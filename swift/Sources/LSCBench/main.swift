// LSCBench: verify LSC Metal kernels v1/v2/v3 against the Python golden output, then time them
// at 12 MP and 48 MP (2x2 tiled) and report achieved DRAM bandwidth.
//
// Author: Xue -c
// Date: 2026-11-21
//
// Run from the repo root:
//   swift run -c release --package-path swift LSCBench [--golden NAME] [--reps 15] [--batch 10]
//                                                      [--peak-gbs 153] [--capture]
// --capture needs MTL_CAPTURE_ENABLED=1 in the environment; writes out/metal/lsc_12mp.gputrace

import Foundation
import Metal

// MARK: - Args / IO

struct Meta: Decodable {
    let height: Int, width: Int, grid_h: Int, grid_w: Int
    let offsets: [[Int]]   // per channel (R, Gr, Gb, B): [row, col] in the 2x2 tile
    let black: [Float]     // per channel
    let white: Float
}

func arg(_ name: String) -> String? {
    let a = CommandLine.arguments
    guard let i = a.firstIndex(of: name), i + 1 < a.count else { return nil }
    return a[i + 1]
}
let root = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
let goldenDir = root.appendingPathComponent("out/golden")
let outDir = root.appendingPathComponent("out/metal")
try FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)

let golden: String = arg("--golden") ?? {
    let files = (try? FileManager.default.contentsOfDirectory(atPath: goldenDir.path)) ?? []
    guard let j = files.filter({ $0.hasSuffix(".json") }).sorted().first else {
        fatalError("no golden files in out/golden; run python/step1_reference.py first")
    }
    return String(j.dropLast(5))
}()
let reps = Int(arg("--reps") ?? "15")!
let batch = Int(arg("--batch") ?? "10")!
let peakGBs = Double(arg("--peak-gbs") ?? "153")!
let doCapture = CommandLine.arguments.contains("--capture")

func readArray<T>(_ name: String, as: T.Type) -> [T] {
    let d = try! Data(contentsOf: goldenDir.appendingPathComponent(name))
    return d.withUnsafeBytes { Array($0.bindMemory(to: T.self)) }
}
let meta = try JSONDecoder().decode(Meta.self, from: Data(contentsOf: goldenDir.appendingPathComponent("\(golden).json")))
let raw12: [UInt16] = readArray("\(golden).raw_u16.bin", as: UInt16.self)
let gridF: [Float] = readArray("\(golden).gain_f32.bin", as: Float.self)     // [gh][gw][4]
let ref: [Float] = readArray("\(golden).out_f32.bin", as: Float.self)        // [4][Hp][Wp]

// CFA position (raster index in the 2x2 tile) -> channel, black, 1/range
var chanOfPos = SIMD4<UInt32>(repeating: 0)
var blackPos = SIMD4<Float>(repeating: 0)
var invRangePos = SIMD4<Float>(repeating: 0)
for (k, o) in meta.offsets.enumerated() {
    let pos = o[0] * 2 + o[1]
    chanOfPos[pos] = UInt32(k)
    blackPos[pos] = meta.black[k]
    invRangePos[pos] = 1 / (meta.white - meta.black[k])
}

// MARK: - Metal setup

struct Params {   // must match metal/lsc.metal (same natural alignment rules)
    var width: UInt32, height: UInt32, gridW: UInt32, gridH: UInt32
    var gridScale: SIMD2<Float>
    var black: SIMD4<Float>
    var invRange: SIMD4<Float>
    var chanOfPos: SIMD4<UInt32>
}
func makeParams(width: Int, height: Int) -> Params {
    Params(width: UInt32(width), height: UInt32(height),
           gridW: UInt32(meta.grid_w), gridH: UInt32(meta.grid_h),
           gridScale: SIMD2(Float(meta.grid_w - 1) / Float(width / 2 - 1),
                            Float(meta.grid_h - 1) / Float(height / 2 - 1)),
           black: blackPos, invRange: invRangePos, chanOfPos: chanOfPos)
}

let device = MTLCreateSystemDefaultDevice()!
let queue = device.makeCommandQueue()!
let metallib = root.appendingPathComponent("build/lsc.metallib")
let library: MTLLibrary
if FileManager.default.fileExists(atPath: metallib.path) {
    library = try device.makeLibrary(URL: metallib)
    print("library: \(metallib.lastPathComponent) (offline compiled)")
} else {
    let src = try String(contentsOf: root.appendingPathComponent("metal/lsc.metal"), encoding: .utf8)
    library = try device.makeLibrary(source: src, options: MTLCompileOptions())
    print("library: metal/lsc.metal (runtime compiled)")
}
func pso(_ n: String) -> MTLComputePipelineState {
    try! device.makeComputePipelineState(function: library.makeFunction(name: n)!)
}
let psoBLC = pso("blc_f32"), psoV1 = pso("lsc_v1_naive"), psoV2 = pso("lsc_v2_sampler")
let psoV3 = pso("lsc_v3_quad"), psoCopy = pso("copy_u16x4")
print("device: \(device.name), 32-bit float filtering: \(device.supports32BitFloatFiltering)")

// Grid resources (shared by both resolutions; the grid is resolution independent).
let gridBuf = device.makeBuffer(bytes: gridF, length: gridF.count * 4)!
func makeTexture(_ fmt: MTLPixelFormat, _ bytes: UnsafeRawPointer, _ bpp: Int) -> MTLTexture {
    let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: fmt, width: meta.grid_w,
                                                     height: meta.grid_h, mipmapped: false)
    d.usage = .shaderRead
    let t = device.makeTexture(descriptor: d)!
    t.replace(region: MTLRegionMake2D(0, 0, meta.grid_w, meta.grid_h), mipmapLevel: 0,
              withBytes: bytes, bytesPerRow: meta.grid_w * bpp)
    return t
}
precondition(device.supports32BitFloatFiltering, "v2 needs RGBA32Float linear filtering")
let texV2 = gridF.withUnsafeBytes { makeTexture(.rgba32Float, $0.baseAddress!, 16) }
// Same grid as fp16 texels: isolates the cost of 32-bit float filtering in the texture unit.
let gridH16 = gridF.map { Float16($0) }
let texV2h = gridH16.withUnsafeBytes { makeTexture(.rgba16Float, $0.baseAddress!, 8) }
// v3: swizzle channels into CFA raster order and fold in 1/(white-black).
var gridV3 = [Float16](repeating: 0, count: gridF.count)
for cell in 0..<(meta.grid_h * meta.grid_w) {
    for pos in 0..<4 {
        gridV3[cell * 4 + pos] = Float16(gridF[cell * 4 + Int(chanOfPos[pos])] * invRangePos[pos])
    }
}
let texV3 = gridV3.withUnsafeBytes { makeTexture(.rgba16Float, $0.baseAddress!, 8) }

// MARK: - Per-resolution test case

struct Case {
    let label: String, w: Int, h: Int
    var params: Params
    let raw, f32a, f32b, half, copyOut: MTLBuffer
    var px: Int { w * h }
}
func makeCase(_ label: String, raw: [UInt16], w: Int, h: Int) -> Case {
    let n = w * h
    return Case(label: label, w: w, h: h, params: makeParams(width: w, height: h),
                raw: device.makeBuffer(bytes: raw, length: n * 2)!,
                f32a: device.makeBuffer(length: n * 4)!, f32b: device.makeBuffer(length: n * 4)!,
                half: device.makeBuffer(length: n * 2)!, copyOut: device.makeBuffer(length: n * 2)!)
}
func tile2x2(_ a: [UInt16], w: Int, h: Int) -> [UInt16] {
    var o = [UInt16](repeating: 0, count: 4 * w * h)
    a.withUnsafeBufferPointer { src in
        o.withUnsafeMutableBufferPointer { dst in
            for y in 0..<(2 * h) {
                let s = (y % h) * w, d = y * 2 * w
                for t in 0..<2 { for x in 0..<w { dst[d + t * w + x] = src[s + x] } }
            }
        }
    }
    return o
}

// MARK: - Encoders

let tg2D = MTLSize(width: 32, height: 8, depth: 1)
func encBLC(_ e: MTLComputeCommandEncoder, _ c: Case) {
    var p = c.params
    e.setComputePipelineState(psoBLC)
    e.setBuffer(c.raw, offset: 0, index: 0); e.setBuffer(c.f32a, offset: 0, index: 1)
    e.setBytes(&p, length: MemoryLayout<Params>.stride, index: 2)
    e.dispatchThreads(MTLSize(width: c.w, height: c.h, depth: 1), threadsPerThreadgroup: tg2D)
}
func encV1(_ e: MTLComputeCommandEncoder, _ c: Case) {
    var p = c.params
    e.setComputePipelineState(psoV1)
    e.setBuffer(c.f32a, offset: 0, index: 0); e.setBuffer(c.f32b, offset: 0, index: 1)
    e.setBytes(&p, length: MemoryLayout<Params>.stride, index: 2); e.setBuffer(gridBuf, offset: 0, index: 3)
    e.dispatchThreads(MTLSize(width: c.w, height: c.h, depth: 1), threadsPerThreadgroup: tg2D)
}
func encV2(_ e: MTLComputeCommandEncoder, _ c: Case, tex: MTLTexture = texV2) {
    var p = c.params
    e.setComputePipelineState(psoV2)
    e.setBuffer(c.f32a, offset: 0, index: 0); e.setBuffer(c.f32b, offset: 0, index: 1)
    e.setBytes(&p, length: MemoryLayout<Params>.stride, index: 2); e.setTexture(tex, index: 0)
    e.dispatchThreads(MTLSize(width: c.w, height: c.h, depth: 1), threadsPerThreadgroup: tg2D)
}
func encV3(_ e: MTLComputeCommandEncoder, _ c: Case) {
    var p = c.params
    e.setComputePipelineState(psoV3)
    e.setBuffer(c.raw, offset: 0, index: 0); e.setBuffer(c.half, offset: 0, index: 1)
    e.setBytes(&p, length: MemoryLayout<Params>.stride, index: 2); e.setTexture(texV3, index: 0)
    e.dispatchThreads(MTLSize(width: c.w / 2, height: c.h / 2, depth: 1), threadsPerThreadgroup: tg2D)
}
func encCopy(_ e: MTLComputeCommandEncoder, _ c: Case) {
    var n4 = UInt32(c.px / 4)
    e.setComputePipelineState(psoCopy)
    e.setBuffer(c.raw, offset: 0, index: 0); e.setBuffer(c.copyOut, offset: 0, index: 1)
    e.setBytes(&n4, length: 4, index: 2)
    e.dispatchThreads(MTLSize(width: c.px / 4, height: 1, depth: 1),
                      threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
}

func run(_ encode: (MTLComputeCommandEncoder) -> Void, times: Int = 1) -> Double {
    let cb = queue.makeCommandBuffer()!
    let e = cb.makeComputeCommandEncoder()!   // serial dispatch: passes run in order
    for _ in 0..<times { encode(e) }
    e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
    if let err = cb.error { fatalError("GPU error: \(err)") }
    return (cb.gpuEndTime - cb.gpuStartTime) * 1000
}
/// Median GPU ms per pipeline invocation; `batch` invocations per command buffer amortize launch cost.
func bench(_ encode: (MTLComputeCommandEncoder) -> Void) -> Double {
    for _ in 0..<3 { _ = run(encode, times: batch) }
    let t = (0..<reps).map { _ in run(encode, times: batch) / Double(batch) }.sorted()
    return t[t.count / 2]
}

// MARK: - Verification (12 MP only, against the Python golden output)

struct Err { let maxAbs: Float, maxRel: Float, rms: Float }
func compare(_ c: Case, _ value: (Int) -> Float) -> Err {
    let wp = c.w / 2, hp = c.h / 2
    var maxAbs: Float = 0, maxRel: Float = 0, sq: Double = 0
    for y in 0..<c.h {
        for x in 0..<c.w {
            let pos = (y & 1) * 2 + (x & 1)
            let r = ref[Int(chanOfPos[pos]) * hp * wp + (y >> 1) * wp + (x >> 1)]
            let d = abs(value(y * c.w + x) - r)
            maxAbs = max(maxAbs, d); maxRel = max(maxRel, d / max(abs(r), 0.01))
            sq += Double(d * d)
        }
    }
    return Err(maxAbs: maxAbs, maxRel: maxRel, rms: Float((sq / Double(c.px)).squareRoot()))
}

let c12 = makeCase("12MP", raw: raw12, w: meta.width, h: meta.height)
print("golden: \(golden)  \(meta.width)x\(meta.height), grid \(meta.grid_w)x\(meta.grid_h)")
print("\n== correctness vs Python golden (12 MP) ==")
var errs: [String: Err] = [:]
_ = run { encBLC($0, c12); encV1($0, c12) }
var pf = c12.f32b.contents().bindMemory(to: Float.self, capacity: c12.px)
errs["v1"] = compare(c12) { pf[$0] }
_ = run { encBLC($0, c12); encV2($0, c12) }
pf = c12.f32b.contents().bindMemory(to: Float.self, capacity: c12.px)
errs["v2"] = compare(c12) { pf[$0] }
_ = run { encBLC($0, c12); encV2($0, c12, tex: texV2h) }
pf = c12.f32b.contents().bindMemory(to: Float.self, capacity: c12.px)
errs["v2h"] = compare(c12) { pf[$0] }
_ = run { encV3($0, c12) }
let ph = c12.half.contents().bindMemory(to: Float16.self, capacity: c12.px)
errs["v3"] = compare(c12) { Float(ph[$0]) }
for k in ["v1", "v2", "v2h", "v3"] {
    let e = errs[k]!
    print(String(format: "  %@  max|err| %.2e   max rel %.2e   rms %.2e", k, e.maxAbs, e.maxRel, e.rms))
}

// MARK: - Capture (optional)

if doCapture {
    let url = outDir.appendingPathComponent("lsc_12mp.gputrace")
    try? FileManager.default.removeItem(at: url)
    let cm = MTLCaptureManager.shared()
    let d = MTLCaptureDescriptor()
    d.captureObject = queue; d.destination = .gpuTraceDocument; d.outputURL = url
    do {
        try cm.startCapture(with: d)
        _ = run { encBLC($0, c12); encV1($0, c12); encV2($0, c12); encV3($0, c12); encCopy($0, c12) }
        cm.stopCapture()
        print("\ncapture -> \(url.path)")
    } catch { print("\ncapture failed (set MTL_CAPTURE_ENABLED=1): \(error)") }
}

// MARK: - Timing

struct Row: Encodable { let res: String, kernel: String, ms: Double, bytesPerPx: Int, gbs: Double, pctPeak: Double }
var rows: [Row] = []
let c48 = makeCase("48MP", raw: tile2x2(raw12, w: meta.width, h: meta.height), w: meta.width * 2, h: meta.height * 2)
for c in [c12, c48] {
    let tests: [(String, Int, (MTLComputeCommandEncoder) -> Void)] = [
        ("copy u16 (roofline)", 4, { encCopy($0, c) }),
        ("BLC pass only", 6, { encBLC($0, c) }),
        ("v1 naive LSC pass", 8, { encV1($0, c) }),
        ("v2 sampler LSC pass", 8, { encV2($0, c) }),
        ("v2h sampler fp16 tex", 8, { encV2($0, c, tex: texV2h) }),
        ("v1 pipeline BLC+LSC", 14, { encBLC($0, c); encV1($0, c) }),
        ("v2 pipeline BLC+LSC", 14, { encBLC($0, c); encV2($0, c) }),
        ("v3 fused half quad", 4, { encV3($0, c) }),
    ]
    for (name, bpp, enc) in tests {
        let ms = bench(enc)
        let gbs = Double(c.px * bpp) / (ms / 1000) / 1e9
        rows.append(Row(res: c.label, kernel: name, ms: ms, bytesPerPx: bpp, gbs: gbs, pctPeak: 100 * gbs / peakGBs))
    }
}
print("\n== timing (median GPU time, \(reps) reps x \(batch) per command buffer; peak \(Int(peakGBs)) GB/s) ==")
func pad(_ s: String, _ n: Int) -> String { s.padding(toLength: n, withPad: " ", startingAt: 0) }
print(pad("res", 6) + pad("kernel", 24) + "     ms  B/px    GB/s  %peak")
for r in rows {
    print(pad(r.res, 6) + pad(r.kernel, 24) +
          String(format: "%7.3f  %4d  %6.1f  %5.1f", r.ms, r.bytesPerPx, r.gbs, r.pctPeak))
}
struct Report: Encodable { let device: String, golden: String, errors: [String: [Float]], rows: [Row] }
let rep = Report(device: device.name, golden: golden,
                 errors: errs.mapValues { [$0.maxAbs, $0.maxRel, $0.rms] }, rows: rows)
let enc = JSONEncoder(); enc.outputFormatting = [.prettyPrinted, .sortedKeys]
try enc.encode(rep).write(to: outDir.appendingPathComponent("results.json"))
print("\nresults -> out/metal/results.json")
