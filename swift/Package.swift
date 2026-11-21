// swift-tools-version:5.9
// LSC Metal benchmark harness.
// Author: Xue -c
// Date: 2026-11-18
import PackageDescription

let package = Package(
    name: "LSCBench",
    platforms: [.macOS(.v14)],
    targets: [
        .executableTarget(name: "LSCBench", path: "Sources/LSCBench")
    ]
)
