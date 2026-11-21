// Lens shading correction kernels: v1 naive, v2 texture sampler, v3 fused half quad.
//
// Author: Xue -c
// Date: 2026-11-16
//
// Conventions match python/lsc_ref.py:
//   * Input is the full-res Bayer mosaic (ushort), raster order.
//   * Plane coords of pixel (x, y) are (x/2, y/2); grid coords are
//     g = plane * gridScale, gridScale = (N - 1) / (planeSize - 1)  (corner-aligned grid).
//   * "pos" is the CFA position inside the 2x2 tile: (y & 1) * 2 + (x & 1).
//   * Output is normalized linear: (raw - black) / (white - black) * gain, unclipped.

#include <metal_stdlib>
using namespace metal;

struct Params {
    uint   width;       // full-res Bayer width
    uint   height;
    uint   gridW;       // gain grid samples (e.g. 17)
    uint   gridH;       // (e.g. 13)
    float2 gridScale;   // ((gridW-1)/(Wp-1), (gridH-1)/(Hp-1))
    float4 black;       // per CFA position
    float4 invRange;    // per CFA position, 1 / (white - black)
    uint4  chanOfPos;   // gain-map channel (R, Gr, Gb, B) used at each CFA position
};

// ---------------------------------------------------------------- shared pre-pass (v1, v2)
// Black level subtraction + normalization as its own pass: ushort -> float.
kernel void blc_f32(device const ushort* raw   [[buffer(0)]],
                    device float*        out   [[buffer(1)]],
                    constant Params&     p     [[buffer(2)]],
                    uint2                gid   [[thread_position_in_grid]])
{
    if (gid.x >= p.width || gid.y >= p.height) return;
    uint pos = (gid.y & 1) * 2 + (gid.x & 1);
    uint i = gid.y * p.width + gid.x;
    out[i] = (float(raw[i]) - p.black[pos]) * p.invRange[pos];
}

// ---------------------------------------------------------------- v1: naive
// One thread per pixel, gain grid in a float buffer [gridH][gridW][4], hand-written bilinear.
kernel void lsc_v1_naive(device const float* in    [[buffer(0)]],
                         device float*       out   [[buffer(1)]],
                         constant Params&    p     [[buffer(2)]],
                         device const float* grid  [[buffer(3)]],
                         uint2               gid   [[thread_position_in_grid]])
{
    if (gid.x >= p.width || gid.y >= p.height) return;
    uint pos = (gid.y & 1) * 2 + (gid.x & 1);
    uint ch  = p.chanOfPos[pos];

    float2 g  = float2(gid >> 1) * p.gridScale;
    uint   x0 = min(uint(g.x), p.gridW - 2);
    uint   y0 = min(uint(g.y), p.gridH - 2);
    float  fx = g.x - float(x0);
    float  fy = g.y - float(y0);

    uint r0 = (y0 * p.gridW + x0) * 4 + ch;
    uint r1 = r0 + p.gridW * 4;
    float top = mix(grid[r0], grid[r0 + 4], fx);
    float bot = mix(grid[r1], grid[r1 + 4], fx);
    float gain = mix(top, bot, fy);

    uint i = gid.y * p.width + gid.x;
    out[i] = in[i] * gain;
}

// ---------------------------------------------------------------- v2: texture sampler
// Same as v1, but the grid is a gridW x gridH RGBA float texture and the sampler's
// bilinear filter does the interpolation. Corner-aligned grid -> texel-center mapping:
//   uv = (g + 0.5) / N
kernel void lsc_v2_sampler(device const float*  in    [[buffer(0)]],
                           device float*        out   [[buffer(1)]],
                           constant Params&     p     [[buffer(2)]],
                           texture2d<float>     grid  [[texture(0)]],
                           uint2                gid   [[thread_position_in_grid]])
{
    constexpr sampler s(coord::normalized, address::clamp_to_edge, filter::linear);
    if (gid.x >= p.width || gid.y >= p.height) return;
    uint pos = (gid.y & 1) * 2 + (gid.x & 1);

    float2 g  = float2(gid >> 1) * p.gridScale;
    float2 uv = (g + 0.5f) / float2(p.gridW, p.gridH);
    float4 gain = grid.sample(s, uv, level(0));

    uint i = gid.y * p.width + gid.x;
    out[i] = in[i] * gain[p.chanOfPos[pos]];
}

// ---------------------------------------------------------------- v3: fused, half, 2x2 quad
// One thread per Bayer quad. The host pre-swizzles the grid so texel.rgba holds the gains
// for CFA positions (0,0),(0,1),(1,0),(1,1) and pre-multiplies them by 1/(white-black),
// so a single sample serves all 4 pixels and normalization is free.
// Reads 2 x ushort2, writes 2 x half2: 4 bytes of DRAM traffic per pixel.
kernel void lsc_v3_quad(device const ushort2* raw   [[buffer(0)]],
                        device half2*         out   [[buffer(1)]],
                        constant Params&      p     [[buffer(2)]],
                        texture2d<half>       grid  [[texture(0)]],
                        uint2                 q     [[thread_position_in_grid]])
{
    constexpr sampler s(coord::normalized, address::clamp_to_edge, filter::linear);
    uint qw = p.width >> 1;
    if (q.x >= qw || q.y >= (p.height >> 1)) return;

    float2 uv = (float2(q) * p.gridScale + 0.5f) / float2(p.gridW, p.gridH);
    half4 gain = grid.sample(s, uv, level(0));

    uint i0 = (q.y << 1) * qw + q.x;   // row 2*qy, in ushort2 units
    uint i1 = i0 + qw;                 // row 2*qy + 1
    ushort2 a = raw[i0];
    ushort2 b = raw[i1];
    half4 v = half4(half(a.x), half(a.y), half(b.x), half(b.y));
    v = (v - half4(p.black)) * gain;
    out[i0] = v.xy;
    out[i1] = v.zw;
}

// ---------------------------------------------------------------- roofline reference
// Pure copy with the same traffic as v3 (2 B read + 2 B write per pixel).
kernel void copy_u16x4(device const ushort4* in  [[buffer(0)]],
                       device ushort4*       out [[buffer(1)]],
                       constant uint&        n4  [[buffer(2)]],
                       uint                  i   [[thread_position_in_grid]])
{
    if (i < n4) out[i] = in[i];
}
