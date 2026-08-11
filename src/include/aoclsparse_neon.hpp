/* ************************************************************************
 * Copyright (c) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */
#ifndef AOCLSPARSE_NEON_HPP
#define AOCLSPARSE_NEON_HPP

/*
 * Lightweight NEON SIMD helpers — supports both AArch64 (64-bit) and
 * ARMv7-A (32-bit) with NEON.
 *
 * Key ARMv7 vs AArch64 differences handled here:
 *   - vfmaq_f32 requires VFPv4 (not on Cortex-A8/A9); fallback to vmlaq_f32
 *   - float64x2_t may not be available on older ARMv7 toolchains;
 *     scalar VFPv3 fallback provided via struct + _f64-named functions
 *   - AArch64 has 32x128b NEON registers; ARMv7 has 16x128b
 *
 * All code guarded by #ifdef __ARM_NEON — safe to include on x86/MSVC.
 */

#ifdef __ARM_NEON
#include <arm_neon.h>

// For ARMv7 toolchains that lack double-precision NEON intrinsics
// (no __ARM_FEATURE_FP64_VECTOR_ARITHMETIC), provide a scalar
// fallback struct at global scope matching the native convention.
#ifndef __ARM_FEATURE_FP64_VECTOR_ARITHMETIC
struct float64x2_t
{
    double val[2];
};
inline double vgetq_lane_f64(float64x2_t v, const int lane)
{
    return v.val[lane];
}
#endif

namespace aoclsparse_neon
{

    // =========================================================================
    // float (float32x4_t — 4-wide, 128-bit) — NEON SIMD
    // =========================================================================

    inline float32x4_t loadu(const float *p)
    {
        return vld1q_f32(p);
    }
    inline void storeu(float *p, float32x4_t v)
    {
        vst1q_f32(p, v);
    }
    inline float32x4_t setzero()
    {
        return vdupq_n_f32(0.0f);
    }
    inline float32x4_t set1(float v)
    {
        return vdupq_n_f32(v);
    }
    inline float32x4_t mul(float32x4_t a, float32x4_t b)
    {
        return vmulq_f32(a, b);
    }

    /*
     * Fused multiply-add: c + a*b
     *   AArch64 / VFPv4:  vfmaq_f32  (single rounding, lower latency)
     *   ARMv7 w/o VFPv4:  vmlaq_f32  (two roundings, compatible with all NEON)
     */
    inline float32x4_t fmadd(float32x4_t a, float32x4_t b, float32x4_t c)
    {
#if defined(__aarch64__) || defined(__ARM_VFPV4__)
        return vfmaq_f32(c, a, b);
#else
        return vmlaq_f32(c, a, b);
#endif
    }

    /* Horizontal sum of a float32x4_t vector */
    inline float hsum(float32x4_t v)
    {
        float32x2_t lo  = vget_low_f32(v);
        float32x2_t hi  = vget_high_f32(v);
        float32x2_t sum = vadd_f32(lo, hi);
        sum             = vpadd_f32(sum, sum);
        return vget_lane_f32(sum, 0);
    }

    /*
     * Gather-load 4 floats from 4 potentially non-contiguous addresses
     * into a float32x4_t. NEON has no native gather, so we do 4 scalar
     * loads and combine with vsetq_lane.
     */
    inline float32x4_t gather(const float *p0,
                              const float *p1,
                              const float *p2,
                              const float *p3)
    {
        float32x4_t v;
        v = vsetq_lane_f32(*p0, v, 0);
        v = vsetq_lane_f32(*p1, v, 1);
        v = vsetq_lane_f32(*p2, v, 2);
        v = vsetq_lane_f32(*p3, v, 3);
        return v;
    }

    /* Software prefetch for gather sources — helps ARMv7's simpler prefetcher */
    inline void prefetch_gather(const float *p0,
                                const float *p1,
                                const float *p2,
                                const float *p3)
    {
        __builtin_prefetch(p0, 0, 3);
        __builtin_prefetch(p1, 0, 3);
        __builtin_prefetch(p2, 0, 3);
        __builtin_prefetch(p3, 0, 3);
    }

    // =========================================================================
    // double (float64x2_t — 2-wide, 128-bit)
    //
    // On ARMv7-A without __ARM_FEATURE_FP64_VECTOR_ARITHMETIC, we use
    // scalar VFPv3 instructions through a custom float64x2_t struct.
    // All double helpers use the _f64 suffix to avoid overloading issues
    // in C++14 code (where if constexpr is not available).
    // =========================================================================

    // -- double load / store ------------------------------------------------

    inline float64x2_t loadu(const double *p)
    {
#ifdef __ARM_FEATURE_FP64_VECTOR_ARITHMETIC
        return vld1q_f64(p);
#else
        float64x2_t r;
        r.val[0] = p[0];
        r.val[1] = p[1];
        return r;
#endif
    }
    inline void storeu(double *p, float64x2_t v)
    {
#ifdef __ARM_FEATURE_FP64_VECTOR_ARITHMETIC
        vst1q_f64(p, v);
#else
        p[0] = v.val[0];
        p[1] = v.val[1];
#endif
    }

    // -- double arithmetic (_f64 suffix) ------------------------------------

    inline float64x2_t setzero_f64()
    {
#ifdef __ARM_FEATURE_FP64_VECTOR_ARITHMETIC
        return vdupq_n_f64(0.0);
#else
        float64x2_t r;
        r.val[0] = 0.0;
        r.val[1] = 0.0;
        return r;
#endif
    }
    inline float64x2_t set1_f64(double v)
    {
#ifdef __ARM_FEATURE_FP64_VECTOR_ARITHMETIC
        return vdupq_n_f64(v);
#else
        float64x2_t r;
        r.val[0] = v;
        r.val[1] = v;
        return r;
#endif
    }
    inline float64x2_t mul_f64(float64x2_t a, float64x2_t b)
    {
#ifdef __ARM_FEATURE_FP64_VECTOR_ARITHMETIC
        return vmulq_f64(a, b);
#else
        float64x2_t r;
        r.val[0] = a.val[0] * b.val[0];
        r.val[1] = a.val[1] * b.val[1];
        return r;
#endif
    }

    /*
     * Fused multiply-add: c + a*b
     *   AArch64:           vfmaq_f64  (FMA for double)
     *   ARMv7 w/ DP NEON:  vaddq_f64 + vmulq_f64
     *   ARMv7 scalar:      plain scalar VFPv3 fmadd
     */
    inline float64x2_t fmadd_f64(float64x2_t a, float64x2_t b, float64x2_t c)
    {
#ifdef __aarch64__
        return vfmaq_f64(c, a, b);
#elif defined(__ARM_FEATURE_FP64_VECTOR_ARITHMETIC)
        return vaddq_f64(c, vmulq_f64(a, b));
#else
        float64x2_t r;
        r.val[0] = c.val[0] + a.val[0] * b.val[0];
        r.val[1] = c.val[1] + a.val[1] * b.val[1];
        return r;
#endif
    }

    /* Horizontal sum of a float64x2_t vector */
    inline double hsum_f64(float64x2_t v)
    {
#ifdef __ARM_FEATURE_FP64_VECTOR_ARITHMETIC
        float64x1_t lo  = vget_low_f64(v);
        float64x1_t hi  = vget_high_f64(v);
        float64x1_t sum = vadd_f64(lo, hi);
        return vget_lane_f64(sum, 0);
#else
        return v.val[0] + v.val[1];
#endif
    }

    /*
     * Gather-load 2 doubles from 2 potentially non-contiguous addresses
     * into a float64x2_t.
     */
    inline float64x2_t gather_f64(const double *p0, const double *p1)
    {
#ifdef __ARM_FEATURE_FP64_VECTOR_ARITHMETIC
        float64x2_t v;
        v = vsetq_lane_f64(*p0, v, 0);
        v = vsetq_lane_f64(*p1, v, 1);
        return v;
#else
        float64x2_t v;
        v.val[0] = *p0;
        v.val[1] = *p1;
        return v;
#endif
    }

    inline void prefetch_gather_f64(const double *p0, const double *p1)
    {
        __builtin_prefetch(p0, 0, 3);
        __builtin_prefetch(p1, 0, 3);
    }

} // namespace aoclsparse_neon

#endif // __ARM_NEON
#endif // AOCLSPARSE_NEON_HPP
