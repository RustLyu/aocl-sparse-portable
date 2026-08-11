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
#ifndef AOCLSPARSE_KERNEL_TEMPLATES_HPP
#define AOCLSPARSE_KERNEL_TEMPLATES_HPP

#include "aoclsparse.h"

/* Portable stub for the Kernel Templates layer.
 * On ARM/portable builds, all KT-based SIMD kernels are unavailable.
 * Only scalar reference implementations are used.
 */

namespace kernel_templates
{
    // Minimal bsz enum (compatible with original code for KAT table sizing)
    enum class bsz
    {
        b128 = 128,
        b256 = 256,
        b512 = 512
    };

    enum class kt_avxext
    {
        AVX2,
        AVX512VL,
        AVX512F
    };

} // namespace kernel_templates

/* KT_INSTANTIATE is a no-op in portable builds */
#define KT_INSTANTIATE(FUNC, SUF)

#endif // AOCLSPARSE_KERNEL_TEMPLATES_HPP
