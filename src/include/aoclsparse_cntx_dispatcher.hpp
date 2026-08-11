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
 * ************************************************************************
 */
#ifndef AOCLSPARSE_DISPATCH_HPP
#define AOCLSPARSE_DISPATCH_HPP

#include "aoclsparse.h"
#include "aoclsparse_context.hpp"
#include "aoclsparse_utils.hpp"

namespace Dispatch
{
    /* Kernel Attribute Table entry */
    template <typename K>
    struct Table
    {
        K                         kernel;
        aoclsparse::context_isa_t flag;
        unsigned int              arch;
    };

    /* ORL: On ARM/portable builds, always returns empty entry (no AVX-512 support) */
    template <typename K>
    K ORL(K /*avx512_f*/, K avx2_f)
    {
        return avx2_f;
    }

    inline aoclsparse::context_isa_t
        get_supported(std::initializer_list<aoclsparse::context_isa_t> isa_list)
    {
        using namespace aoclsparse;

        context_isa_t  supported = context_isa_t::UNSET;
        aoclsparse_int score     = 0;

        for(const auto &isa : isa_list)
        {
            if(context::get_context()->supports(isa))
            {
                aoclsparse_int isa_score = (static_cast<aoclsparse_int>(isa));
                if(isa_score >= score)
                {
                    score     = isa_score;
                    supported = isa;
                }
            }
        }

        return supported;
    }

    template <typename K>
    constexpr Table<K> ORL([[maybe_unused]] Table<K> T)
    {
        // Portable build: no AVX-512, return empty entry
        return {nullptr, aoclsparse::context_isa_t::UNSET, 0U | aoclsparse::archs::UNKNOWN};
    }

    inline bool in_range(aoclsparse_int n, aoclsparse_int lower, aoclsparse_int upper)
    {
        return (lower <= n && n <= upper);
    }

    /* Simplified Oracle: returns the GENERIC (scalar) kernel from the table.
     * In the portable build, KAT tables contain only GENERIC entries,
     * so this always returns the scalar reference implementation.
     */
    template <typename K, aoclsparse_int N>
    K Oracle(const Table<K> (&tbl)[N],
             K                    best_kernel,
             const aoclsparse_int kid   = -1,
             const aoclsparse_int begin = 0,
             const aoclsparse_int end   = N)
    {
        using namespace aoclsparse;

        // Check if user requested kid is NOT auto
        if(kid >= 0)
        {
            if(kid >= (end - begin) || begin >= end || !in_range(begin, 0, N - 1)
               || !in_range(end, 0, N))
            {
                return nullptr;
            }

            if(context::get_context()->supports(tbl[begin + kid].flag))
                return tbl[begin + kid].kernel;
            else
                return nullptr;
        }

        // Always return the first GENERIC kernel
        for(aoclsparse_int kcnt = begin; kcnt < end; ++kcnt)
        {
            if(context::get_context()->supports(tbl[kcnt].flag))
            {
                return tbl[kcnt].kernel;
            }
        }

        return nullptr;
    }
}
#endif
