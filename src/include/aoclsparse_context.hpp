/* ************************************************************************
 * Copyright (c) 2021-2026 Advanced Micro Devices, Inc.
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

#pragma once
#ifndef AOCLSPARSE_CONTEXT_HPP
#define AOCLSPARSE_CONTEXT_HPP

#include "aoclsparse.h"

#include <algorithm>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace aoclsparse
{
    // ISA context preference
    enum class context_isa_t
    {
        UNSET   = 0,
        GENERIC = 1,
        NEON    = 2,
        LENGTH  = 3
    };

    // Architecture enum (portable: minimal)
    enum class archs : unsigned int
    {
        ALL     = ~0U,
        UNKNOWN = 0U
    };

    inline constexpr unsigned int operator|(archs a, archs b)
    {
        return static_cast<unsigned int>(a) | static_cast<unsigned int>(b);
    }
    inline constexpr unsigned int operator|(unsigned int a, archs b)
    {
        return a | static_cast<unsigned int>(b);
    }
    inline constexpr unsigned int operator&(archs a, unsigned int b)
    {
        return static_cast<unsigned int>(a) & b;
    }

    template <typename T>
    T env_get_var(const char *env, const T fallback)
    {
        T     r_val;
        char *str;

        str = getenv(env);
        if(str != NULL)
        {
            if constexpr(std::is_same_v<T, aoclsparse_int>)
            {
                r_val = static_cast<aoclsparse_int>(strtol(str, NULL, 10));
                return r_val;
            }
            else if constexpr(std::is_same_v<T, std::string>)
            {
                return std::string(str);
            }
        }

        return fallback;
    }

    class context
    {
    private:
        bool cpuflags[static_cast<int>(context_isa_t::LENGTH)];

        context_isa_t global_isa_hint = context_isa_t::UNSET;

        context()
        {
            for(int f = 0; f < static_cast<int>(context_isa_t::LENGTH); ++f)
                cpuflags[f] = false;

            // GENERIC is always supported
            cpuflags[static_cast<int>(context_isa_t::GENERIC)] = true;

#ifdef __ARM_NEON
            cpuflags[static_cast<int>(context_isa_t::NEON)] = true;
#endif

            // Portable: only GENERIC and NEON are available
            global_isa_hint = context_isa_t::UNSET;
        }

        aoclsparse_int get_thread_from_env()
        {
            aoclsparse_int nt = 1;

#ifdef _OPENMP
            nt = env_get_var("AOCLSPARSE_NUM_THREADS",
                             static_cast<aoclsparse_int>(omp_get_num_procs()));

            if(nt <= 0)
                nt = 1;
#endif
            return nt;
        }

    public:
        ~context() {}

        context(context &t) = delete;
        void operator=(const context &) = delete;
        context(context &&t) = delete;
        void operator=(context &&t) = delete;

        template <context_isa_t... isa>
        bool supports()
        {
            return (... && this->cpuflags[static_cast<aoclsparse_int>(isa)]);
        }

        bool supports(context_isa_t isa)
        {
            if(isa == context_isa_t::GENERIC)
                return true;
            return this->cpuflags[static_cast<aoclsparse_int>(isa)];
        }

        aoclsparse_int get_num_threads(void)
        {
            aoclsparse_int num_threads = 1;

#ifdef _OPENMP
            aoclsparse_int env_num_threads = this->get_thread_from_env();

            aoclsparse_int max_threads = omp_get_max_threads();
#if _OPENMP >= 200805
            aoclsparse_int thread_limit = omp_get_thread_limit();
#else
            aoclsparse_int thread_limit = max_threads;
#endif

            aoclsparse_int omp_min = (max_threads > thread_limit) ? thread_limit : max_threads;

            num_threads = (omp_min > env_num_threads) ? env_num_threads : omp_min;

#if _OPENMP >= 200805
            aoclsparse_int max_levels    = omp_get_max_active_levels();
            aoclsparse_int current_level = omp_get_active_level();

            if(max_levels <= current_level)
            {
                num_threads = 1;
            }
#elif _OPENMP >= 200505
            if(omp_get_level() > 0)
            {
                num_threads = 1;
            }
#endif
#endif

            return num_threads;
        }

        context_isa_t get_isa_hint()
        {
            return this->global_isa_hint;
        }

        archs get_archs(void)
        {
            return archs::UNKNOWN;
        }

        static context *get_context();
    };

    class isa_hint
    {
        aoclsparse::context_isa_t old_hint;
        aoclsparse::context_isa_t current_hint;

    public:
        isa_hint()
        {
            current_hint = old_hint = context::get_context()->get_isa_hint();
        };

        context_isa_t get_isa_hint()
        {
            return this->current_hint;
        };

        void set_isa_hint(context_isa_t isa)
        {
            this->old_hint = this->current_hint;
            this->current_hint = isa;
        };

        bool is_isa_updated()
        {
            return (this->current_hint != this->old_hint);
        }

        isa_hint(isa_hint &t) = delete;
        void operator=(const isa_hint &) = delete;
        isa_hint(isa_hint &&t) = delete;
        void operator=(isa_hint &&t) = delete;
    };
}

extern thread_local aoclsparse::isa_hint tl_isa_hint;

#endif // AOCLSPARSE_CONTEXT_HPP
