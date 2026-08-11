/* ************************************************************************
 * Copyright (c) 2020-2026 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef AOCLSPARSE_CSRMV_HPP
#define AOCLSPARSE_CSRMV_HPP

/* Portable CSR SpMV dispatcher — scalar reference kernels only.
 * All x86 SIMD (AVX2/AVX512/KT) code paths have been removed.
 */

#include "aoclsparse.h"
#include "aoclsparse_context.hpp"
#include "aoclsparse_descr.h"
#include "aoclsparse_error_check.hpp"
#include "aoclsparse_mat_structures.hpp"
#include "aoclsparse_mtx_dispatcher.hpp"
#include "aoclsparse_neon.hpp"
#include "aoclsparse_utils.hpp"

// ==========================================================================
// Scalar reference kernels (ported from aoclsparse_csrmv_kr.hpp)
// ==========================================================================

/* General (non-transposed) SpMV reference */
template <typename T>
static aoclsparse_status ref_csrmv_gn(aoclsparse_index_base base,
                                       const T               alpha,
                                       aoclsparse_int        m,
                                       const T *__restrict__ csr_val,
                                       const aoclsparse_int *__restrict__ csr_col_ind,
                                       const aoclsparse_int *__restrict__ csr_row_ptr,
                                       const T *__restrict__ x,
                                       const T               beta,
                                       T *__restrict__ y)
{
    const aoclsparse_int *csr_col_ind_fix = csr_col_ind - base;
    const T              *csr_val_fix     = csr_val - base;
    const T              *x_fix           = x - base;

#ifdef _OPENMP
#pragma omp parallel for num_threads(aoclsparse::context::get_context()->get_num_threads())
#endif
    for(aoclsparse_int i = 0; i < m; i++)
    {
        T result = 0.0;

        aoclsparse_int j     = csr_row_ptr[i];
        aoclsparse_int j_end = csr_row_ptr[i + 1];

#ifdef __ARM_NEON
        if constexpr(std::is_same_v<T, float>)
        {
            float32x4_t vsum = aoclsparse_neon::setzero();
            for(; j + 7 < j_end; j += 4)
            {
                // Prefetch the next set of x gather targets
                aoclsparse_neon::prefetch_gather(&x_fix[csr_col_ind_fix[j + 4]],
                                                 &x_fix[csr_col_ind_fix[j + 5]],
                                                 &x_fix[csr_col_ind_fix[j + 6]],
                                                 &x_fix[csr_col_ind_fix[j + 7]]);
                float32x4_t vv = aoclsparse_neon::loadu(&csr_val_fix[j]);
                float32x4_t vx = aoclsparse_neon::gather(&x_fix[csr_col_ind_fix[j]],
                                                          &x_fix[csr_col_ind_fix[j + 1]],
                                                          &x_fix[csr_col_ind_fix[j + 2]],
                                                          &x_fix[csr_col_ind_fix[j + 3]]);
                vsum           = aoclsparse_neon::fmadd(vv, vx, vsum);
            }
            // Tail: 0-6 elements remaining, process 4 at a time + scalar
            for(; j + 3 < j_end; j += 4)
            {
                float32x4_t vv = aoclsparse_neon::loadu(&csr_val_fix[j]);
                float32x4_t vx = aoclsparse_neon::gather(&x_fix[csr_col_ind_fix[j]],
                                                          &x_fix[csr_col_ind_fix[j + 1]],
                                                          &x_fix[csr_col_ind_fix[j + 2]],
                                                          &x_fix[csr_col_ind_fix[j + 3]]);
                vsum           = aoclsparse_neon::fmadd(vv, vx, vsum);
            }
            result = aoclsparse_neon::hsum(vsum);
        }
        else if constexpr(std::is_same_v<T, double>)
        {
            float64x2_t vsum = aoclsparse_neon::setzero();
            for(; j + 3 < j_end; j += 2)
            {
                // Prefetch the next pair of x gather targets
                aoclsparse_neon::prefetch_gather(&x_fix[csr_col_ind_fix[j + 2]],
                                                 &x_fix[csr_col_ind_fix[j + 3]]);
                float64x2_t vv = aoclsparse_neon::loadu(&csr_val_fix[j]);
                float64x2_t vx = aoclsparse_neon::gather(&x_fix[csr_col_ind_fix[j]],
                                                          &x_fix[csr_col_ind_fix[j + 1]]);
                vsum           = aoclsparse_neon::fmadd(vv, vx, vsum);
            }
            // Tail: 0-2 elements remaining
            for(; j + 1 < j_end; j += 2)
            {
                float64x2_t vv = aoclsparse_neon::loadu(&csr_val_fix[j]);
                float64x2_t vx = aoclsparse_neon::gather(&x_fix[csr_col_ind_fix[j]],
                                                          &x_fix[csr_col_ind_fix[j + 1]]);
                vsum           = aoclsparse_neon::fmadd(vv, vx, vsum);
            }
            result = aoclsparse_neon::hsum(vsum);
        }
#endif
        for(; j < j_end; j++)
        {
            result += csr_val_fix[j] * x_fix[csr_col_ind_fix[j]];
        }

        if(alpha != static_cast<T>(1))
        {
            result = alpha * result;
        }

        if(beta != static_cast<T>(0))
        {
            result += beta * y[i];
        }

        y[i] = result;
    }

    return aoclsparse_status_success;
}

/* Transposed / conjugate-transposed SpMV reference */
template <typename T, bool is_herm = false>
static aoclsparse_status ref_csrmv_th(aoclsparse_index_base base,
                                       const T               alpha,
                                       aoclsparse_int        m,
                                       aoclsparse_int        n,
                                       const T *__restrict__ csr_val,
                                       const aoclsparse_int *__restrict__ csr_col_ind,
                                       const aoclsparse_int *__restrict__ csr_row_ptr,
                                       const T *__restrict__ x,
                                       const T               beta,
                                       T *__restrict__ y)
{
    const aoclsparse_int *csr_col_ind_fix = csr_col_ind - base;
    const T              *csr_val_fix     = csr_val - base;
    T                    *y_fix           = y - base;

    if(beta == static_cast<T>(0))
    {
        for(aoclsparse_int i = 0; i < n; i++)
            y[i] = 0.0;
    }
    else if(beta != static_cast<T>(1))
    {
        for(aoclsparse_int i = 0; i < n; i++)
            y[i] = beta * y[i];
    }

    for(aoclsparse_int i = 0; i < m; i++)
    {
        aoclsparse_int row_start = csr_row_ptr[i];
        aoclsparse_int row_end   = csr_row_ptr[i + 1];
        T              axi       = alpha * x[i];
        aoclsparse_int j     = row_start;
        aoclsparse_int j_end = row_end;

#ifdef __ARM_NEON
        if constexpr(!is_herm && std::is_same_v<T, float>)
        {
            float32x4_t vaxi = aoclsparse_neon::set1(axi);
            for(; j + 3 < j_end; j += 4)
            {
                float32x4_t vv = aoclsparse_neon::loadu(&csr_val_fix[j]);
                float32x4_t vp = aoclsparse_neon::mul(vv, vaxi);
                y_fix[csr_col_ind_fix[j]] += vgetq_lane_f32(vp, 0);
                y_fix[csr_col_ind_fix[j + 1]] += vgetq_lane_f32(vp, 1);
                y_fix[csr_col_ind_fix[j + 2]] += vgetq_lane_f32(vp, 2);
                y_fix[csr_col_ind_fix[j + 3]] += vgetq_lane_f32(vp, 3);
            }
        }
        else if constexpr(!is_herm && std::is_same_v<T, double>)
        {
            float64x2_t vaxi = aoclsparse_neon::set1(axi);
            for(; j + 1 < j_end; j += 2)
            {
                float64x2_t vv = aoclsparse_neon::loadu(&csr_val_fix[j]);
                float64x2_t vp = aoclsparse_neon::mul(vv, vaxi);
                y_fix[csr_col_ind_fix[j]] += vgetq_lane_f64(vp, 0);
                y_fix[csr_col_ind_fix[j + 1]] += vgetq_lane_f64(vp, 1);
            }
        }
#endif
        for(; j < j_end; j++)
        {
            aoclsparse_int col_idx = csr_col_ind_fix[j];
            if constexpr(is_herm)
                y_fix[col_idx] += aoclsparse::conj(csr_val_fix[j]) * axi;
            else
                y_fix[col_idx] += csr_val_fix[j] * axi;
        }
    }
    return aoclsparse_status_success;
}

/* Triangular SpMV reference (non-transposed) */
template <typename T>
static aoclsparse_status ref_csrmv_tri(aoclsparse_mat_descr            descr,
                                        const T                         alpha,
                                        aoclsparse_int                  m,
                                        [[maybe_unused]] aoclsparse_int n,
                                        const T *__restrict__ csr_val,
                                        const aoclsparse_int *__restrict__ csr_col_ind,
                                        const aoclsparse_int *__restrict__ crstart,
                                        const aoclsparse_int *__restrict__ crend,
                                        const T *__restrict__ x,
                                        const T               beta,
                                        T *__restrict__ y)
{
    aoclsparse_index_base base            = descr->base;
    const aoclsparse_int *csr_col_ind_fix = csr_col_ind - base;
    const T              *csr_val_fix     = csr_val - base;
    const T              *x_fix           = x - base;
    T                     one             = 1;
    T                     zero            = 0;
    aoclsparse_int        start_offset = 0, end_offset = 0;

    if((descr->type != aoclsparse_matrix_type_general)
       && (descr->diag_type == aoclsparse_diag_type_unit
           || descr->diag_type == aoclsparse_diag_type_zero))
    {
        if(descr->fill_mode == aoclsparse_fill_mode_lower)
            end_offset = -1;
        else
            start_offset = 1;
    }
    bool diag_first = start_offset && descr->diag_type == aoclsparse_diag_type_unit;
    bool diag_last  = end_offset && descr->diag_type == aoclsparse_diag_type_unit;

    if(beta == zero)
    {
        for(aoclsparse_int i = 0; i < m; i++)
            y[i] = 0.0;
    }
    else if(beta != one)
    {
        for(aoclsparse_int i = 0; i < m; i++)
            y[i] = beta * y[i];
    }

    for(aoclsparse_int i = 0; i < m; i++)
    {
        aoclsparse_int rstart = crstart[i] + start_offset;
        aoclsparse_int rend   = crend[i] + end_offset;

        T result = 0.0;
        if(diag_first)
            result += x_fix[i + base];

        for(aoclsparse_int j = rstart; j < rend; j++)
        {
            result += csr_val_fix[j] * x_fix[csr_col_ind_fix[j]];
        }
        if(diag_last)
            result += x_fix[i + base];

        y[i] += alpha * result;
    }
    return aoclsparse_status_success;
}

/* Transposed triangular SpMV reference */
template <typename T, bool is_herm = false>
static aoclsparse_status ref_csrmv_tri_th(const aoclsparse_mat_descr descr,
                                           const T                    alpha,
                                           aoclsparse_int             m,
                                           aoclsparse_int             n,
                                           const T *__restrict__ csr_val,
                                           const aoclsparse_int *__restrict__ csr_col_ind,
                                           const aoclsparse_int *__restrict__ crstart,
                                           const aoclsparse_int *__restrict__ crend,
                                           const T *__restrict__ x,
                                           const T               beta,
                                           T *__restrict__ y)
{
    aoclsparse_index_base base            = descr->base;
    const aoclsparse_int *csr_col_ind_fix = csr_col_ind - base;
    const T              *csr_val_fix     = csr_val - base;
    T                    *y_fix           = y - base;
    aoclsparse_int        start_offset = 0, end_offset = 0;

    if((descr->type != aoclsparse_matrix_type_general)
       && (descr->diag_type == aoclsparse_diag_type_unit
           || descr->diag_type == aoclsparse_diag_type_zero))
    {
        if(descr->fill_mode == aoclsparse_fill_mode_lower)
            end_offset = -1;
        else
            start_offset = 1;
    }
    bool diag_first = start_offset && descr->diag_type == aoclsparse_diag_type_unit;
    bool diag_last  = end_offset && descr->diag_type == aoclsparse_diag_type_unit;

    if(beta == static_cast<T>(0))
    {
        for(aoclsparse_int i = 0; i < n; i++)
            y[i] = 0.0;
    }
    else if(beta != static_cast<T>(1))
    {
        for(aoclsparse_int i = 0; i < n; i++)
            y[i] = beta * y[i];
    }

    for(aoclsparse_int i = 0; i < m; i++)
    {
        aoclsparse_int rstart = crstart[i] + start_offset;
        aoclsparse_int rend   = crend[i] + end_offset;

        T axi = alpha * x[i];
        if(diag_first)
            y_fix[i + base] += axi;
        for(aoclsparse_int j = rstart; j < rend; j++)
        {
            aoclsparse_int col_idx = csr_col_ind_fix[j];
            if constexpr(is_herm)
                y_fix[col_idx] += aoclsparse::conj(csr_val_fix[j]) * axi;
            else
                y_fix[col_idx] += csr_val_fix[j] * axi;
        }
        if(diag_last)
            y_fix[i + base] += axi;
    }
    return aoclsparse_status_success;
}

/* Simple symmetric SpMV */
template <typename T>
static aoclsparse_status csrmv_symm(aoclsparse_index_base base,
                                     const T               alpha,
                                     aoclsparse_int        m,
                                     const T *__restrict__ csr_val,
                                     const aoclsparse_int *__restrict__ csr_col_ind,
                                     const aoclsparse_int *__restrict__ csr_row_ptr,
                                     const T *__restrict__ x,
                                     const T               beta,
                                     T *__restrict__ y)
{
    if(beta == static_cast<T>(0))
    {
        for(aoclsparse_int i = 0; i < m; i++)
            y[i] = 0.;
    }
    else if(beta != static_cast<T>(1))
    {
        for(aoclsparse_int i = 0; i < m; i++)
            y[i] = beta * y[i];
    }

    for(aoclsparse_int i = 0; i < m; i++)
    {
        aoclsparse_int diag_idx      = csr_row_ptr[i + 1] - base - 1;
        aoclsparse_int last_ele_diag = !((csr_col_ind[diag_idx] - base) ^ i);
        y[i] += last_ele_diag * alpha * csr_val[diag_idx] * x[i];
        aoclsparse_int end = csr_row_ptr[i + 1] - base - last_ele_diag;
        for(aoclsparse_int j = (csr_row_ptr[i] - base); j < end; j++)
        {
            y[i] += alpha * csr_val[j] * x[csr_col_ind[j] - base];
            y[csr_col_ind[j] - base] += alpha * csr_val[j] * x[i];
        }
    }
    return aoclsparse_status_success;
}

/* Optimized symmetric SpMV (for L/U triangle with idiag/iurow) */
template <typename T>
static aoclsparse_status csrmv_symm_internal(aoclsparse_index_base base,
                                              T                     alpha,
                                              aoclsparse_int        m,
                                              aoclsparse_diag_type  diag_type,
                                              aoclsparse_fill_mode  fill_mode,
                                              const T *__restrict__ csr_val,
                                              const aoclsparse_int *__restrict__ csr_icol,
                                              const aoclsparse_int *__restrict__ csr_icrow,
                                              const aoclsparse_int *__restrict__ csr_idiag,
                                              const aoclsparse_int *__restrict__ csr_iurow,
                                              const T *__restrict__ x,
                                              T                     beta,
                                              T *__restrict__ y)
{
    const aoclsparse_int *csr_istart, *csr_iend;
    if(fill_mode == aoclsparse_fill_mode_lower)
    {
        csr_istart = csr_icrow;
        csr_iend   = csr_idiag;
    }
    else
    {
        csr_istart = csr_iurow;
        csr_iend   = csr_icrow + 1;
    }

    const aoclsparse_int *col_fix = csr_icol - base;
    const T              *val_fix = csr_val - base;
    const T              *x_fix   = x - base;
    T                    *y_fix   = y - base;

    if(beta == aoclsparse_numeric::zero<T>())
    {
        for(aoclsparse_int i = 0; i < m; i++)
            y[i] = aoclsparse_numeric::zero<T>();
    }
    else if(beta != static_cast<T>(1))
    {
        for(aoclsparse_int i = 0; i < m; i++)
            y[i] = beta * y[i];
    }

    for(aoclsparse_int i = 0; i < m; i++)
    {
        aoclsparse_int idxstart = csr_istart[i];
        aoclsparse_int idxend   = csr_iend[i];
        T              x_val    = x[i];
        T              sum      = aoclsparse_numeric::zero<T>();

        for(aoclsparse_int j = idxstart; j < idxend; j++)
        {
            aoclsparse_int col = col_fix[j];
            T              val = alpha * val_fix[j];
            sum += val * x_fix[col];
            y_fix[col] += val * x_val;
        }
        if(diag_type == aoclsparse_diag_type_non_unit)
            sum += alpha * val_fix[csr_idiag[i]] * x_val;
        else if(diag_type == aoclsparse_diag_type_unit)
            sum += alpha * x_val;
        y[i] += sum;
    }
    return aoclsparse_status_success;
}

/* Conjugate-transpose symmetric SpMV */
template <typename T>
static aoclsparse_status csrmvh_symm_internal(aoclsparse_index_base base,
                                               T                     alpha,
                                               aoclsparse_int        m,
                                               aoclsparse_diag_type  diag_type,
                                               aoclsparse_fill_mode  fill_mode,
                                               const T *__restrict__ csr_val,
                                               const aoclsparse_int *__restrict__ csr_icol,
                                               const aoclsparse_int *__restrict__ csr_icrow,
                                               const aoclsparse_int *__restrict__ csr_idiag,
                                               [[maybe_unused]] const aoclsparse_int *__restrict__,
                                               const T *__restrict__ x,
                                               T         beta,
                                               T *__restrict__ y)
{
    aoclsparse_int i, j, idx, idxstart, idxend;
    T              val;

    if(beta == static_cast<T>(0))
    {
        for(i = 0; i < m; i++)
            y[i] = 0.;
    }
    else if(beta != static_cast<T>(1))
    {
        for(i = 0; i < m; i++)
            y[i] = beta * y[i];
    }

    if(fill_mode == aoclsparse_fill_mode_lower)
    {
        for(i = 0; i < m; i++)
        {
            idxstart = csr_icrow[i] - base;
            idxend = csr_idiag[i] - base;
            for(idx = idxstart; idx < idxend; idx++)
            {
                val = alpha * aoclsparse::conj(csr_val[idx]);
                j   = csr_icol[idx] - base;
                y[i] += val * x[j];
                y[j] += (alpha * csr_val[idx] * x[i]);
            }
            if(diag_type == aoclsparse_diag_type_non_unit)
                y[i] += alpha * csr_val[idxend] * x[i];
            else if(diag_type == aoclsparse_diag_type_unit)
                y[i] += alpha * x[i];
        }
    }
    else
    {
        for(i = 0; i < m; i++)
        {
            idx = csr_idiag[i] - base;
            if(diag_type == aoclsparse_diag_type_non_unit)
                y[i] += alpha * csr_val[idx] * x[i];
            else if(diag_type == aoclsparse_diag_type_unit)
                y[i] += alpha * x[i];

            idxend = csr_icrow[i + 1] - base;
            for(idx = idx + 1; idx < idxend; idx++)
            {
                val = alpha * aoclsparse::conj(csr_val[idx]);
                j   = csr_icol[idx] - base;
                y[i] += val * x[j];
                y[j] += (alpha * csr_val[idx] * x[i]);
            }
        }
    }
    return aoclsparse_status_success;
}

/* Hermitian transposed SpMV */
template <typename T>
static aoclsparse_status csrmv_hermt_internal(aoclsparse_index_base base,
                                               T                     alpha,
                                               aoclsparse_int        m,
                                               aoclsparse_diag_type  diag_type,
                                               aoclsparse_fill_mode  fill_mode,
                                               const T *__restrict__ csr_val,
                                               const aoclsparse_int *__restrict__ csr_icol,
                                               const aoclsparse_int *__restrict__ csr_icrow,
                                               const aoclsparse_int *__restrict__ csr_idiag,
                                               const aoclsparse_int *__restrict__ csr_iurow,
                                               const T *__restrict__ x,
                                               T                     beta,
                                               T *__restrict__ y)
{
    aoclsparse_int i, j, idx, idxstart, idxend;
    T              val;

    if(beta == static_cast<T>(0))
    {
        for(i = 0; i < m; i++)
            y[i] = 0.;
    }
    else if(beta != static_cast<T>(1))
    {
        for(i = 0; i < m; i++)
            y[i] = beta * y[i];
    }

    if(fill_mode == aoclsparse_fill_mode_lower)
    {
        for(i = 0; i < m; i++)
        {
            idxstart = csr_icrow[i] - base;
            idxend = csr_idiag[i] - base;
            for(idx = idxstart; idx < idxend; idx++)
            {
                val = alpha * aoclsparse::conj(csr_val[idx]);
                j   = csr_icol[idx] - base;
                y[i] += val * x[j];
                y[j] += (alpha * csr_val[idx] * x[i]);
            }
            if(diag_type == aoclsparse_diag_type_non_unit)
                y[i] += alpha * csr_val[idxend] * x[i];
            else if(diag_type == aoclsparse_diag_type_unit)
                y[i] += alpha * x[i];
        }
    }
    else
    {
        for(i = 0; i < m; i++)
        {
            idx = csr_idiag[i] - base;
            if(diag_type == aoclsparse_diag_type_non_unit)
                y[i] += alpha * csr_val[idx] * x[i];
            else if(diag_type == aoclsparse_diag_type_unit)
                y[i] += alpha * x[i];

            idxend = csr_icrow[i + 1] - base;
            for(idx = idx + 1; idx < idxend; idx++)
            {
                val = alpha * aoclsparse::conj(csr_val[idx]);
                j   = csr_icol[idx] - base;
                y[i] += val * x[j];
                y[j] += (alpha * csr_val[idx] * x[i]);
            }
        }
    }
    return aoclsparse_status_success;
}

// ==========================================================================
// Main CSR SpMV dispatcher (scalar-only, portable)
// ==========================================================================

template <typename T, bool do_check = true>
aoclsparse_status aoclsparse_csrmv_t(aoclsparse_operation       trans,
                                     const T                   *alpha,
                                     aoclsparse_int             m,
                                     aoclsparse_int             n,
                                     aoclsparse_int             nnz,
                                     const T                   *val,
                                     const aoclsparse_int      *col,
                                     const aoclsparse_int      *row,
                                     const aoclsparse_mat_descr descr,
                                     const T                   *x,
                                     const T                   *beta,
                                     T                         *y,
                                     const aoclsparse_int      *idiag = nullptr,
                                     const aoclsparse_int      *iurow = nullptr,
                                     aoclsparse::doid           d_id  = aoclsparse::doid::len,
                                     aoclsparse_int             kid   = -1)
{
    using namespace aoclsparse;

    doid lcl_doid;

    if constexpr(!do_check)
    {
        lcl_doid = d_id;
    }
    else
    {
        if(alpha == nullptr || beta == nullptr)
            return aoclsparse_status_invalid_pointer;

        if(descr == nullptr)
            return aoclsparse_status_invalid_pointer;

        if(!is_valid_base(descr->base))
            return aoclsparse_status_invalid_value;

        if(!is_valid_mtx_t(descr->type))
            return aoclsparse_status_invalid_value;

        if(!is_valid_op(trans))
            return aoclsparse_status_invalid_value;

        if((descr->type != aoclsparse_matrix_type_general)
           && (descr->type != aoclsparse_matrix_type_symmetric))
        {
            return aoclsparse_status_not_implemented;
        }

        if((descr->type == aoclsparse_matrix_type_symmetric
            || descr->type == aoclsparse_matrix_type_hermitian)
           && m != n)
            return aoclsparse_status_invalid_size;

        if(m < 0 || n < 0 || nnz < 0)
            return aoclsparse_status_invalid_size;

        if(val == nullptr || row == nullptr || col == nullptr || x == nullptr || y == nullptr)
            return aoclsparse_status_invalid_pointer;

        if(descr->type == aoclsparse_matrix_type_triangular && (!idiag || !iurow))
            return aoclsparse_status_invalid_pointer;

        lcl_doid = get_doid<T>(descr, trans);
    }

    // pointers to start/end of the appropriate triangle
    const aoclsparse_int *rstart = nullptr, *rend = nullptr;
    if(lcl_doid == doid::tln || lcl_doid == doid::tlt || lcl_doid == doid::tlh
       || lcl_doid == doid::tlc)
    {
        rstart = row;
        rend   = iurow;
    }
    else if(lcl_doid == doid::tun || lcl_doid == doid::tut || lcl_doid == doid::tuh
            || lcl_doid == doid::tuc)
    {
        rstart = idiag;
        rend   = &row[1];
    }

    // Portable dispatch: all DOIDs mapped directly to scalar reference kernels
    if constexpr(is_dt_complex<T>())
    {
        switch(lcl_doid)
        {
        case doid::gn:
            return ref_csrmv_gn<T>(descr->base, *alpha, m, val, col, row, x, *beta, y);
        case doid::gt:
            return ref_csrmv_th<T, false>(descr->base, *alpha, m, n, val, col, row, x, *beta, y);
        case doid::gh:
            return ref_csrmv_th<T, true>(descr->base, *alpha, m, n, val, col, row, x, *beta, y);
        case doid::gc:
            break;
        case doid::sl:
        case doid::su:
            return csrmv_symm_internal<T>(descr->base,
                                          *alpha, m,
                                          descr->diag_type, descr->fill_mode,
                                          val, col, row, idiag, iurow,
                                          x, *beta, y);
        case doid::slc:
        case doid::suc:
            return csrmvh_symm_internal<T>(descr->base,
                                           *alpha, m,
                                           descr->diag_type, descr->fill_mode,
                                           val, col, row, idiag, iurow,
                                           x, *beta, y);
        case doid::hl:
        case doid::hu:
            return csrmv_symm_internal<T>(descr->base,
                                          *alpha, m,
                                          descr->diag_type, descr->fill_mode,
                                          val, col, row, idiag, iurow,
                                          x, *beta, y);
        case doid::hlc:
        case doid::huc:
            return csrmv_hermt_internal<T>(descr->base,
                                           *alpha, m,
                                           descr->diag_type, descr->fill_mode,
                                           val, col, row, idiag, iurow,
                                           x, *beta, y);
        case doid::tln:
        case doid::tun:
            return ref_csrmv_tri<T>(descr, *alpha, m, n, val, col, rstart, rend, x, *beta, y);
        case doid::tlt:
        case doid::tut:
            return ref_csrmv_tri_th<T, false>(descr, *alpha, m, n, val, col, rstart, rend, x, *beta, y);
        case doid::tlh:
        case doid::tuh:
            return ref_csrmv_tri_th<T, true>(descr, *alpha, m, n, val, col, rstart, rend, x, *beta, y);
        case doid::tlc:
        case doid::tuc:
            break;
        default:
            return aoclsparse_status_internal_error;
        }
        return aoclsparse_status_not_implemented;
    }
    else // Real datatypes
    {
        switch(lcl_doid)
        {
        case doid::gn:
            return ref_csrmv_gn<T>(descr->base, *alpha, m, val, col, row, x, *beta, y);
        case doid::gt:
            return ref_csrmv_th<T, false>(descr->base, *alpha, m, n, val, col, row, x, *beta, y);
        case doid::hl:
        case doid::hu:
        case doid::hlc:
        case doid::huc:
        case doid::sl:
        case doid::su:
        case doid::slc:
        case doid::suc:
            return csrmv_symm<T>(descr->base, *alpha, m, val, col, row, x, *beta, y);
        case doid::tln:
        case doid::tun:
            return ref_csrmv_tri<T>(descr, *alpha, m, n, val, col, rstart, rend, x, *beta, y);
        case doid::tlt:
        case doid::tut:
            return ref_csrmv_tri_th<T, false>(descr, *alpha, m, n, val, col, rstart, rend, x, *beta, y);
        case doid::tlh:
        case doid::tlc:
        case doid::tuh:
        case doid::tuc:
        default:
            return aoclsparse_status_not_implemented;
        }
    }
}

#endif // AOCLSPARSE_CSRMV_HPP
