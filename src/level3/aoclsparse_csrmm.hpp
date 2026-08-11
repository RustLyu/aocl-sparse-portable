/* ***********************************************************************
 * Copyright (c) 2021-2026 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef AOCLSPARSE_CSRMM_HPP
#define AOCLSPARSE_CSRMM_HPP

/* Portable CSRMM dispatcher — scalar reference kernels only.
 * All KT (AVX2/AVX512) kernel paths have been removed.
 */

#include "aoclsparse.h"
#include "aoclsparse_descr.h"
#include "aoclsparse_auxiliary.hpp"
#include "aoclsparse_context.hpp"
#include "aoclsparse_convert.hpp"
#include "aoclsparse_csr_util.hpp"
#include "aoclsparse_neon.hpp"
#include "aoclsparse_utils.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <vector>

// ==========================================================================
// Scalar reference kernels
// ==========================================================================

template <typename T>
static aoclsparse_status aoclsparse_csrmm_col_major_ref(T                          alpha,
                                                         const aoclsparse_mat_descr descr,
                                                         const T *__restrict__ csr_val,
                                                         const aoclsparse_int *__restrict__ csr_col_ind,
                                                         const aoclsparse_int *__restrict__ csr_row_ptr,
                                                         aoclsparse_int m,
                                                         const T       *B,
                                                         aoclsparse_int n,
                                                         aoclsparse_int ldb,
                                                         T              beta,
                                                         T             *C,
                                                         aoclsparse_int ldc)
{
    using namespace aoclsparse;
    aoclsparse_index_base base            = descr->base;
    const aoclsparse_int *csr_col_ind_fix = csr_col_ind - base;
    const T              *csr_val_fix     = csr_val - base;
    const T              *B_fix           = B - base;
#ifdef _OPENMP
#pragma omp parallel num_threads(context::get_context()->get_num_threads())
#endif
    {
#ifdef _OPENMP
        aoclsparse_int num_threads = omp_get_num_threads();
        aoclsparse_int thread_num  = omp_get_thread_num();
        aoclsparse_int start       = n * thread_num / num_threads;
        aoclsparse_int end         = n * (thread_num + 1) / num_threads;
#else
        aoclsparse_int start = 0;
        aoclsparse_int end   = n;
#endif
        for(aoclsparse_int j = start; j < end; ++j)
        {
            for(aoclsparse_int i = 0; i < m; ++i)
            {
                aoclsparse_int row_begin = csr_row_ptr[i];
                aoclsparse_int row_end   = csr_row_ptr[i + 1];
                aoclsparse_int idx_C     = i + j * ldc;
                T              sum       = 0.0;
                aoclsparse_int k     = row_begin;
                aoclsparse_int k_end = row_end;

#ifdef __ARM_NEON
                if constexpr(std::is_same_v<T, float>)
                {
                    float32x4_t vsum = aoclsparse_neon::setzero();
                    for(; k + 7 < k_end; k += 4)
                    {
                        aoclsparse_neon::prefetch_gather(
                            &B_fix[csr_col_ind_fix[k + 4] + j * ldb],
                            &B_fix[csr_col_ind_fix[k + 5] + j * ldb],
                            &B_fix[csr_col_ind_fix[k + 6] + j * ldb],
                            &B_fix[csr_col_ind_fix[k + 7] + j * ldb]);
                        float32x4_t vv = aoclsparse_neon::loadu(&csr_val_fix[k]);
                        float32x4_t vb = aoclsparse_neon::gather(
                            &B_fix[csr_col_ind_fix[k] + j * ldb],
                            &B_fix[csr_col_ind_fix[k + 1] + j * ldb],
                            &B_fix[csr_col_ind_fix[k + 2] + j * ldb],
                            &B_fix[csr_col_ind_fix[k + 3] + j * ldb]);
                        vsum = aoclsparse_neon::fmadd(vv, vb, vsum);
                    }
                    for(; k + 3 < k_end; k += 4)
                    {
                        float32x4_t vv = aoclsparse_neon::loadu(&csr_val_fix[k]);
                        float32x4_t vb = aoclsparse_neon::gather(
                            &B_fix[csr_col_ind_fix[k] + j * ldb],
                            &B_fix[csr_col_ind_fix[k + 1] + j * ldb],
                            &B_fix[csr_col_ind_fix[k + 2] + j * ldb],
                            &B_fix[csr_col_ind_fix[k + 3] + j * ldb]);
                        vsum = aoclsparse_neon::fmadd(vv, vb, vsum);
                    }
                    sum = aoclsparse_neon::hsum(vsum);
                }
                else if constexpr(std::is_same_v<T, double>)
                {
                    float64x2_t vsum = aoclsparse_neon::setzero();
                    for(; k + 3 < k_end; k += 2)
                    {
                        aoclsparse_neon::prefetch_gather(
                            &B_fix[csr_col_ind_fix[k + 2] + j * ldb],
                            &B_fix[csr_col_ind_fix[k + 3] + j * ldb]);
                        float64x2_t vv = aoclsparse_neon::loadu(&csr_val_fix[k]);
                        float64x2_t vb = aoclsparse_neon::gather(
                            &B_fix[csr_col_ind_fix[k] + j * ldb],
                            &B_fix[csr_col_ind_fix[k + 1] + j * ldb]);
                        vsum = aoclsparse_neon::fmadd(vv, vb, vsum);
                    }
                    for(; k + 1 < k_end; k += 2)
                    {
                        float64x2_t vv = aoclsparse_neon::loadu(&csr_val_fix[k]);
                        float64x2_t vb = aoclsparse_neon::gather(
                            &B_fix[csr_col_ind_fix[k] + j * ldb],
                            &B_fix[csr_col_ind_fix[k + 1] + j * ldb]);
                        vsum = aoclsparse_neon::fmadd(vv, vb, vsum);
                    }
                    sum = aoclsparse_neon::hsum(vsum);
                }
#endif
                for(; k < k_end; ++k)
                {
                    aoclsparse_int idx_B = (csr_col_ind_fix[k] + j * ldb);
                    sum                  = csr_val_fix[k] * B_fix[idx_B] + sum;
                }
                C[idx_C] = (beta * C[idx_C]) + (alpha * sum);
            }
        }
    }
    return aoclsparse_status_success;
}

template <typename T>
static aoclsparse_status aoclsparse_csrmm_row_major_ref(T                          alpha,
                                                         const aoclsparse_mat_descr descr,
                                                         const T *__restrict__ csr_val,
                                                         const aoclsparse_int *__restrict__ csr_col_ind,
                                                         const aoclsparse_int *__restrict__ csr_row_ptr,
                                                         aoclsparse_int m,
                                                         const T       *B,
                                                         aoclsparse_int n,
                                                         aoclsparse_int ldb,
                                                         T              beta,
                                                         T             *C,
                                                         aoclsparse_int ldc)
{
    using namespace aoclsparse;
    aoclsparse_index_base base            = descr->base;
    const aoclsparse_int *csr_col_ind_fix = csr_col_ind - base;
    const T              *csr_val_fix     = csr_val - base;
    const T              *B_fix           = B - (base * ldb);

#ifdef _OPENMP
#pragma omp parallel num_threads(context::get_context()->get_num_threads())
#endif
    {
#ifdef _OPENMP
        aoclsparse_int num_threads = omp_get_num_threads();
        aoclsparse_int thread_num  = omp_get_thread_num();
        aoclsparse_int start       = m * thread_num / num_threads;
        aoclsparse_int end         = m * (thread_num + 1) / num_threads;
#else
        aoclsparse_int start = 0;
        aoclsparse_int end   = m;
#endif
        for(aoclsparse_int i = start; i < end; ++i)
        {
            aoclsparse_int row_begin = csr_row_ptr[i];
            aoclsparse_int row_end   = csr_row_ptr[i + 1];
            aoclsparse_int idx_C     = i * ldc;
            for(aoclsparse_int k = 0; k < n; ++k)
            {
                C[idx_C + k] = C[idx_C + k] * beta;
            }
            for(aoclsparse_int j = row_begin; j < row_end; ++j)
            {
                aoclsparse_int idx_B       = csr_col_ind_fix[j] * ldb;
                T              val_scaled   = csr_val_fix[j] * alpha;

#ifdef __ARM_NEON
                if constexpr(std::is_same_v<T, float>)
                {
                    float32x4_t   vval = aoclsparse_neon::set1(val_scaled);
                    aoclsparse_int k   = 0;
                    for(; k + 3 < n; k += 4)
                    {
                        float32x4_t vc = aoclsparse_neon::loadu(&C[idx_C + k]);
                        float32x4_t vb = aoclsparse_neon::loadu(&B_fix[idx_B + k]);
                        vc             = aoclsparse_neon::fmadd(vval, vb, vc);
                        aoclsparse_neon::storeu(&C[idx_C + k], vc);
                    }
                    for(; k < n; ++k)
                        C[idx_C + k] += val_scaled * B_fix[idx_B + k];
                }
                else if constexpr(std::is_same_v<T, double>)
                {
                    float64x2_t   vval = aoclsparse_neon::set1(val_scaled);
                    aoclsparse_int k   = 0;
                    for(; k + 1 < n; k += 2)
                    {
                        float64x2_t vc = aoclsparse_neon::loadu(&C[idx_C + k]);
                        float64x2_t vb = aoclsparse_neon::loadu(&B_fix[idx_B + k]);
                        vc             = aoclsparse_neon::fmadd(vval, vb, vc);
                        aoclsparse_neon::storeu(&C[idx_C + k], vc);
                    }
                    for(; k < n; ++k)
                        C[idx_C + k] += val_scaled * B_fix[idx_B + k];
                }
                else
                {
                    for(aoclsparse_int k = 0; k < n; ++k)
                        C[idx_C + k] += val_scaled * B_fix[idx_B + k];
                }
#else
                for(aoclsparse_int k = 0; k < n; ++k)
                    C[idx_C + k] += val_scaled * B_fix[idx_B + k];
#endif
            }
        }
    }
    return aoclsparse_status_success;
}

template <typename T, bool HERM = false>
static aoclsparse_status aoclsparse_csrmm_sym_row_ref(T                          alpha,
                                                       const aoclsparse_mat_descr descr,
                                                       const T *__restrict__ csr_val,
                                                       const aoclsparse_int *__restrict__ csr_col_ind,
                                                       const aoclsparse_int *__restrict__ csr_row_ptr,
                                                       aoclsparse_int m,
                                                       const T       *B,
                                                       aoclsparse_int n,
                                                       aoclsparse_int ldb,
                                                       T             *C,
                                                       aoclsparse_int ldc)
{
    T                     one  = 1.0;
    aoclsparse_index_base base = descr->base;
    const aoclsparse_fill_mode fill = descr->fill_mode;
    const aoclsparse_diag_type diag = descr->diag_type;
    for(int i = 0; i < m; i++)
    {
        aoclsparse_int row_begin = csr_row_ptr[i] - base;
        aoclsparse_int row_end   = csr_row_ptr[i + 1] - base;
        if(diag == aoclsparse_diag_type_unit)
        {
            for(int j = 0; j < n; j++)
            {
                aoclsparse_int idx_c = i * ldc + j;
                aoclsparse_int idx_b = i * ldb + j;
                C[idx_c] += one * B[idx_b] * alpha;
            }
        }
        for(int k = row_begin; k < row_end; k++)
        {
            bool is_diag = (i == (csr_col_ind[k] - base));
            if(is_diag && (diag == aoclsparse_diag_type_non_unit))
            {
                for(int j = 0; j < n; j++)
                {
                    aoclsparse_int idx_c = i * ldc + j;
                    aoclsparse_int idx_b = (csr_col_ind[k] - base) * ldb + j;
                    C[idx_c] += csr_val[k] * B[idx_b] * alpha;
                }
            }
            else
            {
                if(fill == aoclsparse_fill_mode_lower)
                {
                    for(int j = 0; j < n; j++)
                    {
                        aoclsparse_int idx_c = i * ldc + j;
                        aoclsparse_int idx_b = (csr_col_ind[k] - base) * ldb + j;
                        if(i > (csr_col_ind[k] - base))
                        {
                            C[idx_c] += csr_val[k] * B[idx_b] * alpha;
                            idx_b = i * ldb + j;
                            idx_c = (csr_col_ind[k] - base) * ldc + j;
                            if constexpr(HERM)
                                C[idx_c] += aoclsparse::conj(csr_val[k]) * (B[idx_b]) * alpha;
                            else
                                C[idx_c] += csr_val[k] * (B[idx_b]) * alpha;
                        }
                    }
                }
                else
                {
                    for(int j = 0; j < n; j++)
                    {
                        aoclsparse_int idx_c = i * ldc + j;
                        aoclsparse_int idx_b = (csr_col_ind[k] - base) * ldb + j;
                        if(i < (csr_col_ind[k] - base))
                        {
                            C[idx_c] += csr_val[k] * B[idx_b] * alpha;
                            idx_b = i * ldb + j;
                            idx_c = (csr_col_ind[k] - base) * ldc + j;
                            if constexpr(HERM)
                                C[idx_c] += aoclsparse::conj(csr_val[k]) * (B[idx_b]) * alpha;
                            else
                                C[idx_c] += csr_val[k] * (B[idx_b]) * alpha;
                        }
                    }
                }
            }
        }
    }
    return aoclsparse_status_success;
}

template <typename T, bool HERM = false>
static aoclsparse_status aoclsparse_csrmm_sym_col_ref(T                          alpha,
                                                       const aoclsparse_mat_descr descr,
                                                       const T *__restrict__ csr_val,
                                                       const aoclsparse_int *__restrict__ csr_col_ind,
                                                       const aoclsparse_int *__restrict__ csr_row_ptr,
                                                       aoclsparse_int m,
                                                       const T       *B,
                                                       aoclsparse_int n,
                                                       aoclsparse_int ldb,
                                                       T             *C,
                                                       aoclsparse_int ldc)
{
    T                     one  = 1.0;
    aoclsparse_index_base base = descr->base;
    const aoclsparse_fill_mode fill = descr->fill_mode;
    const aoclsparse_diag_type diag = descr->diag_type;
    for(int i = 0; i < m; i++)
    {
        aoclsparse_int row_begin = csr_row_ptr[i] - base;
        aoclsparse_int row_end   = csr_row_ptr[i + 1] - base;
        if(diag == aoclsparse_diag_type_unit)
        {
            for(int j = 0; j < n; j++)
            {
                aoclsparse_int idx_c = i + j * ldc;
                aoclsparse_int idx_b = i + j * ldb;
                C[idx_c] += one * B[idx_b] * alpha;
            }
        }
        for(int k = row_begin; k < row_end; k++)
        {
            bool is_diag = (i == (csr_col_ind[k] - base));
            if(is_diag && (diag == aoclsparse_diag_type_non_unit))
            {
                for(int j = 0; j < n; j++)
                {
                    aoclsparse_int idx_c = i + j * ldc;
                    aoclsparse_int idx_b = (csr_col_ind[k] - base) + j * ldb;
                    C[idx_c] += csr_val[k] * B[idx_b] * alpha;
                }
            }
            else
            {
                if(fill == aoclsparse_fill_mode_lower)
                {
                    for(int j = 0; j < n; j++)
                    {
                        aoclsparse_int idx_c = i + j * ldc;
                        aoclsparse_int idx_b = (csr_col_ind[k] - base) + j * ldb;
                        if(i > (csr_col_ind[k] - base))
                        {
                            C[idx_c] += csr_val[k] * B[idx_b] * alpha;
                            idx_b = i + j * ldb;
                            idx_c = (csr_col_ind[k] - base) + j * ldc;
                            if constexpr(HERM)
                                C[idx_c] += aoclsparse::conj(csr_val[k]) * (B[idx_b]) * alpha;
                            else
                                C[idx_c] += csr_val[k] * (B[idx_b]) * alpha;
                        }
                    }
                }
                else
                {
                    for(int j = 0; j < n; j++)
                    {
                        aoclsparse_int idx_c = i + j * ldc;
                        aoclsparse_int idx_b = (csr_col_ind[k] - base) + j * ldb;
                        if(i < (csr_col_ind[k] - base))
                        {
                            C[idx_c] += csr_val[k] * B[idx_b] * alpha;
                            idx_b = i + j * ldb;
                            idx_c = (csr_col_ind[k] - base) + j * ldc;
                            if constexpr(HERM)
                                C[idx_c] += aoclsparse::conj(csr_val[k]) * (B[idx_b]) * alpha;
                            else
                                C[idx_c] += csr_val[k] * (B[idx_b]) * alpha;
                        }
                    }
                }
            }
        }
    }
    return aoclsparse_status_success;
}

template <typename T>
static aoclsparse_status scale_dense_matrix(
    aoclsparse_order order, T *mtrx, aoclsparse_int m, aoclsparse_int n, aoclsparse_int ld, T beta)
{
    using namespace aoclsparse;
    if(beta == aoclsparse_numeric::zero<T>())
    {
        if(order == aoclsparse_order_column)
        {
#ifdef _OPENMP
#pragma omp parallel for num_threads(context::get_context()->get_num_threads())
#endif
            for(aoclsparse_int j = 0; j < n; ++j)
                for(aoclsparse_int i = 0; i < m; ++i)
                    mtrx[i + j * ld] = 0;
        }
        else
        {
#ifdef _OPENMP
#pragma omp parallel for num_threads(context::get_context()->get_num_threads())
#endif
            for(aoclsparse_int i = 0; i < m; ++i)
                for(aoclsparse_int j = 0; j < n; ++j)
                    mtrx[i * ld + j] = 0;
        }
    }
    else
    {
        if(order == aoclsparse_order_column)
        {
#ifdef _OPENMP
#pragma omp parallel for num_threads(context::get_context()->get_num_threads())
#endif
            for(aoclsparse_int j = 0; j < n; ++j)
                for(aoclsparse_int i = 0; i < m; ++i)
                    mtrx[i + j * ld] = beta * mtrx[i + j * ld];
        }
        else
        {
#ifdef _OPENMP
#pragma omp parallel for num_threads(context::get_context()->get_num_threads())
#endif
            for(aoclsparse_int i = 0; i < m; ++i)
                for(aoclsparse_int j = 0; j < n; ++j)
                    mtrx[i * ld + j] = beta * mtrx[i * ld + j];
        }
    }
    return aoclsparse_status_success;
}

// ==========================================================================
// Main CSRMM dispatcher (scalar-only, portable)
// ==========================================================================

template <typename T>
aoclsparse_status aoclsparse_csrmm_t(aoclsparse_operation       op,
                                     const T                    alpha,
                                     const aoclsparse_matrix    A,
                                     const aoclsparse_mat_descr descr,
                                     aoclsparse_order           order,
                                     const T                   *B,
                                     aoclsparse_int             n,
                                     aoclsparse_int             ldb,
                                     const T                    beta,
                                     T                         *C,
                                     aoclsparse_int             ldc,
                                     aoclsparse_int             /*kid*/)
{
    using namespace aoclsparse;

    if(A == nullptr || A->mats.empty() || B == nullptr || C == nullptr || descr == nullptr)
        return aoclsparse_status_invalid_pointer;

    if(A->input_format != aoclsparse_csr_mat)
        return aoclsparse_status_not_implemented;

    if(op != aoclsparse_operation_none && op != aoclsparse_operation_transpose
       && op != aoclsparse_operation_conjugate_transpose)
        return aoclsparse_status_invalid_value;

    if(descr->type != aoclsparse_matrix_type_general
       && descr->type != aoclsparse_matrix_type_symmetric
       && descr->type != aoclsparse_matrix_type_hermitian)
        return aoclsparse_status_not_implemented;

    if((descr->type == aoclsparse_matrix_type_symmetric
        || descr->type == aoclsparse_matrix_type_hermitian)
       && A->m != A->n)
        return aoclsparse_status_invalid_size;

    if(order != aoclsparse_order_row && order != aoclsparse_order_column)
        return aoclsparse_status_invalid_value;

    T zero{0.0};
    T one{1.0};

    aoclsparse_int m = A->m;
    aoclsparse_int k = A->n;
    aoclsparse_int m_c{0}, n_c{0};

    aoclsparse::csr *csr_mat = dynamic_cast<aoclsparse::csr *>(A->mats[0]);
    if(!csr_mat)
        return aoclsparse_status_not_implemented;

    if(csr_mat->doid != aoclsparse::doid::gn)
        return aoclsparse_status_not_implemented;

    const aoclsparse_int *csr_col_ind = csr_mat->ind;
    const aoclsparse_int *csr_row_ptr = csr_mat->ptr;
    const T              *csr_val     = static_cast<T *>(csr_mat->val);

    const aoclsparse_matrix_type mat_type = descr->type;

    if(A->val_type != get_data_type<T>())
        return aoclsparse_status_wrong_type;

    if(descr->base != aoclsparse_index_base_zero && descr->base != aoclsparse_index_base_one)
        return aoclsparse_status_invalid_value;

    if(csr_mat->base != descr->base)
        return aoclsparse_status_invalid_value;

    if(m < 0 || n < 0 || k < 0)
        return aoclsparse_status_invalid_size;

    if(m == 0 || n == 0 || k == 0)
        return aoclsparse_status_success;

    if(alpha == zero && beta == one)
        return aoclsparse_status_success;

    if(csr_val == nullptr || csr_row_ptr == nullptr || csr_col_ind == nullptr)
        return aoclsparse_status_invalid_pointer;

    aoclsparse_int check_ldb;
    if(op == aoclsparse_operation_none)
        check_ldb = (order == aoclsparse_order_column ? k : n);
    else
        check_ldb = (order == aoclsparse_order_column ? m : n);
    if(ldb < (((aoclsparse_int)1) >= check_ldb ? (aoclsparse_int)1 : check_ldb))
        return aoclsparse_status_invalid_size;

    aoclsparse_int check_ldc;
    if(op == aoclsparse_operation_none)
        check_ldc = (order == aoclsparse_order_column ? m : n);
    else
        check_ldc = (order == aoclsparse_order_column ? k : n);
    if(ldc < (((aoclsparse_int)1) >= check_ldc ? (aoclsparse_int)1 : check_ldc))
        return aoclsparse_status_invalid_size;

    if(op == aoclsparse_operation_none)
        m_c = m;
    else
        m_c = k;
    n_c = n;

    T                          *val_A;
    aoclsparse_int             *col_ind_A;
    aoclsparse_int             *row_ptr_A;
    std::vector<aoclsparse_int> csr_row_ptr_A;
    std::vector<aoclsparse_int> csr_col_ind_A;
    std::vector<T>              csr_val_A;
    aoclsparse_int              mb;
    aoclsparse_status           status;
    bool                         mat_found = false;
    _aoclsparse_mat_descr        descr_t;
    aoclsparse_copy_mat_descr(&descr_t, descr);
    aoclsparse::doid d_id = aoclsparse::get_doid<T>(descr, op);
    mb                    = m;

    // Overflow check
    {
        aoclsparse_int c_dim, b_dim;
        aoclsparse_int b_rows = (op == aoclsparse_operation_none) ? k : m;

        if(order == aoclsparse_order_column)
        {
            c_dim = n;
            b_dim = n;
        }
        else
        {
            c_dim = m_c;
            b_dim = b_rows;
        }
        if(aoclsparse_lp64_product_overflow(c_dim, ldc)
           || aoclsparse_lp64_product_overflow(b_dim, ldb))
            return aoclsparse_status_invalid_size;
    }

    if(alpha == zero)
    {
        status = scale_dense_matrix(order, C, m_c, n_c, ldc, beta);
        return status;
    }

    if(A->input_format != aoclsparse_csr_mat)
        return aoclsparse_status_not_implemented;

    for(auto mat : A->mats)
    {
        aoclsparse::csr *csr_m = dynamic_cast<aoclsparse::csr *>(mat);
        if(csr_m != nullptr && mat->doid == d_id)
        {
            val_A     = (T *)csr_m->val;
            col_ind_A = csr_m->ind;
            row_ptr_A = csr_m->ptr;
            mb        = csr_m->m;

            if(descr_t.diag_type != mat->mtx_diag)
            {
                status = aoclsparse_set_mat_diag<T>(A->m, descr_t, csr_m);
                if(status != aoclsparse_status_success)
                    return status;
            }
            op                = aoclsparse_operation_none;
            descr_t.type      = aoclsparse_matrix_type_general;
            descr_t.fill_mode = aoclsparse_fill_mode_lower;
            descr_t.base      = csr_m->base;
            mat_found         = true;
            d_id              = doid::gn;
            break;
        }
    }

    switch(d_id)
    {
    case doid::sl:
    case doid::su:
    case doid::slc:
    case doid::suc:
    case doid::hl:
    case doid::hu:
    case doid::hlc:
    case doid::huc:
        status = scale_dense_matrix(order, C, m_c, n_c, ldc, beta);
        if(status != aoclsparse_status_success)
            return status;

        val_A = const_cast<T *>(csr_val);
        if(op != aoclsparse_operation_none)
        {
            try
            {
                csr_val_A.resize(A->nnz);
            }
            catch(std::bad_alloc &)
            {
                return aoclsparse_status_memory_error;
            }
            for(aoclsparse_int idx = 0; idx < A->nnz; idx++)
            {
                if constexpr(std::is_same_v<T, std::complex<double>>
                             || std::is_same_v<T, std::complex<float>>)
                {
                    if(d_id == doid::slc || d_id == doid::suc || d_id == doid::hlc
                       || d_id == doid::huc)
                        csr_val_A[idx] = aoclsparse::conj(csr_val[idx]);
                    else
                        csr_val_A[idx] = csr_val[idx];
                }
                else
                    csr_val_A[idx] = csr_val[idx];
            }
            val_A = csr_val_A.data();
        }
        if(mat_type == aoclsparse_matrix_type_symmetric)
        {
            if(order == aoclsparse_order_column)
                return aoclsparse_csrmm_sym_col_ref<T>(
                    alpha, descr, val_A, csr_col_ind, csr_row_ptr, k, B, n, ldb, C, ldc);
            else
                return aoclsparse_csrmm_sym_row_ref<T>(
                    alpha, descr, val_A, csr_col_ind, csr_row_ptr, k, B, n, ldb, C, ldc);
        }
        else
        {
            if(order == aoclsparse_order_column)
                return aoclsparse_csrmm_sym_col_ref<T, true>(
                    alpha, descr, val_A, csr_col_ind, csr_row_ptr, k, B, n, ldb, C, ldc);
            else
                return aoclsparse_csrmm_sym_row_ref<T, true>(
                    alpha, descr, val_A, csr_col_ind, csr_row_ptr, k, B, n, ldb, C, ldc);
        }
        break;
    case doid::gn:
    case doid::gt:
    case doid::gh:
        if(mat_found)
            break;
        row_ptr_A = const_cast<aoclsparse_int *>(csr_row_ptr);
        col_ind_A = const_cast<aoclsparse_int *>(csr_col_ind);
        val_A     = const_cast<T *>(csr_val);
        mb        = m;

        if(d_id == doid::gt || d_id == doid::gh)
        {
            try
            {
                csr_col_ind_A.resize(A->nnz);
                csr_row_ptr_A.resize(A->n + 1);
                csr_val_A.resize(A->nnz);
            }
            catch(std::bad_alloc &)
            {
                return aoclsparse_status_memory_error;
            }

            aoclsparse_status st = aoclsparse_csr2csc_template(A->m,
                                                               A->n,
                                                               A->nnz,
                                                               descr->base,
                                                               descr->base,
                                                               csr_row_ptr,
                                                               csr_col_ind,
                                                               csr_val,
                                                               csr_col_ind_A.data(),
                                                               csr_row_ptr_A.data(),
                                                               csr_val_A.data());
            if(st != aoclsparse_status_success)
                return aoclsparse_status_internal_error;

            if(d_id == doid::gh)
            {
                for(aoclsparse_int idx = 0; idx < A->nnz; idx++)
                    csr_val_A[idx] = aoclsparse::conj(csr_val_A[idx]);
            }
            row_ptr_A = csr_row_ptr_A.data();
            col_ind_A = csr_col_ind_A.data();
            val_A     = csr_val_A.data();
            mb        = k;
        }
        break;
    default:
        return aoclsparse_status_not_implemented;
    }

    // Portable: always use scalar reference kernel
    if(order == aoclsparse_order_column)
        return aoclsparse_csrmm_col_major_ref<T>(
            alpha, &descr_t, val_A, col_ind_A, row_ptr_A, mb, B, n, ldb, beta, C, ldc);
    else
        return aoclsparse_csrmm_row_major_ref<T>(
            alpha, &descr_t, val_A, col_ind_A, row_ptr_A, mb, B, n, ldb, beta, C, ldc);
}

#endif /* AOCLSPARSE_CSRMM_HPP */
