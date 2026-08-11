/* ************************************************************************
 * Copyright (c) 2022-2026 Advanced Micro Devices, Inc.
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

/* Portable SpMV entry point — CSR-only scalar path.
 * All BSR/ELL/TCSR/blkCSR format paths have been removed.
 */

#include "aoclsparse_descr.h"
#include "aoclsparse.hpp"
#include "aoclsparse_csr_util.hpp"
#include "aoclsparse_csrmv.hpp"
#include "aoclsparse_error_check.hpp"
#include "aoclsparse_mat_structures.hpp"

// Inline helper: simple vector scale
template <typename T>
static aoclsparse_status vscale(T *y, T beta, aoclsparse_int dim)
{
    if(beta == static_cast<T>(0))
    {
        for(aoclsparse_int i = 0; i < dim; i++)
            y[i] = static_cast<T>(0);
    }
    else if(beta != static_cast<T>(1))
    {
        for(aoclsparse_int i = 0; i < dim; i++)
            y[i] = beta * y[i];
    }
    return aoclsparse_status_success;
}

// Inline helper: get kid from optimize hints (always -1 in portable build)
static inline aoclsparse_int get_kid_portable(aoclsparse_optimize_data * /*opt*/,
                                              aoclsparse::doid           /*doid*/,
                                              aoclsparse_hinted_action   /*act*/)
{
    return -1; // Always auto-select (scalar only)
}

// Inline: format support check (only CSR in portable build)
template <typename T>
static bool is_mtx_frmt_supported(aoclsparse_matrix_format_type fmt)
{
    return fmt == aoclsparse_csr_mat;
}

/* Templated SpMV */
template <typename T>
aoclsparse_status aoclsparse::mv(aoclsparse_operation       op,
                                 const T                   *alpha,
                                 aoclsparse_matrix          A,
                                 const aoclsparse_mat_descr descr,
                                 const T                   *x,
                                 const T                   *beta,
                                 T                         *y)
{
    using namespace aoclsparse;

    // Error handling
    if(alpha == nullptr || beta == nullptr)
        return aoclsparse_status_invalid_pointer;

    if(A == nullptr || A->mats.empty() || !A->mats[0])
        return aoclsparse_status_invalid_pointer;

    if(descr == nullptr)
        return aoclsparse_status_invalid_pointer;

    if(x == nullptr || y == nullptr)
        return aoclsparse_status_invalid_pointer;

    if(!is_valid_base(descr->base) || !is_valid_base(A->mats[0]->base))
        return aoclsparse_status_invalid_value;

    if(A->mats[0]->base != descr->base)
        return aoclsparse_status_invalid_value;

    if(!is_valid_op(op))
        return aoclsparse_status_invalid_value;

    if(A->val_type != get_data_type<T>())
        return aoclsparse_status_wrong_type;

    if(!is_valid_mtx_t(descr->type))
        return aoclsparse_status_invalid_value;

    if((descr->type == aoclsparse_matrix_type_symmetric
        || descr->type == aoclsparse_matrix_type_hermitian)
       && A->m != A->n)
        return aoclsparse_status_invalid_size;

    if(!is_mtx_frmt_supported<T>(A->input_format))
        return aoclsparse_status_not_implemented;

    if constexpr(!is_dt_complex<T>())
    {
        if(op == aoclsparse_operation_conjugate_transpose)
            op = aoclsparse_operation_transpose;

        if(descr->type == aoclsparse_matrix_type_hermitian)
            return aoclsparse_status_not_implemented;
    }

    if(A->m == 0 || A->n == 0 || (A->nnz == 0 && descr->type == aoclsparse_matrix_type_general))
    {
        aoclsparse_int dim = op == aoclsparse_operation_none ? A->m : A->n;
        return vscale(y, *beta, dim);
    }

    aoclsparse::doid d_id = aoclsparse::get_doid<T>(descr, op);
    aoclsparse_int   kid  = get_kid_portable(A->optim_data, d_id, aoclsparse_action_mv);

    // Only CSR format is supported in portable build
    aoclsparse::csr *csr_mat = dynamic_cast<aoclsparse::csr *>(A->mats[0]);
    if(!csr_mat)
        return aoclsparse_status_not_implemented;

    // DOID and descriptor adjustment
    _aoclsparse_mat_descr descr_t;
    aoclsparse_copy_mat_descr(&descr_t, descr);
    bool exact_match = false;

    descr_t.base = csr_mat->base;

    if(csr_mat->doid == d_id)
    {
        exact_match       = true;
        op                = aoclsparse_operation_none;
        descr_t.type      = aoclsparse_matrix_type_general;
        descr_t.fill_mode = aoclsparse_fill_mode_lower;
        descr_t.diag_type = aoclsparse_diag_type_non_unit;
        d_id              = doid::gn;
    }
    else
    {
        d_id = aoclsparse::get_effective_doid(csr_mat->doid, d_id);

        if(csr_mat->doid == doid::gt || csr_mat->doid == doid::gh)
        {
            if(descr_t.fill_mode == aoclsparse_fill_mode_upper)
                descr_t.fill_mode = aoclsparse_fill_mode_lower;
            else if(descr_t.fill_mode == aoclsparse_fill_mode_lower)
                descr_t.fill_mode = aoclsparse_fill_mode_upper;
        }
    }

    // CSR-specific diagonal adjustment for exact-match matrices
    if(exact_match && descr->diag_type != csr_mat->mtx_diag)
    {
        aoclsparse_status status = aoclsparse_set_mat_diag<T>(A->m, descr_t, csr_mat);
        if(status != aoclsparse_status_success)
            return status;
    }

    return aoclsparse_csrmv_t<T, false>(op,
                                        alpha,
                                        csr_mat->m,
                                        csr_mat->n,
                                        csr_mat->nnz,
                                        (T *)csr_mat->val,
                                        csr_mat->ind,
                                        csr_mat->ptr,
                                        &descr_t,
                                        x,
                                        beta,
                                        y,
                                        csr_mat->idiag,
                                        csr_mat->iurow,
                                        d_id,
                                        kid);
}

#define MV_DISPATCHER(SUF)                                                                      \
    template DLL_PUBLIC aoclsparse_status aoclsparse::mv<SUF>(aoclsparse_operation       op,    \
                                                              const SUF                 *alpha, \
                                                              aoclsparse_matrix          A,     \
                                                              const aoclsparse_mat_descr descr, \
                                                              const SUF                 *x,     \
                                                              const SUF                 *beta,  \
                                                              SUF                       *y);

INSTANTIATE_FOR_ALL_TYPES(MV_DISPATCHER);

/*
 * C wrappers
 */
extern "C" aoclsparse_status aoclsparse_smv(aoclsparse_operation       op,
                                            const float               *alpha,
                                            aoclsparse_matrix          A,
                                            const aoclsparse_mat_descr descr,
                                            const float               *x,
                                            const float               *beta,
                                            float                     *y)
{
    return aoclsparse::mv<float>(op, alpha, A, descr, x, beta, y);
}

extern "C" aoclsparse_status aoclsparse_dmv(aoclsparse_operation       op,
                                            const double              *alpha,
                                            aoclsparse_matrix          A,
                                            const aoclsparse_mat_descr descr,
                                            const double              *x,
                                            const double              *beta,
                                            double                    *y)
{
    return aoclsparse::mv<double>(op, alpha, A, descr, x, beta, y);
}

extern "C" aoclsparse_status aoclsparse_cmv(aoclsparse_operation            op,
                                            const aoclsparse_float_complex *alpha,
                                            aoclsparse_matrix               A,
                                            const aoclsparse_mat_descr      descr,
                                            const aoclsparse_float_complex *x,
                                            const aoclsparse_float_complex *beta,
                                            aoclsparse_float_complex       *y)
{
    return aoclsparse::mv<std::complex<float>>(op,
                                               ((const std::complex<float> *)alpha),
                                               A,
                                               descr,
                                               (std::complex<float> *)x,
                                               ((const std::complex<float> *)beta),
                                               (std::complex<float> *)y);
}

extern "C" aoclsparse_status aoclsparse_zmv(aoclsparse_operation             op,
                                            const aoclsparse_double_complex *alpha,
                                            aoclsparse_matrix                A,
                                            const aoclsparse_mat_descr       descr,
                                            const aoclsparse_double_complex *x,
                                            const aoclsparse_double_complex *beta,
                                            aoclsparse_double_complex       *y)
{
    return aoclsparse::mv<std::complex<double>>(op,
                                                ((const std::complex<double> *)alpha),
                                                A,
                                                descr,
                                                (std::complex<double> *)x,
                                                ((const std::complex<double> *)beta),
                                                (std::complex<double> *)y);
}
