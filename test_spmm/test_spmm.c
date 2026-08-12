/* Test program for aoclsparse_scsrmm (SPMM) on ARMv7-A */

#include "aoclsparse.h"
#include <stdio.h>
#include <string.h>
#include <time.h>

static int64_t timespec_diff_ns(struct timespec *end, struct timespec *start)
{
    return (end->tv_sec - start->tv_sec) * 1000000000LL
           + (end->tv_nsec - start->tv_nsec);
}

static const char *format_ns(int64_t ns)
{
    static char buf[32];
    if(ns < 1000)
        snprintf(buf, sizeof(buf), "%lld ns", (long long)ns);
    else if(ns < 1000000)
        snprintf(buf, sizeof(buf), "%.2f us", ns / 1000.0);
    else if(ns < 1000000000)
        snprintf(buf, sizeof(buf), "%.2f ms", ns / 1000000.0);
    else
        snprintf(buf, sizeof(buf), "%.2f s", ns / 1000000000.0);
    return buf;
}

#define ITERATIONS 1000

int main()
{
    aoclsparse_status    status;
    aoclsparse_matrix    A     = NULL;
    aoclsparse_mat_descr descr = NULL;
    aoclsparse_int       m = 4, k = 4, n = 2, nnz = 7;
    aoclsparse_index_base base = aoclsparse_index_base_zero;

    /* 4x4 sparse matrix:
     *   [ 1.0  0.0  2.0  0.0 ]
     *   [ 0.0  3.0  0.0  0.0 ]
     *   [ 0.0  0.0  4.0  5.0 ]
     *   [ 6.0  0.0  0.0  7.0 ]
     */
    aoclsparse_int row_ptr[] = {0, 2, 3, 5, 7};
    aoclsparse_int col_idx[] = {0, 2, 1, 2, 3, 0, 3};
    float          val[]     = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};

    /* B 4x2 dense matrix (row-major):
     *   [ 1.0  2.0 ]
     *   [ 3.0  4.0 ]
     *   [ 5.0  6.0 ]
     *   [ 7.0  8.0 ]
     */
    float B[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float C[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    float alpha = 1.0f, beta = 0.0f;

    /* Expected result: C = A * B
     *   [ 11.0  14.0 ]
     *   [  9.0  12.0 ]
     *   [ 55.0  64.0 ]
     *   [ 55.0  68.0 ]
     */
    float expected[] = {11.0f, 14.0f, 9.0f, 12.0f, 55.0f, 64.0f, 55.0f, 68.0f};

    struct timespec t1, t2;
    int64_t         elapsed_cold, elapsed_warm, elapsed_total;

    printf("=== aoclsparse_scsrmm test ===\n\n");

    /* Create matrix descriptor */
    printf("Creating matrix descriptor... ");
    status = aoclsparse_create_mat_descr(&descr);
    if(status != aoclsparse_status_success)
    {
        printf("FAILED (status=%d)\n", status);
        return 1;
    }
    aoclsparse_set_mat_type(descr, aoclsparse_matrix_type_general);
    printf("OK\n");

    /* Create CSR matrix */
    printf("Creating CSR matrix (%dx%d, nnz=%d)... ", m, k, nnz);
    status = aoclsparse_create_scsr(&A, base, m, k, nnz, row_ptr, col_idx, val);
    if(status != aoclsparse_status_success)
    {
        printf("FAILED (status=%d)\n", status);
        return 1;
    }
    printf("OK\n\n");

    /* ---- Cold call ------------------------------------------------- */
    memset(C, 0, sizeof(C));
    clock_gettime(CLOCK_MONOTONIC, &t1);
    status = aoclsparse_scsrmm(aoclsparse_operation_none,
                                alpha, A, descr, aoclsparse_order_row,
                                B, n, n, beta, C, n);
    clock_gettime(CLOCK_MONOTONIC, &t2);
    elapsed_cold = timespec_diff_ns(&t2, &t1);
    if(status != aoclsparse_status_success)
    {
        printf("FAILED (status=%d)\n", status);
        return 1;
    }

    /* ---- Warm-up + timed iterations --------------------------------- */
    clock_gettime(CLOCK_MONOTONIC, &t1);
    for(int iter = 0; iter < ITERATIONS; iter++)
    {
        memset(C, 0, sizeof(C));
        aoclsparse_scsrmm(aoclsparse_operation_none,
                          alpha, A, descr, aoclsparse_order_row,
                          B, n, n, beta, C, n);
    }
    clock_gettime(CLOCK_MONOTONIC, &t2);
    elapsed_total = timespec_diff_ns(&t2, &t1);
    elapsed_warm  = elapsed_total / ITERATIONS;

    /* ---- Timing report ---------------------------------------------- */
    printf("=== Timing (%d iterations) ===\n", ITERATIONS);
    printf("  Cold call  : %12s\n", format_ns(elapsed_cold));
    printf("  Warm avg   : %12s\n", format_ns(elapsed_warm));
    printf("  Total      : %12s\n\n", format_ns(elapsed_total));

    /* ---- Verification ----------------------------------------------- */
    printf("Result matrix C (4x2, row-major):\n");
    int pass = 1;
    for(int i = 0; i < m; i++)
    {
        printf("  Row %d: ", i);
        for(int j = 0; j < n; j++)
        {
            printf(" %8.4f", C[i * n + j]);
        }
        printf("  (expected: %8.4f %8.4f)\n", expected[i * n], expected[i * n + 1]);

        for(int j = 0; j < n; j++)
        {
            float diff = C[i * n + j] - expected[i * n + j];
            if(diff < 0.0f) diff = -diff;
            if(diff > 0.001f) pass = 0;
        }
    }

    printf("\n%s\n", pass ? "TEST PASSED" : "TEST FAILED");

    /* Cleanup */
    aoclsparse_destroy_mat_descr(descr);
    aoclsparse_destroy(&A);

    return pass ? 0 : 1;
}