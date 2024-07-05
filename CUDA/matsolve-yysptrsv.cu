// ref:
// https://github.com/NVIDIA/CUDALibrarySamples/blob/master/cuSPARSE/spsv_csr/spsv_csr_example.c

#include "common.h"
#include "mmio.h"
#include "read_mtx.h"
#include "tranpose.h"
#include "YYSpTRSV.h"

#include "ArrayUtils.hpp"

using namespace uni;

#define CHECK_CUDA(func)                                               \
    {                                                                  \
        cudaError_t status = (func);                                   \
        if (status != cudaSuccess)                                     \
        {                                                              \
            printf("CUDA API failed at line %d with error: %s (%d)\n", \
                   __LINE__, cudaGetErrorString(status), status);      \
            while (1)                                                  \
                ;                                                      \
        }                                                              \
    }

#define CHECK_CUSPARSE(func)                                               \
    {                                                                      \
        cusparseStatus_t status = (func);                                  \
        if (status != CUSPARSE_STATUS_SUCCESS)                             \
        {                                                                  \
            printf("CUSPARSE API failed at line %d with error: %s (%d)\n", \
                   __LINE__, cusparseGetErrorString(status), status);      \
            while (1)                                                      \
                ;                                                          \
        }                                                                  \
    }

using cusp_int = int;
#define my_CUSPARSE_INDEX CUSPARSE_INDEX_32I

#define MAX_DOF_TEST 1

struct benchmark_record
{
    double total_time = 0;
    long flops = 0;
    long bytes = 0;
    long count = 0;
};

benchmark_record benchmark_record_map_lower[MAX_DOF_TEST];
benchmark_record benchmark_record_map_upper[MAX_DOF_TEST];

template <int Dim = 3, int stencil_type = 0, int stencil_width>
void RunBenchmarkLowerWithCusparse(cusp_int M, cusp_int N, cusp_int P, int Dof)
{
    std::string dof_str = std::to_string(Dof);
    // log::FunctionBegin("RunBenchmark_Dof_" + dof_str + "_WithCusparse");
    // cusp_int M = json[dof_str]["M"].get<cusp_int>();
    // cusp_int N = json[dof_str]["N"].get<cusp_int>();
    // cusp_int P = json[dof_str]["P"].get<cusp_int>();

    // for ilu(1)
    // StencilPattern<Dim> stencil_pattern(
    //     ((stencil_type == 0 || stencil_type == 2) ? STENCIL_STAR : STENCIL_BOX),
    //     stencil_width);
    // StencilPattern<Dim> lower_fill_pattern =
    //     stencil_pattern.GetFillInPattern(1).GetLowerWithDiagPattern();

    std::vector<std::array<cusp_int, Dim>> stencil_points;
    if constexpr (stencil_type == 0)
    {
        for (int d = Dim - 1; d >= 0; d--)
        {
            for (int j = stencil_width; j > 0; j--)
            {
                std::array<cusp_int, Dim> pt = {0, 0, 0};
                pt[d] = -j;
                stencil_points.push_back(pt);
            }
        }
        stencil_points.push_back(std::array<cusp_int, Dim>{0, 0, 0});
    }
    // else if constexpr (stencil_type == 1) {
    //     NestedLoop(
    //         constant_array<cusp_int, Dim>(-stencil_width),
    //         constant_array<cusp_int, Dim>(2 * stencil_width + 1), [&](auto pt) {
    //             IndexInt cnt = CartToFlat(
    //                 pt + stencil_width,
    //                 constant_array<IndexInt, Dim>(2 * stencil_width + 1));
    //             if (cnt < (myPow(2 * stencil_width + 1, Dim) / 2)) {
    //                 stencil_points.push_back(pt);
    //             }
    //         });
    //     stencil_points.push_back(std::array<cusp_int, Dim>{0, 0, 0});
    // } else {
    //     for (int cnt = 0; cnt < lower_fill_pattern.length; cnt++) {
    //         stencil_points.push_back(lower_fill_pattern.points(cnt));
    //     }
    // }

    // Host problem definition
    cusp_int A_num_rows = M * N * P * Dof;
    cusp_int A_num_cols = M * N * P * Dof;
    cusp_int A_nnz = 0;
    std::vector<cusp_int> hA_csrOffsets;
    std::vector<cusp_int> hA_columns;
    std::vector<double> hA_values;
    std::vector<double> hX;
    std::vector<double> hY;
    std::vector<double> hY_result;
    double alpha = 1.0f;
    // 注意这里求解的是A* Y = X, 所以这里的Y是输出, X是输入

    // set A & hX
    NestedLoop(
        std::array<cusp_int, Dim>{}, std::array<cusp_int, Dim>{M, N, P},
        [&](auto loc)
        {
            for (int d = 0; d < Dof; d++)
            {
                hA_csrOffsets.push_back(A_nnz);
                cusp_int cnt = 0;
                for (auto pt : stencil_points)
                {
                    if (in_range(loc + pt, std::array<cusp_int, Dim>{},
                                 std::array<cusp_int, Dim>{M, N, P} - 1))
                    {
                        for (int k = 0; k < Dof; k++)
                        {
                            hA_columns.push_back(
                                CartToFlat(loc + pt,
                                           std::array<cusp_int, Dim>{M, N, P}) *
                                    Dof +
                                k);
                            hA_values.push_back(1.);
                            A_nnz++;
                            cnt++;
                        }
                    }
                }
                hX.push_back(cnt);
            }
        });
    hA_csrOffsets.push_back(A_nnz);

    std::cout << "A_nnz = " << A_nnz << std::endl;

    // set hY
    hY.resize(A_num_cols);
    hY_result.resize(A_num_rows);
    for (cusp_int i = 0; i < A_num_cols; i++)
        hY_result[i] = 1.0;

    //--------------------------------------------------------------------------

    /* The border between thread-level and warp-level algorithms, according to
     * the number of non-zero elements in each row of the matrix L*/
    int border = 10;

    /* !!!!!! start computing SpTRSV !!!!!!!! */
    double solve_time, gflops, bandwith, pre_time, warp_occupy, element_occupy;
    int success = YYSpTRSV_csr(
        A_num_rows, A_num_cols, A_nnz, hA_csrOffsets.data(), hA_columns.data(),
        hA_values.data(), hX.data(), hY.data(), border, &solve_time, &gflops,
        &bandwith, &pre_time, &warp_occupy, &element_occupy);

    long readBytes = (sizeof(cusp_int) + sizeof(double)) * A_nnz +
                     sizeof(cusp_int) * A_num_rows +
                     sizeof(double) * A_num_cols;
    long writeBytes = sizeof(double) * A_num_rows;

    // double timing = 0;
    // log::FunctionBegin("Timing");
    // timing
    // for (int i = 0; i < 10; i++) {
    //     log::FunctionBegin("cusparseSpMV");
    //     double time_0 = MPI_Wtime();
    //     CHECK_CUSPARSE(cusparseSpSV_solve(
    //         handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecX,
    //         vecY, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, spsvDescr))
    //     CHECK_CUDA(cudaDeviceSynchronize())
    //     double time_1 = MPI_Wtime();
    //     log::FunctionEnd(2 * A_nnz, readBytes, writeBytes);
    //     timing += time_1 - time_0;
    // }
    // log::FunctionEnd(0, 0, 0);
    benchmark_record_map_lower[Dof - 1] = {solve_time, 2 * A_nnz,
                                           (readBytes + writeBytes), 1};
    std::cout
        << "LowerTime: " << solve_time << ", Gflops: " << gflops << std::endl;

    //--------------------------------------------------------------------------
    // device result check

    int correct = 1;
    for (cusp_int i = 0; i < A_num_rows; i++)
    {
        if (hY[i] !=
            hY_result[i])
        {                // direct doubleing point comparison is not
            correct = 0; // reliable
            // break;
            std::cout << "i = " << i << ", hY[i] = " << hY[i]
                      << ", hY_result[i] = " << hY_result[i] << std::endl;
        }
    }
    if (correct)
        printf("spmv_csr_example test PASSED\n");
    else
        printf("spmv_csr_example test FAILED: wrong result\n");
    //--------------------------------------------------------------------------
    // device memory deallocation
    // log::FunctionEnd(0, 0, 0);
}

int main(int argc, char **argv)
{
    // Json json =
    //     LoadJsonFromFile("example/structure-benchmark/matsolve-yysptrsv.json");
    // std::string platform = json["platform"];
    // std::string remark = json["remark"];
    // std::string problems[] = {"stencilstar", "stencilbox", "stencilstarfill1"};
    // bool if_output = json["output"];
    // MPI_Init(&argc, &argv);
    // int rank, size;
    // MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // MPI_Comm_size(MPI_COMM_WORLD, &size);
    // loop<int, 1>([&](auto i) {
    // loop<int, (1)>([&](auto stencil_width_0) {
    // constexpr int stencil_width = stencil_width_0 + 1;
    // std::string problem = problems[i];
    // if (if_output) {
    //     Logger::InitSingleOutput(
    //         rank, std::string{"results/matsolve-yysptrsv-"} + problem +
    //                   "-stencilwidth" + std::to_string(stencil_width) +
    //                   "-" + platform + "-" + remark + ".out");
    // } else {
    //     Logger::Init();
    // }
    // loop<int, MAX_DOF_TEST>([&](auto dof) {
    // Logger::output(ConsoleAndRank)
    //     << problem << ", width=" << stencil_width
    //     << ", dof=" << dof + 1 << std::endl;
    // log::FunctionBegin(problem + "_lower");
    RunBenchmarkLowerWithCusparse<3, 0, 1>(
        192, 192, 192, 1);
    // log::FunctionEnd(0, 0, 0);
    // Logger::output(ConsoleAndRank) << "Lower:";
    // double total_time = benchmark_record_map_lower[dof].total_time +
    //                     benchmark_record_map_upper[dof].total_time;
    // double total_flops_time =
    //     static_cast<double>(benchmark_record_map_lower[dof].flops +
    //                         benchmark_record_map_upper[dof].flops) /
    //     total_time;
    // double total_bytes_time =
    //     static_cast<double>(benchmark_record_map_lower[dof].bytes +
    //                         benchmark_record_map_upper[dof].bytes) /
    //     total_time;

    // Logger::output(ConsoleAndRank)
    //     << dof + 1 << "," << total_time << ","
    //     << total_flops_time * 1e-9 << "," << total_bytes_time * 1e-9
    //     << std::endl;
    // });
    // log::ReportResult(ConsoleAndRank);
    // log::ClearReportResult();
    // Logger::Finalize();
    // });
    // });
    // MPI_Finalize();
    return 0;
}
