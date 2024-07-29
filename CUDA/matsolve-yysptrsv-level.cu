// ref:
// https://github.com/NVIDIA/CUDALibrarySamples/blob/master/cuSPARSE/spsv_csr/spsv_csr_example.c

#include <fstream>
#include <iostream>
#include <string>

#include "common.h"
#include "mmio.h"
#include "read_mtx.h"
#include "tranpose.h"

#include "YYSpTRSV.h"

#include "unisolver/ArrayUtils.hpp"
#include "unisolver/JsonUtils.hpp"

using namespace uni;

#define CHECK_CUDA(func)                                               \
    {                                                                  \
        cudaError_t status = (func);                                   \
        if (status != cudaSuccess) {                                   \
            printf("CUDA API failed at line %d with error: %s (%d)\n", \
                   __LINE__, cudaGetErrorString(status), status);      \
            while (1);                                                 \
        }                                                              \
    }

#define CHECK_CUSPARSE(func)                                               \
    {                                                                      \
        cusparseStatus_t status = (func);                                  \
        if (status != CUSPARSE_STATUS_SUCCESS) {                           \
            printf("CUSPARSE API failed at line %d with error: %s (%d)\n", \
                   __LINE__, cusparseGetErrorString(status), status);      \
            while (1);                                                     \
        }                                                                  \
    }

using cusp_int = int;
#define my_CUSPARSE_INDEX CUSPARSE_INDEX_32I

#define MAX_DOF_TEST 8

struct benchmark_record {
    double total_time = 0;
    long flops = 0;
    long bytes = 0;
    long count = 0;
};

benchmark_record benchmark_record_map_lower[MAX_DOF_TEST];

void RunBenchmarkLowerWithCusparse(Json json, int Dof, int stencil_type,
                                   int stencil_width) {
    constexpr int Dim = 3;

    std::string dof_str = std::to_string(Dof);
    cusp_int M = json[dof_str]["M"].get<cusp_int>();
    cusp_int N = json[dof_str]["N"].get<cusp_int>();
    cusp_int P = json[dof_str]["P"].get<cusp_int>();

    std::vector<std::array<cusp_int, Dim>> stencil_points;
    if (stencil_type == 0) {
        for (int d = Dim - 1; d >= 0; d--) {
            for (int j = stencil_width; j > 0; j--) {
                std::array<cusp_int, Dim> pt = {0, 0, 0};
                pt[d] = -j;
                stencil_points.push_back(pt);
            }
        }
        stencil_points.push_back(std::array<cusp_int, Dim>{0, 0, 0});
    } else if (stencil_type == 1) {
        NestedLoop(
            constant_array<cusp_int, Dim>(-stencil_width),
            constant_array<cusp_int, Dim>(2 * stencil_width + 1), [&](auto pt) {
                cusp_int cnt = CartToFlat(
                    pt + stencil_width,
                    constant_array<cusp_int, Dim>(2 * stencil_width + 1));
                if (cnt < (myPow(2 * stencil_width + 1, Dim) / 2)) {
                    stencil_points.push_back(pt);
                }
            });
        stencil_points.push_back(std::array<cusp_int, Dim>{0, 0, 0});
    } else {
        stencil_points.push_back(std::array<cusp_int, Dim>{0, 0, -1});
        stencil_points.push_back(std::array<cusp_int, Dim>{1, 0, -1});
        stencil_points.push_back(std::array<cusp_int, Dim>{0, 1, -1});
        stencil_points.push_back(std::array<cusp_int, Dim>{0, -1, 0});
        stencil_points.push_back(std::array<cusp_int, Dim>{1, -1, 0});
        stencil_points.push_back(std::array<cusp_int, Dim>{-1, 0, 0});
        stencil_points.push_back(std::array<cusp_int, Dim>{0, 0, 0});
    }

    // Host problem definition
    cusp_int A_num_rows = M * N * P * Dof;
    cusp_int A_nnz = 0;
    std::vector<cusp_int> hA_csrOffsets;
    std::vector<cusp_int> hA_columns;
    std::vector<double> hA_values;
    std::vector<double> hX;
    std::vector<double> hY;
    std::vector<double> hY_result;
    // 注意这里求解的是A* Y = X, 所以这里的Y是输出, X是输入

    auto get_level = [stencil_type](cusp_int x, cusp_int y, cusp_int z) {
        if (stencil_type == 0) {
            return x + y + z;
        } else {
            return x + 2 * y + 4 * z;
        }
    };

    cusp_int num_levels = get_level(M - 1, N - 1, P - 1) + 1;
    std::vector<cusp_int> level_count(num_levels, 0);
    std::vector<cusp_int> level_start(num_levels, 0);
    std::vector<cusp_int> origin_id_to_id_in_level(M * N * P);
    std::vector<cusp_int> level_id_to_origin_id_map(M * N * P);
    std::vector<cusp_int> origin_id_to_level_id_map(M * N * P);

    NestedLoop(std::array<cusp_int, Dim>{}, std::array<cusp_int, Dim>{M, N, P},
               [&](auto loc) {
                   cusp_int level = get_level(loc[0], loc[1], loc[2]);
                   origin_id_to_id_in_level[CartToFlat(
                       loc, std::array<cusp_int, Dim>{M, N, P})] =
                       level_count[level];
                   level_count[level]++;
               });

    for (cusp_int i = 1; i < num_levels; i++) {
        level_start[i] = level_start[i - 1] + level_count[i - 1];
    }

    NestedLoop(
        std::array<cusp_int, Dim>{}, std::array<cusp_int, Dim>{M, N, P},
        [&](auto loc) {
            cusp_int level = get_level(loc[0], loc[1], loc[2]);
            cusp_int id = CartToFlat(loc, std::array<cusp_int, Dim>{M, N, P});
            cusp_int id_in_level = origin_id_to_id_in_level[id];
            level_id_to_origin_id_map[level_start[level] + id_in_level] = id;
            origin_id_to_level_id_map[id] = level_start[level] + id_in_level;
        });

    // std::cout << "level_count = ";
    // for (int i = 0; i < num_levels; i++) {
    //     std::cout << level_count[i] << " ";
    // }
    // std::cout << "\n";
    // std::cout << "level_start = ";
    // for (int i = 0; i < num_levels; i++) {
    //     std::cout << level_start[i] << " ";
    // }
    // std::cout << "\n";
    // std::cout << "origin_id_to_id_in_level = ";
    // for (int i = 0; i < M * N * P; i++) {
    //     std::cout << origin_id_to_id_in_level[i] << " ";
    // }
    // std::cout << "\n";
    // std::cout << "level_id_to_origin_id_map = ";
    // for (int i = 0; i < M * N * P; i++) {
    //     std::cout << level_id_to_origin_id_map[i] << " ";
    // }
    // std::cout << "\n";
    // std::cout << "origin_id_to_level_id_map = ";
    // for (int i = 0; i < M * N * P; i++) {
    //     std::cout << origin_id_to_level_id_map[i] << " ";
    // }
    // std::cout << "\n";

    // set A & hX

    for (cusp_int i = 0; i < M * N * P; i++) {
        auto loc = FlatToCart(level_id_to_origin_id_map[i],
                              std::array<cusp_int, Dim>{M, N, P});
        for (int d = 0; d < Dof; d++) {
            hA_csrOffsets.push_back(A_nnz);
            cusp_int cnt = 0;
            for (auto pt : stencil_points) {
                if (in_range(loc + pt, std::array<cusp_int, Dim>{},
                             std::array<cusp_int, Dim>{M, N, P} - 1)) {
                    for (int k = 0; k < Dof; k++) {
                        if (pt != std::array<cusp_int, Dim>{0, 0, 0} ||
                            k == d) {
                            hA_columns.push_back(
                                origin_id_to_level_id_map[CartToFlat(
                                    loc + pt,
                                    std::array<cusp_int, Dim>{M, N, P})] *
                                    Dof +
                                k);
                            hA_values.push_back(1.);
                            A_nnz++;
                            cnt++;
                        }
                    }
                }
            }
            hX.push_back(cnt);
        }
    }
    hA_csrOffsets.push_back(A_nnz);

    std::cout << "A_nnz = " << A_nnz << "\n";

    // set hY
    hY.resize(A_num_rows);
    hY_result.resize(A_num_rows);
    for (cusp_int i = 0; i < A_num_rows; i++) hY_result[i] = 1.0;

    //--------------------------------------------------------------------------

    /* The border between thread-level and warp-level algorithms, according to
     * the number of non-zero elements in each row of the matrix L*/
    int border = 10;

    /* !!!!!! start computing SpTRSV !!!!!!!! */
    double solve_time, gflops, bandwith, pre_time, warp_occupy, element_occupy;
    // warm up
    int success = YYSpTRSV_csr(
        A_num_rows, A_num_rows, A_nnz, hA_csrOffsets.data(), hA_columns.data(),
        hA_values.data(), hX.data(), hY.data(), border, &solve_time, &gflops,
        &bandwith, &pre_time, &warp_occupy, &element_occupy);
    // test
    success = YYSpTRSV_csr(A_num_rows, A_num_rows, A_nnz, hA_csrOffsets.data(),
                           hA_columns.data(), hA_values.data(), hX.data(),
                           hY.data(), border, &solve_time, &gflops, &bandwith,
                           &pre_time, &warp_occupy, &element_occupy);

    long readBytes = (sizeof(cusp_int) + sizeof(double)) * A_nnz +
                     sizeof(cusp_int) * A_num_rows +
                     sizeof(double) * A_num_rows;
    long writeBytes = sizeof(double) * A_num_rows;

    benchmark_record_map_lower[Dof - 1] = {solve_time * 1e-3 * 10,
                                           2L * A_nnz * 10,
                                           (readBytes + writeBytes) * 10, 10};
    std::cout << "YYsptrsv LowerTime(ms): " << solve_time
              << ", Gflops: " << gflops << ", Bandwidth=" << bandwith
              << std::endl;

    //--------------------------------------------------------------------------
    // device result check

    int correct = 1;
    for (cusp_int i = 0; i < A_num_rows; i++) {
        if (hY[i] !=
            hY_result[i]) {  // direct doubleing point comparison is not
            correct = 0;     // reliable
            // break;
            std::cout << "i = " << i << ", hY[i] = " << hY[i]
                      << ", hY_result[i] = " << hY_result[i] << std::endl;
        }
    }
    if (correct)
        printf("yysptrsv test PASSED\n");
    else
        printf("yysptrsv test FAILED: wrong result\n");
    //--------------------------------------------------------------------------
    // device memory deallocation
    // log::FunctionEnd(0, 0, 0);
}

int main(int argc, char **argv) {
    Json json = LoadJsonFromFile("matsolve-csrgen.json");
    std::string problems[] = {"stencilstar", "stencilbox", "stencilstarfill1"};
    bool if_output = json["output"];
    for (int i = 0; i < 3; i++) {
        for (int stencil_width_0 = 0; stencil_width_0 < (i == 0 ? 2 : 1);
             stencil_width_0++) {
            int stencil_width = stencil_width_0 + 1;
            std::string problem = problems[i];

            std::ofstream of;
            if (if_output) {
                of.open(std::string{"results/matsolve-yysptrsv-"} + problem +
                        "-stencilwidth" + std::to_string(stencil_width) +
                        ".out");
            } else {
                of.open("/dev/null");
            }
            for (int dof = 0; dof < MAX_DOF_TEST; dof++) {
                of << problem << ", width=" << stencil_width
                   << ", dof=" << dof + 1 << std::endl;
                RunBenchmarkLowerWithCusparse(
                    json[problem + std::to_string(stencil_width)], dof + 1, i,
                    stencil_width);
                of << "Lower:";
                double total_time = benchmark_record_map_lower[dof].total_time;
                double total_flops_time =
                    static_cast<double>(benchmark_record_map_lower[dof].flops) /
                    total_time;
                double total_bytes_time =
                    static_cast<double>(benchmark_record_map_lower[dof].bytes) /
                    total_time;

                of << dof + 1 << "," << total_time << ","
                   << total_flops_time * 1e-9 << "," << total_bytes_time * 1e-9
                   << std::endl;
            }
            of.close();
        }
    }
    return 0;
}
