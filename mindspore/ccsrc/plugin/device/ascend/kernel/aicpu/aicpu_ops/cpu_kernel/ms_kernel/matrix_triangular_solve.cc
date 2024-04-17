/**
 * Copyright 2021 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "cpu_kernel/ms_kernel/matrix_triangular_solve.h"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <vector>

#include "Eigen/Core"
#include "complex"
#include "context/inc/cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"
#include "inc/kernel_log.h"

namespace {
const uint32_t kOutputNum = 1;
const uint32_t kInputNum = 2;
const char *kMatrixTriangularSolve = "MatrixTriangularSolve";
constexpr int64_t kParallelDataNums = 16 * 1024;

#define MATRIXTRIANGULARSOLVE_COMPUTE_CASE(DTYPE, TYPE, CTX)                      \
  case (DTYPE): {                                                                 \
    uint32_t result = MatrixTriangularSolveCompute<TYPE>(CTX);                    \
    if (result != KERNEL_STATUS_OK) {                                             \
      CUST_KERNEL_LOG_ERROR(ctx, "MatrixTriangularSolve kernel compute failed."); \
      return result;                                                              \
    }                                                                             \
    break;                                                                        \
  }
}  // namespace

namespace aicpu {
uint32_t MatrixTriangularSolveCpuKernel::Compute(CpuKernelContext &ctx) {
  CUST_KERNEL_HANDLE_ERROR(ctx, NormalCheck(ctx, kInputNum, kOutputNum),
                           "MatrixTriangularSolve check input and output number failed.");

  CUST_KERNEL_HANDLE_ERROR(ctx, MatrixTriangularSolveCheck(ctx), "MatrixTriangularSolve check params failed.");
  // check the data type of the inputs
  auto data_type = ctx.Input(0)->GetDataType();
  switch (data_type) {
    MATRIXTRIANGULARSOLVE_COMPUTE_CASE(DT_FLOAT, float, ctx)
    MATRIXTRIANGULARSOLVE_COMPUTE_CASE(DT_DOUBLE, double, ctx)
    MATRIXTRIANGULARSOLVE_COMPUTE_CASE(DT_COMPLEX64, std::complex<float>, ctx)
    MATRIXTRIANGULARSOLVE_COMPUTE_CASE(DT_COMPLEX128, std::complex<double>, ctx)
    default:
      CUST_KERNEL_LOG_ERROR(ctx, "MatrixTriangularSolve kernel data type [%s] not support.",
                            DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }

  return KERNEL_STATUS_OK;
}

uint32_t MatrixTriangularSolveCpuKernel::MatrixTriangularSolveCheck(CpuKernelContext &ctx) {
  Tensor *in_matrix = ctx.Input(0);
  Tensor *in_rhs = ctx.Input(1);
  // check same data type constraint
  auto in_type0 = in_matrix->GetDataType();
  auto in_type1 = in_rhs->GetDataType();
  CUST_KERNEL_CHECK_FALSE(ctx, (in_type0 == in_type1), KERNEL_STATUS_PARAM_INVALID,
                          "The data type of input1 [%s] need be same with "
                          "input0 [%s].",
                          DTypeStr(in_type1).c_str(), DTypeStr(in_type0).c_str())
  // check the number of matrix
  auto in_shape0 = in_matrix->GetTensorShape();
  auto in_shape1 = in_rhs->GetTensorShape();

  std::vector<int64_t> dims0 = in_shape0->GetDimSizes();
  std::vector<int64_t> dims1 = in_shape1->GetDimSizes();

  // Check the shape of two inputs
  if (dims0[0] != dims1[0]) {
    CUST_KERNEL_LOG_ERROR(ctx, "The shapes of two inputs are not matched");
    return KERNEL_STATUS_PARAM_INVALID;
  }

  // check square
  int m = dims0.size();
  if (dims0[m - 2] != dims0[m - 1] || dims0[m - 1] == 0) {
    CUST_KERNEL_LOG_ERROR(ctx, "The input0 must be one or more squares.");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t MatrixTriangularSolveCpuKernel::MatrixTriangularSolveCompute(CpuKernelContext &ctx) {
  Tensor *matrix_tensor = ctx.Input(0);
  Tensor *rhs_tensor = ctx.Input(1);
  Tensor *y_tensor = ctx.Output(0);

  auto input_matrix = reinterpret_cast<T *>(matrix_tensor->GetData());
  CUST_KERNEL_CHECK_NULLPTR(ctx, input_matrix, KERNEL_STATUS_PARAM_INVALID, "Get input data0 failed.")
  auto input_rhs = reinterpret_cast<T *>(rhs_tensor->GetData());
  CUST_KERNEL_CHECK_NULLPTR(ctx, input_rhs, KERNEL_STATUS_PARAM_INVALID, "Get input data1 failed.")
  auto output_y = reinterpret_cast<T *>(y_tensor->GetData());
  CUST_KERNEL_CHECK_NULLPTR(ctx, output_y, KERNEL_STATUS_PARAM_INVALID, "Get output data failed.")

  AttrValue *lower_attr = ctx.GetAttr("lower");
  CUST_KERNEL_CHECK_NULLPTR(ctx, lower_attr, KERNEL_STATUS_PARAM_INVALID, "Get attr [lower] failed.");
  AttrValue *adjoint_attr = ctx.GetAttr("adjoint");
  CUST_KERNEL_CHECK_NULLPTR(ctx, adjoint_attr, KERNEL_STATUS_PARAM_INVALID, "Get attr [adjoint] failed.");
  bool lower_data = lower_attr->GetBool();
  bool adjoint_data = adjoint_attr->GetBool();

  auto matrix_shape = matrix_tensor->GetTensorShape();
  auto rhs_shape = rhs_tensor->GetTensorShape();
  auto y_shape = y_tensor->GetTensorShape();

  // Get the number of elements
  auto input1_num = matrix_tensor->NumElements();

  // slice
  std::vector<int64_t> matrix_dims = matrix_shape->GetDimSizes();
  auto last_matrix_dims = *(matrix_dims.end() - 1);
  size_t matrix_size = last_matrix_dims * last_matrix_dims;  // size of a matrix
  size_t matrix_num = input1_num / matrix_size;              // number of matrix

  std::vector<int64_t> rhs_dims = rhs_shape->GetDimSizes();
  auto last_rhs_dims = *(rhs_dims.end() - 1);
  size_t rhs_size = last_matrix_dims * last_rhs_dims;

  auto data_size = matrix_num * matrix_size;

  auto shard_matrix_triangular_solve = [&](size_t start, size_t end) {
    for (size_t k = start; k < end; ++k) {
      Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eigen_input(
        input_matrix + k * matrix_size, last_matrix_dims, last_matrix_dims);
      Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eigen_rhs(
        input_rhs + k * rhs_size, last_matrix_dims, last_rhs_dims);
      Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eigen_output(
        output_y + k * rhs_size, last_matrix_dims, last_rhs_dims);
      if (lower_data) {
        auto triangle = eigen_input.template triangularView<Eigen::Lower>();
        if (adjoint_data) {
          eigen_output.noalias() = triangle.adjoint().solve(eigen_rhs);
        } else {
          eigen_output.noalias() = triangle.solve(eigen_rhs);
        }
      } else {
        auto triangle = eigen_input.template triangularView<Eigen::Upper>();
        if (adjoint_data) {
          eigen_output.noalias() = triangle.adjoint().solve(eigen_rhs);
        } else {
          eigen_output.noalias() = triangle.solve(eigen_rhs);
        }
      }
    }
  };
  if (data_size < kParallelDataNums) {
    shard_matrix_triangular_solve(0, matrix_num);
  } else {
    uint32_t min_core_num = 1;
    uint64_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - 2);
    if (max_core_num > matrix_num) {
      max_core_num = matrix_num;
    }
    CUST_KERNEL_HANDLE_ERROR(
      ctx, CpuKernelUtils::ParallelFor(ctx, matrix_num, matrix_num / max_core_num, shard_matrix_triangular_solve),
      "MatrixTriangularSolve Compute failed.");
  }
  return KERNEL_STATUS_OK;
}
REGISTER_MS_CPU_KERNEL(kMatrixTriangularSolve, MatrixTriangularSolveCpuKernel);
}  // namespace aicpu
