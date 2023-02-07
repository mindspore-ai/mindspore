/**
 * Copyright (c) 2022-2022 Huawei Technologies Co., Ltd.  All rights reserved.
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

#include "matrix_inverse.h"
#include <complex>
#include <vector>
#include "Eigen/Core"
#include "Eigen/LU"
#include "cpu_kernel_utils.h"
#include "utils/kernel_util.h"

namespace {
const uint32_t kOutputNum = 1;
const uint32_t kInputNum = 1;
const char *kMatrixInverse = "MatrixInverse";
// if the data size is larger than the value, call ParallelFor() func
constexpr int64_t kParallelDataNums = 1 * 1024;

#define MATRIXINVERSE_COMPUTE_CASE(DTYPE, TYPE, CTX)            \
  case (DTYPE): {                                               \
    uint32_t result = MatrixInverseCompute<TYPE>(CTX);          \
    if (result != KERNEL_STATUS_OK) {                           \
      KERNEL_LOG_ERROR("MatrixInverse kernel compute failed."); \
      return result;                                            \
    }                                                           \
    break;                                                      \
  }
}  // namespace

namespace aicpu {
uint32_t MatrixInverseCpuKernel::Compute(CpuKernelContext &ctx) {
  // check params
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum), "MatrixInverse check input and output number failed.");
  auto data_type = ctx.Input(0)->GetDataType();
  switch (data_type) {
    MATRIXINVERSE_COMPUTE_CASE(DT_FLOAT, float, ctx)
    MATRIXINVERSE_COMPUTE_CASE(DT_DOUBLE, double, ctx)
    MATRIXINVERSE_COMPUTE_CASE(DT_COMPLEX64, std::complex<float>, ctx)
    MATRIXINVERSE_COMPUTE_CASE(DT_COMPLEX128, std::complex<double>, ctx)
    default:
      KERNEL_LOG_ERROR("MatrixInverse kernel data type [%s] not support.", DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t MatrixInverseCpuKernel::MatrixInverseCompute(CpuKernelContext &ctx) {
  Tensor *input = ctx.Input(0);
  T *input_ptr = reinterpret_cast<T *>(input->GetData());
  Tensor *output = ctx.Output(0);
  T *output_ptr = reinterpret_cast<T *>(output->GetData());
  // Judge whether the input shape matches
  auto shape = input->GetTensorShape();
  uint64_t data_size = input->GetDataSize();
  std::vector<int64_t> dims = shape->GetDimSizes();
  KERNEL_CHECK_FALSE((dims.size() >= 2 && (*(dims.end() - 1) == *(dims.end() - 2))), KERNEL_STATUS_PARAM_INVALID,
                     "Input Shape is wrong");
  auto last_dimsize = *(dims.end() - 1);
  // Output length
  auto input_num = input->NumElements();
  size_t matrix_size = last_dimsize * last_dimsize;
  // Number of matrices
  size_t matrix_num = input_num / matrix_size;
  // Store two-dimensional array of data for slicing
  std::vector<std::vector<T>> temp(matrix_num, std::vector<T>(matrix_size));
  for (size_t i = 0; i < matrix_num; i++) {
    for (size_t j = 0; j < matrix_size; j++) {
      temp[i][j] = *(input_ptr + i * matrix_size + j);
    }
  }
  // Gets the value of the property adjoint
  AttrValue *adjoint_attr = ctx.GetAttr("adjoint");
  bool adjoint__ = adjoint_attr->GetBool();
  if (data_size <= kParallelDataNums) {
    for (size_t i = 0; i < matrix_num; i++) {
      Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> eigen_input(temp[i].data(), last_dimsize,
                                                                               last_dimsize);
      Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> eigen_output(output_ptr + i * matrix_size,
                                                                                last_dimsize, last_dimsize);
      if (adjoint__) {
        eigen_input = eigen_input.adjoint().eval();
      }
      Eigen::FullPivLU<Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>> lu(eigen_input);
      eigen_output = lu.inverse();
    }
  } else {
    uint32_t min_core_num = 1;
    size_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - 2);
    if (max_core_num > matrix_num) {
      max_core_num = matrix_num;
    }
    auto sharedcompute = [&](size_t start, size_t end) {
      for (auto i = start; i < end; i++) {
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> eigen_input(temp[i].data(), last_dimsize,
                                                                                 last_dimsize);
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> eigen_output(output_ptr + i * matrix_size,
                                                                                  last_dimsize, last_dimsize);
        if (adjoint__) {
          eigen_input = eigen_input.adjoint().eval();
        }
        Eigen::FullPivLU<Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>> lu(eigen_input);
        eigen_output = lu.inverse();
      }
    };
    KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, matrix_num, matrix_num / max_core_num, sharedcompute),
                        "Compute failed.");
  }
  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kMatrixInverse, MatrixInverseCpuKernel);
}  // namespace aicpu