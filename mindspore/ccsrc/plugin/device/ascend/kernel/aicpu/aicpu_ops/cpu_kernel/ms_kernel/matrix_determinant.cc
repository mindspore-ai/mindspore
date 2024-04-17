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
#include "cpu_kernel/ms_kernel/matrix_determinant.h"
#include <Eigen/Core>
#include <Eigen/LU>
#include <complex>
#include <iostream>
#include <vector>
#include "./utils/kernel_util.h"
#include "context/inc/cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"

namespace {
const uint32_t kOutputNum = 1;
const uint32_t kInputNum = 1;
const char *kMatrixDeterminant = "MatrixDeterminant";

#define MatrixDeterminant_COMPUTE_CASE(DTYPE, TYPE, CTX)                      \
  case (DTYPE): {                                                             \
    uint32_t result = MatrixDeterminantCompute<TYPE>(CTX);                    \
    if (result != KERNEL_STATUS_OK) {                                         \
      CUST_KERNEL_LOG_ERROR(ctx, "MatrixDeterminant kernel compute failed."); \
      return result;                                                          \
    }                                                                         \
    break;                                                                    \
  }
}  // namespace

namespace aicpu {
uint32_t MatrixDeterminantCpuKernel::Compute(CpuKernelContext &ctx) {
  // check params
  CUST_KERNEL_HANDLE_ERROR(ctx, NormalCheck(ctx, kInputNum, kOutputNum),
                           "MatrixDeterminant check input and output number failed.");
  CUST_KERNEL_HANDLE_ERROR(ctx, MatrixDeterminantCheck(ctx), "MatrixDeterminant check params failed.");
  Tensor *input = ctx.Input(0);
  // Check whether the number of matrices is > 0
  auto shape = input->GetTensorShape();
  std::vector<int64_t> dims = shape->GetDimSizes();
  int k = dims.size();
  for (int i = 0; i < k - 2; i++) {
    if (dims[i] <= 0) {
      CUST_KERNEL_LOG_ERROR(ctx, "The input must be one or more squares.");
      return KERNEL_STATUS_PARAM_INVALID;
    }
  }
  // Check if it's a square array
  if (dims[dims.size() - 1] == 0 || dims[dims.size() - 2] != dims[dims.size() - 1]) {
    CUST_KERNEL_LOG_ERROR(ctx, "The input must be one or more squares.");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  // Check element type
  auto input_dtype = input->GetDataType();
  switch (input_dtype) {
    MatrixDeterminant_COMPUTE_CASE(DT_FLOAT, float, ctx) MatrixDeterminant_COMPUTE_CASE(DT_DOUBLE, double, ctx)
      MatrixDeterminant_COMPUTE_CASE(DT_COMPLEX64, std::complex<float>, ctx)
        MatrixDeterminant_COMPUTE_CASE(DT_COMPLEX128, std::complex<double>, ctx) default
        : CUST_KERNEL_LOG_ERROR(ctx,
                                "MatrixDeterminant kernel dims data_type [%s] "
                                "not support,support data_types: DT_INT32, DT_INT64, "
                                "DT_COMPLEX64, DT_COMPLEX128.",
                                DTypeStr(input_dtype).c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

uint32_t MatrixDeterminantCpuKernel::MatrixDeterminantCheck(CpuKernelContext &ctx) {
  CUST_KERNEL_CHECK_NULLPTR(ctx, ctx.Input(0)->GetData(), KERNEL_STATUS_PARAM_INVALID, "Get input data failed.")
  CUST_KERNEL_CHECK_NULLPTR(ctx, ctx.Output(0)->GetData(), KERNEL_STATUS_PARAM_INVALID, "Get output data failed")
  CUST_KERNEL_LOG_INFO(ctx,
                       "MatrixDeterminantCpuKernel[%s], input: size[%llu];"
                       "output: size[%llu].",
                       ctx.GetOpType().c_str(), ctx.Input(0)->GetDataSize(), ctx.Output(0)->GetDataSize());
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t MatrixDeterminantCpuKernel::MatrixDeterminantCompute(CpuKernelContext &ctx) {
  Tensor *input_tensor = ctx.Input(0);
  Tensor *output_tensor = ctx.Output(0);
  std::vector<int64_t> dims = input_tensor->GetTensorShape()->GetDimSizes();
  T *input = reinterpret_cast<T *>(input_tensor->GetData());
  T *output = reinterpret_cast<T *>(output_tensor->GetData());
  int m = dims[dims.size() - 1];
  int n = 1;
  for (uint i = 0; i < dims.size() - 2; i++) {
    n *= dims[i];
  }
  auto shard_matrix_determinant = [&](size_t start, size_t end) {
    for (size_t k = start; k < end; k++) {
      Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> eMatrix(m, m);
      for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
          eMatrix(i, j) = *(input + k * m * m + i * m + j);
        }
      }
      // use eigen to calculate determinant
      T result = eMatrix.determinant();
      *(output + k) = result;
    }
  };
  CUST_KERNEL_HANDLE_ERROR(ctx, CpuKernelUtils::ParallelFor(ctx, n, 1, shard_matrix_determinant),
                           "MatrixDeterminant Compute failed.");
  return KERNEL_STATUS_OK;
}

REGISTER_MS_CPU_KERNEL(kMatrixDeterminant, MatrixDeterminantCpuKernel);

}  // namespace aicpu
