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
#include <algorithm>
#include <iostream>
#include <type_traits>
#include <vector>

#include "cpu_kernel_utils.h"
#include "sparse_tensor_dense_mat_mul.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

using namespace std;
namespace {
#define COL_SHED 1024 << 1
const uint32_t kOutputNum = 1;
const uint32_t kInputNum = 4;
const char *kSparseTensorDenseMatMul = "SparseTensorDenseMatMul";
}  // namespace
namespace aicpu {
uint32_t SparseTensorDenseMatMulCpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum),
                      "SparseTensorDenseMatMul check input and output number failed.");
  KERNEL_HANDLE_ERROR(SparseTensorDenseMatMulCheck(ctx), "SparseTensorDenseMatMul check params failed.");
  DataType sparse_data_type = ctx.Input(1)->GetDataType();
  DataType indice_data_type = ctx.Input(0)->GetDataType();
  DataType dense_data_type = ctx.Input(3)->GetDataType();
  DataType y_data_type = ctx.Output(0)->GetDataType();
  if (sparse_data_type == DT_FLOAT && indice_data_type == DT_INT64 && dense_data_type == DT_FLOAT &&
      y_data_type == DT_FLOAT)
    regular_calculate<float, int64_t, float, float>(ctx);
  else if (sparse_data_type == DT_FLOAT && indice_data_type == DT_INT64 && dense_data_type == DT_COMPLEX64 &&
           y_data_type == DT_COMPLEX64)
    regular_calculate<float, int64_t, complex<float>, complex<float>>(ctx);
  else if (sparse_data_type == DT_FLOAT && indice_data_type == DT_INT32 && dense_data_type == DT_FLOAT &&
           y_data_type == DT_FLOAT)
    regular_calculate<float, int32_t, float, float>(ctx);
  else if (sparse_data_type == DT_FLOAT && indice_data_type == DT_INT32 && dense_data_type == DT_COMPLEX64 &&
           y_data_type == DT_COMPLEX64)
    regular_calculate<float, int32_t, complex<float>, complex<float>>(ctx);
  else if (sparse_data_type == DT_DOUBLE && indice_data_type == DT_INT64 && dense_data_type == DT_DOUBLE &&
           y_data_type == DT_DOUBLE)
    regular_calculate<double, int64_t, double, double>(ctx);
  else if (sparse_data_type == DT_DOUBLE && indice_data_type == DT_INT64 && dense_data_type == DT_COMPLEX128 &&
           y_data_type == DT_COMPLEX128)
    regular_calculate<double, int64_t, complex<double>, complex<double>>(ctx);
  else if (sparse_data_type == DT_DOUBLE && indice_data_type == DT_INT32 && dense_data_type == DT_DOUBLE &&
           y_data_type == DT_DOUBLE)
    regular_calculate<double, int32_t, double, double>(ctx);
  else if (sparse_data_type == DT_DOUBLE && indice_data_type == DT_INT32 && dense_data_type == DT_COMPLEX128 &&
           y_data_type == DT_COMPLEX128)
    regular_calculate<double, int32_t, complex<double>, complex<double>>(ctx);
  else if (sparse_data_type == DT_INT64 && indice_data_type == DT_INT64 && dense_data_type == DT_INT64 &&
           y_data_type == DT_INT64)
    regular_calculate<int64_t, int64_t, int64_t, int64_t>(ctx);
  else if (sparse_data_type == DT_INT64 && indice_data_type == DT_INT32 && dense_data_type == DT_INT64 &&
           y_data_type == DT_INT64)
    regular_calculate<int64_t, int32_t, int64_t, int64_t>(ctx);
  else if (sparse_data_type == DT_INT32 && indice_data_type == DT_INT64 && dense_data_type == DT_INT32 &&
           y_data_type == DT_INT32)
    regular_calculate<int32_t, int64_t, int32_t, int32_t>(ctx);
  else if (sparse_data_type == DT_INT32 && indice_data_type == DT_INT32 && dense_data_type == DT_INT32 &&
           y_data_type == DT_INT32)
    regular_calculate<int32_t, int32_t, int32_t, int32_t>(ctx);
  else if (sparse_data_type == DT_COMPLEX64 && indice_data_type == DT_INT64 && dense_data_type == DT_FLOAT &&
           y_data_type == DT_COMPLEX64)
    regular_calculate<complex<float>, int64_t, float, complex<float>>(ctx);
  else if (sparse_data_type == DT_COMPLEX64 && indice_data_type == DT_INT64 && dense_data_type == DT_COMPLEX64 &&
           y_data_type == DT_COMPLEX64)
    regular_calculate<complex<float>, int64_t, complex<float>, complex<float>>(ctx);
  else if (sparse_data_type == DT_COMPLEX64 && indice_data_type == DT_INT32 && dense_data_type == DT_FLOAT &&
           y_data_type == DT_COMPLEX64)
    regular_calculate<complex<float>, int32_t, float, complex<float>>(ctx);
  else if (sparse_data_type == DT_COMPLEX64 && indice_data_type == DT_INT32 && dense_data_type == DT_COMPLEX64 &&
           y_data_type == DT_COMPLEX64)
    regular_calculate<complex<float>, int32_t, complex<float>, complex<float>>(ctx);
  else if (sparse_data_type == DT_COMPLEX128 && indice_data_type == DT_INT64 && dense_data_type == DT_DOUBLE &&
           y_data_type == DT_COMPLEX128)
    regular_calculate<complex<double>, int64_t, double, complex<double>>(ctx);
  else if (sparse_data_type == DT_COMPLEX128 && indice_data_type == DT_INT64 && dense_data_type == DT_COMPLEX128 &&
           y_data_type == DT_COMPLEX128)
    regular_calculate<complex<double>, int64_t, complex<double>, complex<double>>(ctx);
  else if (sparse_data_type == DT_COMPLEX128 && indice_data_type == DT_INT32 && dense_data_type == DT_DOUBLE &&
           y_data_type == DT_COMPLEX128)
    regular_calculate<complex<double>, int32_t, double, complex<double>>(ctx);
  else if (sparse_data_type == DT_COMPLEX128 && indice_data_type == DT_INT32 && dense_data_type == DT_COMPLEX128 &&
           y_data_type == DT_COMPLEX128)
    regular_calculate<complex<double>, int32_t, complex<double>, complex<double>>(ctx);
  else if (sparse_data_type == DT_FLOAT16 && indice_data_type == DT_INT64 && dense_data_type == DT_FLOAT16 &&
           y_data_type == DT_FLOAT16)
    regular_calculate<Eigen::half, int64_t, Eigen::half, Eigen::half>(ctx);
  else if (sparse_data_type == DT_FLOAT16 && indice_data_type == DT_INT32 && dense_data_type == DT_FLOAT16 &&
           y_data_type == DT_FLOAT16)
    regular_calculate<Eigen::half, int32_t, Eigen::half, Eigen::half>(ctx);

  else {
    KERNEL_LOG_ERROR(
      "sparse_tensor_dense_mat_mul kernel wrong datatype."
      "sparse_data_type [%s],"
      "indices_data_type [%s],"
      "dense_data_type [%s],"
      "y_data_type [%s].",
      DTypeStr(sparse_data_type).c_str(), DTypeStr(indice_data_type).c_str(), DTypeStr(dense_data_type).c_str(),
      DTypeStr(y_data_type).c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}
template <class SparseType, class IndicesType, class DenseType, class OutputType>
uint32_t SparseTensorDenseMatMulCpuKernel::regular_calculate(CpuKernelContext &ctx) {
  Tensor *x1_indices = ctx.Input(0);
  Tensor *x1_values = ctx.Input(1);
  Tensor *x1_shape = ctx.Input(2);
  Tensor *x2 = ctx.Input(3);
  Tensor *y = ctx.Output(0);
  auto x1_indices_shape = x1_indices->GetTensorShape();
  auto x2_shape = x2->GetTensorShape();
  auto y_shape = y->GetTensorShape();
  int64_t *x1_shape_data = (int64_t *)x1_shape->GetData();
  uint64_t x1_row = x1_shape_data[0];
  uint64_t x1_col = x1_shape_data[1];
  uint64_t x2_row = x2_shape->GetDimSize(0);
  uint64_t x2_col = x2_shape->GetDimSize(1);
  AttrValue *adjoint_a = ctx.GetAttr("adjoint_a");
  AttrValue *adjoint_b = ctx.GetAttr("adjoint_b");
  SparseType *x1_values_data = (SparseType *)x1_values->GetData();
  DenseType *x2_data = (DenseType *)x2->GetData();
  OutputType *y_data = (OutputType *)y->GetData();
  uint64_t y_data_len = y->NumElements();
  for (uint64_t i = 0; i < y_data_len; i++) {
    y_data[i] = static_cast<OutputType>(0);
  }

  if (adjoint_a->GetBool()) {
    swap(x1_row, x1_col);
  }
  if (adjoint_b->GetBool()) {
    swap(x2_row, x2_col);
  }
  uint64_t pairs = x1_indices_shape->GetDimSize(0);
  IndicesType *x1_indices_data = (IndicesType *)x1_indices->GetData();
  for (uint64_t i = 0; i < pairs; i++) {
    uint64_t row = x1_indices_data[i << 1], col = x1_indices_data[1 + (i << 1)];
    SparseType a = x1_values_data[i];
    if (adjoint_a->GetBool()) {
      swap(row, col);
    }
    KERNEL_CHECK_FALSE(row >= 0 && row < x1_row && col >= 0 && col < x1_col, KERNEL_STATUS_PARAM_INVALID,
                       "sparse size invalid.")
    if (x2_col < COL_SHED) {
      for (uint64_t j = 0; j < x2_col; j++) {
        uint64_t idx = adjoint_b->GetBool() ? (j * x2_row + col) : (col * x2_col + j);
        DenseType b = x2_data[idx];
        if constexpr (std::is_same<DenseType, complex<double>>::value || std::is_same<DenseType, complex<float>>::value)
          if (adjoint_b->GetBool()) {
            b = conj(b);
          }
        y_data[row * x2_col + j] += a * b;
      }
      continue;
    }
    uint32_t min_core = 1;
    uint64_t max_core = std::max(min_core, aicpu::CpuKernelUtils::GetCPUNum(ctx) - 2);
    max_core = std::min(max_core, x2_col);
    auto fun = [&](size_t s, size_t t) {
      for (uint64_t j = s; j < t; j++) {
        uint64_t idx = adjoint_b->GetBool() ? (j * x2_row + col) : (col * x2_col + j);
        DenseType b = x2_data[idx];
        if constexpr (std::is_same<DenseType, complex<double>>::value || std::is_same<DenseType, complex<float>>::value)
          if (adjoint_b->GetBool()) {
            b = conj(b);
          }
        y_data[row * x2_col + j] += a * b;
      }
    };
    KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, x2_col, x2_col / max_core, fun),
                        "SparseTensorDenseMatMul Compute failed.");
  }
  return KERNEL_STATUS_OK;
}
uint32_t SparseTensorDenseMatMulCpuKernel::SparseTensorDenseMatMulCheck(CpuKernelContext &ctx) {
  Tensor *x1_indices = ctx.Input(0);
  Tensor *x1_values = ctx.Input(1);
  Tensor *x1_shape = ctx.Input(2);
  Tensor *x2 = ctx.Input(3);
  Tensor *y = ctx.Output(0);
  AttrValue *adjoint_a = ctx.GetAttr("adjoint_a"), *adjoint_b = ctx.GetAttr("adjoint_b");
  KERNEL_CHECK_NULLPTR(x1_indices, KERNEL_STATUS_PARAM_INVALID, "Get input 0 failed.")
  KERNEL_CHECK_NULLPTR(x1_values, KERNEL_STATUS_PARAM_INVALID, "Get input 1 failed.")
  KERNEL_CHECK_NULLPTR(x1_shape, KERNEL_STATUS_PARAM_INVALID, "Get input 2 failed.")
  KERNEL_CHECK_NULLPTR(x2, KERNEL_STATUS_PARAM_INVALID, "Get input 3 failed.")
  KERNEL_CHECK_NULLPTR(y, KERNEL_STATUS_PARAM_INVALID, "Get output 0 failed.")
  KERNEL_CHECK_NULLPTR(adjoint_a, KERNEL_STATUS_PARAM_INVALID, "Get attribute adjoint_a failed.")
  KERNEL_CHECK_NULLPTR(adjoint_b, KERNEL_STATUS_PARAM_INVALID, "Get attribute adjoint_b failed.")
  KERNEL_CHECK_FALSE(x1_indices->GetData(), KERNEL_STATUS_PARAM_INVALID, "Get input 0 data failed.")
  KERNEL_CHECK_FALSE(x1_values->GetData(), KERNEL_STATUS_PARAM_INVALID, "Get input 1 data failed.")
  KERNEL_CHECK_FALSE(x1_indices->GetData(), KERNEL_STATUS_PARAM_INVALID, "Get input 2 data failed.")
  KERNEL_CHECK_FALSE(x2->GetData(), KERNEL_STATUS_PARAM_INVALID, "Get input 3 data failed.")
  KERNEL_CHECK_FALSE(y->GetData(), KERNEL_STATUS_PARAM_INVALID, "Get output 0 data failed.")
  KERNEL_CHECK_FALSE(x1_shape->GetDataType() == DT_INT64, KERNEL_STATUS_PARAM_INVALID, "x1_shape must be DT_INT64")
  KERNEL_CHECK_FALSE(x1_shape->GetTensorShape()->GetDims() == 1 && x1_shape->NumElements() == 2 &&
                       x1_indices->GetTensorShape()->GetDimSize(0) == x1_values->NumElements(),
                     KERNEL_STATUS_PARAM_INVALID, "sparse tensor x1 dimension error.")
  KERNEL_CHECK_FALSE(x2->GetTensorShape()->GetDims() == 2, KERNEL_STATUS_PARAM_INVALID, "matrix x2 dimension error.")
  int64_t *x1_shape_data = (int64_t *)x1_shape->GetData();
  uint64_t x1_col = x1_shape_data[!adjoint_a->GetBool()];
  uint64_t x2_row = x2->GetTensorShape()->GetDimSize(adjoint_b->GetBool());
  KERNEL_CHECK_FALSE(x1_col == x2_row, KERNEL_STATUS_PARAM_INVALID, "can not do matrix multiplication.")
  return KERNEL_STATUS_OK;
}
REGISTER_CPU_KERNEL(kSparseTensorDenseMatMul, SparseTensorDenseMatMulCpuKernel);
}  // namespace aicpu
