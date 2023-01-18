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

#include "tridiagonal_matmul.h"

#include <complex>
#include "Eigen/Core"
#include "Eigen/Dense"
#include "Eigen/LU"
#include "unsupported/Eigen/CXX11/Tensor"

#include "cpu_kernel_utils.h"
#include "kernel_log.h"
#include "status.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
constexpr uint32_t kInputNum = 4;
constexpr uint32_t kOutputNum = 1;
const char *kTridiagonalMatMul = "TridiagonalMatMul";
}  // namespace
namespace aicpu {

uint32_t TridiagonalMatMulCpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum), "TridiagonalMatMul check input and output num failed.");
  KERNEL_HANDLE_ERROR(TridiagonalMatMulDataAndTypeCheck(ctx),
                      "TridiagonalMatMul check input and output params failed.");
  auto data_type = ctx.Input(0)->GetDataType();
  switch (data_type) {
    case DT_FLOAT16:
      return TridiagonalMatMulCompute<Eigen::half>(ctx);
    case DT_FLOAT:
      return TridiagonalMatMulCompute<float>(ctx);
    case DT_DOUBLE:
      return TridiagonalMatMulCompute<double>(ctx);
    case DT_COMPLEX64:
      return TridiagonalMatMulCompute<std::complex<float>>(ctx);
    case DT_COMPLEX128:
      return TridiagonalMatMulCompute<std::complex<double>>(ctx);
    default:
      KERNEL_LOG_ERROR("Unsupported input data type[%s]", DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

uint32_t TridiagonalMatMulCpuKernel::TridiagonalMatMulDataAndTypeCheck(CpuKernelContext &ctx) {
  DataType superdiag_type = ctx.Input(0)->GetDataType();
  DataType maindiag_type = ctx.Input(1)->GetDataType();
  DataType subdiag_type = ctx.Input(2)->GetDataType();
  DataType rhs_type = ctx.Input(3)->GetDataType();
  KERNEL_CHECK_FALSE((superdiag_type == maindiag_type && maindiag_type == subdiag_type && subdiag_type == rhs_type),
                     KERNEL_STATUS_PARAM_INVALID,
                     "The data type of input0 [%s], input1 [%s],input2 [%s] and input3 [%s] "
                     "need be same.",
                     DTypeStr(superdiag_type).c_str(), DTypeStr(maindiag_type).c_str(), DTypeStr(subdiag_type).c_str(),
                     DTypeStr(rhs_type).c_str())

  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t TridiagonalMatMulCpuKernel::TridiagonalMatMulCompute(CpuKernelContext &ctx) {
  auto superdiag_tensor = ctx.Input(0);
  auto superdiag_tensor_shape = superdiag_tensor->GetTensorShape();
  KERNEL_CHECK_FALSE((IsVector(superdiag_tensor_shape->GetDimSizes())), KERNEL_STATUS_PARAM_INVALID,
                     "invalid Input[superdiag]")
  auto maindiag_tensor = ctx.Input(1);
  auto maindiag_tensor_shape = maindiag_tensor->GetTensorShape();
  KERNEL_CHECK_FALSE((IsVector(maindiag_tensor_shape->GetDimSizes())), KERNEL_STATUS_PARAM_INVALID,
                     "invalid Input[maindiag]")
  auto subdiag_tensor = ctx.Input(2);
  auto subdiag_tensor_shape = subdiag_tensor->GetTensorShape();
  KERNEL_CHECK_FALSE((IsVector(subdiag_tensor_shape->GetDimSizes())), KERNEL_STATUS_PARAM_INVALID,
                     "invalid Input[subdiag]")
  auto rhs_tensor = ctx.Input(3);
  auto rhs_tensor_shape = rhs_tensor->GetTensorShape();
  KERNEL_CHECK_FALSE((IsMatrix(rhs_tensor_shape->GetDimSizes())), KERNEL_STATUS_PARAM_INVALID, "invalid Input[rhs]")
  auto superdiag_shape = superdiag_tensor_shape->GetDimSizes();
  auto maindiag_shape = maindiag_tensor_shape->GetDimSizes();
  auto subdiag_shape = subdiag_tensor_shape->GetDimSizes();
  auto rhs_shape = rhs_tensor_shape->GetDimSizes();
  int32_t superdiag_dims = superdiag_tensor_shape->GetDims();
  int32_t maindiag_dims = maindiag_tensor_shape->GetDims();
  int32_t subdiag_dims = subdiag_tensor_shape->GetDims();
  int32_t rhs_dims = rhs_tensor_shape->GetDims();
  int64_t length = rhs_shape[rhs_dims - 2];
  KERNEL_CHECK_FALSE((superdiag_shape[superdiag_dims - 1] == length), KERNEL_STATUS_PARAM_INVALID,
                     "invalid Input superdiag length")
  KERNEL_CHECK_FALSE((maindiag_shape[maindiag_dims - 1] == length), KERNEL_STATUS_PARAM_INVALID,
                     "invalid Input maindiag length")
  KERNEL_CHECK_FALSE((subdiag_shape[subdiag_dims - 1] == length), KERNEL_STATUS_PARAM_INVALID,
                     "invalid Input subdiag length")
  using VectorMap = Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>>;
  using MatrixMap = Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
  VectorMap superdiag(reinterpret_cast<T *>(superdiag_tensor->GetData()), superdiag_shape[superdiag_dims - 1], 1);
  VectorMap maindiag(reinterpret_cast<T *>(maindiag_tensor->GetData()), maindiag_shape[maindiag_dims - 1], 1);
  VectorMap subdiag(reinterpret_cast<T *>(subdiag_tensor->GetData()), subdiag_shape[subdiag_dims - 1], 1);
  MatrixMap rhs(reinterpret_cast<T *>(rhs_tensor->GetData()), rhs_shape[rhs_dims - 2], rhs_shape[rhs_dims - 1]);
  auto y_tensor = ctx.Output(0);
  auto y_shape = y_tensor->GetTensorShape()->GetDimSizes();
  int32_t y_dims = y_tensor->GetTensorShape()->GetDims();
  MatrixMap y(reinterpret_cast<T *>(y_tensor->GetData()), y_shape[y_dims - 2], y_shape[y_dims - 1]);
  y.array() = rhs.array().colwise() * maindiag.array();
  for (int64_t i = 0; i < length - 1; i++) {
    y.array().row(i) += rhs.array().row(i + 1) * superdiag(i);
    y.array().row(i + 1) += rhs.array().row(i) * subdiag(i + 1);
  }
  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kTridiagonalMatMul, TridiagonalMatMulCpuKernel);
}  // namespace aicpu
