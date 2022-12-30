/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
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
#include "matmul.h"

#include <complex>
#include "unsupported/Eigen/CXX11/Tensor"

#include "utils/kernel_util.h"
#include "cpu_kernel_utils.h"
#include "kernel_log.h"
#include "status.h"

using namespace std;

namespace {
const char *kMatmul = "MatMul";
}  // namespace

namespace aicpu {
template <typename T>
uint32_t MatMulCpuKernel::AddCompute(CpuKernelContext &ctx, Bcast &bcast) {
  auto in2 = reinterpret_cast<T *>(ctx.Input(2)->GetData());
  auto out = reinterpret_cast<T *>(ctx.Output(0)->GetData());
  int64_t data_num = ctx.Output(0)->NumElements();

  for (int64_t i = 0; i < data_num; i++) {
    auto input1 = in2 + bcast.GetBroadcastXIndex(i);  // i-th value of input0
    auto input2 = out + bcast.GetBroadcastYIndex(i);  // i-th value of input1
    *(out + i) = (*input1) + (*input2);
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t MatMulCpuKernel::BiasCompute(CpuKernelContext &ctx) {
  auto input0_tensor = ctx.Input(0);
  auto input2_tensor = ctx.Input(2);
  auto input2_shape = input2_tensor->GetTensorShape()->GetDimSizes();
  auto output_tensor = ctx.Output(kFirstOutputIndex);
  auto output_shape = output_tensor->GetTensorShape()->GetDimSizes();

  KERNEL_CHECK_FALSE(input2_tensor->GetTensorShape()->GetDims() == 1, KERNEL_STATUS_PARAM_INVALID,
                     "Input[x3] must be a 1D tensor")

  DataType input0_data_type = input0_tensor->GetDataType();
  DataType input2_data_type = input2_tensor->GetDataType();
  KERNEL_CHECK_FALSE((input0_data_type == input2_data_type), KERNEL_STATUS_PARAM_INVALID,
                     "Input[x1] data type[%s] and input[x3] data type[%s] must be same",
                     DTypeStr(input0_data_type).c_str(), DTypeStr(input2_data_type).c_str())

  Bcast bcast(input2_shape, output_shape);
  if (!bcast.IsValid()) {
    KERNEL_LOG_ERROR("[%s] broadcast failed.", ctx.GetOpType().c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  return AddCompute<T>(ctx, bcast);
}

template <typename T>
uint32_t MatMulCpuKernel::MatMulCompute(CpuKernelContext &ctx) {
  auto input0_tensor = ctx.Input(0);
  auto input0_tensor_shape = input0_tensor->GetTensorShape();
  KERNEL_CHECK_FALSE((IsMatrix(input0_tensor_shape->GetDimSizes())), KERNEL_STATUS_PARAM_INVALID,
                     "Input[x1] must be a matrix")

  auto input1_tensor = ctx.Input(1);
  auto input1_tensor_shape = input1_tensor->GetTensorShape();
  KERNEL_CHECK_FALSE((IsMatrix(input1_tensor_shape->GetDimSizes())), KERNEL_STATUS_PARAM_INVALID,
                     "Input[x2] must be a matrix")

  auto transpose_x1 = ctx.GetAttr("transpose_x1")->GetBool();
  auto transpose_x2 = ctx.GetAttr("transpose_x2")->GetBool();
  KERNEL_LOG_DEBUG(
    "%s Attr[transpose_x1] value[%d], "
    "Attr[transpose_x2] value[%d].",
    kMatmul, transpose_x1, transpose_x2);
  int32_t x1_dim = transpose_x1 ? 0 : 1;
  int32_t x2_dim = transpose_x2 ? 1 : 0;
  KERNEL_CHECK_FALSE((input0_tensor_shape->GetDimSize(x1_dim) == input1_tensor_shape->GetDimSize(x2_dim)),
                     KERNEL_STATUS_PARAM_INVALID,
                     "Matrix size incompatible, input[x1] dim[%d] value[%lld], "
                     "input[x2] dim[%d] value[%lld]",
                     x1_dim, input0_tensor_shape->GetDimSize(x1_dim), x2_dim, input1_tensor_shape->GetDimSize(x2_dim))

  auto input0_shape = input0_tensor_shape->GetDimSizes();
  using MatrixMap = Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
  MatrixMap input0(reinterpret_cast<T *>(input0_tensor->GetData()), input0_shape[0], input0_shape[1]);

  auto input1_shape = input1_tensor_shape->GetDimSizes();
  MatrixMap input1(reinterpret_cast<T *>(input1_tensor->GetData()), input1_shape[0], input1_shape[1]);

  auto output_tensor = ctx.Output(kFirstOutputIndex);
  auto output_shape = output_tensor->GetTensorShape()->GetDimSizes();
  MatrixMap output(reinterpret_cast<T *>(output_tensor->GetData()), output_shape[0], output_shape[1]);
  if (transpose_x1) {
    if (transpose_x2) {
      output = input0.transpose() * input1.transpose();
    } else {
      output = input0.transpose() * input1;
    }
  } else {
    if (transpose_x2) {
      output = input0 * input1.transpose();
    } else {
      output = input0 * input1;
    }
  }
  if (ctx.GetInputsSize() == 3) {
    return BiasCompute<T>(ctx);
  }
  return KERNEL_STATUS_OK;
}

uint32_t MatMulCpuKernel::Compute(CpuKernelContext &ctx) {
  // check params
  uint32_t input_num = ctx.GetInputsSize();
  uint32_t output_num = ctx.GetOutputsSize();
  if ((input_num != 2 && input_num != 3) || output_num != 1) {
    KERNEL_LOG_ERROR("The number of input or output parameters does not match.");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  auto input0_tensor = ctx.Input(0);
  KERNEL_CHECK_NULLPTR(input0_tensor->GetData(), KERNEL_STATUS_PARAM_INVALID, "[%s] Get input[x1] data failed",
                       ctx.GetOpType().c_str())

  auto input1_tensor = ctx.Input(1);
  auto input1_tensor_shape = input1_tensor->GetTensorShape();
  KERNEL_CHECK_NULLPTR(input1_tensor->GetData(), KERNEL_STATUS_PARAM_INVALID, "[%s] Get input[x2] data failed",
                       ctx.GetOpType().c_str())

  DataType input0_data_type = input0_tensor->GetDataType();
  DataType input1_data_type = input1_tensor->GetDataType();
  KERNEL_CHECK_FALSE((input0_data_type == input1_data_type), KERNEL_STATUS_PARAM_INVALID,
                     "Input[x1] data type[%s] and input[x2] data type[%s] must be same",
                     DTypeStr(input0_data_type).c_str(), DTypeStr(input1_data_type).c_str())
  KERNEL_LOG_DEBUG("%s op input[x1] data type is [%s].", kMatmul, DTypeStr(input0_data_type).c_str());
  uint32_t ret = KERNEL_STATUS_OK;
  switch (input0_data_type) {
    case DT_FLOAT:
      ret = MatMulCompute<float>(ctx);
      break;
    case DT_DOUBLE:
      ret = MatMulCompute<double>(ctx);
      break;
    case DT_FLOAT16:
      ret = MatMulCompute<Eigen::half>(ctx);
      break;
    case DT_INT32:
      ret = MatMulCompute<int32_t>(ctx);
      break;
    case DT_COMPLEX64:
      ret = MatMulCompute<std::complex<float>>(ctx);
      break;
    case DT_COMPLEX128:
      ret = MatMulCompute<std::complex<double>>(ctx);
      break;
    default:
      KERNEL_LOG_ERROR("[%s] Data type of input is not support, input data type is [%s].", ctx.GetOpType().c_str(),
                       DTypeStr(input0_data_type).c_str());
      ret = KERNEL_STATUS_PARAM_INVALID;
  }
  return ret;
}

REGISTER_CPU_KERNEL(kMatmul, MatMulCpuKernel);
}  // namespace aicpu
