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
#include "cpu_kernel/ms_kernel/maximum_grad_grad.h"

#include <fstream>
#include <iostream>

#include "cpu_kernel/common/cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
constexpr uint32_t kMaximumGradGradInputNum = 4;
constexpr uint32_t kMaximumGradGradOutputNum = 3;
const char *kMaximumGradGrad = "MaximumGradGrad";

#define MAXIMUMGRADGRAD_COMPUTE_CASE(DTYPE, TYPE, CTX)            \
  case (DTYPE): {                                                 \
    uint32_t result = MaximumGradGradCompute<TYPE>(CTX);          \
    if (result != KERNEL_STATUS_OK) {                             \
      KERNEL_LOG_ERROR("MaximumGradGrad kernel compute failed."); \
      return result;                                              \
    }                                                             \
    break;                                                        \
  }
}  // namespace

namespace aicpu {
uint32_t MaximumGradGradCpuKernel::Compute(CpuKernelContext &ctx) {
  // check params
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kMaximumGradGradInputNum, kMaximumGradGradOutputNum),
                      "MaximumGradGrad check input and output number failed.");
  KERNEL_HANDLE_ERROR(MaximumGradGradParamCheck(ctx), "MaximumGradGrad check params failed.");
  auto data_type = ctx.Input(0)->GetDataType();
  switch (data_type) {
    MAXIMUMGRADGRAD_COMPUTE_CASE(DT_INT32, int32_t, ctx)
    MAXIMUMGRADGRAD_COMPUTE_CASE(DT_FLOAT, float, ctx)
    MAXIMUMGRADGRAD_COMPUTE_CASE(DT_FLOAT16, Eigen::half, ctx)
    default:
      KERNEL_LOG_ERROR("The data type of input is not support, input data type is [%s].", DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

uint32_t MaximumGradGradCpuKernel::MaximumGradGradParamCheck(const CpuKernelContext &ctx) {
  // the non null of inputs and outputs has been verified in NormalCheck
  Tensor *x1 = ctx.Input(0);
  Tensor *x2 = ctx.Input(1);
  Tensor *grad_y1 = ctx.Input(2);
  Tensor *grad_y2 = ctx.Input(3);
  // type check
  DataType grad_y1_type = grad_y1->GetDataType();
  DataType grad_y2_type = grad_y2->GetDataType();
  DataType x1_type = x1->GetDataType();
  DataType x2_type = x2->GetDataType();
  KERNEL_CHECK_FALSE(((grad_y1_type == grad_y2_type) && (grad_y2_type == x1_type) && (x1_type == x2_type)),
                     KERNEL_STATUS_PARAM_INVALID,
                     "The data type of grad_y1 [%s], grad_y2 [%s], x1 [%s] and "
                     "x2 [%s] need to be same.",
                     DTypeStr(grad_y1_type).c_str(), DTypeStr(grad_y2_type).c_str(), DTypeStr(x1_type).c_str(),
                     DTypeStr(x2_type).c_str())
  // shape check
  auto grad_y1_shape = grad_y1->GetTensorShape()->GetDimSizes();
  auto grad_y2_shape = grad_y2->GetTensorShape()->GetDimSizes();
  auto x1_shape = x1->GetTensorShape()->GetDimSizes();
  auto x2_shape = x2->GetTensorShape()->GetDimSizes();
  KERNEL_CHECK_FALSE(grad_y1_shape == x1_shape, KERNEL_STATUS_PARAM_INVALID, "Mismatch in shape of grad_y1 and x1.");
  KERNEL_CHECK_FALSE(grad_y2_shape == x2_shape, KERNEL_STATUS_PARAM_INVALID, "Mismatch in shape of grad_y2 and x2.");
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t MaximumGradGradCpuKernel::MaximumGradGradCompute(const CpuKernelContext &ctx) {
  Tensor *input0_tensor = ctx.Input(0);
  Tensor *input1_tensor = ctx.Input(1);

  auto input0_shape = input0_tensor->GetTensorShape()->GetDimSizes();
  auto input1_shape = input1_tensor->GetTensorShape()->GetDimSizes();

  Bcast bcast(input0_shape, input1_shape);
  if (!bcast.IsValid()) {
    KERNEL_LOG_ERROR("[%s] broadcast failed.", ctx.GetOpType().c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  return BcastCompute<T>(ctx, bcast);
}

template <typename T>
uint32_t MaximumGradGradCpuKernel::BcastCompute(const CpuKernelContext &ctx, const Bcast &bcast) {
  auto in0 = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  auto in1 = reinterpret_cast<T *>(ctx.Input(1)->GetData());
  auto in2 = reinterpret_cast<T *>(ctx.Input(2)->GetData());
  auto in3 = reinterpret_cast<T *>(ctx.Input(3)->GetData());
  auto out0 = reinterpret_cast<T *>(ctx.Output(0)->GetData());
  auto out1 = reinterpret_cast<T *>(ctx.Output(1)->GetData());
  auto out2 = reinterpret_cast<T *>(ctx.Output(2)->GetData());
  *out0 = static_cast<T>(0);
  *out1 = static_cast<T>(0);
  int64_t data_num = ctx.Output(2)->NumElements();

  for (int64_t i = 0; i < data_num; ++i) {
    if (*(in0 + bcast.GetBroadcastXIndex(i)) >= *(in1 + bcast.GetBroadcastYIndex(i))) {
      *(out2 + i) = *(in2 + bcast.GetBroadcastXIndex(i));
    } else {
      *(out2 + i) = *(in3 + bcast.GetBroadcastYIndex(i));
    }
  }

  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kMaximumGradGrad, MaximumGradGradCpuKernel);
}  // namespace aicpu
