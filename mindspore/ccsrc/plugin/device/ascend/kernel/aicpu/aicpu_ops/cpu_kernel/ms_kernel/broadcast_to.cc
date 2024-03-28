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

#include "cpu_kernel/ms_kernel/broadcast_to.h"
#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

#include "Eigen/Core"

#include "context/inc/cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const uint32_t kInputNum = 2;
const uint32_t kOutputNum = 1;
const char *kBroadcastTo = "BroadcastTo";

#define BROADCAST_TO_COMPUTE_CASE(DTYPE, TYPE, ITYPE, CTX) \
  case (DTYPE): {                                          \
    if ((ITYPE) == "DT_INT32") {                           \
      return BcastCompute<TYPE, int32_t>(CTX);             \
    } else {                                               \
      return BcastCompute<TYPE, int64_t>(CTX);             \
    }                                                      \
    break;                                                 \
  }
}  // namespace

namespace aicpu {

uint32_t BroadcastToCpuKernel::Compute(CpuKernelContext &ctx) {
  CUST_KERNEL_HANDLE_ERROR(ctx, NormalCheck(ctx, kInputNum, kOutputNum),
                           "BroadcastTo check input and output number failed.");

  DataType input_data_type = ctx.Input(0)->GetDataType();
  std::string input1_data_type = "DT_INT32";
  if (ctx.Input(1)->GetDataType() == DT_INT64) {
    input1_data_type = "DT_INT64";
  }
  switch (input_data_type) {
    BROADCAST_TO_COMPUTE_CASE(DT_COMPLEX64, std::complex<float>, input1_data_type, ctx)
    BROADCAST_TO_COMPUTE_CASE(DT_COMPLEX128, std::complex<double>, input1_data_type, ctx)
    BROADCAST_TO_COMPUTE_CASE(DT_FLOAT, float, input1_data_type, ctx)
    BROADCAST_TO_COMPUTE_CASE(DT_FLOAT16, Eigen::half, input1_data_type, ctx)
    BROADCAST_TO_COMPUTE_CASE(DT_DOUBLE, double, input1_data_type, ctx)
    BROADCAST_TO_COMPUTE_CASE(DT_INT8, int8_t, input1_data_type, ctx)
    BROADCAST_TO_COMPUTE_CASE(DT_INT16, int16_t, input1_data_type, ctx)
    BROADCAST_TO_COMPUTE_CASE(DT_INT32, int32_t, input1_data_type, ctx)
    BROADCAST_TO_COMPUTE_CASE(DT_INT64, int64_t, input1_data_type, ctx)
    BROADCAST_TO_COMPUTE_CASE(DT_UINT8, uint8_t, input1_data_type, ctx)
    BROADCAST_TO_COMPUTE_CASE(DT_UINT16, uint16_t, input1_data_type, ctx)
    BROADCAST_TO_COMPUTE_CASE(DT_UINT32, uint32_t, input1_data_type, ctx)
    BROADCAST_TO_COMPUTE_CASE(DT_UINT64, uint64_t, input1_data_type, ctx)
    BROADCAST_TO_COMPUTE_CASE(DT_BOOL, bool, input1_data_type, ctx)
    default:
      CUST_KERNEL_LOG_ERROR(ctx, "BroadcastTo kernel data type [%s] not support.", DTypeStr(input_data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

uint32_t BroadcastToCpuKernel::BroadcastToParamCheck(CpuKernelContext &ctx) {
  Tensor *input = ctx.Input(0);
  Tensor *shape = ctx.Input(1);
  Tensor *output = ctx.Output(0);

  // check shape
  auto inputShape = shape->GetTensorShape();
  CUST_KERNEL_CHECK_FALSE(ctx, inputShape->GetDims() == 1, KERNEL_STATUS_PARAM_INVALID, "Input shape must be 1D.")
  DataType shape_type = shape->GetDataType();
  CUST_KERNEL_CHECK_FALSE(ctx, (shape_type == DT_INT32 || shape_type == DT_INT64), KERNEL_STATUS_PARAM_INVALID,
                          "The data type of probs need be DT_INT32 or DT_INT64.")
  CUST_KERNEL_LOG_DEBUG(ctx,
                        "BroadcastToCpuKernel[%s], input: size[%llu];"
                        "shape: size[%llu], output: size[%llu].",
                        ctx.GetOpType().c_str(), input->GetDataSize(), shape->GetDataSize(), output->GetDataSize());
  return KERNEL_STATUS_OK;
}

template <typename T1, typename T2>
uint32_t BroadcastToCpuKernel::BcastCompute(CpuKernelContext &ctx) {
  Tensor *input = ctx.Input(0);
  Tensor *output = ctx.Output(0);

  auto in0 = reinterpret_cast<T1 *>(ctx.Input(0)->GetData());
  auto in1 = reinterpret_cast<T2 *>(ctx.Input(1)->GetData());
  auto out = reinterpret_cast<T1 *>(ctx.Output(0)->GetData());

  auto input_shape = input->GetTensorShape()->GetDimSizes();

  int64_t data_num = ctx.Input(1)->NumElements();
  std::vector<int64_t> dims;
  int length = 1;
  for (int i = 0; i < data_num; i++) {
    dims.push_back(static_cast<int64_t>(*(in1 + i)));
    length = length * (static_cast<int64_t>(*(in1 + i)));
  }
  auto output_shape = output->GetTensorShape();
  output_shape->SetDimSizes(dims);
  auto out_shape = output->GetTensorShape()->GetDimSizes();
  Bcast bcast(ctx, input_shape, out_shape);

  if (!bcast.IsValid()) {
    CUST_KERNEL_LOG_ERROR(ctx, "[%s] broadcast failed!", ctx.GetOpType().c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  for (int i = 0; i < length; i++) {
    *(out + i) = (*(in0 + bcast.GetBroadcastXIndex(i)));
  }
  return KERNEL_STATUS_OK;
}

REGISTER_MS_CPU_KERNEL(kBroadcastTo, BroadcastToCpuKernel);
}  // namespace aicpu
