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

#include "cpu_kernel/ms_kernel/select.h"
#include <vector>
#include <algorithm>
#include "context/inc/cpu_kernel_utils.h"
#include "utils/broadcast_iterator.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const uint32_t kOutputNum = 1;
const uint32_t kInputNum = 3;
const char *kSelect = "Select";

#define SELECT_COMPUTE_CASE(DTYPE, TYPE, CTX)                      \
  case (DTYPE): {                                                  \
    uint32_t result = SelectCompute<TYPE>(CTX);                    \
    if (result != KERNEL_STATUS_OK) {                              \
      CUST_KERNEL_LOG_ERROR(ctx, "Select kernel compute failed."); \
      return result;                                               \
    }                                                              \
    break;                                                         \
  }
}  // namespace

namespace aicpu {
uint32_t SelectCpuKernel::Compute(CpuKernelContext &ctx) {
  // check params
  CUST_KERNEL_HANDLE_ERROR(ctx, NormalCheck(ctx, kInputNum, kOutputNum),
                           "Select check input and output number failed.");
  CUST_KERNEL_HANDLE_ERROR(ctx, SelectParamCheck(ctx), "Select check params failed.");
  auto data_type = ctx.Input(1)->GetDataType();
  switch (data_type) {
    SELECT_COMPUTE_CASE(DT_INT8, int8_t, ctx)
    SELECT_COMPUTE_CASE(DT_INT16, int16_t, ctx)
    SELECT_COMPUTE_CASE(DT_INT32, int32_t, ctx)
    SELECT_COMPUTE_CASE(DT_INT64, int64_t, ctx)
    SELECT_COMPUTE_CASE(DT_UINT8, uint8_t, ctx)
    SELECT_COMPUTE_CASE(DT_UINT16, uint16_t, ctx)
    SELECT_COMPUTE_CASE(DT_UINT32, uint32_t, ctx)
    SELECT_COMPUTE_CASE(DT_UINT64, uint64_t, ctx)
    SELECT_COMPUTE_CASE(DT_BOOL, uint64_t, ctx)
    SELECT_COMPUTE_CASE(DT_FLOAT16, Eigen::half, ctx)
    SELECT_COMPUTE_CASE(DT_FLOAT, float, ctx)
    SELECT_COMPUTE_CASE(DT_DOUBLE, double, ctx)
    SELECT_COMPUTE_CASE(DT_COMPLEX128, std::complex<double>, ctx);
    SELECT_COMPUTE_CASE(DT_COMPLEX64, std::complex<double>, ctx);
    default:
      CUST_KERNEL_LOG_ERROR(ctx, "Select kernel data type [%s] not support.", DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }

  return KERNEL_STATUS_OK;
}

uint32_t SelectCpuKernel::SelectParamCheck(CpuKernelContext &ctx) {
  // the non null of input_0, input_1, output has been verified in NormalCheck
  Tensor *input_0 = ctx.Input(0);
  Tensor *input_1 = ctx.Input(1);
  Tensor *input_2 = ctx.Input(2);
  Tensor *output = ctx.Output(0);
  DataType input0_type = input_0->GetDataType();
  DataType input1_type = input_1->GetDataType();
  DataType input2_type = input_2->GetDataType();

  auto input_shape_a = ctx.Input(1)->GetTensorShape()->GetDimSizes();
  auto input_shape_b = ctx.Input(2)->GetTensorShape()->GetDimSizes();

  if (input0_type != DT_BOOL) {
    CUST_KERNEL_LOG_ERROR(ctx, "[%s] Data type of mask requires bool, but got data type [%s].", ctx.GetOpType().c_str(),
                          DTypeStr(input0_type).c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }

  CUST_KERNEL_CHECK_FALSE(ctx, (input1_type == input2_type), KERNEL_STATUS_PARAM_INVALID,
                          "The data type of input1 [%s] need be same with "
                          "input2 [%s].",
                          DTypeStr(input1_type).c_str(), DTypeStr(input2_type).c_str())

  if (input_shape_a != input_shape_b) {
    CUST_KERNEL_LOG_ERROR(ctx, "The shape of X1 must equal X2.");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  CUST_KERNEL_LOG_DEBUG(ctx,
                        "SelectCpuKernel[%s], input0: size[%llu];"
                        "input1: size[%llu], input2: size[%llu], output: size[%llu].",
                        ctx.GetOpType().c_str(), input_0->GetDataSize(), input_1->GetDataSize(), input_2->GetDataSize(),
                        output->GetDataSize());

  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t SelectCpuKernel::SelectCompute(CpuKernelContext &ctx) {
  bool *condition = static_cast<bool *>(ctx.Input(0)->GetData());
  T *x1 = static_cast<T *>(ctx.Input(1)->GetData());
  T *x2 = static_cast<T *>(ctx.Input(2)->GetData());
  T *y = static_cast<T *>(ctx.Output(0)->GetData());
  auto input_shape_a = ctx.Input(1)->GetTensorShape()->GetDimSizes();
  auto input_shape_mask = ctx.Input(0)->GetTensorShape()->GetDimSizes();
  std::vector<int64_t> output_shape;
  int64_t tensor_size = 1;
  int64_t position = 0;
  if (input_shape_a == input_shape_mask) {
    tensor_size =
      std::accumulate(input_shape_a.begin(), input_shape_a.end(), static_cast<int64_t>(1), std::multiplies<int64_t>());
    for (int64_t i = 0; i < tensor_size; ++i) {
      if (condition[i]) {
        y[position++] = x1[i];
      } else {
        y[position++] = x2[i];
      }
    }
  } else {
    auto ret = GetBroadcastShape(input_shape_a, input_shape_mask, output_shape);
    CUST_KERNEL_CHECK_FALSE(ctx, ret == KERNEL_STATUS_OK, KERNEL_STATUS_PARAM_INVALID,
                            "Shape of x and mask can't be broadcast.");
    tensor_size =
      std::accumulate(output_shape.begin(), output_shape.end(), static_cast<int64_t>(1), std::multiplies<int64_t>());
    BroadcastIterator iter(input_shape_a, input_shape_mask, output_shape);
    iter.SetPos(0);
    for (int64_t i = 0; i < tensor_size; ++i) {
      if (condition[iter.GetInputPosB()]) {
        y[position++] = x1[i];
      } else {
        y[position++] = x2[i];
      }
      iter.GenNextPos();
    }
  }
  ctx.Output(0)->GetTensorShape()->SetDimSizes({position});
  return KERNEL_STATUS_OK;
}

REGISTER_MS_CPU_KERNEL(kSelect, SelectCpuKernel);
}  // namespace aicpu
