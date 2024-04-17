/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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

#include "cpu_kernel/ms_kernel/shuffle_channel.h"
#include <vector>
#include "context/inc/cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const uint32_t kInputNum = 1;
const uint32_t kOutputNum = 1;
const char *kShuffleChannel = "ShuffleChannel";
const int64_t minDimSize = 3;

#define SHUFFLE_CHANNEL_COMPUTE_CASE(DTYPE, TYPE, CTX)                      \
  case (DTYPE): {                                                           \
    uint32_t result = ShuffleChannelCompute<TYPE>(CTX);                     \
    if (result != KERNEL_STATUS_OK) {                                       \
      CUST_KERNEL_LOG_ERROR(ctx, "Shuffle Channel kernel compute failed."); \
      return result;                                                        \
    }                                                                       \
    break;                                                                  \
  }
}  // namespace

namespace aicpu {
uint32_t ShuffleChannelCpuKernel::Compute(CpuKernelContext &ctx) {
  // check params
  CUST_KERNEL_HANDLE_ERROR(ctx, NormalCheck(ctx, kInputNum, kOutputNum),
                           "ShuffleChannel check input and output number failed.");
  CUST_KERNEL_HANDLE_ERROR(ctx, ShuffleChannelParamCheck(ctx), "ShuffleChannel check params failed.");
  auto data_type = ctx.Input(0)->GetDataType();
  switch (data_type) {
    SHUFFLE_CHANNEL_COMPUTE_CASE(DT_INT8, int8_t, ctx)
    SHUFFLE_CHANNEL_COMPUTE_CASE(DT_INT16, int16_t, ctx)
    SHUFFLE_CHANNEL_COMPUTE_CASE(DT_INT32, int32_t, ctx)
    SHUFFLE_CHANNEL_COMPUTE_CASE(DT_INT64, int64_t, ctx)
    SHUFFLE_CHANNEL_COMPUTE_CASE(DT_UINT8, uint8_t, ctx)
    SHUFFLE_CHANNEL_COMPUTE_CASE(DT_UINT16, uint16_t, ctx)
    SHUFFLE_CHANNEL_COMPUTE_CASE(DT_UINT32, uint32_t, ctx)
    SHUFFLE_CHANNEL_COMPUTE_CASE(DT_UINT64, uint64_t, ctx)
    SHUFFLE_CHANNEL_COMPUTE_CASE(DT_FLOAT16, Eigen::half, ctx)
    SHUFFLE_CHANNEL_COMPUTE_CASE(DT_FLOAT, float, ctx)
    default:
      CUST_KERNEL_LOG_ERROR(ctx, "Shuffle kernel data type [%s] not support.", DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

uint32_t ShuffleChannelCpuKernel::ShuffleChannelParamCheck(CpuKernelContext &ctx) {
  // the non null of input_0, input_1, output has been verified in NormalCheck
  Tensor *input = ctx.Input(0);
  auto group = 1;
  if (ctx.GetAttr("group")) {
    group = ctx.GetAttr("group")->GetInt();
  }
  int64_t c = input->GetTensorShape()->GetDimSize(1);
  CUST_KERNEL_CHECK_FALSE(ctx, input->GetTensorShape()->GetDims() >= minDimSize, KERNEL_STATUS_PARAM_INVALID,
                          "ShuffleChannel expect  input with > 2 dims.")
  CUST_KERNEL_CHECK_FALSE(ctx, group > 0, KERNEL_STATUS_PARAM_INVALID,
                          "Number of groups to divide channels in must be positive.")
  CUST_KERNEL_CHECK_FALSE(ctx, (c % group) == 0, KERNEL_STATUS_PARAM_INVALID,
                          "Number of channels must be divisible by groups")
  CUST_KERNEL_LOG_DEBUG(ctx,
                        "ShuffleChannelCpuKernel[%s], input: size[%llu];"
                        "output: size[%llu].",
                        ctx.GetOpType().c_str(), input->GetDataSize(), ctx.Output(0)->GetDataSize());

  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t ShuffleChannelCpuKernel::ShuffleChannelCompute(CpuKernelContext &ctx) {
  Tensor *input = ctx.Input(0);
  Tensor *output = ctx.Output(0);
  auto group = 1;
  if (ctx.GetAttr("group")) {
    group = ctx.GetAttr("group")->GetInt();
  }

  auto shape = input->GetTensorShape();
  int64_t b = shape->GetDimSize(0);
  int64_t c = shape->GetDimSize(1);
  int64_t dims = shape->GetDims();

  if (group == 0) {
    CUST_KERNEL_LOG_ERROR(ctx, "group divided can not be zero");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  int64_t oc = c / group;

  auto out = reinterpret_cast<T *>(output->GetData());
  auto in = reinterpret_cast<T *>(input->GetData());
  int64_t loc_out = 0;
  int64_t loc_in = 0;
  int64_t area = 1;
  for (int64_t i = 2; i < dims; i++) {
    area = area * shape->GetDimSize(i);
  }
  /*
    view the shape to n g c/g h*w,and transpose dim 1 and dim 2
  */
  std::vector<int64_t> temp_shape;
  temp_shape.push_back(b);
  temp_shape.push_back(group);
  temp_shape.push_back(oc);
  temp_shape.push_back(area);
  std::vector<int64_t> temp_loc = {0, 0, 0, 0};
  while (true) {
    int64_t dim = 0;
    for (dim = 0; dim <= minDimSize; dim++) {
      if (dim == minDimSize) {
        loc_in = loc_in + temp_loc[dim] * 1;
        loc_out = loc_out + temp_loc[dim] * 1;
      } else if (dim == (minDimSize - 1)) {
        loc_in = loc_in + temp_loc[dim] * area;
        loc_out = loc_out + temp_loc[1] * area;
      } else if (dim == 1) {
        loc_in = loc_in + temp_loc[dim] * oc * area;
        loc_out = loc_out + temp_loc[minDimSize - 1] * group * area;
      } else if (dim == 0) {
        loc_in = loc_in + temp_loc[dim] * group * oc * area;
        loc_out = loc_out + temp_loc[dim] * group * oc * area;
      }
    }
    *(out + loc_out) = *(in + loc_in);
    loc_in = 0;
    loc_out = 0;
    bool can_plus = false;
    for (dim = 0; dim <= minDimSize; dim++) {
      if (temp_loc[dim] < (temp_shape[dim] - 1)) {
        can_plus = true;
        break;
      }
    }
    if (!can_plus) {
      break;
    }
    for (dim = minDimSize; dim >= 0; dim--) {
      if (temp_loc[dim] == (temp_shape[dim] - 1)) {
        temp_loc[dim] = 0;
      } else {
        temp_loc[dim] = temp_loc[dim] + 1;
        break;
      }
    }
  }
  return KERNEL_STATUS_OK;
}

REGISTER_MS_CPU_KERNEL(kShuffleChannel, ShuffleChannelCpuKernel);
}  // namespace aicpu
