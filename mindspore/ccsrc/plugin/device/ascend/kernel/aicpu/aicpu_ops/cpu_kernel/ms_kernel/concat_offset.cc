/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include "concat_offset.h"
#include <vector>
#include <string>
#include "inc/kernel_log.h"

namespace aicpu {
namespace {
constexpr size_t kConcatOffsetOutputShapeRank = 2;
const char *kConcatOffset = "ConcatOffset";
}  // namespace
bool ConcatOffsetKernel::CheckParams(CpuKernelContext &ctx) {
  auto input_num = input_shapes_.size();
  if (input_num == 0) {
    CUST_AICPU_LOGE(ctx, "For 'ConcatOffset', input tensor number can not be 0.");
    return false;
  }

  // check input shapes
  auto x_rank = input_shapes_[0].size();
  for (size_t i = 1; i < input_num; ++i) {
    if (input_shapes_[i].size() != x_rank) {
      CUST_AICPU_LOGE(
        ctx,
        "For 'ConcatOffset', input tensors shape's rank must be equal, but got input[0] shape's rank = %lu, but got "
        "input[%lu] shape's rank = %lu",
        x_rank, i, input_shapes_[i].size());
      return false;
    }
  }

  // check axis
  auto x_rank_i = static_cast<int64_t>(x_rank);
  if (axis_ < -x_rank_i || axis_ >= x_rank_i) {
    CUST_AICPU_LOGE(ctx, "For 'ConcatOffset', 'axis' must be in range [-%ld, %ld), but got %ld", x_rank_i, x_rank_i,
                    axis_);
    return false;
  }
  if (axis_ < 0) {
    axis_ += x_rank_i;
  }

  // check output shape
  if (output_shape_.size() != kConcatOffsetOutputShapeRank) {
    CUST_AICPU_LOGE(ctx, "For 'ConcatOffset', output tensor shape rank must be %lu, but got %lu",
                    kConcatOffsetOutputShapeRank, output_shape_.size());
    return false;
  }
  auto m = static_cast<size_t>(output_shape_[0]);
  if (m != input_num) {
    CUST_AICPU_LOGE(
      ctx, "For 'ConcatOffset', output tensor shape[0] must be equal to input tensor number, but got: %lu vs %lu", m,
      input_num);
    return false;
  }
  return true;
}

uint32_t ConcatOffsetKernel::Compute(CpuKernelContext &ctx) {
  RETURN_IF_FAILURE(ParseKernelParam(ctx));
  if (!CheckParams(ctx)) {
    return KERNEL_STATUS_INNER_ERROR;
  }

  // calc offset
  std::vector<int64_t> offset{0};
  auto axis = static_cast<size_t>(axis_);
  auto sum_axis = input_shapes_[0][axis];
  for (size_t i = 1; i < input_shapes_.size(); ++i) {
    offset.push_back(sum_axis);
    sum_axis += input_shapes_[i][axis];
  }

  auto out_addr = reinterpret_cast<int64_t *>(ctx.Output(0)->GetData());
  size_t idx = 0;
  for (size_t i = 0; i < static_cast<size_t>(output_shape_[0]); ++i) {
    for (size_t j = 0; j < static_cast<size_t>(output_shape_[1]); ++j) {
      if (j == axis) {
        out_addr[idx] = offset[i];
      } else {
        out_addr[idx] = 0;
      }
      ++idx;
    }
  }

  return KERNEL_STATUS_OK;
}

uint32_t ConcatOffsetKernel::ParseKernelParam(CpuKernelContext &ctx) {
  // get value of attr axis
  auto axis = ctx.GetAttr("axis");
  CUST_KERNEL_CHECK_NULLPTR(ctx, axis, KERNEL_STATUS_INNER_ERROR, "Failed to get attr 'axis'.");
  axis_ = axis->GetInt();

  // get input tensors shape
  for (size_t i = 0; i < ctx.GetInputsSize(); ++i) {
    auto input_shape = ctx.Input(i)->GetTensorShape()->GetDimSizes();
    input_shapes_.push_back(input_shape);
  }

  // get output tensor shape
  if (ctx.GetOutputsSize() != 1) {
    CUST_AICPU_LOGE(ctx, "For 'ConcatOffset', output tensor number must be 1, but got %d", ctx.GetOutputsSize());
    return KERNEL_STATUS_INNER_ERROR;
  }
  output_shape_ = ctx.Output(0)->GetTensorShape()->GetDimSizes();

  return KERNEL_STATUS_OK;
}

REGISTER_MS_CPU_KERNEL(kConcatOffset, ConcatOffsetKernel);
}  // namespace aicpu