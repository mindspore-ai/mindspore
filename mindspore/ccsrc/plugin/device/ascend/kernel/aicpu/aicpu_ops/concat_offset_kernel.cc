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
#include "plugin/device/ascend/kernel/aicpu/aicpu_ops/concat_offset_kernel.h"
#include <vector>
#include <string>
#include "proto/aicpu_tensor.pb.h"

namespace aicpu {
namespace {
constexpr size_t kConcatOffsetOutputShapeRank = 2;

std::vector<int64_t> GetShape(const ::aicpuops::TensorShape &shape) {
  std::vector<int64_t> res;
  for (int i = 0; i < shape.dim_size(); ++i) {
    res.push_back(shape.dim(i).size());
  }
  return res;
}
}  // namespace
bool ConcatOffsetKernel::CheckParams() {
  auto input_num = input_shapes_.size();
  if (input_num == 0) {
    AICPU_LOGE("For 'ConcatOffset', input tensor number can not be 0.");
    return false;
  }

  // check input shapes
  auto x_rank = input_shapes_[0].size();
  for (size_t i = 1; i < input_num; ++i) {
    if (input_shapes_[i].size() != x_rank) {
      AICPU_LOGE(
        "For 'ConcatOffset', input tensors shape's rank must be equal, but got input[0] shape's rank = %lu, but got "
        "input[%lu] shape's rank = %lu",
        x_rank, i, input_shapes_[i].size());
      return false;
    }
  }

  // check axis
  auto x_rank_i = static_cast<int64_t>(SizeToInt(x_rank));
  if (axis_ < -x_rank_i || axis_ >= x_rank_i) {
    AICPU_LOGE("For 'ConcatOffset', 'axis' must be in range [-%ld, %ld), but got %ld", x_rank_i, x_rank_i, axis_);
    return false;
  }
  if (axis_ < 0) {
    axis_ += x_rank_i;
  }

  // check output shape
  if (output_shape_.size() != kConcatOffsetOutputShapeRank) {
    AICPU_LOGE("For 'ConcatOffset', output tensor shape rank must be %lu, but got %lu", kConcatOffsetOutputShapeRank,
               output_shape_.size());
    return false;
  }
  auto m = LongToSize(output_shape_[0]);
  if (m != input_num) {
    AICPU_LOGE("For 'ConcatOffset', output tensor shape[0] must be equal to input tensor number, but got: %lu vs %lu",
               m, input_num);
    return false;
  }
  return true;
}

uint32_t ConcatOffsetKernel::ConcatOffsetTask() {
  if (io_addrs_.empty() || !CheckParams()) {
    return kAicpuKernelStateFailed;
  }

  // calc offset
  std::vector<int64_t> offset{0};
  auto axis = LongToSize(axis_);
  auto sum_axis = input_shapes_[0][axis];
  for (size_t i = 1; i < input_shapes_.size(); ++i) {
    offset.push_back(sum_axis);
    sum_axis += input_shapes_[i][axis];
  }

  auto out_addr = reinterpret_cast<int64_t *>(io_addrs_[io_addrs_.size() - 1]);
  size_t idx = 0;
  for (size_t i = 0; i < LongToSize(output_shape_[0]); ++i) {
    for (size_t j = 0; j < LongToSize(output_shape_[1]); ++j) {
      if (j == axis) {
        out_addr[idx] = offset[i];
      } else {
        out_addr[idx] = 0;
      }
      ++idx;
    }
  }

  return kAicpuKernelStateSucess;
}

uint32_t ConcatOffsetKernel::ParseKernelParam() {
  ::google::protobuf::Map<::std::string, ::aicpuops::AttrValue> attrs = node_def_.attrs();
  // get value of attr axis
  axis_ = attrs["axis"].i();

  // get input tensors shape
  for (int i = 0; i < node_def_.inputs_size(); ++i) {
    aicpuops::Tensor input_tensor = node_def_.inputs(i);
    auto input_shape = GetShape(input_tensor.tensor_shape());
    input_shapes_.push_back(input_shape);
  }

  // get output tensor shape
  if (node_def_.outputs_size() != 1) {
    AICPU_LOGE("For 'ConcatOffset', output tensor number must be 1, but got %d", node_def_.outputs_size());
    return kAicpuKernelStateInvalid;
  }
  aicpuops::Tensor output_tensor = node_def_.outputs(0);
  output_shape_ = GetShape(output_tensor.tensor_shape());

  return kAicpuKernelStateSucess;
}

uint32_t ConcatOffsetKernel::DoCompute() { return ConcatOffsetTask(); }
}  // namespace aicpu

extern "C" {
__attribute__((visibility("default"))) uint32_t ConcatOffset(void *param) {
  aicpu::ConcatOffsetKernel concat_offset_kernel;
  return concat_offset_kernel.Compute(param);
}
}
