/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "backend/kernel_compiler/cpu/scatter_nd_update_cpu_kernel.h"
#include <string>
#include "runtime/device/cpu/cpu_device_address.h"

namespace mindspore {
namespace kernel {
void ScatterNdUpdateCPUKernel::InitKernel(const CNodePtr &kernel_node) {
  Check(kernel_node);
  auto shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  auto indices_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
  auto updates_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 2);
  auto indices_unit_rank = indices_shape.back();
  if (indices_unit_rank > shape.size()) {
    MS_LOG(EXCEPTION) << "Value of last dimension of indices is greater than shape rank";
  }
  if (indices_shape.size() < 2) {
    MS_LOG(EXCEPTION) << "Indices dimension less than 2";
  }
  if (updates_shape.size() != indices_shape.size() - 1 + shape.size() - indices_unit_rank) {
    MS_LOG(EXCEPTION) << "Update, shape rank and indices rank inconsistent";
  }
  for (size_t i = 0; i < indices_shape.size() - 1; ++i) {
    if (updates_shape[i] != indices_shape[i]) {
      MS_LOG(EXCEPTION) << "Value of " << i << "th dimension of indices is not equal to that update";
    }
  }
  indices_unit_rank_ = SizeToInt(indices_unit_rank);
  unit_size_ = 1;
  for (size_t i = indices_shape.size() - 1; i < updates_shape.size(); ++i) {
    unit_size_ *= SizeToInt(updates_shape[i]);
  }
  num_units_ = 1;
  num_units_ *= SizeToInt(updates_shape[indices_shape.size() - 2]);
  for (int i = SizeToInt(indices_shape.size()) - 3; i >= 0; i--) {
    num_units_ *= SizeToInt(updates_shape[i]);
  }
  int out_stride = 1;
  out_strides_.push_back(out_stride);
  for (int i = indices_unit_rank_ - 2; i >= 0; i--) {
    out_stride *= shape[i + 1];
    out_strides_.push_back(out_stride);
  }
  shape_ = shape;
  output_unit_offsets_.reserve(num_units_);
  dtype_ = AnfAlgo::GetPrevNodeOutputInferDataType(kernel_node, 0);
}

bool ScatterNdUpdateCPUKernel::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                      const std::vector<kernel::AddressPtr> & /*workspace*/,
                                      const std::vector<kernel::AddressPtr> & /*outputs*/) {
  if (dtype_ == kNumberTypeFloat16) {
    LaunchKernel<float16>(inputs);
  } else if (dtype_ == kNumberTypeFloat32) {
    LaunchKernel<float>(inputs);
  } else {
    MS_LOG(ERROR) << "Only support float16, float32";
    return false;
  }
  return true;
}

template <typename T>
void ScatterNdUpdateCPUKernel::LaunchKernel(const std::vector<AddressPtr> &inputs) {
  auto x = reinterpret_cast<T *>(inputs[0]->addr);
  auto indices = reinterpret_cast<int *>(inputs[1]->addr);
  auto updates = reinterpret_cast<T *>(inputs[2]->addr);

  for (int i = 0; i < num_units_; ++i) {
    int offset = 0;
    for (int j = 0; j < indices_unit_rank_; ++j) {
      auto index = indices[i * indices_unit_rank_ + j];
      if (index < 0) {
        MS_LOG(EXCEPTION) << "Error, Indices exist element which less than 0. element=" << index;
      }
      offset += index * out_strides_[j] * unit_size_;
    }
    output_unit_offsets_[i] = offset;
  }

  auto mem_size = inputs[0]->size;
  for (int i = 0; i < num_units_; i++) {
    auto ret = memcpy_s(x + output_unit_offsets_[i], mem_size, updates + unit_size_ * i, unit_size_ * sizeof(T));
    if (ret != 0) {
      MS_LOG(EXCEPTION) << "memcpy_s error, errorno" << ret;
    }
  }
}

void ScatterNdUpdateCPUKernel::Check(const CNodePtr &kernel_node) {
  size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
  if (input_num != 3) {
    MS_LOG(EXCEPTION) << "Input number is " << input_num << ", but ScatterNdUpdate needs 3 input.";
  }
  size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
  if (output_num != 1) {
    MS_LOG(EXCEPTION) << "Output number is " << output_num << ", but ScatterNdUpdate needs 1 output.";
  }
}
}  // namespace kernel
}  // namespace mindspore
