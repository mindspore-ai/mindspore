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
#include <algorithm>
#include "backend/kernel_compiler/cpu/slice_cpu_kernel.h"
#include "runtime/device/cpu/cpu_device_address.h"

namespace mindspore {
namespace kernel {
constexpr int MAX_DIMS = 8;
void SliceCPUKernel::InitKernel(const CNodePtr &kernel_node) {
  CheckParam(kernel_node);
  input_shape_ = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  dtype_ = AnfAlgo::GetPrevNodeOutputInferDataType(kernel_node, 0);
  std::vector<int64_t> begin_me = AnfAlgo::GetNodeAttr<std::vector<int64_t>>(kernel_node, BEGIN);
  (void)std::transform(begin_me.begin(), begin_me.end(), std::back_inserter(begin_),
                       [](const int64_t &value) { return static_cast<int>(value); });
  auto prim = AnfAlgo::GetCNodePrimitive(kernel_node);
  MS_EXCEPTION_IF_NULL(prim);
  auto strides = prim->GetAttr(STRIDES);
  if (strides != nullptr) {
    std::vector<int64_t> strides_me = AnfAlgo::GetNodeAttr<std::vector<int64_t>>(kernel_node, STRIDES);
    std::vector<int64_t> end_me = AnfAlgo::GetNodeAttr<std::vector<int64_t>>(kernel_node, END);
    (void)std::transform(strides_me.begin(), strides_me.end(), std::back_inserter(strides_),
                         [](const int64_t &value) { return static_cast<int>(value); });
    (void)std::transform(end_me.begin(), end_me.end(), std::back_inserter(end_),
                         [](const int64_t &value) { return static_cast<int>(value); });
    TransArg();
    ClipBegin();
  } else {
    std::vector<int> sizes;
    std::vector<int64_t> sizes_me = AnfAlgo::GetNodeAttr<std::vector<int64_t>>(kernel_node, SIZE);
    (void)std::transform(sizes_me.begin(), sizes_me.end(), std::back_inserter(sizes),
                         [](const int64_t &value) { return static_cast<int>(value); });
    if (sizes.size() != input_shape_.size() || begin_.size() != input_shape_.size()) {
      MS_LOG(EXCEPTION) << "begin|size|input size must be equal";
    }
    ClipBegin();
    for (size_t i = 0; i < sizes.size(); ++i) {
      while (sizes[i] < 0) {
        sizes[i] = sizes[i] + SizeToInt(input_shape_[i]);
      }
      strides_.emplace_back(1);
      end_.emplace_back(begin_[i] + sizes[i]);
    }
  }
  ExpandAllMemberDims();
  CPUKernelUtils::GetElementNumEveryDim(input_shape_, &input_element_num_);
  CPUKernelUtils::GetElementNumEveryDim(output_shape_, &output_element_num_);
}
void SliceCPUKernel::ClipBegin() {
  for (size_t i = 0; i < begin_.size(); i++) {
    if (begin_[i] < 0) {
      auto k = begin_[i] + SizeToInt(input_shape_[i]);
      begin_[i] = k < 0 ? 0 : k;
    }
    if (begin_[i] > SizeToInt(input_shape_[i])) {
      begin_[i] = SizeToInt(input_shape_[i]);
    }
  }
}
void SliceCPUKernel::ExpandAllMemberDims() {
  auto input_len = input_shape_.size();
  if (input_len < 4) {
    for (size_t i = 0; i < 4 - input_len; ++i) {
      input_shape_.insert(input_shape_.begin(), 1);
      begin_.insert(begin_.begin(), 0);
      strides_.insert(strides_.begin(), 1);
      end_.insert(end_.begin(), 1);
    }
  }
  for (size_t i = 0; i < 4; ++i) {
    if (SignOfStride(i)) {
      int ax = (end_[i] - begin_[i]) * SignOfStride(i);
      if (ax < 0) {
        ax = 0;
      }
      output_shape_.push_back(IntToSize(ax));
    }
  }
}

bool SliceCPUKernel::Launch(const std::vector<kernel::AddressPtr> &inputs,
                            const std::vector<kernel::AddressPtr> & /*workspace*/,
                            const std::vector<kernel::AddressPtr> &outputs) {
  bool ret{true};
  if (dtype_ == kNumberTypeInt32) {
    ret = LaunchKernel<int>(inputs, outputs);
  } else if (dtype_ == kNumberTypeFloat32) {
    ret = LaunchKernel<float>(inputs, outputs);
  } else if (dtype_ == kNumberTypeBool) {
    ret = LaunchKernel<bool>(inputs, outputs);
  } else {
    MS_LOG(ERROR) << "Slice op only support input_x int32 and float32";
    return false;
  }
  return ret;
}

template <typename T>
bool SliceCPUKernel::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                  const std::vector<kernel::AddressPtr> &outputs) {
  T *input_addr = reinterpret_cast<T *>(inputs[0]->addr);
  T *output_addr = reinterpret_cast<T *>(outputs[0]->addr);
  bool can_copy_memory[3] = {CanCopyMemoryOnAxis(0), CanCopyMemoryOnAxis(1), CanCopyMemoryOnAxis(2)};
  int signstride[4] = {SignOfStride(0), SignOfStride(1), SignOfStride(2), SignOfStride(3)};
  size_t in_start_offset[3] = {begin_[0] * input_element_num_[0], begin_[1] * input_element_num_[1],
                               begin_[2] * input_element_num_[2]};
  size_t in_step_size[3] = {strides_[0] * input_element_num_[0], strides_[1] * input_element_num_[1],
                            strides_[2] * input_element_num_[2]};

  auto in_n_offset = in_start_offset[0];
  auto out_n_offset = 0;
  for (int i = begin_[0]; signstride[0] * i < signstride[0] * end_[0];
       i += strides_[0], in_n_offset += in_step_size[0], out_n_offset += output_element_num_[0]) {
    if (can_copy_memory[0]) {
      CopyDataToOutput<T>(inputs, in_n_offset, outputs, out_n_offset, input_element_num_[0], 0);
      continue;
    }
    auto in_c_offset = in_start_offset[1];
    auto out_c_offset = 0;
    for (int j = begin_[1]; signstride[1] * j < signstride[1] * end_[1];
         j += strides_[1], in_c_offset += in_step_size[1], out_c_offset += output_element_num_[1]) {
      if (can_copy_memory[1]) {
        CopyDataToOutput<T>(inputs, in_n_offset + in_c_offset, outputs, out_n_offset + out_c_offset,
                            input_element_num_[1], 1);
        continue;
      }
      auto in_h_offset = in_start_offset[2];
      auto out_h_offset = 0;
      for (int k = begin_[2]; signstride[2] * k < signstride[2] * end_[2];
           k += strides_[2], in_h_offset += in_step_size[2], out_h_offset += output_element_num_[2]) {
        if (can_copy_memory[2]) {
          CopyDataToOutput<T>(inputs, in_n_offset + in_c_offset + in_h_offset, outputs,
                              out_n_offset + out_c_offset + out_h_offset, input_element_num_[2], 2);
          continue;
        }
        for (int m = begin_[3]; signstride[3] * m < signstride[3] * end_[3]; m += strides_[3]) {
          *output_addr++ = input_addr[in_n_offset + in_c_offset + in_h_offset + m];
        }
      }
    }
  }

  return true;
}

bool SliceCPUKernel::CanCopyMemoryOnAxis(size_t dim) const {
  for (size_t i = dim + 1; i < 4; ++i) {
    if (begin_[i] != 0 || end_[i] != SizeToInt(input_shape_[i]) || strides_[i] != 1) {
      return false;
    }
  }
  return true;
}

int SliceCPUKernel::SignOfStride(size_t axis) const {
  if (strides_[axis] > 0) {
    return 1;
  }
  return -1;
}

template <typename T>
void SliceCPUKernel::CopyDataToOutput(const std::vector<kernel::AddressPtr> &inputs, size_t in_offset,
                                      const std::vector<kernel::AddressPtr> &outputs, size_t out_offset,
                                      size_t copy_num, int id) const {
  T *input_addr = reinterpret_cast<T *>(inputs[0]->addr);
  auto in_buff_size = inputs[0]->size;
  T *output_addr = reinterpret_cast<T *>(outputs[0]->addr);
  auto out_buff_size = outputs[0]->size;

  if ((in_offset + copy_num) * sizeof(T) > in_buff_size) {
    MS_LOG(EXCEPTION) << "input memory out of bounds.";
  }
  if ((out_offset + copy_num) * sizeof(T) > out_buff_size) {
    MS_LOG(EXCEPTION) << id << " output memory out of bounds.";
  }

  size_t copy_size = copy_num * sizeof(T);
  auto ret = memcpy_s(output_addr + out_offset, copy_size, input_addr + in_offset, copy_size);
  if (ret != EOK) {
    MS_LOG(EXCEPTION) << "memcpy failed. ret:" << ret;
  }
}

void SliceCPUKernel::TransArg() {
  if (strides_.size() != end_.size() || strides_.size() != input_shape_.size()) {
    MS_LOG(EXCEPTION) << "stride|end|input size must be equal";
  }
  for (size_t i = 0; i < strides_.size(); ++i) {
    if (strides_[i] == 0) {
      MS_LOG(EXCEPTION) << "slice stride cannot be zero";
    }
    if (end_[i] == 0 && begin_[i] < 0) {
      end_[i] = end_[i] + SizeToInt(input_shape_[i]);
    }
    if (end_[i] < 0) {
      end_[i] = end_[i] + SizeToInt(input_shape_[i]) < 0 ? 0 : end_[i] + SizeToInt(input_shape_[i]);
    }
    if (end_[i] > SizeToInt(input_shape_[i])) {
      end_[i] = SizeToInt(input_shape_[i]);
    }
  }
}

void SliceCPUKernel::CheckParam(const CNodePtr &kernel_node) const {
  size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
  if (input_num != 1) {
    MS_LOG(EXCEPTION) << "Input number is " << input_num << ", but SliceCPUKernel needs 1 inputs.";
  }
  size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
  if (output_num != 1) {
    MS_LOG(EXCEPTION) << "Output number is " << output_num << ", but SliceCPUKernel needs 1 output.";
  }
  auto input_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  if (input_shape.size() > MAX_DIMS) {
    MS_LOG(EXCEPTION) << "Input dims is " << input_shape.size() << ", but SliceCPUKernel olny support 4d or lower.";
  }
  if (input_shape.size() == 0) {
    MS_LOG(EXCEPTION) << "Input dims is " << input_shape.size() << ", scalar is not supported.";
  }
}
}  // namespace kernel
}  // namespace mindspore
