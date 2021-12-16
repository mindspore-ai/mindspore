/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "backend/kernel_compiler/cpu/slice_grad_cpu_kernel.h"
#include <algorithm>
#include <string>
#include "runtime/device/cpu/cpu_device_address.h"
#include "ir/primitive.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kSliceGradDynamicInputsNum = 4;
constexpr size_t kStridedSliceGradDynamicInputsNum = 5;
constexpr size_t kOutputsNum = 1;
constexpr size_t kSliceGradMaxInputShapeSize = 4;
}  // namespace

void SliceGradCPUKernel::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = AnfAlgo::GetCNodeName(kernel_node);
  cnode_ptr_ = kernel_node;
  ClearVectors();
  auto input_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  if (input_shape.size() > kSliceGradMaxInputShapeSize) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimension of input tensor should be 4D or lower, but got "
                      << input_shape.size() << "D.";
  }
  output_shape_ = AnfAlgo::GetOutputInferShape(kernel_node, 0);
  dtype_ = AnfAlgo::GetInputDeviceDataType(kernel_node, 0);
  size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
  if (input_num == kSliceGradDynamicInputsNum || input_num == kStridedSliceGradDynamicInputsNum) {
    return;
  }
  // in the case that begin, end, size, stride are const value.
  std::vector<int64_t> begin_me = AnfAlgo::GetNodeAttr<std::vector<int64_t>>(kernel_node, BEGIN);
  (void)std::transform(begin_me.begin(), begin_me.end(), std::back_inserter(begin_),
                       [](const int64_t &value) { return LongToInt(value); });
  auto prim = AnfAlgo::GetCNodePrimitive(kernel_node);
  MS_EXCEPTION_IF_NULL(prim);
  auto strides = prim->GetAttr(STRIDES);
  if (strides != nullptr) {
    std::vector<int64_t> strides_me = AnfAlgo::GetNodeAttr<std::vector<int64_t>>(kernel_node, STRIDES);
    std::vector<int64_t> end_me = AnfAlgo::GetNodeAttr<std::vector<int64_t>>(kernel_node, END);
    (void)std::transform(strides_me.begin(), strides_me.end(), std::back_inserter(strides_),
                         [](const int64_t &value) { return LongToInt(value); });
    (void)std::transform(end_me.begin(), end_me.end(), std::back_inserter(end_),
                         [](const int64_t &value) { return LongToInt(value); });
    if (strides_.size() != end_.size() || strides_.size() != output_shape_.size()) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_
                        << "', the dimension of 'strides|end|output' should be equal, but got the dimension of "
                        << "'strides': " << strides_.size() << ", the dimension of 'end': " << end_.size()
                        << ", and the dimension of output: " << output_shape_.size();
    }
    FormatArgs(true);
  } else {
    std::vector<int64_t> size_me = AnfAlgo::GetNodeAttr<std::vector<int64_t>>(kernel_node, SIZE);
    (void)std::transform(size_me.begin(), size_me.end(), std::back_inserter(size_),
                         [](const int64_t &value) { return LongToInt(value); });
    if (size_.size() != output_shape_.size() || begin_.size() != output_shape_.size()) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_
                        << "', 'begin|size|input' size should be equal, but got 'begin' size: " << begin_.size()
                        << ", 'size' size: " << size_.size() << " and 'input' size: " << output_shape_.size();
    }
    FormatArgs(false);
  }

  ExpandAllMemberDims();
  CPUKernelUtils::GetElementNumEveryDim(input_shape_, &input_element_num_);
  CPUKernelUtils::GetElementNumEveryDim(output_shape_, &output_element_num_);
}

void SliceGradCPUKernel::ClearVectors() {
  begin_.clear();
  size_.clear();
  strides_.clear();
  end_.clear();
  input_element_num_.clear();
  output_element_num_.clear();
  input_shape_.clear();
  output_shape_.clear();
}

void SliceGradCPUKernel::ExpandAllMemberDims() {
  auto output_len = output_shape_.size();
  constexpr size_t expand_dims = 4;
  if (output_len < expand_dims) {
    for (size_t i = 0; i < expand_dims - output_len; ++i) {
      (void)output_shape_.insert(output_shape_.begin(), 1);
      (void)begin_.insert(begin_.begin(), 0);
      (void)strides_.insert(strides_.begin(), 1);
      (void)end_.insert(end_.begin(), 1);
    }
  }
  for (size_t i = 0; i < expand_dims; ++i) {
    if (SignOfStride(i)) {
      int ax = (end_[i] - begin_[i]) * SignOfStride(i);
      if (ax < 0) {
        ax = 0;
      }
      input_shape_.push_back(IntToSize(ax));
    }
  }
}

void SliceGradCPUKernel::InitParams(const std::vector<kernel::AddressPtr> &inputs) {
  auto cnode = cnode_ptr_.lock();
  ClearVectors();
  output_shape_ = AnfAlgo::GetOutputInferShape(cnode, 0);
  std::string kernel_name = AnfAlgo::GetCNodeName(cnode);
  auto begin_shape = AnfAlgo::GetPrevNodeOutputInferShape(cnode, 2);
  auto begin_ptr = reinterpret_cast<int32_t *>(inputs[2]->addr);
  std::vector<int32_t> begin{begin_ptr, begin_ptr + begin_shape[0]};
  (void)std::transform(begin.begin(), begin.end(), std::back_inserter(begin_),
                       [](const int32_t &value) { return value; });
  if (kernel_name == prim::kPrimStridedSliceGrad->name()) {
    auto end_shape = AnfAlgo::GetPrevNodeOutputInferShape(cnode, 3);
    auto stride_shape = AnfAlgo::GetPrevNodeOutputInferShape(cnode, 4);
    if (begin_shape.size() != 1 || end_shape.size() != 1 || stride_shape.size() != 1) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_
                        << "', the dimensions of 'begin', 'end', 'strides' should be 1, "
                           "but got the dimension of 'begin': "
                        << begin_shape.size() << ", the dimension of 'end': " << end_shape.size()
                        << ", and the dimension of 'strides': " << stride_shape.size();
    }

    auto end_ptr = reinterpret_cast<int32_t *>(inputs[3]->addr);
    auto strides_ptr = reinterpret_cast<int32_t *>(inputs[4]->addr);

    std::vector<int32_t> end{end_ptr, end_ptr + end_shape[0]};
    std::vector<int32_t> strides{strides_ptr, strides_ptr + stride_shape[0]};
    (void)std::transform(strides.begin(), strides.end(), std::back_inserter(strides_),
                         [](const int32_t &value) { return value; });
    (void)std::transform(end.begin(), end.end(), std::back_inserter(end_), [](const int32_t &value) { return value; });
    if (strides_.size() != end_.size() || strides_.size() != output_shape_.size()) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_
                        << "', the dimension of 'strides|end|output' should be equal, but got the dimension of "
                        << "'strides': " << strides_.size() << ", the dimension of 'end': " << end_.size()
                        << ", and the dimension of output: " << output_shape_.size();
    }
    FormatArgs(true);
  } else {
    auto size_shape = AnfAlgo::GetPrevNodeOutputInferShape(cnode, 3);
    if (begin_shape.size() != 1 || size_shape.size() != 1) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_
                        << "', the dimensions of 'begin', 'end' should be 1, but got the dimension of 'begin': "
                        << begin_shape.size() << ", and the dimension of 'end': " << size_shape.size();
    }
    auto size_ptr = reinterpret_cast<int32_t *>(inputs[3]->addr);
    std::vector<int32_t> size{size_ptr, size_ptr + size_shape[0]};
    (void)std::transform(size.begin(), size.end(), std::back_inserter(size_),
                         [](const int32_t &value) { return value; });
    if (size_.size() != output_shape_.size() || begin_.size() != output_shape_.size()) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_
                        << "', 'begin|size|input' size should be equal, but got 'begin' size: " << begin_.size()
                        << ", 'size' size: " << size_.size() << " and 'input' size: " << output_shape_.size();
    }
    FormatArgs(false);
  }
  ExpandAllMemberDims();
  CPUKernelUtils::GetElementNumEveryDim(input_shape_, &input_element_num_);
  CPUKernelUtils::GetElementNumEveryDim(output_shape_, &output_element_num_);
}

bool SliceGradCPUKernel::Launch(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &,
                                const std::vector<kernel::AddressPtr> &outputs) {
  if (inputs.empty()) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', input should be not empty.";
  }
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kOutputsNum, kernel_name_);

  bool ret = true;
  if (dtype_ == kNumberTypeInt32) {
    ret = LaunchKernel<int>(inputs, outputs);
  } else if (dtype_ == kNumberTypeFloat32) {
    ret = LaunchKernel<float>(inputs, outputs);
  } else if (dtype_ == kNumberTypeBool) {
    ret = LaunchKernel<bool>(inputs, outputs);
  } else if (dtype_ == kNumberTypeFloat64) {
    ret = LaunchKernel<double>(inputs, outputs);
  } else {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the dtype of input should be bool, int32, float32 or float64, but got "
                      << TypeIdToType(dtype_)->ToString();
  }
  return ret;
}

template <typename T>
bool SliceGradCPUKernel::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                      const std::vector<kernel::AddressPtr> &outputs) {
  auto *input_addr = reinterpret_cast<T *>(inputs[0]->addr);
  auto *output_addr = reinterpret_cast<T *>(outputs[0]->addr);

  // init params for not const inputs
  if (inputs.size() == kSliceGradDynamicInputsNum || inputs.size() == kStridedSliceGradDynamicInputsNum) {
    InitParams(inputs);
  }
  auto ret = memset_s(output_addr, outputs[0]->size, 0, outputs[0]->size);
  if (ret != EOK) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', output buff memset failed. Error no: " << ret;
    return false;
  }
  bool can_copy_memory[3] = {CanCopyMemoryOnAxis(0), CanCopyMemoryOnAxis(1), CanCopyMemoryOnAxis(2)};
  int stride_signs[4] = {SignOfStride(0), SignOfStride(1), SignOfStride(2), SignOfStride(3)};
  size_t out_start_offset[3] = {IntToSize(begin_[0]) * output_element_num_[0],
                                IntToSize(begin_[1]) * output_element_num_[1],
                                IntToSize(begin_[2]) * output_element_num_[2]};
  size_t out_step_size[3] = {IntToSize(strides_[0]) * output_element_num_[0],
                             IntToSize(strides_[1]) * output_element_num_[1],
                             IntToSize(strides_[2]) * output_element_num_[2]};
  size_t in_n_offset = 0;
  size_t out_n_offset = out_start_offset[0];
  size_t input_index = 0;
  for (int i = begin_[0]; stride_signs[0] * i < stride_signs[0] * end_[0];
       i += strides_[0], in_n_offset += input_element_num_[0], out_n_offset += out_step_size[0]) {
    if (can_copy_memory[0]) {
      CopyDataToOutput<T>(inputs, in_n_offset, outputs, out_n_offset, input_element_num_[0], 0);
      continue;
    }
    size_t in_c_offset = 0;
    size_t out_c_offset = out_start_offset[1];
    for (int j = begin_[1]; stride_signs[1] * j < stride_signs[1] * end_[1];
         j += strides_[1], in_c_offset += input_element_num_[1], out_c_offset += out_step_size[1]) {
      if (can_copy_memory[1]) {
        CopyDataToOutput<T>(inputs, in_n_offset + in_c_offset, outputs, out_n_offset + out_c_offset,
                            input_element_num_[1], 1);
        continue;
      }
      size_t in_h_offset = 0;
      size_t out_h_offset = out_start_offset[2];
      for (int k = begin_[2]; stride_signs[2] * k < stride_signs[2] * end_[2];
           k += strides_[2], in_h_offset += input_element_num_[2], out_h_offset += out_step_size[2]) {
        if (can_copy_memory[2]) {
          CopyDataToOutput<T>(inputs, in_n_offset + in_c_offset + in_h_offset, outputs,
                              out_n_offset + out_c_offset + out_h_offset, input_element_num_[2], 2);
          continue;
        }
        for (int m = begin_[3]; stride_signs[3] * m < stride_signs[3] * end_[3]; m += strides_[3]) {
          output_addr[out_n_offset + out_c_offset + out_h_offset + IntToSize(m)] = input_addr[input_index++];
        }
      }
    }
  }
  return true;
}

bool SliceGradCPUKernel::CanCopyMemoryOnAxis(size_t dim) const {
  for (size_t i = dim + 1; i < 4; ++i) {
    if (begin_[i] != 0 || end_[i] != SizeToInt(output_shape_[i]) || strides_[i] != 1) {
      return false;
    }
  }
  return true;
}

int SliceGradCPUKernel::SignOfStride(size_t axis) const {
  if (strides_[axis] > 0) {
    return 1;
  }
  return -1;
}

template <typename T>
void SliceGradCPUKernel::CopyDataToOutput(const std::vector<kernel::AddressPtr> &inputs, size_t in_offset,
                                          const std::vector<kernel::AddressPtr> &outputs, size_t out_offset,
                                          size_t copy_num, int id) const {
  T *input_addr = reinterpret_cast<T *>(inputs[0]->addr);
  auto in_buff_size = inputs[0]->size;
  T *output_addr = reinterpret_cast<T *>(outputs[0]->addr);
  auto out_buff_size = outputs[0]->size;

  if ((in_offset + copy_num) * sizeof(T) > in_buff_size) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << ", " << id << " input memory out of bounds.";
  }
  if ((out_offset + copy_num) * sizeof(T) > out_buff_size) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << ", " << id << " output memory out of bounds.";
  }

  auto ret = memcpy_s(output_addr + out_offset, out_buff_size - out_offset * sizeof(T), input_addr + in_offset,
                      copy_num * sizeof(T));
  if (ret != EOK) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', memcpy failed. Error no: " << ret;
  }
}

void SliceGradCPUKernel::FormatArgs(bool stride) {
  if (stride) {
    for (size_t i = 0; i < strides_.size(); ++i) {
      if (strides_[i] == 0) {
        MS_LOG(EXCEPTION) << "For '" << kernel_name_
                          << ", elements in 'stride' should not be 0, but got 0 in dimension " << i;
      }
      if (end_[i] == 0 && begin_[i] < 0) {
        end_[i] = end_[i] + SizeToInt(output_shape_[i]);
      }
      if (end_[i] < 0) {
        end_[i] = end_[i] + SizeToInt(output_shape_[i]) < 0 ? 0 : end_[i] + SizeToInt(output_shape_[i]);
      }
      if (end_[i] > SizeToInt(output_shape_[i])) {
        end_[i] = SizeToInt(output_shape_[i]);
      }
    }
  }
  for (size_t i = 0; i < begin_.size(); i++) {
    if (begin_[i] < 0) {
      auto k = begin_[i] + SizeToInt(output_shape_[i]);
      begin_[i] = k < 0 ? 0 : k;
    }
    if (begin_[i] > SizeToInt(output_shape_[i])) {
      begin_[i] = SizeToInt(output_shape_[i]);
    }
  }
  if (!stride) {
    for (size_t i = 0; i < size_.size(); ++i) {
      while (size_[i] < 0) {
        size_[i] = size_[i] + SizeToInt(output_shape_[i]);
      }
      (void)strides_.emplace_back(1);
      (void)end_.emplace_back(begin_[i] + size_[i]);
    }
  }
}
}  // namespace kernel
}  // namespace mindspore
