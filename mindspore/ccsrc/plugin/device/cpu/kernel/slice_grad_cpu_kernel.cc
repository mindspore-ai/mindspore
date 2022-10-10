/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/kernel/slice_grad_cpu_kernel.h"
#include <algorithm>
#include <string>
#include <map>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "ir/primitive.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kSliceGradDynamicInputsNum = 4;
constexpr size_t kStridedSliceGradDynamicInputsNum = 5;
constexpr size_t kOutputsNum = 1;
constexpr size_t kSliceGradMaxInputShapeSize = 8;
}  // namespace

void SliceGradCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  cnode_ptr_ = kernel_node;
  constexpr size_t kInputNum2 = 2;
  ClearVectors();
  auto input_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  if (input_shape.size() > kSliceGradMaxInputShapeSize) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimension of input tensor must be 8D or lower, but got "
                      << input_shape.size() << "D.";
  }
  output_shape_ = common::AnfAlgo::GetOutputInferShape(kernel_node, 0);
  dtype_ = AnfAlgo::GetInputDeviceDataType(kernel_node, 0);
  size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
  if (input_num == kSliceGradDynamicInputsNum || input_num == kStridedSliceGradDynamicInputsNum) {
    strides_dtype_ = AnfAlgo::GetInputDeviceDataType(kernel_node, kInputNum2);
    return;
  }
  // in the case that begin, end, size, stride are const value.
  std::vector<int64_t> begin_me = common::AnfAlgo::GetNodeAttr<std::vector<int64_t>>(kernel_node, BEGIN);
  (void)std::transform(begin_me.begin(), begin_me.end(), std::back_inserter(begin_),
                       [](const int64_t &value) { return LongToInt(value); });
  auto prim = common::AnfAlgo::GetCNodePrimitive(kernel_node);
  MS_EXCEPTION_IF_NULL(prim);
  auto strides = prim->GetAttr(STRIDES);
  if (strides != nullptr) {  // StridedSliceGrad
    std::vector<int64_t> strides_me = common::AnfAlgo::GetNodeAttr<std::vector<int64_t>>(kernel_node, STRIDES);
    std::vector<int64_t> end_me = common::AnfAlgo::GetNodeAttr<std::vector<int64_t>>(kernel_node, END);
    (void)std::transform(strides_me.begin(), strides_me.end(), std::back_inserter(strides_),
                         [](const int64_t &value) { return LongToInt(value); });
    (void)std::transform(end_me.begin(), end_me.end(), std::back_inserter(end_),
                         [](const int64_t &value) { return LongToInt(value); });
    if (strides_.size() != end_.size() || strides_.size() != output_shape_.size()) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_
                        << "', the dimension of 'strides|end|output' must be equal, but got the dimension of "
                        << "'strides': " << strides_.size() << ", the dimension of 'end': " << end_.size()
                        << ", and the dimension of output: " << output_shape_.size();
    }
    FormatArgs(true);
  } else {  // SliceGrad
    std::vector<int64_t> size_me = common::AnfAlgo::GetNodeAttr<std::vector<int64_t>>(kernel_node, SIZE);
    (void)std::transform(size_me.begin(), size_me.end(), std::back_inserter(size_),
                         [](const int64_t &value) { return LongToInt(value); });
    if (size_.size() != output_shape_.size() || begin_.size() != output_shape_.size()) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_
                        << "', 'begin|size|input' size must be equal, but got 'begin' size: " << begin_.size()
                        << ", 'size' size: " << size_.size() << " and 'input' size: " << output_shape_.size();
    }
    FormatArgs(false);
  }
  ExpandAllMemberDims(kSliceGradMaxInputShapeSize);

  CPUKernelUtils::GetElementNumEveryDim(input_shape_, &input_element_num_);
  CPUKernelUtils::GetElementNumEveryDim(output_shape_, &output_element_num_);
}

void SliceGradCpuKernelMod::ClearVectors() {
  begin_.clear();
  size_.clear();
  strides_.clear();
  end_.clear();
  input_element_num_.clear();
  output_element_num_.clear();
  input_shape_.clear();
  output_shape_.clear();
}

void SliceGradCpuKernelMod::ExpandAllMemberDims(size_t expand_dims) {
  auto output_len = output_shape_.size();
  auto strides_len = strides_.size();
  // expand begin, end, strides dims equal to output dims
  if (strides_len < output_len) {
    for (size_t i = strides_len; i < output_len; ++i) {
      begin_.push_back(0);
      end_.push_back(output_shape_[i]);
      strides_.push_back(1);
    }
  }
  // expand output, begin, end, strides dims equal to max dims 8
  if (output_len < expand_dims) {
    for (size_t i = 0; i < expand_dims - output_len; ++i) {
      (void)output_shape_.insert(output_shape_.begin(), 1);
      (void)begin_.insert(begin_.begin(), 0);
      (void)strides_.insert(strides_.begin(), 1);
      (void)end_.insert(end_.begin(), 1);
    }
  }

  for (size_t i = 0; i < expand_dims; ++i) {
    int ax = (end_[i] - begin_[i]) * SignOfStride(i);
    if (ax < 0) {
      ax = 0;
    }
    input_shape_.push_back(ax);
  }
}

template <typename T>
void SliceGradCpuKernelMod::InitParams(const std::vector<kernel::AddressPtr> &inputs) {
  auto cnode = cnode_ptr_.lock();
  ClearVectors();
  output_shape_ = common::AnfAlgo::GetOutputInferShape(cnode, 0);
  std::string kernel_name = common::AnfAlgo::GetCNodeName(cnode);
  auto begin_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(cnode, 2);
  auto begin_ptr = static_cast<T *>(inputs[2]->addr);
  std::vector<T> begin{begin_ptr, begin_ptr + begin_shape[0]};
  (void)std::transform(begin.begin(), begin.end(), std::back_inserter(begin_),
                       [](const T &value) { return static_cast<int>(value); });
  if (kernel_name == prim::kPrimStridedSliceGrad->name()) {  // StridedSliceGrad
    auto end_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(cnode, 3);
    auto stride_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(cnode, 4);
    if (begin_shape.size() != 1 || end_shape.size() != 1 || stride_shape.size() != 1) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_
                        << "', the dimensions of 'begin', 'end', 'strides' must be 1, "
                           "but got the dimension of 'begin': "
                        << begin_shape.size() << ", the dimension of 'end': " << end_shape.size()
                        << ", and the dimension of 'strides': " << stride_shape.size();
    }

    auto end_ptr = static_cast<T *>(inputs[3]->addr);
    auto strides_ptr = static_cast<T *>(inputs[4]->addr);

    std::vector<T> end{end_ptr, end_ptr + end_shape[0]};
    std::vector<T> strides{strides_ptr, strides_ptr + stride_shape[0]};
    (void)std::transform(strides.begin(), strides.end(), std::back_inserter(strides_),
                         [](const T &value) { return static_cast<int>(value); });
    (void)std::transform(end.begin(), end.end(), std::back_inserter(end_), [](const T &value) { return value; });
    if (strides_.size() != end_.size()) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_
                        << "', the dimension of 'strides|end|output' must be equal, but got the dimension of "
                        << "'strides': " << strides_.size() << ", the dimension of 'end': " << end_.size()
                        << ", and the dimension of output: " << output_shape_.size();
    }
    FormatArgs(true);
  } else {  // SliceGrad
    auto size_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(cnode, 3);
    if (begin_shape.size() != 1 || size_shape.size() != 1) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_
                        << "', the dimensions of 'begin', 'end' must be 1, but got the dimension of 'begin': "
                        << begin_shape.size() << ", and the dimension of 'end': " << size_shape.size();
    }
    auto size_ptr = static_cast<T *>(inputs[3]->addr);
    std::vector<T> size{size_ptr, size_ptr + size_shape[0]};
    (void)std::transform(size.begin(), size.end(), std::back_inserter(size_),
                         [](const T &value) { return static_cast<int>(value); });
    if (size_.size() != output_shape_.size() || begin_.size() != output_shape_.size()) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_
                        << "', 'begin|size|input' size must be equal, but got 'begin' size: " << begin_.size()
                        << ", 'size' size: " << size_.size() << " and 'input' size: " << output_shape_.size();
    }
    FormatArgs(false);
  }
  ExpandAllMemberDims(kSliceGradMaxInputShapeSize);

  CPUKernelUtils::GetElementNumEveryDim(input_shape_, &input_element_num_);
  CPUKernelUtils::GetElementNumEveryDim(output_shape_, &output_element_num_);
}

bool SliceGradCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                   const std::vector<kernel::AddressPtr> &,
                                   const std::vector<kernel::AddressPtr> &outputs) {
  if (inputs.empty()) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', input can not be empty.";
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
                      << "', the dtype of input must be bool, int32, float32 or float64, but got "
                      << TypeIdToType(dtype_)->ToString();
  }
  return ret;
}

template <typename T>
bool SliceGradCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                         const std::vector<kernel::AddressPtr> &outputs) {
  auto *input_addr = reinterpret_cast<T *>(inputs[0]->addr);
  auto *output_addr = reinterpret_cast<T *>(outputs[0]->addr);

  // init params for not const inputs
  if (inputs.size() == kSliceGradDynamicInputsNum || inputs.size() == kStridedSliceGradDynamicInputsNum) {
    if (strides_dtype_ == kNumberTypeInt32) {
      InitParams<int32_t>(inputs);
    } else {
      InitParams<int64_t>(inputs);
    }
  }
  auto ret = memset_s(output_addr, outputs[0]->size, 0, outputs[0]->size);
  if (ret != EOK) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', output buff memset failed. Error no: " << ret;
    return false;
  }
  return SliceGrad8D<T>(inputs, outputs, input_addr, output_addr);
}

template <typename T>
bool SliceGradCpuKernelMod::SliceGrad8D(const std::vector<kernel::AddressPtr> &inputs,
                                        const std::vector<kernel::AddressPtr> &outputs, T *input_addr, T *output_addr) {
  bool can_copy_memory[7] = {CanCopyMemoryOnAxis(0), CanCopyMemoryOnAxis(1), CanCopyMemoryOnAxis(2),
                             CanCopyMemoryOnAxis(3), CanCopyMemoryOnAxis(4), CanCopyMemoryOnAxis(5),
                             CanCopyMemoryOnAxis(6)};
  int stride_signs[8] = {SignOfStride(0), SignOfStride(1), SignOfStride(2), SignOfStride(3),
                         SignOfStride(4), SignOfStride(5), SignOfStride(6), SignOfStride(7)};
  size_t out_start_offset[7] = {
    IntToSize(begin_[0]) * output_element_num_[0], IntToSize(begin_[1]) * output_element_num_[1],
    IntToSize(begin_[2]) * output_element_num_[2], IntToSize(begin_[3]) * output_element_num_[3],
    IntToSize(begin_[4]) * output_element_num_[4], IntToSize(begin_[5]) * output_element_num_[5],
    IntToSize(begin_[6]) * output_element_num_[6]};
  size_t out_step_size[7] = {
    IntToSize(strides_[0]) * output_element_num_[0], IntToSize(strides_[1]) * output_element_num_[1],
    IntToSize(strides_[2]) * output_element_num_[2], IntToSize(strides_[3]) * output_element_num_[3],
    IntToSize(strides_[4]) * output_element_num_[4], IntToSize(strides_[5]) * output_element_num_[5],
    IntToSize(strides_[6]) * output_element_num_[6]};
  size_t in_n_offset = 0, input_index = 0;
  size_t out_n_offset = out_start_offset[0];
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
        size_t in_w_offset = 0;
        size_t out_w_offset = out_start_offset[3];
        for (int l = begin_[3]; stride_signs[3] * l < stride_signs[3] * end_[3];
             l += strides_[3], in_w_offset += input_element_num_[3], out_w_offset += out_step_size[3]) {
          if (can_copy_memory[3]) {
            CopyDataToOutput<T>(inputs, in_n_offset + in_c_offset + in_h_offset + in_w_offset, outputs,
                                out_n_offset + out_c_offset + out_h_offset + out_w_offset, input_element_num_[3], 3);
            continue;
          }
          size_t in_4_offset = 0;
          size_t out_4_offset = out_start_offset[4];
          for (int m = begin_[4]; stride_signs[4] * m < stride_signs[4] * end_[4];
               m += strides_[4], in_4_offset += input_element_num_[4], out_4_offset += out_step_size[4]) {
            if (can_copy_memory[4]) {
              CopyDataToOutput<T>(inputs, in_n_offset + in_c_offset + in_h_offset + in_w_offset + in_4_offset, outputs,
                                  out_n_offset + out_c_offset + out_h_offset + out_w_offset + out_4_offset,
                                  input_element_num_[4], 4);
              continue;
            }
            size_t in_5_offset = 0;
            size_t out_5_offset = out_start_offset[5];
            for (int n = begin_[5]; stride_signs[5] * n < stride_signs[5] * end_[5];
                 n += strides_[5], in_5_offset += input_element_num_[5], out_5_offset += out_step_size[5]) {
              if (can_copy_memory[5]) {
                CopyDataToOutput<T>(
                  inputs, in_n_offset + in_c_offset + in_h_offset + in_w_offset + in_4_offset + in_5_offset, outputs,
                  out_n_offset + out_c_offset + out_h_offset + out_w_offset + out_4_offset + out_5_offset,
                  input_element_num_[5], 5);
                continue;
              }
              size_t in_6_offset = 0;
              size_t out_6_offset = out_start_offset[6];
              for (int o = begin_[6]; stride_signs[6] * o < stride_signs[6] * end_[6];
                   o += strides_[6], in_6_offset += input_element_num_[6], out_6_offset += out_step_size[6]) {
                if (can_copy_memory[6]) {
                  CopyDataToOutput<T>(
                    inputs,
                    in_n_offset + in_c_offset + in_h_offset + in_w_offset + in_4_offset + in_5_offset + in_6_offset,
                    outputs,
                    out_n_offset + out_c_offset + out_h_offset + out_w_offset + out_4_offset + out_5_offset +
                      out_6_offset,
                    input_element_num_[6], 6);
                  continue;
                }
                for (int p = begin_[7]; stride_signs[7] * p < stride_signs[7] * end_[7]; p += strides_[7]) {
                  output_addr[out_n_offset + out_c_offset + out_h_offset + out_w_offset + out_4_offset + out_5_offset +
                              out_6_offset + IntToSize(p)] = input_addr[input_index++];
                }
              }
            }
          }
        }
      }
    }
  }
  return true;
}

bool SliceGradCpuKernelMod::CanCopyMemoryOnAxis(size_t dim, size_t max_dim) const {
  for (size_t i = dim + 1; i < max_dim; ++i) {
    if (begin_[i] != 0 || end_[i] != SizeToInt(output_shape_[i]) || strides_[i] != 1) {
      return false;
    }
  }
  return true;
}

int SliceGradCpuKernelMod::SignOfStride(size_t axis) const {
  if (strides_[axis] > 0) {
    return 1;
  }
  return -1;
}

template <typename T>
void SliceGradCpuKernelMod::CopyDataToOutput(const std::vector<kernel::AddressPtr> &inputs, size_t in_offset,
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

void SliceGradCpuKernelMod::FormatArgs(bool stride) {
  if (stride) {
    for (size_t i = 0; i < strides_.size(); ++i) {
      if (strides_[i] == 0) {
        MS_LOG(EXCEPTION) << "For '" << kernel_name_ << ", elements in 'stride' can not be 0, but got 0 in dimension "
                          << i;
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

std::vector<KernelAttr> SliceGradCpuKernelMod::GetOpSupport() {
  static std::map<std::string, std::vector<KernelAttr>> support_list_map = {
    {kSliceGrad,
     {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
      KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
      KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
      KernelAttr().AddInputAttr(kNumberTypeBool).AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeBool),
      KernelAttr()
        .AddInputAttr(kNumberTypeFloat32)
        .AddInputAttr(kNumberTypeFloat32)
        .AddInputAttr(kNumberTypeInt32)
        .AddInputAttr(kNumberTypeInt32)
        .AddOutputAttr(kNumberTypeFloat32),
      KernelAttr()
        .AddInputAttr(kNumberTypeFloat64)
        .AddInputAttr(kNumberTypeFloat64)
        .AddInputAttr(kNumberTypeInt32)
        .AddInputAttr(kNumberTypeInt32)
        .AddOutputAttr(kNumberTypeFloat64),
      KernelAttr()
        .AddInputAttr(kNumberTypeInt32)
        .AddInputAttr(kNumberTypeInt32)
        .AddInputAttr(kNumberTypeInt32)
        .AddInputAttr(kNumberTypeInt32)
        .AddOutputAttr(kNumberTypeInt32),
      KernelAttr()
        .AddInputAttr(kNumberTypeBool)
        .AddInputAttr(kNumberTypeBool)
        .AddInputAttr(kNumberTypeInt32)
        .AddInputAttr(kNumberTypeInt32)
        .AddOutputAttr(kNumberTypeBool)}},
    {kStridedSliceGrad,
     {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
      KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
      KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
      KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeBool),
      KernelAttr()
        .AddInputAttr(kNumberTypeFloat32)
        .AddInputAttr(kNumberTypeInt32)
        .AddInputAttr(kNumberTypeInt32)
        .AddInputAttr(kNumberTypeInt32)
        .AddInputAttr(kNumberTypeInt32)
        .AddOutputAttr(kNumberTypeFloat32),
      KernelAttr()
        .AddInputAttr(kNumberTypeInt32)
        .AddInputAttr(kNumberTypeInt32)
        .AddInputAttr(kNumberTypeInt32)
        .AddInputAttr(kNumberTypeInt32)
        .AddInputAttr(kNumberTypeInt32)
        .AddOutputAttr(kNumberTypeInt32),
      KernelAttr()
        .AddInputAttr(kNumberTypeFloat64)
        .AddInputAttr(kNumberTypeInt32)
        .AddInputAttr(kNumberTypeInt32)
        .AddInputAttr(kNumberTypeInt32)
        .AddInputAttr(kNumberTypeInt32)
        .AddOutputAttr(kNumberTypeFloat64),
      KernelAttr()
        .AddInputAttr(kNumberTypeBool)
        .AddInputAttr(kNumberTypeInt32)
        .AddInputAttr(kNumberTypeInt32)
        .AddInputAttr(kNumberTypeInt32)
        .AddInputAttr(kNumberTypeInt32)
        .AddOutputAttr(kNumberTypeBool),
      KernelAttr()
        .AddInputAttr(kNumberTypeFloat32)
        .AddInputAttr(kNumberTypeInt64)
        .AddInputAttr(kNumberTypeInt64)
        .AddInputAttr(kNumberTypeInt64)
        .AddInputAttr(kNumberTypeInt64)
        .AddOutputAttr(kNumberTypeFloat32),
      KernelAttr()
        .AddInputAttr(kNumberTypeInt32)
        .AddInputAttr(kNumberTypeInt64)
        .AddInputAttr(kNumberTypeInt64)
        .AddInputAttr(kNumberTypeInt64)
        .AddInputAttr(kNumberTypeInt64)
        .AddOutputAttr(kNumberTypeInt32),
      KernelAttr()
        .AddInputAttr(kNumberTypeFloat64)
        .AddInputAttr(kNumberTypeInt64)
        .AddInputAttr(kNumberTypeInt64)
        .AddInputAttr(kNumberTypeInt64)
        .AddInputAttr(kNumberTypeInt64)
        .AddOutputAttr(kNumberTypeFloat64),
      KernelAttr()
        .AddInputAttr(kNumberTypeBool)
        .AddInputAttr(kNumberTypeInt64)
        .AddInputAttr(kNumberTypeInt64)
        .AddInputAttr(kNumberTypeInt64)
        .AddInputAttr(kNumberTypeInt64)
        .AddOutputAttr(kNumberTypeBool)}}};

  auto iter = support_list_map.find(kernel_type_);
  if (iter == support_list_map.end()) {
    MS_LOG(EXCEPTION) << "Does not support " << kernel_type_ << "!";
  }

  return iter->second;
}

MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, SliceGrad,
                                 []() { return std::make_shared<SliceGradCpuKernelMod>(kSliceGrad); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, StridedSliceGrad,
                                 []() { return std::make_shared<SliceGradCpuKernelMod>(kStridedSliceGrad); });
}  // namespace kernel
}  // namespace mindspore
