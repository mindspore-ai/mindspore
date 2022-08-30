/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/kernel/mirror_pad_cpu_kernel.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
// preset size of paddings
constexpr int MAX_PADDINGS = 5;
constexpr int PADDING_SIZE = 2;

// define constants for kernel indexing use
constexpr size_t kMirrorPadInputsNum = 2;
constexpr size_t kMirrorPadOutputsNum = 1;
constexpr size_t kTwo = 2;
}  // namespace

void MirrorPadCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  std::string mode = common::AnfAlgo::GetNodeAttr<std::string>(kernel_node, "mode");
  dtype_ = AnfAlgo::GetInputDeviceDataType(kernel_node, 0);
  pad_dtype_ = AnfAlgo::GetInputDeviceDataType(kernel_node, 1);
  if (mode == "REFLECT") {
    mode_ = 1;
  } else if (mode == "SYMMETRIC") {
    mode_ = 0;
  } else {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the 'mode' must be 'REFLECT' or 'SYMMETRIC', but got " << mode;
  }

  ShapeVector input_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  dims_ = int64_t(input_shape.size());

  for (int64_t i = 0; i < dims_; ++i) {
    input_shape_.push_back(input_shape[i]);
    input_elements_ *= input_shape_[i];
  }

  auto padding_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
  num_paddings_ = padding_shape[0];

  auto output_shape = common::AnfAlgo::GetOutputInferShape(kernel_node, 0);
  for (int64_t i = 0; i < dims_; ++i) {
    output_shape_.push_back(output_shape[i]);
    output_elements_ *= output_shape_[i];
  }
}

template <typename T>
void extract_paddings(const T *paddings_arg, int64_t padd_dim, int64_t *extracted_paddings) {
  for (int64_t i = 0; i < padd_dim; i++) {
    extracted_paddings[i * PADDING_SIZE] = int64_t(paddings_arg[i * PADDING_SIZE]);
    extracted_paddings[i * PADDING_SIZE + 1] = int64_t(paddings_arg[i * PADDING_SIZE + 1]);
  }
}

template <typename T>
void MirrorPadCpuKernelMod::paddings_type(const std::vector<AddressPtr> &inputs,
                                          const std::vector<AddressPtr> &outputs) const {
  if (pad_dtype_ == kNumberTypeInt32) {
    LaunchKernel<T, int32_t>(inputs, outputs);
  } else if (pad_dtype_ == kNumberTypeInt64) {
    LaunchKernel<T, int64_t>(inputs, outputs);
  } else {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dtype of 'paddings' should be int32 or int64, but got "
                      << TypeIdLabel(pad_dtype_);
  }
}

bool MirrorPadCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                   const std::vector<kernel::AddressPtr> &,
                                   const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kMirrorPadInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kMirrorPadOutputsNum, kernel_name_);
  if (dtype_ == kNumberTypeFloat16) {
    paddings_type<float16>(inputs, outputs);
  } else if (dtype_ == kNumberTypeFloat32) {
    paddings_type<float>(inputs, outputs);
  } else if (dtype_ == kNumberTypeFloat64) {
    paddings_type<double>(inputs, outputs);
  } else if (dtype_ == kNumberTypeInt8) {
    paddings_type<int8_t>(inputs, outputs);
  } else if (dtype_ == kNumberTypeInt16) {
    paddings_type<int16_t>(inputs, outputs);
  } else if (dtype_ == kNumberTypeInt32) {
    paddings_type<int32_t>(inputs, outputs);
  } else if (dtype_ == kNumberTypeInt64) {
    paddings_type<int64_t>(inputs, outputs);
  } else if (dtype_ == kNumberTypeUInt8) {
    paddings_type<uint8_t>(inputs, outputs);
  } else if (dtype_ == kNumberTypeUInt16) {
    paddings_type<uint16_t>(inputs, outputs);
  } else if (dtype_ == kNumberTypeBool) {
    paddings_type<bool>(inputs, outputs);
  } else if (dtype_ == kNumberTypeComplex64) {
    paddings_type<std::complex<float>>(inputs, outputs);
  } else if (dtype_ == kNumberTypeComplex128) {
    paddings_type<std::complex<double>>(inputs, outputs);
  } else {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the dtype of 'input_x' should be float16, float32, float64, or int8, int16, int32, int64, "
                         "uint8, uint16, bool, complex64, complex128, but got "
                      << TypeIdLabel(dtype_);
  }
  return true;
}

template <typename T1, typename T2>
void MirrorPadCpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                         const std::vector<AddressPtr> &outputs) const {
  auto inputs_addr = reinterpret_cast<T1 *>(inputs[0]->addr);
  auto *paddings_arg = reinterpret_cast<T2 *>(inputs[1]->addr);
  auto outputs_addr = reinterpret_cast<T1 *>(outputs[0]->addr);

  const int64_t padd_dim = num_paddings_;
  const int64_t mode = mode_;
  const int64_t element_size = sizeof(T1);

  int64_t paddings[MAX_PADDINGS * PADDING_SIZE];  // local and fixed size to keep in registers
  for (int i = 0; i < MAX_PADDINGS * PADDING_SIZE; i++) {
    paddings[i] = 0;
  }
  extract_paddings(paddings_arg, padd_dim, paddings);
  // quickly solve one-dimensional simple cases
  if (dims_ == 1) {
    (void)memcpy_s(outputs_addr, paddings[0] * element_size, inputs_addr + mode, paddings[0] * element_size);
    (void)memcpy_s(outputs_addr + paddings[0] + input_elements_, paddings[1] * element_size,
                   inputs_addr + input_elements_ - paddings[1] - mode, paddings[1] * element_size);
    (void)memcpy_s(outputs_addr + paddings[0], input_elements_ * element_size, inputs_addr,
                   input_elements_ * element_size);
    std::reverse(outputs_addr, outputs_addr + paddings[0]);
    std::reverse(outputs_addr + paddings[0] + input_elements_,
                 outputs_addr + paddings[0] + input_elements_ + paddings[1]);
    return;
  }
  // solve other situations
  std::vector<int64_t> output_strides_(dims_, 0);
  output_strides_[dims_ - 1] = 1;
  for (int64_t i = dims_ - 1; i > 0; --i) {
    output_strides_[i - 1] = output_strides_[i] * output_shape_[i];
  }
  std::vector<std::pair<int64_t, int64_t>> index;
  index.resize(dims_);
  index[dims_ - 1] = std::make_pair(output_strides_[dims_ - 1] * paddings[(dims_ - 1) * PADDING_SIZE],
                                    output_strides_[dims_ - 1] * paddings[(dims_ - 1) * PADDING_SIZE + 1]);
  for (int64_t i = dims_ - 1; i > 0; --i) {
    index[i - 1].first = index[i].first + output_strides_[i - 1] * paddings[(i - 1) * PADDING_SIZE];
    index[i - 1].second = index[i].second + output_strides_[i - 1] * paddings[(i - 1) * PADDING_SIZE + 1];
  }

  std::vector<int64_t> pos(dims_ - 1, 0);
  std::vector<int64_t> output_pos;
  std::vector<int64_t> tmp_pos;
  int64_t output_index = index[0].first;
  int64_t copy_size = element_size * input_shape_[dims_ - 1];
  for (int64_t i = 0; i < input_elements_; i += input_shape_[dims_ - 1]) {
    (void)memcpy_s(outputs_addr + output_index, copy_size, inputs_addr + i, copy_size);
    output_pos.push_back(output_index);
    pos[dims_ - kTwo] += 1;
    int64_t dep = dims_ - 1;
    for (int64_t j = dims_ - 2; j >= 0; --j) {
      if (j > 0 && pos[j] >= input_shape_[j]) {
        pos[j] -= input_shape_[j];
        pos[j - 1] += 1;
        dep = j;
      } else {
        break;
      }
    }
    output_index += index[dep].first + index[dep].second + input_shape_[dims_ - 1];
  }
  for (int64_t i = dims_ - 1; i >= 0; --i) {
    int64_t block_size = output_strides_[i];
    int64_t count = 0;
    copy_size = block_size * element_size;
    for (auto item : output_pos) {
      T1 *base_output_ptr1 = outputs_addr + item;
      for (int64_t cnt = 1; cnt <= paddings[i * PADDING_SIZE]; ++cnt) {
        (void)memcpy_s(base_output_ptr1 - cnt * block_size, copy_size, base_output_ptr1 + (cnt - 1 + mode) * block_size,
                       copy_size);
      }
      T1 *base_output_ptr2 = outputs_addr + item + input_shape_[i] * block_size;
      for (int64_t cnt = 1; cnt <= paddings[i * PADDING_SIZE + 1]; ++cnt) {
        (void)memcpy_s(base_output_ptr2 + (cnt - 1) * block_size, copy_size,
                       base_output_ptr2 - (cnt + mode) * block_size, copy_size);
      }
      if (i > 0 && count % input_shape_[i - 1] == 0) tmp_pos.push_back(item - paddings[i * PADDING_SIZE] * block_size);
      ++count;
    }
    output_pos.clear();
    output_pos.resize(tmp_pos.size());
    std::copy(tmp_pos.begin(), tmp_pos.end(), output_pos.begin());
    tmp_pos.clear();
  }
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, MirrorPad, MirrorPadCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
