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
#include <algorithm>
#include "plugin/device/cpu/kernel/nth_element_cpu_kernel.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kNthElementInputsNum = 2;
constexpr size_t kNthElementOutputsNum = 1;
constexpr size_t kParallelDataNums = 32 * 1024;
}  // namespace

size_t get_nth_element_num(const std::vector<size_t> &shape) {
  size_t size = 1;
  for (size_t i = 0; i < shape.size(); i++) {
    size *= shape[i];
  }
  return size;
}

void NthElementCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  input_n_shape_ = AnfAlgo::GetInputDeviceShape(kernel_node, 1);
  if (input_n_shape_.size() != 0) {
    MS_LOG(EXCEPTION) << "For NthElement, the input n must be a scalar or a 0-D tensor but got a "
                      << input_n_shape_.size() << "-D tensor.";
  }
  input_shape_ = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
  if (input_shape_.size() < 1) {
    MS_LOG(EXCEPTION) << "For NthElement, input size must be equal or greater than 1, "
                      << "but got " << input_shape_.size() << ".";
  }
  output_shape_ = common::AnfAlgo::GetOutputInferShape(kernel_node, 0);
  input_elements_ = get_nth_element_num(input_shape_);
  output_elements_ = get_nth_element_num(output_shape_);
  reverse_ = common::AnfAlgo::GetNodeAttr<bool>(kernel_node, "reverse");
  dtype_ = AnfAlgo::GetInputDeviceDataType(kernel_node, 0);
}

bool NthElementCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                    const std::vector<kernel::AddressPtr> &,
                                    const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kNthElementInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kNthElementOutputsNum, kernel_name_);
  switch (dtype_) {
    case kNumberTypeFloat32:
      LaunchKernel<float>(inputs, outputs);
      break;
    case kNumberTypeFloat16:
      LaunchKernel<float16>(inputs, outputs);
      break;
    case kNumberTypeInt8:
      LaunchKernel<int8_t>(inputs, outputs);
      break;
    case kNumberTypeUInt16:
      LaunchKernel<uint16_t>(inputs, outputs);
      break;
    case kNumberTypeInt16:
      LaunchKernel<int16_t>(inputs, outputs);
      break;
    case kNumberTypeUInt8:
      LaunchKernel<uint8_t>(inputs, outputs);
      break;
    case kNumberTypeInt32:
      LaunchKernel<int32_t>(inputs, outputs);
      break;
    case kNumberTypeInt64:
      LaunchKernel<int64_t>(inputs, outputs);
      break;
    case kNumberTypeFloat64:
      LaunchKernel<double>(inputs, outputs);
      break;
    default:
      MS_EXCEPTION(TypeError) << "For NthElement, input data type must be float32, float16, int8, "
                              << "uint16, int 16, uint8, int32, int64 or float64 but got data type "
                              << TypeIdLabel(dtype_) << ".";
  }
  return true;
}

template <typename T>
void NthElementCpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                          const std::vector<AddressPtr> &outputs) {
  auto *n_data = reinterpret_cast<int32_t *>(inputs[1]->addr);
  input_n_val_ = *n_data;
  if (input_n_val_ < 0 || input_n_val_ >= static_cast<int>(input_shape_.back())) {
    MS_LOG(EXCEPTION) << "For NthElement, the value of input n must be in [0, input.shape[-1]), "
                      << "but got " << input_n_val_ << ".";
  }
  auto last_dim = input_shape_.back();
  if (reverse_) {
    input_n_val_ = static_cast<int32_t>(last_dim - input_n_val_ - 1);
  }
  size_t index = IntToSize(input_n_val_);
  T *input_addrs = reinterpret_cast<T *>(inputs[0]->addr);
  T *output_addrs = reinterpret_cast<T *>(outputs[0]->addr);
  if (input_elements_ <= kParallelDataNums) {
    std::vector<T> buf(last_dim);
    for (size_t i = 0; i < output_elements_; i++) {
      const T *input_start = input_addrs + i * last_dim;
      const T *input_end = input_start + last_dim;
      std::copy(input_start, input_end, buf.begin());
      std::nth_element(buf.begin(), buf.begin() + input_n_val_, buf.end());
      output_addrs[i] = buf[index];
    }
  } else {
    auto shard_nth_element = [this, &last_dim, &input_addrs, &output_addrs, &index](size_t start, size_t end) {
      std::vector<T> buf(last_dim);
      for (size_t i = start; i < end; ++i) {
        const T *input_start = input_addrs + i * last_dim;
        const T *input_end = input_start + last_dim;
        std::copy(input_start, input_end, buf.begin());
        std::nth_element(buf.begin(), buf.begin() + input_n_val_, buf.end());
        output_addrs[i] = buf[index];
      }
    };
    ParallelLaunchAutoSearch(shard_nth_element, output_elements_, this, &parallel_search_info_);
  }
}

std::vector<KernelAttr> NthElementCpuKernelMod::GetOpSupport() {
  static std::vector<KernelAttr> support_list = {
    KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat32),
    KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat16),
    KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt8),
    KernelAttr().AddInputAttr(kNumberTypeUInt16).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt16),
    KernelAttr().AddInputAttr(kNumberTypeInt16).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt16),
    KernelAttr().AddInputAttr(kNumberTypeUInt8).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt8),
    KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
    KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt64),
    KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat64)};
  return support_list;
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, NthElement, NthElementCpuKernelMod);
// }
}  // namespace kernel
}  // namespace mindspore
