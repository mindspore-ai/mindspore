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
#include "mindspore/core/ops/nth_element.h"
namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kNthElementInputsNum = 2;
constexpr size_t kNthElementOutputsNum = 1;
constexpr size_t kParallelDataNums = 32 * 1024;
}  // namespace

bool NthElementCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                  const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kNthElementInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kNthElementOutputsNum, kernel_name_);
  auto kernel_ptr = std::dynamic_pointer_cast<ops::NthElement>(base_operator);
  MS_EXCEPTION_IF_NULL(kernel_ptr);
  reverse_ = kernel_ptr->get_reverse();
  return MatchKernelFunc(base_operator, inputs, outputs);
}

int NthElementCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                   const std::vector<KernelTensorPtr> &outputs,
                                   const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  if (auto ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost); ret != KRET_OK) {
    return ret;
  }
  input_shape_ = inputs[kIndex0]->GetDeviceShapeAdaptively();
  output_shape_ = outputs[kIndex0]->GetShapeVector();
  input_elements_ = SizeOf(input_shape_);
  output_elements_ = SizeOf(output_shape_);
  return KRET_OK;
}

template <typename T>
bool NthElementCpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                                          const std::vector<AddressPtr> &outputs) {
  auto *n_data = static_cast<int32_t *>(inputs[1]->addr);
  input_n_val_ = *n_data;
  if (input_n_val_ < 0 || input_n_val_ >= static_cast<int>(input_shape_.back())) {
    MS_LOG(ERROR) << "For NthElement, the value of input n must be in [0, input.shape[-1]), "
                  << "but got " << input_n_val_ << ".";
    return false;
  }
  auto last_dim = input_shape_.back();
  if (reverse_) {
    input_n_val_ = static_cast<int32_t>(last_dim - input_n_val_ - 1);
  }
  size_t index = IntToSize(input_n_val_);
  T *input_addrs = static_cast<T *>(inputs[0]->addr);
  T *output_addrs = static_cast<T *>(outputs[0]->addr);
  if (input_elements_ <= kParallelDataNums) {
    std::vector<T> buf(last_dim);
    for (size_t i = 0; i < output_elements_; i++) {
      const T *input_start = input_addrs + SizeToLong(i) * last_dim;
      const T *input_end = input_start + last_dim;
      (void)std::copy(input_start, input_end, buf.begin());
      std::nth_element(buf.begin(), buf.begin() + input_n_val_, buf.end());
      output_addrs[i] = buf[index];
    }
  } else {
    auto shard_nth_element = [this, &last_dim, &input_addrs, &output_addrs, &index](size_t start, size_t end) {
      std::vector<T> buf(last_dim);
      for (size_t i = start; i < end; ++i) {
        const T *input_start = input_addrs + SizeToLong(i) * last_dim;
        const T *input_end = input_start + last_dim;
        (void)std::copy(input_start, input_end, buf.begin());
        std::nth_element(buf.begin(), buf.begin() + input_n_val_, buf.end());
        output_addrs[i] = buf[index];
      }
    };
    ParallelLaunchAutoSearch(shard_nth_element, output_elements_, this, &parallel_search_info_);
  }
  return true;
}

const std::vector<std::pair<KernelAttr, NthElementCpuKernelMod::KernelRunFunc>> &NthElementCpuKernelMod::GetFuncList()
  const {
  static const std::vector<std::pair<KernelAttr, NthElementCpuKernelMod::KernelRunFunc>> func_list = {
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat32),
     &NthElementCpuKernelMod::LaunchKernel<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat16),
     &NthElementCpuKernelMod::LaunchKernel<float16>},
    {KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt8),
     &NthElementCpuKernelMod::LaunchKernel<int8_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt16).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt16),
     &NthElementCpuKernelMod::LaunchKernel<uint16_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt16).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt16),
     &NthElementCpuKernelMod::LaunchKernel<int16_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt8),
     &NthElementCpuKernelMod::LaunchKernel<uint8_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
     &NthElementCpuKernelMod::LaunchKernel<int32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt64),
     &NthElementCpuKernelMod::LaunchKernel<int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat64),
     &NthElementCpuKernelMod::LaunchKernel<double>},
  };
  return func_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, NthElement, NthElementCpuKernelMod);
// }
}  // namespace kernel
}  // namespace mindspore
