/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/gpu/kernel/math/equalcount_gpu_kernel.h"

namespace mindspore {
namespace kernel {
bool EqualCountGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                  const std::vector<KernelTensorPtr> &outputs) {
  constexpr size_t input_num = 2;
  constexpr size_t output_num = 1;
  kernel_name_ = base_operator->GetPrim()->name();
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), input_num, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), output_num, kernel_name_);

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;

  return true;
}

int EqualCountGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                   const std::vector<KernelTensorPtr> &outputs,
                                   const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  if (auto ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost); ret != KRET_OK) {
    return ret;
  }

  auto input_shape = inputs[0]->GetShapeVector();
  is_null_input_ = CHECK_SHAPE_NULL(input_shape, kernel_name_, "input");
  input_shape_size_ = SizeOf(input_shape);
  return KRET_OK;
}

template <typename T>
bool EqualCountGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                          const std::vector<AddressPtr> &workspace,
                                          const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  if (is_null_input_) {
    return true;
  }
  VARIABLE_NOT_USED(workspace);
  T *input1 = GetDeviceAddress<T>(inputs, 0);
  T *input2 = GetDeviceAddress<T>(inputs, 1);
  T *output = GetDeviceAddress<T>(outputs, 0);
  int size = SizeToInt(input_shape_size_);
  CalEqualCount(size, input1, input2, output, reinterpret_cast<cudaStream_t>(stream_ptr));
  return true;
}

std::vector<std::pair<KernelAttr, EqualCountGpuKernelMod::EqualCountLaunchFunc>> EqualCountGpuKernelMod::func_list_ = {
  {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
   &EqualCountGpuKernelMod::LaunchKernel<half>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
   &EqualCountGpuKernelMod::LaunchKernel<float>},
  {KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
   &EqualCountGpuKernelMod::LaunchKernel<int>},
};

std::vector<KernelAttr> EqualCountGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(
    func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
    [](const std::pair<KernelAttr, EqualCountGpuKernelMod::EqualCountLaunchFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, EqualCount, EqualCountGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
