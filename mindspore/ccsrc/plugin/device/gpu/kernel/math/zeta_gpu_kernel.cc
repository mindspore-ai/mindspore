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

#include "plugin/device/gpu/kernel/math/zeta_gpu_kernel.h"

namespace mindspore {
namespace kernel {
bool ZetaGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                            const std::vector<KernelTensorPtr> &outputs) {
  auto kernel_ptr_ = std::dynamic_pointer_cast<ops::Zeta>(base_operator);
  kernel_name_ = kernel_ptr_->name();
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' got empty inputs or outputs, which is invalid.";
    return false;
  }
  constexpr int INPUT_NUM = 2;
  if (inputs.size() != INPUT_NUM) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of inputs should be 2, but got " << inputs.size();
  }
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', the kernel type should be in [float32, float64], but got: " << kernel_attr << ".";
    return false;
  }
  kernel_func_ = func_list_[index].second;
  unit_size_ = abstract::TypeIdSize(kernel_attr.GetInputAttr(kIndex0).first);
  return true;
}

int ZetaGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                             const std::vector<KernelTensorPtr> &outputs,
                             const std::map<uint32_t, tensor::TensorPtr> &) {
  for (const auto &input : inputs) {
    // If any input shape contains -1, means input shape is dynamic, so just return do nothing.
    auto input_shape = input->GetShapeVector();
    if (!IsValidShape(input_shape)) {
      return KRET_UNKNOWN_SHAPE;
    }
  }
  ResetResource();
  std::vector<int64_t> output_shape = std::vector<int64_t>(outputs.at(kIndex0)->GetDeviceShapeAdaptively().begin(),
                                                           outputs.at(kIndex0)->GetDeviceShapeAdaptively().end());
  output_elements_ = std::accumulate(output_shape.begin(), output_shape.end(), int64_t(1), std::multiplies<int64_t>());
  if (output_elements_ == 0) {
    is_null_input_ = true;
  }
  InitSizeLists();
  return KRET_OK;
}

template <typename T>
bool ZetaGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                                    const std::vector<AddressPtr> &outputs) {
  T *x = GetDeviceAddress<T>(inputs, 0);
  T *dimension = GetDeviceAddress<T>(inputs, 1);
  T *output = GetDeviceAddress<T>(outputs, 0);
  CalZeta(output_elements_, x, dimension, output, device_id_, reinterpret_cast<cudaStream_t>(cuda_stream_));
  return true;
}

std::vector<std::pair<KernelAttr, ZetaGpuKernelMod::ZetaFunc>> ZetaGpuKernelMod::func_list_ = {
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
   &ZetaGpuKernelMod::LaunchKernel<float>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
   &ZetaGpuKernelMod::LaunchKernel<double>}};

std::vector<KernelAttr> ZetaGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, ZetaFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, Zeta, ZetaGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
