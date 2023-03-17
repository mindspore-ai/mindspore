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

#include "plugin/device/gpu/kernel/nn/kl_div_loss_grad_kernel.h"
#include <map>
#include <utility>
#include "mindspore/core/ops/grad/kl_div_loss_grad.h"

namespace mindspore {
namespace kernel {
bool KLDivLossGradGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                     const std::vector<KernelTensorPtr> &outputs) {
  constexpr size_t input_num = 3;
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
  auto input_data_type = inputs.at(kIndex0)->GetDtype();
  type_id_size_ = abstract::TypeIdSize(input_data_type);
  return true;
}

int KLDivLossGradGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                      const std::vector<KernelTensorPtr> &outputs,
                                      const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  if (auto ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost); ret != KRET_OK) {
    return ret;
  }
  auto input_shape = inputs[kIndex1]->GetShapeVector();
  input_size_ = 1;
  input_size_ *= SizeOf(input_shape);
  auto kl_div_loss_grad_ptr = std::dynamic_pointer_cast<ops::KLDivLossGrad>(base_operator);
  string reduction = kl_div_loss_grad_ptr->get_reduction();
  reduction_ = kReductionModeMap[reduction];
  if (reduction_ == ReductionMode::kNone) {
    input_size_list_[0] = input_size_ * type_id_size_;
  } else {
    input_size_list_[0] = type_id_size_;
  }
  return KRET_OK;
}

template <typename T>
bool KLDivLossGradGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                                             const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  T *dloss = GetDeviceAddress<T>(inputs, kIndex0);
  T *input_x = GetDeviceAddress<T>(inputs, kIndex1);
  T *input_y = GetDeviceAddress<T>(inputs, kIndex2);
  T *dx = GetDeviceAddress<T>(outputs, kIndex0);
  KLDivLossGrad(input_size_, reduction_, input_x, input_y, dloss, dx, reinterpret_cast<cudaStream_t>(stream_ptr));
  return true;
}

std::vector<std::pair<KernelAttr, KLDivLossGradGpuKernelMod::KLDivLossLaunchFunc>>
  KLDivLossGradGpuKernelMod::func_list_ = {
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeFloat64)
       .AddOutputAttr(kNumberTypeFloat64),
     &KLDivLossGradGpuKernelMod::LaunchKernel<double>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeFloat32),
     &KLDivLossGradGpuKernelMod::LaunchKernel<float>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddOutputAttr(kNumberTypeFloat16),
     &KLDivLossGradGpuKernelMod::LaunchKernel<half>},
};

std::vector<KernelAttr> KLDivLossGradGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(
    func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
    [](const std::pair<KernelAttr, KLDivLossGradGpuKernelMod::KLDivLossLaunchFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, KLDivLossGrad, KLDivLossGradGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
