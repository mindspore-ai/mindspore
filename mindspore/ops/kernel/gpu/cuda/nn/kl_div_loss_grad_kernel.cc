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

#include "kernel/gpu/cuda/nn/kl_div_loss_grad_kernel.h"
#include <map>
#include <utility>
#include "mindspore/ops/infer/grad/kl_div_loss_grad.h"

namespace mindspore {
namespace kernel {
bool KLDivLossGradGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                     const std::vector<KernelTensor *> &outputs) {
  constexpr size_t input_num = 3;
  constexpr size_t output_num = 1;

  CHECK_KERNEL_INPUTS_NUM(inputs.size(), input_num, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), output_num, kernel_name_);
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  auto input_data_type = inputs[kIndex0]->dtype_id();
  type_id_size_ = abstract::TypeIdSize(input_data_type);
  return true;
}

int KLDivLossGradGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                      const std::vector<KernelTensor *> &outputs) {
  if (auto ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  auto input_shape = inputs[kIndex1]->GetShapeVector();
  input_size_ = 1;
  input_size_ *= SizeOf(input_shape);
  string reduction = GetValue<std::string>(primitive_->GetAttr("reduction"));
  reduction_ = kReductionModeMap[reduction];

  return KRET_OK;
}

template <typename T>
bool KLDivLossGradGpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                             const std::vector<KernelTensor *> &,
                                             const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  T *dloss = GetDeviceAddress<T>(inputs, kIndex0);
  T *input_x = GetDeviceAddress<T>(inputs, kIndex1);
  T *input_y = GetDeviceAddress<T>(inputs, kIndex2);
  T *dx = GetDeviceAddress<T>(outputs, kIndex0);
  auto status =
    KLDivLossGrad(input_size_, reduction_, input_x, input_y, dloss, dx, reinterpret_cast<cudaStream_t>(stream_ptr));
  CHECK_CUDA_STATUS(status, kernel_name_);
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
