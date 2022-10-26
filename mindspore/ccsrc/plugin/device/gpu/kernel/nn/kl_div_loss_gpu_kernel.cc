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

#include "plugin/device/gpu/kernel/nn/kl_div_loss_gpu_kernel.h"
#include <string>
#include <algorithm>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/loss_with_reduction_impl.cuh"
#include "kernel/common_utils.h"
#include "ops/kl_div_loss.h"

namespace mindspore {
namespace kernel {
constexpr size_t kKLDivLossInputsNum = 2;

template <typename T>
bool KLDivLossGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                         const std::vector<AddressPtr> &workspace,
                                         const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  T *input_x = GetDeviceAddress<T>(inputs, 0);
  T *input_y = GetDeviceAddress<T>(inputs, 1);
  T *loss = GetDeviceAddress<T>(outputs, 0);
  T *tmp_loss = GetDeviceAddress<T>(workspace, 0);
  KLDivLoss(input_size_, reduction_, input_x, input_y, loss, tmp_loss, reinterpret_cast<cudaStream_t>(stream_ptr));
  return true;
}

bool KLDivLossGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                 const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();
  auto kernel_ptr = std::dynamic_pointer_cast<ops::KLDivLoss>(base_operator);
  string reduction = kernel_ptr->get_reduction();
  reduction_ = kReductionModeMap[reduction];
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  type_size_ = abstract::TypeIdSize(kernel_attr.GetInputAttr(0).first);
  return true;
}

int KLDivLossGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                  const std::vector<KernelTensorPtr> &outputs,
                                  const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kKLDivLossInputsNum, kernel_name_);
  int ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost);
  if (ret != KRET_OK) {
    return ret;
  }

  auto input_shape = inputs[0]->GetShapeVector();
  is_null_input_ = CHECK_SHAPE_NULL(input_shape, kernel_name_, "logits");
  if (is_null_input_) {
    return ret;
  }

  workspace_size_list_.clear();
  input_size_ = 1;
  input_size_ *= SizeOf(input_shape);
  size_t workspace_size = type_size_;
  if (reduction_ != ReductionMode::kNone) {
    workspace_size *= input_size_;
  }
  workspace_size_list_.push_back(workspace_size);
  return ret;
}

std::vector<std::pair<KernelAttr, KLDivLossGpuKernelMod::KLDivLossFunc>> KLDivLossGpuKernelMod::func_list_ = {
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
   &KLDivLossGpuKernelMod::LaunchKernel<float>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
   &KLDivLossGpuKernelMod::LaunchKernel<half>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
   &KLDivLossGpuKernelMod::LaunchKernel<double>},
};

std::vector<KernelAttr> KLDivLossGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, KLDivLossFunc> &item) { return item.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, KLDivLoss, KLDivLossGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
