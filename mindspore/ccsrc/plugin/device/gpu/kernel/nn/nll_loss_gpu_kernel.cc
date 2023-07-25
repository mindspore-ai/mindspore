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

#include "plugin/device/gpu/kernel/nn/nll_loss_gpu_kernel.h"
#include <map>
#include <utility>
#include "mindspore/core/ops/nllloss.h"
#include "kernel/common_utils.h"

namespace mindspore {
namespace kernel {
namespace {
std::map<Reduction, ReductionMode> kReductionMap = {{Reduction::MEAN, ReductionMode::kMean},
                                                    {Reduction::REDUCTION_SUM, ReductionMode::kSum},
                                                    {Reduction::NONE, ReductionMode::kNone}};
}
bool NLLLossGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                               const std::vector<KernelTensorPtr> &outputs) {
  auto kernel_ptr = std::dynamic_pointer_cast<ops::NLLLoss>(base_operator);
  if (!kernel_ptr) {
    MS_LOG(ERROR) << "cast NLLLoss ops failed!";
    return false;
  }
  kernel_name_ = kernel_ptr->GetPrim()->name();

  auto reduction = kernel_ptr->get_reduction();
  reduction_ = kReductionMap[reduction];

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  ignore_index_ = static_cast<int32_t>(kernel_ptr->get_ignore_index());
  return true;
}

int NLLLossGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                const std::vector<KernelTensorPtr> &outputs,
                                const std::map<uint32_t, tensor::TensorPtr> &) {
  int ret = 0;
  if ((ret = KernelMod::Resize(base_operator, inputs, outputs)) != 0) {
    return ret;
  }

  auto logits_shape = inputs[kIndex0]->GetShapeVector();
  label_size_ = logits_shape[0];
  num_classes_ = logits_shape[1];
  return KRET_OK;
}

template <typename T, typename S>
bool NLLLossGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                                       const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  T *logits = GetDeviceAddress<T>(inputs, 0);
  auto *labels = GetDeviceAddress<int32_t>(inputs, 1);
  S *weights = GetDeviceAddress<S>(inputs, 2);
  T *loss = GetDeviceAddress<T>(outputs, 0);
  S *total_weight = GetDeviceAddress<S>(outputs, 1);
  auto status = NLLLoss(logits, labels, weights, loss, total_weight, label_size_, num_classes_, reduction_,
                        ignore_index_, reinterpret_cast<cudaStream_t>(stream_ptr));
  CHECK_CUDA_STATUS(status, kernel_name_);
  return true;
}

std::vector<std::pair<KernelAttr, NLLLossGpuKernelMod::NLLLossLaunchFunc>> NLLLossGpuKernelMod::func_list_ = {
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32),
   &NLLLossGpuKernelMod::LaunchKernel<float, float>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat16),
   &NLLLossGpuKernelMod::LaunchKernel<float, half>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeFloat32),
   &NLLLossGpuKernelMod::LaunchKernel<half, float>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeFloat16),
   &NLLLossGpuKernelMod::LaunchKernel<half, half>}};

std::vector<KernelAttr> NLLLossGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(
    func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
    [](const std::pair<KernelAttr, NLLLossGpuKernelMod::NLLLossLaunchFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, NLLLoss, NLLLossGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
