/**
 * Copyright 2021-2023 Huawei Technologies Co., Ltd
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

#include "kernel/gpu/cuda/nn/nll_loss_grad_gpu_kernel.h"
#include <map>
#include <utility>
#include "kernel/gpu/cuda_impl/cuda_ops/loss_with_reduction_impl.cuh"

namespace mindspore {
namespace kernel {
namespace {
constexpr auto kReductionIdx = 5;
constexpr auto kIgnoreIndexIdx = 6;
}  // namespace

bool NLLLossGradGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                   const std::vector<KernelTensor *> &outputs) {
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int NLLLossGradGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                    const std::vector<KernelTensor *> &outputs) {
  int ret = 0;
  if ((ret = KernelMod::Resize(inputs, outputs)) != 0) {
    return ret;
  }
  auto reduction = inputs[kReductionIdx]->GetValueWithCheck<int64_t>();
  reduction_ = kEnumReductionModeMap[reduction];
  ignore_index_ = inputs[kIgnoreIndexIdx]->GetValueWithCheck<int64_t>();

  auto logits_shape = inputs[kIndex0]->GetShapeVector();
  size_t kMinShapeSize = 2;
  if (logits_shape.size() < kMinShapeSize) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimension of logits cannot be less than 2, but "
                      << "got the " << logits_shape.size();
  }
  n_ = LongToInt(logits_shape[0]);
  c_ = LongToInt(logits_shape[1]);
  if (reduction_ == ReductionMode::kNone) {
    num_dloss_ = n_;
  }
  return KRET_OK;
}

template <typename T, typename S>
bool NLLLossGradGpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                           const std::vector<KernelTensor *> &workspace,
                                           const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  T *input_device = GetDeviceAddress<T>(inputs, 0);
  T *dloss_device = GetDeviceAddress<T>(inputs, 1);
  int32_t *target_device = GetDeviceAddress<int32_t>(inputs, 2);  // nll_loss_grad only supports int32 target
  S *weight_device = GetDeviceAddress<S>(inputs, 3);
  S *total_weight_device = GetDeviceAddress<S>(inputs, 4);

  T *dinput_device = GetDeviceAddress<T>(outputs, 0);

  auto status = NLLLossGrad(n_, c_, reduction_, input_device, target_device, weight_device, total_weight_device,
                            dloss_device, dinput_device, ignore_index_, reinterpret_cast<cudaStream_t>(stream_ptr));
  CHECK_CUDA_STATUS(status, kernel_name_);
  return true;
}

std::vector<std::pair<KernelAttr, NLLLossGradGpuKernelMod::NLLLossGradLaunchFunc>> NLLLossGradGpuKernelMod::func_list_ =
  {{KernelAttr()
      .AddInputAttr(kNumberTypeFloat32)
      .AddInputAttr(kNumberTypeFloat32)
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeFloat32)
      .AddInputAttr(kNumberTypeFloat32)
      .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
      .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
      .AddOutputAttr(kNumberTypeFloat32),
    &NLLLossGradGpuKernelMod::LaunchKernel<float, float>},
   {KernelAttr()
      .AddInputAttr(kNumberTypeFloat32)
      .AddInputAttr(kNumberTypeFloat32)
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeFloat16)
      .AddInputAttr(kNumberTypeFloat16)
      .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
      .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
      .AddOutputAttr(kNumberTypeFloat32),
    &NLLLossGradGpuKernelMod::LaunchKernel<float, half>},
   {KernelAttr()
      .AddInputAttr(kNumberTypeFloat16)
      .AddInputAttr(kNumberTypeFloat16)
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeFloat32)
      .AddInputAttr(kNumberTypeFloat32)
      .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
      .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
      .AddOutputAttr(kNumberTypeFloat16),
    &NLLLossGradGpuKernelMod::LaunchKernel<half, float>},
   {KernelAttr()
      .AddInputAttr(kNumberTypeFloat16)
      .AddInputAttr(kNumberTypeFloat16)
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeFloat16)
      .AddInputAttr(kNumberTypeFloat16)
      .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
      .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
      .AddOutputAttr(kNumberTypeFloat16),
    &NLLLossGradGpuKernelMod::LaunchKernel<half, half>}};

std::vector<KernelAttr> NLLLossGradGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(
    func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
    [](const std::pair<KernelAttr, NLLLossGradGpuKernelMod::NLLLossGradLaunchFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, NLLLossGrad, NLLLossGradGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
