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

#include "kernel/gpu/cuda/nn/nll_loss_gpu_kernel.h"
#include <map>
#include <utility>
#include "include/runtime/hardware_abstract/kernel_base/common_utils.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr auto kReductionIdx = 3;
constexpr auto kIgnoreIndexIdx = 4;
constexpr size_t kNLLLossLogitsDim = 2;
constexpr size_t kNLLLossLabelsDim = 1;
}  // namespace
bool NLLLossGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int NLLLossGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  int ret = 0;
  if ((ret = KernelMod::Resize(inputs, outputs)) != 0) {
    return ret;
  }
  auto reduction = inputs[kReductionIdx]->GetValueWithCheck<int64_t>();
  reduction_ = kEnumReductionModeMap[reduction];
  ignore_index_ = inputs[kIgnoreIndexIdx]->GetValueWithCheck<int64_t>();
  auto logits_shape = inputs[kIndex0]->GetShapeVector();
  auto labels_shape = inputs[kIndex1]->GetShapeVector();
  auto weight_shape = inputs[kIndex2]->GetShapeVector();
  size_t logits_dim = logits_shape.size();
  size_t labels_dim = labels_shape.size();
  size_t weight_dim = weight_shape.size();
  if (logits_dim != kNLLLossLogitsDim) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimension of 'logits' should be 2, but got: " << logits_dim;
  }
  if ((labels_dim != kNLLLossLabelsDim) || (weight_dim != kNLLLossLabelsDim)) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the dimension of 'labels' and 'weight' should all be 1, but got: " << labels_dim << " and "
                      << weight_dim << " respectively.";
  }
  if ((logits_shape[kIndex0] != labels_shape[kIndex0]) || (logits_shape[kIndex1] != weight_shape[kIndex0])) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', logits_shape[0] should be equal to labels_shape[0], logits_shape[1] should be equal "
                         "to weight_shape[kIndex0], but got logits_shape: "
                      << logits_shape << ", labels_shape: " << labels_shape << ", weight_shape: " << weight_shape
                      << ".";
  }

  label_size_ = logits_shape[0];
  num_classes_ = logits_shape[1];
  return KRET_OK;
}

template <typename T, typename S>
bool NLLLossGpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                       const std::vector<KernelTensor *> &workspace,
                                       const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
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
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32),
   &NLLLossGpuKernelMod::LaunchKernel<float, float>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat16),
   &NLLLossGpuKernelMod::LaunchKernel<float, half>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeFloat32),
   &NLLLossGpuKernelMod::LaunchKernel<half, float>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
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
