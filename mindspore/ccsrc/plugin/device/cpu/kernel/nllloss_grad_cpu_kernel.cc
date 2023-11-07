/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/kernel/nllloss_grad_cpu_kernel.h"
#include <algorithm>
#include <map>
#include <string>
#include <utility>
#include <unordered_map>

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kNLLLossGradInputsNum = 7;
constexpr size_t kNLLLossGradOutputsNum = 1;
constexpr auto kReductionIdx = 5;
constexpr auto kIgnoreIndexIdx = 6;
}  // namespace

bool NLLLossGradCpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                   const std::vector<KernelTensor *> &outputs) {
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);

  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(EXCEPTION) << kernel_name_ << " does not support this kernel data type: " << kernel_attr;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int NLLLossGradCpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                    const std::vector<KernelTensor *> &outputs) {
  int ret = 0;
  if ((ret = KernelMod::Resize(inputs, outputs)) != 0) {
    return ret;
  }
  auto reduction = inputs[kReductionIdx]->GetValueWithCheck<int64_t>();
  reduction_type_ = static_cast<MsPyEnum::Reduction>(reduction);
  ignore_index_ = inputs[kIgnoreIndexIdx]->GetValueWithCheck<int64_t>();
  auto logits_shape = inputs[0]->GetShapeVector();
  nllloss_param_.batch_ = LongToInt(logits_shape[0]);
  nllloss_param_.class_num_ = LongToInt(logits_shape[1]);
  return KRET_OK;
}

template <typename T>
bool NLLLossGradCpuKernelMod::LaunchKernel(const std::vector<kernel::KernelTensor *> &inputs,
                                           const std::vector<kernel::KernelTensor *> &workspace,
                                           const std::vector<kernel::KernelTensor *> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(kNLLLossGradInputsNum, inputs.size(), kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(kNLLLossGradOutputsNum, outputs.size(), kernel_name_);

  const auto *logits = reinterpret_cast<float *>(inputs[0]->device_ptr());
  const auto *loss_grad = reinterpret_cast<float *>(inputs[1]->device_ptr());
  const auto *labels = static_cast<T *>(inputs[2]->device_ptr());
  const auto *weight = reinterpret_cast<float *>(inputs[3]->device_ptr());
  const auto *total_weight = reinterpret_cast<float *>(inputs[4]->device_ptr());
  auto *logits_grad = reinterpret_cast<float *>(outputs[0]->device_ptr());

  if (logits == nullptr || loss_grad == nullptr || labels == nullptr || weight == nullptr || total_weight == nullptr) {
    MS_LOG(ERROR) << "For NLLLossGrad, it does not support NULL input";
  }
  auto ret =
    memset_s(logits_grad, outputs[0]->size(), 0, nllloss_param_.batch_ * nllloss_param_.class_num_ * sizeof(float));
  if (ret != EOK) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', memset_s failed, ret=" << ret;
  }
  for (int i = 0; i < nllloss_param_.batch_; i++) {
    if (labels[i] == ignore_index_) {
      continue;
    }
    if (!(labels[i] < nllloss_param_.class_num_)) {
      MS_EXCEPTION(ValueError) << "For '" << kernel_name_
                               << "', the labels should be smaller than the number of classes, but got " << labels[i];
    }
    int index = i * nllloss_param_.class_num_ + labels[i];
    float n_weight = weight[labels[i]];
    if (reduction_type_ == MsPyEnum::Reduction::SUM) {
      logits_grad[index] = -loss_grad[0] * n_weight;
    } else if (reduction_type_ == MsPyEnum::Reduction::MEAN) {
      logits_grad[index] = -loss_grad[0] * n_weight / *total_weight;
    } else {
      logits_grad[index] = -loss_grad[i] * n_weight;
    }
  }
  return true;
}

std::vector<std::pair<KernelAttr, NLLLossGradCpuKernelMod::NLLLossGradFunc>> NLLLossGradCpuKernelMod::func_list_ = {
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeFloat32),
   &NLLLossGradCpuKernelMod::LaunchKernel<int32_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeFloat32),
   &NLLLossGradCpuKernelMod::LaunchKernel<int64_t>}};

std::vector<KernelAttr> NLLLossGradCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(
    func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
    [](const std::pair<KernelAttr, NLLLossGradCpuKernelMod::NLLLossGradFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, NLLLossGrad, NLLLossGradCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
