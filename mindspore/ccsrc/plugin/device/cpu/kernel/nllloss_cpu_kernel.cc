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

#include "plugin/device/cpu/kernel/nllloss_cpu_kernel.h"
#include <algorithm>
#include <map>
#include <string>
#include <utility>

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kNLLLossInputsNum = 5;
constexpr size_t kNLLLossOutputsNum = 2;
constexpr int minLabelNum = 0;
constexpr auto kReductionIdx = 3;
constexpr auto kIgnoreIndexIdx = 4;
}  // namespace

bool NLLLossCpuKernelMod::Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);

  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int NLLLossCpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  int ret = 0;
  if ((ret = KernelMod::Resize(inputs, outputs)) != 0) {
    return ret;
  }
  auto reduction = inputs[kReductionIdx]->GetValueWithCheck<int64_t>();
  reduction_type_ = static_cast<ops::Reduction>(reduction);
  ignore_index_ = inputs[kIgnoreIndexIdx]->GetValueWithCheck<int64_t>();

  auto logits_shape = inputs[kIndex0]->GetShapeVector();
  nllloss_param_.batch_ = LongToInt(logits_shape[kIndex0]);
  nllloss_param_.class_num_ = LongToInt(logits_shape[kIndex1]);

  return KRET_OK;
}

template <typename T>
bool NLLLossCpuKernelMod::LaunchKernel(const std::vector<kernel::KernelTensor *> &inputs,
                                       const std::vector<kernel::KernelTensor *> &workspace,
                                       const std::vector<kernel::KernelTensor *> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(kNLLLossInputsNum, inputs.size(), kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(kNLLLossOutputsNum, outputs.size(), kernel_name_);

  const auto *logits = static_cast<float *>(inputs[kIndex0]->device_ptr());
  const auto *labels = static_cast<T *>(inputs[kIndex1]->device_ptr());
  const auto *weight = static_cast<float *>(inputs[kIndex2]->device_ptr());
  auto *loss = static_cast<float *>(outputs[kIndex0]->device_ptr());
  auto *total_weight = static_cast<float *>(outputs[kIndex1]->device_ptr());
  if (logits == nullptr || labels == nullptr || weight == nullptr) {
    MS_LOG(EXCEPTION) << "Nllloss does not support null input";
  }

  float total_loss = 0.0;
  float tmp_total_weight = 0.0;
  for (int i = 0; i < nllloss_param_.batch_; i++) {
    if (labels[i] == ignore_index_) {
      continue;
    }
    if (labels[i] < minLabelNum || labels[i] > nllloss_param_.class_num_) {
      MS_EXCEPTION(ValueError) << "For '" << kernel_name_ << "', the label must in scope[0, C-1], but got" << labels[i];
    }
    if (!(labels[i] < nllloss_param_.class_num_)) {
      MS_EXCEPTION(ValueError) << "For '" << kernel_name_
                               << "', the labels should be smaller than the number of classes, but got " << labels[i];
    }
    int index = i * nllloss_param_.class_num_ + labels[i];
    float n_weight = weight[labels[i]];
    float n_loss = -logits[index] * n_weight;
    tmp_total_weight += n_weight;
    total_loss += n_loss;
    if (reduction_type_ == ops::Reduction::NONE) {
      loss[i] = n_loss;
    }
  }

  *total_weight = tmp_total_weight;
  if (reduction_type_ == ops::Reduction::SUM) {
    *loss = total_loss;
  } else if (reduction_type_ == ops::Reduction::MEAN) {
    *loss = total_loss / tmp_total_weight;
  }
  return true;
}

std::vector<std::pair<KernelAttr, NLLLossCpuKernelMod::NLLLossFunc>> NLLLossCpuKernelMod::func_list_ = {
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32),
   &NLLLossCpuKernelMod::LaunchKernel<int32_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32),
   &NLLLossCpuKernelMod::LaunchKernel<int64_t>}};

std::vector<KernelAttr> NLLLossCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, NLLLossCpuKernelMod::NLLLossFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, NLLLoss, NLLLossCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
