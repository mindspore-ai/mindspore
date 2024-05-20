/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/kernel/fftfreq_cpu_kernel.h"
#include "ops/op_utils.h"
#include "kernel/kernel.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr auto kInputNum = 3;
constexpr auto kOutputNum = 1;

}  // namespace
bool FFTFreqCpuKernelMod::Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(EXCEPTION) << kernel_name_ << " valid cpu kernel does not support this kernel data type: " << kernel_attr;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int FFTFreqCpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  if (auto ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }

  n_ = inputs[kIndex0]->GetValueWithCheck<int64_t>();
  auto d_opt = inputs[kIndex1]->GetOptionalValueWithCheck<float>();
  if (d_opt.has_value()) {
    d_ = d_opt.value();
  }
  return KRET_OK;
}

template <typename T>
bool FFTFreqCpuKernelMod::LaunchKernel(const std::vector<kernel::KernelTensor *> &inputs,
                                       const std::vector<kernel::KernelTensor *> &outputs) {
  auto *output_ptr = reinterpret_cast<T *>(outputs[kIndex0]->device_ptr());
  int64_t index = 0;
  int64_t mid = (n_ + 1) / 2;
  int64_t r = (n_ % 2) ? 1 : 0;
  double weight = 1.0 / (d_ * n_);
  while (index < mid) {
    output_ptr[index] = static_cast<T>(index * weight);
    index++;
  }
  if (kernel_name_ == prim::kPrimRFFTFreq->name()) {
    output_ptr[index] = static_cast<T>(index * weight);
  } else {
    int64_t k = 0;
    while (k < n_ - mid) {
      output_ptr[index + k] = static_cast<T>((-index + r + k) * weight);
      k++;
    }
  }
  return true;
}

std::vector<std::pair<KernelAttr, FFTFreqCpuKernelMod::FFTFreqFunc>> FFTFreqCpuKernelMod::func_list_ = {
  {KernelAttr()
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
     .AddOptionalInputAttr(kNumberTypeFloat32)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeFloat32),
   &FFTFreqCpuKernelMod::LaunchKernel<float>}};

std::vector<KernelAttr> FFTFreqCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, FFTFreqFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, FFTFreq, FFTFreqCpuKernelMod);
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, RFFTFreq, FFTFreqCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
