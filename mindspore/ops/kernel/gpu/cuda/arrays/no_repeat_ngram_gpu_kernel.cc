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

#include "kernel/gpu/cuda/arrays/no_repeat_ngram_gpu_kernel.h"
#include <functional>
#include <utility>
#include <string>
#include <algorithm>
#include <memory>
#include "kernel/gpu/cuda_impl/cuda_ops/no_repeat_ngram_impl.cuh"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kNoRepeatNGramInputNum = 2;
}  // namespace
bool NoRepeatNGramGpuKernelMode::Init(const std::vector<KernelTensor *> &inputs,
                                      const std::vector<KernelTensor *> &outputs) {
  ngram_ = GetValue<int64_t>(primitive_->GetAttr("ngram_size"));
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(WARNING) << "For '" << kernel_name_ << "' does not support this kernel type: " << kernel_attr;
    return false;
  }
  state_size_ = abstract::TypeIdSize(kernel_attr.GetInputAttr(kIndex0).dtype);
  logit_size_ = abstract::TypeIdSize(kernel_attr.GetInputAttr(kIndex1).dtype);
  kernel_func_ = func_list_[index].second;

  return true;
}

int NoRepeatNGramGpuKernelMode::Resize(const std::vector<KernelTensor *> &inputs,
                                       const std::vector<KernelTensor *> &outputs) {
  int ret = KernelMod::Resize(inputs, outputs);
  if (ret != KRET_OK) {
    return ret;
  }
  if (inputs.size() != kNoRepeatNGramInputNum) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' input size must be equal 2 , but got " << inputs.size();
    return KRET_RESIZE_FAILED;
  }
  state_seq_shape_ = inputs[kIndex0]->GetShapeVector();
  log_probs_shape_ = inputs[kIndex1]->GetShapeVector();
  batch_size_ = log_probs_shape_[kIndex0];
  beam_size_ = log_probs_shape_[kIndex1];
  seq_len_ = state_seq_shape_[kIndex2];
  vocab_size_ = log_probs_shape_[kIndex2];

  return KRET_OK;
}

template <typename StateType, typename LogProbType>
bool NoRepeatNGramGpuKernelMode::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                              const std::vector<KernelTensor *> &outputs) {
  StateType *input_state = GetDeviceAddress<StateType>(inputs, kIndex0);
  LogProbType *log_probs = GetDeviceAddress<LogProbType>(inputs, kIndex1);
  LogProbType *output = GetDeviceAddress<LogProbType>(outputs, kIndex0);
  MS_EXCEPTION_IF_NULL(input_state);
  MS_EXCEPTION_IF_NULL(log_probs);
  MS_EXCEPTION_IF_NULL(output);
  auto blocks = batch_size_ * beam_size_;
  auto mem_size = (seq_len_ + 1) * sizeof(StateType);

  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(output, log_probs, inputs[kIndex1]->size(), cudaMemcpyDeviceToDevice,
                    reinterpret_cast<cudaStream_t>(cuda_stream_)),
    "For 'no_repeat_ngram', it launch memcopy failed.");

  auto status = CalculateNoRepeatNGram(input_state, log_probs, output, seq_len_, ngram_, device_id_, vocab_size_,
                                       blocks, mem_size, reinterpret_cast<cudaStream_t>(cuda_stream_));
  CHECK_CUDA_STATUS(status, kernel_name_);
  return true;
}

std::vector<std::pair<KernelAttr, NoRepeatNGramGpuKernelMode::NoRepeatNGramFunc>>
  NoRepeatNGramGpuKernelMode::func_list_ = {
    {KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
     &NoRepeatNGramGpuKernelMode::LaunchKernel<int32_t, half>},
    {KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     &NoRepeatNGramGpuKernelMode::LaunchKernel<int32_t, float>},
    {KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
     &NoRepeatNGramGpuKernelMode::LaunchKernel<int32_t, double>}};

std::vector<KernelAttr> NoRepeatNGramGpuKernelMode::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, NoRepeatNGramFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, NoRepeatNGram, NoRepeatNGramGpuKernelMode);
}  // namespace kernel
}  // namespace mindspore
