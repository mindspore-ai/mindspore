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

#include "plugin/device/cpu/kernel/no_repeat_ngram_cpu_kernel.h"

#include <algorithm>
#include <memory>
#include <utility>
#include <map>
#include <cmath>
#include <limits>

#include "mindspore/core/ops/no_repeat_ngram.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "include/common/thread_pool.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kNoRepeatNGramInputsNum = 2;
constexpr size_t kNoRepeatNGramOutputsNum = 1;
constexpr size_t kNoRepeatNGramDim = 3;
constexpr int64_t kNoRepeatNGramParamValue = 0;
}  // namespace

bool NoRepeatNGramCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                     const std::vector<KernelTensorPtr> &outputs) {
  auto kernel_ptr = std::dynamic_pointer_cast<ops::NoRepeatNGram>(base_operator);
  if (!kernel_ptr) {
    MS_LOG(ERROR) << "Cast NoRepeatNGram ops failed!";
    return false;
  }
  kernel_name_ = kernel_ptr->name();
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kNoRepeatNGramInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kNoRepeatNGramOutputsNum, kernel_name_);

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "NoRepeatNGram does not support this kernel data type: " << kernel_attr;
  }

  base_operator_ = base_operator;
  kernel_func_ = func_list_[index].second;
  return true;
}

int NoRepeatNGramCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                      const std::vector<KernelTensorPtr> &outputs,
                                      const std::map<uint32_t, tensor::TensorPtr> &others) {
  int ret = 0;
  if ((ret = NativeCpuKernelMod::Resize(base_operator, inputs, outputs, others)) != 0) {
    MS_LOG(WARNING) << kernel_name_ << " resize failed.";
    return ret;
  }
  state_seq_shape_ = inputs[kIndex0]->GetShapeVector();
  log_probs_shape_ = inputs[kIndex1]->GetShapeVector();
  ngram_size_ = GetValue<int64_t>(base_operator->GetAttr("ngram_size"));
  return 0;
}

void NoRepeatNGramCpuKernelMod::CheckAndInitParams() {
  size_t state_seq_shape_len = state_seq_shape_.size();
  size_t log_probs_shape_len = log_probs_shape_.size();
  if ((state_seq_shape_len != log_probs_shape_len) || (state_seq_shape_len != kNoRepeatNGramDim)) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimension of 'state_seq' " << state_seq_shape_len
                      << " and 'log_probs' " << log_probs_shape_len << " is illegal.";
  }
  state_dim_ = state_seq_shape_[state_seq_shape_len - 1];
  output_dim_ = log_probs_shape_[log_probs_shape_len - 1];
  if ((ngram_size_ <= kNoRepeatNGramParamValue) || (ngram_size_ >= state_dim_)) {
    MS_LOG(EXCEPTION) << "Param ngram_size is illegal";
  }
  num_seq_ = 1;
  for (size_t i = 0; i < log_probs_shape_len - 1; i++) {
    num_seq_ *= log_probs_shape_[i];
  }
  output_size_ = num_seq_ * output_dim_;
}

template <typename T>
bool NoRepeatNGramCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                             const std::vector<kernel::AddressPtr> &,
                                             const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kNoRepeatNGramInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kNoRepeatNGramOutputsNum, kernel_name_);
  auto *state_seq = reinterpret_cast<int32_t *>(inputs[kIndex0]->addr);
  auto *log_probs = reinterpret_cast<T *>(inputs[kIndex1]->addr);
  auto *output = reinterpret_cast<T *>(outputs[kIndex0]->addr);
  CheckAndInitParams();

  for (size_t i = 0; i < LongToSize(output_size_); i++) {
    output[i] = log_probs[i];
  }

  std::vector<int32_t> array_dim(state_dim_);
  std::vector<int32_t> array_ngram(ngram_size_ - 1);
  for (int64_t i = 0; i < num_seq_; i++) {
    int64_t src_index_i = state_dim_ * i;
    int64_t output_index_i = output_dim_ * i;
    for (int64_t k = 0; k < state_dim_; k++) {
      int64_t src_index_k = k + src_index_i;
      array_dim[LongToSize(k)] = static_cast<int32_t>(state_seq[LongToSize(src_index_k)]);
      if (k > (state_dim_ - ngram_size_)) {
        array_ngram[LongToSize(k + ngram_size_ - state_dim_ - 1)] =
          static_cast<int32_t>(state_seq[LongToSize(src_index_k)]);
      }
    }
    for (int64_t j = 0; j < state_dim_ - ngram_size_ + 1; j++) {
      if (equal(array_ngram.begin(), array_ngram.end(), array_dim.begin() + j)) {
        int64_t output_index_j = static_cast<int64_t>(array_dim[LongToSize(j + ngram_size_ - 1)]);
        if (output_index_j < 0 || output_index_j >= output_dim_) {
          MS_EXCEPTION(ValueError) << "For NoRepeatNGram, the id in the state input is not valid. "
                                   << "Please make the value in the state_seq in the valid range between [0,"
                                   << output_dim_ << ").";
        }
        output[output_index_i + output_index_j] = -(std::numeric_limits<T>::max)();
      }
    }
  }
  return true;
}

std::vector<std::pair<KernelAttr, NoRepeatNGramCpuKernelMod::NoRepeatNGramFunc>> NoRepeatNGramCpuKernelMod::func_list_ =
  {{KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
    &NoRepeatNGramCpuKernelMod::LaunchKernel<double>},
   {KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
    &NoRepeatNGramCpuKernelMod::LaunchKernel<float>},
   {KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
    &NoRepeatNGramCpuKernelMod::LaunchKernel<float16>}};

std::vector<KernelAttr> NoRepeatNGramCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, NoRepeatNGramFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, NoRepeatNGram, NoRepeatNGramCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
