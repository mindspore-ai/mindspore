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

#include "plugin/device/cpu/kernel/ctcloss_v2_cpu_kernel.h"
#include <string>
#include <limits>
#include <algorithm>
#include "mindspore/core/ops/ctc_loss_v2.h"
#include "abstract/utils.h"
namespace mindspore {
namespace kernel {
namespace {
template <typename T>
T log_sum_exp(T a, T b) {
  constexpr T neg_inf = -std::numeric_limits<T>::infinity();
  if (a < b) {
    std::swap(a, b);
  }
  if (b == neg_inf) {
    return a;
  }
  return a + std::log1p(std::exp(b - a));
}
}  // namespace
bool CTCLossV2CpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                 const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it got empty inputs or outputs, which is invalid.";
    return false;
  }

  // Save for later use
  outputs_ = outputs;

  // Getting values
  auto kernel_ptr = std::make_shared<ops::CTCLossV2>(base_operator->GetPrim());
  blank_ = kernel_ptr->get_blank();
  auto reduction = kernel_ptr->get_reduction();

  static const HashMap<std::string, ReductionType> kReductionMap = {{MEAN, Mean}, {SUM, Sum}, {NONE, None}};
  auto iter = kReductionMap.find(reduction);
  if (iter == kReductionMap.end()) {
    MS_EXCEPTION(ValueError) << "For " << kernel_name_
                             << ", the attr 'reduction' only support 'mean', 'sum' and 'none', but got " << reduction;
  }
  reduction_ = iter->second;

  // Dealing with multiple types
  {
    auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
    auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
    if (!is_match) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel type: " << kernel_attr;
      return false;
    }
    kernel_func_ = func_list_[index].second;
  }
  return true;
}
int CTCLossV2CpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                  const std::vector<KernelTensorPtr> &outputs,
                                  const std::map<uint32_t, tensor::TensorPtr> &) {
  if (int ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
    dyamic_shape_ = ret == KRET_UNKNOWN_SHAPE;
    return ret;
  }
  // log_probs_shape.size() is 3, checked in CTCLossV2InferShape
  auto log_probs_shape = inputs[kIndex0]->GetShapeVector();
  time_series_ = log_probs_shape[kIndex0];
  batch_ = log_probs_shape[kIndex1];
  num_classes_ = log_probs_shape[kIndex2];
  // target_shape.size() is 2, checked in CTCLossV2InferShape
  target_shape_ = inputs[kIndex1]->GetShapeVector();
  padded_targets = target_shape_.size() != 1;
  // Deal with workspace_size_list_
  workspace_size_list_.clear();
  if (reduction_ != None) {
    const size_t value_size = abstract::TypeIdSize(inputs.at(kIndex0)->GetDtype());
    workspace_size_list_ = {batch_ * value_size};
  }

  return KRET_OK;
}

template <typename S>
std::vector<S> CTCLossV2CpuKernelMod::IndexProcessing(const S *input_lengths, const S *target_lengths) {
  std::vector<S> target_offsets(batch_);
  if (padded_targets) {
    const auto target_length = target_shape_[kIndex1];
    for (int i = 0; i < batch_; ++i) {
      if (target_lengths[i] < 0 || target_lengths[i] > target_length) {
        MS_EXCEPTION(ValueError) << "For '" << kernel_name_ << "', the target_lengths[" << i
                                 << "] = " << target_lengths[i]
                                 << " is negative or larger than target.shape[1] = " << target_length << ".";
      }
      target_offsets[i] = target_length * i;
    }
  } else {
    S current = 0;
    for (int i = 0; i < batch_; ++i) {
      target_offsets[i] = current;
      current += target_lengths[i];
    }
    const int64_t target_length = target_shape_[kIndex0];
    if (current != target_length) {
      MS_EXCEPTION(ValueError) << "For '" << kernel_name_ << "', the sum of target_lengths " << current
                               << " should be equal to targets.shape[0] " << target_length << ".";
    }
  }

  for (int64_t b = 0; b < batch_; ++b) {
    const auto input_length = input_lengths[b];
    const auto target_length = target_lengths[b];
    if (input_length > time_series_) {
      MS_EXCEPTION(ValueError) << "For '" << kernel_name_ << "', the input_lengths[" << b << "] = " << input_length
                               << " should be smaller than probs.shape[0] = " << time_series_;
    }
    if (input_length < 0 || input_length < target_length) {
      MS_EXCEPTION(ValueError) << "For '" << kernel_name_ << "', the input_lengths[" << b << "] = " << input_length
                               << " should be non-negative and smaller than target_lengths[" << b
                               << "] = " << target_length;
    }
  }
  return target_offsets;
}

template <typename T, typename S>
bool CTCLossV2CpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                         const std::vector<kernel::AddressPtr> &workspace,
                                         const std::vector<kernel::AddressPtr> &outputs) {
  auto probs = reinterpret_cast<T *>(inputs[kIndex0]->addr);
  auto targets = reinterpret_cast<S *>(inputs[kIndex1]->addr);
  auto input_lengths = reinterpret_cast<S *>(inputs[kIndex2]->addr);
  auto target_lengths = reinterpret_cast<S *>(inputs[kIndex3]->addr);
  auto batched_log_alpha = reinterpret_cast<T *>(outputs[kIndex1]->addr);

  auto neg_log_likelihood =
    reinterpret_cast<T *>((reduction_ == None) ? outputs[kIndex0]->addr : workspace[kIndex0]->addr);

  std::vector<S> target_offsets = IndexProcessing(input_lengths, target_lengths);

  const auto max_target_length = padded_targets
                                   ? target_shape_[kIndex1]
                                   : static_cast<int64_t>(*std::max_element(target_lengths, target_lengths + batch_));
  const auto padded_max_target_length = 2 * max_target_length + 1;
  const auto batched_log_alpha_offset = time_series_ * padded_max_target_length;

  NdTensorIterator<kDim3> probs_it(time_series_, batch_, num_classes_);
  NdTensorIterator<kDim2> log_alpha_it(time_series_, padded_max_target_length);

  // Set log_alpha shape
  if (dyamic_shape_) {
    const std::vector<int64_t> log_alpha_shape = {batch_, time_series_, padded_max_target_length};
    outputs_[kIndex1]->SetShapeVector(log_alpha_shape);
  }

  std::fill(batched_log_alpha, batched_log_alpha + (batch_ * time_series_ * padded_max_target_length),
            -std::numeric_limits<T>::infinity());

  auto task = [this, probs, &probs_it, input_lengths, targets, target_lengths, target_offsets, batched_log_alpha,
               batched_log_alpha_offset, &log_alpha_it, neg_log_likelihood](size_t start, size_t end) {
    for (size_t b = start; b < end; b++) {
      const auto input_length = input_lengths[b];
      const auto target_length = target_lengths[b];
      const auto current_target = targets + target_offsets[b];
      T *log_alpha = batched_log_alpha + batched_log_alpha_offset * b;
      log_alpha[log_alpha_it(0, 0)] = probs[probs_it(0, b, blank_)];
      log_alpha[log_alpha_it(0, 1)] = probs[probs_it(0, b, current_target[0])];
      for (int64_t t = 1; t < input_length; ++t) {
        for (int64_t i = 0; i < 2 * target_length + 1; ++i) {
          S target = GetBlankPaddedTarget(current_target, i);
          T alpha = log_alpha[log_alpha_it(t - 1, i)];
          if (i - 1 >= 0) {
            alpha = log_sum_exp(alpha, log_alpha[log_alpha_it(t - 1, i - 1)]);
          }
          if ((target != blank_) && (i - 2 >= 0) && (target != GetBlankPaddedTarget(current_target, i - 2))) {
            alpha = log_sum_exp(alpha, log_alpha[log_alpha_it(t - 1, i - 2)]);
          }
          log_alpha[log_alpha_it(t, i)] = alpha + probs[probs_it(t, b, target)];
        }
      }
      if (target_length != 0) {
        neg_log_likelihood[b] = -log_sum_exp(log_alpha[log_alpha_it(input_length - 1, 2 * target_length)],
                                             log_alpha[log_alpha_it(input_length - 1, 2 * target_length - 1)]);
      } else {
        neg_log_likelihood[b] = -log_alpha[log_alpha_it(input_length - 1, 0)];
      }
    }
  };

  ParallelLaunchAutoSearch(task, batch_, this, &parallel_search_info_);

  if (reduction_ != None) {
    auto reduced_result = reinterpret_cast<T *>(outputs[kIndex0]->addr);
    *reduced_result = DoReduce(neg_log_likelihood, target_lengths);
  }

  return true;
}

template <typename T, typename S>
T CTCLossV2CpuKernelMod::DoReduce(T *neg_log_likelihood, const S *target_lengths) {
  if (reduction_ == Mean) {
    for (int b = 0; b < batch_; ++b) {
      neg_log_likelihood[b] = neg_log_likelihood[b] / target_lengths[b];
    }
    T sum = std::accumulate(neg_log_likelihood, neg_log_likelihood + batch_, 0.0);
    return sum / batch_;
  } else {  // Sum
    return std::accumulate(neg_log_likelihood, neg_log_likelihood + batch_, 0.0);
  }
}

std::vector<KernelAttr> CTCLossV2CpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, CTCLossV2Func> &pair) { return pair.first; });
  return support_list;
}

std::vector<std::pair<KernelAttr, CTCLossV2CpuKernelMod::CTCLossV2Func>> CTCLossV2CpuKernelMod::func_list_ = {
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32),
   &CTCLossV2CpuKernelMod::LaunchKernel<float, int>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeFloat64)
     .AddOutputAttr(kNumberTypeFloat64),
   &CTCLossV2CpuKernelMod::LaunchKernel<double, int>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32),
   &CTCLossV2CpuKernelMod::LaunchKernel<float, int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeFloat64)
     .AddOutputAttr(kNumberTypeFloat64),
   &CTCLossV2CpuKernelMod::LaunchKernel<double, int64_t>},
};
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, CTCLossV2, CTCLossV2CpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
