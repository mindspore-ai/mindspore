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

#include "plugin/device/cpu/kernel/bce_with_logits_loss_cpu_kernel.h"
#include <vector>
#include <algorithm>
#include <utility>
#include <string>
#include <map>
#include <functional>
#include <memory>
#include "mindspore/core/ops/bce_with_logits_loss.h"

namespace mindspore {
namespace kernel {
bool BCEWithLogitsLossCpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                         const std::vector<KernelTensorPtr> &inputs,
                                         const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' got empty inputs or outputs, which is invalid.";
    return false;
  }
  if (kernel_name_ != prim::kPrimBCEWithLogitsLoss->name()) {
    MS_LOG(ERROR) << "For 'BCEWithLogitsLoss', kernl name invalid, got" << kernel_name_;
    return false;
  }
  auto kernel_ptr = std::make_shared<ops::BCEWithLogitsLoss>(base_operator->GetPrim());
  const auto reduction = kernel_ptr->get_reduction();
  if (reduction == NONE) {
    reduction_ = kNone;
  } else if (reduction == MEAN) {
    reduction_ = kMean;
  } else if (reduction == SUM) {
    reduction_ = kSum;
  } else {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the 'reduction' must be 'none', 'mean', or 'sum', but got "
                  << reduction;
    return false;
  }
  if (!MatchKernelFunc(base_operator, inputs, outputs)) {
    return false;
  }
  return true;
}

int BCEWithLogitsLossCpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                          const std::vector<KernelTensorPtr> &inputs,
                                          const std::vector<KernelTensorPtr> &outputs,
                                          const std::map<uint32_t, tensor::TensorPtr> &) {
  ResetResource();
  if (auto ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
    return ret;
  }

  auto input_logits_shape = inputs.at(kIndex0)->GetShapeVector();
  (void)std::transform(input_logits_shape.begin(), input_logits_shape.end(), std::back_inserter(input_logits_shape_),
                       LongToSize);
  input_size_ = std::accumulate(input_logits_shape_.begin(), input_logits_shape_.end(), 1, std::multiplies<size_t>());

  auto input_label_shape = inputs.at(kIndex1)->GetShapeVector();
  (void)std::transform(input_label_shape.begin(), input_label_shape.end(), std::back_inserter(input_label_shape_),
                       LongToSize);
  auto input_weight_shape = inputs.at(kIndex2)->GetShapeVector();
  (void)std::transform(input_weight_shape.begin(), input_weight_shape.end(), std::back_inserter(input_weight_shape_),
                       LongToSize);
  auto input_post_weight_shape = inputs.at(kIndex3)->GetShapeVector();
  (void)std::transform(input_post_weight_shape.begin(), input_post_weight_shape.end(),
                       std::back_inserter(input_post_weight_shape_), LongToSize);

  // output_size_list_ should be clear and reset.
  output_size_list_.clear();
  size_t unit_byte_size = GetTypeByte(TypeIdToType(outputs.at(kIndex0)->GetDtype()));
  size_t input_byte_size = input_size_ * unit_byte_size;
  if (reduction_ == kNone) {
    // The output is a Tensor in ReductionType none.
    output_size_list_.emplace_back(input_byte_size);
  } else {
    // The output is a scalar in ReductionType mean or sum.
    output_size_list_.emplace_back(unit_byte_size);
  }
  return KRET_OK;
}

template <typename T>
bool BCEWithLogitsLossCpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                                 const std::vector<AddressPtr> &workspace,
                                                 const std::vector<AddressPtr> &outputs) {
  const auto input_logits = GetDeviceAddress<T>(inputs, kIndex0);
  const auto input_label = GetDeviceAddress<T>(inputs, kIndex1);
  const auto input_weight = GetDeviceAddress<T>(inputs, kIndex2);
  const auto input_pos_weight = GetDeviceAddress<T>(inputs, kIndex3);
  auto output = GetDeviceAddress<T>(outputs, kIndex0);
  ReductionType reduction = reduction_;
  // High precision.
  float middle_output[1] = {};
  if (input_post_weight_shape_ == input_label_shape_ && input_weight_shape_ == input_label_shape_) {
    auto task = [&input_logits, &input_label, &input_weight, &input_pos_weight, &output, &reduction, &middle_output](
                  size_t start, size_t end) {
      const auto template_zero = static_cast<T>(0);
      const auto template_one = static_cast<T>(1);
      for (size_t i = start; i < end; i++) {
        auto logits_value = input_logits[i];
        auto label_value = input_label[i];
        auto weight_value = input_weight[i];
        auto post_weight_value = input_pos_weight[i];
        T max_value = -logits_value;
        max_value = max_value > template_zero ? max_value : template_zero;
        const auto log_weight = (post_weight_value - template_one) * label_value + template_one;
        const auto log_exp_value = static_cast<T>(
          std::log(std::exp(static_cast<float>(-max_value)) + std::exp(static_cast<float>(-logits_value - max_value))));
        T loss = (template_one - label_value) * logits_value + log_weight * (log_exp_value + max_value);
        if (reduction == kNone) {
          output[i] = loss * weight_value;
        } else {
          middle_output[0] += static_cast<float>(loss * weight_value);
        }
      }
    };
    ParallelLaunchAutoSearch(task, input_size_, this, &parallel_search_info_, pool_);
  } else {
    MultipleBroadcastIterator multi_broadcast_iterator(
      {input_logits_shape_, input_label_shape_, input_weight_shape_, input_post_weight_shape_}, input_logits_shape_);
    auto task = [&multi_broadcast_iterator, &input_logits, &input_label, &input_weight, &input_pos_weight, &output,
                 &reduction, &middle_output](size_t start, size_t end) {
      const auto template_zero = static_cast<T>(0);
      const auto template_one = static_cast<T>(1);
      auto iter = multi_broadcast_iterator;
      iter.SetPos(start);
      for (size_t i = start; i < end; i++) {
        auto logits_value = input_logits[iter.GetInputPos(kIndex0)];
        auto label_value = input_label[iter.GetInputPos(kIndex1)];
        auto weight_value = input_weight[iter.GetInputPos(kIndex2)];
        auto post_weight_value = input_pos_weight[iter.GetInputPos(kIndex3)];
        T max_value = -logits_value;
        max_value = max_value > template_zero ? max_value : template_zero;
        const auto log_weight = (post_weight_value - template_one) * label_value + template_one;
        const auto log_exp_value = static_cast<T>(
          std::log(std::exp(static_cast<float>(-max_value)) + std::exp(static_cast<float>(-logits_value - max_value))));
        T loss = (template_one - label_value) * logits_value + log_weight * (log_exp_value + max_value);
        if (reduction == kNone) {
          output[i] = loss * weight_value;
        } else {
          middle_output[0] += static_cast<float>(loss * weight_value);
        }
        iter.GenNextPos();
      }
    };
    ParallelLaunchAutoSearch(task, input_size_, this, &parallel_search_info_, pool_);
  }
  if (reduction == kMean) {
    output[0] = static_cast<T>(middle_output[0] / static_cast<double>(input_size_));
  } else if (reduction == kSum) {
    output[0] = static_cast<T>(middle_output[0]);
  }
  return true;
}

void BCEWithLogitsLossCpuKernelMod::ResetResource() noexcept {
  input_label_shape_.clear();
  input_logits_shape_.clear();
  input_weight_shape_.clear();
  input_post_weight_shape_.clear();
  input_size_list_.clear();
  output_size_list_.clear();
  workspace_size_list_.clear();
}

const std::vector<std::pair<KernelAttr, BCEWithLogitsLossCpuKernelMod::KernelRunFunc>>
  &BCEWithLogitsLossCpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, BCEWithLogitsLossCpuKernelMod::KernelRunFunc>> func_list = {
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddOutputAttr(kNumberTypeFloat16),
     &BCEWithLogitsLossCpuKernelMod::LaunchKernel<float16>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeFloat32),
     &BCEWithLogitsLossCpuKernelMod::LaunchKernel<float>},
  };
  return func_list;
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, BCEWithLogitsLoss, BCEWithLogitsLossCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
