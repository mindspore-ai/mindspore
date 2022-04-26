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

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kBceWithLogitsLossInputsNum = 4;
constexpr size_t kBceWithLogitsLossOutputsNum = 1;
}  // namespace

BCEWithLogitsLossCpuKernelMod::~BCEWithLogitsLossCpuKernelMod() {
  workspace_size_list_.clear();
  input_size_list_.clear();
  output_size_list_.clear();
}

template <typename T>
bool BCEWithLogitsLossCpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                                 const std::vector<AddressPtr> &workspace,
                                                 const std::vector<AddressPtr> &outputs) {
  const auto input_logits = reinterpret_cast<T *>(inputs.at(kIndex0)->addr);
  const auto input_label = reinterpret_cast<T *>(inputs.at(kIndex1)->addr);
  const auto input_weight = reinterpret_cast<T *>(inputs.at(kIndex2)->addr);
  const auto input_pos_weight = reinterpret_cast<T *>(inputs.at(kIndex3)->addr);
  auto output = reinterpret_cast<T *>(outputs.at(kIndex0)->addr);
  ReductionType reduction = reduction_;
  double middle_output[1] = {};
  if (input_post_weight_shape_ == input_label_shape_ && input_weight_shape_ == input_label_shape_) {
    auto task = [&input_logits, &input_label, &input_weight, &input_pos_weight, &output, &reduction, &middle_output](
                  size_t start, size_t end) {
      const auto template_zero = static_cast<double>(0);
      const auto template_one = static_cast<double>(1);
      for (size_t i = start; i < end; i++) {
        auto logits_value = static_cast<double>(input_logits[i]);
        auto label_value = static_cast<double>(input_label[i]);
        auto weight_value = static_cast<double>(input_weight[i]);
        auto post_weight_value = static_cast<double>(input_pos_weight[i]);
        double max_value = -logits_value;
        max_value = max_value > template_zero ? max_value : template_zero;
        const auto log_weight = (post_weight_value - template_one) * label_value + template_one;
        double loss = (template_one - label_value) * logits_value +
                      log_weight * (std::log(std::exp(-max_value) + std::exp(-logits_value - max_value)) + max_value);
        if (reduction == kNone) {
          output[i] = static_cast<T>(loss * weight_value);
        } else {
          middle_output[0] += loss * weight_value;
        }
      }
    };
    ParallelLaunchAutoSearch(task, input_size_, this, &parallel_search_info_);
  } else {
    MultipleBroadcastIterator multi_broadcast_iterator(
      {input_logits_shape_, input_label_shape_, input_weight_shape_, input_post_weight_shape_}, input_logits_shape_);
    auto task = [&multi_broadcast_iterator, &input_logits, &input_label, &input_weight, &input_pos_weight, &output,
                 &reduction, &middle_output](size_t start, size_t end) {
      const auto template_zero = static_cast<double>(0);
      const auto template_one = static_cast<double>(1);
      auto iter = multi_broadcast_iterator;
      iter.SetPos(start);
      for (size_t i = start; i < end; i++) {
        auto logits_value = static_cast<double>(input_logits[iter.GetInputPos(kIndex0)]);
        auto label_value = static_cast<double>(input_label[iter.GetInputPos(kIndex1)]);
        auto weight_value = static_cast<double>(input_weight[iter.GetInputPos(kIndex2)]);
        auto post_weight_value = static_cast<double>(input_pos_weight[iter.GetInputPos(kIndex3)]);
        double max_value = -logits_value;
        max_value = max_value > template_zero ? max_value : template_zero;
        const auto log_weight = (post_weight_value - template_one) * label_value + template_one;
        double loss = (template_one - label_value) * logits_value +
                      log_weight * (std::log(std::exp(-max_value) + std::exp(-logits_value - max_value)) + max_value);
        if (reduction == kNone) {
          output[i] = static_cast<T>(loss * weight_value);
        } else {
          middle_output[0] += loss * weight_value;
        }
        iter.GenNextPos();
      }
    };
    ParallelLaunchAutoSearch(task, input_size_, this, &parallel_search_info_);
  }
  if (reduction == kMean) {
    output[0] = static_cast<T>(middle_output[0] / static_cast<double>(input_size_));
  } else if (reduction == kSum) {
    output[0] = static_cast<T>(middle_output[0]);
  }
  return true;
}

void BCEWithLogitsLossCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
  CHECK_KERNEL_INPUTS_NUM(input_num, kBceWithLogitsLossInputsNum, kernel_name_);
  size_t output_num = common::AnfAlgo::GetOutputTensorNum(kernel_node);
  CHECK_KERNEL_OUTPUTS_NUM(output_num, kBceWithLogitsLossOutputsNum, kernel_name_);
  auto kernel_attr = GetKernelAttrFromNode(kernel_node);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(EXCEPTION) << kernel_name_ << " does not support this kernel data type: " << kernel_attr;
  }
  kernel_func_ = func_list_[index].second;
  input_data_type_ = AnfAlgo::GetInputDeviceDataType(kernel_node, kIndex0);
  input_logits_shape_ = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, kIndex0);
  input_label_shape_ = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, kIndex1);
  input_weight_shape_ = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, kIndex2);
  input_post_weight_shape_ = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, kIndex3);
  for (const auto &shape : input_logits_shape_) {
    input_size_ *= shape;
  }
  const auto reduction = common::AnfAlgo::GetNodeAttr<string>(kernel_node, REDUCTION);
  if (reduction == NONE) {
    reduction_ = kNone;
  } else if (reduction == MEAN) {
    reduction_ = kMean;
  } else if (reduction == SUM) {
    reduction_ = kSum;
  } else {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the 'reduction' must be 'none', 'mean', or 'sum', but got "
                      << reduction;
  }

  size_t unit_size = GetTypeByte(TypeIdToType(input_data_type_));
  size_t input_byte_size = input_size_ * unit_size;
  input_size_list_.emplace_back(input_byte_size);
  input_size_list_.emplace_back(input_byte_size);
  input_size_list_.emplace_back(input_byte_size);
  if (reduction_ == kNone) {
    // The output is a Tensor in ReductionType none.
    output_size_list_.emplace_back(input_byte_size);
  } else {
    // The output is a scalar in ReductionType mean or sum.
    output_size_list_.emplace_back(unit_size);
  }
}

std::vector<std::pair<KernelAttr, BCEWithLogitsLossCpuKernelMod::BceFunc>> BCEWithLogitsLossCpuKernelMod::func_list_ = {
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
   &BCEWithLogitsLossCpuKernelMod::LaunchKernel<float>}};
std::vector<KernelAttr> BCEWithLogitsLossCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, BceFunc> &pair) { return pair.first; });
  return support_list;
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, BCEWithLogitsLoss, BCEWithLogitsLossCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
