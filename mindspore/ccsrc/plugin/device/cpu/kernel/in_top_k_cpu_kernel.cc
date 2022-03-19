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

#include "plugin/device/cpu/kernel/in_top_k_cpu_kernel.h"
#include <algorithm>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "include/common/thread_pool.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kInTopKInputsNum = 2;
constexpr size_t kInTopKOutputsNum = 1;
constexpr size_t kInTopKShapeRank = 2;
constexpr size_t kInTopkTargetShapeSize = 1;
}  // namespace

void InTopKCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  auto prediction_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  auto target_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
  if (prediction_shape.size() != kInTopKShapeRank) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the rank of the first input should be equal to "
                      << kInTopKShapeRank << ", but got " << prediction_shape.size();
  }
  if (target_shape.size() != kInTopkTargetShapeSize || target_shape[0] != prediction_shape[0]) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the rank of the second input should be equal to "
                      << kInTopkTargetShapeSize << ", and the shape size should be euqal to " << prediction_shape[0]
                      << ", but got " << target_shape;
  }
  outer_size_ = 1;
  for (size_t i = 0; i < prediction_shape.size() - 1; ++i) {
    outer_size_ *= prediction_shape[i];
  }
  inner_size_ = prediction_shape[prediction_shape.size() - 1];
  k_ = common::AnfAlgo::GetNodeAttr<int64_t>(kernel_node, "k");

  auto kernel_attr = GetKernelAttrFromNode(kernel_node);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(EXCEPTION) << "InTopK does not support this kernel data type: " << kernel_attr;
  }
  kernel_func_ = func_list_[index].second;
}

template <typename T>
bool InTopKCpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kInTopKInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kInTopKOutputsNum, kernel_name_);

  auto predictions = reinterpret_cast<T *>(inputs[0]->addr);
  auto targets = reinterpret_cast<int32_t *>(inputs[1]->addr);
  auto output = reinterpret_cast<bool *>(outputs[0]->addr);

  if (k_ < 1) {
    MS_LOG(WARNING) << "For '" << kernel_name_ << "', the 'k' should be greater than 0, but got " << k_;
    auto ret = memset_s(output, outputs[0]->size, 0, outputs[0]->size);
    if (ret != EOK) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', memset output failed. Error no: " << ret;
    }
    return true;
  }

  size_t k_num = std::min<size_t>(inner_size_ - 1, LongToSize(k_ - 1));
  const std::function<bool(size_t, size_t)> comparator = [predictions](size_t index_1, size_t index_2) {
    return predictions[index_1] > predictions[index_2];
  };

  auto task = [&](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      std::vector<size_t> idx(inner_size_);
      auto base_input = i * inner_size_;
      std::iota(idx.begin(), idx.end(), base_input);
      std::nth_element(idx.begin(), idx.begin() + SizeToLong(k_num), idx.end(), comparator);
      size_t p_index = base_input + IntToSize(targets[i]);
      output[i] = (predictions[p_index] >= predictions[idx[k_num]]) ? true : false;
    }
  };
  ParallelLaunchAutoSearch(task, outer_size_, this, &parallel_search_info_);
  return true;
}

std::vector<std::pair<KernelAttr, InTopKCpuKernelMod::InTopKFunc>> InTopKCpuKernelMod::func_list_ = {
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeBool),
   &InTopKCpuKernelMod::LaunchKernel<float>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeBool),
   &InTopKCpuKernelMod::LaunchKernel<float16>}};

std::vector<KernelAttr> InTopKCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, InTopKFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, InTopK, InTopKCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
