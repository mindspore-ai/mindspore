/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/kernel/searchsorted_cpu_kernel.h"

#include <vector>
#include <numeric>
#include <functional>
#include <algorithm>
#include <utility>

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kSearchSortedInputsNum = 2;
constexpr size_t kSearchSortedOutputsNum = 1;
}  // namespace

void SearchSortedCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  right_ = common::AnfAlgo::GetNodeAttr<bool>(kernel_node, "right");
  sequence_shape_ = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
  values_shape_ = AnfAlgo::GetInputDeviceShape(kernel_node, 1);
  search_len = sequence_shape_.back();

  auto kernel_attr = GetKernelAttrFromNode(kernel_node);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(EXCEPTION) << "SearchSorted does not support this kernel data type: " << kernel_attr;
  }
  kernel_func_ = func_list_[index].second;
}

template <typename S>
const S *SearchSortedCpuKernelMod::CustomizedLowerBound(const S *seq_start, const S *seq_end, const S key) {
  while (seq_start < seq_end) {
    const S *mid = seq_start + ((seq_end - seq_start) / 2);
    if (!(key <= *mid)) {
      seq_start = mid + 1;
    } else {
      seq_end = mid;
    }
  }
  return seq_start;
}

template <typename S, typename T>
bool SearchSortedCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                            const std::vector<kernel::AddressPtr> &outputs) {
  CheckParam<S, T>(inputs, outputs);
  auto sequence = reinterpret_cast<S *>(inputs[0]->addr);
  auto values = reinterpret_cast<S *>(inputs[1]->addr);
  auto output = reinterpret_cast<T *>(outputs[0]->addr);
  size_t elem_num = inputs[1]->size / sizeof(S);
  size_t seq_dim = sequence_shape_.size();
  size_t search_repeat = values_shape_.back();

  auto task = [this, &sequence, &values, &output, seq_dim, search_repeat](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      auto seq_start = (seq_dim == 1) ? sequence : sequence + (i / search_repeat) * search_len;
      auto result = right_ ? std::upper_bound(seq_start, seq_start + search_len, values[i]) - seq_start
                           : CustomizedLowerBound<S>(seq_start, seq_start + search_len, values[i]) - seq_start;
      output[i] = static_cast<T>(result);
    }
  };
  ParallelLaunchAutoSearch(task, elem_num, this, &parallel_search_info_);
  return true;
}

template <typename S, typename T>
void SearchSortedCpuKernelMod::CheckParam(const std::vector<AddressPtr> &inputs,
                                          const std::vector<AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kSearchSortedInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kSearchSortedOutputsNum, kernel_name_);

  if (outputs[0]->size / sizeof(T) != inputs[1]->size / sizeof(S)) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the dimension of `v` and output should be equal, but got the dimension of `v` "
                      << inputs[1]->size << " and the dimension of output " << outputs[0]->size;
  }

  auto sequence = reinterpret_cast<S *>(inputs[0]->addr);
  int list_count = accumulate(sequence_shape_.begin(), sequence_shape_.end() - 1, 1, std::multiplies<int>());
  auto task = [this, &sequence](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      for (size_t j = 0; j < search_len - 1; j++) {
        if (sequence[i * search_len + j] > sequence[i * search_len + j + 1]) {
          MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the input sequence should be forward sequence. But got "
                            << sequence[i * search_len + j] << '>' << sequence[i * search_len + j + 1];
        }
      }
    }
  };
  ParallelLaunchAutoSearch(task, IntToSize(list_count), this, &parallel_search_info_);
}

std::vector<std::pair<KernelAttr, SearchSortedCpuKernelMod::SearchSortedFunc>> SearchSortedCpuKernelMod::func_list_ = {
  {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeInt32),
   &SearchSortedCpuKernelMod::LaunchKernel<double, int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeInt32),
   &SearchSortedCpuKernelMod::LaunchKernel<float, int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt32),
   &SearchSortedCpuKernelMod::LaunchKernel<int64_t, int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
   &SearchSortedCpuKernelMod::LaunchKernel<int32_t, int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt16).AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt32),
   &SearchSortedCpuKernelMod::LaunchKernel<int16_t, int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt32),
   &SearchSortedCpuKernelMod::LaunchKernel<int8_t, int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeInt64),
   &SearchSortedCpuKernelMod::LaunchKernel<double, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeInt64),
   &SearchSortedCpuKernelMod::LaunchKernel<float, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
   &SearchSortedCpuKernelMod::LaunchKernel<int64_t, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt64),
   &SearchSortedCpuKernelMod::LaunchKernel<int32_t, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt16).AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt64),
   &SearchSortedCpuKernelMod::LaunchKernel<int16_t, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt64),
   &SearchSortedCpuKernelMod::LaunchKernel<int8_t, int64_t>}};

std::vector<KernelAttr> SearchSortedCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                 [](const std::pair<KernelAttr, SearchSortedFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, SearchSorted, SearchSortedCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
