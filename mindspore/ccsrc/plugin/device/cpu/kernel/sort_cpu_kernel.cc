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
#include "plugin/device/cpu/kernel/sort_cpu_kernel.h"
#include <algorithm>
#include <utility>
#include "include/common/thread_pool.h"

namespace mindspore {
namespace kernel {
void SortCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  size_t input_count = common::AnfAlgo::GetInputTensorNum(kernel_node);
  if (input_count != 1) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of inputs must be 1, but got " << input_count
                      << " input(s).";
  }

  size_t output_count = common::AnfAlgo::GetOutputTensorNum(kernel_node);
  if (output_count != 2) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of outputs must be 2, but got " << output_count
                      << " output(s).";
  }

  auto input_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  descending_ = common::AnfAlgo::GetNodeAttr<bool>(kernel_node, "descending");
  auto axis = common::AnfAlgo::GetNodeAttr<int64_t>(kernel_node, AXIS);
  size_t axis_t = axis < 0 ? LongToSize(axis + SizeToLong(input_shape.size())) : LongToSize(axis);
  if (axis_t >= input_shape.size()) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the 'axis' must be less than the dimension of input tensor "
                      << input_shape.size() << "D, but got " << axis_t;
  }

  axisIterator_.Init(input_shape, axis_t);

  auto kernel_attr = GetKernelAttrFromNode(kernel_node);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(EXCEPTION) << "Sort does not support this kernel data type: " << kernel_attr;
  }
  kernel_func_ = func_list_[index].second;
}

void SortCpuKernelMod::InitInputOutputSize(const CNodePtr &kernel_node) {
  DeprecatedNativeCpuKernelMod::InitInputOutputSize(kernel_node);
  size_t element_size = axisIterator_.OuterSize() * axisIterator_.InnerSize() * axisIterator_.AxisSize();
  // id
  (void)workspace_size_list_.emplace_back((sizeof(size_t) * element_size));
}

template <typename T>
bool SortCpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                                    const std::vector<AddressPtr> &outputs) {
  if (inputs.size() != 1) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of inputs must be 1, but got " << inputs.size()
                      << " input(s).";
  }
  if (outputs.size() != 2) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of outputs must be 2, but got " << outputs.size()
                      << " output(s).";
  }
  if (inputs[0]->size != axisIterator_.OuterSize() * axisIterator_.AxisSize() * axisIterator_.InnerSize() * sizeof(T)) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the memory size of inputs error.";
  }
  auto input = reinterpret_cast<T *>(inputs[0]->addr);
  auto ids_addr = reinterpret_cast<size_t *>(workspace[0]->addr);
  auto output = reinterpret_cast<T *>(outputs[0]->addr);
  auto indices = reinterpret_cast<int *>(outputs[1]->addr);

  if (outputs[0]->size != inputs[0]->size) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the memory size of the first output must be equal to "
                         "the memory size of input, but got the memory size of the first output "
                      << outputs[0]->size << " and the memory size of input " << inputs[0]->size;
  }

  std::function<bool(size_t, size_t)> comparator;
  if (descending_) {
    comparator = [&input](size_t index_1, size_t index_2) { return input[index_1] > input[index_2]; };
  } else {
    comparator = [&input](size_t index_1, size_t index_2) { return input[index_1] < input[index_2]; };
  }

  auto task = [this, ids_addr, input, indices, output, &comparator](size_t start, size_t end) {
    size_t axis_size = axisIterator_.AxisSize();
    AxisIterator iter(axisIterator_);
    for (size_t index = start; index < end; index++) {
      iter.SetOffset(index);

      size_t offset = index * axis_size;
      size_t *idx = ids_addr + offset;
      for (size_t k = 0; k < axis_size; ++k) {
        idx[k] = iter.GetPos(k);
      }

      std::stable_sort(idx, idx + axis_size, comparator);

      for (size_t k = 0; k < axis_size; ++k) {
        const auto output_index = iter.GetPos(k);
        indices[output_index] = SizeToInt(iter.RevertPos(idx[k]));
        output[output_index] = input[idx[k]];
      }
    }
  };
  ParallelLaunchAutoSearch(task, axisIterator_.OuterSize() * axisIterator_.InnerSize(), this, &parallel_search_info_);

  return true;
}

std::vector<std::pair<KernelAttr, SortCpuKernelMod::SortFunc>> SortCpuKernelMod::func_list_ = {
  {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeInt32),
   &SortCpuKernelMod::LaunchKernel<float16>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeInt32),
   &SortCpuKernelMod::LaunchKernel<float>}};

std::vector<KernelAttr> SortCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, SortFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, Sort, SortCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
