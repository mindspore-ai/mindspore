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
#include "kernel/cpu/native/sort_cpu_kernel.h"
#include <algorithm>
#include <utility>
#include <map>
#include <memory>
#include "include/utils/thread_pool.h"
#include "mindspore/ops/infer/sort.h"

namespace mindspore {
namespace kernel {
namespace sort_cpu {
constexpr int kSortInputsNum = 1;
constexpr int kSortOutputsNum = 2;

bool SortCpuKernelMod::Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kSortInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kSortOutputsNum, kernel_name_);

  descending_ = GetValue<bool>(primitive_->GetAttr(ops::kDescending));
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(EXCEPTION) << "Sort does not support this kernel data type: " << kernel_attr;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

template <typename T>
bool SortCpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                    const std::vector<KernelTensor *> &workspace,
                                    const std::vector<KernelTensor *> &outputs) {
  if (inputs.size() != 1) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of inputs must be 1, but got " << inputs.size()
                      << " input(s).";
  }
  if (outputs.size() != 2) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of outputs must be 2, but got " << outputs.size()
                      << " output(s).";
  }
  if (inputs[0]->size() !=
      axisIterator_.OuterSize() * axisIterator_.AxisSize() * axisIterator_.InnerSize() * sizeof(T)) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the memory size of inputs error.";
  }
  auto input = GetDeviceAddress<T>(inputs, kIndex0);
  auto ids_addr = GetDeviceAddress<size_t>(workspace, kIndex0);
  auto output = GetDeviceAddress<T>(outputs, kIndex0);
  auto indices = GetDeviceAddress<int>(outputs, kIndex1);

  if (outputs[0]->size() != inputs[0]->size()) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the memory size of the first output must be equal to "
                         "the memory size of input, but got the memory size of the first output "
                      << outputs[0]->size() << " and the memory size of input " << inputs[0]->size();
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

int SortCpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  auto ret = KernelMod::Resize(inputs, outputs);
  if (ret != KRET_OK) {
    return ret;
  }
  auto input_shape = inputs[0]->GetShapeVector();
  auto axis = GetValue<int64_t>(primitive_->GetAttr(ops::kAxis));
  size_t axis_t = axis < 0 ? LongToSize(axis + SizeToLong(input_shape.size())) : LongToSize(axis);
  if (axis_t >= input_shape.size()) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the 'axis' must be less than the dimension of input tensor "
                      << input_shape.size() << "D, but got " << axis_t;
  }

  axisIterator_.Init(input_shape, axis_t);
  size_t element_size = axisIterator_.OuterSize() * axisIterator_.InnerSize() * axisIterator_.AxisSize();
  (void)workspace_size_list_.emplace_back((sizeof(size_t) * element_size));
  return KRET_OK;
}

std::vector<std::pair<KernelAttr, SortCpuKernelMod::SortFunc>> SortCpuKernelMod::func_list_ = {
  {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeInt32),
   &SortCpuKernelMod::LaunchKernel<float16>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeInt32),
   &SortCpuKernelMod::LaunchKernel<float>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeInt32),
   &SortCpuKernelMod::LaunchKernel<uint8_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt32),
   &SortCpuKernelMod::LaunchKernel<int8_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt32),
   &SortCpuKernelMod::LaunchKernel<int16_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
   &SortCpuKernelMod::LaunchKernel<int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt32),
   &SortCpuKernelMod::LaunchKernel<int64_t>}};

std::vector<KernelAttr> SortCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, SortFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, Sort, SortCpuKernelMod);
}  // namespace sort_cpu
}  // namespace kernel
}  // namespace mindspore
