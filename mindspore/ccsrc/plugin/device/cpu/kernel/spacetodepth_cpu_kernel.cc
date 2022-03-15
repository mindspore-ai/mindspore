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

#include "plugin/device/cpu/kernel/spacetodepth_cpu_kernel.h"
#include <algorithm>
#include <vector>
#include <utility>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kSpaceToDepthInputsNum = 1;
constexpr size_t kSpaceToDepthOutputsNum = 1;
constexpr size_t kSpaceToDepthInputShapeSize = 4;
constexpr size_t kSpaceToDepthMinBlockSize = 2;
}  // namespace
void SpaceToDepthCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);

  input_shape_ = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
  output_shape_ = AnfAlgo::GetOutputDeviceShape(kernel_node, 0);
  block_size_ = LongToSize(common::AnfAlgo::GetNodeAttr<int64_t>(kernel_node, "block_size"));
  if (input_shape_.size() != kSpaceToDepthInputShapeSize) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimension of input tensor should be 4-D, but got "
                      << input_shape_.size() << "-D";
  }
  if (block_size_ < kSpaceToDepthMinBlockSize) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the 'block_size' should be greater than or equal to "
                      << kSpaceToDepthMinBlockSize << ", but got " << block_size_;
  }

  auto kernel_attr = GetKernelAttrFromNode(kernel_node);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(EXCEPTION) << "SpaceToDepth does not support this kernel data type: " << kernel_attr;
  }
  kernel_func_ = func_list_[index].second;
}

template <typename T>
bool SpaceToDepthCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                            const std::vector<kernel::AddressPtr> &,
                                            const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kSpaceToDepthInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kSpaceToDepthOutputsNum, kernel_name_);
  auto input_addr = reinterpret_cast<T *>(inputs[0]->addr);
  auto output_addr = reinterpret_cast<T *>(outputs[0]->addr);
  size_t size = inputs[0]->size / sizeof(T);

  std::vector<size_t> input_shape = input_shape_;
  std::vector<size_t> output_shape = output_shape_;
  size_t block_size = block_size_;
  size_t input_dimension = input_shape.size();
  size_t input_strides[3] = {1, 1, 1};

  for (size_t i = input_dimension - 1; i >= 1; --i) {
    for (size_t j = 0; j < i; ++j) {
      input_strides[j] *= input_shape[i];
    }
  }

  auto task = [&, input_addr, output_addr](size_t start, size_t end) {
    std::vector<size_t> input_pos_array(input_dimension, 0);
    for (size_t i = start; i < end; ++i) {
      size_t tmp_pos = i;
      for (size_t j = 0; j < input_dimension - 1; ++j) {
        input_pos_array[j] = tmp_pos / input_strides[j];
        tmp_pos %= input_strides[j];
      }
      input_pos_array.back() = tmp_pos;
      size_t output_pos = input_pos_array[0];
      output_pos =
        (output_pos * output_shape[1]) +
        (input_pos_array[1] +
         (block_size * (input_pos_array[2] % block_size) + input_pos_array[3] % block_size) * input_shape[1]);
      output_pos = (output_pos * output_shape[2]) + (input_pos_array[2] / block_size);
      output_pos = (output_pos * output_shape[3]) + (input_pos_array[3] / block_size);
      output_addr[output_pos] = input_addr[i];
    }
  };

  ParallelLaunchAutoSearch(task, size, this, &parallel_search_info_);
  return true;
}

std::vector<std::pair<KernelAttr, SpaceToDepthCpuKernelMod::SpaceToDepthFunc>> SpaceToDepthCpuKernelMod::func_list_ = {
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
   &SpaceToDepthCpuKernelMod::LaunchKernel<float>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
   &SpaceToDepthCpuKernelMod::LaunchKernel<float16>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
   &SpaceToDepthCpuKernelMod::LaunchKernel<int8_t>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),
   &SpaceToDepthCpuKernelMod::LaunchKernel<int16_t>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
   &SpaceToDepthCpuKernelMod::LaunchKernel<int>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
   &SpaceToDepthCpuKernelMod::LaunchKernel<int64_t>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
   &SpaceToDepthCpuKernelMod::LaunchKernel<uint8_t>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16),
   &SpaceToDepthCpuKernelMod::LaunchKernel<uint16_t>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32),
   &SpaceToDepthCpuKernelMod::LaunchKernel<uint32_t>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt64),
   &SpaceToDepthCpuKernelMod::LaunchKernel<uint64_t>}};

std::vector<KernelAttr> SpaceToDepthCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                 [](const std::pair<KernelAttr, SpaceToDepthFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, SpaceToDepth, SpaceToDepthCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
