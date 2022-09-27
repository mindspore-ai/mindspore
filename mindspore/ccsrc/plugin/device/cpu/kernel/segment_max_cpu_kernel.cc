/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include <algorithm>
#include "plugin/device/cpu/kernel/segment_max_cpu_kernel.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace {
const size_t kSegmentsThreshold = 2 * 1024;
const size_t kDataSizeThreshold = 2 * 1024;
}  // namespace

namespace mindspore {
namespace kernel {
void SegmentMaxCPUKernelMod::InitKernel(const CNodePtr &kernel_node) {
  input_x_shape_ = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
  segment_ids_shape_ = AnfAlgo::GetInputDeviceShape(kernel_node, 1);
  output_shape_ = AnfAlgo::GetOutputDeviceShape(kernel_node, 0);
  output_dtype_ = AnfAlgo::GetOutputDeviceDataType(kernel_node, 0);
  input_x_num_ = SizeOf(input_x_shape_);
  segment_ids_num_ = SizeOf(segment_ids_shape_);
  output_num_ = SizeOf(output_shape_);
  auto kernel_attr = GetKernelAttrFromNode(kernel_node);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(EXCEPTION) << "SegmentMax does not support this kernel data type: " << kernel_attr;
  }
  kernel_func_ = func_list_[index].second;
}

std::vector<std::pair<KernelAttr, SegmentMaxCPUKernelMod::SegmentMaxFunc>> SegmentMaxCPUKernelMod::func_list_ = {
  {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat16),
   &SegmentMaxCPUKernelMod::LaunchKernel<float16, int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat32),
   &SegmentMaxCPUKernelMod::LaunchKernel<float, int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat64),
   &SegmentMaxCPUKernelMod::LaunchKernel<double, int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt8),
   &SegmentMaxCPUKernelMod::LaunchKernel<int8_t, int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt16).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt16),
   &SegmentMaxCPUKernelMod::LaunchKernel<int16_t, int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
   &SegmentMaxCPUKernelMod::LaunchKernel<int32_t, int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt64),
   &SegmentMaxCPUKernelMod::LaunchKernel<int64_t, int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt8),
   &SegmentMaxCPUKernelMod::LaunchKernel<uint8_t, int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt16).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt16),
   &SegmentMaxCPUKernelMod::LaunchKernel<uint16_t, int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt32),
   &SegmentMaxCPUKernelMod::LaunchKernel<uint32_t, int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt64).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt64),
   &SegmentMaxCPUKernelMod::LaunchKernel<uint64_t, int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat16),
   &SegmentMaxCPUKernelMod::LaunchKernel<float16, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat32),
   &SegmentMaxCPUKernelMod::LaunchKernel<float, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat64),
   &SegmentMaxCPUKernelMod::LaunchKernel<double, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt8),
   &SegmentMaxCPUKernelMod::LaunchKernel<int8_t, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt16).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt16),
   &SegmentMaxCPUKernelMod::LaunchKernel<int16_t, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt32),
   &SegmentMaxCPUKernelMod::LaunchKernel<int32_t, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
   &SegmentMaxCPUKernelMod::LaunchKernel<int64_t, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeUInt8),
   &SegmentMaxCPUKernelMod::LaunchKernel<uint8_t, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt16).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeUInt16),
   &SegmentMaxCPUKernelMod::LaunchKernel<uint16_t, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt32).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeUInt32),
   &SegmentMaxCPUKernelMod::LaunchKernel<uint32_t, int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeUInt64),
   &SegmentMaxCPUKernelMod::LaunchKernel<uint64_t, int64_t>}};

std::vector<KernelAttr> SegmentMaxCPUKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, SegmentMaxFunc> &pair) { return pair.first; });
  return support_list;
}

template <typename T1, typename T2>
bool SegmentMaxCPUKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                          const std::vector<kernel::AddressPtr> &,
                                          const std::vector<kernel::AddressPtr> &outputs) {
  auto input_x_data_addr = static_cast<T1 *>(inputs[0]->addr);
  auto segment_ids_data_addr = static_cast<T2 *>(inputs[1]->addr);
  auto output_data_addr = static_cast<T1 *>(outputs[0]->addr);
  std::vector<int64_t> segments = CPUKernelUtils::CalcSegmentIds(segment_ids_data_addr, segment_ids_num_);
  for (size_t i = 0; i < output_num_; ++i) {
    output_data_addr[i] = static_cast<T1>(0);
  }
  if (input_x_shape_[0] == 0) {
    MS_LOG(EXCEPTION) << "For SegmentMaxCPUKernelMod, input_x_shape_[0] can not be 0";
  }
  const size_t num_compare_per = input_x_num_ / LongToSize(input_x_shape_[0]);
  const size_t num_segments = segments.size();
  if (num_segments < kSegmentsThreshold) {
    for (size_t i = 0; i < num_segments; ++i) {
      const size_t count = static_cast<size_t>(segments[i]);
      int64_t count_no = 0;
      for (size_t j = 0; j < i; ++j) {
        count_no += segments[j];
      }
      size_t input_addr_base = LongToSize(count_no) * num_compare_per;
      auto task = [&](size_t start, size_t end) {
        for (size_t j = start; j < end; ++j) {
          size_t max_init_addr = input_addr_base + j;
          T1 max_value = input_x_data_addr[max_init_addr];
          for (size_t k = 1; k < count; ++k) {
            int cmp_addr = max_init_addr + k * num_compare_per;
            if (max_value < input_x_data_addr[cmp_addr]) {
              max_value = input_x_data_addr[cmp_addr];
            }
          }
          output_data_addr[segment_ids_data_addr[LongToSize(count_no)] * num_compare_per + j] = max_value;
        }
      };
      if (num_compare_per < kDataSizeThreshold) {
        task(0, num_compare_per);
      } else {
        CPUKernelUtils::ParallelFor(task, num_compare_per);
      }
    }
  } else {
    auto task = [&](size_t start, size_t end) {
      for (size_t i = start; i < end; ++i) {
        const size_t count = static_cast<size_t>(segments[i]);
        int64_t count_no = 0;
        for (size_t j = 0; j < i; ++j) {
          count_no += segments[j];
        }
        size_t input_addr_base = LongToSize(count_no) * num_compare_per;
        for (size_t j = 0; j < num_compare_per; ++j) {
          size_t max_init_addr = input_addr_base + j;
          T1 max_value = input_x_data_addr[max_init_addr];
          for (size_t k = 1; k < count; ++k) {
            int cmp_addr = max_init_addr + k * num_compare_per;
            if (max_value < input_x_data_addr[cmp_addr]) {
              max_value = input_x_data_addr[cmp_addr];
            }
          }
          output_data_addr[segment_ids_data_addr[LongToSize(count_no)] * num_compare_per + j] = max_value;
        }
      }
    };
    CPUKernelUtils::ParallelFor(task, num_segments);
  }
  return true;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, SegmentMax, SegmentMaxCPUKernelMod);
}  // namespace kernel
}  // namespace mindspore
