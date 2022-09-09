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

#include "plugin/device/cpu/kernel/sparse_segment_sqrt_n_with_num_segments_cpu_kernel.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kSparseSegmentSqrtNWithNumSegmentsInputsNum = 4;
constexpr size_t kSparseSegmentSqrtNWithNumSegmentsOutputsNum = 1;

#define ADD_KERNEL(t1, t2, t3, t4, t5) \
  KernelAttr()                         \
    .AddInputAttr(kNumberType##t1)     \
    .AddInputAttr(kNumberType##t2)     \
    .AddInputAttr(kNumberType##t3)     \
    .AddInputAttr(kNumberType##t4)     \
    .AddOutputAttr(kNumberType##t5)
}  // namespace

void SparseSegmentSqrtNWithNumSegmentsCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
  CHECK_KERNEL_INPUTS_NUM(input_num, kSparseSegmentSqrtNWithNumSegmentsInputsNum, kernel_name_);
  size_t output_num = common::AnfAlgo::GetOutputTensorNum(kernel_node);
  CHECK_KERNEL_OUTPUTS_NUM(output_num, kSparseSegmentSqrtNWithNumSegmentsOutputsNum, kernel_name_);
  xdtype_ = AnfAlgo::GetInputDeviceDataType(kernel_node, kIndex0);
  dtype1_ = AnfAlgo::GetInputDeviceDataType(kernel_node, kIndex1);
  x_shape_ = AnfAlgo::GetInputDeviceShape(kernel_node, kIndex0);
  indices_shape_ = AnfAlgo::GetInputDeviceShape(kernel_node, kIndex1);
  segment_ids_shape_ = AnfAlgo::GetInputDeviceShape(kernel_node, kIndex2);
  num_segments_shape_ = AnfAlgo::GetInputDeviceShape(kernel_node, kIndex3);
  y_shape_ = AnfAlgo::GetOutputDeviceShape(kernel_node, kIndex0);
}

bool SparseSegmentSqrtNWithNumSegmentsCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                                           const std::vector<kernel::AddressPtr> &,
                                                           const std::vector<kernel::AddressPtr> &outputs) {
  switch (xdtype_) {
    case (kNumberTypeFloat16):
      if (dtype1_ == kNumberTypeInt32) {
        LaunchKernel<float16, int32_t>(inputs, outputs);
        break;
      } else if (dtype1_ == kNumberTypeInt64) {
        LaunchKernel<float16, int64_t>(inputs, outputs);
        break;
      } else {
        MS_EXCEPTION(TypeError) << "For '" << kernel_name_
                                << "', data type of indices, segment_ids and num_segments is " << TypeIdLabel(dtype1_)
                                << ", which is not supported.";
        break;
      }
    case (kNumberTypeFloat32):
      if (dtype1_ == kNumberTypeInt32) {
        LaunchKernel<float, int32_t>(inputs, outputs);
        break;
      } else if (dtype1_ == kNumberTypeInt64) {
        LaunchKernel<float, int64_t>(inputs, outputs);
        break;
      } else {
        MS_EXCEPTION(TypeError) << "For '" << kernel_name_
                                << "', data type of indices, segment_ids and num_segments is " << TypeIdLabel(dtype1_)
                                << ", which is not supported.";
        break;
      }
    case (kNumberTypeFloat64):
      if (dtype1_ == kNumberTypeInt32) {
        LaunchKernel<double, int32_t>(inputs, outputs);
        break;
      } else if (dtype1_ == kNumberTypeInt64) {
        LaunchKernel<double, int64_t>(inputs, outputs);
        break;
      } else {
        MS_EXCEPTION(TypeError) << "For '" << kernel_name_
                                << "', data type of indices, segment_ids and num_segments is " << TypeIdLabel(dtype1_)
                                << ", which is not supported.";
        break;
      }
    default:
      MS_EXCEPTION(TypeError) << "For '" << kernel_name_ << "', data type of x is " << TypeIdLabel(xdtype_)
                              << ", which is not supported.";
  }
  return true;
}

template <typename T1, typename T2>
void SparseSegmentSqrtNWithNumSegmentsCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                                                 const std::vector<kernel::AddressPtr> &outputs) {
  size_t n = static_cast<size_t>(
    std::accumulate(x_shape_.begin(), x_shape_.end(), kIndex1, std::multiplies<int64_t>()) / x_shape_[kIndex0]);
  size_t m = static_cast<size_t>(
    std::accumulate(segment_ids_shape_.begin(), segment_ids_shape_.end(), kIndex1, std::multiplies<int64_t>()));
  size_t k =
    static_cast<size_t>(std::accumulate(y_shape_.begin(), y_shape_.end(), kIndex1, std::multiplies<int64_t>()));
  auto x_shape_0 = static_cast<T2>(x_shape_[kIndex0]);
  auto x_addr = static_cast<T1 *>(inputs[kIndex0]->addr);
  auto indices_addr = static_cast<T2 *>(inputs[kIndex1]->addr);
  auto segment_ids_addr = static_cast<T2 *>(inputs[kIndex2]->addr);
  auto num_segments_addr = static_cast<T2 *>(inputs[kIndex3]->addr);
  auto y_addr = static_cast<T1 *>(outputs[kIndex0]->addr);

  for (size_t i = 0; i < k; i++) {
    y_addr[i] = static_cast<T1>(0);
  }
  for (size_t i = 1; i < m; i++) {
    if (segment_ids_addr[i] < segment_ids_addr[i - 1]) {
      MS_EXCEPTION(ValueError) << "For '" << kernel_name_ << "', segment_ids should be sorted.";
    }
  }
  if (segment_ids_addr[m - 1] >= num_segments_addr[kIndex0]) {
    MS_EXCEPTION(ValueError) << "For '" << kernel_name_
                             << "', num_segments must be bigger than the largest id of segment_ids.";
  }
  for (size_t i = 0; i < m; i++) {
    if (indices_addr[i] >= x_shape_0) {
      MS_EXCEPTION(ValueError) << "For '" << kernel_name_ << "', indices is out of range of x's first dimension.";
    }
  }

  int oldindex = -1;
  int countnum = 0;
  for (size_t i = 0; i < m; i++) {
    if (oldindex == static_cast<int>(segment_ids_addr[i])) {
      countnum++;
    } else {
      if (countnum != 0) {
        for (size_t j = 0; j < n; j++) {
          y_addr[j + static_cast<size_t>(oldindex) * n] /= static_cast<T1>(sqrt(countnum));
        }
      }
      countnum = 1;
      oldindex = static_cast<int>(segment_ids_addr[i]);
      for (size_t j = 0; j < n; j++) {
        y_addr[j + static_cast<size_t>(oldindex) * n] = static_cast<T1>(0);
      }
    }
    for (size_t j = 0; j < n; j++) {
      y_addr[j + static_cast<size_t>(oldindex) * n] += x_addr[j + static_cast<size_t>(indices_addr[i]) * n];
    }
  }
  if (countnum != 0) {
    for (size_t j = 0; j < n; j++) {
      y_addr[j + static_cast<size_t>(oldindex) * n] /= static_cast<T1>(sqrt(countnum));
    }
  }
}

std::vector<KernelAttr> SparseSegmentSqrtNWithNumSegmentsCpuKernelMod::GetOpSupport() {
  static std::vector<KernelAttr> kernel_attr_list = {
    ADD_KERNEL(Float16, Int32, Int32, Int32, Float16), ADD_KERNEL(Float16, Int64, Int64, Int64, Float16),
    ADD_KERNEL(Float32, Int32, Int32, Int32, Float32), ADD_KERNEL(Float32, Int64, Int64, Int64, Float32),
    ADD_KERNEL(Float64, Int32, Int32, Int32, Float64), ADD_KERNEL(Float64, Int64, Int64, Int64, Float64)};

  return kernel_attr_list;
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, SparseSegmentSqrtNWithNumSegments,
                      SparseSegmentSqrtNWithNumSegmentsCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
