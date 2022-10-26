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
 * once_compute_thread_sizetributed under the License is
 * once_compute_thread_sizetributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
 * CONDITIONS OF ANY KIND, either express or implied. See the License for the
 * specific language governing permissions and limitations under the License.
 */

#include "plugin/device/cpu/kernel/sparse_segment_sum_cpu_kernel.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kSparseSegmentSumInputsNum = 3;
constexpr size_t kSparseSegmentSumOutputsNum = 1;

#define ADD_KERNEL(t1, t2, t3, t4) \
  KernelAttr()                     \
    .AddInputAttr(kNumberType##t1) \
    .AddInputAttr(kNumberType##t2) \
    .AddInputAttr(kNumberType##t3) \
    .AddOutputAttr(kNumberType##t4)
}  // namespace

void SparseSegmentSumCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  CheckParam(kernel_node);
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  x_shape_ = AnfAlgo::GetInputDeviceShape(kernel_node, kIndex0);
  indices_dtype_ = AnfAlgo::GetInputDeviceDataType(kernel_node, kIndex1);
  segment_shape_ = AnfAlgo::GetInputDeviceShape(kernel_node, kIndex2);
  x_dtype_ = AnfAlgo::GetInputDeviceDataType(kernel_node, kIndex0);
}

bool SparseSegmentSumCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                          const std::vector<kernel::AddressPtr> &,
                                          const std::vector<kernel::AddressPtr> &outputs) {
  switch (x_dtype_) {
    case (kNumberTypeInt8):
      if (indices_dtype_ == kNumberTypeInt32) {
        LaunchKernel<int8_t, int32_t>(inputs, outputs);
        break;
      } else {
        LaunchKernel<int8_t, int64_t>(inputs, outputs);
        break;
      }
    case (kNumberTypeInt16):
      if (indices_dtype_ == kNumberTypeInt32) {
        LaunchKernel<int16_t, int32_t>(inputs, outputs);
        break;
      } else {
        LaunchKernel<int16_t, int64_t>(inputs, outputs);
        break;
      }
    case (kNumberTypeInt32):
      if (indices_dtype_ == kNumberTypeInt32) {
        LaunchKernel<int32_t, int32_t>(inputs, outputs);
        break;
      } else {
        LaunchKernel<int32_t, int64_t>(inputs, outputs);
        break;
      }
    case (kNumberTypeInt64):
      if (indices_dtype_ == kNumberTypeInt32) {
        LaunchKernel<int64_t, int32_t>(inputs, outputs);
        break;
      } else {
        LaunchKernel<int64_t, int64_t>(inputs, outputs);
        break;
      }
    case (kNumberTypeUInt8):
      if (indices_dtype_ == kNumberTypeInt32) {
        LaunchKernel<uint8_t, int32_t>(inputs, outputs);
        break;
      } else {
        LaunchKernel<uint8_t, int64_t>(inputs, outputs);
        break;
      }
    case (kNumberTypeUInt16):
      if (indices_dtype_ == kNumberTypeInt32) {
        LaunchKernel<uint16_t, int32_t>(inputs, outputs);
        break;
      } else {
        LaunchKernel<uint16_t, int64_t>(inputs, outputs);
        break;
      }
    case (kNumberTypeFloat16):
      if (indices_dtype_ == kNumberTypeInt32) {
        LaunchKernel<float16, int32_t>(inputs, outputs);
        break;
      } else {
        LaunchKernel<float16, int64_t>(inputs, outputs);
        break;
      }
    case (kNumberTypeFloat32):
      if (indices_dtype_ == kNumberTypeInt32) {
        LaunchKernel<float, int32_t>(inputs, outputs);
        break;
      } else {
        LaunchKernel<float, int64_t>(inputs, outputs);
        break;
      }
    case (kNumberTypeFloat64):
      if (indices_dtype_ == kNumberTypeInt32) {
        LaunchKernel<double, int32_t>(inputs, outputs);
        break;
      } else {
        LaunchKernel<double, int64_t>(inputs, outputs);
        break;
      }
    default:
      MS_EXCEPTION(TypeError) << "For '" << kernel_name_ << "', data type of x is " << TypeIdLabel(x_dtype_)
                              << " which is not supported.";
  }
  return true;
}

template <typename T1, typename T2>
void SparseSegmentSumCpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                                const std::vector<AddressPtr> &outputs) {
  constexpr size_t kMultiply = 1;
  size_t n = std::accumulate(x_shape_.begin(), x_shape_.end(), kMultiply, std::multiplies<int>()) / x_shape_[kIndex0];
  size_t m = std::accumulate(segment_shape_.begin(), segment_shape_.end(), kMultiply, std::multiplies<int>());
  auto x_shape0 = static_cast<T2>(x_shape_[kIndex0]);
  auto dataptr = reinterpret_cast<T1 *>(inputs[kIndex0]->addr);
  auto indicesptr = reinterpret_cast<T2 *>(inputs[kIndex1]->addr);
  auto segment_idsptr = reinterpret_cast<T2 *>(inputs[kIndex2]->addr);
  auto yptr = reinterpret_cast<T1 *>(outputs[kIndex0]->addr);
  if (segment_idsptr[0] != 0) {
    MS_EXCEPTION(ValueError) << "For '" << kernel_name_ << "', indices should start from 0.";
  }
  for (size_t i = 1; i < m; i++) {
    if (segment_idsptr[i] < segment_idsptr[i - 1]) {
      MS_EXCEPTION(ValueError) << "For '" << kernel_name_ << "', input segment_ids should be sorted.";
    }
    if (segment_idsptr[i] - segment_idsptr[i - 1] > 1) {
      MS_EXCEPTION(ValueError) << "For '" << kernel_name_ << "', indices in 'segment_ids' should be contiguous.";
    }
  }
  for (size_t i = 0; i < m; i++) {
    if (indicesptr[i] >= x_shape0) {
      MS_EXCEPTION(ValueError) << "For '" << kernel_name_ << "', input indices is out of range of x's first dimension.";
    }
  }
  int oldindex = -1;
  for (size_t i = 0; i < m; i++) {
    if (oldindex != segment_idsptr[i]) {
      oldindex = segment_idsptr[i];
      for (size_t j = 0; j < n; j++) {
        yptr[j + oldindex * n] = (T1)0;
      }
    }
    for (size_t j = 0; j < n; j++) {
      yptr[j + oldindex * n] += dataptr[j + indicesptr[i] * n];
    }
  }
}

void SparseSegmentSumCpuKernelMod::CheckParam(const CNodePtr &kernel_node) {
  size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
  CHECK_KERNEL_INPUTS_NUM(input_num, kSparseSegmentSumInputsNum, kernel_name_);
  size_t output_num = common::AnfAlgo::GetOutputTensorNum(kernel_node);
  CHECK_KERNEL_OUTPUTS_NUM(output_num, kSparseSegmentSumOutputsNum, kernel_name_);
}

std::vector<KernelAttr> SparseSegmentSumCpuKernelMod::GetOpSupport() {
  static std::vector<KernelAttr> kernel_attr_list = {
    ADD_KERNEL(Int8, Int32, Int32, Int8),       ADD_KERNEL(Float16, Int32, Int32, Float16),
    ADD_KERNEL(Int16, Int32, Int32, Int16),     ADD_KERNEL(Float32, Int32, Int32, Float32),
    ADD_KERNEL(Int32, Int32, Int32, Int32),     ADD_KERNEL(Float64, Int32, Int32, Float64),
    ADD_KERNEL(Int64, Int32, Int32, Int64),     ADD_KERNEL(UInt8, Int32, Int32, UInt8),
    ADD_KERNEL(UInt16, Int32, Int32, UInt16),   ADD_KERNEL(Int8, Int64, Int64, Int8),
    ADD_KERNEL(Float16, Int64, Int64, Float16), ADD_KERNEL(Int16, Int64, Int64, Int16),
    ADD_KERNEL(Float32, Int64, Int64, Float32), ADD_KERNEL(Int32, Int64, Int64, Int32),
    ADD_KERNEL(Float64, Int64, Int64, Float64), ADD_KERNEL(Int64, Int64, Int64, Int64),
    ADD_KERNEL(UInt8, Int64, Int64, UInt8),     ADD_KERNEL(UInt16, Int64, Int64, UInt16)};
  return kernel_attr_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, SparseSegmentSum, SparseSegmentSumCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
