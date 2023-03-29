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

#include "plugin/device/cpu/kernel/sparse_segment_sum_with_num_segments_cpu_kernel.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kInputsNum = 4;
constexpr size_t kOutputsNum = 1;

#define ADD_KERNEL(T1, T2, T3, T4, T5, T6, T7)                           \
  {                                                                      \
    KernelAttr()                                                         \
      .AddInputAttr(kNumberType##T1)                                     \
      .AddInputAttr(kNumberType##T2)                                     \
      .AddInputAttr(kNumberType##T3)                                     \
      .AddInputAttr(kNumberType##T4)                                     \
      .AddOutputAttr(kNumberType##T5),                                   \
      &SparseSegmentSumWithNumSegmentsCpuKernelMod::LaunchKernel<T6, T7> \
  }
}  // namespace

bool SparseSegmentSumWithNumSegmentsCpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                                       const std::vector<KernelTensorPtr> &inputs,
                                                       const std::vector<KernelTensorPtr> &outputs) {
  MS_ERROR_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kOutputsNum, kernel_name_);
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "The kernel '" << kernel_name_ << "' does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = f_list_[index].second;
  return true;
}

int SparseSegmentSumWithNumSegmentsCpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                                        const std::vector<KernelTensorPtr> &inputs,
                                                        const std::vector<KernelTensorPtr> &outputs,
                                                        const std::map<uint32_t, tensor::TensorPtr> &) {
  auto ret = KernelMod::Resize(base_operator, inputs, outputs);
  if (ret != KRET_OK) {
    return ret;
  }
  x_dtype_ = inputs[kIndex0]->GetDtype();
  indices_dtype_ = inputs[kIndex1]->GetDtype();
  x_shape_ = inputs[kIndex0]->GetDeviceShapeAdaptively();
  segment_ids_shape_ = inputs[kIndex2]->GetDeviceShapeAdaptively();
  y_shape_ = outputs[kIndex0]->GetDeviceShapeAdaptively();
  return KRET_OK;
}

bool SparseSegmentSumWithNumSegmentsCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                                         const std::vector<kernel::AddressPtr> &,
                                                         const std::vector<kernel::AddressPtr> &outputs) {
  MS_ERROR_IF_NULL(kernel_func_);
  kernel_func_(this, inputs, outputs);
  return true;
}

template <typename T1, typename T2>
void SparseSegmentSumWithNumSegmentsCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                                               const std::vector<kernel::AddressPtr> &outputs) {
  constexpr size_t kMultiply = 1;
  size_t n = std::accumulate(x_shape_.begin(), x_shape_.end(), kMultiply, std::multiplies<int>()) /
             static_cast<size_t>(x_shape_[kIndex0]);
  size_t m = std::accumulate(segment_ids_shape_.begin(), segment_ids_shape_.end(), kMultiply, std::multiplies<int>());
  size_t num_elements = std::accumulate(y_shape_.begin(), y_shape_.end(), kMultiply, std::multiplies<int>());
  auto x_shape0 = static_cast<T2>(x_shape_[kIndex0]);
  auto x_addr = static_cast<T1 *>(inputs[kIndex0]->addr);
  auto indices_addr = static_cast<T2 *>(inputs[kIndex1]->addr);
  auto segment_ids_addr = static_cast<T2 *>(inputs[kIndex2]->addr);
  auto num_segments_addr = static_cast<T2 *>(inputs[kIndex3]->addr);
  auto y_addr = static_cast<T1 *>(outputs[kIndex0]->addr);
  for (size_t i = 1; i < m; i++) {
    if (segment_ids_addr[i] < segment_ids_addr[i - 1]) {
      MS_EXCEPTION(ValueError) << "For '" << kernel_name_ << "', input segment_ids should be sorted.";
    }
  }
  if (segment_ids_addr[m - 1] >= num_segments_addr[0]) {
    MS_EXCEPTION(ValueError) << "For '" << kernel_name_
                             << "', num_segments must be bigger than the largest id of segment_ids.";
  }
  for (size_t i = 0; i < m; i++) {
    if (indices_addr[i] >= x_shape0) {
      MS_EXCEPTION(ValueError) << "For '" << kernel_name_ << "', input indices is out of range of x's first dimension.";
    }
  }
  for (size_t i = 0; i < num_elements; i++) {
    y_addr[i] = (T1)0;
  }
  int oldindex = -1;
  for (size_t i = 0; i < m; i++) {
    if (oldindex != segment_ids_addr[i]) {
      oldindex = segment_ids_addr[i];
      for (size_t j = 0; j < n; j++) {
        y_addr[j + IntToSize(oldindex) * n] = (T1)0;
      }
    }
    for (size_t j = 0; j < n; j++) {
      y_addr[j + IntToSize(oldindex) * n] += x_addr[j + static_cast<size_t>(indices_addr[i]) * n];
    }
  }
}

std::vector<std::pair<KernelAttr, SparseSegmentSumWithNumSegmentsCpuKernelMod::LaunchKernelFunc>>
  SparseSegmentSumWithNumSegmentsCpuKernelMod::f_list_ = {
    ADD_KERNEL(Int8, Int32, Int32, Int32, Int8, int8_t, int32_t),
    ADD_KERNEL(Int8, Int64, Int64, Int64, Int8, int8_t, int64_t),
    ADD_KERNEL(Int16, Int32, Int32, Int32, Int16, int16_t, int32_t),
    ADD_KERNEL(Int16, Int64, Int64, Int64, Int16, int16_t, int64_t),
    ADD_KERNEL(Int32, Int32, Int32, Int32, Int32, int32_t, int32_t),
    ADD_KERNEL(Int32, Int64, Int64, Int64, Int32, int32_t, int64_t),
    ADD_KERNEL(Int64, Int32, Int32, Int32, Int64, int64_t, int32_t),
    ADD_KERNEL(Int64, Int64, Int64, Int64, Int64, int64_t, int64_t),
    ADD_KERNEL(UInt8, Int32, Int32, Int32, UInt8, uint8_t, int32_t),
    ADD_KERNEL(UInt8, Int64, Int64, Int64, UInt8, uint8_t, int64_t),
    ADD_KERNEL(UInt16, Int32, Int32, Int32, UInt16, uint16_t, int32_t),
    ADD_KERNEL(UInt16, Int64, Int64, Int64, UInt16, uint16_t, int64_t),
    ADD_KERNEL(Float16, Int32, Int32, Int32, Float16, float16, int32_t),
    ADD_KERNEL(Float16, Int64, Int64, Int64, Float16, float16, int64_t),
    ADD_KERNEL(Float32, Int32, Int32, Int32, Float32, float, int32_t),
    ADD_KERNEL(Float32, Int64, Int64, Int64, Float32, float, int64_t),
    ADD_KERNEL(Float64, Int32, Int32, Int32, Float64, double, int32_t),
    ADD_KERNEL(Float64, Int64, Int64, Int64, Float64, double, int64_t)};

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, SparseSegmentSumWithNumSegments, SparseSegmentSumWithNumSegmentsCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
