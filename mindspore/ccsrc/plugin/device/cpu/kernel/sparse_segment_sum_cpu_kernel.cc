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
#include "functional"
#include "mindspore/core/abstract/utils.h"
#include "kernel/common_utils.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
bool SparseSegmentSumCpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                        const std::vector<KernelTensor *> &outputs) {
  constexpr size_t input_num = 3;
  constexpr size_t output_num = 1;
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), input_num, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), output_num, kernel_name_);
  return MatchKernelFunc(kernel_name_, inputs, outputs);
}

int SparseSegmentSumCpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                         const std::vector<KernelTensor *> &outputs) {
  if (auto ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  x_shape_ = inputs[kIndex0]->GetDeviceShapeVector();
  segment_shape_ = inputs[kIndex2]->GetDeviceShapeVector();
  return KRET_OK;
}

template <typename T1, typename T2>
bool SparseSegmentSumCpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                                const std::vector<KernelTensor *> &,
                                                const std::vector<KernelTensor *> &outputs) {
  constexpr size_t kMultiply = 1;
  size_t n = std::accumulate(x_shape_.begin(), x_shape_.end(), kMultiply, std::multiplies<int>()) /
             static_cast<size_t>(x_shape_[kIndex0]);
  size_t m = std::accumulate(segment_shape_.begin(), segment_shape_.end(), kMultiply, std::multiplies<int>());
  auto x_shape0 = static_cast<T2>(x_shape_[kIndex0]);
  auto dataptr = static_cast<T1 *>(inputs[kIndex0]->device_ptr());
  auto indicesptr = static_cast<T2 *>(inputs[kIndex1]->device_ptr());
  auto segment_idsptr = static_cast<T2 *>(inputs[kIndex2]->device_ptr());
  auto yptr = static_cast<T1 *>(outputs[kIndex0]->device_ptr());
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
        yptr[j + IntToSize(oldindex) * n] = (T1)0;
      }
    }
    for (size_t j = 0; j < n; j++) {
      yptr[j + IntToSize(oldindex) * n] += dataptr[j + static_cast<size_t>(indicesptr[i]) * n];
    }
  }
  return true;
}

#define SPARSE_SEGMENT_SUM_CPU_REG(MS_T, MS_S, T, S)                                         \
  KernelAttr().AddInputAttr(MS_T).AddInputAttr(MS_S).AddInputAttr(MS_S).AddOutputAttr(MS_T), \
    &SparseSegmentSumCpuKernelMod::LaunchKernel<T, S>

#define SPARSE_SEGMENT_SUM_CPU_INDEX_REG(MS_T, T)                     \
  {SPARSE_SEGMENT_SUM_CPU_REG(MS_T, kNumberTypeInt32, T, int32_t)}, { \
    SPARSE_SEGMENT_SUM_CPU_REG(MS_T, kNumberTypeInt64, T, int64_t)    \
  }

const std::vector<std::pair<KernelAttr, SparseSegmentSumCpuKernelMod::KernelRunFunc>>
  &SparseSegmentSumCpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, SparseSegmentSumCpuKernelMod::KernelRunFunc>> func_list = {
    SPARSE_SEGMENT_SUM_CPU_INDEX_REG(kNumberTypeInt8, int8_t),
    SPARSE_SEGMENT_SUM_CPU_INDEX_REG(kNumberTypeInt16, int16_t),
    SPARSE_SEGMENT_SUM_CPU_INDEX_REG(kNumberTypeInt32, int32_t),
    SPARSE_SEGMENT_SUM_CPU_INDEX_REG(kNumberTypeInt64, int64_t),
    SPARSE_SEGMENT_SUM_CPU_INDEX_REG(kNumberTypeUInt8, uint8_t),
    SPARSE_SEGMENT_SUM_CPU_INDEX_REG(kNumberTypeUInt16, uint16_t),
    SPARSE_SEGMENT_SUM_CPU_INDEX_REG(kNumberTypeFloat16, float16),
    SPARSE_SEGMENT_SUM_CPU_INDEX_REG(kNumberTypeFloat32, float),
    SPARSE_SEGMENT_SUM_CPU_INDEX_REG(kNumberTypeFloat64, double),
  };
  return func_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, SparseSegmentSum, SparseSegmentSumCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
