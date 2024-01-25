/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/kernel/fftshift_cpu_kernel.h"
#include <functional>
#include <algorithm>
#include <utility>
#include <memory>
#include <complex>

namespace mindspore {
namespace kernel {
namespace {
using complex64 = std::complex<float>;
using complex128 = std::complex<double>;
}  // namespace

bool FFTShiftCpuKernelMod::Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(EXCEPTION) << kernel_name_ << " valid cpu kernel does not support this kernel data type: " << kernel_attr;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int FFTShiftCpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                 const std::vector<KernelTensor *> &outputs) {
  if (int ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }

  tensor_shape_ = inputs[kIndex0]->GetShapeVector();
  x_rank_ = SizeToLong(tensor_shape_.size());
  // No need to process when input is empty tensor.
  if (x_rank_ == 0) {
    return KRET_OK;
  }

  // Get or set attribute axes
  axes_ = inputs[kIndex1]->GetValueWithCheck<std::vector<int64_t>>();
  if (axes_.empty()) {
    // Process all dimensions.
    for (int64_t i = 0; i < x_rank_; ++i) {
      (void)axes_.emplace_back(i);
    }
  } else {
    (void)std::for_each(axes_.begin(), axes_.end(), [this](auto &axis) { axis = axis < 0 ? x_rank_ + axis : axis; });
  }
  forward_ = inputs[kIndex2]->GetValueWithCheck<bool>();
  element_nums_ = SizeOf(tensor_shape_);

  return KRET_OK;
}

template <typename T>
bool FFTShiftCpuKernelMod::LaunchKernel(const std::vector<kernel::KernelTensor *> &inputs,
                                        const std::vector<kernel::KernelTensor *> &outputs) {
  T *input_ptr = reinterpret_cast<T *>(inputs[0]->device_ptr());
  T *output_ptr = reinterpret_cast<T *>(outputs[0]->device_ptr());
  // No need to process when input is empty tensor.
  if (x_rank_ == 0) {
    output_ptr[0] = input_ptr[0];
    return true;
  }
  // Calculate the offset of input[i]
  std::vector<int64_t> offsets_(element_nums_, 0);
  for (size_t j = 0; j < axes_.size(); j++) {
    int64_t size_j = tensor_shape_[axes_[j]];
    int64_t size_back =
      std::accumulate(tensor_shape_.begin() + axes_[j] + 1, tensor_shape_.end(), 1, std::multiplies<int64_t>());
    int64_t size_tmp1 = size_j * size_back;
    int64_t size_tmp2 = size_j / 2 * size_back;

    for (int64_t i = 0; i < element_nums_; i++) {
      if (forward_ == true) {
        if ((i + offsets_[i]) % size_tmp1 >= size_tmp1 - size_tmp2) {
          offsets_[i] -= size_tmp1 - size_tmp2;
        } else {
          offsets_[i] += size_tmp2;
        }
      } else {
        if ((i + offsets_[i]) % size_tmp1 < size_tmp2) {
          offsets_[i] += size_tmp1 - size_tmp2;
        } else {
          offsets_[i] -= size_tmp2;
        }
      }
    }
  }

  // Update output according to offset
  for (int64_t i = 0; i < element_nums_; i++) {
    output_ptr[i + offsets_[i]] = input_ptr[i];
  }

  return true;
}

#define FFTSHIFT_CPU_REG(T1, T2)                                    \
  KernelAttr()                                                      \
    .AddInputAttr(T1)                                 /* x */       \
    .AddInputAttr(kObjectTypeTuple, kNumberTypeInt64) /* axes */    \
    .AddInputAttr(kObjectTypeNumber, kNumberTypeBool) /* forward */ \
    .AddOutputAttr(T1),                                             \
    &FFTShiftCpuKernelMod::LaunchKernel<T2>

std::vector<std::pair<KernelAttr, FFTShiftCpuKernelMod::FFTShiftFunc>> FFTShiftCpuKernelMod::func_list_ = {
  {FFTSHIFT_CPU_REG(kNumberTypeUInt8, uint8_t)},         {FFTSHIFT_CPU_REG(kNumberTypeUInt16, uint16_t)},
  {FFTSHIFT_CPU_REG(kNumberTypeUInt32, uint32_t)},       {FFTSHIFT_CPU_REG(kNumberTypeUInt64, uint64_t)},
  {FFTSHIFT_CPU_REG(kNumberTypeInt8, int8_t)},           {FFTSHIFT_CPU_REG(kNumberTypeInt16, int16_t)},
  {FFTSHIFT_CPU_REG(kNumberTypeInt32, int32_t)},         {FFTSHIFT_CPU_REG(kNumberTypeInt64, int64_t)},
  {FFTSHIFT_CPU_REG(kNumberTypeFloat16, float16)},       {FFTSHIFT_CPU_REG(kNumberTypeFloat32, float)},
  {FFTSHIFT_CPU_REG(kNumberTypeFloat64, double)},        {FFTSHIFT_CPU_REG(kNumberTypeComplex64, complex64)},
  {FFTSHIFT_CPU_REG(kNumberTypeComplex128, complex128)}, {FFTSHIFT_CPU_REG(kNumberTypeBool, bool)}};

std::vector<KernelAttr> FFTShiftCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, FFTShiftFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, FFTShift, FFTShiftCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
