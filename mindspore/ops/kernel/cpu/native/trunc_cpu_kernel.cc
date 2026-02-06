/**
 * Copyright 2022-2025 Huawei Technologies Co., Ltd
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

#include "kernel/cpu/native/trunc_cpu_kernel.h"
#include <functional>

namespace mindspore {
namespace kernel {
namespace trunc_cpu {
namespace {
const size_t kZero = 0;
constexpr size_t kTruncInputsNum = 1;
constexpr size_t kTruncOutputsNum = 1;
constexpr size_t kSizeGapMin = 1024;
constexpr size_t kSizeGapMax = 102400;

template <typename T>
void Trunc(const T *in0, T *out0, size_t start, size_t end) {
  for (size_t index = start; index < end; index++) {
    if constexpr (std::is_integral_v<T>) {
      out0[index] = in0[index];
    } else if constexpr (std::is_same_v<T, mindspore::BFloat16>) {
      out0[index] = mindspore::BFloat16(std::trunc(static_cast<float>(in0[index])));
    } else if constexpr (std::is_same_v<T, float16>) {
      out0[index] = float16(std::trunc(static_cast<float>(in0[index])));
    } else if constexpr (std::is_floating_point_v<T>) {
      out0[index] = std::trunc(in0[index]);
    }
  }
}
}  // namespace

bool TruncCpuKernelMod::Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  dtype_ = inputs[kZero]->dtype_id();
  return true;
}

int TruncCpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  if (int ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }

  auto input_shape = inputs[kZero]->GetShapeVector();
  input_size_ = std::accumulate(input_shape.begin(), input_shape.end(), size_t(1), std::multiplies<size_t>());

  return KRET_OK;
}

bool TruncCpuKernelMod::Launch(const std::vector<kernel::KernelTensor *> &inputs,
                               const std::vector<kernel::KernelTensor *> &workspace,
                               const std::vector<kernel::KernelTensor *> &outputs) {
  bool ret = true;
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kTruncInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kTruncOutputsNum, kernel_name_);
  if (dtype_ == kNumberTypeFloat32) {
    ret = LaunchKernel<float>(inputs, outputs);
  } else if (dtype_ == kNumberTypeFloat64) {
    ret = LaunchKernel<double>(inputs, outputs);
  } else if (dtype_ == kNumberTypeInt8) {
    ret = LaunchKernel<int8_t>(inputs, outputs);
  } else if (dtype_ == kNumberTypeUInt8) {
    ret = LaunchKernel<uint8_t>(inputs, outputs);
  } else if (dtype_ == kNumberTypeInt32) {
    ret = LaunchKernel<int32_t>(inputs, outputs);
  } else if (dtype_ == kNumberTypeBool) {
    ret = LaunchKernel<bool>(inputs, outputs);
  } else if (dtype_ == kNumberTypeBFloat16) {
    ret = LaunchKernel<mindspore::BFloat16>(inputs, outputs);
  } else if (dtype_ == kNumberTypeInt64) {
    ret = LaunchKernel<int64_t>(inputs, outputs);
  } else if (dtype_ == kNumberTypeInt16) {
    ret = LaunchKernel<int16_t>(inputs, outputs);
  } else if (dtype_ == kNumberTypeUInt16) {
    ret = LaunchKernel<uint16_t>(inputs, outputs);
  } else if (dtype_ == kNumberTypeUInt32) {
    ret = LaunchKernel<uint32_t>(inputs, outputs);
  } else if (dtype_ == kNumberTypeUInt64) {
    ret = LaunchKernel<uint64_t>(inputs, outputs);
  } else if (dtype_ == kNumberTypeFloat16) {
    ret = LaunchKernel<float16>(inputs, outputs);
  } else {
    MS_EXCEPTION(TypeError) << "Unsupported input data type for operator [" << kernel_name_
                            << "]: " << TypeIdToType(dtype_)->ToString();
  }
  return ret;
}

template <typename T>
bool TruncCpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                     const std::vector<KernelTensor *> &outputs) {
  const T *input_0_addr = reinterpret_cast<T *>(inputs[kZero]->device_ptr());
  T *output_0_addr = reinterpret_cast<T *>(outputs[kZero]->device_ptr());
  auto task = std::bind(Trunc<T>, input_0_addr, output_0_addr, std::placeholders::_1, std::placeholders::_2);
  if (input_size_ <= kSizeGapMin) {
    Trunc(input_0_addr, output_0_addr, 0, input_size_ * kTruncInputsNum);
  } else if (input_size_ <= kSizeGapMax) {
    ParallelLaunchAutoSearch(task, input_size_ * kTruncInputsNum, this, &parallel_search_info_);
  } else {
    ParallelLaunch(task, input_size_ * kTruncInputsNum, 0, this);
  }
  return true;
}

std::vector<KernelAttr> TruncCpuKernelMod::GetOpSupport() {
  static std::vector<KernelAttr> support_list = {
    KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
    KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
    KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
    KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeBFloat16).AddOutputAttr(kNumberTypeBFloat16),
    KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeBool),
    KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
    KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),
    KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
    KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
    KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
    KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16),
    KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32),
    KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt64)};

  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, Trunc, TruncCpuKernelMod);
}  // namespace trunc_cpu
}  // namespace kernel
}  // namespace mindspore
