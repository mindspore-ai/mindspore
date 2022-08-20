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

#include "plugin/device/cpu/kernel/trunc_cpu_kernel.h"
#include <functional>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
const size_t kZero = 0;
constexpr size_t kTruncInputsNum = 1;
constexpr size_t kTruncOutputsNum = 1;
constexpr size_t kSizeGapMin = 1024;
constexpr size_t kSizeGapMax = 102400;

template <typename T>
void Trunc(const T *in0, T *out0, size_t start, size_t end) {
  for (size_t index = start; index < end; index++) {
    if constexpr ((std::is_same_v<T, uint8_t>) || (std::is_same_v<T, int8_t>) || (std::is_same_v<T, int32_t>)) {
      out0[index] = in0[index];
    } else if constexpr ((std::is_same_v<T, float>) || (std::is_same_v<T, double>)) {
      out0[index] = std::trunc(in0[index]);
    }
  }
}
}  // namespace

bool TruncCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                             const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  dtype_ = inputs[kZero]->GetDtype();
  return true;
}

int TruncCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                              const std::vector<KernelTensorPtr> &outputs,
                              const std::map<uint32_t, tensor::TensorPtr> &) {
  if (int ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
    return ret;
  }

  auto input_shape = inputs[kZero]->GetShapeVector();
  input_size_ = std::accumulate(input_shape.begin(), input_shape.end(), size_t(1), std::multiplies<size_t>());

  return KRET_OK;
}

bool TruncCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs,
                               const std::vector<kernel::AddressPtr> &workspace,
                               const std::vector<kernel::AddressPtr> &outputs) {
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
  } else {
    MS_EXCEPTION(TypeError) << "Unsupported input data type for operator [" << kernel_name_
                            << "]: " << TypeIdToType(dtype_)->ToString();
  }
  return ret;
}

template <typename T>
bool TruncCpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs) {
  const T *input_0_addr = reinterpret_cast<T *>(inputs[kZero]->addr);
  T *output_0_addr = reinterpret_cast<T *>(outputs[kZero]->addr);
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
    KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
    KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
    KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
    KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
    KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32)};

  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, Trunc, TruncCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
