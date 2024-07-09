/**
 * Copyright 2021-2023 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/kernel/tensor_shape_cpu_kernel.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore::kernel {
namespace {
constexpr size_t kTensorShapeOutputNum = 1;
}  // namespace
bool TensorShapeCpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                   const std::vector<KernelTensor *> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kTensorShapeOutputNum, kernel_name_);
  return true;
}

int TensorShapeCpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                    const std::vector<KernelTensor *> &outputs) {
  if (auto ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }

  input_shape_ = inputs[kIndex0]->GetShapeVector();
  output_shape_ = outputs[kIndex0]->GetShapeVector();
  if (output_shape_.size() != 1) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the dimension of output must be 1-D, but got: " << output_shape_.size();
  }
  if (output_shape_[0] != SizeToLong(input_shape_.size())) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', 'output_shape[0]' must be equal to the dimension of input, but got 'output_shape[0]': "
                      << output_shape_[0] << " and the dimension of input: " << input_shape_.size();
  }
  return KRET_OK;
}

bool TensorShapeCpuKernelMod::Launch(const std::vector<kernel::KernelTensor *> &inputs,
                                     const std::vector<kernel::KernelTensor *> &,
                                     const std::vector<kernel::KernelTensor *> &outputs) {
  auto output_addr = reinterpret_cast<int64_t *>(outputs[0]->device_ptr());
  for (size_t i = 0; i < LongToSize(output_shape_[0]); ++i) {
    output_addr[i] = input_shape_[i];
  }
  return true;
}

std::vector<KernelAttr> TensorShapeCpuKernelMod::GetOpSupport() {
  static const std::vector<KernelAttr> support_list = {
    KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeInt64),
    KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeInt64),
    KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeInt64),
    KernelAttr().AddInputAttr(kNumberTypeBFloat16).AddOutputAttr(kNumberTypeInt64),
    KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt64),
    KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt64),
    KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt64),
    KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
    KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeInt64),
    KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeInt64),
    KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeInt64),
    KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeInt64),
    KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeInt64),
    KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeInt64),
    KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeInt64)};
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, DynamicShape, TensorShapeCpuKernelMod);
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, TensorShape, TensorShapeCpuKernelMod);
}  // namespace mindspore::kernel
