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

#include "plugin/device/cpu/kernel/dynamic_shape_cpu_kernel.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kDynamicShapeOutputNum = 1;
}  // namespace

bool TensorShapeCpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                   const std::vector<KernelTensor *> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kDynamicShapeOutputNum, kernel_name_);
  return true;
}

int TensorShapeCpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                    const std::vector<KernelTensor *> &outputs) {
  if (auto ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  input_shape_ = inputs.at(kIndex0)->GetShapeVector();
  output_shape_ = outputs.at(kIndex0)->GetShapeVector();
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
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kDynamicShapeOutputNum, kernel_name_);
  auto output_addr = reinterpret_cast<int64_t *>(outputs[0]->device_ptr());
  for (size_t i = 0; i < LongToSize(output_shape_[0]); ++i) {
    output_addr[i] = input_shape_[i];
  }
  return true;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, DynamicShape, TensorShapeCpuKernelMod);
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, TensorShape, TensorShapeCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
