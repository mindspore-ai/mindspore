/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/kernel/shape_cpu_kernel.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kShapeInputsNum = 1;
constexpr size_t kShapeOutputsNum = 1;
}  // namespace

bool ShapeCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                             const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kShapeInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kShapeOutputsNum, kernel_name_);
  return MatchKernelFunc(base_operator, inputs, outputs);
}

int ShapeCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                              const std::vector<KernelTensorPtr> &outputs,
                              const std::map<uint32_t, tensor::TensorPtr> &) {
  if (auto ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
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

bool ShapeCpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                                     const std::vector<AddressPtr> &outputs) {
  auto output_addr = reinterpret_cast<int64_t *>(outputs[0]->addr);
  for (size_t i = 0; i < LongToSize(output_shape_[0]); ++i) {
    output_addr[i] = input_shape_[i];
  }
  return true;
}

const std::vector<std::pair<KernelAttr, ShapeCpuKernelMod::KernelRunFunc>> &ShapeCpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, ShapeCpuKernelMod::KernelRunFunc>> func_list = {
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kObjectTypeTuple, kNumberTypeInt64),
     &ShapeCpuKernelMod::LaunchKernel},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kObjectTypeTuple, kNumberTypeInt64),
     &ShapeCpuKernelMod::LaunchKernel},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kObjectTypeTuple, kNumberTypeInt64),
     &ShapeCpuKernelMod::LaunchKernel},
    {KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kObjectTypeTuple, kNumberTypeInt64),
     &ShapeCpuKernelMod::LaunchKernel},
    {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kObjectTypeTuple, kNumberTypeInt64),
     &ShapeCpuKernelMod::LaunchKernel},
    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kObjectTypeTuple, kNumberTypeInt64),
     &ShapeCpuKernelMod::LaunchKernel},
    {KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kObjectTypeTuple, kNumberTypeInt64),
     &ShapeCpuKernelMod::LaunchKernel}};
  return func_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, Shape, ShapeCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
