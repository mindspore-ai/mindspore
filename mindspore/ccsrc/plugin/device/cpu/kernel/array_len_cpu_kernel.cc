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

#include "plugin/device/cpu/kernel/array_len_cpu_kernel.h"

#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <map>
#include <string>
#include <functional>
#include "kernel/common_utils.h"
namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kInputNum = 1;
constexpr size_t kOutputNum = 1;
constexpr char kKernelName[] = "array_len";
}  // namespace
bool ArrayLenCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kInputNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kOutputNum, kernel_name_);
  return true;
}

int ArrayLenCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                 const std::vector<KernelTensorPtr> &outputs,
                                 const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  int ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost);
  if (ret != 0) {
    return ret;
  }
  input_shape_ = inputs[0]->GetShapeVector();
  return KRET_OK;
}

bool ArrayLenCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                  const std::vector<kernel::AddressPtr> &,
                                  const std::vector<kernel::AddressPtr> &outputs) {
  auto output_addr = reinterpret_cast<int *>(outputs[0]->addr);
  output_addr[0] = input_shape_[0];
  return true;
}

std::vector<KernelAttr> ArrayLenCpuKernelMod::GetOpSupport() {
  static std::vector<KernelAttr> kernel_attr_list = {
    KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kObjectTypeTuple, kNumberTypeInt64),
    KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kObjectTypeTuple, kNumberTypeInt64),
    KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kObjectTypeTuple, kNumberTypeInt64),
    KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kObjectTypeTuple, kNumberTypeInt64),
    KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kObjectTypeTuple, kNumberTypeInt64),
    KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kObjectTypeTuple, kNumberTypeInt64),
    KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kObjectTypeTuple, kNumberTypeInt64),
  };
  return kernel_attr_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, array_len, ArrayLenCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
