/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/kernel/debug_cpu_kernel.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "utils/ms_utils.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kDebugInputsNum = 1;
constexpr size_t kDebugOutputsNum = 1;
}  // namespace

bool DebugCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                             const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->GetPrim()->name();
  return true;
}

bool DebugCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &,
                               const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kDebugInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kDebugOutputsNum, kernel_name_);
  const auto *val = reinterpret_cast<int *>(inputs[0]->addr);
  MS_LOG(DEBUG) << " launch DebugCpuKernelMod";

  auto output = reinterpret_cast<int *>(outputs[0]->addr);
  size_t elem_num = inputs[0]->size / sizeof(int);
  for (size_t i = 0; i < elem_num; i++) {
    output[i] = static_cast<int>(val[i]);
  }
  return true;
}

std::vector<KernelAttr> DebugCpuKernelMod::GetOpSupport() {
  static std::vector<KernelAttr> support_list = {
    KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeInt32)};
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, Debug, DebugCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
