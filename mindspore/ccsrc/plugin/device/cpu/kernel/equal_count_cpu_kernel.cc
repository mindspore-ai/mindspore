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

#include "plugin/device/cpu/kernel/equal_count_cpu_kernel.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kEqualCountInputsNum = 2;
constexpr size_t kEqualCountOutputsNum = 1;
}  // namespace

bool EqualCountCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                  const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->GetPrim()->name();

  return true;
}

bool EqualCountCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                    const std::vector<kernel::AddressPtr> &,
                                    const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kEqualCountInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kEqualCountOutputsNum, kernel_name_);
  if (inputs[0]->size != inputs[1]->size) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the address size of inputs must be the same, but got the address size of 'inputs[0]': "
                      << inputs[0]->size << " and the address size of 'inputs[1]': " << inputs[1]->size;
  }

  int count = 0;
  auto left = reinterpret_cast<int *>(inputs[0]->addr);
  auto right = reinterpret_cast<int *>(inputs[1]->addr);
  size_t elem_num = inputs[0]->size / sizeof(int);
  for (size_t i = 0; i < elem_num; i++) {
    if (left[i] == right[i]) {
      count++;
    }
  }
  auto output = reinterpret_cast<int *>(outputs[0]->addr);
  output[0] = count;
  return true;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, EqualCount, EqualCountCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
