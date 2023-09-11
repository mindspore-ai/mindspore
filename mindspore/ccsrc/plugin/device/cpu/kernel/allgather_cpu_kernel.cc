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

#include "plugin/device/cpu/kernel/allgather_cpu_kernel.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "plugin/device/cpu/hal/device/mpi/mpi_interface.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr auto kRanksGroup = "group";
}  // namespace

int AllGatherCpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                  const std::vector<KernelTensor *> &outputs) {
  if (int ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  ranks_group_ = GetValue<std::vector<int>>(primitive_->GetAttr(kRanksGroup));
  return KRET_OK;
}

bool AllGatherCpuKernelMod::Launch(const std::vector<kernel::KernelTensor *> &inputs,
                                   const std::vector<kernel::KernelTensor *> &,
                                   const std::vector<kernel::KernelTensor *> &outputs) {
  auto *input_addr = reinterpret_cast<float *>(inputs[0]->device_ptr());
  auto *output_addr = reinterpret_cast<float *>(outputs[0]->device_ptr());
  auto input_data_num = inputs[0]->size() / sizeof(float);
  return MPIAllGather(input_addr, output_addr, ranks_group_, input_data_num);
}

std::vector<KernelAttr> AllGatherCpuKernelMod::GetOpSupport() {
  static std::vector<KernelAttr> kernel_attr_list = {
    KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32)};
  return kernel_attr_list;
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, _HostAllGather, AllGatherCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
