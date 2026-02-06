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

#include "kernel/cpu/native/environ/environ_cpu_destroy_all.h"
#include "include/runtime/hardware_abstract/kernel_base/environ_manager.h"

namespace mindspore {
namespace kernel {
int EnvironDestroyAllCpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                          const std::vector<KernelTensor *> &outputs) {
  if (auto ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  // Check the output type.
  auto output_type = outputs[kIndex0]->dtype_id();
  if (output_type != TypeId::kNumberTypeBool) {
    MS_LOG(EXCEPTION) << "The output type is invalid: " << output_type;
  }

  output_size_list_.clear();
  output_size_list_.push_back(sizeof(bool));
  return KRET_OK;
}

bool EnvironDestroyAllCpuKernelMod::Launch(const std::vector<KernelTensor *> &, const std::vector<KernelTensor *> &,
                                           const std::vector<KernelTensor *> &) {
  MS_LOG(INFO) << "Clear the global environ data.";
  // Clear the global data which are generated in the kernel running.
  EnvironMgr::GetInstance().Clear();

  return true;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, EnvironDestroyAll, EnvironDestroyAllCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
