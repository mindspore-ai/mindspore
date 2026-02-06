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

#include "kernel/cpu/native/environ/environ_cpu_create.h"
#include "include/runtime/hardware_abstract/kernel_base/environ_manager.h"
#include "include/runtime/hardware_abstract/kernel_base/common_utils.h"

namespace mindspore {
namespace kernel {
int EnvironCreateCpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                      const std::vector<KernelTensor *> &outputs) {
  if (auto ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  // Check the output handle.
  auto handle_type = outputs[kIndex0]->dtype_id();
  const auto &handle_shapes = outputs[kIndex0]->GetShapeVector();
  if (!EnvironMgr::GetInstance().IsScalarTensor(handle_type, handle_shapes)) {
    MS_LOG(EXCEPTION) << "The output handle checks invalid, kernel: " << kernel_name_;
  }

  handle_size_ = sizeof(int64_t);
  output_size_list_.clear();
  output_size_list_.push_back(handle_size_);
  return KRET_OK;
}

bool EnvironCreateCpuKernelMod::Launch(const std::vector<KernelTensor *> &, const std::vector<KernelTensor *> &,
                                       const std::vector<KernelTensor *> &outputs) {
  // Generate an unique handle.
  int64_t env_handle = EnvironMgr::GetInstance().Create();

  auto output = GetDeviceAddress<int64_t>(outputs, kIndex0);
  MS_EXCEPTION_IF_NULL(output);
  output[kIndex0] = env_handle;
  MS_LOG(DEBUG) << "Create env handle: " << output[kIndex0];

  return true;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, EnvironCreate, EnvironCreateCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
