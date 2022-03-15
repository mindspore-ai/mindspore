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

#include "plugin/device/cpu/kernel/rl/tensor_array_close_kernel.h"
#include "kernel/common_utils.h"
#include "plugin/device/cpu/hal/device/cpu_tensor_array.h"
#include "runtime/device/tensor_array_manager.h"

namespace mindspore {
namespace kernel {
using mindspore::device::TensorArrayMgr;
using mindspore::device::cpu::CPUTensorArray;
using mindspore::device::cpu::CPUTensorArrayPtr;
TensorArrayCloseCpuKernelMod::TensorArrayCloseCpuKernelMod() {}

void TensorArrayCloseCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  input_size_list_.push_back(sizeof(int64_t));
  output_size_list_.push_back(sizeof(int64_t));
}

bool TensorArrayCloseCpuKernelMod::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                                          const std::vector<AddressPtr> &) {
  auto handle_addr = GetDeviceAddress<int64_t>(inputs, 0);
  MS_EXCEPTION_IF_NULL(handle_addr);
  CPUTensorArrayPtr tensors_ =
    std::dynamic_pointer_cast<CPUTensorArray>(TensorArrayMgr::GetInstance().GetTensorArray(handle_addr[0]));
  MS_ERROR_IF_NULL(tensors_);
  // Free device mem
  tensors_->Free();
  // Erase tensorarray
  if (!TensorArrayMgr::GetInstance().EraseTensorArray(handle_addr[0])) {
    MS_LOG(EXCEPTION) << "Free tensorarray failed";
  }
  return true;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, TensorArrayClose, TensorArrayCloseCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
