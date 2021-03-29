/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "runtime/hardware/cpu/cpu_device_context.h"
#include <string>
#include "runtime/device/cpu/cpu_device_address.h"
#include "runtime/device/cpu/cpu_memory_manager.h"
#include "backend/kernel_compiler/cpu/cpu_kernel_factory.h"
#include "runtime/device/cpu/kernel_select_cpu.h"
#include "utils/trace_base.h"

namespace mindspore {
namespace device {
namespace cpu {
bool CPUDeviceContext::Initialize() {
  if (initialized_) {
    return true;
  }
  mem_manager_ = std::make_shared<CPUMemoryManager>();
  MS_EXCEPTION_IF_NULL(mem_manager_);
  initialized_ = true;
  return true;
}

bool CPUDeviceContext::AllocateMemory(DeviceAddress *const &address, size_t size) const {
  address->ptr_ = static_cast<CPUMemoryManager *>(mem_manager_.get())->StaticMemMalloc(size);
  return true;
}

void CPUDeviceContext::FreeMemory(DeviceAddress *const &address) const {
  static_cast<CPUMemoryManager *>(mem_manager_.get())->MemFree(address->ptr_);
  address->ptr_ = nullptr;
}

void CPUDeviceContext::SetOperatorInfo(const std::vector<CNodePtr> &nodes) const {
  for (const auto &node : nodes) {
    SetKernelInfo(node);
  }
}

void CPUDeviceContext::CreateKernel(const std::vector<CNodePtr> &nodes) const {
  for (const auto &node : nodes) {
    MS_EXCEPTION_IF_NULL(node);
    std::string kernel_name = AnfAlgo::GetCNodeName(node);
    std::shared_ptr<kernel::CPUKernel> cpu_kernel = kernel::CPUKernelFactory::GetInstance().Create(kernel_name, node);
    if (!cpu_kernel) {
      MS_LOG(EXCEPTION) << "Build cpu operator[" << node->fullname_with_scope() << "] failed";
    }

    cpu_kernel->Init(node);
    AnfAlgo::SetKernelMod(cpu_kernel, node.get());
  }
}

bool CPUDeviceContext::LaunchKernel(KernelMod *kernel_mod, const std::vector<AddressPtr> &inputs,
                                    const std::vector<AddressPtr> &workspace,
                                    const std::vector<AddressPtr> &outputs) const {
  MS_EXCEPTION_IF_NULL(kernel_mod);
  return kernel_mod->Launch(inputs, workspace, outputs, nullptr);
}

MS_REGISTER_DEVICE(kCPUDevice, CPUDeviceContext);
}  // namespace cpu
}  // namespace device
}  // namespace mindspore
