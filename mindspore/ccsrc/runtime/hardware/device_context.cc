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

#include "runtime/hardware/device_context.h"
#include "backend/common/optimizer/common_backend_optimization.h"

namespace mindspore {
namespace device {
void *DeviceResManager::AllocateOffloadMemory(size_t size) const { return offloaded_mem_pool_->MallocHost(size); }

void DeviceResManager::FreeOffloadMemory(void *ptr) const { offloaded_mem_pool_->FreeHost(ptr); }

bool DeviceResManager::AllocateMemory(DeviceAddress *const &address) const {
  MS_EXCEPTION_IF_NULL(address);
  if (address->GetPtr() != nullptr) {
    MS_LOG(ERROR) << "Memory leak detected!";
    return false;
  }

  auto device_ptr = AllocateMemory(address->GetSize());
  if (!device_ptr) {
    MS_LOG(WARNING) << "Allocate memory failed for size: " << address->GetSize();
    return false;
  }
  address->set_ptr(device_ptr);
  address->set_from_mem_pool(true);
  return true;
}

void DeviceResManager::FreeMemory(DeviceAddress *const &address) const {
  MS_EXCEPTION_IF_NULL(address);
  if (address->GetPtr() == nullptr) {
    MS_LOG(EXCEPTION) << "Device ptr is null in device address to release!";
  }

  if (!address->from_mem_pool()) {
    return;
  }

  FreeMemory(address->GetMutablePtr());
  address->set_ptr(nullptr);
}

void DeprecatedKernelExecutor::UnifyMindIR(const KernelGraphPtr &graph) const { opt::CommonUnifyMindIR(graph); }
void DeprecatedKernelExecutor::AddUnifyMindIRPass(const std::shared_ptr<opt::GraphOptimizer> &opt) const {
  opt->AddPassManager(opt::GetCommonUnifyMindIRPassManager());
}
}  // namespace device
}  // namespace mindspore
