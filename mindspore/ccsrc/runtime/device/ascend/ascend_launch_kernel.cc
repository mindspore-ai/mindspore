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

#include "runtime/device/ascend/ascend_launch_kernel.h"

#include <vector>
#include <memory>
#include "runtime/device/memory_manager.h"
#include "runtime/device/ascend/ascend_memory_pool.h"
#include "runtime/device/ascend/kernel_build_ascend.h"
#include "runtime/device/ascend/kernel_select_ascend.h"

namespace mindspore::device::ascend {
void AscendLaunchKernel::FreeDeviceMem(void *addr) { AscendMemoryPool::GetInstance().FreeTensorMem(addr); }

size_t AscendLaunchKernel::AlignSizeForLaunchKernel(size_t size) { return MemoryManager::GetCommonAlignSize(size); }

uint8_t *AscendLaunchKernel::AllocDeviceMem(size_t size) {
  auto device_memory = AscendMemoryPool::GetInstance().AllocTensorMem(size);
  MS_EXCEPTION_IF_NULL(device_memory);
  return static_cast<uint8_t *>(device_memory);
}

void AscendLaunchKernel::KernelSelect(std::shared_ptr<session::KernelGraph> kernel_graph) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto node_list = kernel_graph->execution_order();
  for (size_t i = 0; i < node_list.size(); ++i) {
    auto status = device::ascend::SelectKernelInfo(node_list[i]);
    if (status == ascend::kNoMatched) {
      MS_LOG(ERROR) << "cnode name : " << node_list[i]->fullname_with_scope() << " kernel select failed";
    }
  }
}

void AscendLaunchKernel::KernelBuild(std::shared_ptr<session::KernelGraph> kernel_graph) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  device::ascend::KernelBuild(kernel_graph->execution_order());
}
}  // namespace mindspore::device::ascend
