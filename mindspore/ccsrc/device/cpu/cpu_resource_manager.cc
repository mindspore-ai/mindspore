/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#include "device/cpu/cpu_resource_manager.h"
#include "session/anf_runtime_algorithm.h"

namespace mindspore {
namespace device {
namespace cpu {
CPUResourceManager::~CPUResourceManager() { MemFree(); }

void CPUResourceManager::MemFree() {
  if (mem_ptr_ != nullptr) {
    free(mem_ptr_);
    mem_ptr_ = nullptr;
    mem_size_ = 0;
  }

  for (auto &&iter : dynamic_mem_) {
    free(iter.first);
  }
  dynamic_mem_.clear();
}

void CPUResourceManager::MemPlan(const session::KernelGraph *graph) {
  mem_plan_.MemPlan(graph);
  size_t graph_mem_size = mem_plan_.GetGraphMemSize(graph);
  if (graph_mem_size > mem_size_) {
    MemFree();
    mem_ptr_ = reinterpret_cast<uint8_t *>(malloc(graph_mem_size));
    if (mem_ptr_ != nullptr) {
      mem_size_ = graph_mem_size;
      dynamic_malloc_ = false;
    } else {
      MS_LOG(INFO) << "Switch to dynamic malloc";
      dynamic_malloc_ = true;
    }
  }
}

void CPUResourceManager::MemMalloc(const session::KernelGraph *graph) {
  if (dynamic_malloc_) {
    return;
  }
  mem_plan_.MemAssign(graph, mem_ptr_);
}

void *CPUResourceManager::MemMalloc(size_t mem_size) {
  void *ptr = malloc(mem_size);
  if (ptr != nullptr) {
    memset_s(ptr, mem_size, 0, mem_size);
    dynamic_mem_[ptr] = mem_size;
    return ptr;
  } else {
    MS_LOG(EXCEPTION) << "Malloc memory failed: size " << mem_size;
  }
}

void CPUResourceManager::MemFree(void *ptr) {
  auto iter = dynamic_mem_.find(ptr);
  if (iter != dynamic_mem_.end()) {
    (void)dynamic_mem_.erase(iter);
    free(ptr);
  }
}

void CPUResourceManager::ResetAddressRefCount(const session::KernelGraph *graph) {
  if (!dynamic_malloc_) {
    return;
  }
  MS_EXCEPTION_IF_NULL(graph);
  auto kernels = graph->execution_order();
  for (const auto &kernel : kernels) {
    MS_EXCEPTION_IF_NULL(kernel);
    size_t input_num = AnfAlgo::GetInputTensorNum(kernel);
    for (size_t i = 0; i < input_num; ++i) {
      auto address = AnfAlgo::GetPrevNodeMutableOutputAddr(kernel, i);
      MS_EXCEPTION_IF_NULL(address);
      address->ref_count_++;
    }

    auto kernel_mod = AnfAlgo::GetKernelMod(kernel);
    MS_EXCEPTION_IF_NULL(kernel_mod);
    for (size_t i = 0; i < kernel_mod->GetWorkspaceSizeList().size(); ++i) {
      auto address = AnfAlgo::GetWorkspaceAddr(kernel, i);
      MS_EXCEPTION_IF_NULL(address);
      address->ref_count_++;
    }
  }
}

void CPUResourceManager::DecreaseAddressRefCount(const AnfNodePtr &kernel) {
  if (!dynamic_malloc_) {
    return;
  }
  MS_EXCEPTION_IF_NULL(kernel);
  size_t input_num = AnfAlgo::GetInputTensorNum(kernel);
  for (size_t i = 0; i < input_num; ++i) {
    auto address = AnfAlgo::GetPrevNodeMutableOutputAddr(kernel, i);
    MS_EXCEPTION_IF_NULL(address);
    address->ref_count_--;
    if (address->ref_count_ == 0 && address->ptr_ != nullptr) {
      MemFree(address->ptr_);
      address->ptr_ = nullptr;
    }
  }

  auto kernel_mod = AnfAlgo::GetKernelMod(kernel);
  MS_EXCEPTION_IF_NULL(kernel_mod);
  for (size_t i = 0; i < kernel_mod->GetWorkspaceSizeList().size(); ++i) {
    auto address = AnfAlgo::GetWorkspaceAddr(kernel, i);
    MS_EXCEPTION_IF_NULL(address);
    address->ref_count_--;
    if (address->ref_count_ == 0 && address->ptr_ != nullptr) {
      MemFree(address->ptr_);
      address->ptr_ = nullptr;
    }
  }
}
}  // namespace cpu
}  // namespace device
}  // namespace mindspore
