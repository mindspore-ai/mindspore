/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/hal/device/cpu_memory_manager.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "utils/ms_context.h"
#include "include/common/utils/convert_utils.h"
namespace mindspore {
namespace device {
namespace cpu {
uint8_t *CPUMemoryManager::MemMalloc(size_t size) {
  auto block = std::make_shared<std::vector<uint8_t>>();
  try {
    block->resize(size, 0);
    auto ptr = block->data();
    mem_block_map_[ptr] = block;
    return ptr;
  } catch (const std::exception &e) {
    MS_LOG(EXCEPTION) << "Malloc memory failed: size " << size;
  }
}

uint8_t *CPUMemoryManager::MallocStaticMem(size_t size, bool, uint32_t) {
  auto ptr = MemMalloc(size);
  static_mem_[ptr] = size;
  return ptr;
}

uint8_t *CPUMemoryManager::MallocDynamicMem(size_t size, bool) {
  void *ptr = nullptr;
  size_t min_size = 0;
  // first find the smallest cached_mem_ which fits the size
  for (auto &&iter : cached_mem_) {
    if (iter.second >= size) {
      if (min_size == 0 || iter.second < min_size) {
        ptr = iter.first;
        min_size = iter.second;
      }
    }
  }
  if (ptr != nullptr) {
    if (memset_s(ptr, size, 0, size) != EOK) {
      free(ptr);
      MS_LOG(EXCEPTION) << "Failed to init memory.";
    }
    dynamic_mem_[ptr] = min_size;
    (void)cached_mem_.erase(ptr);
    return reinterpret_cast<uint8_t *>(ptr);
  }
  // if not found, malloc
  auto new_ptr = MemMalloc(size);
  dynamic_mem_[new_ptr] = size;
  return new_ptr;
}

void CPUMemoryManager::ResetDynamicMemory() {
  // don't free, for multi graph
  for (auto &&iter : dynamic_mem_) {
    cached_mem_[iter.first] = iter.second;
  }
  dynamic_mem_.clear();
}

CPUMemoryManager::~CPUMemoryManager() { MemFree(); }

void CPUMemoryManager::MemFree() noexcept {
  if (mem_ptr_ != nullptr) {
    mem_ptr_ = nullptr;
    mem_size_ = 0;
  }
  static_mem_.clear();
  dynamic_mem_.clear();
  cached_mem_.clear();
  mem_block_map_.clear();
}

void CPUMemoryManager::AssignMemory(const session::KernelGraph *graph) {
  size_t graph_mem_size = mem_plan_.MemPlan(graph);
  if (graph_mem_size > mem_size_) {
    if (mem_size_ > 0) {
      dynamic_mem_[mem_ptr_] = mem_size_;
      mem_size_ = 0;
    }
    mem_ptr_ = MemMalloc(graph_mem_size);
    if (mem_ptr_ != nullptr) {
      MS_LOG(INFO) << "Simple MemPlan GraphMemSize [" << graph_mem_size << "]";
      mem_size_ = graph_mem_size;
      dynamic_malloc_ = false;
    } else {
      MS_LOG(INFO) << "Switch to dynamic malloc";
      dynamic_malloc_ = true;
    }
  }
  if (dynamic_malloc_) {
    return;
  }
  mem_plan_.MemAssign(graph, mem_ptr_);
}

void *CPUMemoryManager::StaticMemMalloc(size_t mem_size) {
  auto ptr = MemMalloc(mem_size);
  if (ptr != nullptr) {
    static_mem_[ptr] = mem_size;
    return ptr;
  } else {
    MS_LOG(EXCEPTION) << "Malloc memory failed: size " << mem_size;
  }
}

void CPUMemoryManager::MemFree(void *ptr) {
  auto iter = static_mem_.find(ptr);
  if (iter != static_mem_.end()) {
    (void)static_mem_.erase(iter);
    auto block_iter = mem_block_map_.find(ptr);
    if (block_iter != mem_block_map_.end()) {
      (void)mem_block_map_.erase(block_iter);
    }
  }
}

void CPUMemoryManager::IncreaseSummaryRefCount(const session::NamedSummaryOutputs &summary_outputs) const {
  if (!dynamic_malloc_) {
    return;
  }
  if (summary_outputs.empty()) {
    return;
  }
  for (auto &output_item : summary_outputs) {
    auto node = output_item.second.first;
    size_t index = IntToSize(output_item.second.second);
    auto address = AnfAlgo::GetMutableOutputAddr(node, index);
    MS_EXCEPTION_IF_NULL(address);
    address->ref_count_++;
  }
}

void CPUMemoryManager::DecreaseSummaryRefCount(const session::NamedSummaryOutputs &summary_outputs) {
  if (!dynamic_malloc_) {
    return;
  }
  if (summary_outputs.empty()) {
    return;
  }
  for (auto &output_item : summary_outputs) {
    auto node = output_item.second.first;
    size_t index = IntToSize(output_item.second.second);
    auto address = AnfAlgo::GetMutableOutputAddr(node, index);
    MS_EXCEPTION_IF_NULL(address);
    address->ref_count_--;
    if (address->ref_count_ == 0 && address->ptr_ != nullptr) {
      MemFree(address->ptr_);
      address->ptr_ = nullptr;
    }
  }
}

void CPUMemoryManager::IncreaseAddressRefCount(const session::KernelGraph *graph) const {
  if (!dynamic_malloc_) {
    return;
  }
  MS_EXCEPTION_IF_NULL(graph);
  auto kernels = graph->execution_order();
  for (const auto &kernel : kernels) {
    MS_EXCEPTION_IF_NULL(kernel);
    size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel);
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

void CPUMemoryManager::DecreaseAddressRefCount(const AnfNodePtr &kernel) {
  if (!dynamic_malloc_) {
    return;
  }
  MS_EXCEPTION_IF_NULL(kernel);
  size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel);
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
