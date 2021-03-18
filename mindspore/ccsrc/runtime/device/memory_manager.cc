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

#include "runtime/device/memory_manager.h"
#include <string>
#include "backend/session/anf_runtime_algorithm.h"
#ifdef ENABLE_DUMP_IR
#include "debug/rdr/running_data_recorder.h"
#endif
#include "utils/ms_context.h"

using mindspore::memreuse::BestFitMemReuse;
using mindspore::memreuse::MemReuseUtilPtr;

namespace mindspore {
namespace device {
size_t MemoryManager::GetCommonAlignSize(size_t input_size) {
  return (input_size + kMemAlignSize + 31) / kMemAlignSize * kMemAlignSize;
}

size_t MemoryManager::GetCommunicationAlignSize(size_t input_size) const {
  return (input_size + kMemAlignSize - 1) / kMemAlignSize * kMemAlignSize + 2 * kMemAlignSize;
}

void MemoryManager::MallocReusedDynamicMem(const session::KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(graph);
  MemReuseUtilPtr mem_reuse_util_ptr = std::make_shared<memreuse::MemReuseUtil>();
  MS_EXCEPTION_IF_NULL(mem_reuse_util_ptr);
  // set all infos
  mem_reuse_util_ptr->SetAllInfo(graph);
  auto bestfit_mem_reuse = std::make_shared<BestFitMemReuse>();
  MS_EXCEPTION_IF_NULL(bestfit_mem_reuse);
  bestfit_mem_reuse->Reuse(mem_reuse_util_ptr.get());
  size_t total_allocated_size = bestfit_mem_reuse->GetAllocatedSize();
  MS_LOG(INFO) << "TotalReuseDynamicSize [" << total_allocated_size << "]";
  mem_reuse_util_ptr_ = mem_reuse_util_ptr;
  auto base_ptr = MallocDynamicMem(total_allocated_size, false);
  MS_LOG(INFO) << "Reuse Memory from [" << reinterpret_cast<void *>(base_ptr) << "] to ["
               << reinterpret_cast<void *>(base_ptr + total_allocated_size) << "]";
  mem_reuse_util_ptr_->set_mem_base(base_ptr);
}

void MemoryManager::MallocSomasDynamicMem(const session::KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(graph);
  SomasPtr somas_reuse_util_ptr = std::make_shared<somas::Somas>();
  MS_EXCEPTION_IF_NULL(somas_reuse_util_ptr);
  somas_reuse_util_ptr_ = somas_reuse_util_ptr;

  if (!(somas_reuse_util_ptr->Allocate(graph))) {
    MS_LOG(EXCEPTION) << "Somas Allocate Failed.";
  }

  size_t total_allocated_size = somas_reuse_util_ptr->GetTotalMemSize();
  MS_LOG(INFO) << "Graph " << graph->graph_id() << ": TotalSomasReuseDynamicSize [" << total_allocated_size << "]";
  auto base_ptr = MallocDynamicMem(total_allocated_size, false);
  MS_LOG(INFO) << "Somas Reuse Memory Base Address [" << static_cast<void *>(base_ptr) << "], End Address ["
               << static_cast<void *>(base_ptr + total_allocated_size) << "]";
  somas_reuse_util_ptr->set_mem_base_addr(base_ptr);

  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
#ifdef ENABLE_DUMP_IR
  SubModuleId module = SubModuleId::SM_OPTIMIZER;

  std::string name = "somas_allocate_info." + std::to_string(graph->graph_id());
  mindspore::RDR::RecordString(module, name, somas_reuse_util_ptr_->SomasInfo());

  name = "somas_mem_info." + std::to_string(graph->graph_id());
  mindspore::RDR::RecordString(module, name, somas_reuse_util_ptr_->SomasMemory());
#endif
  bool save_graphs = context_ptr->get_param<bool>(MS_CTX_SAVE_GRAPHS_FLAG);
  auto save_graphs_path = context_ptr->get_param<std::string>(MS_CTX_SAVE_GRAPHS_PATH);
  if (save_graphs_path.empty()) {
    save_graphs_path = ".";
  }
  if (save_graphs) {
    std::string file_path = save_graphs_path + "/" + "somas_allocate_info_" + std::to_string(graph->graph_id()) + ".ir";
    somas_reuse_util_ptr_->DumpSomasInfoIR(file_path);

    std::string mem_file_path = save_graphs_path + "/" + "somas_mem_info_" + std::to_string(graph->graph_id()) + ".ir";
    somas_reuse_util_ptr_->DumpSomasMemoryIR(mem_file_path);
  }
}

uint8_t *MemoryManager::MallocOutputMem(const AnfNodePtr &node, size_t index, MemType type, size_t size,
                                        const DeviceAddressPtr &address, bool comm_mem) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(address);
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  uint8_t *ptr = nullptr;
  if (comm_mem) {
    bool communication_mem = false;
    if (context_ptr->get_param<bool>(MS_CTX_ENABLE_HCCL)) {
      communication_mem = true;
    }
    if (type == kStaticMem) {
      ptr = MallocStaticMem(size, communication_mem);
      address->from_mem_pool_ = true;
      if (communication_mem) {
        address->communication_ptr_ = ptr - kMemAlignSize;
      }
    } else if (type == kReuseDynamicCommMem) {
      MS_EXCEPTION_IF_NULL(mem_reuse_util_ptr_);
      ptr = mem_reuse_util_ptr_->GetNodeOutputPtr(node, index);
    } else if (type == kSomasReuseDynamicMem) {
      MS_EXCEPTION_IF_NULL(somas_reuse_util_ptr_);
      ptr = somas_reuse_util_ptr_->GetNodeOutputPtr(node, index);
    } else {
      ptr = MallocDynamicMem(size, communication_mem);
    }
    address->ptr_ = ptr;
    return ptr;
  }

  if (type == kStaticMem) {
    ptr = MallocStaticMem(size, false);
    address->from_mem_pool_ = true;
  } else if (type == kDynamicMem) {
    ptr = MallocDynamicMem(size, false);
  } else if (type == kReuseDynamicMem) {
    MS_EXCEPTION_IF_NULL(mem_reuse_util_ptr_);
    ptr = mem_reuse_util_ptr_->GetNodeOutputPtr(node, index);
  } else if (type == kSomasReuseDynamicMem) {
    MS_EXCEPTION_IF_NULL(somas_reuse_util_ptr_);
    ptr = somas_reuse_util_ptr_->GetNodeOutputPtr(node, index);
  }
  address->ptr_ = ptr;
  return ptr;
}

uint8_t *MemoryManager::MallocWorkSpaceMem(const AnfNodePtr &node, size_t index, MemType type, size_t size) {
  if (type == kReuseDynamicMem) {
    MS_EXCEPTION_IF_NULL(mem_reuse_util_ptr_);
    return mem_reuse_util_ptr_->GetNodeWorkSpacePtr(node, index);
  } else if (type == kSomasReuseDynamicMem) {
    MS_EXCEPTION_IF_NULL(somas_reuse_util_ptr_);
    return somas_reuse_util_ptr_->GetNodeWorkSpacePtr(node, index);
  }
  return MallocDynamicMem(size, false);
}

uint8_t *MemoryManager::MallocMem(MemType type, size_t size, const DeviceAddressPtr &address, uint32_t graph_id) {
  MS_EXCEPTION_IF_NULL(address);
  uint8_t *ptr = nullptr;
  if (type == kStaticMem) {
    ptr = MallocStaticMem(size, false, graph_id);
    address->from_mem_pool_ = true;
  } else if (type == kDynamicMem) {
    ptr = MallocDynamicMem(size, false);
  }
  address->ptr_ = ptr;
  return ptr;
}

uint8_t *MemoryManager::MallocStaticMem(size_t size, bool communication_mem, uint32_t graph_id) {
  size_t align_size = 0;
  if (communication_mem) {
    align_size = GetCommunicationAlignSize(size);
  } else {
    align_size = GetCommonAlignSize(size);
  }

  MS_LOG(INFO) << "Malloc Memory for Static: total[" << device_mem_size_ << "](dynamic[" << total_dynamic_size_
               << "] static[" << total_static_size_ << "])"
               << " malloc [" << align_size << "] communication_mem: " << communication_mem;

  if (static_mem_offset_ < align_size) {
    MS_LOG(EXCEPTION) << "Out of memory!!! total[" << device_mem_size_ << "](dynamic[" << total_dynamic_size_
                      << "] static[" << total_static_size_ << "])"
                      << " malloc [" << align_size << "] failed!";
  }
  total_static_size_ += align_size;
  auto offset = static_mem_offset_ - align_size;
  if (dynamic_mem_offset_ > offset) {
    MS_LOG(EXCEPTION) << "Out of memory!!! total[" << device_mem_size_ << "](dynamic[" << total_dynamic_size_
                      << "] static[" << total_static_size_ << "])"
                      << " malloc [" << align_size << "] failed!";
  }
  static_mem_offset_ = offset;
  if (communication_mem) {
    return device_mem_base_ + offset + kMemAlignSize;
  } else {
    return device_mem_base_ + offset;
  }
}

uint8_t *MemoryManager::MallocDynamicMem(size_t size, bool communication_mem) {
  size_t align_size = 0;
  if (communication_mem) {
    align_size = GetCommunicationAlignSize(size);
  } else {
    align_size = GetCommonAlignSize(size);
  }

  MS_LOG(INFO) << "Malloc Memory for Dynamic: total[" << device_mem_size_ << "](dynamic[" << total_dynamic_size_
               << "] static[" << total_static_size_ << "])"
               << " malloc [" << align_size << "] communication_mem: " << communication_mem;

  uint64_t offset = dynamic_mem_offset_;
  auto new_offset = dynamic_mem_offset_ + align_size;
  if (new_offset > static_mem_offset_) {
    MS_LOG(EXCEPTION) << "Out of memory!!! total[" << device_mem_size_ << "](dynamic[" << total_dynamic_size_
                      << "] static[" << total_static_size_ << "])"
                      << " malloc [" << align_size << "] failed!";
  }
  total_dynamic_size_ += align_size;
  dynamic_mem_offset_ = new_offset;

  if (communication_mem) {
    return device_mem_base_ + offset + kMemAlignSize;
  } else {
    return device_mem_base_ + offset;
  }
}

bool MemoryManager::MallocMemFromMemPool(const DeviceAddressPtr address, size_t size) {
  auto device_ptr = MallocMemFromMemPool(size);
  if (!device_ptr) {
    return false;
  }
  address->ptr_ = device_ptr;
  address->size_ = size;
  address->from_mem_pool_ = true;
  return true;
}

void *MemoryManager::MallocMemFromMemPool(size_t size) {
  if (size == 0) {
    MS_LOG(ERROR) << "MallocMemFromMemPool size is 0.";
  }
  return nullptr;
}

void MemoryManager::FreeMemFromMemPool(const DeviceAddressPtr address) {
  MS_EXCEPTION_IF_NULL(address);
  MS_EXCEPTION_IF_NULL(address->ptr_);
  FreeMemFromMemPool(address->ptr_);
  address->ptr_ = nullptr;
}

void MemoryManager::FreeMemFromMemPool(void *device_ptr) {
  if (device_ptr == nullptr) {
    MS_LOG(ERROR) << "FreeMemFromMemPool device_ptr is null.";
  }
}

bool MemoryManager::MallocContinuousMemFromMemPool(const DeviceAddressPtrList addr_list, size_t total_size,
                                                   std::vector<size_t> size_list) {
  auto device_ptr_list = MallocContinuousMemFromMemPool(total_size, size_list);
  if (device_ptr_list.size() == 0) {
    return false;
  }
  if (addr_list.size() != device_ptr_list.size()) {
    MS_LOG(EXCEPTION) << "The size of device list is not equal to the size of address list.";
  }
  for (size_t i = 0; i < addr_list.size(); i++) {
    MS_EXCEPTION_IF_NULL(device_ptr_list[i]);
    MS_EXCEPTION_IF_NULL(addr_list[i]);
    addr_list[i]->ptr_ = device_ptr_list[i];
    addr_list[i]->from_mem_pool_ = true;
  }
  return true;
}

std::vector<void *> MemoryManager::MallocContinuousMemFromMemPool(size_t total_size, std::vector<size_t> size_list) {
  if (total_size == 0) {
    MS_LOG(ERROR) << "MallocContinuousMemFromMemPool total_size is 0.";
  }
  std::vector<void *> device_ptr_list;
  device_ptr_list.emplace_back(nullptr);
  return device_ptr_list;
}
}  // namespace device
}  // namespace mindspore
