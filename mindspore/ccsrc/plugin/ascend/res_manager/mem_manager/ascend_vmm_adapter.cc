/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#include "plugin/ascend/res_manager/mem_manager/ascend_vmm_adapter.h"
#include <map>
#include <vector>
#include <tuple>

#include "utils/convert_utils_base.h"
#include "utils/ms_utils.h"
#include "plugin/ascend/res_manager/symbol_interface/symbol_utils.h"
#include "plugin/ascend/res_manager/symbol_interface/acl_rt_symbol.h"

namespace mindspore {
namespace device {
namespace ascend {
size_t AscendVmmAdapter::GetRoundUpAlignSize(size_t input_size) const {
  return ((input_size + vmm_align_size_ - 1) / vmm_align_size_) * vmm_align_size_;
}

size_t AscendVmmAdapter::GetRoundDownAlignSize(size_t input_size) const {
  return (input_size / vmm_align_size_) * vmm_align_size_;
}

size_t AscendVmmAdapter::GetHandleSize(size_t input_size) {
  if (input_size % vmm_align_size_ != 0) {
    MS_LOG(EXCEPTION) << "Input size must be multiple of 2MB, but got " << input_size;
  }
  return input_size / vmm_align_size_;
}

DeviceMemPtr AscendVmmAdapter::FindVmmSegment(const DeviceMemPtr addr) {
  auto it = vmm_map_.upper_bound(addr);
  if (it == vmm_map_.begin()) {
    return nullptr;
  } else {
    --it;
    return it->first;
  }
  return nullptr;
}

void AscendVmmAdapter::ClearAllMemory() {
  for (const auto &[addr, handle] : vmm_map_) {
    if (handle == nullptr) {
      continue;
    }
    auto ret = CALL_ASCEND_API(aclrtUnmapMem, addr);
    if (ret != ACL_SUCCESS) {
      MS_LOG(ERROR) << "Unmap memory failed.";
    }
    ret = CALL_ASCEND_API(aclrtFreePhysical, handle);
    if (ret != ACL_SUCCESS) {
      MS_LOG(ERROR) << "Free physical memory failed.";
    }
  }
  while (!cached_handle_sets_.empty()) {
    auto handle = *cached_handle_sets_.begin();
    cached_handle_sets_.erase(cached_handle_sets_.begin());
    auto ret = CALL_ASCEND_API(aclrtFreePhysical, handle);
    if (ret != ACL_SUCCESS) {
      MS_LOG(ERROR) << "Free physical memory failed.";
    }
  }
  for (auto &addr : all_reserve_mems_) {
    CALL_ASCEND_API(aclrtReleaseMemAddress, addr);
  }
  all_reserve_mems_.clear();
  vmm_map_.clear();
}

namespace {
void MoveBackMappedHandle(std::map<DeviceMemPtr, aclrtDrvMemHandle> *mapped_vmm_handle,
                          std::map<DeviceMemPtr, aclrtDrvMemHandle> *vmm_map,
                          std::set<aclrtDrvMemHandle> *cached_handle_sets_) {
  for (const auto [address, handle] : *mapped_vmm_handle) {
    auto ret = CALL_ASCEND_API(aclrtUnmapMem, address);
    if (ret != ACL_SUCCESS) {
      MS_LOG(ERROR) << "Unmap memory failed, address : " << address << ".";
    } else {
      auto iter = vmm_map->find(address);
      if (iter == vmm_map->end()) {
        MS_LOG(ERROR) << "Find vmm map address : " << address << " failed.";
      } else {
        iter->second = nullptr;
        cached_handle_sets_->insert(handle);
      }
    }
  }
}
};  // namespace

size_t AscendVmmAdapter::MmapDeviceMem(const size_t size, const DeviceMemPtr addr, const size_t max_size) {
  if (common::IsCompileSimulation()) {
    MS_LOG(EXCEPTION) << "VMM is not supported in dry run mode.";
  }
  MS_EXCEPTION_IF_NULL(addr);
  MS_LOG(DEBUG) << "VMM MmapDeviceMem size:" << size << ", addr:" << addr
                << ", cached_handle_sets_ size : " << cached_handle_sets_.size() << ".";
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  auto device_id = context->get_param<uint32_t>(MS_CTX_DEVICE_ID);

  auto vmm_start_addr = FindVmmSegment(addr);
  if (vmm_start_addr == nullptr) {
    MS_LOG(ERROR) << "Can not find the vmm segment.";
    return 0;
  }
  aclrtPhysicalMemProp prop = {};
  prop.handleType = ACL_MEM_HANDLE_TYPE_NONE;
  prop.allocationType = ACL_MEM_ALLOCATION_TYPE_PINNED;
  prop.memAttr = enable_mem_huge_1g_ ? ACL_HBM_MEM_HUGE1G : ACL_HBM_MEM_HUGE;
  prop.location.type = ACL_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = device_id;
  prop.reserve = 0;
  auto start_offset = CalAddressOffset(addr, vmm_start_addr);
  auto align_size = GetRoundUpAlignSize(size + start_offset);
  auto handle_size = GetHandleSize(align_size);
  auto iter = vmm_map_.find(vmm_start_addr);

  std::map<DeviceMemPtr, aclrtDrvMemHandle> mapped_vmm_handle;
  for (size_t i = 0; i < handle_size; ++i) {
    auto new_addr = AddressOffset(vmm_start_addr, i * vmm_align_size_);
    if (iter == vmm_map_.end() || iter->first != new_addr) {
      MS_LOG(ERROR) << "Can not find the vmm segment.";
      return 0;
    }
    if (iter->second != nullptr) {
      iter++;
      continue;
    }
    aclrtDrvMemHandle handle = nullptr;
    if (!cached_handle_sets_.empty()) {
      handle = *cached_handle_sets_.begin();
      cached_handle_sets_.erase(cached_handle_sets_.begin());
    } else {
      if (physical_handle_size_ * vmm_align_size_ >= max_size) {
        MS_LOG(INFO) << "Mapped too much memory, physical_handle_size_ : " << physical_handle_size_
                     << ", max_size : " << max_size << ", addr : " << addr << ", size : " << size << ".";
        MoveBackMappedHandle(&mapped_vmm_handle, &vmm_map_, &cached_handle_sets_);
        return 0;
      }
      if (!AllocPhysicalMem(&handle, vmm_align_size_, &prop, 0, mapped_vmm_handle)) {
        return 0;
      }
      ++physical_handle_size_;
      if (physical_handle_size_ * vmm_align_size_ >= max_size) {
        MS_LOG(WARNING) << "Mapped too much memory, physical_handle_size_ : " << physical_handle_size_
                        << ", max_size : " << max_size << ".";
      }
    }

    auto ret = CALL_ASCEND_API(aclrtMapMem, new_addr, vmm_align_size_, 0, handle, 0);
    if (ret != ACL_SUCCESS) {
      MS_LOG(ERROR) << "Map memory failed, return code " << ret;
      cached_handle_sets_.insert(handle);
      MoveBackMappedHandle(&mapped_vmm_handle, &vmm_map_, &cached_handle_sets_);
      return 0;
    }
    mapped_vmm_handle[iter->first] = handle;
    iter->second = handle;
    iter++;
  }

  static bool enable_trace_mem = memory::mem_pool::IsEnableMemTrack();
  if (enable_trace_mem) {
    MS_LOG(INFO) << "Total physical memory handle size : " << physical_handle_size_ << ".";
  }
  return size;
}

size_t AscendVmmAdapter::AllocDeviceMem(size_t size, DeviceMemPtr *addr) {
  MS_EXCEPTION_IF_NULL(addr);
  size_t align_size = GetRoundUpAlignSize(size);
  MS_LOG(INFO) << "VMM AllocDeviceMem size:" << size << ", align_size:" << align_size;
  auto ret = CALL_ASCEND_API(aclrtReserveMemAddress, addr, align_size, 0, nullptr, 1);
  if (ret != ACL_SUCCESS) {
    MS_LOG(ERROR) << "Reserve memory address failed.";
    return 0;
  }
  all_reserve_mems_.push_back(*addr);
  auto handle_size = GetHandleSize(align_size);
  for (size_t i = 0; i < handle_size; i++) {
    auto new_addr = AddressOffset(*addr, i * vmm_align_size_);
    vmm_map_[new_addr] = nullptr;
  }
  return align_size;
}

size_t AscendVmmAdapter::EagerFreeDeviceMem(const DeviceMemPtr addr, const size_t size) {
  MS_EXCEPTION_IF_NULL(addr);
  if (common::IsCompileSimulation()) {
    MS_LOG(EXCEPTION) << "VMM is not supported in dry run mode.";
  }
  MS_LOG(DEBUG) << "Eager free device mem addr :" << addr << ", size :" << size
                << ", cached_handle_sets_ size : " << cached_handle_sets_.size() << ".";
  size_t ret_size = 0;
  auto iter = vmm_map_.lower_bound(addr);
  if (iter == vmm_map_.end()) {
    // Memory less than 2MB may be at the end of a vmm segment, and it's a normal case.
    if (size >= vmm_align_size_) {
      MS_LOG(ERROR) << "Can not find the vmm segment.";
    }
    return 0;
  }
  auto vmm_start_addr = iter->first;
  auto free_end_addr = AddressOffset(addr, size);
  while (true) {
    auto vmm_end_addr = AddressOffset(vmm_start_addr, vmm_align_size_);
    if (vmm_end_addr > free_end_addr) {
      break;
    }
    if (iter == vmm_map_.end() || iter->first != vmm_start_addr) {
      MS_LOG(ERROR) << "Can not find the vmm segment.";
      return 0;
    }
    if (iter->second == nullptr) {
      iter++;
      vmm_start_addr = vmm_end_addr;
      // Here nullptr may be huge, skip do logging.
      continue;
    }
    auto ret = CALL_ASCEND_API(aclrtUnmapMem, vmm_start_addr);
    if (ret != ACL_SUCCESS) {
      MS_LOG(ERROR) << "Unmap memory failed.";
      return 0;
    }
    cached_handle_sets_.insert(iter->second);
    iter->second = nullptr;
    iter++;
    vmm_start_addr = vmm_end_addr;
    ret_size += vmm_align_size_;
  }
  MS_LOG(DEBUG) << "After eager free, cached_handle_sets_ size : " << cached_handle_sets_.size()
                << ", expected free size : " << size << ", real size : " << ret_size << ".";
  return ret_size;
}

size_t AscendVmmAdapter::EmptyCache() {
  size_t empty_size = 0L;
  for (auto iter = cached_handle_sets_.begin(); iter != cached_handle_sets_.end(); iter++) {
    auto ret = CALL_ASCEND_API(aclrtFreePhysical, *iter);
    if (ret != ACL_SUCCESS) {
      MS_LOG(EXCEPTION) << "Free physical memory failed.";
    }
  }

  size_t cache_handle_size = cached_handle_sets_.size();
  physical_handle_size_ -= cache_handle_size;
  empty_size += cache_handle_size * vmm_align_size_;
  cached_handle_sets_.clear();
  MS_LOG(INFO) << "Empty cache size: " << empty_size << ", cached handle set size: " << cached_handle_sets_.size()
               << ".";
  return empty_size;
}

bool AscendVmmAdapter::AllocPhysicalMem(aclrtDrvMemHandle *handle, size_t size, aclrtPhysicalMemProp *prop,
                                        uint64_t flags, std::map<DeviceMemPtr, aclrtDrvMemHandle> &mapped_vmm_handle) {
  auto ret = CALL_ASCEND_API(aclrtMallocPhysical, handle, size, prop, flags);
  if (prop->memAttr == ACL_HBM_MEM_HUGE1G && ret != ACL_RT_SUCCESS) {
    if (ret == ACL_ERROR_RT_FEATURE_NOT_SUPPORT) {
      const auto soc_name = CALL_ASCEND_API(aclrtGetSocName);
      MS_LOG(EXCEPTION)
        << "Current device " << soc_name
        << " does not support ACL_HBM_MEM_HUGE1G (1G huge page memory)."
           " Please visit https://www.hiascend.com/document/detail/zh/CANNCommunityEdition for more information.";
    } else {
      MS_LOG(WARNING) << "Malloc 1G physical memory failed, error code " << ret
                      << ". Try to malloc 2M physical memory.";
      aclrtPhysicalMemProp new_prop = {prop->handleType,
                                       prop->allocationType,
                                       ACL_HBM_MEM_HUGE,
                                       {prop->location.id, prop->location.type},
                                       prop->reserve};
      ret = CALL_ASCEND_API(aclrtMallocPhysical, handle, size, &new_prop, flags);
    }
  }

  if (ret != ACL_SUCCESS) {
    size_t used_handle_size = 0;
    for (const auto &[addr, handle] : vmm_map_) {
      if (handle != nullptr) {
        MS_LOG(DEBUG) << "Inuse handle address : " << addr << ", handle : " << handle << ".";
        used_handle_size += 1;
      }
    }
    used_handle_size += cached_handle_sets_.size();
    // This may be a normal case at the memory usage boundary.
    MS_LOG(WARNING) << "Malloc physical memory failed, return code " << ret
                    << ", inuse physical memory handle size : " << used_handle_size
                    << ", physical_handle_size_ size : " << physical_handle_size_ << ".";
    MoveBackMappedHandle(&mapped_vmm_handle, &vmm_map_, &cached_handle_sets_);
    return false;
  }
  return true;
}
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
