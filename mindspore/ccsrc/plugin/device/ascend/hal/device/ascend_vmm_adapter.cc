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
#include "plugin/device/ascend/hal/device/ascend_vmm_adapter.h"
#include <map>
#include <vector>
#include <tuple>

#include "utils/convert_utils_base.h"
#include "utils/ms_utils.h"
#include "transform/symbol/symbol_utils.h"
#include "transform/symbol/acl_rt_symbol.h"

namespace mindspore {
namespace device {
namespace ascend {
size_t AscendVmmAdapter::GetRoundUpAlignSize(size_t input_size) const {
  return ((input_size + kVmmAlignSize - 1) / kVmmAlignSize) * kVmmAlignSize;
}

size_t AscendVmmAdapter::GetRoundDownAlignSize(size_t input_size) const {
  return (input_size / kVmmAlignSize) * kVmmAlignSize;
}

size_t AscendVmmAdapter::GetHandleSize(size_t input_size) {
  if (input_size % kVmmAlignSize != 0) {
    MS_LOG(EXCEPTION) << "Input size must be multiple of 2MB, but got " << input_size;
  }
  return input_size / kVmmAlignSize;
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
  for (auto &kv : vmm_map_) {
    if (kv.second == nullptr) {
      continue;
    }
    auto ret = CALL_ASCEND_API(aclrtUnmapMem, kv.first);
    if (ret != ACL_ERROR_NONE) {
      MS_LOG(ERROR) << "Unmap memory failed.";
    }
    ret = CALL_ASCEND_API(aclrtFreePhysical, kv.second);
    if (ret != ACL_ERROR_NONE) {
      MS_LOG(ERROR) << "Free physical memory failed.";
    }
  }
  while (!handle_queue_.empty()) {
    auto handle = handle_queue_.front();
    handle_queue_.pop();
    auto ret = CALL_ASCEND_API(aclrtFreePhysical, handle);
    if (ret != ACL_ERROR_NONE) {
      MS_LOG(ERROR) << "Free physical memory failed.";
    }
  }
  for (auto &addr : all_reserve_mems_) {
    CALL_ASCEND_API(aclrtReleaseMemAddress, addr);
  }
  all_reserve_mems_.clear();
  vmm_map_.clear();
}

AscendVmmAdapter::~AscendVmmAdapter() { ClearAllMemory(); }

size_t AscendVmmAdapter::MmapDeviceMem(const size_t size, const DeviceMemPtr addr, const size_t max_size) {
  MS_EXCEPTION_IF_NULL(addr);
  MS_LOG(DEBUG) << "VMM MmapDeviceMem size:" << size << ", addr:" << addr
                << ", handle_queue_ size : " << handle_queue_.size() << ".";
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
  prop.memAttr = ACL_HBM_MEM_HUGE;
  prop.location.type = ACL_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = device_id;
  prop.reserve = 0;
  auto start_offset = CalAddressOffset(addr, vmm_start_addr);
  auto align_size = GetRoundUpAlignSize(size + start_offset);
  auto handle_size = GetHandleSize(align_size);
  auto iter = vmm_map_.find(vmm_start_addr);

  std::map<DeviceMemPtr, aclrtDrvMemHandle> mapped_vmm_handle;
  for (size_t i = 0; i < handle_size; ++i) {
    auto new_addr = AddressOffset(vmm_start_addr, i * kVmmAlignSize);
    if (iter == vmm_map_.end() || iter->first != new_addr) {
      MS_LOG(ERROR) << "Can not find the vmm segment.";
      return 0;
    }
    if (iter->second != nullptr) {
      iter++;
      continue;
    }
    aclrtDrvMemHandle handle = nullptr;
    if (!handle_queue_.empty()) {
      handle = handle_queue_.front();
      handle_queue_.pop();
    } else {
      if (physical_handle_size_ * kVmmAlignSize >= max_size) {
        MS_LOG(INFO) << "Mapped too much memory, physical_handle_size_ : " << physical_handle_size_
                     << ", max_size : " << max_size << ", addr : " << addr << ", size : " << size << ".";
        for (const auto [device_mem_ptr, handle] : mapped_vmm_handle) {
          vmm_map_[device_mem_ptr] = nullptr;
          handle_queue_.push(handle);
        }
        return 0;
      }

      auto ret = CALL_ASCEND_API(aclrtMallocPhysical, &handle, kVmmAlignSize, &prop, 0);
      if (ret != ACL_ERROR_NONE) {
        if (common::IsNeedProfileMemory()) {
          return size;
        }
        size_t used_handle_size = 0;
        for (const auto &[k, v] : vmm_map_) {
          if (v != nullptr) {
            MS_LOG(DEBUG) << "Inuse handle address : " << k << ", handle : " << v << ".";
            used_handle_size += 1;
          }
        }
        used_handle_size += handle_queue_.size();
        MS_LOG(ERROR) << "Malloc physical memory failed, inuse physical memory handle size : " << used_handle_size
                      << ", physical_handle_size_ size : " << physical_handle_size_ << ".";
        return 0;
      } else {
        physical_handle_size_++;
        if (physical_handle_size_ * kVmmAlignSize >= max_size) {
          MS_LOG(WARNING) << "Mapped too much memory, physical_handle_size_ : " << physical_handle_size_
                          << ", max_size : " << max_size << ".";
        }
      }
    }

    auto ret = CALL_ASCEND_API(aclrtMapMem, new_addr, kVmmAlignSize, 0, handle, 0);
    if (ret != ACL_ERROR_NONE) {
      if (common::IsNeedProfileMemory()) {
        return size;
      }
      MS_LOG(ERROR) << "Map memory failed.";
      handle_queue_.push(handle);
      return 0;
    }
    mapped_vmm_handle[new_addr] = handle;
    iter->second = handle;
    iter++;
  }

  static bool enable_trace_mem = common::IsEnableAlllocConfig(common::kAllocMemoryTracker);
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
  if (ret != ACL_ERROR_NONE) {
    MS_LOG(ERROR) << "Reserve memory address failed.";
    return 0;
  }
  all_reserve_mems_.push_back(*addr);
  auto handle_size = GetHandleSize(align_size);
  for (size_t i = 0; i < handle_size; i++) {
    auto new_addr = AddressOffset(*addr, i * kVmmAlignSize);
    vmm_map_[new_addr] = nullptr;
  }
  return align_size;
}

size_t AscendVmmAdapter::EagerFreeDeviceMem(const DeviceMemPtr addr, const size_t size) {
  MS_LOG(DEBUG) << "Eager free device mem addr :" << addr << ", size :" << size
                << ", handle_queue_ size : " << handle_queue_.size() << ".";
  if (common::IsNeedProfileMemory()) {
    return size;
  }

  size_t ret_size = 0;
  auto iter = vmm_map_.lower_bound(addr);
  if (iter == vmm_map_.end()) {
    MS_LOG(ERROR) << "Can not find the vmm segment.";
    return 0;
  }
  auto vmm_start_addr = iter->first;
  auto free_end_addr = AddressOffset(addr, size);
  while (true) {
    auto vmm_end_addr = AddressOffset(vmm_start_addr, kVmmAlignSize);
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
    if (ret != ACL_ERROR_NONE) {
      MS_LOG(ERROR) << "Unmap memory failed.";
      return 0;
    }
    handle_queue_.push(iter->second);
    iter->second = nullptr;
    iter++;
    vmm_start_addr = vmm_end_addr;
    ret_size += kVmmAlignSize;
  }
  MS_LOG(DEBUG) << "After eager free, handle_queue_ size : " << handle_queue_.size()
                << ", expected free size : " << size << ", real size : " << ret_size << ".";
  return ret_size;
}
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
