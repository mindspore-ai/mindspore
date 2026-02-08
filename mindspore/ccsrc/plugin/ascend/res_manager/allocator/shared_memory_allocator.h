/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_ALLOCATOR_SHARED_H
#define MINDSPORE_ALLOCATOR_SHARED_H

#include <utility>
#include <unordered_set>
#include <vector>
#include <algorithm>
#include <numeric>
#include <set>
#include <tuple>
#include <memory>
#include <string>
#include <unordered_map>

#include "plugin/ascend/res_manager/hal_manager/ascend_hal_manager.h"
#include "plugin/ascend/res_manager/symbol_interface/acl_base_symbol.h"
#include "plugin/ascend/res_manager/symbol_interface/hal_symbol.h"
#include "plugin/ascend/res_manager/symbol_interface/symbol_utils.h"
#include "plugin/ascend/res_manager/symbol_interface/acl_rt_symbol.h"
#include "include/runtime/hardware_abstract/device_context/device_context.h"

namespace mindspore {
namespace device {
namespace ascend {

enum class RmaDevModel {
  SVM_MAP_DEV,  // 910_93
  PCIE_TH_DEV,  // 910B
  ERROR_CHIP_TYPE
};

class SharedMemoryAllocator : public AddressAllocator {
 public:
  SharedMemoryAllocator() = default;
  virtual ~SharedMemoryAllocator() = default;
  SharedMemoryAllocator(const SharedMemoryAllocator &) = delete;
  SharedMemoryAllocator &operator=(const SharedMemoryAllocator &) = delete;
  static std::shared_ptr<SharedMemoryAllocator> &getInstance();

  void *Alloc(size_t size, uint32_t stream_id) override;
  bool Free(void *address_ptr) override;
  std::tuple<void *, void *> AllocTmp(size_t size, uint32_t stream_id);
  bool FreeTmp(void *registered_ptr, void *ptr);
  void *GetHostPtrByDevicePtr(void *devicePtr) override;

 private:
  static std::shared_ptr<SharedMemoryAllocator> instance;
  std::string socName_ = "unknown";
  std::mutex mutex_;
  RmaDevModel g_rmaDevModel = RmaDevModel::ERROR_CHIP_TYPE;
  std::unordered_map<void *, int> shm_map_;     // ptr -> shm_id
  std::unordered_map<void *, void *> ptr_map_;  // ascend ptr -> cpu ptr
  uint32_t GetRegisterFlag(RmaDevModel mode);

  void *RegisterMem(void *memory, uint64_t datalen, int deviceId);

  RmaDevModel GetRmDevModel();

  // 910B
  void *AllocWithSharedMemory(size_t size);

  // 910_93
  void *AllocWithHostMemory(size_t size);

  bool FreeSharedMemory(void *ptr, int deviceId);

  bool FreeHostMemory(void *ptr);

  bool GetAndRemoveShmId(void *ptr, int *shmId);
};
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
#endif
