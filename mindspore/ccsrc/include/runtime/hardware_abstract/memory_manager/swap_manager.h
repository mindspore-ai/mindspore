/**
 * Copyright 2023-2025 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_INCLUDE_RUNTIMR_HARDWARE_ABSTRACT_MEMORY_MANAGER_SWAP_MANAGER_H_
#define MINDSPORE_CCSRC_INCLUDE_RUNTIMR_HARDWARE_ABSTRACT_MEMORY_MANAGER_SWAP_MANAGER_H_

#include <memory>
#include <queue>
#include <string>
#include <utility>
#include <vector>

#include "include/runtime/memory/mem_pool/dynamic_mem_pool.h"
#include "device_address/device_address.h"
#include "include/runtime/hardware_abstract/memory_manager/pin_mem_pool.h"
#include "runtime/hardware_abstract/visible.h"

namespace mindspore {
namespace device {
class IOHandle;
using IOHandlePtr = std::shared_ptr<IOHandle>;
using AsyncIOToken = size_t;

class RUNTIME_HARDWARE_EXPORT SwapManager {
 public:
  SwapManager(size_t stream_id, DynamicMemPool *device_memory_pool, PinMemPool *pin_mem_pool);
  ~SwapManager() = default;
  // Device memory
  void *AllocDeviceMemory(size_t size, uint32_t stream_id = kDefaultStreamIndex);
  std::vector<void *> AllocDeviceContinuousMem(const std::vector<size_t> &size_list,
                                               uint32_t stream_id = kDefaultStreamIndex);
  void FreeDeviceMemory(void *ptr);

  // Host memory
  void *AllocHostMemory(size_t size, uint32_t stream_id = kDefaultStreamIndex);
  void FreeHostMemory(void *ptr);

  // File
  bool CreateFile(const std::string &file_name, size_t file_size);
  bool DeleteFile(const std::string &file_name);
  bool FileToHostMemory(void *host_memory, const std::string &file_name, size_t byte_num, bool async,
                        AsyncIOToken *sync_token);
  bool HostMemoryToFile(const std::string &file_name, const void *data, size_t byte_num, bool async,
                        AsyncIOToken *sync_token);
  bool WaitAsyncIO(AsyncIOToken sync_token);
  std::string GetSwapFileName(uint32_t device_id) const;

  PinMemPool *GetPinMemPool() { return pin_mem_pool_; }

 private:
  bool EnoughFileSpace(const size_t &size) const;

 private:
  size_t stream_id_;
  DynamicMemPool *device_memory_pool_;
  PinMemPool *pin_mem_pool_;
  size_t max_file_size_{0};
  size_t current_used_file_size_{0};
  HashMap<std::string, size_t> file_size_;
  struct compare {
    bool operator()(const DeviceAddressPtr &l, const DeviceAddressPtr &r) const { return l->GetSize() < r->GetSize(); }
  };
  const size_t size_level_num_{0};
  IOHandlePtr io_handle_;
};
using SwapManagerPtr = std::shared_ptr<SwapManager>;
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_INCLUDE_RUNTIMR_HARDWARE_ABSTRACT_MEMORY_MANAGER_SWAP_MANAGER_H_
