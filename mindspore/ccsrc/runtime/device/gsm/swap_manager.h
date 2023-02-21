/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_RUNTIME_DEVICE_GSM_SWAP_MANAGER_H_
#define MINDSPORE_CCSRC_RUNTIME_DEVICE_GSM_SWAP_MANAGER_H_

#include <memory>
#include <queue>
#include <string>
#include <vector>

#include "common/mem_reuse/mem_dynamic_allocator.h"
#include "runtime/device/device_address.h"
#include "runtime/device/gsm/io_handle.h"
#include "runtime/device/gsm/host_mem_pool.h"
#include "include/backend/visible.h"

namespace mindspore {
namespace device {
class BACKEND_EXPORT SwapManager {
 public:
  SwapManager(size_t stream_id, DynamicMemPoolBestFit *device_memory_pool, HostMemPoolPtr host_memory_pool);
  ~SwapManager() = default;
  // Device memory
  void *AllocDeviceMemory(size_t size);
  std::vector<void *> AllocDeviceContinuousMem(const std::vector<size_t> &size_list);
  void FreeDeviceMemory(void *ptr);

  // Host memory
  void *AllocHostMemory(size_t size);
  void FreeHostMemory(void *ptr);

  // File
  bool CreateFile(const std::string &file_name);
  bool DeleteFile(const std::string &file_name);
  bool FileToHostMemory(void *host_memory, const std::string &file_name, size_t byte_num, bool async,
                        AsyncIOToken *sync_token);
  bool HostMemoryToFile(const std::string &file_name, const void *data, size_t byte_num, bool async,
                        AsyncIOToken *sync_token);
  bool WaitAsyncIO(AsyncIOToken sync_token);

  // Swapping and swappable tensors
  void AddSwappableTensor(const DeviceAddressPtr &device_address);
  void AddSwappingTensor(const DeviceAddress *device_address);

 private:
  void *AllocDeviceMemorySimply(const size_t &size);
  std::vector<void *> AllocDeviceContinuousMemSimply(const std::vector<size_t> &size_list);
  void *AllocHostMemorySimply(const size_t &size);
  bool EnoughFileSpace(const size_t &size);

  template <class Input, class Output>
  bool TryAllocate(std::queue<const DeviceAddress *> queue, const Input &input,
                   Output (SwapManager::*allocate_func)(const Input &), const std::function<bool(Output)> &success,
                   Output *output);

 private:
  size_t stream_id_;
  DynamicMemPoolBestFit *device_memory_pool_;
  HostMemPoolPtr host_memory_pool_;
  size_t max_file_size_{0};
  size_t current_used_file_size_{0};
  HashMap<std::string, size_t> file_size_;
  struct compare {
    bool operator()(const DeviceAddressPtr &l, const DeviceAddressPtr &r) const { return l->GetSize() < r->GetSize(); }
  };
  std::priority_queue<DeviceAddressPtr, std::vector<DeviceAddressPtr>, compare> swappable_tensors_;
  std::mutex swapping_tensors_device_mutex_;
  std::queue<const DeviceAddress *> swapping_tensors_device_;
  std::mutex swapping_tensors_host_mutex_;
  std::queue<const DeviceAddress *> swapping_tensors_host_;
  std::mutex swapping_tensors_file_mutex_;
  std::queue<const DeviceAddress *> swapping_tensors_file_;
  IOHandlePtr io_handle_;
};
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_DEVICE_GSM_SWAP_MANAGER_H_
