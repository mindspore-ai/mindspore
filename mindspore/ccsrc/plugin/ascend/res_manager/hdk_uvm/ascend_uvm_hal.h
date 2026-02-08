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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_HAL_DEVICE_ASCEND_UVM_HAL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_HAL_DEVICE_ASCEND_UVM_HAL_H_

#include <memory>
#include <vector>
#include "acl/acl_rt.h"
#include "plugin/ascend/res_manager/visible.h"

namespace mindspore {
namespace device {
namespace ascend {
// UVM memory management interface for Ascend device
class ASCEND_RES_MANAGER_EXPORT AscendUvmHal {
 public:
  AscendUvmHal() = default;
  ~AscendUvmHal() = default;
  AscendUvmHal(const AscendUvmHal &) = delete;
  AscendUvmHal &operator=(const AscendUvmHal &) = delete;

  static AscendUvmHal &GetInstance();
  // Query UVM memory information
  // void QueryUvmInfo(void *addr, uvm_query_info *attr) const;

  // Synchronize stream
  void SyncStream(void *stream_ptr) const;

  // Wait for update to finish
  void WaitUpdateFinished(int32_t dst_device, void *addr, void *stream_ptr) const;

  // Wait for uvm memcpy to finish
  void WaitMemCpyFinished(int32_t dst_device, void *addr, void *stream_ptr) const;

  // Set UVM memory read mostly
  void SetUvmReadMostly(void *addr, size_t size) const;

  // Update device memory to remote memory
  bool UpdateDeviceToRemote(int32_t device_id, void *addr, size_t size, void *stream_ptr, size_t offset = 0,
                            bool sync = false) const;

  // Update remote memory to device memory
  bool UpdateRemoteToDevice(int32_t device_id, void *addr, size_t size, void *stream_ptr, size_t offset = 0,
                            bool sync = false) const;

  // Check if memory has allocated physical memory on device
  bool HasDeviceMem(int32_t device_id, void *addr) const;

  // Check if memory is being updated between remote and device
  bool IsUpdating(int32_t device_id, void *addr, size_t size) const;

  // Detach device memory
  bool DetachDevice(int32_t device_id, void *addr, size_t size, bool sync, void *stream_ptr) const;

  // Copy between host and hyper offload
  bool CopyHostToRemote(void *src_addr, void *dst_addr, size_t size, void *stream_ptr, size_t offset = 0,
                        bool sync = false) const;
  bool CopyRemoteToHost(void *src_addr, void *dst_addr, size_t size, void *stream_ptr, size_t offset = 0,
                        bool sync = false) const;
};

}  // namespace ascend
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_HAL_DEVICE_ASCEND_UVM_HAL_H_
