/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_RUNTIME_HARDWARE_ASCEND_ASCEND_RES_MANAGER_H_
#define MINDSPORE_CCSRC_RUNTIME_HARDWARE_ASCEND_ASCEND_RES_MANAGER_H_

#include <vector>
#include <memory>
#include <string>
#include <set>
#include <map>
#include "runtime/hardware/device_context.h"
#include "runtime/device/memory_manager.h"
#include "plugin/device/ascend/hal/device/ascend_kernel_runtime.h"
#include "plugin/device/ascend/hal/device/ascend_device_address.h"

namespace mindspore {
namespace device {
namespace ascend {
class AscendDeviceResManager : public DeviceResManager {
 public:
  AscendDeviceResManager() : compute_stream_(nullptr), communication_stream_(nullptr), mem_manager_(nullptr) {}
  ~AscendDeviceResManager() override = default;

  void Initialize() override;

  void Destroy() override;

  // set rt_context_ to this thread to control device
  bool BindDeviceToCurrentThread() const override;

  // Relevant function to allocate and free device memory of raw ptr.
  void *AllocateMemory(size_t size) const override;
  void FreeMemory(void *ptr) const override;
  bool AllocateMemory(DeviceAddress *const &address) const override;

  // Allocate continuous device memory according to size list.
  // Communication operators may need continuous memory for input and output
  // to optimize the communication performance.
  std::vector<void *> AllocateContinuousMemory(const std::vector<size_t> &size_list) const override;

  // Create concrete device address according different device type.
  DeviceAddressPtr CreateDeviceAddress(void *const device_ptr, size_t device_size, const string &format, TypeId type_id,
                                       const ShapeVector &shape) const override;

  // Synchronize stream, device such as GPU and Ascend need stream to launch kernel asynchronously,
  // using 'SyncStream' to block thread and wait for completing all tasks in stream.
  // Devices that do not need stream could ignore the implementation of this function.
  bool SyncStream(size_t stream_id) const override;

 protected:
  // Really create an ascend stream.
  bool CreateStream(void **stream) const override;

  // Really destroy an ascend stream.
  bool DestroyStream(void *stream) const override;

 private:
  friend class AscendKernelExecutor;
  friend class AscendGraphExecutor;
  friend class AscendDeviceContext;

  // rank id of physical device
  uint32_t rank_id_{0};
  void *compute_stream_;
  void *communication_stream_;
  // Kernel Runtime  --- only for task sink
  AscendKernelRuntime *runtime_instance_{nullptr};
  std::shared_ptr<MemoryManager> mem_manager_{nullptr};
};
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_RUNTIME_HARDWARE_ASCEND_ASCEND_RES_MANAGER_H_
