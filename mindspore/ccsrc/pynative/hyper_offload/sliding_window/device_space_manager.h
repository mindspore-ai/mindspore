/**
 * Copyright 2026 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_UTILS_SLIDING_WINDOW_DEVICE_SPACE_MANAGER_H_
#define MINDSPORE_CCSRC_UTILS_SLIDING_WINDOW_DEVICE_SPACE_MANAGER_H_

#include <cstdint>
#include <unordered_set>
#include <vector>
#include <memory>
#include "ir/tensor.h"
#include "device_address/device_address.h"
#include "pynative/hyper_offload/sliding_window/source_group_manager.h"

namespace mindspore {
namespace utils {
namespace sliding_window {

using TensorPtr = tensor::TensorPtr;

class DeviceSpaceManager {
 public:
  explicit DeviceSpaceManager(SourceGroupManager *source_group_mgr);

  ~DeviceSpaceManager() = default;

  // Ensure enough device memory space, offload tensors if needed
  bool EnsureDeviceSpaceEnough(size_t expected_size, const std::unordered_set<TensorId> &protected_source_ids);

  // Ensure all input tensors are on device
  bool EnsureInputsOnDevice(const std::vector<TensorPtr> &inputs);

  // Purge offloaded entries whose original tensor has expired (lifecycle ended)
  void PurgeExpiredOffloaded();

  // Ensure all offloaded tensors are on device
  bool EnsureAllOffloadedOnDevice();

 private:
  // Offload one source group to host. Returns bytes freed on success; 0 when
  // the group is expired, cannot be offloaded, its memory is still event-held,
  // or the host copy failed.
  size_t TryOffloadGroup(SourceGroup *group);

  // Flush async offload tasks and sync events so freed memory is available.
  void FinalizeOffload();

  // Reload one offloaded entry through the shared DeviceAddress. This is used when the
  // source Tensor has been destroyed while a view still keeps the DeviceAddress alive.
  bool ReloadDeviceAddress(TensorId source_id, const device::DeviceAddressPtr &device_addr, const TensorPtr &host_copy);

  bool ReloadGroup(SourceGroup *group, const TensorPtr &addr_tensor);

  SourceGroupManager *source_group_mgr_;
};

}  // namespace sliding_window
}  // namespace utils
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_UTILS_SLIDING_WINDOW_DEVICE_SPACE_MANAGER_H_
