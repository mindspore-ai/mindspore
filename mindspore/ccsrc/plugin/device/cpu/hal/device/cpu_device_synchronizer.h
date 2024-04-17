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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_HAL_DEVICE_CPU_DEVICE_SYNCHRONIZER_H
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_HAL_DEVICE_CPU_DEVICE_SYNCHRONIZER_H

#include <string>
#include "include/backend/device_synchronizer.h"
#include "include/backend/visible.h"

namespace mindspore {
namespace device {
namespace cpu {
class BACKEND_EXPORT CPUDeviceSynchronizer : public DeviceSynchronizer {
 public:
  CPUDeviceSynchronizer() = default;
  ~CPUDeviceSynchronizer() override = default;

  // Copy device memory to host side synchronously.
  bool SyncDeviceToHost(void *host_ptr, const void *device_ptr, size_t size, const std::string &device_name,
                        uint32_t device_id, mindspore::Format format, const ShapeVector &shape, size_t stream_id,
                        const UserDataPtr &user_data = nullptr) const override;

  // Copy host memory to device side synchronously.
  bool SyncHostToDevice(void *device_ptr, const void *host_ptr, size_t size, const std::string &device_name,
                        uint32_t device_id, mindspore::Format format, const ShapeVector &shape, size_t stream_id,
                        const UserDataPtr &user_data = nullptr) const override;
};
}  // namespace cpu
}  // namespace device
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_HAL_DEVICE_CPU_DEVICE_SYNCHRONIZER_H
