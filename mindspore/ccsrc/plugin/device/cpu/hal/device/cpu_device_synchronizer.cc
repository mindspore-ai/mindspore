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

#include "plugin/device/cpu/hal/device/cpu_device_synchronizer.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace device {
namespace cpu {
bool CPUDeviceSynchronizer::SyncDeviceToHost(void *host_ptr, void *device_ptr, size_t size, mindspore::Format format,
                                             const ShapeVector &shape, size_t stream_id,
                                             const UserDataPtr &user_data) const {
  MS_EXCEPTION_IF_NULL(host_ptr);
  MS_EXCEPTION_IF_NULL(device_ptr);

  // For the CPU, device is the Host side, use memcpy_s to copy data.
  auto ret = memcpy_s(host_ptr, size, device_ptr, size);
  if (ret != EOK) {
    MS_LOG(ERROR) << "Memcpy for sync device memory to host side failed, errno[" << ret << "]";
    return false;
  }

  return true;
}

bool CPUDeviceSynchronizer::SyncHostToDevice(void *device_ptr, void *host_ptr, size_t size, mindspore::Format format,
                                             const ShapeVector &shape, size_t stream_id,
                                             const UserDataPtr &user_data) const {
  MS_EXCEPTION_IF_NULL(device_ptr);
  MS_EXCEPTION_IF_NULL(host_ptr);

  // For the CPU, device is the Host side, use memcpy_s to copy data.
  auto ret = memcpy_s(device_ptr, size, host_ptr, size);
  if (ret != EOK) {
    MS_LOG(ERROR) << "Memcpy for sync host memory to device side failed, errno[" << ret << "]";
    return false;
  }

  return true;
}
}  // namespace cpu
}  // namespace device
}  // namespace mindspore
