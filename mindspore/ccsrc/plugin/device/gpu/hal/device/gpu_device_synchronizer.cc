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

#include "plugin/device/gpu/hal/device/gpu_device_synchronizer.h"
#include "plugin/device/gpu/hal/device/gpu_device_manager.h"
#include "plugin/device/gpu/hal/device/gpu_common.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace device {
namespace gpu {
bool GPUDeviceSynchronizer::SyncDeviceToHost(void *host_ptr, void *device_ptr, size_t size, mindspore::Format format,
                                             const ShapeVector &shape, size_t stream_id,
                                             const UserDataPtr &user_data) const {
  MS_EXCEPTION_IF_NULL(host_ptr);
  MS_EXCEPTION_IF_NULL(device_ptr);
  const auto stream = GPUDeviceManager::GetInstance().GetStream(stream_id);
  MS_ERROR_IF_NULL(stream);
  CHECK_RET_WITH_RETURN_ERROR(CudaDriver::CopyDeviceMemToHostAsync(host_ptr, device_ptr, size, stream),
                              "CopyHostMemToDeviceAsync failed");

  CHECK_RET_WITH_RETURN_ERROR(CudaDriver::SyncStream(stream), "SyncStream failed");

  return true;
}

bool GPUDeviceSynchronizer::SyncHostToDevice(void *device_ptr, void *host_ptr, size_t size, mindspore::Format format,
                                             const ShapeVector &shape, size_t stream_id,
                                             const UserDataPtr &user_data) const {
  MS_EXCEPTION_IF_NULL(device_ptr);
  MS_EXCEPTION_IF_NULL(host_ptr);
  const auto stream = GPUDeviceManager::GetInstance().GetStream(stream_id);
  MS_ERROR_IF_NULL(stream);
  CHECK_RET_WITH_RETURN_ERROR(CudaDriver::CopyHostMemToDeviceAsync(device_ptr, host_ptr, size, stream),
                              "CopyHostMemToDeviceAsync failed");

  CHECK_RET_WITH_RETURN_ERROR(CudaDriver::SyncStream(stream), "SyncStream failed");

  return true;
}
}  // namespace gpu
}  // namespace device
}  // namespace mindspore
