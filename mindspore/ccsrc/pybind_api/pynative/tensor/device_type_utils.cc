/**
 * Copyright 2020-2025 Huawei Technologies Co., Ltd
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

#include "pybind_api/pynative/tensor/device_type_utils.h"
#include "utils/log_adapter.h"
#include "utils/ms_context.h"

namespace mindspore {

device::DeviceType DeviceTypeUtils::DLDeviceTypeToMsDeviceTarget(DLDeviceType dl_device) {
  if (dl_device == DLDeviceType::kDLCPU) {
    return device::DeviceType::kCPU;
  }
  if (dl_device == DLDeviceType::kDLExtDev) {
    return device::DeviceType::kAscend;
  }
  MS_LOG(EXCEPTION) << "Unsupported dl_device target: " << dl_device;
}

DLDeviceType DeviceTypeUtils::MsDeviceTargetToDLDeviceType(device::DeviceType device_type) {
  if (device_type == device::DeviceType::kCPU) {
    return DLDeviceType::kDLCPU;
  }
  if (device_type == device::DeviceType::kAscend) {
    // Ascend uses kDLExtDev in DLPack for custom device type
    return DLDeviceType::kDLExtDev;
  }
  MS_LOG(EXCEPTION) << "Unsupported device target for DLPack: " << device_type;
}

}  // namespace mindspore
