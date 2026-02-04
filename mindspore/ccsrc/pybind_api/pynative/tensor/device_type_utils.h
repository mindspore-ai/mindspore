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

#ifndef MINDSPORE_CCSRC_FRONTEND_IR_DEVICE_TYPE_UTILS_H_
#define MINDSPORE_CCSRC_FRONTEND_IR_DEVICE_TYPE_UTILS_H_

#include <string>
#include "pybind_api/pynative/tensor/dlpack.h"
#include "device_address/device_type.h"

namespace mindspore {
class DeviceTypeUtils {
 public:
  // Convert DL device target string to device::DeviceType
  static device::DeviceType DLDeviceTypeToMsDeviceTarget(DLDeviceType dl_device);

  // Convert MindSpore context device target string to DLPack DLDeviceType
  static DLDeviceType MsDeviceTargetToDLDeviceType(device::DeviceType device_type);
};
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_IR_DEVICE_TYPE_UTILS_H_
