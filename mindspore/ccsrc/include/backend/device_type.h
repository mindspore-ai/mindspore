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

#ifndef MINDSPORE_CCSRC_RUNTIME_HARDWARE_DEVICE_TYPE_H_
#define MINDSPORE_CCSRC_RUNTIME_HARDWARE_DEVICE_TYPE_H_

#include <string>
#include <map>
#include "include/backend/visible.h"

namespace mindspore {
namespace device {
enum class RunMode { kUnknown, kKernelMode, kGraphMode };
enum class DeviceType { kUnknown, kCPU, kAscend, kGPU };
const std::map<DeviceType, std::string> device_type_to_name_map = {{DeviceType::kUnknown, "Unknown"},
                                                                   {DeviceType::kAscend, "Ascend"},
                                                                   {DeviceType::kCPU, "CPU"},
                                                                   {DeviceType::kGPU, "GPU"}};

const std::map<std::string, DeviceType> device_name_to_type_map = {{"Unknown", DeviceType::kUnknown},
                                                                   {"Ascend", DeviceType::kAscend},
                                                                   {"CPU", DeviceType::kCPU},
                                                                   {"GPU", DeviceType::kGPU}};

BACKEND_EXPORT std::string GetDeviceNameByType(const DeviceType &type);
BACKEND_EXPORT DeviceType GetDeviceTypeByName(const std::string &name);
}  // namespace device
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_RUNTIME_HARDWARE_DEVICE_TYPE_H_
