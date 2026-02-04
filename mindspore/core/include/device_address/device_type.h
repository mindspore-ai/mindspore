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

#ifndef MINDSPORE_MINDSPORE_CORE_INCLUDE_IR_DEVICE_TYPE_H_
#define MINDSPORE_MINDSPORE_CORE_INCLUDE_IR_DEVICE_TYPE_H_
#include <string>
#include <map>
#include "mindapi/base/macros.h"

namespace mindspore {
namespace device {
enum class RunMode { kUnknown, kKernelMode, kGraphMode };
enum class DeviceType { kUnknown = 0, kCPU = 1, kAscend = 2, kGPU = 3, kNone = 4, kDeviceEnd };

const std::map<RunMode, std::string> run_mode_to_name_map = {
  {RunMode::kUnknown, "Unknown"}, {RunMode::kKernelMode, "KernelMode"}, {RunMode::kGraphMode, "GraphMode"}};

const std::map<DeviceType, std::string> device_type_to_name_map = {{DeviceType::kUnknown, "Unknown"},
                                                                   {DeviceType::kAscend, "Ascend"},
                                                                   {DeviceType::kCPU, "CPU"},
                                                                   {DeviceType::kGPU, "GPU"}};

const std::map<std::string, DeviceType> device_name_to_type_map = {{"Unknown", DeviceType::kUnknown},
                                                                   {"Ascend", DeviceType::kAscend},
                                                                   {"CPU", DeviceType::kCPU},
                                                                   {"GPU", DeviceType::kGPU}};

MS_CORE_API const std::string &GetDeviceNameByType(const DeviceType &type);
MS_CORE_API DeviceType GetDeviceTypeByName(const std::string &name);
MS_CORE_API bool IsAscendDeviceType(const DeviceType &type);
MS_CORE_API bool IsCPUDeviceType(const DeviceType &type);

}  // namespace device
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CORE_INCLUDE_IR_DEVICE_TYPE_H_
