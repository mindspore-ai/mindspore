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

#include "include/backend/device_type.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace device {
std::string GetDeviceNameByType(const DeviceType &type) {
  auto iter = device_type_to_name_map.find(type);
  if (iter == device_type_to_name_map.end()) {
    MS_LOG(EXCEPTION) << "Illegal device type: " << type;
  }
  return iter->second;
}

DeviceType GetDeviceTypeByName(const std::string &name) {
  auto iter = device_name_to_type_map.find(name);
  if (iter == device_name_to_type_map.end()) {
    MS_LOG(EXCEPTION) << "Illegal device name: " << name;
  }
  return iter->second;
}
}  // namespace device
}  // namespace mindspore
