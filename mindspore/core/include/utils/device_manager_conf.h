/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CORE_INCLUDE_UTILS_DEVICE_MANAGER_CONF_H_
#define MINDSPORE_CORE_INCLUDE_UTILS_DEVICE_MANAGER_CONF_H_

#include <memory>
#include <string>
#include <map>
#include "mindapi/base/macros.h"
#include "utils/log_adapter.h"
#include "device_address/device_type.h"

namespace mindspore {
const char kDeterministic[] = "deterministic";
const char kDeviceTargetType[] = "device_target";

class MS_CORE_API DeviceManagerConf {
 public:
  DeviceManagerConf() = default;
  ~DeviceManagerConf() = default;
  DeviceManagerConf(const DeviceManagerConf &) = delete;
  DeviceManagerConf &operator=(const DeviceManagerConf &) = delete;
  static std::shared_ptr<DeviceManagerConf> GetInstance();

  void set_device(const std::string &device_target, uint32_t device_id, bool is_default_device_id) {
    SetDeviceType(device_target);
    conf_status_[kDeviceTargetType] = true;
    device_id_ = device_id;
    is_default_device_id_ = is_default_device_id;
  }
  void distributed_refresh_device_id(uint32_t device_id) {
    MS_LOG(INFO) << "Refresh device id to " << device_id << " for distributed.";
    device_id_ = device_id;
  }
  const std::string &GetDeviceTarget() { return device::GetDeviceNameByType(device_type_); }
  const uint32_t &device_id() { return device_id_; }
  bool is_default_device_id() { return is_default_device_id_; }
  bool IsDeviceEnable() { return conf_status_.count(kDeviceTargetType); }

  void set_deterministic(bool deterministic) {
    deterministic_ = deterministic ? "ON" : "OFF";
    conf_status_[kDeterministic] = true;
  }
  const std::string &deterministic() { return deterministic_; }
  bool IsDeterministicConfigured() { return conf_status_.count(kDeterministic); }

  device::DeviceType device_type() const { return device_type_; }
  void SetDeviceType(const std::string &device_target) {
    if (IsDeviceEnable()) {
      return;
    }
    auto iter = device::device_name_to_type_map.find(device_target);
    if (iter != device::device_name_to_type_map.end()) {
      device_type_ = iter->second;
    }
  }

 private:
  static std::shared_ptr<DeviceManagerConf> instance_;

  device::DeviceType device_type_{device::DeviceType::kUnknown};
  uint32_t device_id_{0};
  bool is_default_device_id_{true};

  std::string deterministic_{"OFF"};

  std::map<std::string, bool> conf_status_;
};
}  // namespace mindspore

#endif  // MINDSPORE_CORE_INCLUDE_UTILS_DEVICE_MANAGER_CONF_H_
