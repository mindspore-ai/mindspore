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

#ifndef MINDSPORE_TOOLS_ERROR_HANDLER_ERROR_CONFIG_H_
#define MINDSPORE_TOOLS_ERROR_HANDLER_ERROR_CONFIG_H_
#include <map>
#include <string>
#include <memory>
#include "tools/visible.h"
#include "nlohmann/json.hpp"
#include "utils/log_adapter.h"
#include "pybind11/pybind11.h"

namespace py = pybind11;

namespace mindspore {
constexpr auto kStatusRecord = "HCCL_STATUS_SAVE";
constexpr auto kStatusSavePath = "CCAE_HCCL_STATUS_SAVE_PATH";
constexpr auto kStatusSaveInterval = "CCAE_HCCL_STATUS_SAVE_INTERVAL";
constexpr auto kWatchdog = "HCCL_WATCHDOG";
namespace tools {
class TOOLS_EXPORT TftConfig {
 public:
  TftConfig() = default;
  ~TftConfig() = default;
  static std::shared_ptr<TftConfig> GetInstance();
  void RegisterConfig(const py::object &configs);
  bool IsEnableWatchdog();
  bool IsEnableSaveHcclOpStatus();
  bool CheckSupport(const std::string &key, bool def_value);
  template <typename T>
  T GetConfigValue(const std::string &key, const T &default_value) {
    if (config_json_.is_null()) {
      MS_LOG(INFO) << "Config is null, using default value.";
      return default_value;
    }
    if (!config_json_.contains(key)) {
      MS_LOG(INFO) << "Key:" << key << " not found, using default value.";
      return default_value;
    }
    try {
      return config_json_[key].get<T>();
    } catch (const std::exception &e) {
      MS_LOG(INFO) << "Get value of " << key << " fault, exception info: " << e.what() << ". Using default value";
    }
    return default_value;
  }

  static bool IsEnableTRE();
  static bool IsEnableStepTRE();
  static int GetSnapShotSteps();

  bool IsEnableUCE();
  bool IsEnableHCCE();
  bool IsEnableARF();
  bool IsEnableRsc();

 private:
  nlohmann::json config_json_;
  std::map<std::string, bool> mark_check_;

  static std::map<std::string, std::string> &GetConfigMap();
  static bool IsEnableFeature(const std::string &feature_name);
};
}  // namespace tools
}  // namespace mindspore
#endif  // MINDSPORE_TOOLS_ERROR_HANDLER_ERROR_CONFIG_H_
