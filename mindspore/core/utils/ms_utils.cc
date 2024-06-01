/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#include "utils/ms_utils.h"
#include <map>
#include <string>
#include <sstream>

namespace mindspore {
namespace common {
namespace {
const int CACHED_STR_NUM = 1 << 8;
const int CACHED_STR_MASK = CACHED_STR_NUM - 1;
std::vector<std::string> STR_HOLDER(CACHED_STR_NUM);
}  // namespace
const char *SafeCStr(const std::string &&str) {
  static std::atomic<uint32_t> index{0};
  uint32_t cur_index = index++;
  cur_index = cur_index & CACHED_STR_MASK;
  STR_HOLDER[cur_index] = str;
  return STR_HOLDER[cur_index].c_str();
}

std::string GetRuntimeConfigValue(const std::string &runtime_config) {
  static std::map<std::string, std::string> runtime_configs;
  static bool first_get_runtime_config_value = true;
  // Parse runtime config.
  if (first_get_runtime_config_value) {
    first_get_runtime_config_value = false;
    std::string env_value = GetEnv("MS_DEV_RUNTIME_CONF");
    if (env_value.empty()) {
      return "";
    }

    std::stringstream ss(env_value);
    std::string item;
    while (std::getline(ss, item, ',')) {
      std::size_t delimiterPos = item.find(':');
      if (delimiterPos != std::string::npos) {
        std::string key = item.substr(0, delimiterPos);
        std::string value = item.substr(delimiterPos + 1);
        runtime_configs[key] = value;
      }
    }
  }

  if (runtime_configs.count(runtime_config) == 0) {
    return "";
  }
  return runtime_configs.at(runtime_config);
}

bool IsEnableRuntimeConfig(const std::string &runtime_config) {
  const auto &value = GetRuntimeConfigValue(runtime_config);
  return ((value == "True") || (value == "true"));
}

bool IsDisableRuntimeConfig(const std::string &runtime_config) {
  const auto &value = GetRuntimeConfigValue(runtime_config);
  return ((value == "False") || (value == "false"));
}
}  // namespace common
}  // namespace mindspore
