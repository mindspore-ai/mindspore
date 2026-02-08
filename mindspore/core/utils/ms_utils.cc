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
#include <mutex>
#include <set>
#include <string>
#include <sstream>
#include <ostream>
#include <iostream>

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

namespace {
class Config {
 public:
  static std::string GetValue(const std::string &config, const std::string &config_key);
  static void Reset(const std::string &config);

 private:
  static std::map<std::string, std::map<std::string, std::string>> configs;
  static std::set<std::string> has_parsed_config;

  static std::mutex mutex_;
};

std::map<std::string, std::map<std::string, std::string>> Config::configs;
std::set<std::string> Config::has_parsed_config;
std::mutex Config::mutex_;

std::string Config::GetValue(const std::string &config, const std::string &config_key) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto ret_val = has_parsed_config.insert(config);
  if (ret_val.second) {
    // Parse config.
    const auto env_config = EnvHelper::GetInstance()->GetEnv(config.c_str(), false);
    if (env_config == nullptr) {
      return "";
    }
    std::string env_value = std::string(env_config);
    if (env_value.empty()) {
      return "";
    }

    // Remove the single or double quotes at the beginning and ending of the string. 'xxx', "xxxx" '""xxxx""' they are
    // equal
    while ((env_value.front() == kSingleQuote || env_value.front() == kDoubleQuote) &&
           env_value.front() == env_value.back()) {
      env_value.erase(0, 1);
      env_value.pop_back();
    }

    std::ostringstream oss_buf;
    oss_buf << "[" << config << "] config:";
    // Replace semicolon with commas to standardize delimiter
    std::replace(env_value.begin(), env_value.end(), kSemicolon, kComma);
    std::stringstream ss(env_value);
    std::string item;
    while (std::getline(ss, item, kComma)) {
      // Trim spaces around the item
      item.erase(0, item.find_first_not_of(kWhiteSpace));
      item.erase(item.find_last_not_of(kWhiteSpace) + 1);
      std::size_t delimiterPos = item.find(kColon);
      if (delimiterPos != std::string::npos) {
        std::string key = item.substr(0, delimiterPos);
        std::string value = item.substr(delimiterPos + 1);
        // Trim spaces around key and value
        key.erase(0, key.find_first_not_of(kWhiteSpace));
        key.erase(key.find_last_not_of(kWhiteSpace) + 1);

        value.erase(0, value.find_first_not_of(kWhiteSpace));
        value.erase(value.find_last_not_of(kWhiteSpace) + 1);
        oss_buf << "  " << key << ":" << value;
        configs[config][key] = value;
      }
    }
    std::cout << oss_buf.str() << std::endl;
  }
  auto configs_iter = configs.find(config);
  if (configs_iter == configs.end()) {
    return "";
  }
  if (configs_iter->second.count(config_key) == 0) {
    return "";
  }
  return configs_iter->second.at(config_key);
}

void Config::Reset(const std::string &config) {
  std::lock_guard<std::mutex> lock(mutex_);
  (void)has_parsed_config.erase(config);
}
}  // namespace

void ResetConfig(const std::string &config) { Config::Reset(config); }

std::string GetConfigValue(const std::string &config, const std::string &config_key) {
  return Config::GetValue(config, config_key);
}

EnvHelperPtr &EnvHelper::GetInstance() {
  static std::once_flag init_flag = {};
  static EnvHelperPtr instance = nullptr;
  std::call_once(init_flag, [&]() { instance = std::make_shared<EnvHelper>(); });
  return instance;
}

const char *EnvHelper::GetEnv(const char *conf, bool cache_env) {
  if (conf == nullptr) {
    return nullptr;
  }

  std::lock_guard<std::mutex> lock(mutex_);
  if (!cache_env) {
    return std::getenv(conf);
  }

  auto &&it = env_cache_.find(conf);
  if (it != env_cache_.end()) {
    return it->second.c_str();
  }

  auto ret = std::getenv(conf);
  if (ret != nullptr) {
    env_cache_[conf] = ret;
  }
  return ret;
}

void EnvHelper::ResetCache(const char *conf) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (conf == nullptr) {
    env_cache_.clear();
  } else {
    (void)env_cache_.erase(conf);
  }
}
}  // namespace common
}  // namespace mindspore
