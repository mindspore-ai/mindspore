/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "debug/env_config_parser.h"
#include <algorithm>
#include <fstream>
#include "nlohmann/json.hpp"
#include "utils/log_adapter.h"
#include "debug/common.h"
#include "utils/ms_context.h"
#include "utils/convert_utils_base.h"

namespace {
constexpr auto ENV_RDR_ENABLE = "MS_RDR_ENABLE";
constexpr auto ENV_RDR_PATH = "MS_RDR_PATH";
constexpr auto KEY_RDR_SETTINGS = "rdr";
constexpr auto KEY_PATH = "path";
constexpr auto KEY_ENABLE = "enable";
constexpr auto kSysSettings = "sys";
constexpr auto kMemReuse = "mem_reuse";
}  // namespace

namespace mindspore {
std::optional<bool> GetRdrEnableFromEnv() {
  // get environment variable to configure RDR
  const char *env_enable_char = std::getenv(ENV_RDR_ENABLE);
  if (env_enable_char != nullptr) {
    std::string env_enable_str = std::string(env_enable_char);
    (void)std::transform(env_enable_str.begin(), env_enable_str.end(), env_enable_str.begin(), ::tolower);
    if (env_enable_str != "0" && env_enable_str != "1") {
      MS_LOG(WARNING) << "The environment variable '" << ENV_RDR_ENABLE << "' should be 0 or 1.";
    }
    if (env_enable_str == "1") {
      return true;
    }
    return false;
  }
  return std::nullopt;
}

std::optional<std::string> GetRdrPathFromEnv() {
  // get environment variable to configure RDR
  const char *path_char = std::getenv(ENV_RDR_PATH);
  if (path_char != nullptr) {
    std::string err_msg = "RDR path parse from environment variable failed. Please check the settings about '" +
                          std::string(ENV_RDR_PATH) + "' in environment variables.";
    std::string path = std::string(path_char);
    if (!Common::IsPathValid(path, MAX_DIRECTORY_LENGTH, err_msg)) {
      return std::string("");
    }
    return path;
  }
  return std::nullopt;
}

bool EnvConfigParser::CheckJsonStringType(const nlohmann::json &content, const std::string &setting_key,
                                          const std::string &key) {
  if (!content.is_string()) {
    MS_LOG(WARNING) << "Json Parse Failed. The '" << key << "' in '" << setting_key << "' should be string."
                    << " Please check the config file '" << config_file_ << "' set by 'env_config_path' in context.";
    return false;
  }
  return true;
}

std::optional<nlohmann::detail::iter_impl<const nlohmann::json>> EnvConfigParser::CheckJsonKeyExist(
  const nlohmann::json &content, const std::string &setting_key, const std::string &key) {
  auto iter = content.find(key);
  if (iter == content.end()) {
    MS_LOG(WARNING) << "Check json failed, '" << key << "' not found in '" << setting_key << "'."
                    << " Please check the config file '" << config_file_ << "' set by 'env_config_path' in context.";
    return std::nullopt;
  }
  return iter;
}

std::string EnvConfigParser::GetIfstreamString(const std::ifstream &ifstream) {
  std::stringstream buffer;
  buffer << ifstream.rdbuf();
  return buffer.str();
}

void EnvConfigParser::ParseFromEnv() {
  // Get RDR seetings from environment variables
  auto rdr_enable_env = GetRdrEnableFromEnv();
  if (rdr_enable_env.has_value()) {
    has_rdr_setting_ = true;
    rdr_enabled_ = rdr_enable_env.value();
  }
  auto path_env = GetRdrPathFromEnv();
  if (path_env.has_value()) {
    has_rdr_setting_ = true;
    std::string path = path_env.value();
    if (!path.empty()) {
      rdr_path_ = path;
    }
  }
}

void EnvConfigParser::ParseFromFile() {
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  auto config_file = context->get_param<std::string>(MS_CTX_ENV_CONFIG_PATH);
  if (config_file.empty()) {
    MS_LOG(INFO) << "The 'env_config_path' in 'mindspore.context.set_context(env_config_path={path})' is empty.";
    return;
  }
  config_file_ = config_file;
  std::ifstream json_file(config_file_);
  if (!json_file.is_open()) {
    MS_LOG(WARNING) << "Env config file:" << config_file_ << " open failed."
                    << " Please check the config file '" << config_file_ << "' set by 'env_config_path' in context.";
    return;
  }

  nlohmann::json j;
  try {
    json_file >> j;
  } catch (nlohmann::json::parse_error &e) {
    MS_LOG(WARNING) << "Env config json contents '" << GetIfstreamString(json_file) << "' in config file '"
                    << config_file_ << "' set by 'env_config_path' in context.";
    return;
  }

  // convert json to string
  std::stringstream ss;
  ss << j;
  std::string cfg = ss.str();
  MS_LOG(INFO) << "Env config json:" << cfg;

  // Parse rdr seetings from file
  ParseRdrSetting(j);
  ParseMemReuseSetting(j);

  ConfigToString();
}

void EnvConfigParser::Parse() {
  std::lock_guard<std::mutex> guard(lock_);
  if (already_parsed_) {
    return;
  }
  already_parsed_ = true;
  ParseFromEnv();
  ParseFromFile();
}

void EnvConfigParser::ParseMemReuseSetting(const nlohmann::json &content) {
  auto sys_setting = content.find(kSysSettings);
  if (sys_setting == content.end()) {
    MS_LOG(INFO) << "The '" << kSysSettings << "' not exists. Please check the config file '" << config_file_
                 << "' set by 'env_config_path' in context.";
    return;
  }
  auto sys_memreuse = CheckJsonKeyExist(*sys_setting, kSysSettings, kMemReuse);
  if (sys_memreuse.has_value()) {
    ParseSysMemReuse(**sys_memreuse);
  }
}

void EnvConfigParser::ParseSysMemReuse(const nlohmann::json &content) {
  if (!content.is_boolean()) {
    MS_LOG(WARNING) << "Json parse failed. 'enable' in " << kSysSettings << " should be boolean."
                    << " Please check the config file '" << config_file_ << "' set by 'env_config_path' in context.";
    return;
  }
  sys_memreuse_ = content;
}

void EnvConfigParser::ParseRdrSetting(const nlohmann::json &content) {
  auto rdr_setting = content.find(KEY_RDR_SETTINGS);
  if (rdr_setting == content.end()) {
    MS_LOG(WARNING) << "The '" << KEY_RDR_SETTINGS << "' not exists. Please check the config file '" << config_file_
                    << "' set by 'env_config_path' in context.";
    return;
  }

  has_rdr_setting_ = true;

  auto rdr_enable = CheckJsonKeyExist(*rdr_setting, KEY_RDR_SETTINGS, KEY_ENABLE);
  if (rdr_enable.has_value()) {
    ParseRdrEnable(**rdr_enable);
  }

  auto rdr_path = CheckJsonKeyExist(*rdr_setting, KEY_RDR_SETTINGS, KEY_PATH);
  if (rdr_path.has_value()) {
    ParseRdrPath(**rdr_path);
  }
}

void EnvConfigParser::ParseRdrPath(const nlohmann::json &content) {
  std::string err_msg = "RDR path parse failed. The RDR path will be a default value: '" + rdr_path_ +
                        "'. Please check the settings about '" + KEY_RDR_SETTINGS + "' in config file '" +
                        config_file_ + "' set by 'env_config_path' in context.";

  if (!CheckJsonStringType(content, KEY_RDR_SETTINGS, KEY_PATH)) {
    MS_LOG(WARNING) << err_msg;
    return;
  }

  std::string path = content;
  if (!Common::IsPathValid(path, MAX_DIRECTORY_LENGTH, err_msg)) {
    return;
  }

  if (path.back() != '/') {
    path += '/';
  }
  rdr_path_ = path;
}

void EnvConfigParser::ParseRdrEnable(const nlohmann::json &content) {
  if (!content.is_boolean()) {
    MS_LOG(WARNING) << "Json parse failed. 'enable' in " << KEY_RDR_SETTINGS << " should be boolean."
                    << " Please check the config file '" << config_file_ << "' set by 'env_config_path' in context.";
    return;
  }
  rdr_enabled_ = content;
}

void EnvConfigParser::ConfigToString() {
  std::string cur_config;
  cur_config.append("After parsed, rdr path: ");
  cur_config.append(rdr_path_);
  cur_config.append(", rdr_enable: ");
  cur_config.append(std::to_string(rdr_enabled_));
  MS_LOG(INFO) << cur_config;
}
}  // namespace mindspore
