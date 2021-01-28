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
#include <fstream>
#include "nlohmann/json.hpp"
#include "utils/log_adapter.h"
#include "debug/common.h"
#include "utils/ms_context.h"
#include "utils/convert_utils_base.h"

namespace {
constexpr auto kRdrSettings = "rdr";
constexpr auto kPath = "path";
constexpr auto kEnable = "enable";
}  // namespace

namespace mindspore {
bool EnvConfigParser::CheckJsonStringType(const nlohmann::json &content, const std::string &setting_key,
                                          const std::string &key) {
  if (!content.is_string()) {
    MS_LOG(WARNING) << "Json Parse Failed. The '" << key << "' in '" << setting_key << "' should be string."
                    << " Please check the config file '" << config_file_ << "' set by 'env_config_path' in context.";
    return false;
  }
  return true;
}

auto EnvConfigParser::CheckJsonKeyExist(const nlohmann::json &content, const std::string &setting_key,
                                        const std::string &key) {
  auto iter = content.find(key);
  if (iter == content.end()) {
    MS_LOG(WARNING) << "Check json failed, '" << key << "' not found in '" << setting_key << "'."
                    << " Please check the config file '" << config_file_ << "' set by 'env_config_path' in context.";
  }
  return iter;
}

std::string EnvConfigParser::GetIfstreamString(const std::ifstream &ifstream) {
  std::stringstream buffer;
  buffer << ifstream.rdbuf();
  return buffer.str();
}

void EnvConfigParser::Parse() {
  std::lock_guard<std::mutex> guard(lock_);
  if (already_parsed_) {
    return;
  }
  already_parsed_ = true;

  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  auto config_file = context->get_param<std::string>(MS_CTX_ENV_CONFIG_PATH);
  if (config_file.empty()) {
    MS_LOG(INFO) << "Get env config file failed. Please check the 'env_config_path' set in context.";
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

  ParseRdrSetting(j);
  ConfigToString();
}

void EnvConfigParser::ParseRdrSetting(const nlohmann::json &content) {
  auto rdr_setting = content.find(kRdrSettings);
  if (rdr_setting == content.end()) {
    MS_LOG(WARNING) << "The '" << kRdrSettings << "' not exists. Please check the config file '" << config_file_
                    << "' set by 'env_config_path' in context.";
    return;
  }

  auto rdr_enable = CheckJsonKeyExist(*rdr_setting, kRdrSettings, kEnable);
  auto rdr_path = CheckJsonKeyExist(*rdr_setting, kRdrSettings, kPath);

  ParseRdrEnable(*rdr_enable);
  ParseRdrPath(*rdr_path);
}

void EnvConfigParser::ParseRdrPath(const nlohmann::json &content) {
  if (!CheckJsonStringType(content, kRdrSettings, kPath)) {
    MS_LOG(WARNING) << "The RDR path will be a default value: '" << rdr_path_ << "'.";
    return;
  }
  std::string path = content;
  if (!std::all_of(path.begin(), path.end(),
                   [](char c) { return ::isalpha(c) || ::isdigit(c) || c == '-' || c == '_' || c == '/'; })) {
    MS_LOG(WARNING) << "The path in " << kRdrSettings
                    << " only support alphabets, digit or {'-', '_', '/'}, but got:" << path << "."
                    << " Please check the config file '" << config_file_ << "' set by 'env_config_path' in context.";
    return;
  }
  if (path.empty()) {
    MS_LOG(WARNING) << "The path in " << kRdrSettings << " is empty."
                    << " Please check the config file '" << config_file_ << "' set by 'env_config_path' in context.";
    return;
  }
  if (path[0] != '/') {
    MS_LOG(WARNING) << "The path in " << kRdrSettings << " only support absolute path and should start with '/'."
                    << " Please check the config file '" << config_file_ << "' set by 'env_config_path' in context.";
    return;
  }
  if (path.back() != '/') {
    path += '/';
  }
  rdr_path_ = path;
}

void EnvConfigParser::ParseRdrEnable(const nlohmann::json &content) {
  if (!content.is_boolean()) {
    MS_LOG(WARNING) << "Json Parse Failed. 'enable' in " << kRdrSettings << " should be boolean."
                    << " Please check the config file '" << config_file_ << "' set by 'env_config_path' in context.";
    rdr_enabled_ = false;
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
