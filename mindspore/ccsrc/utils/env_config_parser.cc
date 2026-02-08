/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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
#include "include/utils/env_config_parser.h"
#include <algorithm>
#include <fstream>
#include "nlohmann/json.hpp"
#include "utils/log_adapter.h"
#include "include/utils/common.h"
#include "utils/ms_context.h"
#include "utils/ms_utils.h"

namespace {
#ifdef ENABLE_DUMP_IR
constexpr auto KEY_ENABLE = "enable";
constexpr auto KEY_MODE = "mode";
constexpr auto KEY_PATH = "path";
#endif
constexpr auto KEY_MEM_REUSE_SETTINGS = "sys";
constexpr auto KEY_MEM_REUSE = "mem_reuse";
}  // namespace

namespace mindspore {
EnvConfigParser &EnvConfigParser::GetInstance() {
  static EnvConfigParser instance = EnvConfigParser();
  instance.Parse();
  return instance;
}

bool EnvConfigParser::CheckJsonStringType(const nlohmann::json &content, const std::string &setting_key,
                                          const std::string &key) const {
  if (!content.is_string()) {
    MS_LOG(WARNING) << "Json Parse Failed. The '" << key << "' in '" << setting_key << "' should be string."
                    << " Please check the config file '" << config_file_ << "' set by 'env_config_path' in context.";
    return false;
  }
  return true;
}

std::optional<nlohmann::detail::iter_impl<const nlohmann::json>> EnvConfigParser::CheckJsonKeyExist(
  const nlohmann::json &content, const std::string &setting_key, const std::string &key) const {
  auto iter = content.find(key);
  if (iter == content.end()) {
    MS_LOG(WARNING) << "Check json failed, '" << key << "' not found in '" << setting_key << "'."
                    << " Please check the config file '" << config_file_ << "' set by 'env_config_path' in context.";
    return std::nullopt;
  }
  return iter;
}

std::string EnvConfigParser::GetIfstreamString(const std::ifstream &ifstream) const {
  std::stringstream buffer;
  buffer << ifstream.rdbuf();
  return buffer.str();
}

void EnvConfigParser::ParseFromEnv() {}

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
                    << " Please check the config file '" << config_file_ << "' set by 'env_config_path' in context."
                    << ErrnoToString(errno);
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
  auto sys_setting = content.find(KEY_MEM_REUSE_SETTINGS);
  if (sys_setting == content.end()) {
    MS_LOG(INFO) << "The '" << KEY_MEM_REUSE_SETTINGS << "' isn't existed. Please check the config file '"
                 << config_file_ << "' set by 'env_config_path' in context.";
    return;
  }
  auto sys_memreuse = CheckJsonKeyExist(*sys_setting, KEY_MEM_REUSE_SETTINGS, KEY_MEM_REUSE);
  if (sys_memreuse.has_value()) {
    ParseSysMemReuse(**sys_memreuse);
  }
}

void EnvConfigParser::ParseSysMemReuse(const nlohmann::json &content) {
  if (!content.is_boolean()) {
    MS_LOG(INFO) << "the json object parses failed. 'enable' in " << KEY_MEM_REUSE_SETTINGS << " should be boolean."
                 << " Please check the config file '" << config_file_ << "' set by 'env_config_path' in context.";
    return;
  }
  sys_memreuse_ = content;
}

void EnvConfigParser::ConfigToString() {
  std::string cur_config;
  MS_LOG(INFO) << cur_config;
}
}  // namespace mindspore
