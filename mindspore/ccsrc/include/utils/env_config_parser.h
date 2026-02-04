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
#ifndef MINDSPORE_CCSRC_INCLUDE_COMMON_DEBUG_ENV_CONFIG_PARSER_H_
#define MINDSPORE_CCSRC_INCLUDE_COMMON_DEBUG_ENV_CONFIG_PARSER_H_

#include <string>
#include <map>
#include <set>
#include <mutex>
#include <optional>
#include "nlohmann/json.hpp"
#include "include/utils/visible.h"

namespace mindspore {
enum RdrModes : int { Exceptional = 1, Normal = 2 };

class COMMON_EXPORT EnvConfigParser {
 public:
  static EnvConfigParser &GetInstance();

  void Parse();
  std::string ConfigPath() const { return config_file_; }

#ifdef ENABLE_DUMP_IR
  bool HasRdrSetting() const { return has_rdr_setting_; }
  bool RdrEnabled() const { return rdr_enabled_; }
  int RdrMode() const { return rdr_mode_; }
  std::string RdrPath() const { return rdr_path_; }
#endif
  bool GetSysMemreuse() const { return sys_memreuse_; }
  void SetSysMemreuse(bool set_memreuse) { sys_memreuse_ = set_memreuse; }

 private:
  EnvConfigParser() {}
  ~EnvConfigParser() {}

  std::mutex lock_;
  std::string config_file_{""};
  bool already_parsed_{false};

#ifdef ENABLE_DUMP_IR
  // rdr
  bool has_rdr_setting_{false};
  bool rdr_enabled_{false};
  int rdr_mode_{1};
  std::string rdr_path_{"./"};
#endif

  // memreuse
  bool sys_memreuse_{true};

  void ParseFromFile();
  void ParseFromEnv();
  std::string GetIfstreamString(const std::ifstream &ifstream) const;
  bool CheckJsonStringType(const nlohmann::json &content, const std::string &setting_key, const std::string &key) const;
  std::optional<nlohmann::detail::iter_impl<const nlohmann::json>> CheckJsonKeyExist(const nlohmann::json &content,
                                                                                     const std::string &setting_key,
                                                                                     const std::string &key) const;
#ifdef ENABLE_DUMP_IR
  void ParseRdrSetting(const nlohmann::json &content);
  void ParseRdrPath(const nlohmann::json &content);
  void ParseRdrEnable(const nlohmann::json &content);
  void ParseRdrMode(const nlohmann::json &content);
#endif
  void ParseMemReuseSetting(const nlohmann::json &content);
  void ParseSysMemReuse(const nlohmann::json &content);

  void ConfigToString();
};
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_INCLUDE_COMMON_DEBUG_ENV_CONFIG_PARSER_H_
