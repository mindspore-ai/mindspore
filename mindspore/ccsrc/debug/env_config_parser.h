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
#ifndef MINDSPORE_CCSRC_DEBUG_ENV_CONFIG_PARSER_H_
#define MINDSPORE_CCSRC_DEBUG_ENV_CONFIG_PARSER_H_

#include <string>
#include <map>
#include <set>
#include <mutex>
#include "nlohmann/json.hpp"
#include "utils/ms_utils.h"
namespace mindspore {
class EnvConfigParser {
 public:
  static EnvConfigParser &GetInstance() {
    static EnvConfigParser instance;
    instance.Parse();
    return instance;
  }

  void Parse();
  std::string ConfigPath() const { return config_file_; }

  bool HasRdrSetting() const { return has_rdr_setting_; }
  bool RdrEnabled() const { return rdr_enabled_; }
  std::string RdrPath() const { return rdr_path_; }
  bool GetSysMemreuse() { return sys_memreuse_; }
  void SetSysMemreuse(bool set_memreuse) { sys_memreuse_ = set_memreuse; }

 private:
  EnvConfigParser() {}
  ~EnvConfigParser() {}

  std::mutex lock_;
  std::string config_file_{""};
  bool already_parsed_{false};

  bool rdr_enabled_{false};
  bool has_rdr_setting_{false};
  std::string rdr_path_{"./rdr/"};

  bool sys_memreuse_{true};
  void ParseFromFile();
  void ParseFromEnv();
  std::string GetIfstreamString(const std::ifstream &ifstream);
  void ParseRdrSetting(const nlohmann::json &content);

  bool CheckJsonStringType(const nlohmann::json &content, const std::string &setting_key, const std::string &key);
  std::optional<nlohmann::detail::iter_impl<const nlohmann::json>> CheckJsonKeyExist(const nlohmann::json &content,
                                                                                     const std::string &setting_key,
                                                                                     const std::string &key);

  void ParseRdrPath(const nlohmann::json &content);
  void ParseRdrEnable(const nlohmann::json &content);
  void ParseMemReuseSetting(const nlohmann::json &content);
  void ParseSysMemReuse(const nlohmann::json &content);

  void ConfigToString();
};
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_DEBUG_ENV_CONFIG_PARSER_H_
