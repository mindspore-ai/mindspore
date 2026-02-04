/**
 * Copyright 2024-2025 Huawei Technologies Co., Ltd
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
#include "kernel/ascend/dvm/pyboost_impl/lazy_fusion_flags.h"

#include <string>
#include <utility>
#include <iostream>
#include <sstream>
#include <cstdlib>
#include "nlohmann/json.hpp"
#include "include/utils/utils.h"
#include "utils/ms_utils.h"

namespace {
constexpr auto kLogValidFlag =
  "Valid flag format is \"--key=value\", flags are separated by spaces(e.g. \"--key1=value1 --key2=value2\"). bool "
  "flag's value can be implicit, the \"--key\" means \"--key=true\".";

// Split string to tokens
std::vector<std::string> GetTokens(const std::string &str, const std::string &delim) {
  std::vector<std::string> tokens;
  size_t start = 0;
  while (start < str.size()) {
    size_t pos = str.find_first_of(delim, start);
    if (pos == std::string::npos) {
      tokens.emplace_back(str.substr(start));
      break;
    }
    if (pos > start) {
      tokens.emplace_back(str.substr(start, pos - start));
    }
    start = pos + delim.size();
  }
  return tokens;
}

// Parse flag string to key-value pair.
// Flag format: "--key=value", bool flag's value can be implicit, the "--key" means "--key=true"
std::pair<std::string, std::string> ParseFlag(const std::string &flag) {
  auto i = flag.find("--");
  // check the string starts with "--".
  constexpr size_t leading_size = 2;
  if (flag.size() <= leading_size || i != 0) {
    return std::pair<std::string, std::string>();
  }
  i += leading_size;

  auto j = flag.find('=', i + 1);  // the key should not be empty, "--=" is invalid
  if (j >= flag.size()) {
    // no value, treated as bool flag.
    return std::make_pair(flag.substr(i), "");
  } else if (j + 1 < flag.size() && flag.find('=', j + 1) == std::string::npos) {
    // normal "--key=value" format
    return std::make_pair(flag.substr(i, j - i), flag.substr(j + 1));
  }
  // string with two "=" is invalid.
  return std::pair<std::string, std::string>();
}

std::map<std::string, std::string> ParseFlags(const std::string &flags) {
  std::map<std::string, std::string> flag_map;
  auto tokens = GetTokens(flags, " ");
  for (const auto &token : tokens) {
    auto flag = ParseFlag(token);
    if (!flag.first.empty()) {
      if (!flag_map.insert(flag).second) {
        MS_LOG(WARNING) << "Warning: The flag '" << flag.first << "' is repeated.";
      }
    } else {
      MS_LOG(WARNING) << "Warning: The flag '" << token << "' is invalid. " << kLogValidFlag;
    }
  }
  return flag_map;
}

class FlagRegister {
 public:
  explicit FlagRegister(std::map<std::string, std::string> *flag_map) : flag_map_(*flag_map) {}
  ~FlagRegister() = default;

  template <typename T>
  void AddFlag(const std::string &flag_name, T *flag_var, T default_value) const {
    *flag_var = std::move(default_value);
    AddFlag(flag_name, flag_var);
  }

  template <typename T>
  void AddFlag(const std::string &flag_name, T *flag_var) const {
    const auto iter = flag_map_.find(flag_name);
    if (iter != flag_map_.end()) {
      T var;
      bool ret = ParseValue(iter->second, &var);
      if (ret) {
        *flag_var = std::move(var);
      } else {
        if (iter->second.empty()) {
          MS_LOG(WARNING) << "Warning: The flag --" << iter->first << " is invalid. " << kLogValidFlag;
        } else {
          MS_LOG(WARNING) << "Warning: The flag --" << iter->first << "=" << iter->second << " is invalid. "
                          << kLogValidFlag;
        }
      }
      (void)flag_map_.erase(iter);
    }
  }

 private:
  bool ParseValue(const std::string &s, std::vector<std::string> *result) const {
    *result = GetTokens(s, ",");
    return !result->empty();
  }

  bool ParseValue(const std::string &s, bool *result) const {
    *result = (s.empty() || s == "true" || s == "True" || s == "on" || s == "1");
    return *result || s == "false" || s == "False" || s == "off" || s == "0";
  }

  template <typename T>
  bool ParseValue(const std::string &s, T *result) const {
    if (s.empty()) {
      return false;
    }
    std::istringstream iss(s);
    iss >> (*result);
    return iss.eof();
  }

  template <typename T>
  bool ParseValue(const std::string &s, std::vector<T> *result) const {
    result->clear();
    auto tokens = GetTokens(s, ",");
    if (tokens.empty()) {
      return false;
    }
    for (const auto &tok : tokens) {
      T temp;
      if (!ParseValue(tok, &temp)) {
        result->clear();
        return false;
      }
      result->emplace_back(temp);
    }
    return true;
  }

  std::map<std::string, std::string> &flag_map_;
};
}  // namespace
namespace mindspore {
const LazyFusionFlags &LazyFusionFlags::GetInstance() {
  static LazyFusionFlags instance;
  return instance;
}

LazyFusionFlags::LazyFusionFlags() {
  auto env = common::EnvHelper::GetInstance()->GetEnv("MS_DEV_PYNATIVE_FUSION_FLAGS");
  std::string str_flags = env == nullptr ? "" : std::string(env);
  std::map<std::string, std::string> flag_map = ParseFlags(str_flags);
  RegisterFlags(&flag_map);
  MS_LOG(INFO) << "lazy_fusion_flags :" << DumpAllFlags();
}

void LazyFusionFlags::RegisterFlags(std::map<std::string, std::string> *flag_map) {
  FlagRegister reg(flag_map);

  reg.AddFlag("dump_as_text", &dump_as_text);
  reg.AddFlag("dump_dir", &dump_dir);
  reg.AddFlag("synchronize", &synchronize);
  reg.AddFlag("flush_threshold", &flush_threshold);
  reg.AddFlag("opt_level", &opt_level);
  reg.AddFlag("online_tuning", &online_tuning);
  reg.AddFlag("disable_ops", &disable_ops);
  reg.AddFlag("enable_ops", &enable_ops);
  reg.AddFlag("enable_ops_only", &enable_ops_only);
  for (const auto &item : *flag_map) {
    MS_LOG(WARNING) << "Unknown flag: " << item.first;
  }
  if (!flag_map->empty()) {
    MS_LOG(WARNING) << "The flags listed above are invalid. Valid flags include: opt_level, enable_ops. For more "
                       "details, please refer to 'MS_DEV_PYNATIVE_FUSION_FLAGS' at https://www.mindspore.cn";
  }
}

std::string LazyFusionFlags::DumpAllFlags() const {
  nlohmann::json j;
  j["dump_as_text"] = dump_as_text;
  j["dump_dir"] = dump_dir;
  j["synchronize"] = synchronize;
  j["disable_ops"] = disable_ops;
  j["enable_ops"] = enable_ops;
  j["enable_ops_only"] = enable_ops_only;
  j["opt_level"] = opt_level;
  j["online_tuning"] = online_tuning;
  j["flush_threshold"] = flush_threshold;
  return j.dump();
}
}  // namespace mindspore
