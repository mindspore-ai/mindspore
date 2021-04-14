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

#include "utils/context/graph_kernel_flags.h"

#include <map>
#include <string>
#include <cstring>
#include <vector>
#include <utility>
#include "nlohmann/json.hpp"
#include "utils/ms_context.h"

namespace mindspore {
namespace context {
namespace {
// Split string to tokens
std::vector<std::string> GetTokens(const std::string &str, const std::string &delim) {
  std::vector<std::string> tokens;
  std::vector<char> c_str(str.begin(), str.end());
  c_str.push_back('\0');
  char *saveptr;
  char *pch = strtok_r(&c_str[0], delim.c_str(), &saveptr);
  while (pch != NULL) {
    tokens.emplace_back(pch);
    pch = strtok_r(NULL, delim.c_str(), &saveptr);
  }
  return tokens;
}

// Parse flag string to key-value pair.
// Flag format: "--key=value", bool flag's value can be implicit, the "--key" means "--key=true"
std::pair<std::string, std::string> ParseFlag(const std::string &flag) {
  auto i = flag.find("--");
  // check the string starts with "--".
  if (i != 0 || flag.size() == 2) {
    return std::pair<std::string, std::string>();
  }
  i += 2;

  auto j = flag.find('=', i + 1);  // the key should not be empty, "--=" is invalid
  if (j == std::string::npos) {
    // no value, treated as bool flag.
    return std::make_pair(flag.substr(i), "");
  } else if (j + 1 != flag.size() && flag.find('=', j + 1) == std::string::npos) {
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
    if (flag.first != "") {
      if (!flag_map.insert(flag).second) {
        MS_LOG(WARNING) << "Repeated GraphKernel flag: " << flag.first;
      }
    } else {
      MS_LOG(WARNING) << "Invalid GraphKernel flag: " << token;
    }
  }
  return flag_map;
}

class FlagRegister {
 public:
  explicit FlagRegister(std::map<std::string, std::string> *flag_map) : flag_map_(*flag_map) {}
  ~FlagRegister() = default;

  template <typename T>
  void AddFlag(std::string flag_name, T *flag_var) {
    auto iter = flag_map_.find(flag_name);
    if (iter != flag_map_.end()) {
      T var;
      bool ret = ParseValue(iter->second, &var);
      if (ret) {
        *flag_var = std::move(var);
      } else {
        if (iter->second.empty()) {
          MS_LOG(WARNING) << "Invalid GraphKernel flag: --" << iter->first;
        } else {
          MS_LOG(WARNING) << "Invalid GraphKernel flag: --" << iter->first << "=" << iter->second;
        }
      }
      flag_map_.erase(iter);
    }
  }

 private:
  bool ParseValue(const std::string &s, std::vector<std::string> *result) {
    *result = GetTokens(s, ",");
    return !result->empty();
  }

  bool ParseValue(const std::string &s, bool *result) {
    *result = (s.empty() || s == "true" || s == "on" || s == "1");
    return *result || s == "false" || s == "off" || s == "0";
  }

  template <typename T>
  bool ParseValue(const std::string &s, T *result) {
    if (s.empty()) {
      return false;
    }
    std::istringstream iss(s);
    iss >> (*result);
    return iss.eof();
  }

  template <typename T>
  bool ParseValue(const std::string &s, std::vector<T> *result) {
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

void GraphKernelFlags::Refresh() {
  auto flag_map = ParseFlags(flags_cache_);
  RegisterFlags(&flag_map);
  for (auto &item : flag_map) {
    MS_LOG(WARNING) << "Unknown GraphKernel flag: " << item.first;
  }
}

void GraphKernelFlags::RegisterFlags(std::map<std::string, std::string> *flag_map) {
  FlagRegister reg(flag_map);

  // Boolean flags
  reg.AddFlag("dump_as_text", &dump_as_text);
  reg.AddFlag("enable_stitch_fusion", &enable_stitch_fusion);
  reg.AddFlag("enable_parallel_fusion", &enable_parallel_fusion);

  // Integer flags
  reg.AddFlag("opt_level", &opt_level);
  reg.AddFlag("auto_tune", &auto_tune);
  reg.AddFlag("cluster_limit", &cluster_limit);

  // String list flags
  reg.AddFlag("enable_expand_ops", &enable_expand_ops);
  reg.AddFlag("enable_expand_ops_only", &enable_expand_ops_only);
  reg.AddFlag("disable_expand_ops", &disable_expand_ops);
  reg.AddFlag("enable_cluster_ops", &enable_cluster_ops);
  reg.AddFlag("enable_cluster_ops_only", &enable_cluster_ops_only);
  reg.AddFlag("disable_cluster_ops", &disable_cluster_ops);
  reg.AddFlag("enable_pass_only", &enable_pass_only);
  reg.AddFlag("disable_pass", &disable_pass);
}

std::string GraphKernelFlags::DumpAllFlags() const {
  nlohmann::json json;

  json["dump_as_text"] = dump_as_text;
  json["enable_stitch_fusion"] = enable_stitch_fusion;
  json["enable_parallel_fusion"] = enable_parallel_fusion;

  json["opt_level"] = opt_level;
  json["auto_tune"] = auto_tune;
  json["cluster_limit"] = cluster_limit;

  json["enable_expand_ops"] = enable_expand_ops;
  json["enable_expand_ops_only"] = enable_expand_ops_only;
  json["disable_expand_ops"] = disable_expand_ops;
  json["enable_cluster_ops"] = enable_cluster_ops;
  json["enable_cluster_ops_only"] = enable_cluster_ops_only;
  json["disable_cluster_ops"] = disable_cluster_ops;
  json["enable_pass_only"] = enable_pass_only;
  json["disable_pass"] = disable_pass;

  return json.dump();
}
}  // namespace context
}  // namespace mindspore
