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
  char *saveptr = nullptr;
  char *pch = strtok_r(&c_str[0], delim.c_str(), &saveptr);
  while (pch != nullptr) {
    tokens.emplace_back(pch);
    pch = strtok_r(nullptr, delim.c_str(), &saveptr);
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
  void AddFlag(const std::string &flag_name, T *const flag_var, T default_value = T()) const {
    auto iter = flag_map_.find(flag_name);
    if (iter != flag_map_.end()) {
      T var;
      bool ret = ParseValue(iter->second, &var);
      if (ret) {
        *flag_var = std::move(var);
      } else {
        *flag_var = std::move(default_value);
        if (iter->second.empty()) {
          MS_LOG(WARNING) << "Invalid GraphKernel flag: --" << iter->first;
        } else {
          MS_LOG(WARNING) << "Invalid GraphKernel flag: --" << iter->first << "=" << iter->second;
        }
      }
      flag_map_.erase(iter);
    } else {
      *flag_var = std::move(default_value);
    }
  }

 private:
  bool ParseValue(const std::string &s, std::vector<std::string> *result) const {
    *result = GetTokens(s, ",");
    return !result->empty();
  }

  bool ParseValue(const std::string &s, bool *result) const {
    *result = (s.empty() || s == "true" || s == "on" || s == "1");
    return *result || s == "false" || s == "off" || s == "0";
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

void GraphKernelFlags::Refresh() {
  auto flag_map = ParseFlags(flags_cache_);
  RegisterFlags(&flag_map);
  for (auto &item : flag_map) {
    MS_LOG(WARNING) << "Unknown GraphKernel flag: " << item.first;
  }
  if (IsEnableGraphKernel()) {
    auto context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(context);
    if (context->get_param<int>(MS_CTX_EXECUTION_MODE) != kGraphMode) {
      MS_LOG(WARNING) << "GraphKernel only support GRAPH_MODE";
      opt_level = OptLevel_0;
    }
  }
  // Dump flags so that people can check the setting.
  MS_LOG(INFO) << "graph_kernel_flags = \"" << flags_cache_ << "\", all flags: " << DumpAllFlags();
}

void GraphKernelFlags::RegisterFlags(std::map<std::string, std::string> *flag_map) {
  FlagRegister reg(flag_map);
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  bool is_gpu = (context_ptr->get_param<std::string>(MS_CTX_DEVICE_TARGET) == kGPUDevice);

  // Set opt_level first, some flags' default value depends on it.
  // Default optimization level is level 2 when enable graphkernel
  reg.AddFlag("opt_level", &opt_level, enable_graph_kernel_ ? OptLevel_2 : OptLevel_0);
  if (opt_level > OptLevel_3) {
    MS_LOG(WARNING) << "GraphKernelFlag: opt_level should be in the range [0,3] but got " << opt_level;
    opt_level = OptLevel_3;
  }

  // Boolean flags
  reg.AddFlag("dump_as_text", &dump_as_text);
  reg.AddFlag("enable_stitch_fusion", &enable_stitch_fusion, opt_level == OptLevel_3);
  reg.AddFlag("enable_recompute_fusion", &enable_recompute_fusion, opt_level >= OptLevel_2);
  reg.AddFlag("enable_parallel_fusion", &enable_parallel_fusion, opt_level == OptLevel_3);
  reg.AddFlag("enable_low_precision", &enable_low_precision);
  reg.AddFlag("enable_trans_op_optimize", &enable_trans_op_optimize);

  // Integer flags
  reg.AddFlag("online_tuning", &online_tuning);
  reg.AddFlag("fusion_ops_level", &fusion_ops_level, is_gpu ? OpLevel_MAX : OpLevel_0);

  // String flags
  reg.AddFlag("repository_path", &repository_path);

  // String list flags
  reg.AddFlag("enable_expand_ops", &enable_expand_ops);
  reg.AddFlag("enable_expand_ops_only", &enable_expand_ops_only);
  reg.AddFlag("disable_expand_ops", &disable_expand_ops);
  reg.AddFlag("enable_cluster_ops", &enable_cluster_ops);
  reg.AddFlag("enable_cluster_ops_only", &enable_cluster_ops_only);
  reg.AddFlag("disable_cluster_ops", &disable_cluster_ops);
  reg.AddFlag("enable_simplify_exprs_only", &enable_simplify_exprs_only);
  reg.AddFlag("disable_simplify_exprs", &disable_simplify_exprs);
  reg.AddFlag("enable_pass", &enable_pass);
  reg.AddFlag("disable_pass", &disable_pass);
}

std::string GraphKernelFlags::DumpAllFlags() const {
  nlohmann::json json;

  json["dump_as_text"] = dump_as_text;
  json["enable_stitch_fusion"] = enable_stitch_fusion;
  json["enable_recompute_fusion"] = enable_recompute_fusion;
  json["enable_parallel_fusion"] = enable_parallel_fusion;
  json["enable_low_precision"] = enable_low_precision;
  json["enable_trans_op_optimize"] = enable_trans_op_optimize;

  json["opt_level"] = opt_level;
  json["fusion_ops_level"] = fusion_ops_level;
  json["online_tuning"] = online_tuning;

  json["repository_path"] = repository_path;

  json["enable_expand_ops"] = enable_expand_ops;
  json["enable_expand_ops_only"] = enable_expand_ops_only;
  json["disable_expand_ops"] = disable_expand_ops;
  json["enable_cluster_ops"] = enable_cluster_ops;
  json["enable_cluster_ops_only"] = enable_cluster_ops_only;
  json["disable_cluster_ops"] = disable_cluster_ops;
  json["enable_simplify_exprs_only"] = enable_simplify_exprs_only;
  json["disable_simplify_exprs"] = disable_simplify_exprs;
  json["enable_pass"] = enable_pass;
  json["disable_pass"] = disable_pass;

  return json.dump();
}
}  // namespace context
}  // namespace mindspore
