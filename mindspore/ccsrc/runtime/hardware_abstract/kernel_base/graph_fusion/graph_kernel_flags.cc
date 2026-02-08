/**
 * Copyright 2021-2025 Huawei Technologies Co., Ltd
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

#include "include/runtime/hardware_abstract/kernel_base/graph_fusion/graph_kernel_flags.h"
#include <memory>
#include <iostream>
#include <map>
#include <string>
#include <cstring>
#include <vector>
#include <utility>
#include "nlohmann/json.hpp"
#include "utils/ms_context.h"
#include "utils/ms_utils.h"
#include "utils/file_utils.h"
#include "ops_utils/op_constants.h"

namespace mindspore::graphkernel {
namespace {
constexpr auto kLogValidFlag =
  "Valid flag format is \"--key=value\", flags are separated by spaces(e.g. \"--key1=value1 --key2=value2\"). bool "
  "flag's value can be implicit, the \"--key\" means \"--key=true\".";

// Split string to tokens
std::vector<std::string> GetTokens(const std::string &str, const std::string &delim) {
  std::vector<std::string> tokens;
  std::vector<char> c_str(str.begin(), str.end());
  c_str.push_back('\0');
  char *saveptr = nullptr;
#ifdef _MSC_VER
  char *pch = strtok_s(&c_str[0], delim.c_str(), &saveptr);
#else
  char *pch = strtok_r(&c_str[0], delim.c_str(), &saveptr);
#endif
  while (pch != nullptr) {
    (void)tokens.emplace_back(pch);
#ifdef _MSC_VER
    pch = strtok_s(nullptr, delim.c_str(), &saveptr);
#else
    pch = strtok_r(nullptr, delim.c_str(), &saveptr);
#endif
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

std::map<std::string, std::string> ParseFlagsByJson(const std::string &json_path) {
  std::map<std::string, std::string> flag_map;
  auto realpath = FileUtils::GetRealPath(json_path.c_str());
  if (!realpath.has_value()) {
    MS_LOG(ERROR) << "Failed to get real path.";
  }
  std::ifstream jsonFile(realpath.value(), std::ifstream::in);
  if (!jsonFile.is_open()) {
    MS_LOG(ERROR) << "Failed to open graph kernel flags file. ";
  }
  nlohmann::json flags_json = nlohmann::json::parse(jsonFile);
  for (auto it = flags_json.begin(); it != flags_json.end(); ++it) {
    if (it.value().is_string()) {
      flag_map[it.key()] = it.value();
    } else {
      MS_LOG(ERROR) << "Flags's value should be string.";
    }
  }
  return flag_map;
}

std::map<std::string, std::string> ParseFlags(const std::string &flags) {
  std::map<std::string, std::string> flag_map;
  auto tokens = GetTokens(flags, " ");
  for (const auto &token : tokens) {
    auto flag = ParseFlag(token);
    if (flag.first == "path") {
      return ParseFlagsByJson(flag.second);
    }
    if (flag.first != "") {
      if (!flag_map.insert(flag).second) {
        MS_LOG(WARNING) << "For 'context.set_context', the flag '" << flag.first
                        << "' in the parameter 'graph_kernel_flags' is repeated.";
      }
    } else {
      MS_LOG(WARNING) << "For 'context.set_context', the flag '" << token
                      << "' in the parameter 'graph_kernel_flags' is invalid. " << kLogValidFlag;
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
          MS_LOG(WARNING) << "For 'context.set_context', the flag --" << iter->first
                          << " in the parameter 'graph_kernel_flags' is invalid. " << kLogValidFlag;
        } else {
          MS_LOG(WARNING) << "For 'context.set_context', the flag --" << iter->first << "=" << iter->second
                          << " in the parameter 'graph_kernel_flags' is invalid. " << kLogValidFlag;
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

bool GraphKernelFlags::IsEnableKernelPacket() const {
  // when jit_level is O1 (enable dvm), default enable kernel packet, user can use the environment to disable it.
  // whne jit_level is O0, default disable kernel packet, user can use the environment to enable it.
  if (IsEnableGraphKernel()) {
    return common::GetEnv("MS_DEV_ENABLE_KERNEL_PACKET") != "off";
  }
  return common::GetEnv("MS_DEV_ENABLE_KERNEL_PACKET") == "on";
}

const GraphKernelFlags &GraphKernelFlags::GetInstance() {
  static std::unique_ptr<GraphKernelFlags> flags(nullptr);
  auto config = GetGraphKernelConfig();
  if (flags == nullptr || config.first != flags->flags_cache_ || config.second != flags->enable_graph_kernel_) {
    flags.reset(new GraphKernelFlags(config.first, config.second));
    flags->Refresh();
  }
  return *flags;
}

void GraphKernelFlags::SaveJitConfig(const std::map<std::string, std::string> &jit_config) {
  auto &configs = GetJitConfig();
  configs.clear();
  auto level_iter = jit_config.find(kAttrJitLevel);
  if (level_iter != jit_config.end()) {
    configs[kAttrJitLevel] = level_iter->second;
    MS_LOG(DEBUG) << "Save jit_level from jit config, level: " << level_iter->second;
  }
  auto flags_iter = jit_config.find("graph_kernel_flags");
  if (flags_iter != jit_config.end()) {
    configs["graph_kernel_flags"] = flags_iter->second;
    MS_LOG(DEBUG) << "Save graph_kernel_flags from jit config, flags: " << flags_iter->second;
  }
}

std::pair<std::string, bool> GraphKernelFlags::GetGraphKernelConfig() {
  const auto &jit_config = GetJitConfig();
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);

  auto jit_level_iter = jit_config.find(kAttrJitLevel);
  auto jit_level = (jit_level_iter != jit_config.end() ? jit_level_iter->second : "");
  bool enable_gk = context->get_param<bool>(MS_CTX_ENABLE_GRAPH_KERNEL);
  auto device_target = context->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  if (!enable_gk && device_target == kAscendDevice) {
    enable_gk = (jit_level == kAttrJitLevelO1);
  }
  // use environ flags in priority
  auto flags_env = common::EnvHelper::GetInstance()->GetEnv("MS_DEV_GRAPH_KERNEL_FLAGS");
  if (flags_env != nullptr) {
    return std::make_pair(std::string(flags_env), enable_gk);
  }
  // get flags string from context or jitconfig
  auto flags = context->get_param<std::string>(MS_CTX_GRAPH_KERNEL_FLAGS);
  auto iter = jit_config.find("graph_kernel_flags");
  if (iter != jit_config.end()) {
    static bool print_warning_once = true;
    if (!flags.empty() && print_warning_once) {
      print_warning_once = false;
      MS_LOG(WARNING) << "The 'graph_kernel_flags' in 'mindspore.context' and 'JitConfig' is set in the same time, "
                         "only the JitConfig's setting is efficient.";
    }
    flags = iter->second;
  }
  return std::make_pair(flags, enable_gk);
}

void GraphKernelFlags::CheckSupport() const {
  if (IsEnableGraphKernel()) {
    auto context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(context);
#ifndef USE_LLVM
    auto is_cpu = (context->get_param<std::string>(MS_CTX_DEVICE_TARGET) == kCPUDevice);
    if (is_cpu && const_cast<GraphKernelFlags *>(this)->kernel_generator == "AKG") {
      MS_LOG(WARNING)
        << "Graph Kernel Fusion is not supported without LLVM on cpu platform, and it will be turned off now. Please "
           "refer to https://www.mindspore.cn/install and install the required version of LLVM.";
      const_cast<GraphKernelFlags *>(this)->opt_level = OptLevel_0;
      return;
    }
#endif
    auto is_ascend = (context->get_param<std::string>(MS_CTX_DEVICE_TARGET) == kAscendDevice);
    if (is_ascend) {
#ifndef ENABLE_DVM
      MS_LOG(WARNING) << "Graph Kernel Fusion is not supported without the prebuild binary file tracked by git lfs, "
                         "and it will be turned off now. Please perform the following steps:\n\n"
                         "1. Install git lfs, refer https://github.com/git-lfs/git-lfs/wiki/installation\n"
                         "2. After installing git lfs, do not forget executing the following command:\n"
                         "   git lfs install\n"
                         "3. Re-clone the source codes, the files tracked by git lfs will be downloaded automatically\n"
                         "4. Re-compile the source codes\n";
      const_cast<GraphKernelFlags *>(this)->opt_level = OptLevel_0;
      return;
#else
      if (const_cast<GraphKernelFlags *>(this)->kernel_generator == "DVM") {
        auto const &soc_version = context->ascend_soc_version();
        if (!soc_version.empty() && soc_version != "ascend910b" && soc_version != "ascend910_93") {
          MS_LOG(WARNING) << "DVM does not support " << soc_version << ".";
          const_cast<GraphKernelFlags *>(this)->opt_level = OptLevel_0;
          return;
        }
      }
#endif
    }
  }
}

void GraphKernelFlags::Refresh() {
  auto flag_map = ParseFlags(flags_cache_);
  RegisterFlags(&flag_map);
  for (const auto &item : flag_map) {
    MS_LOG(WARNING) << "Unknown flag: " << item.first;
  }
  if (!flag_map.empty()) {
    MS_LOG(WARNING)
      << "For 'context.set_context', the flags listed above in the parameter 'graph_kernel_flags' are invalid. For "
         "valid flags, please refer to the source code file graph_kernel_flags.h at "
         "https://gitee.com/mindspore/mindspore.";
  }
  if (IsEnableGraphKernel()) {
    CheckSupport();
  }
  // If enable graphkernel, Dump flags so that people can check the setting.
  if (IsEnableGraphKernel()) {
    MS_LOG(INFO) << "graph_kernel_flags = \"" << flags_cache_ << "\", all flags: " << DumpAllFlags();
  }

  GraphKernelPassChecker::GetInstance().Init();
}

void GraphKernelFlags::RegisterFlags(std::map<std::string, std::string> *flag_map) {
  bool has_kernel_generator = (flag_map->find("kernel_generator") != flag_map->end());
  bool has_enable_dynamic_shape_fusion = (flag_map->find("enable_dynamic_shape_fusion") != flag_map->end());
  FlagRegister reg(flag_map);
  bool is_ascend{false};
  bool is_910b{false};
  auto context_ptr = MsContext::GetInstance();
  if (context_ptr != nullptr) {
    auto const &soc_version = context_ptr->ascend_soc_version();
    is_910b = (soc_version == "ascend910b") || (soc_version == "ascend910_93");
    is_ascend = (context_ptr->get_param<std::string>(MS_CTX_DEVICE_TARGET) == kAscendDevice);
  }

  // Set opt_level first, some flags' default value depends on it.
  // Default optimization level is level 2 when enable graphkernel
  reg.AddFlag("opt_level", &opt_level, enable_graph_kernel_ ? OptLevel_2 : OptLevel_0);
  if (opt_level > OptLevel_3) {
    MS_LOG(WARNING) << "For 'context.set_context', the flag opt_level in the parameter 'graph_kernel_flags' must be in "
                       "the range [0, 3], but got "
                    << opt_level << ". It will be set to " << OptLevel_3
                    << ". For more details, please refer to 'graph_kernel_flags' at https://www.mindspore.cn.";
    opt_level = OptLevel_3;
  }

  // Boolean flags
  reg.AddFlag("dump_as_text", &dump_as_text);
  reg.AddFlag("enable_stitch_fusion", &enable_stitch_fusion, (opt_level == OptLevel_3 && !is_910b));
  reg.AddFlag("enable_recompute_fusion", &enable_recompute_fusion, opt_level >= OptLevel_2);
  reg.AddFlag("enable_parallel_fusion", &enable_parallel_fusion, opt_level == OptLevel_3);
  reg.AddFlag("enable_horizontal_fusion", &enable_horizontal_fusion);
  reg.AddFlag("enable_auto_tensor_inplace", &enable_auto_tensor_inplace);
  reg.AddFlag("enable_dynamic_batch", &enable_dynamic_batch);
  reg.AddFlag("enable_csr_fusion", &enable_csr_fusion);
  reg.AddFlag("enable_debug_mode", &enable_debug_mode);
  reg.AddFlag("enable_lite_conv_tuning", &enable_lite_conv_tuning);
  reg.AddFlag("enable_vectorization", &enable_vectorization);
  reg.AddFlag("enable_dynamic_shape_fusion", &enable_dynamic_shape_fusion);
  reg.AddFlag("disable_matmul_post_fusion", &disable_matmul_post_fusion);
  reg.AddFlag("enable_allreduce_prologue_fusion", &enable_allreduce_prologue_fusion);
  reg.AddFlag("enable_allreduce_epilogue_fusion", &enable_allreduce_epilogue_fusion);
  reg.AddFlag("exact_precision_mode", &exact_precision_mode);

  // Integer flags
  reg.AddFlag("reduce_fuse_depth", &reduce_fuse_depth);
  reg.AddFlag("online_tuning", &online_tuning);
  reg.AddFlag("cpu_refer_thread_num", &cpu_refer_thread_num);
  reg.AddFlag("fusion_ops_level", &fusion_ops_level, is_ascend ? OpLevel_0 : OpLevel_1);
  reg.AddFlag("parallel_ops_level", &parallel_ops_level);
  reg.AddFlag("recompute_increment_threshold", &recompute_increment_threshold);
  reg.AddFlag("recompute_peak_threshold", &recompute_peak_threshold);
  reg.AddFlag("composite_op_limit_size", &composite_op_limit_size);

  // String flags
  reg.AddFlag("repository_path", &repository_path);
  reg.AddFlag("target_os", &target_os);
  reg.AddFlag("cpu_arch", &cpu_arch);
  reg.AddFlag("cpu_feature", &cpu_feature);
  reg.AddFlag("cpu_type", &cpu_type);
  reg.AddFlag("kernel_generator", &kernel_generator);

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
  reg.AddFlag("enable_cce_lib", &enable_cce_lib);
  reg.AddFlag("enable_cce_lib_ops", &enable_cce_lib_ops);
  reg.AddFlag("enable_cce_lib_ops_only", &enable_cce_lib_ops_only);
  reg.AddFlag("disable_cce_lib_ops", &disable_cce_lib_ops);
  reg.AddFlag("enable_packet_ops_only", &enable_packet_ops_only);
  reg.AddFlag("disable_packet_ops", &disable_packet_ops);
  reg.AddFlag("enable_fusion_pattern_only", &enable_fusion_pattern_only);
  reg.AddFlag("disable_fusion_pattern", &disable_fusion_pattern);

  if (enable_dynamic_shape_fusion && !is_ascend) {
    kernel_generator = "AKG_V2";
    return;
  }

  if (is_ascend && !has_kernel_generator) {
    kernel_generator = "DVM";
  }
  if (kernel_generator == "DVM" && !has_enable_dynamic_shape_fusion) {
    enable_dynamic_shape_fusion = true;
  }
  if (is_ascend && enable_auto_tensor_inplace) {
    MS_LOG(WARNING)
      << "For Graph Kernel Fusion, the flag '--enable_auto_tensor_inplace' set in 'graph_kernel_flags' is "
         "not supported on Ascend and will be turned off now";
    enable_auto_tensor_inplace = false;
  }
  if (is_ascend && enable_parallel_fusion) {
    std::cout
      << "[WARNING] The flag '--enable_parallel_fusion' set in 'graph_kernel_flags' will enable parallel fusion, "
         "which is an experimental feature, please check if 'graph_kernel_flags' is purposely set."
      << std::endl;
  }
}

std::string GraphKernelFlags::DumpAllFlags() const {
  nlohmann::json json;

  json["dump_as_text"] = dump_as_text;
  json["enable_stitch_fusion"] = enable_stitch_fusion;
  json["enable_recompute_fusion"] = enable_recompute_fusion;
  json["enable_parallel_fusion"] = enable_parallel_fusion;
  json["enable_horizontal_fusion"] = enable_horizontal_fusion;
  json["enable_auto_tensor_inplace"] = enable_auto_tensor_inplace;
  json["enable_dynamic_batch"] = enable_dynamic_batch;
  json["enable_csr_fusion"] = enable_csr_fusion;
  json["enable_debug_mode"] = enable_debug_mode;
  json["enable_lite_conv_tuning"] = enable_lite_conv_tuning;
  json["enable_vectorization"] = enable_vectorization;
  json["enable_dynamic_shape_fusion"] = enable_dynamic_shape_fusion;
  json["exact_precision_mode"] = exact_precision_mode;

  json["opt_level"] = opt_level;
  json["fusion_ops_level"] = fusion_ops_level;
  json["parallel_ops_level"] = parallel_ops_level;
  json["reduce_fuse_depth"] = reduce_fuse_depth;
  json["online_tuning"] = online_tuning;
  json["cpu_refer_thread_num"] = cpu_refer_thread_num;
  json["recompute_increment_threshold"] = recompute_increment_threshold;
  json["recompute_peak_threshold"] = recompute_peak_threshold;
  json["composite_op_limit_size"] = composite_op_limit_size;

  json["repository_path"] = repository_path;
  json["target_os"] = target_os;
  json["cpu_arch"] = cpu_arch;
  json["cpu_feature"] = cpu_feature;
  json["cpu_type"] = cpu_type;

  json["kernel_generator"] = kernel_generator;

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
  json["enable_cce_lib"] = enable_cce_lib;
  json["enable_cce_lib_ops"] = enable_cce_lib_ops_only;
  json["enable_cce_lib_ops_only"] = enable_cce_lib_ops_only;
  json["disable_cce_lib_ops"] = disable_cce_lib_ops;
  json["enable_packet_ops_only"] = enable_packet_ops_only;
  json["disable_packet_ops"] = disable_packet_ops;
  json["disable_matmul_post_fusion"] = disable_matmul_post_fusion;
  json["enable_allreduce_prologue_fusion"] = enable_allreduce_prologue_fusion;
  json["enable_allreduce_epilogue_fusion"] = enable_allreduce_epilogue_fusion;
  json["enable_fusion_pattern_only"] = enable_fusion_pattern_only;
  json["disable_fusion_pattern"] = disable_fusion_pattern;

  return json.dump();
}

GraphKernelPassChecker &GraphKernelPassChecker::GetInstance() {
  static GraphKernelPassChecker instance;
  return instance;
}

void GraphKernelPassChecker::Init() {
  enable_pass_active_ = std::vector<bool>(GraphKernelFlags::GetInstance().enable_pass.size(), false);
  disable_pass_active_ = std::vector<bool>(GraphKernelFlags::GetInstance().disable_pass.size(), false);
}

void GraphKernelPassChecker::SetEnablePassActive(size_t index, bool value) {
  if (index < GraphKernelFlags::GetInstance().enable_pass.size()) {
    enable_pass_active_[index] = value;
  }
}
void GraphKernelPassChecker::SetDisablePassActive(size_t index, bool value) {
  if (index < GraphKernelFlags::GetInstance().disable_pass.size()) {
    disable_pass_active_[index] = value;
  }
}

void GraphKernelPassChecker::PassFlagsValidation() {
  for (size_t i = 0; i < enable_pass_active_.size(); ++i) {
    if (enable_pass_active_[i]) {
      continue;
    }
    MS_LOG(WARNING) << "graph kernel pass enable flag \"" << GraphKernelFlags::GetInstance().enable_pass[i]
                    << "\" is not valid!";
  }

  for (size_t i = 0; i < disable_pass_active_.size(); ++i) {
    if (disable_pass_active_[i]) {
      continue;
    }
    MS_LOG(WARNING) << "graph kernel pass disable flag \"" << GraphKernelFlags::GetInstance().disable_pass[i]
                    << "\" is not valid!";
  }
}
}  // namespace mindspore::graphkernel
