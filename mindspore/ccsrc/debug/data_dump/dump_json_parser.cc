/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "debug/data_dump/dump_json_parser.h"
#include <fstream>
#include "utils/log_adapter.h"
#include "debug/common.h"
#include "utils/ms_context.h"
#include "utils/convert_utils_base.h"
#include "backend/session/anf_runtime_algorithm.h"
#include "debug/data_dump/npy_header.h"
#include "debug/anf_ir_utils.h"
#include "utils/comm_manager.h"

namespace {
constexpr auto kCommonDumpSettings = "common_dump_settings";
constexpr auto kAsyncDumpSettings = "async_dump_settings";
constexpr auto kE2eDumpSettings = "e2e_dump_settings";
constexpr auto kDumpMode = "dump_mode";
constexpr auto kPath = "path";
constexpr auto kNetName = "net_name";
constexpr auto kIteration = "iteration";
constexpr auto kInputOutput = "input_output";
constexpr auto kKernels = "kernels";
constexpr auto kSupportDevice = "support_device";
constexpr auto kEnable = "enable";
constexpr auto kOpDebugMode = "op_debug_mode";
constexpr auto kTransFlag = "trans_flag";
constexpr auto kDumpInputAndOutput = 0;
constexpr auto kDumpInputOnly = 1;
constexpr auto kDumpOutputOnly = 2;
constexpr auto kMindsporeDumpConfig = "MINDSPORE_DUMP_CONFIG";
}  // namespace

namespace mindspore {
auto DumpJsonParser::CheckJsonKeyExist(const nlohmann::json &content, const std::string &key) {
  auto iter = content.find(key);
  if (iter == content.end()) {
    MS_LOG(EXCEPTION) << "Check dump json failed, " << key << " not found";
  }
  return iter;
}

std::string GetIfstreamString(const std::ifstream &ifstream) {
  std::stringstream buffer;
  buffer << ifstream.rdbuf();
  return buffer.str();
}

bool DumpJsonParser::IsDumpEnabled() {
  auto config_path = common::GetEnv(kMindsporeDumpConfig);
  if (config_path.empty()) {
    return false;
  }
  MS_LOG(INFO) << "Dump config path is " << config_path;

  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  if (context->get_param<int>(MS_CTX_EXECUTION_MODE) == kPynativeMode) {
    MS_LOG(WARNING) << "Dump is disabled in PyNative mode";
    return false;
  }
  return true;
}

void DumpJsonParser::Parse() {
  std::lock_guard<std::mutex> guard(lock_);
  if (already_parsed_) {
    return;
  }
  already_parsed_ = true;
  if (!IsDumpEnabled()) {
    return;
  }

  auto dump_config_file = Common::GetConfigFile(kMindsporeDumpConfig);
  if (!dump_config_file.has_value()) {
    MS_LOG(EXCEPTION) << "Get dump config file failed";
  }

  auto dump_file_realpath = Common::GetRealPath(dump_config_file.value());
  if (!dump_file_realpath.has_value()) {
    MS_LOG(EXCEPTION) << "Get real path failed in Parse.";
  }

  std::ifstream json_file(dump_file_realpath.value());
  if (!json_file.is_open()) {
    MS_LOG(EXCEPTION) << "Dump file:" << dump_config_file.value() << " open failed.";
  }

  nlohmann::json j;
  try {
    json_file >> j;
  } catch (nlohmann::json::parse_error &e) {
    MS_LOG(ERROR) << "Dump json contents:" << GetIfstreamString(json_file);
    json_file.close();
    MS_LOG(EXCEPTION) << "Parse dump json failed, error:" << e.what();
  }

  // convert json to string
  std::stringstream ss;
  ss << j;
  std::string cfg = ss.str();
  json_file.close();
  MS_LOG(INFO) << "Dump json:" << cfg;

  ParseE2eDumpSetting(j);
  ParseCommonDumpSetting(j);
  JudgeDumpEnabled();
}

void DumpJsonParser::CopyJsonToDir(uint32_t rank_id) {
  this->Parse();
  if (!IsDumpEnabled()) {
    return;
  }
  auto dump_config_file = Common::GetConfigFile(kMindsporeDumpConfig);
  if (!dump_config_file.has_value()) {
    MS_LOG(EXCEPTION) << "Get dump config file failed";
  }
  std::ifstream json_file(dump_config_file.value());
  if (async_dump_enabled_ || e2e_dump_enabled_) {
    auto realpath = Common::GetRealPath(path_ + "/rank_" + std::to_string(rank_id) + "/.dump_metadata/data_dump.json");
    if (!realpath.has_value()) {
      MS_LOG(ERROR) << "Get real path failed in CopyJsonDir.";
    } else {
      const std::string file_path = realpath.value();
      ChangeFileMode(file_path, S_IWUSR);
      std::ofstream json_copy(file_path);
      json_copy << json_file.rdbuf();
      json_copy.close();
      ChangeFileMode(file_path, S_IRUSR);
    }
  }
}

void DumpJsonParser::CopyHcclJsonToDir(uint32_t rank_id) {
  if (!IsDumpEnabled()) {
    return;
  }
  std::string config_path = common::GetEnv("MINDSPORE_HCCL_CONFIG_PATH");
  if (config_path.empty()) {
    config_path = common::GetEnv("RANK_TABLE_FILE");
    if (config_path.empty()) {
      MS_LOG(INFO) << "Get hccl json config failed.";
      return;
    }
  }
  std::ifstream json_file(config_path);
  auto realpath = Common::GetRealPath(path_ + "/rank_" + std::to_string(rank_id) + "/.dump_metadata/hccl.json");
  if (!realpath.has_value()) {
    MS_LOG(ERROR) << "Get real path failed in CopyHcclJsonToDir.";
  } else {
    const std::string file_path = realpath.value();
    ChangeFileMode(file_path, S_IWUSR);
    std::ofstream json_copy(file_path);
    json_copy << json_file.rdbuf();
    json_copy.close();
    ChangeFileMode(file_path, S_IRUSR);
  }
}

void DumpJsonParser::CopyMSCfgJsonToDir(uint32_t rank_id) {
  if (!IsDumpEnabled()) {
    return;
  }
  auto realpath = Common::GetRealPath(path_ + "/rank_" + std::to_string(rank_id) + "/.dump_metadata/config.json");
  if (!realpath.has_value()) {
    MS_LOG(ERROR) << "Get real path failed in CopyMSConfigJsonToDir.";
  } else {
    nlohmann::json ms_info;
    auto context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(context);
    ms_info["device_target"] = context->get_param<std::string>(MS_CTX_DEVICE_TARGET);
    ms_info["ms_version"] = "1.3.0";
    const std::string file_path = realpath.value();
    ChangeFileMode(file_path, S_IWUSR);
    std::ofstream json_create(file_path);
    json_create << ms_info;
    json_create.close();
    ChangeFileMode(file_path, S_IRUSR);
  }
}

bool DumpJsonParser::GetIterDumpFlag() const { return e2e_dump_enabled_ && IsDumpIter(cur_dump_iter_); }

bool DumpJsonParser::DumpToFile(const std::string &filename, const void *data, size_t len, const ShapeVector &shape,
                                TypeId type) {
  if (filename.empty() || data == nullptr || len == 0) {
    MS_LOG(ERROR) << "Incorrect parameter.";
    return false;
  }

  auto realpath = Common::GetRealPath(filename + ".npy");
  if (!realpath.has_value()) {
    MS_LOG(ERROR) << "Get real path failed.";
    return false;
  }
  const std::string file_path = realpath.value();
  ChangeFileMode(file_path, S_IWUSR);
  std::ofstream fd(file_path, std::ios::out | std::ios::trunc | std::ios::binary);
  if (!fd.is_open()) {
    MS_LOG(ERROR) << "Open file " << file_path << " failed.";
    return false;
  }
  std::string npy_header = GenerateNpyHeader(shape, type);
  if (!npy_header.empty()) {
    fd << npy_header;
    (void)fd.write(reinterpret_cast<const char *>(data), SizeToLong(len));
    fd.close();
    ChangeFileMode(file_path, S_IRUSR);
  }
  return true;
}

void DumpJsonParser::ParseCommonDumpSetting(const nlohmann::json &content) {
  // async_dump is enabled by default, if e2e dump is enabled it will override this
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  if (context->get_param<std::string>(MS_CTX_DEVICE_TARGET) == kAscendDevice) {
    async_dump_enabled_ = true;
  } else if (context->get_param<std::string>(MS_CTX_DEVICE_TARGET) == kGPUDevice) {
    if (!e2e_dump_enabled_) {
      e2e_dump_enabled_ = true;
      trans_flag_ = true;
    }
  }

  auto common_dump_settings = CheckJsonKeyExist(content, kCommonDumpSettings);
  auto dump_mode = CheckJsonKeyExist(*common_dump_settings, kDumpMode);
  auto path = CheckJsonKeyExist(*common_dump_settings, kPath);
  auto net_name = CheckJsonKeyExist(*common_dump_settings, kNetName);
  auto iteration = CheckJsonKeyExist(*common_dump_settings, kIteration);
  auto input_output = CheckJsonKeyExist(*common_dump_settings, kInputOutput);
  auto kernels = CheckJsonKeyExist(*common_dump_settings, kKernels);
  auto support_device = CheckJsonKeyExist(*common_dump_settings, kSupportDevice);

  nlohmann::detail::iter_impl<const nlohmann::json> op_debug_mode;
  if (!e2e_dump_enabled_) {
    op_debug_mode = CheckJsonKeyExist(*common_dump_settings, kOpDebugMode);
  }

  ParseDumpMode(*dump_mode);
  ParseDumpPath(*path);
  ParseNetName(*net_name);
  ParseIteration(*iteration);
  ParseInputOutput(*input_output);
  ParseKernels(*kernels);
  ParseSupportDevice(*support_device);
  if (!e2e_dump_enabled_) {
    ParseOpDebugMode(*op_debug_mode);
  }
}

void DumpJsonParser::ParseE2eDumpSetting(const nlohmann::json &content) {
  auto e2e_dump_setting = content.find(kE2eDumpSettings);
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  if (e2e_dump_setting == content.end()) {
    MS_LOG(INFO) << "No e2e_dump_settings";
    return;
  }

  auto e2e_dump_enable = CheckJsonKeyExist(*e2e_dump_setting, kEnable);
  auto trans_flag = CheckJsonKeyExist(*e2e_dump_setting, kTransFlag);

  e2e_dump_enabled_ = ParseEnable(*e2e_dump_enable);
  if (e2e_dump_enabled_ && context->get_param<std::string>(MS_CTX_DEVICE_TARGET) == kAscendDevice) {
    MS_LOG(WARNING) << "Deprecated: Synchronous dump mode is deprecated and will be removed in a future release";
  }
  trans_flag_ = ParseEnable(*trans_flag);
}

void CheckJsonUnsignedType(const nlohmann::json &content, const std::string &key) {
  if (!content.is_number_unsigned()) {
    MS_LOG(EXCEPTION) << "Dump config parse failed, " << key << " should be unsigned int type";
  }
}

void CheckJsonStringType(const nlohmann::json &content, const std::string &key) {
  if (!content.is_string()) {
    MS_LOG(EXCEPTION) << "Dump config parse failed, " << key << " should be string type";
  }
}

void CheckJsonArrayType(const nlohmann::json &content, const std::string &key) {
  if (!content.is_array()) {
    MS_LOG(EXCEPTION) << "Dump config parse failed, " << key << " should be array type";
  }
}

void DumpJsonParser::ParseDumpMode(const nlohmann::json &content) {
  CheckJsonUnsignedType(content, kDumpMode);
  dump_mode_ = content;
  if (dump_mode_ != 0 && dump_mode_ != 1) {
    MS_LOG(EXCEPTION) << "Dump config parse failed, dump_mode should be 0 or 1, but got " << dump_mode_;
  }
}

void DumpJsonParser::ParseDumpPath(const nlohmann::json &content) {
  CheckJsonStringType(content, kPath);
  path_ = content;
  if (!std::all_of(path_.begin(), path_.end(),
                   [](char c) { return ::isalpha(c) || ::isdigit(c) || c == '-' || c == '_' || c == '/'; })) {
    MS_LOG(EXCEPTION) << "Dump path only support alphabets, digit or {'-', '_', '/'}, but got:" << path_;
  }
  if (path_.empty()) {
    MS_LOG(EXCEPTION) << "Dump path is empty";
  }
  if (path_[0] != '/') {
    MS_LOG(EXCEPTION) << "Dump path only support absolute path and should start with '/'";
  }
}

void DumpJsonParser::ParseNetName(const nlohmann::json &content) {
  CheckJsonStringType(content, kNetName);
  net_name_ = content;
  if (net_name_.empty() || !std::all_of(net_name_.begin(), net_name_.end(),
                                        [](char c) { return ::isalpha(c) || ::isdigit(c) || c == '-' || c == '_'; })) {
    MS_LOG(EXCEPTION) << "net_name only supports alphabetic, digit, or {'-', '_'}, but got: " << net_name_;
  }
}

void DumpJsonParser::ParseIteration(const nlohmann::json &content) {
  CheckJsonStringType(content, kIteration);
  auto context = MsContext::GetInstance();
  if (e2e_dump_enabled_ || async_dump_enabled_) {
    iteration_ = content;
    if (iteration_.empty() || (!std::all_of(iteration_.begin(), iteration_.end(), [](char c) {
          return ::isdigit(c) || c == '-' || c == '|';
        }) && iteration_ != "all")) {
      MS_LOG(EXCEPTION) << "iteration only supports digits, {'-', '|'}, or just \"all\" but got: " << iteration_;
    }
  } else if (context->get_param<std::string>(MS_CTX_DEVICE_TARGET) == kCPUDevice) {
    MS_LOG(WARNING) << "Dump is not enabled. ";
  } else {
    MS_LOG(EXCEPTION) << "Dump Json Parse Failed. Async or E2E should be enabled. ";
  }
}

bool DumpJsonParser::IsDumpIter(uint32_t iteration) const {
  // bool DumpJsonParser::IsDumpIter(uint32_t iteration) --> checks if iteration should be dumped or not.
  if (iteration_ == "all") {
    return true;
  }
  const std::string vertical_bar = "|";
  const std::string dash = "-";
  int start = 0;
  unsigned int end = (unsigned int)(iteration_.find(vertical_bar));
  while (iteration_.find(vertical_bar, IntToSize(start)) != std::string::npos) {
    std::string temp = iteration_.substr(IntToSize(start), IntToSize(end - start));
    unsigned int range_idx = (unsigned int)(temp.find(dash));
    if (temp.find(dash) != std::string::npos) {
      uint32_t low_range = std::stoul(temp.substr(0, IntToSize(range_idx)));
      uint32_t high_range = std::stoul(temp.substr(IntToSize((range_idx + 1)), -1));
      if ((low_range <= iteration) && (iteration <= high_range)) {
        return true;
      }
    } else if (iteration == std::stoul(temp)) {
      return true;
    }
    start = end + 1;
    end = (unsigned int)(iteration_.find(vertical_bar, IntToSize(start)));
  }
  std::string temp = iteration_.substr(IntToSize(start), IntToSize(end - start));
  int range_idx = (unsigned int)(temp.find(dash));
  if (temp.find(dash) != std::string::npos) {
    uint32_t low_range = std::stoul(temp.substr(0, IntToSize(range_idx)));
    uint32_t high_range = std::stoul(temp.substr(IntToSize(range_idx + 1), -1));
    if ((low_range <= iteration) && (iteration <= high_range)) {
      return true;
    }
  } else if (iteration == std::stoul(temp)) {
    return true;
  }
  return false;
}

bool DumpJsonParser::IsSingleIter() {
  // bool DumpJsonParser::IsSingleIter() --> checks if iteration in json dump file is single or not.
  if (iteration_ != "all" && iteration_.find("-") == std::string::npos && iteration_.find("|") == std::string::npos) {
    return true;
  }
  return false;
}

void DumpJsonParser::ParseInputOutput(const nlohmann::json &content) {
  CheckJsonUnsignedType(content, kInputOutput);
  input_output_ = content;
  const uint32_t max_inout_num = 2;
  if (input_output_ < 0 || input_output_ > max_inout_num) {
    MS_LOG(EXCEPTION) << "Dump Json Parse Failed. input_output should be 0, 1, 2";
  }
}

void DumpJsonParser::ParseKernels(const nlohmann::json &content) {
  CheckJsonArrayType(content, kKernels);

  for (const auto &kernel : content) {
    auto kernel_str = kernel.dump();
    kernel_str.erase(std::remove(kernel_str.begin(), kernel_str.end(), '\"'), kernel_str.end());
    MS_LOG(INFO) << "Need dump kernel:" << kernel_str;
    auto ret = kernels_.try_emplace({kernel_str, 0});
    if (!ret.second) {
      MS_LOG(WARNING) << "Duplicate dump kernel name:" << kernel_str;
    }
  }
}

void DumpJsonParser::ParseSupportDevice(const nlohmann::json &content) {
  CheckJsonArrayType(content, kSupportDevice);
  for (const auto &device : content) {
    uint32_t device_id = device;
    MS_LOG(INFO) << "Dump support device:" << device_id;
    auto ret = support_devices_.emplace(device_id);
    if (!ret.second) {
      MS_LOG(WARNING) << "Duplicate support device:" << device_id;
    }
  }
}

bool DumpJsonParser::ParseEnable(const nlohmann::json &content) {
  if (!content.is_boolean()) {
    MS_LOG(EXCEPTION) << "Dump Json Parse Failed. 'enable' should be boolean type";
  }
  return content;
}

void DumpJsonParser::ParseOpDebugMode(const nlohmann::json &content) {
  CheckJsonUnsignedType(content, kOpDebugMode);
  op_debug_mode_ = content;
  const size_t max_mode = 3;
  if (op_debug_mode_ < 0 || op_debug_mode_ > max_mode) {
    MS_LOG(EXCEPTION) << "Dump Json Parse Failed. op_debug_mode should be 0, 1, 2, 3";
  }
}

void DumpJsonParser::JsonConfigToString() {
  std::string cur_config;
  cur_config.append("dump_mode:");
  cur_config.append(std::to_string(dump_mode_));
  cur_config.append(" path:");
  cur_config.append(path_);
  cur_config.append(" net_name:");
  cur_config.append(net_name_);
  cur_config.append(" iteration:");
  cur_config.append(iteration_);
  cur_config.append(" input_output:");
  cur_config.append(std::to_string(input_output_));
  cur_config.append("e2e_enable:");
  cur_config.append(std::to_string(static_cast<int>(e2e_dump_enabled_)));
  cur_config.append(" async_dump_enable:");
  cur_config.append(std::to_string(static_cast<int>(async_dump_enabled_)));
  MS_LOG(INFO) << cur_config;
}

void DumpJsonParser::JudgeDumpEnabled() {
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  if (context->get_param<std::string>(MS_CTX_DEVICE_TARGET) == kGPUDevice) {
    async_dump_enabled_ = false;
  }

  if (context->get_param<std::string>(MS_CTX_DEVICE_TARGET) == kAscendDevice) {
    if (async_dump_enabled_ && e2e_dump_enabled_) {
      async_dump_enabled_ = false;
      MS_LOG(INFO) << "Disable async dump";
    }
  }

  if (!async_dump_enabled_ && !e2e_dump_enabled_) {
    MS_LOG(WARNING) << "Dump json parse failed. Dump is not enabled";
  }
  if (context->get_param<std::string>(MS_CTX_DEVICE_TARGET) != kCPUDevice) {
    auto device_id = context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
    if (support_devices_.find(device_id) == support_devices_.end()) {
      async_dump_enabled_ = false;
      e2e_dump_enabled_ = false;
      MS_LOG(WARNING) << "Dump is not enabled. device_id:" << device_id << " not support";
    }
  }
  JsonConfigToString();
}

bool DumpJsonParser::NeedDump(const std::string &op_full_name) const {
  if (dump_mode_ == 0) {
    return true;
  }
  auto iter = kernels_.find(op_full_name);
  return iter != kernels_.end();
}

void DumpJsonParser::MatchKernel(const std::string &kernel_name) {
  auto iter = kernels_.find(kernel_name);
  if (iter == kernels_.end()) {
    return;
  }
  iter->second = iter->second + 1;
  MS_LOG(INFO) << "Match dump kernel:" << iter->first << " match times:" << iter->second;
}

void DumpJsonParser::PrintUnusedKernel() {
  if (!e2e_dump_enabled_ && !async_dump_enabled_) {
    return;
  }
  for (const auto &iter : kernels_) {
    if (iter.second == 0) {
      MS_LOG(WARNING) << "[DataDump] Unused Kernel in json:" << iter.first;
    }
  }
}

std::string DumpJsonParser::GetOpOverflowBinPath(uint32_t graph_id) const {
  std::string bin_path;
  (void)bin_path.append(path_);
  (void)bin_path.append("/");
  (void)bin_path.append("rank_");

  uint32_t rank_id = 0;
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto env_rank_id = common::GetEnv("RANK_ID");
  if (ms_context->get_param<bool>(MS_CTX_ENABLE_HCCL) && !env_rank_id.empty()) {
    // get actual rank id if it's distribution training case.
    if (!CommManager::GetInstance().GetRankID(kHcclWorldGroup, &rank_id)) {
      MS_LOG(INFO) << "Failed to get rank id.";
    }
  }
  bin_path.append(std::to_string(rank_id));

  bin_path.append("/");
  bin_path.append(net_name_);
  bin_path.append("/");
  bin_path.append(std::to_string(graph_id));
  bin_path.append("/");
  bin_path.append(std::to_string(cur_dump_iter_));
  bin_path.append("/");

  return bin_path;
}

bool DumpJsonParser::InputNeedDump() const {
  return input_output_ == kDumpInputAndOutput || input_output_ == kDumpInputOnly;
}

bool DumpJsonParser::OutputNeedDump() const {
  return input_output_ == kDumpInputAndOutput || input_output_ == kDumpOutputOnly;
}

void DumpJsonParser::UpdateNeedDumpKernels(NotNull<const session::KernelGraph *> kernel_graph) {
  if (!async_dump_enabled_) {
    return;
  }
  MS_LOG(INFO) << "Update async dump kernel list for hccl";
  std::map<std::string, uint32_t> update_kernels;
  for (const auto &kernel : kernel_graph->execution_order()) {
    MS_EXCEPTION_IF_NULL(kernel);
    if (AnfAlgo::GetKernelType(kernel) == HCCL_KERNEL &&
        DumpJsonParser::GetInstance().NeedDump(GetKernelNodeName(kernel))) {
      auto input_size = AnfAlgo::GetInputTensorNum(kernel);
      for (size_t i = 0; i < input_size; ++i) {
        auto input_with_index = AnfAlgo::GetPrevNodeOutput(kernel, i);
        auto input = input_with_index.first;
        if (input->isa<CNode>()) {
          MS_LOG(INFO) << "[AsyncDump] Match Hccl Node:" << GetKernelNodeName(kernel)
                       << " Input:" << GetKernelNodeName(input);
          (void)update_kernels.try_emplace(GetKernelNodeName(input), 0);
        }
      }
    }
  }
  kernels_.insert(update_kernels.begin(), update_kernels.end());
}
}  // namespace mindspore
