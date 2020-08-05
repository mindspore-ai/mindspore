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

#include "debug/data_dump_parser.h"

#include <fstream>
#include "utils/ms_context.h"
#include "debug/common.h"

static constexpr auto kDataDumpConfigPtah = "DATA_DUMP_CONFIG_PATH";
static constexpr auto kEnableDataDump = "ENABLE_DATA_DUMP";
static constexpr auto kDataDumpPath = "DATA_DUMP_PATH";
static constexpr auto kConfigDumpMode = "dump_mode";
static constexpr auto kConfigOpDebugMode = "op_debug_mode";
static constexpr auto kConfigNetName = "net_name";
static constexpr auto kConfigIteration = "iteration";
static constexpr auto kConfigKernels = "kernels";

namespace mindspore {
void DataDumpParser::ResetParam() {
  enable_ = false;
  net_name_.clear();
  dump_mode_ = 0;
  dump_step_ = 0;
  kernel_map_.clear();
}

bool DataDumpParser::DumpEnabled() const {
  auto enable_dump = std::getenv(kEnableDataDump);
  if (enable_dump == nullptr) {
    MS_LOG(INFO) << "[DataDump] enable dump is null. Please export ENABLE_DATA_DUMP";
    return false;
  }

  auto enabled = std::atoi(enable_dump);
  if (enabled != 1) {
    MS_LOG(WARNING) << "[DataDump] Please export ENABLE_DATA_DUMP=1";
    return false;
  }

  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  if (context->execution_mode() == kPynativeMode) {
    MS_LOG(EXCEPTION) << "[DataDump] PyNative mode not support data dump";
  }
  return true;
}

std::optional<std::string> DataDumpParser::GetDumpPath() const {
  auto dump_path = std::getenv(kDataDumpPath);
  if (dump_path == nullptr) {
    MS_LOG(ERROR) << "[DataDump] dump path is null. Please export DATA_DUMP_PATH";
    return {};
  }
  std::string dump_path_str(dump_path);
  if (!std::all_of(dump_path_str.begin(), dump_path_str.end(),
                   [](char c) { return ::isalpha(c) || ::isdigit(c) || c == '-' || c == '_' || c == '/'; })) {
    MS_LOG(EXCEPTION) << "[DataDump] dump path only support alphabets, digit or {'-', '_', '/'}, but got:"
                      << dump_path_str;
  }
  return dump_path_str;
}

std::string GetIfstreamString(const std::ifstream &ifstream) {
  std::stringstream buffer;
  buffer << ifstream.rdbuf();
  return buffer.str();
}

void DataDumpParser::ParseDumpConfig() {
  std::lock_guard<std::mutex> guard(lock_);
  MS_LOG(INFO) << "[DataDump] parse start";
  if (!DumpEnabled()) {
    MS_LOG(INFO) << "[DataDump] dump not enable";
    return;
  }

  ResetParam();

  auto dump_config_file = Common::GetConfigFile(kDataDumpConfigPtah);
  if (!dump_config_file.has_value()) {
    MS_LOG(EXCEPTION) << "[DataDump] Get config file failed";
  }

  std::ifstream json_file(dump_config_file.value());
  if (!json_file.is_open()) {
    MS_LOG(EXCEPTION) << "[DataDump] " << dump_config_file.value() << " open failed.";
  }

  nlohmann::json j;
  try {
    json_file >> j;
  } catch (nlohmann::json::parse_error &e) {
    MS_LOG(ERROR) << "[DataDump] json contents:" << GetIfstreamString(json_file);
    MS_LOG(EXCEPTION) << "[DataDump] parse json failed, error:" << e.what();
  }
  if (j.find("DumpSettings") == j.end()) {
    MS_LOG(EXCEPTION) << "[DataDump] DumpSettings is not exist.";
  }

  nlohmann::json dump_settings = j.at("DumpSettings");
  // convert json to string
  std::stringstream ss;
  ss << dump_settings;
  std::string cfg = ss.str();
  MS_LOG(INFO) << "[DataDump] Async dump settings Json: " << cfg;
  if (!IsConfigExist(dump_settings)) {
    MS_LOG(EXCEPTION) << "[DataDump] Async dump json invalid";
  }

  if (!ParseDumpSetting(dump_settings)) {
    MS_LOG(EXCEPTION) << "[DataDump] Parse dump json failed";
  }
}

bool DataDumpParser::NeedDump(const std::string &op_full_name) const {
  if (!DumpEnabled()) {
    return false;
  }
  if (dump_mode_ == 0) {
    return true;
  }
  auto iter = kernel_map_.find(op_full_name);
  return iter != kernel_map_.end();
}

bool DataDumpParser::IsConfigExist(const nlohmann::json &dump_settings) const {
  if (dump_settings.find(kConfigDumpMode) == dump_settings.end() ||
      dump_settings.find(kConfigNetName) == dump_settings.end() ||
      dump_settings.find(kConfigOpDebugMode) == dump_settings.end() ||
      dump_settings.find(kConfigIteration) == dump_settings.end() ||
      dump_settings.find(kConfigKernels) == dump_settings.end()) {
    MS_LOG(ERROR) << "[DataDump] DumpSettings keys are not exist.";
    return false;
  }
  return true;
}

bool DataDumpParser::ParseDumpSetting(const nlohmann::json &dump_settings) {
  auto mode = dump_settings.at(kConfigDumpMode);
  auto op_debug_mode = dump_settings.at(kConfigOpDebugMode);
  auto net_name = dump_settings.at(kConfigNetName);
  auto iteration = dump_settings.at(kConfigIteration);
  auto kernels = dump_settings.at(kConfigKernels);
  if (!(mode.is_number_unsigned() && op_debug_mode.is_number_unsigned() && net_name.is_string() &&
        iteration.is_number_unsigned() && kernels.is_array())) {
    MS_LOG(ERROR) << "[DataDump] Element's type in Dump config json is invalid.";
    enable_ = false;
    return false;
  }

  CheckDumpMode(mode);
  CheckOpDebugMode(op_debug_mode);

  enable_ = true;
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  dump_mode_ = mode;
  op_debug_mode_ = op_debug_mode;
  net_name_ = net_name;
  dump_step_ = iteration;
  for (const auto &kernel : kernels) {
    auto kernel_str = kernel.dump();
    kernel_str.erase(std::remove(kernel_str.begin(), kernel_str.end(), '\"'), kernel_str.end());
    MS_LOG(INFO) << "[DataDump] Need dump kernel:" << kernel_str;
    kernel_map_.insert({kernel_str, 0});
  }
  return true;
}

void DataDumpParser::MatchKernel(const std::string &kernel_name) {
  auto iter = kernel_map_.find(kernel_name);
  if (iter == kernel_map_.end()) {
    return;
  }
  iter->second = iter->second + 1;
  MS_LOG(INFO) << "Match dump kernel:" << iter->first << " match times:" << iter->second;
}

void DataDumpParser::PrintUnusedKernel() {
  for (const auto &iter : kernel_map_) {
    if (iter.second == 0) {
      MS_LOG(WARNING) << "[DataDump] Unused Kernel in json:" << iter.first;
    }
  }
}

void DataDumpParser::CheckDumpMode(uint32_t dump_mode) const {
  if (dump_mode != 0 && dump_mode != 1) {
    MS_LOG(EXCEPTION) << "[DataDump] dump_mode in config json should be 0 or 1";
  }
}

void DataDumpParser::CheckOpDebugMode(uint32_t op_debug_mode) const {
  if (op_debug_mode < 0 || op_debug_mode > 3) {
    MS_LOG(EXCEPTION) << "[DataDump] op_debug_mode in config json file should be [0-3]";
  }
}
}  // namespace mindspore
