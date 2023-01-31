/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
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
#include <algorithm>
#include "utils/log_adapter.h"
#include "include/common/debug/common.h"
#include "debug/utils.h"
#include "utils/ms_context.h"
#include "utils/convert_utils_base.h"
#include "backend/common/session/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "debug/data_dump/npy_header.h"
#include "include/common/debug/anf_dump_utils.h"
#include "include/common/utils/comm_manager.h"
#include "mindspore/core/utils/file_utils.h"

namespace {
constexpr auto kCommonDumpSettings = "common_dump_settings";
constexpr auto kE2eDumpSettings = "e2e_dump_settings";
constexpr auto kDumpMode = "dump_mode";
constexpr auto kPath = "path";
constexpr auto kNetName = "net_name";
constexpr auto kSavedData = "saved_data";
constexpr auto kIteration = "iteration";
constexpr auto kInputOutput = "input_output";
constexpr auto kKernels = "kernels";
constexpr auto kSupportDevice = "support_device";
constexpr auto kEnable = "enable";
constexpr auto kOpDebugMode = "op_debug_mode";
constexpr auto kTransFlag = "trans_flag";
constexpr auto kStatisticDump = "statistic";
constexpr auto kTensorDump = "tensor";
constexpr auto kFullDump = "full";
constexpr auto kFileFormat = "file_format";
constexpr auto kDumpInputAndOutput = 0;
constexpr auto kDumpInputOnly = 1;
constexpr auto kDumpOutputOnly = 2;
constexpr auto kMindsporeDumpConfig = "MINDSPORE_DUMP_CONFIG";
}  // namespace

namespace mindspore {
auto DumpJsonParser::CheckJsonKeyExist(const nlohmann::json &content, const std::string &key) {
  nlohmann::json::const_iterator iter = content.find(key);
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
  auto single_op = common::GetEnv(kGraphOpRun);
  auto config_path = common::GetEnv(kMindsporeDumpConfig);
  if (config_path.empty()) {
    return false;
  }
  // Dump is supported with Ascend kernel-by-kernel mode (mindRT) when kGraphOpRun is set.
  if (!single_op.empty() && single_op == "1" && !MsContext::GetInstance()->get_param<bool>(MS_CTX_ENABLE_MINDRT)) {
    if (!dump_enabled_warning_printed_) {
      MS_LOG(WARNING) << "Dump is not supported when task is not sink. Please set env GRAPH_OP_RUN to 0 to enable task "
                         "sink, so that the data can be dumped.";
      // Only print the warning once.
      dump_enabled_warning_printed_ = true;
    }

    return false;
  }
  MS_LOG(INFO) << "Dump config path is " << config_path;

  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  if (context->get_param<int>(MS_CTX_EXECUTION_MODE) == kPynativeMode) {
    MS_LOG(EXCEPTION) << "Dump is disabled in PyNative mode. Please set mode to GRAPH_MODE in context.";
  }
  return true;
}

/*
 * Feature group: Dump.
 * Target device group: Ascend, GPU and CPU.
 * Runtime category: Old runtime, MindRT.
 * Description: Parse the configuration option in dump json file pointed by environment variable MINDSPORE_DUMP_CONFIG.
 */
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

  std::ifstream json_file(dump_config_file.value());
  if (!json_file.is_open()) {
    MS_LOG(EXCEPTION) << "Dump file:" << dump_config_file.value() << " open failed."
                      << " Errno:" << errno;
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

void WriteJsonFile(const std::string &file_path, const std::ifstream &json_file) {
  ChangeFileMode(file_path, S_IWUSR);
  std::ofstream json_copy(file_path);
  if (!json_copy.is_open()) {
    MS_LOG(EXCEPTION) << "Json file " << file_path << "open failed!";
  }
  json_copy << json_file.rdbuf();
  json_copy.close();
  ChangeFileMode(file_path, S_IRUSR);
}

/*
 * Feature group: Dump.
 * Target device group: Ascend, GPU and CPU.
 * Runtime category: Old runtime, MindRT.
 * Description: Copy the dump configuration file to the root directory of dump path.
 */
void DumpJsonParser::CopyDumpJsonToDir(uint32_t rank_id) {
  this->Parse();
  if (!IsDumpEnabled()) {
    return;
  }
  auto dump_config_file = Common::GetConfigFile(kMindsporeDumpConfig);
  if (!dump_config_file.has_value()) {
    MS_LOG(EXCEPTION) << "Get dump config file failed.";
  }
  std::ifstream json_file(dump_config_file.value());
  if (async_dump_enabled_ || e2e_dump_enabled_) {
    auto realpath =
      Common::CreatePrefixPath(path_ + "/rank_" + std::to_string(rank_id) + "/.dump_metadata/data_dump.json");
    if (!realpath.has_value()) {
      MS_LOG(ERROR) << "Get real path failed in CopyDumpJsonToDir.";
    } else {
      if (!Common::FileExists(realpath.value())) {
        WriteJsonFile(realpath.value(), json_file);
      } else {
        MS_LOG(WARNING) << "The file: " << realpath.value() << " is already exist, skip copy it.";
      }
    }
  }
}

/*
 * Feature group: Dump.
 * Target device group: Ascend.
 * Runtime category: Old runtime, MindRT.
 * Description: Copy the hccl configuration file to the root directory of dump path.
 */
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
  auto realpath = Common::CreatePrefixPath(path_ + "/rank_" + std::to_string(rank_id) + "/.dump_metadata/hccl.json");
  if (!realpath.has_value()) {
    MS_LOG(ERROR) << "Get real path failed in CopyHcclJsonToDir.";
  } else {
    WriteJsonFile(realpath.value(), json_file);
  }
}

/*
 * Feature group: Dump.
 * Target device group: Ascend, GPU and CPU.
 * Runtime category: Old runtime, MindRT.
 * Description: Copy the mindspore configuration file to the root directory of dump path. It provides the device and
 * ms_version information.
 */
void DumpJsonParser::CopyMSCfgJsonToDir(uint32_t rank_id) {
  if (!IsDumpEnabled()) {
    return;
  }
  auto realpath = Common::CreatePrefixPath(path_ + "/rank_" + std::to_string(rank_id) + "/.dump_metadata/config.json");
  if (!realpath.has_value()) {
    MS_LOG(ERROR) << "Get real path failed in CopyMSConfigJsonToDir.";
  } else {
    if (Common::FileExists(realpath.value())) {
      MS_LOG(WARNING) << "The file: " << realpath.value() << " is already exist, skip copy it.";
      return;
    }
    nlohmann::json ms_info;
    auto context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(context);
    ms_info["device_target"] = context->get_param<std::string>(MS_CTX_DEVICE_TARGET);
    ms_info["ms_version"] = MSVERSION;
    const std::string file_path = realpath.value();
    ChangeFileMode(file_path, S_IWUSR);
    std::ofstream json_create(file_path);
    if (!json_create.is_open()) {
      MS_LOG(EXCEPTION) << "Json file " << file_path << "open failed!";
    }
    json_create << ms_info;
    json_create.close();
    ChangeFileMode(file_path, S_IRUSR);
  }
}

bool DumpJsonParser::GetIterDumpFlag() const { return e2e_dump_enabled_ && IsDumpIter(cur_dump_iter_); }

bool DumpJsonParser::DumpEnabledForIter() const {
  return ((e2e_dump_enabled_ || async_dump_enabled_) && IsDumpIter(cur_dump_iter_));
}

/*
 * Feature group: Dump.
 * Target device group: Ascend, GPU and CPU.
 * Runtime category: Old runtime, MindRT.
 * Description: Dump data in the given address into npy file.
 */
bool DumpJsonParser::DumpToFile(const std::string &filename, const void *data, size_t len, const ShapeVector &shape,
                                TypeId type) {
  if (filename.empty() || data == nullptr || len == 0) {
    MS_LOG(ERROR) << "Incorrect parameter.";
    return false;
  }
  std::string npy_suffix = ".npy";
  std::string origin_file_path = filename + npy_suffix;
  std::optional<std::string> prefix_path;
  std::optional<std::string> origin_name;
  std::optional<std::string> mapped_name;
  bool need_map = Common::MappingName(origin_file_path, &prefix_path, &origin_name, &mapped_name);
  if (!prefix_path.has_value() || !origin_name.has_value() || !mapped_name.has_value()) {
    MS_LOG(ERROR) << "Cannot get prefix_path or file_name from: " << origin_file_path;
    return false;
  }
  std::string final_file_path = origin_file_path;
  if (need_map) {
    std::string origin_name_str = origin_name.value();
    std::string mapped_name_str = mapped_name.value();
    auto mapping_file = Common::CreatePrefixPath(prefix_path.value() + "/mapping.csv");
    if (!mapping_file.has_value()) {
      MS_LOG(ERROR) << "CreatePrefixPath for mapping.csv failed.";
      return false;
    }
    const std::string mapping_file_str = mapping_file.value();
    // try to open file
    ChangeFileMode(mapping_file_str, S_IWUSR);
    std::ofstream fout(mapping_file_str, std::ofstream::app);
    if (!fout.is_open()) {
      MS_LOG(WARNING) << "Open file for mapping.csv failed.";
      return false;
    }
    fout << mapped_name_str << "," << origin_name_str << "\n";
    fout.close();
    ChangeFileMode(mapping_file_str, S_IRUSR);
    origin_file_path = prefix_path.value() + "/" + mapped_name_str;
  }
  auto file_path = Common::CreatePrefixPath(final_file_path);
  if (!file_path.has_value()) {
    MS_LOG(ERROR) << "CreatePrefixPath failed.";
    return false;
  }
  const std::string file_path_str = file_path.value();
  MS_LOG(INFO) << "Dump path is " << file_path_str;
  ChangeFileMode(file_path_str, S_IWUSR);
  std::ofstream fd(file_path_str, std::ios::out | std::ios::trunc | std::ios::binary);
  if (!fd.is_open()) {
    MS_LOG(EXCEPTION) << "Open file " << file_path_str << " failed." << ErrnoToString(errno);
  }
  std::string npy_header = GenerateNpyHeader(shape, type);
  if (!npy_header.empty()) {
    fd << npy_header;
    (void)fd.write(reinterpret_cast<const char *>(data), SizeToLong(len));
    if (fd.bad()) {
      fd.close();
      MS_LOG(EXCEPTION) << "Write mem to file " << file_path_str << " failed.";
    }
    fd.close();
    ChangeFileMode(file_path_str, S_IRUSR);
  }
  return true;
}

void DumpJsonParser::ParseCommonDumpSetting(const nlohmann::json &content) {
  // async_dump is enabled by default, if e2e dump is enabled it will override this
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  if (context->get_param<std::string>(MS_CTX_DEVICE_TARGET) == kAscendDevice) {
    async_dump_enabled_ = true;
  } else if (!e2e_dump_enabled_) {
    e2e_dump_enabled_ = true;
    trans_flag_ = true;
  }

  auto common_dump_settings = CheckJsonKeyExist(content, kCommonDumpSettings);
  auto dump_mode = CheckJsonKeyExist(*common_dump_settings, kDumpMode);
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
  ParseDumpPath(*common_dump_settings);  // Pass in the whole json string to parse because the path field is optional.
  ParseNetName(*net_name);
  ParseIteration(*iteration);
  ParseInputOutput(*input_output);
  ParseKernels(*kernels);
  ParseSupportDevice(*support_device);
  if (!e2e_dump_enabled_) {
    ParseOpDebugMode(*op_debug_mode);
    ParseFileFormat(
      *common_dump_settings);  // Pass in the whole json string to parse because file_format field is optional.
  }
  ParseSavedData(*common_dump_settings);  // saved data optional
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
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  CheckJsonUnsignedType(content, kDumpMode);
  dump_mode_ = content;
  if (dump_mode_ > static_cast<uint32_t>(DUMP_KERNELS_WITH_FLAG)) {
    MS_LOG(EXCEPTION) << "Dump config parse failed, dump_mode should be 0, 1 or 2, but got " << dump_mode_;
  }
  if (dump_mode_ == static_cast<uint32_t>(DUMP_KERNELS_WITH_FLAG)) {
    if (context->get_param<std::string>(MS_CTX_DEVICE_TARGET) != kAscendDevice || e2e_dump_enabled_) {
      MS_LOG(EXCEPTION) << "Cell dump is only supported in Ascend async dump. Please set dump_mode to 0 or 1.";
    }
  }
}

void DumpJsonParser::ParseDumpPath(const nlohmann::json &content) {
  std::string dump_path;
  auto json_iter = content.find(kPath);
  // Check if `path` field exists in dump json file.
  if (json_iter != content.end()) {
    CheckJsonStringType(*json_iter, kPath);
    dump_path = *json_iter;
  }
  if (dump_path.empty()) {
    // If no path is found or path is set as empty in dump json file, use MS_DIAGNOSTIC_DATA_PATH/debug_dump as the dump
    // path value if the env exists.
    dump_path = common::GetEnv("MS_DIAGNOSTIC_DATA_PATH");
    if (dump_path.empty()) {
      MS_LOG(EXCEPTION)
        << "Dump path is empty. Please set it in dump json file or environment variable `MS_DIAGNOSTIC_DATA_PATH`.";
    } else {
      dump_path += "/debug_dump";
    }
  }
  path_ = dump_path;
  if (!std::all_of(path_.begin(), path_.end(),
                   [](char c) { return ::isalpha(c) || ::isdigit(c) || c == '-' || c == '_' || c == '/'; })) {
    MS_LOG(EXCEPTION) << "Dump path only support alphabets, digit or {'-', '_', '/'}, but got:" << path_;
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

void DumpJsonParser::ParseSavedData(const nlohmann::json &content) {
  saved_data_ = kTensorDump;  // default to tensor data dump
  auto json_iter = content.find(kSavedData);
  if (json_iter != content.end()) {
    CheckJsonStringType(*json_iter, kSavedData);
    saved_data_ = *json_iter;
  }
  if (saved_data_ != kStatisticDump && saved_data_ != kTensorDump && saved_data_ != kFullDump) {
    MS_LOG(EXCEPTION) << "Dump Json parse failed, saved_data only supports statistic, tensor, or full, but got: "
                      << saved_data_ << ". Please set saved_data to either statistic, tensor, or full";
  }
  auto context = MsContext::GetInstance();
  if (IsStatisticDump() && context->get_param<std::string>(MS_CTX_DEVICE_TARGET) == kCPUDevice) {
    MS_LOG(EXCEPTION) << "Dump Json parse failed, storing statistic dump is only supported on GPU and Ascend, please "
                         "set saved_data to tensor or use a GPU or Ascend device";
  }
  if (IsStatisticDump() && context->get_param<std::string>(MS_CTX_DEVICE_TARGET) == kAscendDevice) {
    if (!IsNpyFormat()) {
      MS_LOG(EXCEPTION) << "Dump Json parse failed, storing statistic dump is only supported on Ascend when "
                           "file_format is set to 'npy'.";
    }
    if (e2e_dump_enabled_) {
      MS_LOG(EXCEPTION)
        << "Dump Json parse failed, storing statistic dump is only supported on Ascend asynchronous mode.";
    }
  }
}

void DumpJsonParser::ParseIteration(const nlohmann::json &content) {
  CheckJsonStringType(content, kIteration);
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
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

bool IsIterInRange(uint32_t iteration, const std::string &range) {
  if (range.empty()) {
    return false;
  }
  const std::string dash = "-";
  std::size_t range_idx = range.find(dash);
  // no dash in range, compare the value directly
  if (range_idx == std::string::npos) {
    size_t range_d = 0;
    if (!CheckStoul(&range_d, range)) {
      MS_LOG(INFO) << "Failed to convert the single step range: " << range
                   << " into an integer, so the iteration: " << iteration << " is regarded as not in dump range.";
      return false;
    }
    return iteration == range_d;
  }
  // make sure there is only one dash in range
  if (range.find(dash, range_idx + 1) != std::string::npos) {
    return false;
  }
  auto low_range_str = range.substr(0, range_idx);
  auto high_range_str = range.substr(range_idx + 1);
  if (low_range_str.empty() || high_range_str.empty()) {
    return false;
  }
  size_t low_range = 0;
  if (!CheckStoul(&low_range, low_range_str)) {
    MS_LOG(INFO) << "Failed to convert the low_range_str: " << low_range_str
                 << " into an integer, so the iteration: " << iteration << " is regarded as not in dump range.";
    return false;
  }
  size_t high_range = 0;
  if (!CheckStoul(&high_range, high_range_str)) {
    MS_LOG(INFO) << "Failed to convert the high_range_str: " << high_range_str
                 << " into an integer, so the iteration: " << iteration << " is regarded as not in dump range.";
    return false;
  }
  return (low_range <= iteration) && (iteration <= high_range);
}

bool DumpJsonParser::IsStatisticDump() const { return saved_data_ == kStatisticDump || IsFullDump(); }

bool DumpJsonParser::IsTensorDump() const { return saved_data_ == kTensorDump || IsFullDump(); }

bool DumpJsonParser::IsFullDump() const { return saved_data_ == kFullDump; }

bool DumpJsonParser::IsNpyFormat() const { return file_format_ == JsonFileFormat::FORMAT_NPY; }

bool DumpJsonParser::IsDumpIter(uint32_t iteration) const {
  // bool DumpJsonParser::IsDumpIter(uint32_t iteration) --> checks if iteration should be dumped or not.
  if (iteration_ == "all") {
    return true;
  }
  const std::string vertical_bar = "|";
  std::size_t start = 0;
  std::size_t end = iteration_.find(vertical_bar);
  while (end != std::string::npos) {
    std::string temp = iteration_.substr(start, end - start);
    auto found = IsIterInRange(iteration, temp);
    if (found) {
      return true;
    }
    start = end + 1;
    end = iteration_.find(vertical_bar, start);
  }
  std::string temp = iteration_.substr(start);
  return IsIterInRange(iteration, temp);
}

void DumpJsonParser::ParseInputOutput(const nlohmann::json &content) {
  CheckJsonUnsignedType(content, kInputOutput);
  input_output_ = content;
  const uint32_t max_inout_num = 2;
  if (input_output_ > max_inout_num) {
    MS_LOG(EXCEPTION) << "Dump Json Parse Failed. input_output should be 0, 1, 2";
  }
}

void DumpJsonParser::ParseKernels(const nlohmann::json &content) {
  CheckJsonArrayType(content, kKernels);
  if (dump_mode_ != static_cast<uint32_t>(DUMP_KERNEL)) {
    MS_LOG(INFO) << "Dump config field <" << kKernels << "> is not used as the dump mode is not 1.";
    return;
  }
  for (const auto &kernel : content) {
    bool ret;
    auto kernel_str = kernel.dump();
    kernel_str.erase(std::remove(kernel_str.begin(), kernel_str.end(), '\"'), kernel_str.end());
    MS_LOG(INFO) << "Need dump kernel:" << kernel_str;
    if (static_cast<int>(kernel_str.rfind('/')) == -1 && static_cast<int>(kernel_str.rfind("-op")) == -1) {
      ret = kernel_types_.try_emplace({kernel_str, 0}).second;
    } else {
      ret = kernels_.try_emplace({kernel_str, 0}).second;
      dump_layer_ += kernel_str + " ";
    }
    if (!ret) {
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

bool DumpJsonParser::ParseEnable(const nlohmann::json &content) const {
  if (!content.is_boolean()) {
    MS_LOG(EXCEPTION) << "Dump Json Parse Failed. 'enable' should be boolean type";
  }
  return content;
}

void DumpJsonParser::ParseOpDebugMode(const nlohmann::json &content) {
  CheckJsonUnsignedType(content, kOpDebugMode);
  op_debug_mode_ = content;
  const size_t max_mode = 3;
  if (op_debug_mode_ > max_mode) {
    MS_LOG(EXCEPTION) << "Dump Json Parse Failed. op_debug_mode should be 0, 1, 2, 3";
  }
}

void DumpJsonParser::ParseFileFormat(const nlohmann::json &content) {
  auto iter = content.find(kFileFormat);
  if (iter == content.end()) {
    file_format_ = JsonFileFormat::FORMAT_BIN;
  } else {
    CheckJsonStringType(*iter, kFileFormat);
    std::string file_format = *iter;
    const std::map<std::string, JsonFileFormat> str_to_fmt_enum = {{"bin", JsonFileFormat::FORMAT_BIN},
                                                                   {"npy", JsonFileFormat::FORMAT_NPY}};
    if (str_to_fmt_enum.find(file_format) == str_to_fmt_enum.end()) {
      MS_LOG(EXCEPTION) << "Dump Json Parse Failed. 'file_format' should be either 'npy' or 'bin', but got: "
                        << file_format;
    }
    file_format_ = str_to_fmt_enum.at(file_format);
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

/*
 * Feature group: Dump.
 * Target device group: Ascend, GPU and CPU.
 * Runtime category: Old runtime, MindRT.
 * Description: Check if the given op needs to be dumped based the configuration option.
 */
bool DumpJsonParser::NeedDump(const std::string &op_full_name) const {
  bool need_dump = false;

  switch (dump_mode_) {
    case DUMP_ALL:
      need_dump = true;
      break;
    case DUMP_KERNEL:
      if (kernels_.find(op_full_name) != kernels_.end()) {
        need_dump = true;
        break;
      }
      for (const auto &iter : kernel_types_) {
        int start_index = static_cast<int>(op_full_name.rfind('/')) + 1;
        int end_index = static_cast<int>(op_full_name.rfind('-'));
        if (end_index == -1) {
          end_index = static_cast<int>(op_full_name.length());
        }
        std::string op_name = op_full_name.substr(start_index, end_index - start_index);
        transform(op_name.begin(), op_name.end(), op_name.begin(), ::tolower);
        std::string kernel_type(iter.first);
        transform(kernel_type.begin(), kernel_type.end(), kernel_type.begin(), ::tolower);
        if (op_name.find(kernel_type) != std::string::npos) {
          need_dump = true;
          break;
        }
      }
      break;
    case DUMP_KERNELS_WITH_FLAG:
      if (std::find(cell_dump_kernels_.begin(), cell_dump_kernels_.end(), op_full_name) != cell_dump_kernels_.end()) {
        need_dump = true;
      }
      break;
    default:
      break;
  }
  return need_dump;
}

/*
 * Feature group: Dump.
 * Target device group: Ascend, GPU and CPU.
 * Runtime category: Old runtime, MindRT.
 * Description: Increment the count of dumping for given kernel.
 */
void DumpJsonParser::MatchKernel(const std::string &kernel_name) {
  auto iter = kernels_.find(kernel_name);
  if (iter == kernels_.end()) {
    return;
  }
  iter->second = iter->second + 1;
  MS_LOG(INFO) << "Match dump kernel:" << iter->first << " match times:" << iter->second;
}

void DumpJsonParser::PrintUnusedKernel() {
  if ((!e2e_dump_enabled_ && !async_dump_enabled_) || dump_mode_ != static_cast<uint32_t>(DUMP_KERNEL)) {
    return;
  }
  for (const auto &iter : kernels_) {
    if (iter.second == 0) {
      MS_LOG(WARNING) << "[DataDump] Unused Kernel in json:" << iter.first;
    }
  }
}

/*
 * Feature group: Online debugger.
 * Target device group: Ascend.
 * Runtime category: Old runtime, MindRT.
 * Description: Generate the directory path where overflow bin file locates.
 */
std::string DumpJsonParser::GetOpOverflowBinPath(uint32_t graph_id) const {
  std::string bin_path;
  bin_path.append(path_);
  bin_path.append("/");
  bin_path.append("rank_");

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

/*
 * Feature group: Dump.
 * Target device group: Ascend.
 * Runtime category: Old runtime, MindRT.
 * Description: Obtain the cell dump flag of each operators in the given kernel graph.
 */
void DumpJsonParser::GetCellDumpFlag(const session::KernelGraph &kernel_graph) {
  if (dump_mode_ != static_cast<uint32_t>(DUMP_KERNELS_WITH_FLAG)) {
    return;
  }
  for (const auto &kernel : kernel_graph.execution_order()) {
    MS_EXCEPTION_IF_NULL(kernel);
    auto dump_flag = common::AnfAlgo::GetDumpFlag(kernel);
    if (dump_flag.has_value() && dump_flag.value().compare("true") == 0) {
      MS_LOG(DEBUG) << "Dump flag is true for " << GetKernelNodeName(kernel);
      cell_dump_kernels_.push_back(GetKernelNodeName(kernel));
    }
  }
}

void DumpJsonParser::UpdateNeedDumpKernels(const session::KernelGraph &kernel_graph) {
  if (!async_dump_enabled_) {
    return;
  }

  MS_LOG(INFO) << "Get async kernel dump flag";
  GetCellDumpFlag(kernel_graph);

  MS_LOG(INFO) << "Update async dump kernel list for hccl";
  std::map<std::string, uint32_t> update_kernels;
  for (const auto &kernel : kernel_graph.execution_order()) {
    MS_EXCEPTION_IF_NULL(kernel);
    if (AnfAlgo::GetKernelType(kernel) == HCCL_KERNEL &&
        DumpJsonParser::GetInstance().NeedDump(GetKernelNodeName(kernel))) {
      auto input_size = common::AnfAlgo::GetInputTensorNum(kernel);
      for (size_t i = 0; i < input_size; ++i) {
        auto input_with_index = common::AnfAlgo::GetPrevNodeOutput(kernel, i);
        auto input = input_with_index.first;
        MS_EXCEPTION_IF_NULL(input);
        if (input->isa<CNode>()) {
          MS_LOG(INFO) << "[AsyncDump] Match Hccl Node:" << GetKernelNodeName(kernel)
                       << " Input:" << GetKernelNodeName(input);
          update_kernels.try_emplace(GetKernelNodeName(input), 0);
          cell_dump_kernels_.push_back(GetKernelNodeName(input));
        }
      }
    }
  }
  kernels_.insert(update_kernels.begin(), update_kernels.end());
}
}  // namespace mindspore
