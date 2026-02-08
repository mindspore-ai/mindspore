/**
 * Copyright 2020-2025 Huawei Technologies Co., Ltd
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
#include "tools/data_dump/dump_json_parser.h"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include "include/backend/common/kernel_graph/anf_runtime_algorithm.h"
#include "include/utils/anfalgo.h"
#include "include/utils/comm_manager.h"
#include "include/utils/common.h"
#include "mindspore/ccsrc/utils/ir_dump/anf_dump_utils.h"
#include "primitive/ascend_op_name.h"
#include "tools/data_dump/dump_control.h"
#include "tools/data_dump/npy_header.h"
#include "tools/data_dump/utils.h"
#include "utils/convert_utils_base.h"
#include "utils/file_utils.h"
#include "utils/log_adapter.h"
#include "utils/ms_context.h"
#include "utils/ms_utils.h"

namespace {
constexpr auto kCommonDumpSettings = "common_dump_settings";
constexpr auto kE2eDumpSettings = "e2e_dump_settings";
constexpr auto kDumpMode = "dump_mode";
constexpr auto kPath = "path";
constexpr auto kNetName = "net_name";
constexpr auto kSavedData = "saved_data";
constexpr auto kIteration = "iteration";
constexpr auto kInitialIteration = "initial_iteration";
constexpr auto kInputOutput = "input_output";
constexpr auto kKernels = "kernels";
constexpr auto kSupportDevice = "support_device";
constexpr auto kOverflowNumber = "overflow_number";
constexpr auto kEnable = "enable";
constexpr auto kOpDebugMode = "op_debug_mode";
constexpr auto kTransFlag = "trans_flag";
constexpr auto kSaveArgs = "save_kernel_args";
constexpr auto kSampleMode = "sample_mode";
constexpr auto kSampleNum = "sample_num";
constexpr auto kStatCalcMode = "stat_calc_mode";
constexpr auto kHost = "host";
constexpr auto kDevice = "device";
constexpr auto kStatisticDump = "statistic";
constexpr auto kDeviceStatPrecisionMode = "device_stat_precision_mode";
constexpr auto kHighPrecision = "high";
constexpr auto kLowPrecision = "low";
constexpr auto kTensorDump = "tensor";
constexpr auto kFullDump = "full";
constexpr auto kFileFormat = "file_format";
constexpr auto kStatisticCategory = "statistic_category";
constexpr auto kDumpInputAndOutput = 0;
constexpr auto kDumpInputOnly = 1;
constexpr auto kDumpOutputOnly = 2;
constexpr auto kMindsporeDumpConfig = "MINDSPORE_DUMP_CONFIG";
constexpr auto kBracketsOffset = 1;
constexpr auto kRegexPrefixLength = 11;
const std::vector<std::string> kDefaultStatisticCategory = {"max", "min", "l2norm"};
const std::set<std::string> kDeviceStatisticCategory = {"max", "min", "avg", "l2norm"};
const std::set<std::string> kHostStatisticCategory = {"max",
                                                      "min",
                                                      "avg",
                                                      "count",
                                                      "negative zero count",
                                                      "positive zero count",
                                                      "nan count",
                                                      "negative inf count",
                                                      "positive inf count",
                                                      "zero count",
                                                      "md5",
                                                      "l2norm",
                                                      "hash",
                                                      "hash:sha1",
                                                      "hash:md5"};
constexpr auto kDeviceStatisticsategory = "['max', 'min', 'avg', 'l2norm']";
constexpr auto kSupportedStatisticsategory =
  "['max', 'min', 'avg', 'count', 'negative zero count', 'positive zero count', 'nan count', 'negative inf count', "
  "'positive inf count', 'zero count', 'md5', 'l2norm', 'hash', 'hash:sha1', 'hash:md5']";
}  // namespace

namespace mindspore {

void DumpJsonParser::SetInitialIteration(uint32_t initial_iteration) {
  Parse();
  dump_user_step_flag_ = true;
  cur_dump_iter_ = initial_iteration;
  cur_user_dump_iter_ = initial_iteration;
  initial_dump_iter_ = initial_iteration;
  MS_VLOG(VL_DUMP) << "Set initial iteration to " << initial_iteration;
}

auto DumpJsonParser::CheckJsonKeyExist(const nlohmann::json &content, const std::string &key) {
  nlohmann::json::const_iterator iter = content.find(key);
  if (iter == content.end()) {
    MS_LOG(EXCEPTION) << "Check dump json failed, " << key << " not found";
  }
  return iter;
}

bool DumpJsonParser::CheckSelectableKeyExist(const nlohmann::json &content, const std::string &key) {
  nlohmann::json::const_iterator iter = content.find(key);
  if (iter == content.end()) {
    return false;
  }
  return true;
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
  MS_VLOG(VL_DUMP) << "Dump config path is " << config_path;
  return true;
}

void DumpJsonParser::CheckE2eSetting() {
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  if (e2e_dump_enabled()) {
    CheckStatCalcModeVaild();
  }
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
  if (!IsDumpEnabled()) {
    return;
  }

  auto dump_config_file = Common::GetConfigFile(kMindsporeDumpConfig);
  if (!dump_config_file.has_value()) {
    MS_LOG(EXCEPTION) << "Get dump config file failed";
  }

  std::ifstream json_file(dump_config_file.value());
  if (!json_file.is_open()) {
    MS_LOG(EXCEPTION) << "Dump file:" << dump_config_file.value() << " open failed. Errno:" << errno;
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
  MS_VLOG(VL_DUMP) << "Dump json:" << cfg;

  ParseE2eDumpSetting(j);
  ParseCommonDumpSetting(j);
  CheckE2eSetting();
  JudgeDumpEnabled();
  CheckStatCalcModeVaild();
  ParseStatisticCategory(j);
  already_parsed_ = true;
}

void DumpJsonParser::ParseStatisticCategory(const nlohmann::json &content) {
  if (!IsStatisticDump()) {
    return;
  }
  auto common_dump_settings = CheckJsonKeyExist(content, kCommonDumpSettings);
  auto set_statistic_category = CheckSelectableKeyExist(*common_dump_settings, kStatisticCategory);
  if (set_statistic_category) {
    auto user_statistics = CheckJsonKeyExist(*common_dump_settings, kStatisticCategory);
    CheckJsonArrayType(*user_statistics, kStatisticCategory);
    std::string unsupported_items = "";
    if (IsDeviceCalcStats()) {
      HandleDeviceStatisticCategory(user_statistics, &unsupported_items);
    } else {
      HandleHostStatisticCategory(user_statistics, &unsupported_items);
    }
    if (!unsupported_items.empty()) {
      MS_LOG(EXCEPTION) << "The following statistic_category is invalid:" << unsupported_items
                        << "the valid statistic_category are as follows:" << kSupportedStatisticsategory;
    }
  } else {
    statistic_category_ = kDefaultStatisticCategory;
    MS_VLOG(VL_DUMP) << "Statistic category is not set, use the default items as follows:";
    for (auto &itm : kDefaultStatisticCategory) {
      MS_VLOG(VL_DUMP) << itm;
    }
  }
  CsvHeaderUtil::GetInstance().SetStatCsvHeader(statistic_category_);
}

void DumpJsonParser::HandleHostStatisticCategory(const nlohmann::json::const_iterator user_statistics,
                                                 std::string *unsupported_items) {
  std::unordered_set<std::string> categorySet;
  for (const auto &statistic_item_json : *user_statistics) {
    std::string statistic_item = statistic_item_json;
    auto rt_find = kHostStatisticCategory.find(statistic_item);
    if (rt_find == kHostStatisticCategory.end()) {
      *unsupported_items += statistic_item + ", ";
    } else {
      if (statistic_item == "hash:md5") {
        statistic_item = "md5";
      } else if (statistic_item == "hash" || statistic_item == "hash:sha1") {
        statistic_item = "sha1";
      }
      if (categorySet.insert(statistic_item).second) {
        statistic_category_.push_back(statistic_item);
      }
      MS_VLOG(VL_DUMP) << "The item: " << statistic_item
                       << " is a valid statistic category, it will be computed on host.";
    }
  }
}

void DumpJsonParser::HandleDeviceStatisticCategory(const nlohmann::json::const_iterator user_statistics,
                                                   std::string *unsupported_items) {
  std::string device_unsupported_items = "";
  for (const auto &statistic_item_json : *user_statistics) {
    std::string statistic_item = statistic_item_json;
    auto rt_find = kDeviceStatisticCategory.find(statistic_item);
    if (rt_find == kDeviceStatisticCategory.end()) {
      auto in_host_category = kHostStatisticCategory.find(statistic_item);
      if (in_host_category == kHostStatisticCategory.end()) {
        *unsupported_items += statistic_item + ", ";
      } else {
        device_unsupported_items += statistic_item + ", ";
      }
    } else {
      statistic_category_.push_back(statistic_item);
      MS_VLOG(VL_DUMP) << "The item: " << statistic_item
                       << " is a valid statistic category, it will be computed on device.";
    }
  }
  if (!device_unsupported_items.empty()) {
    MS_LOG(WARNING) << "The following statistic_category only support to be compute on host:"
                    << device_unsupported_items
                    << "the valid statistic_category on device are as follows:" << kDeviceStatisticsategory;
  }
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

uint32_t DumpJsonParser::cur_dump_iter() const { return dump_user_step_flag_ ? cur_user_dump_iter_ : cur_dump_iter_; }

void DumpJsonParser::UpdateDumpIter(int cur_step_count) {
  cur_dump_iter_ = static_cast<uint32_t>(cur_step_count) + initial_dump_iter_;
}

void DumpJsonParser::UpdateUserDumpStep(const uint32_t step) {
  MS_VLOG(VL_DUMP) << "Do dump step update:" << step << "before update cur_user_dump_iter_: " << cur_user_dump_iter_;
  if (!dump_user_step_flag_) {
    MS_LOG(WARNING) << "Costomized step function has not enabled, step update does not work!";
  }
  cur_user_dump_iter_ += step;
}

bool DumpJsonParser::GetIterDumpFlag() const { return e2e_dump_enabled_ && IsDumpIter(cur_dump_iter()); }

bool DumpJsonParser::DumpEnabledForIter() const {
  const auto &dump_control = DumpControl::GetInstance();
  if (dump_control.dynamic_switch()) {
    MS_VLOG(VL_DUMP) << (dump_control.dump_switch() ? "Dynamic switch is on, dump for every iteration."
                                                    : "Dump is end.");
    return dump_control.dump_switch() && (e2e_dump_enabled_ || async_dump_enabled_);
  }
  return (e2e_dump_enabled_ || async_dump_enabled_) && IsDumpIter(cur_dump_iter());
}

/*
 * Feature group: Dump.
 * Target device group: Ascend, GPU and CPU.
 * Runtime category: Old runtime, MindRT.
 * Description: Dump data in the given address into npy file.
 */
bool DumpJsonParser::DumpToFile(const std::string &filename, const void *data, size_t len, const ShapeVector &shape,
                                TypeId type) {
  if (filename.empty() && (data == nullptr || len == 0)) {
    MS_LOG(ERROR) << "Filename and data are empty or null.";
    return false;
  } else if (filename.empty()) {
    MS_LOG(ERROR) << "Filename is empty.";
    return false;
  } else if (data == nullptr || len == 0) {
    MS_LOG(WARNING) << "Data is empty or null for file: " << filename;
    return false;
  }
  std::string npy_header = GenerateNpyHeader(shape, type);
  if (npy_header.empty()) {
    MS_LOG(WARNING) << "Failed to generate npy_header for file: " << filename;
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
    std::lock_guard<std::mutex> guard(lock_);
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
    final_file_path = prefix_path.value() + "/" + mapped_name_str;
  }
  auto file_path = Common::CreatePrefixPath(final_file_path);
  if (!file_path.has_value()) {
    MS_LOG(ERROR) << "CreatePrefixPath failed.";
    return false;
  }
  const std::string file_path_str = file_path.value();
  MS_VLOG(VL_DUMP) << "Dump path is " << file_path_str;
  ChangeFileMode(file_path_str, S_IWUSR);

  MSLogTime msTime;
  msTime.Start();
  std::ofstream fd(file_path_str, std::ios::out | std::ios::trunc | std::ios::binary);
  if (!fd.is_open()) {
    MS_LOG(EXCEPTION) << "Open file " << file_path_str << " failed." << ErrnoToString(errno);
  }
  fd << npy_header;
  (void)fd.write(reinterpret_cast<const char *>(data), SizeToLong(len));
  if (fd.bad()) {
    fd.close();
    MS_LOG(EXCEPTION)
      << "Write mem to file " << file_path_str
      << " failed. This error may be caused by insufficient disk space. Please check the available disk space.";
  }
  fd.close();
  msTime.End();
  MS_LOG(DEBUG) << "Dump file costs time : " << msTime.GetRunTimeUS() << " microseconds.";

  ChangeFileMode(file_path_str, S_IRUSR);
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
    sample_mode_ = 0;
    sample_num_ = 100;
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
  } else {
    if (CheckSelectableKeyExist(*common_dump_settings, kOpDebugMode)) {
      op_debug_mode = CheckJsonKeyExist(*common_dump_settings, kOpDebugMode);
    }
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
    CheckOverflowSetting();
    ParseFileFormat(
      *common_dump_settings);  // Pass in the whole json string to parse because file_format field is optional.
  } else {
    if (CheckSelectableKeyExist(*common_dump_settings, kOpDebugMode)) {
      ParseOpDebugMode(*op_debug_mode);
      CheckOverflowSetting();
    }
  }
  ParseOverflowNumber(*common_dump_settings);    // The overflow number field is optional.
  ParseSavedData(*common_dump_settings);         // saved data optional
  ParseInitialIteration(*common_dump_settings);  // saved initial iter
}

void DumpJsonParser::ParseE2eSyncDumpEnable(const nlohmann::json &content) {
  auto enable_value_iter = content.find(kEnable);
  e2e_sync_dump_enabled_ = false;
  if (enable_value_iter != content.end()) {
    e2e_sync_dump_enabled_ = ParseEnable(*enable_value_iter);
  }
}

void DumpJsonParser::ParseE2eDumpSetting(const nlohmann::json &content) {
  auto e2e_dump_setting = content.find(kE2eDumpSettings);
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  if (e2e_dump_setting == content.end()) {
    constexpr int kMaxWarnings = 3;
    constexpr auto kWarningInterval = std::chrono::milliseconds(1000);
    for (int i = 1; i <= kMaxWarnings; ++i) {
      MS_LOG(WARNING) << "[Dump Alert " << i << "/" << kMaxWarnings << "]: "
                      << "For 'Dump', in the scenario where 'backend' is 'ms_backend', please configure "
                         "'e2e_dump_setting', otherwise there will be no data.";
      std::this_thread::sleep_for(kWarningInterval);
    }
    return;
  }

  e2e_dump_enabled_ = true;
  ParseE2eSyncDumpEnable(*e2e_dump_setting);
  bool set_trans_flag = CheckSelectableKeyExist(*e2e_dump_setting, kTransFlag);
  if (set_trans_flag) {
    auto trans_flag = CheckJsonKeyExist(*e2e_dump_setting, kTransFlag);
    trans_flag_ = ParseEnable(*trans_flag);
  } else {
    trans_flag_ = true;
  }
  if (CheckSelectableKeyExist(*e2e_dump_setting, kSaveArgs)) {
    auto save_args_flag = CheckJsonKeyExist(*e2e_dump_setting, kSaveArgs);
    save_args_flag_ = ParseEnable(*save_args_flag);
  }

  ParseStatCalcMode(*e2e_dump_setting);
  ParseDeviceStatPrecisionMode(*e2e_dump_setting);
  if (CheckSelectableKeyExist(*e2e_dump_setting, kSampleMode)) {
    auto sample_mode = CheckJsonKeyExist(*e2e_dump_setting, kSampleMode);
    ParseSampleMode(*sample_mode);
    if (CheckSelectableKeyExist(*e2e_dump_setting, kSampleNum) &&
        sample_mode_ == static_cast<uint32_t>(DUMP_HEAD_AND_TAIL)) {
      auto sample_num = CheckJsonKeyExist(*e2e_dump_setting, kSampleNum);
      ParseSampleNum(*sample_num);
    }
  }
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
    if (context->get_param<std::string>(MS_CTX_DEVICE_TARGET) != kAscendDevice) {
      MS_LOG(EXCEPTION) << "Set dump is only supported in Ascend async dump. Please set dump_mode to 0 or 1.";
    }
  }
}

void DumpJsonParser::ParseOverflowNumber(const nlohmann::json &content) {
  auto json_iter = content.find(kOverflowNumber);
  if (op_debug_mode_ == static_cast<uint32_t>(DUMP_BOTH_OVERFLOW)) {
    overflow_number_ = 0;
    if (json_iter != content.end()) {
      CheckJsonUnsignedType(*json_iter, kOverflowNumber);
      overflow_number_ = *json_iter;
      const uint32_t min_input_num = 0;
      if (overflow_number_ < min_input_num) {
        MS_LOG(EXCEPTION) << "Dump config parse failed, overflow_number should not be less than 0, but got "
                          << overflow_number_;
      }
    }
  } else {
    if (json_iter != content.end()) {
      MS_LOG(EXCEPTION) << "overflow_number only need to be set when op_debug_mode is 3.";
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
  if (e2e_dump_enabled_ && op_debug_mode_ == static_cast<uint32_t>(DUMP_LITE_EXCEPTION) && saved_data_ != kTensorDump) {
    MS_LOG(WARNING) << "E2e exception dump only support save tensor, saved_data is set to tensor";
    saved_data_ = kTensorDump;
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
    if (!IsNpyFormat() && !e2e_dump_enabled_) {
      MS_LOG(EXCEPTION) << "Dump Json parse failed, storing statistic dump is only supported on Ascend when "
                           "file_format is set to 'npy'.";
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

void DumpJsonParser::ParseInitialIteration(const nlohmann::json &content) {
  auto json_iter = content.find(kInitialIteration);
  if (json_iter != content.end()) {
    CheckJsonUnsignedType(*json_iter, kInitialIteration);
    initial_dump_iter_ = *json_iter;
    cur_dump_iter_ = initial_dump_iter_;
    cur_user_dump_iter_ = initial_dump_iter_;
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
      MS_VLOG(VL_DUMP) << "Failed to convert the single step range: " << range
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
    MS_VLOG(VL_DUMP) << "Failed to convert the low_range_str: " << low_range_str
                     << " into an integer, so the iteration: " << iteration << " is regarded as not in dump range.";
    return false;
  }
  size_t high_range = 0;
  if (!CheckStoul(&high_range, high_range_str)) {
    MS_VLOG(VL_DUMP) << "Failed to convert the high_range_str: " << high_range_str
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
    MS_VLOG(VL_DUMP) << "Dump config field <" << kKernels << "> is not used as the dump mode is not 1.";
    return;
  }
  kernels_json_ = content;
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  for (const auto &kernel : content) {
    bool ret;
    auto kernel_str = kernel.dump();
    MS_VLOG(VL_DUMP) << "Need dump kernel:" << kernel_str;
    kernel_str.erase(std::remove(kernel_str.begin(), kernel_str.end(), '\"'), kernel_str.end());
    kernel_str.erase(std::remove(kernel_str.begin(), kernel_str.end(), ' '), kernel_str.end());
    if (kernel_str == "") {
      continue;
    }
    if (static_cast<int>(kernel_str.find("name-regex(")) == 0 &&
        static_cast<int>(kernel_str.rfind(")")) == static_cast<int>(kernel_str.length()) - kBracketsOffset) {
      std::string kernel_reg_exp = kernel_str.substr(
        kRegexPrefixLength, static_cast<int>(kernel_str.length()) - kRegexPrefixLength - kBracketsOffset);
      ret = kernel_regs_.try_emplace(kernel_str, std::regex(kernel_reg_exp)).second;
    } else {
      if (static_cast<int>(kernel_str.rfind('/')) == -1 && static_cast<int>(kernel_str.rfind("-op")) == -1) {
        ret = kernel_types_.try_emplace({kernel_str, 0}).second;
      } else {
        ret = kernels_.try_emplace({kernel_str, 0}).second;
      }
    }
    kernel_strings_.try_emplace({kernel_str, 0});
    if (!ret) {
      MS_LOG(WARNING) << "Duplicate dump kernel name:" << kernel_str;
    }
  }
  if (kernel_strings_.empty()) {
    kernel_types_.try_emplace({"", 0});
    kernel_strings_.try_emplace({"", 0});
  }
}

void DumpJsonParser::ParseStatCalcMode(const nlohmann::json &content) {
  auto iter = content.find(kStatCalcMode);
  stat_calc_mode_ = kHost;
  if (iter == content.end()) {
    MS_VLOG(VL_DUMP) << "'stat_calc_mode' is not set, default is " << stat_calc_mode_;
    return;
  }
  CheckJsonStringType(*iter, kStatCalcMode);
  std::string calc_mode = *iter;
  if (calc_mode != kHost && calc_mode != kDevice) {
    MS_LOG(EXCEPTION) << "Dump Json parse failed, 'stat_calc_mode' only supports 'host' or 'device', but got: "
                      << calc_mode << ". Please set 'stat_cal_mode' to 'host' or 'device'";
  }
  stat_calc_mode_ = calc_mode;
}

void DumpJsonParser::ParseDeviceStatPrecisionMode(const nlohmann::json &content) {
  auto iter = content.find(kDeviceStatPrecisionMode);
  device_stat_precision_mode_ = kHighPrecision;
  if (iter == content.end()) {
    MS_VLOG(VL_DUMP) << "'device_stat_precision_mode' is not set, default is " << device_stat_precision_mode_;
    return;
  }
  CheckJsonStringType(*iter, kDeviceStatPrecisionMode);
  std::string presicion_mode = *iter;
  if (presicion_mode != kHighPrecision && presicion_mode != kLowPrecision) {
    MS_LOG(EXCEPTION) << "Dump Json parse failed, 'device_stat_precision_mode' only supports 'high' or 'low', but got: "
                      << presicion_mode << ". Please set 'stat_cal_mode' to 'high' or 'low'";
  }
  device_stat_precision_mode_ = presicion_mode;
}

void DumpJsonParser::CheckStatCalcModeVaild() {
  if (IsTensorDump() && stat_calc_mode_ == kDevice) {
    MS_LOG(WARNING) << "When 'saved_data' is 'tensor' or 'full', the device cannot be used to calculate statistics and "
                       "the 'stat_calc_mode' is forced to 'host'.";
    stat_calc_mode_ = kHost;
  }
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  auto device_target = context->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  if (device_target != kAscendDevice && stat_calc_mode_ == kDevice) {
    MS_LOG(WARNING)
      << "The 'device' option of 'stat_calc_mode' currently only supports the ascend platform. The current platform is "
      << device_target << ", and the 'stat_calc_mode' option is forcibly set to 'host'.";
    stat_calc_mode_ = kHost;
  }
  MS_VLOG(VL_DUMP) << "stat_calc_mode is set to " << stat_calc_mode_;
}

bool DumpJsonParser::IsDeviceCalcStats() const { return stat_calc_mode_ == kDevice; }

void DumpJsonParser::ParseSupportDevice(const nlohmann::json &content) {
  CheckJsonArrayType(content, kSupportDevice);
  for (const auto &device : content) {
    uint32_t device_id = device;
    MS_VLOG(VL_DUMP) << "Dump support device:" << device_id;
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

void DumpJsonParser::ParseSampleMode(const nlohmann::json &content) {
  CheckJsonUnsignedType(content, kSampleMode);
  sample_mode_ = content;
  const uint32_t max_inout_num = 1;
  if (sample_mode_ > max_inout_num) {
    MS_LOG(EXCEPTION) << "Dump Json Parse Failed. sample_mode should be 0, 1";
  }
}

void DumpJsonParser::ParseSampleNum(const nlohmann::json &content) {
  CheckJsonUnsignedType(content, kSampleMode);
  sample_num_ = content;
  const uint32_t min_inout_num = 1;
  if (sample_num_ < min_inout_num) {
    MS_LOG(EXCEPTION) << "Dump Json Parse Failed. sample_num should be greater than 0";
  }
}

void DumpJsonParser::ParseOpDebugMode(const nlohmann::json &content) {
  CheckJsonUnsignedType(content, kOpDebugMode);
  op_debug_mode_ = content;
  switch (op_debug_mode_) {
    case static_cast<uint32_t>(DUMP_WHOLE):
      break;
    case static_cast<uint32_t>(DUMP_AICORE_OVERFLOW):
    case static_cast<uint32_t>(DUMP_ATOMIC_OVERFLOW): {
      MS_VLOG(VL_DUMP)
        << "Op_debug_mode should be 0, 3, 4. When set to 1 or 2, it would be reset to 3 and overflow dump "
           "is enabled.";
      op_debug_mode_ = static_cast<uint32_t>(DUMP_BOTH_OVERFLOW);
    } break;
    case static_cast<uint32_t>(DUMP_BOTH_OVERFLOW): {
      break;
    }
    case static_cast<uint32_t>(DUMP_LITE_EXCEPTION): {
      auto context = MsContext::GetInstance();
      MS_EXCEPTION_IF_NULL(context);
      auto device_target = context->get_param<std::string>(MS_CTX_DEVICE_TARGET);
      if (device_target == "CPU" || device_target == "GPU") {
        MS_LOG(WARNING) << "Abnormal dump is not supported on " << device_target
                        << " backend, and none operator data would be saved when abnormal dump is enabled. ";
      }
      if (e2e_dump_enabled_ && iteration_ != "all") {
        MS_LOG(WARNING) << "For e2e exception dump, it is not support to specify iteration, set iteration to all.";
        iteration_ = "all";
      }
      if (e2e_dump_enabled_ && sample_mode_ != 0) {
        MS_LOG(WARNING) << "For e2e exception dump, it is not support to sample dump, set sample_mode to 0, the "
                           "whole tensor would be saved when exception occur.";
        sample_mode_ = 0;
      }
      break;
    }
    default:
      MS_LOG(EXCEPTION) << "Dump Json Parse Failed. op_debug_mode should be 0, 3, 4";
  }
  if (op_debug_mode_ != static_cast<uint32_t>(DUMP_WHOLE) && dump_mode_ != static_cast<uint32_t>(DUMP_ALL)) {
    MS_LOG(WARNING) << "Overflow dump or exception dump do not support specify kernels, the dump_mode is set to 0";
    dump_mode_ = static_cast<uint32_t>(DUMP_ALL);
  }
}

void DumpJsonParser::CheckOverflowSetting() {
  if (op_debug_mode_ == static_cast<uint32_t>(DUMP_BOTH_OVERFLOW)) {
    if (input_output_ != 0) {
      input_output_ = 0;
      MS_LOG(WARNING) << "When the `op_debug` mode is set to 3, `input_output` should be set to 0.";
    }
    if (stat_calc_mode_ == kDevice && !e2e_sync_dump_enabled_) {
      stat_calc_mode_ = kHost;
      MS_LOG(WARNING) << "When op_debug_mode is set to 3 and e2e_enable is set to false,"
                         "stat_calc_mode has been be set to 'host'.";
    }
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
  MS_VLOG(VL_DUMP) << cur_config;
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
      MS_VLOG(VL_DUMP) << "Disable async dump";
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
bool DumpJsonParser::NeedDump(const std::string &op_full_name) {
  bool need_dump = false;

  switch (dump_mode_) {
    case DUMP_ALL:
      need_dump = true;
      break;
    case DUMP_KERNEL:
      for (const auto &iter : kernel_regs_) {
        if (regex_match(op_full_name, iter.second)) {
          need_dump = true;
          MatchKernel(iter.first);
          break;
        }
      }
      if (need_dump) {
        break;
      }
      if (kernels_.find(op_full_name) != kernels_.end()) {
        need_dump = true;
        MatchKernel(op_full_name);
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
        std::string kernel_type_in_lower_case = kernel_type;
        transform(kernel_type_in_lower_case.begin(), kernel_type_in_lower_case.end(), kernel_type_in_lower_case.begin(),
                  ::tolower);
        if (op_name.find(kernel_type_in_lower_case) != std::string::npos) {
          need_dump = true;
          MatchKernel(kernel_type);
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
  auto iter = kernel_strings_.find(kernel_name);
  if (iter == kernel_strings_.end()) {
    return;
  }
  iter->second = iter->second + 1;
  MS_VLOG(VL_DUMP) << "Match dump kernel:" << iter->first << " match times:" << iter->second;
}

void DumpJsonParser::PrintUnusedKernel() {
  if ((!e2e_dump_enabled_ && !async_dump_enabled_) || dump_mode_ != static_cast<uint32_t>(DUMP_KERNEL)) {
    return;
  }
  for (const auto &iter : kernel_strings_) {
    if (iter.second == 0) {
      MS_LOG(WARNING) << "[DataDump] Unused Kernel in json: " << iter.first;
    }
  }
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
  MS_VLOG(VL_DUMP) << "Get kernel dump flag";
  GetCellDumpFlag(kernel_graph);
}

bool DumpJsonParser::IsDeviceStatHighPrecisionMode() const { return device_stat_precision_mode_ == kHighPrecision; }

void DumpJsonParserParse() { DumpJsonParser::GetInstance().Parse(); }
bool AsyncDumpEnabled() { return DumpJsonParser::GetInstance().async_dump_enabled(); }
bool E2eDumpEnabled() { return DumpJsonParser::GetInstance().e2e_dump_enabled(); }
std::string DumpJsonParserPath() { return DumpJsonParser::GetInstance().path(); }
bool InputNeedDump() { return DumpJsonParser::GetInstance().InputNeedDump(); }
void CopyDumpJsonToDir(uint32_t rank_id) { return DumpJsonParser::GetInstance().CopyDumpJsonToDir(rank_id); }
void CopyMSCfgJsonToDir(uint32_t rank_id) { return DumpJsonParser::GetInstance().CopyMSCfgJsonToDir(rank_id); }
void DumpJsonParserFinalize() { return DumpJsonParser::GetInstance().Finalize(); }
void UpdateNeedDumpKernels(const session::KernelGraph &kernel_graph) {
  return DumpJsonParser::GetInstance().UpdateNeedDumpKernels(kernel_graph);
}
bool DumpJsonParserDumpToFile(const std::string &filename, const void *data, size_t len, const ShapeVector &shape,
                              TypeId type) {
  return DumpJsonParser::GetInstance().DumpToFile(filename, data, len, shape, type);
}
}  // namespace mindspore
