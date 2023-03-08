/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/ascend/kernel/tbe/tbe_kernel_compile.h"
#include <unistd.h>
#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include "plugin/device/ascend/kernel/tbe/tbe_json/tbe_json_creator.h"
#include "plugin/device/ascend/kernel/tbe/tbe_json/tbe_json_utils.h"
#include "plugin/device/ascend/kernel/tbe/tbe_json/single_tbe_json_creator.h"
#include "plugin/device/ascend/kernel/tbe/tbe_json/fusion_tbe_json_creator.h"
#include "kernel/common_utils.h"
#include "plugin/device/ascend/kernel/tbe/tbe_utils.h"
#include "plugin/device/ascend/kernel/tbe/tbe_kernel_mod.h"
#include "plugin/device/ascend/kernel/tbe/dynamic_tbe_kernel_mod.h"
#include "plugin/device/ascend/kernel/tbe/tbe_convert_utils.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "backend/common/session/kernel_build_client.h"
#include "common/util/error_manager/error_manager.h"
#include "include/common/debug/anf_ir_dump.h"
#include "frontend/operator/ops.h"
#include "utils/ms_context.h"
#include "utils/ms_utils.h"
#include "utils/trace_base.h"
#include "include/common/utils/utils.h"
#include "include/common/utils/json_operation_utils.h"

namespace mindspore {
namespace kernel {
namespace ascend {
using mindspore::kernel::tbe::TbeAdapter;
using mindspore::kernel::tbe::TbeUtils;
const int indent = 4;  // for dump json
const int kAscii_0 = 48;
const int kAscii_9 = 57;
const uint32_t kDEFAULT_PROCESS_NUM = 24;
constexpr auto kInitialize = "Initialize";
constexpr auto kPreCompile = "PreCompile";
constexpr auto kFinalize = "Finalize";
constexpr auto kCompile = "Compile";
constexpr auto kFusionCompile = "FusionOpCompile";
constexpr auto kTune = "Tune";
constexpr auto kOfflineTune = "offlineTune";
constexpr auto kCheckSupport = "CheckSupport";
constexpr auto kSelectFormat = "SelectFormat";
constexpr auto kFullySupported = "FULLY_SUPPORTED";
constexpr auto kNotSupported = "NOT_SUPPORTED";
constexpr auto kPartiallySupported = "PARTIALLY_SUPPORTED";
constexpr auto kUnSupportedReason = "The shape is not support now";
constexpr auto kLevel = "level";
constexpr auto kMessage = "message";
constexpr auto kErrorCode = "errCode";
constexpr auto kIndex = "index";
constexpr auto kStatus = "status";
constexpr auto kJobType = "job_type";
constexpr auto kJobId = "job_id";
constexpr auto kSourceId = "source_id";
constexpr auto kTuneMode = "tune_mode";
constexpr auto kTuneType = "tune_type";
constexpr auto kJobContent = "job_content";
constexpr auto kProcessInfo = "process_info";
constexpr auto kReturnValue = "return_value";
constexpr auto kFusionOpName = "fusion_op_name";
constexpr auto kResult = "result";
constexpr auto kOpList = "op_list";
constexpr auto kSuccess = "SUCCESS";
constexpr auto kRunning = "RUNNING";
constexpr auto kFailed = "FAILED";
constexpr auto kQuery = "Query";
constexpr auto kTrue = "True";
constexpr auto kGLOG_v = "GLOG_v";
constexpr auto kSocInfo = "SocInfo";
constexpr auto kTuneInfo = "TuneInfo";
constexpr auto kLicInfo = "LicInfo";
constexpr auto kTuneOpList = "tune_op_list";
constexpr auto kProcessNum = "process_num";
constexpr auto kLogLevel = "log_level";
constexpr auto kEnableEvent = "enable_event";
constexpr auto kTuneDumpPath = "tune_dump_path";
constexpr auto kTuneBankPath = "tune_bank_path";
constexpr auto kTbeImplPath = "tbe_impl_path";
constexpr auto kParaDebugPath = "para_debug_path";
constexpr auto kTbePrebuildRes = "kernel_meta/tbe_prebuild_res/";
constexpr auto kNotSupportFusionOp = "kernel_meta/not_support_fusion_op.cache";
constexpr auto kMS_BUILD_PROCESS_NUM = "MS_BUILD_PROCESS_NUM";
constexpr auto kMS_PARA_DEBUG_PATH = "PARA_DEBUG_PATH";
constexpr auto kTBE_IMPL_PATH = "TBE_IMPL_PATH";
constexpr auto kTUNE_OPS_NAME = "TUNE_OPS_NAME";
constexpr auto KSleepUSeconds = 3000;
constexpr auto KSleepInterval = 1000;
const uint64_t kUSecondInSecond = 1000000;

namespace {
inline bool Order(const nlohmann::json &json1, const nlohmann::json &json2) {
  return json1[kIndex].dump() < json2[kIndex].dump();
}

void ReportToErrorManager(const string &message) {
  nlohmann::json exception_message;
  if (!ParseJson(message, &exception_message)) {
    MS_LOG(EXCEPTION) << "Parse tbe exception message error.";
  }
  const auto &error_code = GetJsonValue<std::string>(exception_message, kErrorCode);
  std::map<std::string, std::string> arg_map;
  for (auto it = exception_message.begin(); it != exception_message.end(); ++it) {
    const std::string arg_key = it.key();
    if (it.key() == kErrorCode) {
      continue;
    }
    const auto &arg_value = GetJsonValue<std::string>(exception_message, arg_key);
    arg_map[arg_key] = arg_value;
  }
  const auto report_ret = ErrorManager::GetInstance().ReportErrMessage(error_code, arg_map);
  if (report_ret != 0) {
    MS_LOG(WARNING) << "Report error message failed, raw error message: " << message;
  }
}

void PrintInfo(const nlohmann::json &info, const std::string &job_name, const int job_id, int adjust_log_level) {
  auto level = GetJsonValue<int>(info, kLevel);
  level = level > adjust_log_level ? adjust_log_level : level;
  auto message = GetJsonValue<std::string>(info, kMessage);
  if (level == 0) {
    MS_LOG(DEBUG) << "Job id:" << job_id << ", name :" << job_name << ", message:" << message;
  } else if (level == static_cast<int>(MsLogLevel::kInfo)) {
    MS_LOG(INFO) << "Job id:" << job_id << ", name :" << job_name << ", message:" << message;
  } else if (level == static_cast<int>(MsLogLevel::kWarning)) {
    MS_LOG(WARNING) << "Job id:" << job_id << ", name :" << job_name << ", message:" << message;
  } else if (level == static_cast<int>(MsLogLevel::kError)) {
    MS_LOG(ERROR) << "Job id:" << job_id << ", name :" << job_name << ", message:" << message;
  } else if (level == static_cast<int>(MsLogLevel::kException)) {
    ReportToErrorManager(message);
  }
}

std::string FilterExceptionMessage(const std::vector<nlohmann::json> &all_logs) {
  std::ostringstream exception_buffer;
  // print all logs if exception log is empty
  std::ostringstream all_message_buffer;
  for (const auto &item : all_logs) {
    auto message = GetJsonValue<std::string>(item, kMessage);
    all_message_buffer << message;
    all_message_buffer << "\n";
    if (message.find("except_msg") != std::string::npos || message.find("except_tuple_msg") != std::string::npos ||
        message.find("Error message") != std::string::npos) {
      exception_buffer << message;
      exception_buffer << "\n";
    }
  }
  auto res = exception_buffer.str().empty() ? all_message_buffer.str() : exception_buffer.str();
  return res;
}

bool IsDigit(const std::string &str) {
  if (str.empty()) {
    return false;
  }
  size_t i = 0;
  while (i < str.size()) {
    if (static_cast<int>(str[i]) < kAscii_0 || static_cast<int>(str[i]) > kAscii_9) {
      return false;
    }
    i++;
  }
  return true;
}

uint32_t GetProcessNum() {
  uint32_t process_num = kDEFAULT_PROCESS_NUM;
  auto env_process_num = common::GetEnv(kMS_BUILD_PROCESS_NUM);
  if (!env_process_num.empty()) {
    if (!IsDigit(env_process_num)) {
      MS_LOG(EXCEPTION) << "Invalid environment variable '" << kMS_BUILD_PROCESS_NUM
                        << "', it should be a digit, but got: " << env_process_num;
    }
    process_num = UlongToUint(std::stoul(env_process_num));
    if (process_num < IntToUint(1) || process_num > kDEFAULT_PROCESS_NUM) {
      MS_LOG(EXCEPTION) << "Invalid environment variable '" << kMS_BUILD_PROCESS_NUM
                        << "', the value should be in [1, 24], but got: " << process_num;
    }
  }
  return process_num;
}

std::string GetParaDebugPath() {
  auto save_path = common::GetEnv(kMS_PARA_DEBUG_PATH);
  char real_path[PATH_MAX] = {0};
  if (!save_path.empty()) {
    if (realpath(save_path.c_str(), real_path)) {
      save_path = real_path;
    } else {
      MS_LOG(EXCEPTION) << "Invalid environment variable '" << kMS_PARA_DEBUG_PATH << "', the path is " << save_path
                        << ". Please check (1) whether the path exists, (2) whether the path has the access "
                           "permission, (3) whether the path is too long.";
    }
  } else {
    save_path = "";
  }
  return save_path;
}

std::vector<std::string> GetTuneOpsList(const std::string &d) {
  std::vector<string> res;
  auto ops = common::GetEnv(kTUNE_OPS_NAME);
  if (ops.empty()) {
    return {};
  }
  size_t p1 = 0;
  size_t p2 = ops.find(d);
  while (p2 != std::string::npos) {
    if (p1 < ops.length() && (p2 - p1) < ops.length()) {
      (void)res.emplace_back(ops.substr(p1, p2 - p1));
    }

    p1 = p2 + 1;
    p2 = ops.find(d, p1);
  }
  if (p1 <= ops.length()) {
    (void)res.emplace_back(ops.substr(p1));
  }
  return res;
}
}  // namespace

void TbeKernelCompileManager::PrintProcessLog(const nlohmann::json &json,
                                              int adjust_log_level = MsLogLevel::kException) const {
  auto all_logs = GetJsonValue<std::vector<nlohmann::json>>(json, kProcessInfo);
  auto job_id = GetJsonValue<int>(json, kJobId);
  auto json_name = GetJsonValue<std::string>(json, kFusionOpName);
  std::sort(all_logs.begin(), all_logs.end(), Order);
  for (const auto &item : all_logs) {
    PrintInfo(item, json_name, job_id, adjust_log_level);
  }
}

void TbeKernelCompileManager::PrintCompileResult(const nlohmann::json &json) {
  auto job_type = GetJsonValue<std::string>(json, kJobType);
  auto json_name = GetJsonValue<std::string>(json, kFusionOpName);
  if (json.at(kStatus) == kFailed) {
    if (job_type == kFusionCompile || job_type == kPreCompile || job_type == kTune) {
      auto all_logs = GetJsonValue<std::vector<nlohmann::json>>(json, kProcessInfo);
      auto message = FilterExceptionMessage(all_logs);
      MS_LOG(INFO) << json_name << " " << job_type << " running failed.\n except_msg: " << message;
      return;
    } else {
      PrintProcessLog(json);
      auto task_id = GetJsonValue<int>(json, kJobId);
      auto target_node = job_id_to_node_[task_id];
      MS_LOG(EXCEPTION) << json_name << " " << job_type << " running failed." << trace::DumpSourceLines(target_node);
    }
  }
}

void TbeKernelCompileManager::ParseTargetJobStatus(const nlohmann::json &json, TargetJobStatus *target_status) const {
  MS_EXCEPTION_IF_NULL(target_status);
  if (GetJsonValue<std::string>(json, kStatus) == kSuccess) {
    nlohmann::json query_result;
    if (!ParseJson(GetJsonValue<std::string>(json, kResult), &query_result)) {
      MS_LOG(EXCEPTION) << "Parse query result error.";
    }
    auto json_name = GetJsonValue<std::string>(query_result, kFusionOpName);
    auto target_job_id = GetJsonValue<int>(query_result, kJobId);
    auto status = GetJsonValue<std::string>(query_result, kStatus);
    auto all_logs = GetJsonValue<std::vector<nlohmann::json>>(query_result, kProcessInfo);
    auto message = FilterExceptionMessage(all_logs);
    // save job status and exception message
    target_status->target_job_id = target_job_id;
    target_status->json_name = json_name;
    target_status->except_msg = message;
    if (status == kSuccess) {
      target_status->job_status = kSuccess;
    } else if (status != kSuccess && status != kRunning) {
      target_status->job_status = kFailed;
    }
  }
}

std::string TbeKernelCompileManager::ParseOpPattern(const std::string &json_str) const {
  nlohmann::json result;
  if (!ParseJson(json_str, &result)) {
    MS_LOG(WARNING) << "Parse op pattern json error. Origin result: " << json_str;
    return kernel::kPatternOpaque;
  }
  return GetJsonValue<std::string>(result, "pattern");
}

nlohmann::json TbeKernelCompileManager::TurnStrToJson(const std::string &string) const {
  nlohmann::json json;
  if (!ParseJson(string, &json)) {
    MS_LOG(EXCEPTION) << "Parse build result error.";
  }
  if (!json.is_object()) {
    MS_LOG(EXCEPTION) << "Json str is not an object, str: " << string;
  }
  return json;
}

void TbeKernelCompileManager::JsonAssemble(const std::string &job_type, const nlohmann::json &src_json,
                                           nlohmann::json *dst_json) const {
  MS_EXCEPTION_IF_NULL(src_json);
  MS_EXCEPTION_IF_NULL(dst_json);
  static size_t job_id = 0;
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  static uint32_t source_id = context_ptr->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  (*dst_json)[kJobType] = job_type;
  (*dst_json)[kJobId] = job_id++;
  (*dst_json)[kSourceId] = source_id;
  if (job_type == kInitialize || job_type == kFinalize) {
    nlohmann::json job_info;
    static auto process_num = GetProcessNum();
    job_info[kProcessNum] = process_num;
    job_info[kParaDebugPath] = GetParaDebugPath();
    job_info[kTbeImplPath] = "";
    job_info[kSocInfo] = src_json;
    nlohmann::json tune_infos;
    tune_infos[kTuneOpList] = GetTuneOpsList(",");
    tune_infos[kTuneDumpPath] = TbeUtils::GetTuneDumpPath();
    tune_infos[kTuneBankPath] = TbeUtils::GetBankPath();
    job_info[kTuneInfo] = tune_infos;
    nlohmann::json lic_infos;
    kernel::tbe::TbeUtils::GenLicInfo(&lic_infos);
    job_info[kLicInfo] = lic_infos;
    (*dst_json)[kJobContent] = job_info;
  } else if (job_type == kQuery) {
    nlohmann::json content;
    content[kSourceId] = GetJsonValue<int>(src_json, kSourceId);
    content[kJobId] = GetJsonValue<int>(src_json, kJobId);
    (*dst_json)[kJobContent] = content;
  } else {
    (*dst_json)[kJobContent] = src_json;
  }
}

void TbeKernelCompileManager::GetAllTbeNodes(const std::shared_ptr<session::KernelGraph> &kernel_graph,
                                             std::vector<CNodePtr> *tbe_nodes) const {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  MS_EXCEPTION_IF_NULL(tbe_nodes);
  auto all_nodes = kernel_graph->execution_order();
  for (const auto &anf_node : all_nodes) {
    MS_EXCEPTION_IF_NULL(anf_node);
    if (!AnfUtils::IsRealKernel(anf_node) || IsPrimitiveCNode(anf_node, prim::kPrimCallInline)) {
      continue;
    }
    KernelType kernel_type = AnfAlgo::GetKernelType(anf_node);
    if (kernel_type == TBE_KERNEL) {
      if (AnfAlgo::GetKernelMod(anf_node) == nullptr) {
        (void)tbe_nodes->push_back(anf_node);
      }
    }
  }
}

std::string TbeKernelCompileManager::DispatchCompileTask(const nlohmann::json &kernel_json) const {
  return AscendKernelBuildClient::Instance().DispatchToServer(kernel_json.dump());
}

void TbeKernelCompileManager::SavePreBuildResult(const std::string &json_name, const std::string &pre_build_result) {
  nlohmann::json result;
  if (!ParseJson(pre_build_result, &result)) {
    MS_LOG(WARNING) << "Parse pre-build result error. Origin result: " << pre_build_result;
    return;
  }
  auto op_pattern = ParseOpPattern(GetJsonValue<std::string>(result, "op_pattern"));
  auto output_data_desc = GetJsonValue<nlohmann::json>(result, "op_params");
  auto core_type = GetJsonValue<nlohmann::json>(result, "core_type");
  // save pre build result
  struct PreBuildResult pre_res;
  pre_res.json_name = json_name;
  pre_res.fusion_type = op_pattern;
  pre_res.core_type = core_type;
  pre_res.output_data_desc = output_data_desc;
  prebuild_res_map_[json_name] = pre_res;
}

void TbeKernelCompileManager::SaveFailedTaskCompileResult(int task_id) {
  const auto task_iter = task_map_.find(task_id);
  if (task_iter == task_map_.end()) {
    MS_LOG(WARNING) << "Can not find pre-build task_id:" << task_id;
    return;
  }
  auto json_name = task_iter->second.json_name;
  auto config_path = TbeUtils::GetOpDebugPath();
  auto cache_file = config_path + kNotSupportFusionOp;
  if (cache_file.size() > PATH_MAX) {
    MS_LOG(WARNING) << "File path length should be smaller than " << PATH_MAX << ", but got " << cache_file;
    return;
  }
  std::ofstream file_write;
  file_write.open(cache_file, std::ios::app);
  if (!file_write.is_open()) {
    MS_LOG(WARNING) << "Create info file failed. [" << cache_file << "]";
    return;
  }
  file_write << json_name << std::endl;
  file_write.close();
  not_support_fusion_ops_.insert(json_name);
}

void TbeKernelCompileManager::SaveSucceedTaskCompileResult(int task_id, const std::string &compile_info,
                                                           const std::string &job_type) {
  if (job_type == kPreCompile) {
    MS_LOG(DEBUG) << "Find pre-build task_id: " << task_id << ", result:" << compile_info;
    auto task_iter = task_map_.find(task_id);
    if (task_iter == task_map_.end()) {
      MS_EXCEPTION(ArgumentError) << "Can not find pre-build task_id:" << task_id;
    }
    auto json_name = task_iter->second.json_name;
    SavePreBuildResult(json_name, compile_info);
    // save pre_build result to json file
    TbeUtils::SaveJsonInfo(json_name, compile_info, tbe::saveType::TBE_PREBUILD);
    return;
  } else {
    auto task_info = task_map_[task_id];
    auto json_name = task_info.json_name;
    TbeUtils::UpdateCache(json_name);
    if (task_info.is_dynamic) {
      bool save_flag = true;
      TbeUtils::SaveCompileInfo(json_name, compile_info, &save_flag);
      if (!save_flag) {
        MS_LOG(EXCEPTION) << "Save json file failed, op: " << json_name << ", compile_info:" << compile_info;
      }
    }
  }
}

void TbeKernelCompileManager::SaveIOSizeInfo(const nlohmann::json &json, const std::string &json_name,
                                             const std::vector<AnfNodePtr> &output_nodes) {
  MS_LOG(DEBUG) << "Save io size info for " << json_name;
  if (kernel_io_size_info_.find(json_name) != kernel_io_size_info_.end()) {
    MS_LOG(DEBUG) << "Io size info of " << json_name << " already exist, skip.";
    return;
  }
  struct KernelIOSizeInfo info;
  std::vector<size_t> input_size_list;
  std::vector<size_t> output_size_list;
  if (!output_nodes.empty()) {
    (void)TbeKernelBuild::GetIOSize(GetJsonValue<nlohmann::json>(json, kOpList), output_nodes, &input_size_list,
                                    &output_size_list);
  } else {
    (void)TbeKernelBuild::GetIOSize(json, &input_size_list, &output_size_list);
  }
  info.json_name = json_name;
  info.input_size_list.assign(input_size_list.begin(), input_size_list.end());
  info.output_size_list.assign(output_size_list.begin(), output_size_list.end());
  kernel_io_size_info_[json_name] = info;
}

void TbeKernelCompileManager::SaveTaskInfo(const bool is_dynamic, const nlohmann::json &json,
                                           const std::string &json_name, const std::string &full_name, int task_id,
                                           int64_t scope_id) {
  struct TaskInfo task_info;
  task_info.json_name = json_name;
  task_info.full_name = full_name;
  task_info.build_json = json;
  task_info.task_id = task_id;
  task_info.is_dynamic = is_dynamic;
  // only fusion save scope_id
  if (scope_id != INT64_MAX) {
    task_info.scope_id = scope_id;
  }
  task_map_[task_id] = task_info;
}

void TbeKernelCompileManager::QueryProcess(const std::string &type, const std::string &job_result,
                                           std::vector<int> *success_job, std::vector<int> *failed_job) {
  MS_EXCEPTION_IF_NULL(success_job);
  MS_EXCEPTION_IF_NULL(failed_job);
  auto json_obj = TurnStrToJson(job_result);
  // the query job' status.
  if (json_obj.at(kStatus) == kSuccess) {
    nlohmann::json query_obj;
    if (!ParseJson(GetJsonValue<std::string>(json_obj, kResult), &query_obj)) {
      MS_LOG(EXCEPTION) << "Parse query result error.";
    }
    auto kernel_name = GetJsonValue<std::string>(query_obj, kFusionOpName);
    struct TargetJobStatus target_status;
    ParseTargetJobStatus(json_obj, &target_status);
    if (target_status.job_status == kSuccess) {
      MS_LOG(DEBUG) << kernel_name << type << " running success.";
      std::string build_result = GetJsonValue<std::string>(query_obj, kResult);
      SaveSucceedTaskCompileResult(target_status.target_job_id, build_result, type);
      (void)success_job->emplace_back(target_status.target_job_id);
    } else if (target_status.job_status == kFailed) {
      SaveFailedTaskCompileResult(target_status.target_job_id);
      if (type == kPreCompile) {
        MS_LOG(INFO) << "Single op pre build failed, op: " << kernel_name
                     << "\n except_msg : " << target_status.except_msg;
        (void)failed_job->emplace_back(target_status.target_job_id);
      } else if (type == kCompile) {
        auto target_job_id = target_status.target_job_id;
        auto target_cnode = job_id_to_node_[target_job_id];
        std::ostringstream oss;
        (void)failed_job->emplace_back(target_job_id);
        oss << "op: " << kernel_name << ".#dmsg#Operator Compilation Exception Message:#dmsg#"
            << target_status.except_msg << trace::DumpSourceLines(target_cnode) << "\n";
        failed_log_ += oss.str();
        MS_LOG(INFO) << "Single op compile failed. " << oss.str();
      } else {
        MS_LOG(INFO) << "Op " << kernel_name << " " << type << " failed,\n except_msg : " << target_status.except_msg;
        (void)failed_job->emplace_back(target_status.target_job_id);
      }
    }
    return;
  }
  MS_LOG(EXCEPTION) << "Query job failed.";
}

void TbeKernelCompileManager::Query(const std::string &type) {
  size_t query_cnt = 0;
  size_t last_sleep = 0;
  size_t sleep_time = 0;
  while (!task_map_.empty()) {
    std::vector<int> success_job;
    std::vector<int> failed_job;
    auto iter = task_map_.begin();
    while (iter != task_map_.end()) {
      nlohmann::json query_json;
      auto task_info = iter->second;
      auto kernel_json = task_info.build_json;
      JsonAssemble(kQuery, kernel_json, &query_json);
      auto job_result = DispatchCompileTask(query_json);
      query_cnt++;
      QueryProcess(type, job_result, &success_job, &failed_job);
      (void)iter++;
    }

    bool sleep_flag = true;
    for (auto k : success_job) {
      (void)task_map_.erase(k);
      sleep_flag = false;
    }
    success_job.clear();
    for (auto k : failed_job) {
      (void)task_map_.erase(k);
      sleep_flag = false;
    }
    failed_job.clear();

    if (sleep_flag && !task_map_.empty()) {
      if ((query_cnt - last_sleep) > KSleepInterval * task_map_.size()) {
        MS_LOG(INFO) << "Querying Parallel Compilation Job. Current Query Count: " << query_cnt;
        last_sleep = query_cnt;
        (void)usleep(KSleepUSeconds * (1U << sleep_time));
        sleep_time++;
      }
    }
  }
}

std::pair<std::vector<CNodePtr>, std::vector<CNodePtr>> TbeKernelCompileManager::GenKernelMod(
  const std::vector<CNodePtr> &node_list) {
  MS_LOG(INFO) << "Gen kernel mod start!";
  std::vector<CNodePtr> success_nodes;
  std::vector<CNodePtr> failed_nodes;

  for (auto &node : node_list) {
    MS_EXCEPTION_IF_NULL(node);
    if (AnfAlgo::GetKernelMod(node) != nullptr) {
      (void)success_nodes.emplace_back(node);
      continue;  // kernel mod already exist, continue;
    }
    auto op_name = common::AnfAlgo::GetCNodeName(node);

    auto full_name = node->fullname_with_scope();
    if (common::AnfAlgo::HasNodeAttr(kAttrOriFusionName, node)) {
      full_name = common::AnfAlgo::GetNodeAttr<std::string>(node, kAttrOriFusionName);
    }
    auto json_name = full_name_to_json_name_[full_name];
    auto kernel_pack = tbe::TbeUtils::SearchCache(json_name, false);
    if (kernel_pack == nullptr) {
      auto *bin_map = tbe::KernelMeta::GetInstance();
      MS_EXCEPTION_IF_NULL(bin_map);
      kernel_pack = bin_map->SearchInFile(json_name);
      if (kernel_pack == nullptr) {
        MS_LOG(INFO) << "Can not find .json file or the .o file for op:" << json_name << trace::DumpSourceLines(node);
        (void)failed_nodes.emplace_back(node);
        continue;
      }
    }
    auto kernel_info_json = kernel_pack->kernel_json_info();
    std::shared_ptr<TbeKernelMod> kernel_mod_ptr;
    if (common::AnfAlgo::IsDynamicShape(node)) {
      kernel_mod_ptr = std::make_shared<DynamicTbeKernelMod>(kernel_pack, node);
    } else {
      kernel_mod_ptr = std::make_shared<TbeKernelMod>(kernel_pack, node);
    }
    MS_EXCEPTION_IF_NULL(kernel_mod_ptr);

    auto iter = kernel_io_size_info_.find(json_name);
    if (iter == kernel_io_size_info_.end() || iter->second.json_name != json_name) {
      MS_LOG(EXCEPTION) << "Can not find node io size info for: " << full_name << trace::DumpSourceLines(node);
    }
    kernel_mod_ptr->SetInputSizeList(iter->second.input_size_list);
    kernel_mod_ptr->SetOutputSizeList(iter->second.output_size_list);
    kernel_mod_ptr->SetWorkspaceSizeList(kernel_info_json.workspaces);
    if (op_name == kNPUClearFloatStatusV2OpName || op_name == kNPUGetFloatStatusV2OpName) {
      constexpr size_t io_byte_size = 32;
      const std::vector<size_t> size_list = {io_byte_size};
      kernel_mod_ptr->SetInputSizeList(size_list);
      kernel_mod_ptr->SetOutputSizeList(size_list);
    }
    AnfAlgo::SetKernelMod(kernel_mod_ptr, node.get());
    (void)success_nodes.emplace_back(node);
  }
  MS_LOG(INFO) << "Gen kernel mod end!";
  return std::make_pair(success_nodes, failed_nodes);
}

void TbeKernelCompileManager::UpdateFusionTypeAndOutputDataDesc(const std::vector<CNodePtr> &nodes) {
  // save prebuild result: fusion_type, output_data_desc
  MS_LOG(INFO) << "Start update fusion type after pre build";
  for (auto &node : nodes) {
    MS_EXCEPTION_IF_NULL(node);
    auto full_name = node->fullname_with_scope();
    auto kernel_name = pre_build_full_name_to_json_name_[full_name];
    if (prebuild_res_map_.find(kernel_name) == prebuild_res_map_.end()) {
      MS_LOG(WARNING) << kernel_name << " not in prebuild_res_map_. Op name: " << full_name;
      continue;
    }
    auto pre_res = prebuild_res_map_[kernel_name];
    auto fusion_type = pre_res.fusion_type;
    auto output_data_desc = pre_res.output_data_desc;
    auto core_type = pre_res.core_type;
    AnfAlgo::SetCoreType(node, core_type);
    AnfAlgo::SetFusionType(node, fusion_type);
    common::AnfAlgo::SetNodeAttr(kAttrTbeFusionType, MakeValue(fusion_type), node);
    AnfAlgo::SetOutputDataDesc(node, {output_data_desc});
  }
  MS_LOG(INFO) << "End update fusion type after pre build";
}

void TbeKernelCompileManager::PrintInitResult(const nlohmann::json &json) const {
  if (json.at(kStatus) == kFailed) {
    PrintProcessLog(json);
    MS_LOG(EXCEPTION) << "TbeInitialize running failed.";
  }
  MS_LOG(INFO) << "TbeInitialize running success.";
}

std::string TbeKernelCompileManager::ParseSelectAndCheckResult(const nlohmann::json &json, const CNodePtr &node) const {
  // for check supported and format select
  MS_EXCEPTION_IF_NULL(node);
  auto job_type = GetJsonValue<std::string>(json, kJobType);
  auto json_name = GetJsonValue<std::string>(json, kFusionOpName);
  auto res = GetJsonValue<std::string>(json, kResult);
  if (job_type == kCheckSupport) {
    if (json.at(kStatus) == kFailed) {
      auto all_logs = GetJsonValue<std::vector<nlohmann::json>>(json, kProcessInfo);
      auto message = FilterExceptionMessage(all_logs);
      MS_LOG(INFO) << json_name << " " << kCheckSupport << " failed.\nRes: " << message;
      return kFailed;
    }
    if (res != kFullySupported) {
      PrintProcessLog(json, static_cast<int>(MsLogLevel::kDebug));
    }
  } else if (json.at(kStatus) == kFailed) {
    auto all_logs = GetJsonValue<std::vector<nlohmann::json>>(json, kProcessInfo);
    auto except_msg = FilterExceptionMessage(all_logs);
    MS_LOG(EXCEPTION) << job_type << " running failed, op: " << json_name << "\nexception message:" << except_msg
                      << trace::DumpSourceLines(node);
  }
  MS_LOG(DEBUG) << json_name << " " << job_type << " success, get: " << res;
  return res;
}

JsonNameMap TbeKernelCompileManager::GetAllSuccessFusion() {
  auto iter = all_fusion_ops_.begin();
  while (iter != all_fusion_ops_.end()) {
    auto scope_id = iter->first;
    auto full_name = iter->second;
    auto json_name = full_name_to_json_name_[full_name];
    if (TbeUtils::SearchCache(json_name) != nullptr) {
      (void)success_fusion_ops_.emplace(scope_id, full_name);
    }
    (void)iter++;
  }
  return success_fusion_ops_;
}

void TbeKernelCompileManager::DistributePreBuildTask(const std::vector<CNodePtr> &node_list) {
  auto json_creator = std::make_shared<BuildTbeJsonCreator>();
  MS_EXCEPTION_IF_NULL(json_creator);
  nlohmann::json kernel_json;  // for gen json
  nlohmann::json build_json;   // for assemble json
  for (const auto &node : node_list) {
    MS_EXCEPTION_IF_NULL(node);
    kernel_json.clear();
    build_json.clear();
    if (!json_creator->GenJson(node, &kernel_json)) {
      MS_LOG(EXCEPTION) << "Generate pre build json failed, op: " << node->fullname_with_scope()
                        << trace::DumpSourceLines(node);
    }
    auto json_name = json_creator->GetJsonName();
    auto full_name = node->fullname_with_scope();
    pre_build_full_name_to_json_name_[full_name] = json_name;  // cache kernel name
    if (prebuild_res_map_.find(json_name) != prebuild_res_map_.end()) {
      // cache exist, no need pre_build
      continue;
    }
    if (TbeUtils::IsOneOf(pre_build_single_processed_kernels_, json_name)) {
      // same op skip prebuild
      continue;
    }
    pre_build_single_processed_kernels_.insert(json_name);
    JsonAssemble(kPreCompile, kernel_json, &build_json);
    auto task_id = GetJsonValue<int>(build_json, kJobId);
    auto is_dynamic = common::AnfAlgo::IsDynamicShape(node);
    SaveTaskInfo(is_dynamic, build_json, json_name, full_name, task_id, INT64_MAX);

    // save pair<task_id, node> for exception print and get node trace
    (void)job_id_to_node_.emplace(std::pair<int, CNodePtr>(task_id, node));
    // start compile
    auto build_result = DispatchCompileTask(build_json);
    auto json_obj = TurnStrToJson(build_result);
    // print message of build
    PrintCompileResult(json_obj);
  }
}

void TbeKernelCompileManager::DistributeCompileTask(const std::vector<CNodePtr> &node_list,
                                                    const std::string &job_type) {
  if (job_type == kPreCompile) {
    DistributePreBuildTask(node_list);
    return;
  }
  auto json_creator = std::make_shared<BuildTbeJsonCreator>();
  MS_EXCEPTION_IF_NULL(json_creator);
  // for gen json
  nlohmann::json kernel_json;
  // for assemble json
  nlohmann::json build_json;
  for (const auto &node : node_list) {
    MS_EXCEPTION_IF_NULL(node);
    kernel_json.clear();
    build_json.clear();
    if (common::AnfAlgo::HasNodeAttr(kAttrIsUBFusionOp, node) &&
        common::AnfAlgo::GetNodeAttr<bool>(node, kAttrIsUBFusionOp)) {
      // skip fusion op, if node has the attr: kAttrIsUBFusionOp, means already done fusion compile, can not do single
      // op compile
      continue;
    }
    if (!json_creator->GenJson(node, &kernel_json)) {
      MS_LOG(EXCEPTION) << "Generate compile json failed, [" << node->fullname_with_scope() << "]"
                        << trace::DumpSourceLines(node);
    }
    auto json_name = json_creator->GetJsonName();
    auto full_name = node->fullname_with_scope();
    full_name_to_json_name_[full_name] = json_name;
    if (common::AnfAlgo::IsDynamicShape(node)) {
      common::AnfAlgo::SetNodeAttr(kAttrJsonFileName, MakeValue(json_name), node);
    }
    // save all task io size info for gen kernel mod
    SaveIOSizeInfo(kernel_json, json_name);
    if (tbe::TbeUtils::SearchCache(json_name, false) != nullptr) {
      // cache exist, no need compile
      continue;
    }
    if (TbeUtils::IsOneOf(single_processed_kernels_, json_name)) {
      // same op only compile once
      continue;
    }
    single_processed_kernels_.insert(json_name);
    JsonAssemble(job_type, kernel_json, &build_json);
    // save compile json to file; cache io size for gen kernel mod
    auto build_str = build_json.dump(indent);
    TbeUtils::SaveJsonInfo(json_name, build_str);
    auto task_id = GetJsonValue<int>(build_json, kJobId);
    auto is_dynamic = common::AnfAlgo::IsDynamicShape(node);
    SaveTaskInfo(is_dynamic, build_json, json_name, full_name, task_id, INT64_MAX);

    // save pair<task_id, node> for exception print and get node trace
    (void)job_id_to_node_.emplace(std::pair<int, CNodePtr>(task_id, node));
    // start compile
    auto build_result = DispatchCompileTask(build_json);
    auto json_obj = TurnStrToJson(build_result);
    // print message of build
    PrintCompileResult(json_obj);
  }
}

void TbeKernelCompileManager::TbePreBuild(const KernelGraphPtr &kernel_graph) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  MS_LOG(INFO) << "Single op pre build start.";
  struct timeval start_time;
  struct timeval end_time;
  (void)gettimeofday(&start_time, nullptr);
  std::vector<CNodePtr> node_list;
  GetAllTbeNodes(kernel_graph, &node_list);
  if (node_list.empty()) {
    return;
  }
  DistributeCompileTask(node_list, kPreCompile);
  Query(kPreCompile);
  UpdateFusionTypeAndOutputDataDesc(node_list);
  (void)gettimeofday(&end_time, nullptr);
  uint64_t cost = kUSecondInSecond * static_cast<uint64_t>(end_time.tv_sec - start_time.tv_sec);
  cost += static_cast<uint64_t>(end_time.tv_usec - start_time.tv_usec);
  MS_LOG(INFO) << "Kernel PreBuild run in " << cost << " us.";
  MS_LOG(INFO) << "Single op pre build end.";
}

std::pair<std::vector<CNodePtr>, std::vector<CNodePtr>> TbeKernelCompileManager::TbeSingleOpCompile(
  const std::vector<CNodePtr> &node_list) {
  MS_LOG(INFO) << "Single op parallel build start.";
  auto job_type = is_tune_flag_ ? kTune : kCompile;
  DistributeCompileTask(node_list, job_type);
  Query(job_type);
  auto ret = GenKernelMod(node_list);
  ClearOldTask();
  MS_LOG(INFO) << "TBE Single op parallel build result: all:" << node_list.size() << " success:" << ret.first.size()
               << " failed:" << ret.second.size() << ".";
  return ret;
}

JsonNameMap TbeKernelCompileManager::TbeFusionOpCompile(const std::vector<FusionScopeInfo> &fusion_scopes) {
  MS_LOG(INFO) << "Fusion op build start";
  auto json_creator = std::make_shared<FusionBuildTbeJsonCreator>();
  MS_EXCEPTION_IF_NULL(json_creator);
  const std::string job_type = is_tune_flag_ ? kTune : kFusionCompile;
  success_fusion_ops_.clear();
  nlohmann::json fusion_op;
  nlohmann::json build_json;
  for (const auto &fusion_scope_iter : fusion_scopes) {
    fusion_op.clear();
    build_json.clear();
    if (!json_creator->GenJson(fusion_scope_iter, &fusion_op)) {
      MS_LOG(WARNING) << "Generate fusion json failed, fusion info: " << fusion_scope_iter.full_name;
      continue;
    }
    auto full_name = fusion_scope_iter.full_name;
    auto json_name = json_creator->GetJsonName();
    full_name_to_json_name_[full_name] = json_name;
    // save all fusion ops to filter those compile succeed ops
    all_fusion_ops_[fusion_scope_iter.scope_id] = full_name;
    SaveIOSizeInfo(fusion_op, json_name, fusion_scope_iter.output_nodes);
    if (tbe::TbeUtils::SearchCache(json_name, false) != nullptr) {
      // cache exist, no need compile
      continue;
    }
    if (TbeUtils::IsOneOf(fusion_processed_kernels_, json_name)) {
      // same fusion op only compile once
      continue;
    }
    if (TbeUtils::IsOneOf(not_support_fusion_ops_, json_name)) {
      // fusion op not support
      continue;
    }
    fusion_processed_kernels_.insert(json_name);
    JsonAssemble(job_type, fusion_op, &build_json);
    auto build_str = build_json.dump(indent);
    MS_LOG(DEBUG) << "FusionOp build json file : " << build_str;
    TbeUtils::SaveJsonInfo(json_name, build_str);
    auto build_result = DispatchCompileTask(build_json);
    auto json_obj = TurnStrToJson(build_result);
    PrintCompileResult(json_obj);
    auto task_id = GetJsonValue<int>(json_obj, kJobId);
    SaveTaskInfo(false, build_json, json_name, full_name, task_id, fusion_scope_iter.scope_id);
  }
  Query(job_type);
  // only fusion op compile succeed can be return
  return GetAllSuccessFusion();
}

std::string TbeKernelCompileManager::TbeOpSelectFormat(const CNodePtr &node) const {
  MS_EXCEPTION_IF_NULL(node);
  auto full_name = node->fullname_with_scope();
  MS_LOG(INFO) << "Op select format start for op [" << full_name << "]";
  auto json_creator = std::make_shared<SelectTbeJsonCreator>();
  MS_EXCEPTION_IF_NULL(json_creator);
  nlohmann::json kernel_info;
  nlohmann::json select_json;
  if (!json_creator->GenJson(node, &kernel_info)) {
    MS_LOG(EXCEPTION) << "Gen select json failed. [" << full_name << "]" << trace::DumpSourceLines(node);
  }
  JsonAssemble(kSelectFormat, kernel_info, &select_json);
  auto select_ret = DispatchCompileTask(select_json);
  auto json_ret = TurnStrToJson(select_ret);
  MS_LOG(INFO) << "Op select format result: " << select_ret;
  return ParseSelectAndCheckResult(json_ret, node);
}

bool TbeKernelCompileManager::TbeOpCheckSupported(const CNodePtr &node, nlohmann::json *kernel_json) const {
  MS_EXCEPTION_IF_NULL(node);
  auto full_name = node->fullname_with_scope();
  MS_LOG(DEBUG) << "Check supported for op [" << full_name << "]";
  auto json_creator = std::make_shared<BuildTbeJsonCreator>();
  MS_EXCEPTION_IF_NULL(json_creator);
  auto &compute_json = (*kernel_json)[kJOpList].back();
  auto inputs_json_tmp = compute_json[kJInputDesc];
  if (!json_creator->GenInputsJson(node, &compute_json)) {
    MS_LOG(ERROR) << "update inputs json failed, node full name:" << node->fullname_with_scope();
  }
  nlohmann::json check_json;
  JsonAssemble(kCheckSupport, *kernel_json, &check_json);
  auto check_ret = DispatchCompileTask(check_json);
  compute_json[kJInputDesc] = inputs_json_tmp;
  auto json_ret = TurnStrToJson(check_ret);
  if (json_ret.at(kStatus) == kFailed) {
    MS_LOG(DEBUG) << "Call check supported api failed, result info: " << check_ret;
    return false;
  }
  auto check_result = json_ret.at(kResult);
  if (check_result == kPartiallySupported || check_result == kFullySupported) {
    return true;
  }
  MS_LOG(DEBUG) << "The shape is not support, result info: " << check_ret;
  return false;
}

void TbeKernelCompileManager::LoadNotSupportFusionOp() {
  static bool has_load = false;
  if (!has_load) {
    auto config_path = TbeUtils::GetOpDebugPath();
    auto cache_file = config_path + kNotSupportFusionOp;
    std::ifstream file_read;
    file_read.open(cache_file.c_str());
    if (!file_read.is_open()) {
      MS_LOG(INFO) << "Note: File is not open. File: " << cache_file;
      return;
    }
    std::string json_name;
    while (file_read >> json_name) {
      not_support_fusion_ops_.insert(json_name);
    }
    file_read.close();
    has_load = true;
  }
}

void TbeKernelCompileManager::LoadPreBuildResult() {
  static bool has_load = false;
  if (!has_load) {
    auto config_path = TbeUtils::GetOpDebugPath();
    auto bin_dir = config_path + kTbePrebuildRes;
    DIR *dir = opendir(bin_dir.c_str());
    if (dir == nullptr) {
      MS_LOG(INFO) << "Open dir failed. Dir:" << bin_dir;
      return;
    }
    struct dirent *entry;
    constexpr size_t SUFFIX_LENS = 5;
    while ((entry = readdir(dir)) != nullptr) {
      string bin_dir_tmp = bin_dir;
      std::string tbe_prebuild_json = entry->d_name;
      if (tbe_prebuild_json.length() <= SUFFIX_LENS) {
        continue;
      }
      std::string suffix = tbe_prebuild_json.substr(tbe_prebuild_json.length() - SUFFIX_LENS);
      if (suffix != kJsonSuffix) {
        continue;
      }
      auto sp = tbe_prebuild_json.rfind('/');
      if (sp != std::string::npos) {
        continue;
      }
      sp = tbe_prebuild_json.rfind('.');
      if (sp == std::string::npos) {
        continue;
      }
      auto kernel_name = tbe_prebuild_json.substr(0, sp);
      (void)bin_dir_tmp.append("/");
      (void)bin_dir_tmp.append(tbe_prebuild_json);
      std::ifstream file(bin_dir_tmp.c_str());
      if (!file.is_open()) {
        MS_LOG(WARNING) << "File is not open. File: " << bin_dir_tmp;
        continue;
      }
      std::string pre_build_result =
        std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
      SavePreBuildResult(kernel_name, pre_build_result);
    }
    (void)closedir(dir);
    has_load = true;
  }
}

void TbeKernelCompileManager::TbeInitialize() {
  if (tbe_init_flag_) {
    MS_LOG(DEBUG) << "TbeInitialize already complete, no need do again";
    return;
  }
  MS_LOG(INFO) << "TbeInitialize start";
  nlohmann::json init_json;
  nlohmann::json soc_info = TbeUtils::GenSocInfo();
  JsonAssemble(kInitialize, soc_info, &init_json);
  auto offline_tune = (init_json[kJobContent][kSocInfo][kOfflineTune]).get<bool>();
  op_debug_level_ = (init_json[kJobContent][kSocInfo]["op_debug_level"]).get<std::string>();
  op_debug_config_ = (init_json[kJobContent][kSocInfo]["op_debug_config"]).get<std::string>();
  auto auto_tiling_mode = (init_json[kJobContent][kSocInfo]["autoTilingMode"]).get<std::string>();
  tbe_init_flag_ = true;
  is_tune_flag_ = offline_tune || (auto_tiling_mode != "NO_TUNE");

  auto init_str = init_json.dump();
  MS_LOG(INFO) << "TbeInitialize json file : " << init_str;
  TbeUtils::SaveJsonInfo(kInitialize, init_json.dump(indent));
  auto init_ret = DispatchCompileTask(init_json);
  auto json_ret = TurnStrToJson(init_ret);
  PrintInitResult(json_ret);
  // load cache before kernel build
  LoadPreBuildResult();
  // load not support op
  LoadNotSupportFusionOp();
  TbeUtils::LoadCache();
}

void TbeKernelCompileManager::TbeFinalize() {
  MS_LOG(INFO) << "TbeFinalize start.";
  if (!tbe_init_flag_) {
    MS_LOG(DEBUG) << "TbeFinalize already complete, no need do again";
    return;
  }
  op_debug_level_ = "";
  op_debug_config_ = "";
  tbe_init_flag_ = false;
  is_tune_flag_ = false;
  ClearOldTask();
  MS_LOG(INFO) << "TbeFinalize end.";
}

void TbeKernelCompileManager::ClearOldTask() {
  task_map_.clear();
  all_fusion_ops_.clear();
  job_id_to_node_.clear();
  success_fusion_ops_.clear();
  kernel_io_size_info_.clear();
  full_name_to_json_name_.clear();
  single_processed_kernels_.clear();
  fusion_processed_kernels_.clear();
}

TbeKernelCompileManager::~TbeKernelCompileManager() { TbeFinalize(); }
bool TbeKernelCompileManager::tbe_init_flag_ = false;
bool TbeKernelCompileManager::is_tune_flag_ = false;
}  // namespace ascend
}  // namespace kernel
}  // namespace mindspore
