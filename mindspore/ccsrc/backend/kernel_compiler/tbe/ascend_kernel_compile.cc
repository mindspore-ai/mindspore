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

#include "backend/kernel_compiler/tbe/ascend_kernel_compile.h"
#include <sys/syscall.h>
#include <unistd.h>
#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "mindspore/ccsrc/backend/kernel_compiler/tbe/tbe_json/tbe_json_creator.h"
#include "mindspore/ccsrc/backend/kernel_compiler/tbe/tbe_json/single_tbe_json_creator.h"
#include "mindspore/ccsrc/backend/kernel_compiler/tbe/tbe_json/fusion_tbe_json_creator.h"
#include "backend/kernel_compiler/tbe/tbe_utils.h"
#include "backend/kernel_compiler/tbe/tbe_convert_utils.h"
#include "backend/session/anf_runtime_algorithm.h"
#include "common/util/error_manager/error_manager.h"
#include "debug/anf_ir_dump.h"
#include "frontend/operator/ops.h"
#include "utils/ms_context.h"
#include "utils/ms_utils.h"
#include "utils/trace_base.h"
#include "utils/utils.h"
#include "utils/json_operation_utils.h"

namespace mindspore {
namespace kernel {
namespace ascend {
using mindspore::kernel::tbe::TbeAdapter;
using mindspore::kernel::tbe::TbeUtils;
const int indent = 4;  // for dump json
constexpr auto kInitialize = "Initialize";
constexpr auto kPreCompile = "PreCompile";
constexpr auto kFinalize = "Finalize";
constexpr auto kCompile = "Compile";
constexpr auto kTune = "Tune";
constexpr auto kOfflineTune = "offlineTune";
constexpr auto kCheckSupport = "CheckSupport";
constexpr auto kSelectFormat = "SelectFormat";
constexpr auto kFullySupported = "FULLY_SUPPORTED";
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
constexpr auto kMS_BUILD_PROCESS_NUM = "MS_BUILD_PROCESS_NUM";
constexpr auto kMS_PARA_DEBUG_PATH = "PARA_DEBUG_PATH";
constexpr auto kTBE_IMPL_PATH = "TBE_IMPL_PATH";
constexpr auto kTUNE_OPS_NAME = "TUNE_OPS_NAME";
constexpr auto kDefPath = "/usr/local/Ascend/ascend-toolkit/latest/opp/op_impl/built-in/ai_core/tbe/";
constexpr auto kBkPath = "/usr/local/Ascend/opp/op_impl/built-in/ai_core/tbe/";
constexpr int KSleepSeconds = 3;
constexpr int KSleepInterval = 1000;

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
  for (auto it = exception_message.begin(); it != exception_message.end(); it++) {
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
  } else if (level == 1) {
    MS_LOG(INFO) << "Job id:" << job_id << ", name :" << job_name << ", message:" << message;
  } else if (level == 2) {
    MS_LOG(WARNING) << "Job id:" << job_id << ", name :" << job_name << ", message:" << message;
  } else if (level == 3) {
    MS_LOG(ERROR) << "Job id:" << job_id << ", name :" << job_name << ", message:" << message;
  } else if (level == 4) {
    ReportToErrorManager(message);
  }
}

uint32_t GetProcessNum() {
  uint32_t process_num = 24;
  auto env_process_num = common::GetEnv(kMS_BUILD_PROCESS_NUM);
  if (!env_process_num.empty()) {
    try {
      process_num = UlongToUint(std::stoul(env_process_num));
    } catch (std::invalid_argument &e) {
      MS_LOG(EXCEPTION) << "Invalid MS_BUILD_PROCESS_NUM env:" << env_process_num
                        << ". Please set the value of MS_BUILD_PROCESS_NUM in [0, 24]";
    }
  }
  return process_num;
}

int StrToInt(const std::string &env) {
  if (env == "0") {
    return DEBUG;
  } else if (env == "1") {
    return INFO;
  } else if (env == "3") {
    return ERROR;
  }
  return WARNING;
}

int GetLogLevel() {
  auto env = common::GetEnv(kGLOG_v);
  int ms_level = StrToInt(env);
  return ms_level;
}

std::string GetParaDebugPath() {
  auto save_path = common::GetEnv(kMS_PARA_DEBUG_PATH);
  char real_path[PATH_MAX] = {0};
  if (!save_path.empty()) {
    if (realpath(save_path.c_str(), real_path)) {
      save_path = real_path;
    } else {
      MS_LOG(EXCEPTION) << "Invalid env PARA_DEBUG_PATH, path : " << save_path;
    }
  } else {
    save_path = "";
  }
  return save_path;
}

std::string GetTbePath() {
  auto save_path = common::GetEnv(kTBE_IMPL_PATH);
  char real_path[PATH_MAX] = {0};
  if (!save_path.empty()) {
    if (realpath(save_path.c_str(), real_path)) {
      save_path = real_path;
    } else {
      MS_LOG(EXCEPTION) << "Invalid env TBE_IMPL_PATH, path : " << save_path;
    }
  } else {
    if (realpath(kDefPath, real_path)) {
      save_path = real_path;
    } else if (realpath(kBkPath, real_path)) {
      save_path = real_path;
    } else {
      MS_LOG(WARNING) << "Can not get access to [" << kDefPath << "] or [" << kBkPath << "]";
    }
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
    res.emplace_back(ops.substr(p1, p2 - p1));
    p1 = p2 + 1;
    p2 = ops.find(d, p1);
  }
  if (p1 != ops.length()) {
    res.emplace_back(ops.substr(p1));
  }
  return res;
}
}  // namespace

void AscendKernelCompileManager::ResetOldTask() {
  if (build_manager_ != nullptr) {
    build_manager_->ResetTaskInfo();
  }
  job_list_.clear();
}

void AscendKernelCompileManager::PrintProcessLog(const nlohmann::json &json, int adjust_log_level = EXCEPTION) {
  auto logs = GetJsonValue<std::vector<nlohmann::json>>(json, kProcessInfo);
  auto job_id = GetJsonValue<int>(json, kJobId);
  auto json_name = GetJsonValue<std::string>(json, kFusionOpName);
  std::vector<nlohmann::json> all_logs;
  std::copy(logs.begin(), logs.end(), std::back_inserter(all_logs));
  std::sort(all_logs.begin(), all_logs.end(), Order);
  for (const auto &item : all_logs) {
    PrintInfo(item, json_name, job_id, adjust_log_level);
  }
}

void AscendKernelCompileManager::PrintInitResult(const nlohmann::json &json) {
  auto job_type = GetJsonValue<std::string>(json, kJobType);
  auto json_name = GetJsonValue<std::string>(json, kFusionOpName);
  MS_LOG(DEBUG) << "Job: " << job_type << " post processing.";
  PrintProcessLog(json);
  if (json.at(kStatus) == kFailed) {
    MS_LOG(EXCEPTION) << "Job " << job_type << " running failed, job_id: " << json.at(kJobId)
                      << ", source_id: " << json[kSourceId] << ".";
  }
  MS_LOG(INFO) << "Job: " << job_type << " running success, job id: " << json.at(kJobId) << ", " << json_name;
}

void AscendKernelCompileManager::PrintSingleBuildResult(const nlohmann::json &json) {
  auto job_type = GetJsonValue<std::string>(json, kJobType);
  auto json_name = GetJsonValue<std::string>(json, kFusionOpName);
  MS_LOG(DEBUG) << "Job: " << job_type << " post process";
  PrintProcessLog(json);
  if (json.at(kStatus) == kFailed) {
    MS_LOG(ERROR) << "Job " << job_type << " running failed, job_id: " << json.at(kJobId) << ", : " << json_name;
    return;
  }
  MS_LOG(INFO) << "Job " << job_type << " running " << json.at(kStatus) << ", job id: " << json.at(kJobId) << ", "
               << json_name;
}

void AscendKernelCompileManager::PrintFusionOpBuildResult(const nlohmann::json &json) {
  auto job_type = GetJsonValue<std::string>(json, kJobType);
  auto json_name = GetJsonValue<std::string>(json, kFusionOpName);
  MS_LOG(DEBUG) << "Job: " << job_type << " post process";
  PrintProcessLog(json, INFO);
  if (json.at(kStatus) == kFailed) {
    MS_LOG(INFO) << "Job fusion running failed, job_id: " << json.at(kJobId) << ", " << json_name;
    return;
  }
  MS_LOG(INFO) << "Job " << job_type << " running " << json.at(kStatus) << ", job id: " << json.at(kJobId) << ", "
               << json_name;
}

std::string AscendKernelCompileManager::FormatSelectResultProcess(const nlohmann::json &json) {
  // for check supported and format select
  auto job_type = GetJsonValue<std::string>(json, kJobType);
  auto json_name = GetJsonValue<std::string>(json, kFusionOpName);
  MS_LOG(DEBUG) << "Job: " << job_type << " post process";
  if (json.at(kStatus) == kFailed) {
    if (job_type == kCheckSupport) {
      PrintProcessLog(json, WARNING);
      MS_LOG(WARNING) << "Job:" << job_type << " running failed, job_id: " << json.at(kJobId) << ", " << json_name;
      return kFailed;
    } else {
      PrintProcessLog(json);
      MS_LOG(EXCEPTION) << "Job:" << job_type << " running failed, job_id: " << json.at(kJobId) << ", " << json_name;
    }
  }
  auto res = GetJsonValue<std::string>(json, kResult);
  MS_LOG(INFO) << "Job:" << job_type << " running success, id: " << json.at(kJobId) << ", " << json_name
               << ", get: " << res;
  return res;
}

void AscendKernelCompileManager::QueryResultProcess(const nlohmann::json &json, TargetJobStatus *task_info,
                                                    int adjust_log_level = EXCEPTION) {
  auto job_type = GetJsonValue<std::string>(json, kJobType);
  auto json_name = GetJsonValue<std::string>(json, kFusionOpName);
  MS_LOG(DEBUG) << "Job: " << job_type << " post processing";
  if (GetJsonValue<std::string>(json, kStatus) == kSuccess) {
    nlohmann::json query_result;
    if (!ParseJson(GetJsonValue<std::string>(json, kResult), &query_result)) {
      MS_LOG(EXCEPTION) << "Parse query result error.";
    }
    auto target_job_id = query_result.at(kJobId);
    auto target_status = query_result.at(kStatus);
    task_info->target_job_id = target_job_id;
    // target job result
    auto target_job = query_result.at(kJobType);
    if (target_status == kSuccess) {
      MS_LOG(DEBUG) << "Target job: " << target_job << " running success, job id: " << target_job_id << ", "
                    << json_name;
      task_info->job_status = kSuccess;
      return;
    } else if (target_status != kSuccess && target_status != kRunning) {
      MS_LOG(INFO) << "Target job running failed, target_job_type:" << target_job << ", job id: " << target_job_id
                   << ", target_result: " << query_result.dump();
      task_info->job_status = kFailed;
    }
    PrintProcessLog(query_result, adjust_log_level);
  }
}

nlohmann::json AscendKernelCompileManager::TurnStrToJson(const std::string &string) {
  nlohmann::json json;
  if (!ParseJson(string, &json)) {
    MS_LOG(EXCEPTION) << "Parse build result error.";
  }
  if (!json.is_object()) {
    MS_LOG(EXCEPTION) << "Json str is not an object, str: " << string;
  }
  return json;
}

void AscendKernelCompileManager::ParseTargetJobStatus(const std::string &type, const std::string &build_result,
                                                      std::vector<int> *success_job) {
  MS_EXCEPTION_IF_NULL(success_job);
  auto json_obj = TurnStrToJson(build_result);
  if (json_obj.at(kStatus) == kSuccess) {
    nlohmann::json query_obj;
    if (!ParseJson(GetJsonValue<std::string>(json_obj, kResult), &query_obj)) {
      MS_LOG(EXCEPTION) << "Parse query result error.";
    }
    struct TargetJobStatus task_info;
    QueryResultProcess(json_obj, &task_info);
    if (task_info.job_status == kSuccess) {
      MS_LOG(DEBUG) << "Job " << GetJsonValue<std::string>(query_obj, kJobType) << " running success.";
      std::string build_result = GetJsonValue<std::string>(query_obj, kResult);
      if (type == kPreCompile) {
        build_manager_->PreTaskFinishProcess(task_info.target_job_id, build_result);
      } else {
        build_manager_->TaskFinishProcess(task_info.target_job_id, build_result);
      }
      success_job->emplace_back(task_info.target_job_id);
    } else if (task_info.job_status == kFailed) {
      if (type == kPreCompile) {
        success_job->emplace_back(task_info.target_job_id);
        MS_LOG(WARNING) << "Single op pre build failed ,res: " << query_obj;
      } else {
        ResetOldTask();
        single_processed_kernels_.clear();
        MS_LOG(EXCEPTION) << "Single op compile failed ,res: " << query_obj;
      }
    }
  } else {
    auto file_name = GetJsonValue<std::string>(json_obj, kJobType) + "_" + json_obj.at(kJobId).dump();
    TbeUtils::SaveJsonInfo(file_name, build_result);
    MS_LOG(EXCEPTION) << "Query job failed";
  }
}

void AscendKernelCompileManager::QueryFinishJob(const std::string &job_type) {
  MS_EXCEPTION_IF_NULL(build_manager_);
  size_t query_cnt = 0;
  while (!job_list_.empty()) {
    std::vector<int> success_job;
    auto iter = job_list_.begin();
    while (iter != job_list_.end()) {
      nlohmann::json query_json;
      auto kernel_json = iter->second;
      JsonAssemble(kQuery, kernel_json, &query_json);
      auto build_result = build_manager_->ProcessTbeJob(query_json);
      query_cnt++;
      ParseTargetJobStatus(job_type, build_result, &success_job);
      iter++;
    }
    for (auto k : success_job) {
      job_list_.erase(k);
    }
    success_job.clear();
    if (!job_list_.empty()) {
      if (query_cnt % KSleepInterval == 0) {
        MS_LOG(INFO) << "Querying Parallel Compilation Job, Current Query Count: " << query_cnt;
        sleep(KSleepSeconds);
      }
    }
  }
}

void AscendKernelCompileManager::QueryFusionFinishJob(KernelModMap *kernel_mode_ret) {
  MS_EXCEPTION_IF_NULL(build_manager_);
  MS_EXCEPTION_IF_NULL(kernel_mode_ret);
  int build_failed_nums = 0;
  size_t query_cnt = 0;
  while (!job_list_.empty()) {
    std::vector<int> success_job;
    auto iter = job_list_.begin();
    while (iter != job_list_.end()) {
      nlohmann::json query_json;
      auto kernel_json = iter->second;
      JsonAssemble(kQuery, kernel_json, &query_json);
      auto build_result = build_manager_->ProcessTbeJob(query_json);
      query_cnt++;
      auto json_obj = TurnStrToJson(build_result);
      if (json_obj.at(kStatus) == kSuccess) {
        struct TargetJobStatus task_info;
        QueryResultProcess(json_obj, &task_info, INFO);
        if (task_info.job_status == kSuccess) {
          MS_LOG(DEBUG) << "Job " << GetJsonValue<std::string>(json_obj, kJobType) << " running success.";
          std::string build_res = GetJsonValue<std::string>(json_obj, kResult);
          auto kernel_mode_item = build_manager_->TaskFinishProcess(task_info.target_job_id, build_res, false);
          if (kernel_mode_item.second != nullptr) {
            (void)kernel_mode_ret->emplace(kernel_mode_item);
          }
          success_job.emplace_back(task_info.target_job_id);
        } else if (task_info.job_status == kFailed) {
          MS_LOG(INFO) << "FusionOp compile failed.";
          auto target_id = task_info.target_job_id;
          success_job.emplace_back(target_id);
          build_failed_nums += 1;
        }
      } else {
        auto file_name = GetJsonValue<std::string>(json_obj, kJobType) + "_" + json_obj.at(kJobId).dump();
        TbeUtils::SaveJsonInfo(file_name, json_obj.dump());
        PrintProcessLog(json_obj);
        MS_LOG(EXCEPTION) << "Query job Failed";
      }
      iter++;
    }
    for (auto k : success_job) {
      job_list_.erase(k);
    }
    success_job.clear();
    if (!job_list_.empty()) {
      if (query_cnt % KSleepInterval == 0) {
        MS_LOG(INFO) << "Querying Parallel Compilation Job, Current Query Count: " << query_cnt;
        sleep(KSleepSeconds);
      }
    }
  }
  MS_LOG(INFO) << "Compile Fusion Kernel Failed Num: " << build_failed_nums;
}

bool AscendKernelCompileManager::JsonAssemble(const std::string &job_type, const nlohmann::json &src_json,
                                              nlohmann::json *dst_json) {
  MS_EXCEPTION_IF_NULL(src_json);
  MS_EXCEPTION_IF_NULL(dst_json);
  static size_t job_id = 0;
  static auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  static int source_id = context_ptr->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  (*dst_json)[kJobType] = job_type;
  (*dst_json)[kJobId] = job_id++;
  (*dst_json)[kSourceId] = source_id;
  if (job_type == kInitialize || job_type == kFinalize) {
    nlohmann::json job_info;
    static auto process_num = GetProcessNum();
    job_info[kProcessNum] = process_num;
    job_info[kLogLevel] = GetLogLevel();
    job_info[kEnableEvent] = false;
    job_info[kParaDebugPath] = GetParaDebugPath();
    job_info[kTbeImplPath] = GetTbePath();
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
    content[kSourceId] = src_json[kSourceId];
    content[kJobId] = src_json[kJobId];
    (*dst_json)[kJobContent] = content;
  } else {
    (*dst_json)[kJobContent] = src_json;
  }
  return true;
}

void AscendKernelCompileManager::GetAllAscendNodes(const std::shared_ptr<session::KernelGraph> &kernel_graph,
                                                   std::vector<AnfNodePtr> *tbe_nodes) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto all_nodes = kernel_graph->execution_order();
  for (const auto &anf_node : all_nodes) {
    MS_EXCEPTION_IF_NULL(anf_node);
    if (!AnfAlgo::IsRealKernel(anf_node)) {
      continue;
    }
    KernelType kernel_type = AnfAlgo::GetKernelType(anf_node);
    if (kernel_type == TBE_KERNEL) {
      if (AnfAlgo::GetKernelMod(anf_node) == nullptr) {
        tbe_nodes->push_back(anf_node);
      }
    }
  }
}

void AscendKernelCompileManager::AscendPreBuild(const std::shared_ptr<session::KernelGraph> &kernel_graph) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  MS_LOG(INFO) << "Single op pre build start.";
  struct timeval start_time, end_time;
  (void)gettimeofday(&start_time, nullptr);
  MS_EXCEPTION_IF_NULL(build_manager_);
  std::vector<AnfNodePtr> anf_nodes;
  GetAllAscendNodes(kernel_graph, &anf_nodes);
  if (anf_nodes.empty()) {
    return;
  }
  auto json_creator = std::make_shared<BuildTbeJsonCreator>();
  MS_EXCEPTION_IF_NULL(json_creator);
  for (const auto &node : anf_nodes) {
    MS_EXCEPTION_IF_NULL(node);
    auto op_name = AnfAlgo::GetCNodeName(node);
    nlohmann::json kernel_json;
    if (!json_creator->GenJson(node, &kernel_json)) {
      MS_LOG(EXCEPTION) << "Generate prebuild json failed, [" << op_name << ", " << node->fullname_with_scope() << "]";
    }
    auto json_name = json_creator->GetJsonName();
    nlohmann::json build_json;
    if (!JsonAssemble(kPreCompile, kernel_json, &build_json)) {
      MS_LOG(EXCEPTION) << "Assemble json failed. job type: " << kPreCompile;
    }
    auto build_result = build_manager_->ProcessTbeJob(build_json);
    auto json_obj = TurnStrToJson(build_result);
    PrintSingleBuildResult(json_obj);
    auto task_id = GetJsonValue<int>(json_obj, kJobId);
    build_manager_->SavePreBuildTaskInfo(task_id, node, json_name);
    if (json_obj.at(kStatus) == kRunning) {
      std::pair<int, nlohmann::json> pair(task_id, build_json);
      job_list_.insert(pair);
    } else if (json_obj.at(kStatus) == kSuccess) {
      std::string build_res = GetJsonValue<std::string>(json_obj, kResult);
      build_manager_->PreTaskFinishProcess(task_id, build_res);
    } else {
      MS_LOG(WARNING) << "Kernel prebuild failed, op: " << op_name << ", json_name: " << json_name;
    }
  }
  QueryFinishJob(kPreCompile);
  (void)gettimeofday(&end_time, nullptr);
  const uint64_t kUSecondInSecond = 1000000;
  uint64_t cost = kUSecondInSecond * static_cast<uint64_t>(end_time.tv_sec - start_time.tv_sec);
  cost += static_cast<uint64_t>(end_time.tv_usec - start_time.tv_usec);
  MS_LOG(INFO) << "Kernel PreBuild run in  " << PRIu64 << " us " << cost;
  MS_LOG(INFO) << "Single op pre build end.";
}

bool AscendKernelCompileManager::AscendSingleOpCompile(const std::vector<AnfNodePtr> &anf_nodes) {
  MS_LOG(INFO) << "Single op parallel build start";
  MS_EXCEPTION_IF_NULL(build_manager_);
  auto json_creator = std::make_shared<BuildTbeJsonCreator>();
  MS_EXCEPTION_IF_NULL(json_creator);
  std::string job_type;
  for (const auto &node : anf_nodes) {
    MS_EXCEPTION_IF_NULL(node);
    if (AnfAlgo::GetKernelMod(node) != nullptr && !is_tune_flag_) {
      continue;
    }
    auto op_name = AnfAlgo::GetCNodeName(node);
    nlohmann::json kernel_json;
    if (!json_creator->GenJson(node, &kernel_json)) {
      MS_LOG(EXCEPTION) << "Generate compile json failed, [" << op_name << ", " << node->fullname_with_scope() << "]";
    }
    auto json_name = json_creator->GetJsonName();
    std::vector<size_t> in_size_list;
    std::vector<size_t> out_size_list;
    (void)TbeKernelBuild::GetIOSize2(kernel_json, &in_size_list, &out_size_list, node);
    if (!is_tune_flag_ && op_debug_level_ != "1" &&
        build_manager_->SearchInCache(json_name, in_size_list, out_size_list, node.get())) {
      continue;
    }
    if (single_processed_kernels_.find(json_name) != single_processed_kernels_.end()) {
      build_manager_->SaveSameOpInfo(node, json_name, in_size_list, out_size_list);
      continue;
    }
    (void)single_processed_kernels_.insert(json_name);

    nlohmann::json build_json;
    job_type = is_tune_flag_ ? kTune : kCompile;
    if (!JsonAssemble(job_type, kernel_json, &build_json)) {
      MS_LOG(EXCEPTION) << "Assemble json failed. job type: " << kCompile << ", op:[" << op_name << ", "
                        << node->fullname_with_scope() << "]";
    }
    auto build_str = build_json.dump(indent);
    MS_LOG(DEBUG) << "Op build json file : " << build_str;
    TbeUtils::SaveJsonInfo(json_name, build_str);
    auto build_result = build_manager_->ProcessTbeJob(build_json);
    auto json_obj = TurnStrToJson(build_result);
    PrintSingleBuildResult(json_obj);
    auto task_id = GetJsonValue<int>(json_obj, kJobId);
    build_manager_->SaveTaskInfo(task_id, node, json_name, in_size_list, out_size_list);
    if (json_obj.at(kStatus) == kRunning) {
      std::pair<int, nlohmann::json> pair(task_id, build_json);
      job_list_.insert(pair);
    } else if (json_obj.at(kStatus) == kSuccess) {
      std::string build_res = GetJsonValue<std::string>(json_obj, kResult);
      build_manager_->TaskFinishProcess(task_id, build_res);
    } else {
      ResetOldTask();
      single_processed_kernels_.clear();
      MS_LOG(EXCEPTION) << "Kernel compile failed, op [" << op_name << "], build result: " << build_result;
    }
  }
  QueryFinishJob(job_type);
  return build_manager_->GenSameOpKernelMod();
}

KernelModMap AscendKernelCompileManager::AscendFusionOpCompile(const std::vector<FusionScopeInfo> &fusion_scopes) {
  MS_LOG(INFO) << "fusion op build start";
  KernelModMap kernel_mode_ret;
  MS_EXCEPTION_IF_NULL(build_manager_);
  auto json_creator = std::make_shared<FusionBuildTbeJsonCreator>();
  MS_EXCEPTION_IF_NULL(json_creator);
  for (const auto &fusion_scope_iter : fusion_scopes) {
    nlohmann::json fusion_op;
    if (!json_creator->GenJson(fusion_scope_iter, &fusion_op)) {
      MS_LOG(WARNING) << "Generate fusion json failed";
      continue;
    }
    auto json_name = json_creator->GetJsonName();
    std::vector<size_t> input_size_list;
    std::vector<size_t> output_size_list;
    if (!TbeKernelBuild::GetIOSize(fusion_op[kOpList], fusion_scope_iter.output_nodes, &input_size_list,
                                   &output_size_list)) {
      continue;
    }
    // cache
    if (!is_tune_flag_ && op_debug_level_ != "1") {
      auto kernel_pack = TbeUtils::SearchCache(json_name);
      if (kernel_pack != nullptr) {
        auto kernel_mod = build_manager_->GenKernelMod(input_size_list, output_size_list, kernel_pack);
        if (kernel_mod != nullptr) {
          kernel_mode_ret[fusion_scope_iter.scope_id] = kernel_mod;
          continue;
        }
      }
    }

    // same op no need build, but need wait build finish to set kernel mode
    if (fusion_processed_kernels_.find(json_name) != fusion_processed_kernels_.end()) {
      build_manager_->SaveSameFusionOpInfo(fusion_scope_iter.scope_id, json_name, tbe::kProcessorAiCore,
                                           input_size_list, output_size_list);
      continue;
    }
    (void)fusion_processed_kernels_.insert(json_name);

    nlohmann::json build_json;
    const std::string job_type = is_tune_flag_ ? kTune : kCompile;
    if (!JsonAssemble(job_type, fusion_op, &build_json)) {
      MS_LOG(EXCEPTION) << "Assemble json failed. job type: [" << kCompile << "]";
    }
    auto build_str = build_json.dump(indent);
    MS_LOG(DEBUG) << "FusionOp build json file : " << build_str;
    TbeUtils::SaveJsonInfo(json_name, build_str);
    auto build_result = build_manager_->ProcessTbeJob(build_json);
    auto json_obj = TurnStrToJson(build_result);
    PrintFusionOpBuildResult(json_obj);
    auto task_id = GetJsonValue<int>(json_obj, kJobId);
    fusion_op_names_[task_id] = json_name;
    build_manager_->SaveTaskInfo(task_id, nullptr, json_name, input_size_list, output_size_list,
                                 fusion_scope_iter.scope_id);
    if (json_obj.at(kStatus) == kRunning) {
      std::pair<int, nlohmann::json> pair(task_id, build_json);
      job_list_.insert(pair);
    } else if (json_obj.at(kStatus) == kSuccess) {
      std::string build_res = GetJsonValue<std::string>(json_obj, kResult);
      auto kernel_mode_item = build_manager_->TaskFinishProcess(task_id, build_res, false);
      if (kernel_mode_item.second != nullptr) {
        (void)kernel_mode_ret.emplace(kernel_mode_item);
      }
    } else {
      MS_LOG(INFO) << "Kernel compile failed for << " << fusion_scope_iter.full_name << ", " << build_result;
    }
  }
  QueryFusionFinishJob(&kernel_mode_ret);
  if (!build_manager_->GenSameFusionOpKernelMod(&kernel_mode_ret)) {
    MS_LOG(INFO) << "Fusion warning: cache failed.";
  }
  return kernel_mode_ret;
}

void AscendKernelCompileManager::TbeInitialize() {
  if (tbe_init_flag_) {
    MS_LOG(DEBUG) << "TbeInitialize already complete, no need do again";
    return;
  }
  MS_LOG(INFO) << "TbeInitialize start";
  build_manager_ = std::make_shared<ParallelBuildManager>();
  MS_EXCEPTION_IF_NULL(build_manager_);
  nlohmann::json init_json;
  nlohmann::json soc_info;
  TbeUtils::GenSocInfo(&soc_info);
  if (!JsonAssemble(kInitialize, soc_info, &init_json)) {
    MS_LOG(EXCEPTION) << "Assemble json failed. job type: Initialize.";
  }
  auto offline_tune = (init_json[kJobContent][kSocInfo][kOfflineTune]).get<bool>();
  op_debug_level_ = (init_json[kJobContent][kSocInfo]["op_debug_level"]).get<std::string>();
  auto auto_tiling_mode = (init_json[kJobContent][kSocInfo]["autoTilingMode"]).get<std::string>();
  tbe_init_flag_ = true;
  is_tune_flag_ = offline_tune || (auto_tiling_mode != "NO_TUNE");

  auto init_str = init_json.dump();
  MS_LOG(INFO) << "TbeInitialize json file : " << init_str;
  TbeUtils::SaveJsonInfo(kInitialize, init_str);
  auto init_ret = build_manager_->ProcessTbeJob(init_json);
  auto json_ret = TurnStrToJson(init_ret);
  PrintInitResult(json_ret);
  MS_LOG(INFO) << "TbeInitialize end";
}

std::string AscendKernelCompileManager::AscendOpSelectFormat(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto op_name = AnfAlgo::GetCNodeName(node);
  MS_LOG(INFO) << "Op select format start for op [" << op_name << ", " << node->fullname_with_scope() << "]";
  MS_EXCEPTION_IF_NULL(build_manager_);
  auto json_creator = std::make_shared<SelectTbeJsonCreator>();
  MS_EXCEPTION_IF_NULL(json_creator);
  nlohmann::json kernel_info;
  nlohmann::json select_json;
  if (!json_creator->GenJson(node, &kernel_info)) {
    MS_LOG(EXCEPTION) << "Gen select json failed. [" << op_name << ", " << node->fullname_with_scope() << "]";
  }
  if (!JsonAssemble(kSelectFormat, kernel_info, &select_json)) {
    MS_LOG(EXCEPTION) << "Assemble json failed. job type: SelectFormat";
  }
  auto select_ret = build_manager_->ProcessTbeJob(select_json);
  auto json_ret = TurnStrToJson(select_ret);
  return FormatSelectResultProcess(json_ret);
}

bool AscendKernelCompileManager::AscendOpCheckSupported(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto op_name = AnfAlgo::GetCNodeName(node);
  MS_LOG(INFO) << "Check supported for op [" << op_name << ", " << node->fullname_with_scope() << "]";
  MS_EXCEPTION_IF_NULL(build_manager_);
  auto json_creator = std::make_shared<CheckTbeJsonCreator>();
  MS_EXCEPTION_IF_NULL(json_creator);
  nlohmann::json kernel_info;
  nlohmann::json check_json;
  if (!json_creator->GenJson(node, &kernel_info)) {
    MS_LOG(EXCEPTION) << "Gen check supported json failed.[" << op_name << ", " << node->fullname_with_scope() << "]";
  }
  if (!JsonAssemble(kCheckSupport, kernel_info, &check_json)) {
    MS_LOG(EXCEPTION) << "Assemble json failed. job type: CheckSupport";
  }
  auto check_ret = build_manager_->ProcessTbeJob(check_json);
  auto json_ret = TurnStrToJson(check_ret);
  std::string check_info = FormatSelectResultProcess(json_ret);
  return check_info == kFullySupported;
}

void AscendKernelCompileManager::TbeFinalize() {
  MS_LOG(INFO) << "TbeFinalize start";
  if (!tbe_init_flag_) {
    MS_LOG(DEBUG) << "TbeFinalize already complete, no need do again";
    return;
  }
  build_manager_ = nullptr;
  tbe_init_flag_ = false;
  is_tune_flag_ = false;
  job_list_.clear();
  single_processed_kernels_.clear();
  fusion_processed_kernels_.clear();
  MS_LOG(INFO) << "TbeFinalize end";
}

AscendKernelCompileManager::~AscendKernelCompileManager() { TbeFinalize(); }

bool AscendKernelCompileManager::tbe_init_flag_ = false;
bool AscendKernelCompileManager::is_tune_flag_ = false;
}  // namespace ascend
}  // namespace kernel
}  // namespace mindspore
