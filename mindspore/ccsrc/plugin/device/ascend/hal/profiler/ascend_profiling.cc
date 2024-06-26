/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "plugin/device/ascend/hal/profiler/ascend_profiling.h"
#include <map>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>
#include "utils/log_adapter.h"
#include "utils/ms_context.h"
#include "include/common/utils/utils.h"
#include "include/backend/mem_reuse/mem_tracker.h"
#include "plugin/device/ascend/hal/common/ascend_utils.h"
#include "plugin/device/ascend/hal/profiler/memory_profiling.h"
#include "plugin/device/ascend/hal/profiler/parallel_strategy_profiling.h"
#include "common/debug/profiler/profiling_framework_data.h"
#include "runtime/hardware/device_context_manager.h"
#include "transform/symbol/acl_prof_symbol.h"
#include "transform/symbol/acl_rt_symbol.h"
#include "transform/symbol/symbol_utils.h"

using mindspore::device::ascend::ErrorManagerAdapter;
using mindspore::profiler::ascend::MemoryProfiling;

namespace mindspore {
namespace profiler {
namespace ascend {
namespace {
PROFILER_REG(kAscendDevice, AscendProfiler);

constexpr auto kAclProfStepStartTag = 60000;
constexpr auto kAclProfStepEndTag = 60001;
}  // namespace

std::map<std::string, aclprofAicoreMetrics> kAicMetrics{{"ArithmeticUtilization", ACL_AICORE_ARITHMETIC_UTILIZATION},
                                                        {"PipeUtilization", ACL_AICORE_PIPE_UTILIZATION},
                                                        {"Memory", ACL_AICORE_MEMORY_BANDWIDTH},
                                                        {"MemoryL0", ACL_AICORE_L0B_AND_WIDTH},
                                                        {"ResourceConflictRatio", ACL_AICORE_RESOURCE_CONFLICT_RATIO},
                                                        {"MemoryUB", ACL_AICORE_MEMORY_UB},
                                                        {"L2Cache", ACL_AICORE_L2_CACHE},
                                                        {"None", ACL_AICORE_NONE}};

std::map<std::string, uint64_t> profLevelMap{{"Level0", Level0}, {"Level1", Level1}, {"Level2", Level2}};

std::shared_ptr<AscendProfiler> AscendProfiler::GetInstance() {
  auto instance = Profiler::GetInstance(kAscendDevice);
  MS_EXCEPTION_IF_NULL(instance);
  return std::dynamic_pointer_cast<AscendProfiler>(instance);
}

void AscendProfiler::StepProfilingEnable(const bool enable_flag) {
  MS_LOG(INFO) << "Start profiling";
  enable_flag_ = enable_flag;
}

void AscendProfiler::Init(const std::string &profiling_path, uint32_t device_id, const std::string &profiling_options) {
  mindspore::device::ascend::InitializeAcl();
  MS_LOG(INFO) << "Begin to init profiling and call aclprofInit function.";
  profiling_options_ = profiling_options;
  profile_data_path_ = profiling_path;
  device_id_ = device_id;

  nlohmann::json options;
  try {
    options = nlohmann::json::parse(profiling_options);
  } catch (nlohmann::json::exception &e) {
    MS_LOG(EXCEPTION) << "Failed to parse profiling options because of format error, current options is " << options;
  }
  if (options["parallel_strategy"] == "off") {
    is_parallel_strategy = false;
  } else {
    is_parallel_strategy = true;
  }
  aclError ret = CALL_ASCEND_API(aclrtSetDevice, static_cast<int32_t>(device_id));
  if (ret != ACL_ERROR_NONE) {
    MS_LOG(EXCEPTION) << "Device " << device_id << " call aclrtSetDevice failed, ret[" << static_cast<int>(ret) << "]";
  }

  // Init ErrorManager instance in order to get error msg reported by Ascend.
  (void)ErrorManagerAdapter::Init();

  MemoryProfiling::GetInstance().SetMemoryProfilingInitialize(profiling_options_);

  aclError aclRet = CALL_ASCEND_API(aclprofInit, profile_data_path_.c_str(), profile_data_path_.length());
  if (aclRet != ACL_SUCCESS) {
    MS_LOG(EXCEPTION) << "Failed to call aclprofInit function. error_code : " << static_cast<int>(aclRet);
  }

  if (options["hbm_ddr"] == "on" || options["profile_memory"] == "on") {
    const char *hbmFreq = "100";
    aclError hbmRet = aclprofSetConfig(ACL_PROF_SYS_HARDWARE_MEM_FREQ, hbmFreq, strlen(hbmFreq));
    if (hbmRet != ACL_SUCCESS) {
      MS_LOG(EXCEPTION) << "Failed to set hbm profiling config. error_code : " << static_cast<int>(hbmRet);
    }
  }

  if (options["profile_memory"] == "on") {
    MS_LOG(INFO) << "profile_memory is on, profile_data_path:" << profile_data_path_;
    enable_prof_mem_ = true;
    MS_LOG(INFO) << "Enable profiling memory.";
    auto ms_context = MsContext::GetInstance();
    ms_context->set_param<std::string>(MS_CTX_PROF_MEM_OUTPUT_PATH, profile_data_path_);
    ms_context->set_param<bool>(MS_CTX_ENABLE_PROF_MEM, true);
  }

  if (options["pcie"] == "on") {
    const char *pcieFreq = "50";
    aclError pcieRet = aclprofSetConfig(ACL_PROF_SYS_INTERCONNECTION_FREQ, pcieFreq, strlen(pcieFreq));
    if (pcieRet != ACL_SUCCESS) {
      MS_LOG(EXCEPTION) << "Failed to set pcie profiling config. error_code : " << static_cast<int>(pcieRet);
    }
  }
  if (options["host_stack"] == "on") {
    host_stack_ = true;
  } else {
    host_stack_ = false;
  }
  uint32_t device_list[1] = {device_id_};
  uint32_t device_num = 1;
  aclprofAicoreMetrics aic_metrics = GetAicMetrics();
  uint64_t mask = GetOptionsMask(aic_metrics);
  acl_config_ = CALL_ASCEND_API(aclprofCreateConfig, device_list, device_num, aic_metrics, nullptr, mask);
  if (acl_config_ == nullptr) {
    MS_LOG(EXCEPTION) << "Failed to call aclprofCreateConfig function.";
  }
  MS_LOG(INFO) << "Start profiling, options mask is " << mask << " aic_metrics is " << aic_metrics;

  init_flag_ = true;
}

uint64_t AscendProfiler::GetOptionsMask(aclprofAicoreMetrics aic_metrics) const {
  uint64_t mask = ACL_PROF_ACL_API;
  nlohmann::json options_json;
  try {
    options_json = nlohmann::json::parse(profiling_options_);
  } catch (const std::exception &err) {
    MS_LOG(ERROR) << "Failed to parse profiling options.";
    return ACL_AICORE_NONE;
  }

  if (aic_metrics != ACL_AICORE_NONE) {
    mask |= ACL_PROF_AICORE_METRICS;
  }

  if (options_json["task_trace"] == "on") {
    mask |= ACL_PROF_TASK_TIME;
  }

  if (options_json["aicpu"] == "on") {
    mask |= ACL_PROF_AICPU;
  }

  if (options_json["hccl"] == "on") {
    mask |= ACL_PROF_HCCL_TRACE;
  }

  if (options_json["profiler_level"] != "off" &&
      profLevelMap.find(options_json["profiler_level"]) != profLevelMap.end()) {
    mask = ACL_PROF_ACL_API;  // reset mask
    mask |= profLevelMap[options_json["profiler_level"]];
  }

  if (options_json["training_trace"] == "on") {
    mask |= ACL_PROF_TRAINING_TRACE;
  }

  if (options_json["l2_cache"] == "on") {
    mask |= ACL_PROF_L2CACHE;
  }
  if (options_json["profile_memory"] == "on") {
    mask |= ACL_PROF_TASK_MEMORY;
  }
  return mask;
}

aclprofAicoreMetrics AscendProfiler::GetAicMetrics() const {
  nlohmann::json options_json;
  try {
    options_json = nlohmann::json::parse(profiling_options_);
  } catch (const std::exception &err) {
    MS_LOG(ERROR) << "Failed to parse profiling options.";
    return ACL_AICORE_NONE;
  }
  auto result = std::find_if(kAicMetrics.begin(), kAicMetrics.end(), [&options_json](const auto &metric) {
    return metric.first == options_json["aic_metrics"];
  });
  if (result == kAicMetrics.end()) {
    return ACL_AICORE_NONE;
  }
  return result->second;
}

void AscendProfiler::Start() {
  MS_LOG(INFO) << "Begin to profiling.";
  aclError aclRet = CALL_ASCEND_API(aclprofStart, acl_config_);
  if (aclRet != ACL_SUCCESS) {
    MS_LOG(EXCEPTION) << "Failed to call aclprofStart function. error_code : " << static_cast<int>(aclRet);
  }

  MemoryProfiling::GetInstance().StartMemoryProfiling();
  if (enable_prof_mem_) {
    MS_LOG(INFO) << "Start profiling memory.";
    auto &mem_tracker = device::tracker::MemTrackerManager::GetInstance();
    mem_tracker.UpdateProfilingPos();
    auto &&ms_context = MsContext::GetInstance();
    ms_context->set_param<bool>(MS_CTX_ENABLE_PROF_MEM, true);
  }

  profiler::ascend::ParallelStrategy::GetInstance()->SaveParallelStrategyToFile();
  std::string op_range_dir = profile_data_path_ + "/FRAMEWORK";
  uint32_t global_rank_id_ = 0;
  device::DeviceContextKey host_key = {"CPU", 0};
  auto host_ctx_ = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(host_key);
  MS_EXCEPTION_IF_NULL(host_ctx_);
  auto host_comm_lib_instance_ = host_ctx_->device_res_manager_->collective_comm_lib();
  if (host_comm_lib_instance_ != nullptr) {
    global_rank_id_ = host_comm_lib_instance_->global_rank_id();
  } else if (!common::GetEnv("RANK_ID").empty()) {
    global_rank_id_ = static_cast<int32_t>(std::atoi(common::GetEnv("RANK_ID").c_str()));
  }
  ProfilingFrameworkData::Device_Id = global_rank_id_;
  ProfilingDataDumper::GetInstance().Init(op_range_dir);
  ProfilingDataDumper::GetInstance().Start();
  StepProfilingEnable(true);
}

void AscendProfiler::Stop() {
  MS_LOG(INFO) << "Begin to stop profiling.";

  if (acl_config_ == nullptr) {
    MS_LOG(EXCEPTION)
      << "Failed to stop profiling because of null aReportDatacl config.Please make sure call Profiler.Start function "
         "before call Profiler.Stop function.";
  }

  ProfilingDataDumper::GetInstance().Stop();
  aclError aclRet = CALL_ASCEND_API(aclprofStop, acl_config_);
  if (aclRet != ACL_SUCCESS) {
    MS_LOG(EXCEPTION) << "Failed to call aclprofStop function. error_code : " << static_cast<int>(aclRet);
  }

  MemoryProfiling::GetInstance().StopMemoryProfiling();

  if (enable_prof_mem_) {
    std::string csvPrefix = "operator_memory";
    device::tracker::MemTrackerManager::GetInstance().DumpProfilingMemInfo(profile_data_path_, csvPrefix);
    device::tracker::MemTrackerManager::GetInstance().Dump();

    MS_LOG(INFO) << "End profiling memory.";
    auto &&ms_context = MsContext::GetInstance();
    ms_context->set_param<bool>(MS_CTX_ENABLE_PROF_MEM, false);
  }

  StepProfilingEnable(false);
}

struct aclprofStepInfoInner {
  bool startFlag;
  bool endFlag;
  uint64_t indexId;
};

void AscendProfiler::StepStart(uint64_t step_id, void *stream) {
  acl_stream_ = static_cast<aclrtStream>(stream);
  acl_prof_step_info_ = CALL_ASCEND_API(aclprofCreateStepInfo);
  aclprofStepInfoInner *ptr_info = reinterpret_cast<aclprofStepInfoInner *>(acl_prof_step_info_);
  ptr_info->indexId = step_id;
  auto ret =
    CALL_ASCEND_API(aclprofGetStepTimestamp, acl_prof_step_info_, (aclprofStepTag)kAclProfStepStartTag, acl_stream_);
  if (ret != ACL_SUCCESS) {
    MS_LOG(WARNING) << "Failed to call aclprofGetStepTimestamp with tag " << kAclProfStepStartTag << ".";
  }
}

void AscendProfiler::StepStop() {
  auto ret =
    CALL_ASCEND_API(aclprofGetStepTimestamp, acl_prof_step_info_, (aclprofStepTag)kAclProfStepEndTag, acl_stream_);
  if (ret != ACL_SUCCESS) {
    MS_LOG(WARNING) << "Failed to call aclprofGetStepTimestamp with tag " << kAclProfStepEndTag << ".";
  }
  if (acl_prof_step_info_ != nullptr) {
    CALL_ASCEND_API(aclprofDestroyStepInfo, acl_prof_step_info_);
    acl_prof_step_info_ = nullptr;
  }
  acl_stream_ = nullptr;
}

void AscendProfiler::Finalize() {
  aclError aclRet = CALL_ASCEND_API(aclprofDestroyConfig, acl_config_);
  if (aclRet != ACL_SUCCESS) {
    MS_LOG(EXCEPTION) << "Failed to call aclprofDestoryConfig function. error_code : " << static_cast<int>(aclRet);
  }
  MS_LOG(INFO) << "Begin to finalize profiling";
  aclRet = CALL_ASCEND_API(aclprofFinalize);
  if (aclRet != ACL_SUCCESS) {
    MS_LOG(EXCEPTION) << "Failed to call aclprofDestroyConfig function. error_code : " << static_cast<int>(aclRet);
  }
}
}  // namespace ascend
}  // namespace profiler
}  // namespace mindspore
