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
#include "utils/log_adapter.h"
#include "include/common/utils/utils.h"
#include "plugin/device/ascend/hal/common/ascend_utils.h"
#include "plugin/device/ascend/hal/profiler/memory_profiling.h"
#include "plugin/device/ascend/hal/device/profiling/profiling_manager.h"
#include "plugin/device/ascend/hal/profiler/parallel_strategy_profiling.h"
#include <nlohmann/json.hpp>
#include "plugin/device/ascend/hal/device/profiling/profiling_reporter.h"
#include "kernel/kernel.h"
#include "acl/acl_rt.h"

using mindspore::device::ascend::ErrorManagerAdapter;
using mindspore::device::ascend::ProfilingManager;
using mindspore::device::ascend::ProfilingReporter;
using mindspore::profiler::ascend::MemoryProfiling;

namespace mindspore {
namespace profiler {
namespace ascend {
namespace {
PROFILER_REG(kAscendDevice, AscendProfiler);
}  // namespace

std::map<std::string, aclprofAicoreMetrics> kAicMetrics{{"ArithmeticUtilization", ACL_AICORE_ARITHMETIC_UTILIZATION},
                                                        {"PipeUtilization", ACL_AICORE_PIPE_UTILIZATION},
                                                        {"Memory", ACL_AICORE_MEMORY_BANDWIDTH},
                                                        {"MemoryL0", ACL_AICORE_L0B_AND_WIDTH},
                                                        {"ResourceConflictRatio", ACL_AICORE_RESOURCE_CONFLICT_RATIO},
                                                        {"MemoryUB", ACL_AICORE_MEMORY_UB},
                                                        {"None", ACL_AICORE_NONE}};

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
  aclError ret = aclrtSetDevice(static_cast<int32_t>(device_id));
  if (ret != ACL_ERROR_NONE) {
    MS_LOG(EXCEPTION) << "Device " << device_id << " call aclrtSetDevice failed, ret[" << static_cast<int>(ret) << "]";
  }

  // Init ErrorManager instance in order to get error msg reported by Ascend.
  (void)ErrorManagerAdapter::Init();

  if (options["op_time"] == "on") {
    (void)ProfilingManager::GetInstance().InitProfiling(profiling_path, device_id);
  }

  MemoryProfiling::GetInstance().SetMemoryProfilingInitialize(profiling_options_);

  aclError aclRet = aclprofInit(profile_data_path_.c_str(), profile_data_path_.length());
  if (aclRet != ACL_SUCCESS) {
    MS_LOG(EXCEPTION) << "Failed to call aclprofInit function.";
  }

  init_flag_ = true;
}

uint64_t AscendProfiler::GetOptionsMask() const {
  uint64_t mask = ACL_PROF_ACL_API | ACL_PROF_AICORE_METRICS;

  nlohmann::json options_json;
  try {
    options_json = nlohmann::json::parse(profiling_options_);
  } catch (const std::exception &err) {
    MS_LOG(ERROR) << "Failed to parse profiling options.";
    return ACL_AICORE_NONE;
  }

  if (options_json["task_trace"] == "on") {
    mask |= ACL_PROF_TASK_TIME;
  }

  if (options_json["training_trace"] == "on") {
    mask |= ACL_PROF_TRAINING_TRACE;
  }

  if (options_json["aicpu"] == "on") {
    mask |= ACL_PROF_AICPU;
  }
  if (options_json["hccl"] == "on") {
    mask |= ACL_PROF_HCCL_TRACE;
  }
  if (options_json["l2_cache"] == "on") {
    mask |= ACL_PROF_L2CACHE;
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
  uint32_t device_list[1] = {device_id_};
  uint32_t device_num = 1;
  uint64_t mask = GetOptionsMask();
  aclprofAicoreMetrics aic_metrics = GetAicMetrics();
  acl_config_ = aclprofCreateConfig(device_list, device_num, aic_metrics, nullptr, GetOptionsMask());
  if (acl_config_ == nullptr) {
    MS_LOG(EXCEPTION) << "Failed to call aclprofCreateConfig function.";
  }
  aclError aclRet = aclprofStart(acl_config_);
  if (aclRet != ACL_SUCCESS) {
    MS_LOG(EXCEPTION) << "Failed to call aclprofStart function.";
  }
  MS_LOG(INFO) << "Start profiling, options mask is " << mask << " aic_metrics is " << aic_metrics;

  MemoryProfiling::GetInstance().StartMemoryProfiling();

  profiler::ascend::ParallelStrategy::GetInstance()->SaveParallelStrategyToFile();

  StepProfilingEnable(true);
}

void AscendProfiler::Stop() {
  MS_LOG(INFO) << "Begin to stop profiling.";
  if (acl_config_ == nullptr) {
    MS_LOG(EXCEPTION)
      << "Failed to stop profiling because of null aReportDatacl config.Please make sure call Profiler.Start function "
         "before call Profiler.Stop function.";
  }

  aclError aclRet = aclprofStop(acl_config_);
  if (aclRet != ACL_SUCCESS) {
    MS_LOG(EXCEPTION) << "Failed to call aclprofStop function.";
  }
  aclRet = aclprofDestroyConfig(acl_config_);
  if (aclRet != ACL_SUCCESS) {
    MS_LOG(EXCEPTION) << "Failed to call aclprofDestroyConfig function.";
  }

  MemoryProfiling::GetInstance().StopMemoryProfiling();

  StepProfilingEnable(false);
}

void AscendProfiler::Finalize() {
  MS_LOG(INFO) << "Begin to finalize profiling";
  aclError aclRet = aclprofFinalize();
  if (aclRet != ACL_SUCCESS) {
    MS_LOG(EXCEPTION) << "Failed to call aclprofDestroyConfig function.";
  }
}

void AscendProfiler::MsprofInitProfiler() const {
  if (ProfilingManager::GetInstance().IsMsprofiling()) {
    ProfilingManager::GetInstance().ProfRegisterCtrlCallback();
    MsprofInit(MSPROF_CTRL_INIT_DYNA, nullptr, 0);
  }
}

void AscendProfiler::MsprofStopProfiler() const {
  if (ProfilingManager::GetInstance().IsMsprofiling()) {
    MsprofFinalize();
  }
}

void AscendProfiler::GetNodeTaskIdStreamId(const CNodePtr &kernel, uint32_t graph_id, int device_id,
                                           const KernelType kernel_type, int32_t kernel_mod_task_id) {
  uint32_t stream_id;
  uint32_t task_id;
  uint32_t aicpu_task_id;
  uint32_t rt_model_id = 0;
  std::vector<CNodePtr> cnode_list;
  std::vector<uint32_t> stream_ids;
  std::vector<uint32_t> task_ids;
  std::thread::id t_id = std::this_thread::get_id();
  auto rt_ret = rtGetTaskIdAndStreamID(&task_id, &stream_id);
  if (rt_ret != RT_ERROR_NONE) {
    MS_LOG(EXCEPTION) << "Profiling get task_id and stream_id failed.";
  }
  if (kernel_mod_task_id != -1) {
    task_id = static_cast<uint32_t>(kernel_mod_task_id);
  }
  ProfilingReporter reporter(device_id, graph_id, rt_model_id, cnode_list, stream_ids, task_ids);
  if (task_id <= last_tid_[t_id] && stream_id == last_streamid_[t_id]) {
    MS_LOG(INFO) << "No task id is allocated to the node <" << kernel->fullname_with_scope() << ">.";
  } else {
    if (task_id >= max_op_taskid_limit_ && (uint32_t)kernel_type == aicpu_kernel_type_) {
      aicpu_task_id = task_id % max_op_taskid_limit_;
      reporter.DynamicNodeReport(kernel, stream_id, aicpu_task_id, kernel_type);
    } else {
      reporter.DynamicNodeReport(kernel, stream_id, task_id, kernel_type);
    }
  }
  last_tid_[t_id] = task_id;
  last_streamid_[t_id] = stream_id;
}
}  // namespace ascend
}  // namespace profiler
}  // namespace mindspore
