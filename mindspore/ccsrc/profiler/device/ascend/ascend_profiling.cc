/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "profiler/device/ascend/ascend_profiling.h"
#include <map>
#include <string>
#include "common/util/error_manager/error_manager.h"
#include "pybind_api/api_register.h"
#include "utils/log_adapter.h"
#include "utils/utils.h"
#include "profiler/device/ascend/memory_profiling.h"
#include "runtime/device/ascend/profiling/profiling_manager.h"
#include "profiler/device/ascend/parallel_strategy_profiling.h"
#include <nlohmann/json.hpp>

using mindspore::device::ascend::ProfilingManager;
using mindspore::profiler::ascend::MemoryProfiling;

namespace mindspore {
namespace profiler {
namespace ascend {
constexpr auto kUnknownErrorString = "Unknown error occurred";

std::map<std::string, aclprofAicoreMetrics> kAicMetrics{
  {"ArithmeticUtilization", ACL_AICORE_ARITHMETIC_UTILIZATION},
  {"PipeUtilization", ACL_AICORE_PIPE_UTILIZATION},
  {"Memory", ACL_AICORE_MEMORY_BANDWIDTH},
  {"MemoryLO", ACL_AICORE_L0B_AND_WIDTH},
  {"ResourceConflictRatio", ACL_AICORE_RESOURCE_CONFLICT_RATIO},
};

std::shared_ptr<AscendProfiler> AscendProfiler::ascend_profiler_ = std::make_shared<AscendProfiler>();

std::shared_ptr<AscendProfiler> &AscendProfiler::GetInstance() { return ascend_profiler_; }

void AscendProfiler::ReportErrorMessage() const {
  const std::string &error_message = ErrorManager::GetInstance().GetErrorMessage();
  if (!error_message.empty() && error_message.find(kUnknownErrorString) == std::string::npos) {
    MS_LOG(ERROR) << "Ascend error occurred, error message:\n" << error_message;
  }
}

void AscendProfiler::StepProfilingEnable(const bool enable_flag) {
  MS_LOG(INFO) << "Start profiling";
  enable_flag_ = enable_flag;
}

void AscendProfiler::InitProfiling(const std::string &profiling_path, uint32_t device_id,
                                   const std::string &profiling_options) {
  MS_LOG(INFO) << "Begin to init profiling and call aclprofInit function.";
  profiling_options_ = profiling_options;
  profile_data_path_ = profiling_path;
  device_id_ = device_id;
  (void)ProfilingManager::GetInstance().InitProfiling(profiling_path, device_id);

  MemoryProfiling::GetInstance().SetMemoryProfilingInitialize(profiling_options_);

  aclError aclRet = aclprofInit(profile_data_path_.c_str(), profile_data_path_.length());
  if (aclRet != ACL_SUCCESS) {
    ReportErrorMessage();
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

  if (options_json["aicpu"] == "on") {
    mask |= ACL_PROF_AICPU;
  }
  if (options_json["hccl"] == "on") {
    mask |= ACL_PROF_HCCL_TRACE;
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
    ReportErrorMessage();
    MS_LOG(EXCEPTION) << "Failed to call aclprofCreateConfig function.";
  }
  aclError aclRet = aclprofStart(acl_config_);
  if (aclRet != ACL_SUCCESS) {
    ReportErrorMessage();
    MS_LOG(EXCEPTION) << "Failed to call aclprofStart function.";
  }
  MS_LOG(INFO) << "Start profiling, options mask is " << mask << " aic_metrics is " << aic_metrics;

  MemoryProfiling::GetInstance().StartMemoryProfiling();

  SaveParallelStrategyToFile();

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
    ReportErrorMessage();
    MS_LOG(EXCEPTION) << "Failed to call aclprofStop function.";
  }
  aclRet = aclprofDestroyConfig(acl_config_);
  if (aclRet != ACL_SUCCESS) {
    ReportErrorMessage();
    MS_LOG(EXCEPTION) << "Failed to call aclprofDestroyConfig function.";
  }

  MemoryProfiling::GetInstance().StopMemoryProfiling();

  StepProfilingEnable(false);
}

void AscendProfiler::Finalize() const {
  MS_LOG(INFO) << "Begin to finalize profiling";
  aclError aclRet = aclprofFinalize();
  if (aclRet != ACL_SUCCESS) {
    ReportErrorMessage();
    MS_LOG(EXCEPTION) << "Failed to call aclprofDestroyConfig function.";
  }
}

REGISTER_PYBIND_DEFINE(AscendProfiler_, ([](const py::module *m) {
                         (void)py::class_<AscendProfiler, std::shared_ptr<AscendProfiler>>(*m, "AscendProfiler")
                           .def_static("get_instance", &AscendProfiler::GetInstance, "AscendProfiler get_instance.")
                           .def("init", &AscendProfiler::InitProfiling, py::arg("profiling_path"), py::arg("device_id"),
                                py::arg("profiling_options"), "init")
                           .def("start", &AscendProfiler::Start, "start")
                           .def("stop", &AscendProfiler::Stop, "stop")
                           .def("finalize", &AscendProfiler::Finalize, "finalize");
                       }));
}  // namespace ascend
}  // namespace profiler
}  // namespace mindspore
