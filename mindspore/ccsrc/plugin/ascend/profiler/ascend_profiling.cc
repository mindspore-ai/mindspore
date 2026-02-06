/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "plugin/ascend/profiler/ascend_profiling.h"
#include <map>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

#include "utils/log_adapter.h"
#include "utils/ms_context.h"
#include "include/utils/utils.h"
#include "tools/profiler/profiling_framework_data.h"
#include "tools/profiler/profiling_python.h"
#include "plugin/ascend/profiler/mstx/mstx_dispatcher.h"
#include "plugin/ascend/res_manager/mem_manager/ascend_memory_pool.h"
#include "plugin/ascend/profiler/parallel_strategy_profiling.h"
#include "include/runtime/hardware_abstract/device_context/device_context_manager.h"
#include "plugin/ascend/res_manager/symbol_interface/acl_prof_symbol.h"
#include "plugin/ascend/res_manager/symbol_interface/acl_rt_symbol.h"
#include "plugin/ascend/res_manager/symbol_interface/symbol_utils.h"
#include "plugin/ascend/res_manager/hal_manager/ascend_err_manager.h"
#include "plugin/ascend/res_manager/hal_manager/ascend_hal_manager.h"

using mindspore::device::ascend::ErrorManagerAdapter;

namespace mindspore {
namespace profiler {
namespace ascend {

namespace {
PROFILER_REG(kAscendDevice, AscendProfiler);

constexpr auto kAclProfStepStartTag = 60000;
constexpr auto kAclProfStepEndTag = 60001;
}  // namespace

using mindspore::profiler::PythonTracer;

static constexpr int ACL_AICORE_MEMORY_ACCESS = 8;

std::map<std::string, aclprofAicoreMetrics> kAicMetrics = {
  {"ArithmeticUtilization", ACL_AICORE_ARITHMETIC_UTILIZATION},
  {"PipeUtilization", ACL_AICORE_PIPE_UTILIZATION},
  {"Memory", ACL_AICORE_MEMORY_BANDWIDTH},
  {"MemoryL0", ACL_AICORE_L0B_AND_WIDTH},
  {"ResourceConflictRatio", ACL_AICORE_RESOURCE_CONFLICT_RATIO},
  {"MemoryUB", ACL_AICORE_MEMORY_UB},
  {"L2Cache", ACL_AICORE_L2_CACHE},
  {"MemoryAccess", static_cast<aclprofAicoreMetrics>(ACL_AICORE_MEMORY_ACCESS)},
  {"None", ACL_AICORE_NONE}};

std::map<std::string, uint64_t> profLevelMap{
  {"LevelNone", LevelNone}, {"Level0", Level0}, {"Level1", Level1}, {"Level2", Level2}};

std::shared_ptr<AscendProfiler> AscendProfiler::GetInstance() {
  auto instance = Profiler::GetInstance(kAscendDevice);
  MS_EXCEPTION_IF_NULL(instance);
  return std::dynamic_pointer_cast<AscendProfiler>(instance);
}

void AscendProfiler::StepProfilingEnable(const bool enable_flag) {
  enable_flag_ = enable_flag;
  MS_LOG(INFO) << "Step profiling enable flag is " << enable_flag;
}

void AscendProfiler::InitAscendProfilerConfig(const std::string &profiling_path, uint32_t device_id,
                                              const std::string &profiling_options) {
  nlohmann::json options;
  try {
    options = nlohmann::json::parse(profiling_options);
  } catch (nlohmann::json::exception &e) {
    MS_LOG(ERROR) << "Failed to parse profiling options json data, current options is " << options;
    return;
  }

  config_.frameworkDataPath = options["framework_path"];
  config_.deviceId = device_id;
  config_.rankId = options["rank_id"];
  config_.profileMemory = options["profile_memory"];
  config_.l2Cache = options["l2_cache"];
  config_.hbmDdr = options["hbm_ddr"];
  config_.pcie = options["pcie"];
  config_.sysIo = options["sys_io"];
  config_.sysInterconnection = options["sys_interconnection"];
  config_.hostSys = options["host_sys"];
  config_.withStack = options["with_stack"];
  config_.parallelStrategy = options["parallel_strategy"];
  config_.profilerLevel = options["profiler_level"];
  config_.aicoreMetrics = options["aicore_metrics"];
  config_.cpuTrace = options["cpu_trace"];
  config_.npuTrace = options["npu_trace"];
  config_.mstx = options["mstx"];
  config_.recordShapes = options["record_shapes"];
  config_.mstxDomainInclude = options["mstx_domain_include"].get<std::vector<std::string>>();
  config_.mstxDomainExclude = options["mstx_domain_exclude"].get<std::vector<std::string>>();
  config_.outputPath = profiling_path;

  is_parallel_strategy = config_.parallelStrategy;
}

void AscendProfiler::InitAclConfig() {
  aclError ret = CALL_ASCEND_API(aclrtSetDevice, static_cast<int32_t>(config_.deviceId));
  if (ret != ACL_SUCCESS) {
    MS_LOG(ERROR) << "Device " << config_.deviceId
                  << " call aclrtSetDevice failed, error_code : " << static_cast<int>(ret);
  }

  aclError aclRet = CALL_ASCEND_API(aclprofInit, config_.outputPath.c_str(), config_.outputPath.length());
  if (aclRet != ACL_SUCCESS) {
    MS_LOG(ERROR) << "Failed call aclprofInit function. error_code : " << static_cast<int>(aclRet);
  }

  if (config_.profileMemory || config_.hbmDdr) {
    const char *hbmFreq = "50";
    aclError hbmRet = aclprofSetConfig(ACL_PROF_SYS_HARDWARE_MEM_FREQ, hbmFreq, strlen(hbmFreq));
    if (hbmRet != ACL_SUCCESS) {
      MS_LOG(ERROR) << "Failed call aclprofSetConfig to ACL_PROF_SYS_HARDWARE_MEM_FREQ. error_code : "
                    << static_cast<int>(hbmRet);
    }
  }

  if (config_.sysIo) {
    const char *sysIoFreq = "100";
    aclError sysIoRet = aclprofSetConfig(ACL_PROF_SYS_IO_FREQ, sysIoFreq, strlen(sysIoFreq));
    if (sysIoRet != ACL_SUCCESS) {
      MS_LOG(ERROR) << "Failed call aclprofSetConfig to ACL_PROF_SYS_IO_FREQ. error_code : "
                    << static_cast<int>(sysIoRet);
    }
  }

  if (config_.sysInterconnection || config_.pcie) {
    const char *sysInterconnectionFreq = "50";
    aclError sysInterconnectionRet =
      aclprofSetConfig(ACL_PROF_SYS_INTERCONNECTION_FREQ, sysInterconnectionFreq, strlen(sysInterconnectionFreq));
    if (sysInterconnectionRet != ACL_SUCCESS) {
      MS_LOG(ERROR) << "Failed call aclprofSetConfig to ACL_PROF_SYS_INTERCONNECTION_FREQ. error_code : "
                    << static_cast<int>(sysInterconnectionRet);
    }
  }

  aclprofAicoreMetrics aicMetrics = CheckAicMetricsFeature(GetAicMetrics(), config_.profilerLevel);
  uint64_t mask = GetAclProfMask(aicMetrics);
  uint32_t deviceList[1] = {config_.deviceId};
  uint32_t deviceNum = 1;
  aclConfig_ = CALL_ASCEND_API(aclprofCreateConfig, deviceList, deviceNum, aicMetrics, nullptr, mask);
  if (aclConfig_ == nullptr) {
    MS_LOG(ERROR) << "Failed call aclprofCreateConfig function.";
  }
}

aclprofAicoreMetrics AscendProfiler::GetAicMetrics() const {
  aclprofAicoreMetrics aicMetrics = ACL_AICORE_NONE;
  if (kAicMetrics.find(config_.aicoreMetrics) != kAicMetrics.end()) {
    aicMetrics = kAicMetrics[config_.aicoreMetrics];
    MS_LOG(INFO) << "aicore_metrics is " << config_.aicoreMetrics << ", aicMetrics is " << aicMetrics;
  }
  return aicMetrics;
}

aclprofAicoreMetrics AscendProfiler::CheckAicMetricsFeature(aclprofAicoreMetrics aic_metrics,
                                                            const std::string &profiler_level) {
  if (aic_metrics == ACL_AICORE_MEMORY_ACCESS &&
      !FeatureMgr::Instance().IsSupportFeature(FeatureType::FEATURE_MEMORY_ACCESS)) {
    MS_LOG(WARNING) << "AicMetrics is not supported to set to MemoryAccess. "
                    << "When profiler_level is set to " << profiler_level << ", AicMetrics will be set to "
                    << (profiler_level == "Level1" || profiler_level == "Level2" ? "PipeUtilization" : "AicoreNone")
                    << ".";
    return (profiler_level == "Level1" || profiler_level == "Level2") ? ACL_AICORE_PIPE_UTILIZATION : ACL_AICORE_NONE;
  }
  return aic_metrics;
}

uint64_t AscendProfiler::GetAclProfMask(aclprofAicoreMetrics aicMetrics) {
  auto level_iter = profLevelMap.find(config_.profilerLevel);
  uint64_t mask = (level_iter == profLevelMap.end()) ? Level0 : profLevelMap[config_.profilerLevel];
  MS_LOG(INFO) << "profiler_level is " << config_.profilerLevel << ", mask is " << mask;

  if (aicMetrics != ACL_AICORE_NONE) {
    mask |= ACL_PROF_AICORE_METRICS;
    MS_LOG(INFO) << "aicore_metrics is " << config_.aicoreMetrics << ", mask is " << mask;
  }

  if (config_.l2Cache) {
    mask |= ACL_PROF_L2CACHE;
    MS_LOG(INFO) << "l2_cache is enabled, mask is " << mask;
  }

  if (config_.profileMemory) {
    mask |= ACL_PROF_TASK_MEMORY;
    MS_LOG(INFO) << "profile_memory is enabled, mask is " << mask;
  }
  if (config_.mstx) {
    mask |= ACL_PROF_MSPROFTX;
    MS_LOG(INFO) << "mstx is enabled, mask is " << mask;
  }
  return mask;
}

void AscendProfiler::Init(const std::string &profiling_path, uint32_t device_id, const std::string &profiling_options) {
  MS_LOG(INFO) << "Init AscendProfiler";
  device::ascend::AscendHalManager::GetInstance().InitializeAcl();
  (void)ErrorManagerAdapter::Init();
  InitAscendProfilerConfig(profiling_path, device_id, profiling_options);
  FeatureMgr::Instance().Init();

  if (config_.cpuTrace) {
    ProfilingFrameworkData::Device_Id = config_.rankId;
    ProfilingDataDumper::GetInstance().Init(config_.frameworkDataPath);
    profiler::ascend::ParallelStrategy::GetInstance()->SetOutputPath(config_.frameworkDataPath);
    InitFwkMemProfiling();
    MS_LOG(INFO) << "cpu_trace is enabled";
  }

  if (config_.npuTrace) {
    InitAclConfig();
    MS_LOG(INFO) << "npu_trace is enabled";
  }
  init_flag_ = true;
}

void AscendProfiler::InitFwkMemProfiling() {
  if (config_.profileMemory) {
    auto msContext = MsContext::GetInstance();
    msContext->set_param<std::string>(MS_CTX_PROF_MEM_OUTPUT_PATH, config_.frameworkDataPath);
    MS_LOG(INFO) << "profile_memory enabled, save path:" << config_.frameworkDataPath;
  }
}

void AscendProfiler::StartFwkMemProfiling() {
  if (config_.profileMemory) {
    device::ascend::AscendMemoryPool::GetInstance().SetEnableTimeEvent(true);
    device::ascend::AscendMemoryPool::SetEnhancedMemoryPool(true);
    MS_LOG(INFO) << "Start framework memory profiling";
  }
}

void AscendProfiler::StopFwkMemProfiling() {
  if (config_.profileMemory) {
    device::ascend::AscendMemoryPool::GetInstance().SetEnableTimeEvent(false);
    if (!memory::mem_pool::IsEnableMemTrack()) {
      device::ascend::AscendMemoryPool::SetEnhancedMemoryPool(false);
    }
    MS_LOG(INFO) << "Stop framework memory profiling, save path: " << config_.frameworkDataPath;
  }
}

void AscendProfiler::Start() {
  MS_LOG(INFO) << "Start AscendProfiler begin";

  if (!config_.hostSys.empty()) {
    aclError hostSysRet = aclprofSetConfig(ACL_PROF_HOST_SYS, config_.hostSys.c_str(), strlen(config_.hostSys.c_str()));
    if (hostSysRet != ACL_SUCCESS) {
      MS_LOG(ERROR) << "Failed call aclprofSetConfig to ACL_PROF_HOST_SYS. error_code : "
                    << static_cast<int>(hostSysRet);
    }
  }

  if (config_.npuTrace) {
    aclError aclRet = CALL_ASCEND_API(aclprofStart, aclConfig_);
    if (aclRet != ACL_SUCCESS) {
      MS_LOG(ERROR) << "Failed call aclprofStart function. error_code : " << static_cast<int>(aclRet);
    }
    if (config_.mstx) {
      MstxDispatcher::GetInstance().Enable();
      MstxDispatcher::GetInstance().SetDomain(config_.mstxDomainInclude, config_.mstxDomainExclude);
    }
    MS_LOG(INFO) << "Start AscendProfiler npu trace";
  }

  if (config_.cpuTrace) {
    ProfilingDataDumper::GetInstance().Start();
    StartFwkMemProfiling();
    if (config_.withStack) {
      pybind11::gil_scoped_acquire gil;
      PythonTracer::call(Command::kStartOne, config_.rankId);
    }
    MS_LOG(INFO) << "Start AscendProfiler cpu trace";
  }
  StepProfilingEnable(true);
  MS_LOG(INFO) << "Start AscendProfiler end";
}

void AscendProfiler::Stop() {
  MS_LOG(INFO) << "Stop AscendProfiler begin";

  if (config_.npuTrace) {
    if (config_.mstx) {
      MstxDispatcher::GetInstance().Disable();
    }
    aclError aclRet = CALL_ASCEND_API(aclprofStop, aclConfig_);
    if (aclRet != ACL_SUCCESS) {
      MS_LOG(ERROR) << "Failed call aclprofStop function. error_code : " << static_cast<int>(aclRet);
    }
    MS_LOG(INFO) << "Stop AscendProfiler npu trace";
  }

  if (config_.cpuTrace) {
    StopFwkMemProfiling();
    if (config_.withStack) {
      pybind11::gil_scoped_acquire gil;
      PythonTracer::call(Command::kStop, config_.rankId);
    }
    profiler::ascend::ParallelStrategy::GetInstance()->SaveParallelStrategyToFile();
    ProfilingDataDumper::GetInstance().Stop();
    ProfilingDataDumper::GetInstance().UnInit();
    MS_LOG(INFO) << "Stop AscendProfiler cpu trace";
  }
  StepProfilingEnable(false);
  MS_LOG(INFO) << "Stop AscendProfiler end";
}

void AscendProfiler::Finalize() {
  MS_LOG(INFO) << "Finalize AscendProfiler begin";

  if (aclConfig_ != nullptr) {
    aclError aclRetDestroy = CALL_ASCEND_API(aclprofDestroyConfig, aclConfig_);
    if (aclRetDestroy != ACL_SUCCESS) {
      MS_LOG(ERROR) << "Failed to call aclprofDestoryConfig function. error_code : " << static_cast<int>(aclRetDestroy);
    }

    aclError aclRetFinalize = CALL_ASCEND_API(aclprofFinalize);
    if (aclRetFinalize != ACL_SUCCESS) {
      MS_LOG(ERROR) << "Failed to call aclprofFinalize function. error_code : " << static_cast<int>(aclRetFinalize);
    }
    MS_LOG(INFO) << "Finalize AscendProfiler aclprofFinalize";
  }

  if (config_.cpuTrace) {
    profiler::ascend::ParallelStrategy::GetInstance()->ClearOutputPath();
  }
  ClearInst();
  MS_LOG(INFO) << "Finalize AscendProfiler end";
}

struct aclprofStepInfoInner {
  bool startFlag;
  bool endFlag;
  uint64_t indexId;
};

void AscendProfiler::StepStart(uint64_t step_id, void *stream) {
  aclStream_ = static_cast<aclrtStream>(stream);
  aclProfStepInfo_ = CALL_ASCEND_API(aclprofCreateStepInfo);
  aclprofStepInfoInner *ptr_info = reinterpret_cast<aclprofStepInfoInner *>(aclProfStepInfo_);
  ptr_info->indexId = step_id;
  auto ret =
    CALL_ASCEND_API(aclprofGetStepTimestamp, aclProfStepInfo_, (aclprofStepTag)kAclProfStepStartTag, aclStream_);
  if (ret != ACL_SUCCESS) {
    MS_LOG(WARNING) << "Failed to call aclprofGetStepTimestamp with tag " << kAclProfStepStartTag << ".";
  }
}

void AscendProfiler::StepStop() {
  auto ret = CALL_ASCEND_API(aclprofGetStepTimestamp, aclProfStepInfo_, (aclprofStepTag)kAclProfStepEndTag, aclStream_);
  if (ret != ACL_SUCCESS) {
    MS_LOG(WARNING) << "Failed to call aclprofGetStepTimestamp with tag " << kAclProfStepEndTag << ".";
  }
  if (aclProfStepInfo_ != nullptr) {
    CALL_ASCEND_API(aclprofDestroyStepInfo, aclProfStepInfo_);
    aclProfStepInfo_ = nullptr;
  }
  aclStream_ = nullptr;
}

void AscendProfiler::MstxMark(const std::string &message, void *stream, const std::string &domain_name) {
  MS_LOG(INFO) << "Ascend mstx mark, message: " << message;
  MstxDispatcher::GetInstance().Mark(message.c_str(), stream, domain_name);
}

int AscendProfiler::MstxRangeStart(const std::string &message, void *stream, const std::string &domain_name) {
  MS_LOG(INFO) << "Ascend mstx range start, message: " << message;
  return MstxDispatcher::GetInstance().RangeStart(message.c_str(), stream, domain_name);
}

void AscendProfiler::MstxRangeEnd(int range_id, const std::string &domain_name) {
  MS_LOG(INFO) << "Ascend mstx range end, range_id: " << range_id;
  MstxDispatcher::GetInstance().RangeEnd(range_id, domain_name);
}

bool AscendProfiler::EnableRecordShapes() { return config_.recordShapes; }

}  // namespace ascend
}  // namespace profiler
}  // namespace mindspore
