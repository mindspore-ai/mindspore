/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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

#include "utils/ms_context.h"
#include <thread>
#include <atomic>
#include <fstream>
#include <algorithm>
#include <utility>
#include "ir/tensor.h"
#include "utils/ms_utils.h"

namespace mindspore {
std::atomic<bool> thread_1_must_end(false);

MsContext::DeviceSeter MsContext::seter_ = nullptr;
MsContext::DeviceTypeSeter MsContext::device_type_seter_ = nullptr;
std::shared_ptr<MsContext> MsContext::inst_context_ = nullptr;
std::map<std::string, MsBackendPolicy> MsContext::policy_map_ = {{"ge", kMsBackendGePrior},
                                                                 {"vm", kMsBackendVmOnly},
                                                                 {"ms", kMsBackendMsPrior},
                                                                 {"ge_only", kMsBackendGeOnly},
                                                                 {"vm_prior", kMsBackendVmPrior}};

std::map<MsCtxParam, std::string> kUnresetParamCheckList = {
  {MsCtxParam::MS_CTX_DEVICE_ID, "device_id"},
  {MsCtxParam::MS_CTX_VARIABLE_MEMORY_MAX_SIZE, "variable_memory_max_size"},
  {MsCtxParam::MS_CTX_MAX_DEVICE_MEMORY, "max_device_memory"},
  {MsCtxParam::MS_CTX_MEMPOOL_BLOCK_SIZE, "mempool_block_size"}};

MsContext::MsContext(const std::string &policy, const std::string &target) {
#ifndef ENABLE_SECURITY
  set_param<bool>(MS_CTX_SAVE_GRAPHS_FLAG, false);
  set_param<std::string>(MS_CTX_SAVE_GRAPHS_PATH, ".");
  set_param<std::string>(MS_CTX_COMPILE_CACHE_PATH, "");
#else
  // Need set a default value for arrays even if running in the security mode.
  bool_params_[MS_CTX_SAVE_GRAPHS_FLAG - MS_CTX_TYPE_BOOL_BEGIN] = false;
  string_params_[MS_CTX_SAVE_GRAPHS_PATH - MS_CTX_TYPE_STRING_BEGIN] = ".";
#endif
  set_param<std::string>(MS_CTX_PYTHON_EXE_PATH, "python");
  set_param<std::string>(MS_CTX_KERNEL_BUILD_SERVER_DIR, "");
  set_param<bool>(MS_CTX_ENABLE_DUMP, false);
  set_param<std::string>(MS_CTX_SAVE_DUMP_PATH, ".");
  set_param<std::string>(MS_CTX_ENV_CONFIG_PATH, "");
  set_param<std::string>(MS_CTX_TUNE_MODE, "NO_TUNE");
  set_param<std::string>(MS_CTX_GRAPH_KERNEL_FLAGS, "");
  set_param<uint32_t>(MS_CTX_TSD_REF, 0);
  set_param<uint32_t>(MS_CTX_GE_REF, 0);

  set_param<bool>(MS_CTX_IS_MULTI_GRAPH_SINK, false);
  set_param<bool>(MS_CTX_IS_PYNATIVE_GE_INIT, false);
  set_param<bool>(MS_CTX_ENABLE_REDUCE_PRECISION, true);
  auto env_device = common::GetEnv("DEVICE_ID");
  if (!env_device.empty()) {
    try {
      uint32_t device_id = UlongToUint(std::stoul(env_device));
      set_param<uint32_t>(MS_CTX_DEVICE_ID, device_id);
    } catch (std::invalid_argument &e) {
      MS_LOG(WARNING) << "Invalid DEVICE_ID env:" << env_device << ". Please set DEVICE_ID to 0-7";
      set_param<uint32_t>(MS_CTX_DEVICE_ID, 0);
    }
  } else {
    set_param<uint32_t>(MS_CTX_DEVICE_ID, 0);
  }

  set_param<uint32_t>(MS_CTX_MAX_CALL_DEPTH, MAX_CALL_DEPTH_DEFAULT);
  set_param<std::string>(MS_CTX_DEVICE_TARGET, target);
  set_param<int>(MS_CTX_EXECUTION_MODE, kGraphMode);
  set_param<bool>(MS_CTX_ENABLE_TASK_SINK, true);
  set_param<bool>(MS_CTX_IR_FUSION_FLAG, true);
  set_param<bool>(MS_CTX_ENABLE_HCCL, false);
  set_param<bool>(MS_CTX_ENABLE_GPU_SUMMARY, true);
  set_param<bool>(MS_CTX_PRECOMPILE_ONLY, false);
  set_param<bool>(MS_CTX_ENABLE_PYNATIVE_INFER, false);
  set_param<bool>(MS_CTX_ENABLE_PYNATIVE_HOOK, false);
  set_param<bool>(MS_CTX_ENABLE_DYNAMIC_MEM_POOL, true);
  set_param<std::string>(MS_CTX_GRAPH_MEMORY_MAX_SIZE, "0");
  set_param<std::string>(MS_CTX_VARIABLE_MEMORY_MAX_SIZE, "0");
  set_param<bool>(MS_CTX_ENABLE_LOOP_SINK, target == kAscendDevice || target == kDavinciDevice);
  set_param<bool>(MS_CTX_ENABLE_PROFILING, false);
  set_param<std::string>(MS_CTX_PROFILING_OPTIONS, "training_trace");
  set_param<bool>(MS_CTX_CHECK_BPROP_FLAG, false);
  set_param<float>(MS_CTX_MAX_DEVICE_MEMORY, kDefaultMaxDeviceMemory);
  set_param<float>(MS_CTX_MEMPOOL_BLOCK_SIZE, kDefaultMempoolBlockSize);
  set_param<std::string>(MS_CTX_PRINT_FILE_PATH, "");
  set_param<bool>(MS_CTX_ENABLE_GRAPH_KERNEL, false);
  set_param<bool>(MS_CTX_ENABLE_PARALLEL_SPLIT, false);
  set_param<bool>(MS_CTX_ENABLE_INFER_OPT, false);
  set_param<bool>(MS_CTX_GRAD_FOR_SCALAR, false);
  set_param<bool>(MS_CTX_ENABLE_MINDRT, false);
  set_param<bool>(MS_CTX_ENABLE_PYNATIVE_SYNCHRONIZE, false);
  set_param<bool>(MS_CTX_ENABLE_PYNATIVE_OP_GRAPH_CACHE, true);
  set_param<bool>(MS_CTX_ENABLE_MEM_SCHEDULER, false);
  set_param<bool>(MS_CTX_ENABLE_RECOVERY, false);
  set_param<bool>(MS_CTX_ENABLE_GE_HETEROGENOUS, false);
  set_param<bool>(MS_CTX_DISABLE_FORMAT_TRANSFORM, false);
  set_param<int>(MS_CTX_MEMORY_OPTIMIZE_LEVEL, kOptimizeO0);
  set_param<uint32_t>(MS_CTX_OP_TIMEOUT, kOpTimeout);

  uint32_t kDefaultRuntimeNumThreads = 30;
  uint32_t cpu_core_num = std::thread::hardware_concurrency() - 1;
  uint32_t runtime_num_threads_default = std::min(cpu_core_num, kDefaultRuntimeNumThreads);
  set_param<uint32_t>(MS_CTX_RUNTIME_NUM_THREADS, runtime_num_threads_default);

  backend_policy_ = policy_map_[policy];

  params_read_status_ = std::vector<bool>(
    static_cast<size_t>(MsCtxParam::NUM_BOOL_PARAMS + MsCtxParam::NUM_UINT32_PARAMS + MsCtxParam::NUM_INT_PARAMS +
                        MsCtxParam::NUM_FLOAT_PARAMS + MsCtxParam::NUM_STRING_PARAMS),
    false);
}

std::shared_ptr<MsContext> MsContext::GetInstance() {
  if (inst_context_ == nullptr) {
    MS_LOG(DEBUG) << "Create new mindspore context";
    if (device_type_seter_) {
      device_type_seter_(inst_context_);
    }
  }
  return inst_context_;
}

bool MsContext::set_backend_policy(const std::string &policy) {
  auto policy_new = policy;
#ifdef ENABLE_D
  auto enable_ge = mindspore::common::GetEnv("MS_ENABLE_GE");
  if (enable_ge == "1") {
    policy_new = "ge";
  }
#endif
  if (policy_map_.find(policy_new) == policy_map_.end()) {
    MS_LOG(ERROR) << "invalid backend policy name: " << policy_new;
    return false;
  }
  backend_policy_ = policy_map_[policy_new];
  MS_LOG(INFO) << "ms set context backend policy:" << policy_new;
  return true;
}

std::string MsContext::backend_policy() const {
  auto res = std::find_if(
    policy_map_.begin(), policy_map_.end(),
    [&, this](const std::pair<std::string, MsBackendPolicy> &item) { return item.second == backend_policy_; });
  if (res != policy_map_.end()) {
    return res->first;
  }
  return "unknown";
}

bool MsContext::enable_dump_ir() const {
#ifdef ENABLE_DUMP_IR
  return true;
#else
  return false;
#endif
}

void MsContext::MarkReadStatus(MsCtxParam param) const {
#if !(defined(ENABLE_TEST) || defined(ENABLE_TESTCASES) || defined(BUILD_LITE))
  // unit tests will set device_id many times in one process
  if (static_cast<size_t>(param) < params_read_status_.size()) {
    params_read_status_[static_cast<size_t>(param)] = true;
  }
#endif
}

template <typename T>
void MsContext::CheckReadStatus(MsCtxParam param, const T &value) const {
#if !(defined(ENABLE_TEST) || defined(ENABLE_TESTCASES) || defined(BUILD_LITE))
  // unit tests will set device_id many times in one process
  if (static_cast<size_t>(param) >= params_read_status_.size()) {
    return;
  }
  auto iter = kUnresetParamCheckList.find(param);
  if (iter == kUnresetParamCheckList.end()) {
    return;
  }
  auto origin_status = params_read_status_;
  T origin_value = get_param<T>(param);
  params_read_status_ = origin_status;
  if (params_read_status_[static_cast<size_t>(param)] && value != origin_value) {
    MS_EXCEPTION(TypeError) << "For 'set_context', the parameter " << iter->second
                            << " can not be set repeatedly, origin value [" << origin_value << "] has been in effect.";
  }
#endif
}
template MS_CORE_API void MsContext::CheckReadStatus<bool>(MsCtxParam, const bool &) const;
template MS_CORE_API void MsContext::CheckReadStatus<uint32_t>(MsCtxParam, const uint32_t &) const;
template MS_CORE_API void MsContext::CheckReadStatus<int>(MsCtxParam, const int &) const;
template MS_CORE_API void MsContext::CheckReadStatus<float>(MsCtxParam, const float &) const;
template MS_CORE_API void MsContext::CheckReadStatus<std::string>(MsCtxParam, const std::string &) const;
}  // namespace mindspore
