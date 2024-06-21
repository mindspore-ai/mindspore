/**
 * Copyright 2019-2023 Huawei Technologies Co., Ltd
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
#include <string>
#include <thread>
#include <atomic>
#include <fstream>
#include <algorithm>
#include <utility>
#include "utils/ms_utils.h"
#include "include/common/utils/utils.h"
#include "utils/convert_utils_base.h"
#include "utils/phase.h"

#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace mindspore {
namespace {
std::map<std::string, MsBackendPolicy> kPolicyMap = {{"ge", kMsBackendGePrior},     {"bisheng", kMsBackendBishengPrior},
                                                     {"vm", kMsBackendVmOnly},      {"ms", kMsBackendMsPrior},
                                                     {"ge_only", kMsBackendGeOnly}, {"vm_prior", kMsBackendVmPrior}};

constexpr auto kDeviceTargetSize2 = 2;
}  // namespace
std::atomic<bool> thread_1_must_end(false);

MsContext::DeviceSeter MsContext::seter_ = nullptr;
MsContext::LoadPluginError MsContext::load_plugin_error_ = nullptr;
std::shared_ptr<MsContext> MsContext::inst_context_ = nullptr;

std::map<MsCtxParam, std::string> kUnresetParamCheckList = {
  {MsCtxParam::MS_CTX_DEVICE_ID, "device_id"},
  {MsCtxParam::MS_CTX_VARIABLE_MEMORY_MAX_SIZE, "variable_memory_max_size"},
  {MsCtxParam::MS_CTX_MAX_DEVICE_MEMORY, "max_device_memory"},
  {MsCtxParam::MS_CTX_MEMPOOL_BLOCK_SIZE, "mempool_block_size"}};

MsContext::MsContext(const std::string &policy, const std::string &target) {
#ifndef ENABLE_SECURITY
  set_param<int>(MS_CTX_SAVE_GRAPHS_FLAG, 0);
  set_param<std::string>(MS_CTX_SAVE_GRAPHS_PATH, ".");
  set_param<bool>(MS_CTX_ENABLE_COMPILE_CACHE, false);
  set_param<std::string>(MS_CTX_COMPILE_CACHE_PATH, "");
#else
  // Need set a default value for arrays even if running in the security mode.
  int_params_[MS_CTX_SAVE_GRAPHS_FLAG - MS_CTX_TYPE_BOOL_BEGIN] = 0;
  string_params_[MS_CTX_SAVE_GRAPHS_PATH - MS_CTX_TYPE_STRING_BEGIN] = ".";
#endif
  InitBoolTypeDefaultValue();
  InitStringTypeDefaultValue();
  InitDigitalTypeDefaultValue();
  MsContext::SetDeviceId();
  string_params_[MS_CTX_DEVICE_TARGET - MS_CTX_TYPE_STRING_BEGIN] = target;
  set_param<bool>(MS_CTX_ENABLE_LOOP_SINK, target == kAscendDevice || target == kDavinciDevice);

  backend_policy_ = kPolicyMap[policy];
  ascend_soc_version_ = "";

  params_read_status_ = std::vector<bool>(
    static_cast<size_t>(MsCtxParam::NUM_BOOL_PARAMS + MsCtxParam::NUM_UINT32_PARAMS + MsCtxParam::NUM_INT_PARAMS +
                        MsCtxParam::NUM_FLOAT_PARAMS + MsCtxParam::NUM_STRING_PARAMS),
    false);
  params_write_status_ = std::vector<bool>(
    static_cast<size_t>(MsCtxParam::NUM_BOOL_PARAMS + MsCtxParam::NUM_UINT32_PARAMS + MsCtxParam::NUM_INT_PARAMS +
                        MsCtxParam::NUM_FLOAT_PARAMS + MsCtxParam::NUM_STRING_PARAMS),
    false);

  SetAscendConfig();
}

std::shared_ptr<MsContext> MsContext::GetInstance() {
  static std::once_flag inst_context_init_flag_ = {};
  std::call_once(inst_context_init_flag_, [&]() {
    if (inst_context_ == nullptr) {
      MS_LOG(DEBUG) << "Create new mindspore context";
      inst_context_ = std::make_shared<MsContext>("vm", kCPUDevice);
    }
  });
  MS_EXCEPTION_IF_NULL(inst_context_);
  return inst_context_;
}

void MsContext::SetDeviceId() {
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
}

void MsContext::Refresh() {
  RefreshExecutionMode();
  RefreshMemoryOffload();
}

void MsContext::RefreshExecutionMode() {
  const std::string &target = get_param<std::string>(MS_CTX_DEVICE_TARGET);
  if (target == kAscendDevice) {
    if (MsContext::GetInstance()->get_param<int>(MS_CTX_EXECUTION_MODE) == kPynativeMode) {
      set_param<bool>(MS_CTX_IS_MULTI_GRAPH_SINK, false);
    } else if (IsKByKExecutorMode()) {
      set_param<bool>(MS_CTX_ENABLE_TASK_SINK, false);
    }
  }
}

void MsContext::RefreshMemoryOffload() {
  const bool enable_mem_offload = get_param<bool>(MS_CTX_ENABLE_MEM_OFFLOAD);
  if (!enable_mem_offload) {
    return;
  }
  const std::string &target = get_param<std::string>(MS_CTX_DEVICE_TARGET);
  if (target == kCPUDevice) {
    MS_LOG(WARNING) << "Memory offload is not available on CPU device.";
    set_param(MS_CTX_ENABLE_MEM_OFFLOAD, false);
    return;
  }
  if (target == kAscendDevice && get_param<int>(MS_CTX_EXECUTION_MODE) != kPynativeMode && !IsKByKExecutorMode()) {
    MS_LOG(WARNING) << "Run graph mode with kernel by kernel because memory offload is ON.";
    set_param<bool>(MS_CTX_ENABLE_TASK_SINK, false);
    return;
  }
  if (get_param<int>(MS_CTX_MEMORY_OPTIMIZE_LEVEL) == kOptimizeO1) {
    MS_LOG(WARNING) << "Memory offload is not available when memory_optimize_level is set to O1.";
    set_param(MS_CTX_ENABLE_MEM_OFFLOAD, false);
    return;
  }
  MS_LOG(INFO) << "Set memory pool block size to max device memory size for memory offload.";
  set_param_inner(MS_CTX_MEMPOOL_BLOCK_SIZE, get_param<float>(MS_CTX_MAX_DEVICE_MEMORY));
}

bool MsContext::set_backend_policy(const std::string &policy) {
  auto iter = kPolicyMap.find(policy);
  if (iter == kPolicyMap.end()) {
    MS_LOG(ERROR) << "invalid backend policy name: " << policy;
    return false;
  }
  backend_policy_ = iter->second;
  MS_LOG(INFO) << "ms set context backend policy:" << policy;
  return true;
}

std::string MsContext::backend_policy() const {
  auto res = std::find_if(
    kPolicyMap.begin(), kPolicyMap.end(),
    [&, this](const std::pair<std::string, MsBackendPolicy> &item) { return item.second == backend_policy_; });
  if (res != kPolicyMap.end()) {
    return res->first;
  }
  return "unknown";
}

void MsContext::set_ascend_soc_name(const std::string &soc_name) { ascend_soc_name_ = soc_name; }

std::string MsContext::ascend_soc_name() const { return ascend_soc_name_; }

bool MsContext::set_ascend_soc_version(const std::string &soc_version) {
  ascend_soc_version_ = soc_version;
  return true;
}

std::string MsContext::ascend_soc_version() const { return ascend_soc_version_; }

bool MsContext::enable_dump_ir() const {
#ifdef ENABLE_DUMP_IR
  return true;
#else
  return false;
#endif
}

std::map<std::string, MsContext::InitDeviceTargetAndPolicy> &MsContext::InitFuncMap() {
  static std::map<std::string, InitDeviceTargetAndPolicy> init_func_map = {};
  return init_func_map;
}

std::map<std::string, std::string> &MsContext::PluginPathMap() {
  static std::map<std::string, std::string> plugin_path_map = {};
  return plugin_path_map;
}

void MsContext::RegisterInitFunc(const std::string &name, MsContext::InitDeviceTargetAndPolicy func) {
  (void)InitFuncMap().emplace(name, func);
  if (GetInstance() != nullptr) {
    GetInstance()->SetDefaultDeviceTarget();
  }
  std::string plugin_path;
#if !defined(_WIN32) && !defined(_WIN64)
  Dl_info dl_info;
  if (dladdr(reinterpret_cast<void *>(func), &dl_info) == 0) {
    MS_LOG(EXCEPTION) << "Get dladdr error for " << name;
  }
  plugin_path = dl_info.dli_fname;
#else
  HMODULE h_module = nullptr;
  if (GetModuleHandleEx(GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT | GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS,
                        (LPCSTR)func, &h_module) == 0) {
    MS_LOG(EXCEPTION) << "Get GetModuleHandleEx failed for " << name;
  }
  char sz_path[MAX_PATH];
  if (GetModuleFileName(h_module, sz_path, sizeof(sz_path)) == 0) {
    MS_LOG(EXCEPTION) << "Get GetModuleFileName failed for " << name;
  }
  plugin_path = std::string(sz_path);
#endif
  (void)PluginPathMap().emplace(name, plugin_path);
}

void MsContext::ResisterLoadPluginErrorFunc(MsContext::LoadPluginError func) { load_plugin_error_ = func; }

bool MsContext::IsAscendPluginLoaded() const {
#ifdef WITH_BACKEND
  return InitFuncMap().find("Ascend") != InitFuncMap().end();
#else
  // for ut test
  return true;
#endif
}

void MsContext::SetDefaultDeviceTarget() {
  auto cpu_iter = InitFuncMap().find(kCPUDevice);
  if (cpu_iter == InitFuncMap().end()) {
    return;
  }
  if (InitFuncMap().size() == 1) {
    // when only cpu in map
    cpu_iter->second(inst_context_.get());
  } else if (InitFuncMap().size() == kDeviceTargetSize2) {
    // when cpu and another in map
    for (auto [name, func] : InitFuncMap()) {
      if (name != kCPUDevice) {
        inst_context_ = std::make_shared<MsContext>("ms", name);
        func(inst_context_.get());
      }
    }
  } else {
    cpu_iter->second(inst_context_.get());
  }
  default_device_target_ = true;
}

void MsContext::SetDeviceTargetFromInner(const std::string &device_target) {
  if (seter_ != nullptr) {
    if (!InitFuncMap().empty()) {
      if (auto iter = InitFuncMap().find(device_target); iter == InitFuncMap().end()) {
        CheckEnv(device_target);
        std::string device_list = "[";
        for (auto citer = InitFuncMap().cbegin(); citer != InitFuncMap().cend(); ++citer) {
          if (device_list == "[") {
            device_list += "\'" + citer->first + "\'";
          } else {
            device_list += ", \'" + citer->first + "\'";
          }
        }
        device_list += "]";
        if (load_plugin_error_ != nullptr) {
          auto load_plugin_error_str = load_plugin_error_();
          if (!load_plugin_error_str.empty()) {
            MS_EXCEPTION(RuntimeError) << "Unsupported device target " << device_target
                                       << ". This process only supports one of the " << device_list
                                       << ". Please check whether the " << device_target
                                       << " environment is installed and configured correctly, and check whether "
                                          "current mindspore wheel package was built with \"-e "
                                       << device_target
                                       << "\". For details, please refer to \"Device load error message\"." << std::endl
                                       << "#umsg#Device load error message:#umsg#" << load_plugin_error_str;
          }
        }
        MS_EXCEPTION(RuntimeError) << "Unsupported device target " << device_target
                                   << ". This process only supports one of the " << device_list
                                   << ". Please check whether the " << device_target
                                   << " environment is installed and configured correctly, and check whether "
                                      "current mindspore wheel package was built with \"-e "
                                   << device_target << "\".";
      } else {
        iter->second(this);
        SetEnv(device_target);
      }
    }
    MS_LOG(INFO) << "ms set context device target:" << device_target;
    seter_(device_target);
  }
  if (!CheckWriteStatus(MS_CTX_MEMORY_OPTIMIZE_LEVEL)) {
    MS_LOG(INFO) << "Set memory_optimize_level to O0 as default.";
    int_params_[MS_CTX_MEMORY_OPTIMIZE_LEVEL - MS_CTX_TYPE_INT_BEGIN] = kOptimizeO0;
  }
  string_params_[MS_CTX_DEVICE_TARGET - MS_CTX_TYPE_STRING_BEGIN] = device_target;
}

void MsContext::SetDeviceTargetFromUser(const std::string &device_target) {
  SetDeviceTargetFromInner(device_target);
  default_device_target_ = false;
}

bool MsContext::IsDefaultDeviceTarget() const { return default_device_target_; }

void MsContext::RegisterSetEnv(const EnvFunc &func) { set_env_ = func; }
void MsContext::RegisterCheckEnv(const EnvFunc &func) { check_env_ = func; }

void MsContext::SetEnv(const std::string &device) {
  if (set_env_ == nullptr) {
    return;
  }

  if (auto iter = PluginPathMap().find(device); iter != PluginPathMap().end()) {
    const auto &library_path = iter->second;
    set_env_(device, library_path);
  }
}

void MsContext::CheckEnv(const std::string &device) {
  if (check_env_ == nullptr) {
    return;
  }

  check_env_(device, "");
}

std::string MsContext::GetSaveGraphsPath() const {
  std::string path = common::GetEnv("MS_DEV_SAVE_GRAPHS_PATH");
  if (!path.empty()) {
    return path;
  } else {
    return MsContext::GetInstance()->get_param<std::string>(MS_CTX_SAVE_GRAPHS_PATH);
  }
}

int MsContext::GetSaveGraphsLevel() const {
  static std::string save_env = common::GetEnv("MS_DEV_SAVE_GRAPHS");
  if (save_env.size() == 1) {
    int save_graphs_by_env = -1;
    try {
      save_graphs_by_env = std::stoi(save_env);
    } catch (const std::invalid_argument &ia) {
      MS_LOG(EXCEPTION) << "Invalid argument: " << ia.what() << " when parse " << save_env;
    }
    if (save_graphs_by_env < 0 || save_graphs_by_env > kFully) {
      MS_LOG(EXCEPTION) << "Dump level can only be from 0 to 3";
    }
    return save_graphs_by_env;
  } else if (save_env.size() > 1) {
    MS_LOG(EXCEPTION) << "MS_DEV_SAVE_GRAPHS should be a single number with one digit.";
  }
  return MsContext::GetInstance()->get_param<int>(MS_CTX_SAVE_GRAPHS_FLAG);
}

bool MsContext::CanDump(const DumpLevel &level) const { return GetSaveGraphsLevel() >= level; }

void MsContext::MarkReadStatus(MsCtxParam param) const {
#if !(defined(ENABLE_TEST) || defined(ENABLE_TESTCASES) || defined(BUILD_LITE))
  // unit tests will set device_id many times in one process
  if (static_cast<size_t>(param) < params_read_status_.size()) {
    params_read_status_[static_cast<size_t>(param)] = true;
  }
#endif
}

void MsContext::MarkWriteStatus(MsCtxParam param) const {
  if (static_cast<size_t>(param) < params_write_status_.size()) {
    params_write_status_[static_cast<size_t>(param)] = true;
  }
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
                            << " can not be set repeatedly, origin value [" << origin_value << "] has been in effect."
                            << " Maybe 'mindspore.communication.init()' has been called before 'set_context()'.";
  }
#endif
}

bool MsContext::CheckWriteStatus(MsCtxParam param) const {
  if (static_cast<size_t>(param) >= params_write_status_.size()) {
    return false;
  }
  return params_write_status_[static_cast<size_t>(param)];
}

// Reset ms context. Only called in child process after fork occurs.
void MsContext::ChildAfterFork() {
  MS_LOG(DEBUG) << "Reset context after fork.";
  // configs can be modified again.
  params_read_status_ = std::vector<bool>(
    static_cast<size_t>(MsCtxParam::NUM_BOOL_PARAMS + MsCtxParam::NUM_UINT32_PARAMS + MsCtxParam::NUM_INT_PARAMS +
                        MsCtxParam::NUM_FLOAT_PARAMS + MsCtxParam::NUM_STRING_PARAMS),
    false);
  params_write_status_ = std::vector<bool>(
    static_cast<size_t>(MsCtxParam::NUM_BOOL_PARAMS + MsCtxParam::NUM_UINT32_PARAMS + MsCtxParam::NUM_INT_PARAMS +
                        MsCtxParam::NUM_FLOAT_PARAMS + MsCtxParam::NUM_STRING_PARAMS),
    false);
  std::string device_target_ = get_param<std::string>(MS_CTX_DEVICE_TARGET);
  if (device_target_ != kCPUDevice) {
    // set device_target to 'CPU' as default.
    MS_LOG(INFO) << "Process " << getpid() << " config changed: 'device_target' is reset to 'CPU'.";
    SetDeviceTargetFromUser("CPU");
  }
}

bool MsContext::EnableAoeOnline() const {
  std::string aoe_tune_mode = MsContext::GetInstance()->get_param<std::string>(MS_CTX_AOE_TUNE_MODE);
  return aoe_tune_mode == "online";
}

bool MsContext::EnableAoeOffline() const {
  std::string aoe_tune_mode = MsContext::GetInstance()->get_param<std::string>(MS_CTX_AOE_TUNE_MODE);
  return aoe_tune_mode == "offline";
}

namespace {
void PrintJitLevelAndExecMode(bool is_jit_level_changed, const std::string &jit_level, const std::string &exec_mode) {
  if (!is_jit_level_changed) {
    return;
  }

  MS_LOG(INFO) << "The jit_level is: " << jit_level << ", and " << exec_mode;
  static std::string is_enable_runtime_cfg = common::GetEnv("MS_DEV_RUNTIME_CONF");
  if (!is_enable_runtime_cfg.empty()) {
    std::cout << "[MS_RUNTIME_PROF]The jit_level is: " << jit_level << ", and " << exec_mode << std::endl;
  }
}
}  // namespace

void MsContext::SetJitLevel(const std::string &jit_level) const {
  if (jit_level.empty()) {
    return;
  }
  std::map<std::string, std::string> jit_config = PhaseManager::GetInstance().jit_config();
  jit_config["jit_level"] = jit_level;
  PhaseManager::GetInstance().set_jit_config(jit_config);
}

std::string MsContext::GetJitLevel() const {
  const auto &jit_config = PhaseManager::GetInstance().jit_config();
  std::string jit_level = "";
  auto iter = jit_config.find("jit_level");
  if (iter != jit_config.end()) {
    jit_level = iter->second;
  }

  auto global_jit_level = get_param<std::string>(MS_CTX_JIT_LEVEL);
  auto mode = get_param<int>(MS_CTX_EXECUTION_MODE);
  auto device_target = get_param<std::string>(MS_CTX_DEVICE_TARGET);
  if (jit_level.empty()) {
    if (!global_jit_level.empty()) {
      jit_level = global_jit_level;
    } else if (device_target == kAscendDevice && mode == kGraphMode) {
      jit_level = kAttrJitLevelO2;
    } else {
      jit_level = kAttrJitLevelO0;
    }
  }

  if (mode == kPynativeMode && jit_level == kAttrJitLevelO2) {
    MS_LOG(WARNING) << "Pynative mode can not set jit_level to O2, use O0 instead.";
    return kAttrJitLevelO0;
  }
  return jit_level;
}

bool MsContext::IsKByKExecutorMode() const {
  // Get jit level.
  std::string jit_level = GetJitLevel();
  static std::string jit_level_log = "";
  bool is_jit_level_changed = false;
  auto mode = get_param<int>(MS_CTX_EXECUTION_MODE);
  if (jit_level_log != jit_level) {
    is_jit_level_changed = true;
    jit_level_log = jit_level;
  }

  if (get_param<bool>(MS_CTX_ENABLE_MEM_OFFLOAD)) {
    PrintJitLevelAndExecMode(is_jit_level_changed, jit_level, "enable kernelbykernel executor by mem offload.");
    return true;
  }

  if (mode == kPynativeMode) {
    if (jit_level == kAttrJitLevelO2) {
      PrintJitLevelAndExecMode(is_jit_level_changed, jit_level, "enable graph_sink executor in the PYNATIVE mode.");
      return false;
    }
    PrintJitLevelAndExecMode(is_jit_level_changed, jit_level, "enable kernelbykernel executor in the PYNATIVE mode.");
    return true;
  }

  if (mode == kGraphMode) {
    if (common::GetEnv("GRAPH_OP_RUN") == "1" || jit_level == kAttrJitLevelO0 || jit_level == kAttrJitLevelO1) {
      PrintJitLevelAndExecMode(is_jit_level_changed, jit_level, "enable kernelbykernel executor in the GRAPH mode.");
      return true;
    }
    PrintJitLevelAndExecMode(is_jit_level_changed, jit_level, "enable graph_sink executor in the GRAPH mode.");
    return false;
  }

  MS_LOG(ERROR) << "No valid executor mode.";
  return false;
}

void MsContext::SetAscendConfig() {
  set_param<std::string>(MS_CTX_PRECISION_MODE, "");
  set_param<std::string>(MS_CTX_ENABLE_JIT_COMPILE, "");
  set_param<std::string>(MS_CTX_ATOMIC_CLEAN_POLICY, "");
  set_param<std::string>(MS_CTX_MATMUL_ALLOW_HF32, "");
  set_param<std::string>(MS_CTX_CONV_ALLOW_HF32, "");
  set_param<std::string>(MS_CTX_OP_PRECISION_MODE, "");
  set_param<std::string>(MS_CTX_HOST_SCHEDULING_MAX_THRESHOLD, "");
  set_param<std::string>(MS_CTX_GE_OPTIONS, "");
}

void MsContext::InitBoolTypeDefaultValue() {
  set_param<bool>(MS_CTX_ENABLE_DUMP, false);
  set_param<bool>(MS_CTX_IS_MULTI_GRAPH_SINK, false);
  set_param<bool>(MS_CTX_IS_PYNATIVE_GE_INIT, false);
  set_param<bool>(MS_CTX_ENABLE_REDUCE_PRECISION, true);
  set_param<bool>(MS_CTX_ENABLE_TASK_SINK, true);
  set_param<bool>(MS_CTX_IR_FUSION_FLAG, true);
  set_param<bool>(MS_CTX_ENABLE_HCCL, false);
  set_param<bool>(MS_CTX_ENABLE_GPU_SUMMARY, true);
  set_param<bool>(MS_CTX_PRECOMPILE_ONLY, false);
  set_param<bool>(MS_CTX_ENABLE_PYNATIVE_INFER, false);
  set_param<bool>(MS_CTX_ENABLE_PYNATIVE_HOOK, false);
  set_param<bool>(MS_CTX_ENABLE_DYNAMIC_MEM_POOL, true);
  set_param<bool>(MS_CTX_ENABLE_GRAPH_KERNEL, false);
  set_param<bool>(MS_CTX_ENABLE_PARALLEL_SPLIT, false);
  set_param<bool>(MS_CTX_ENABLE_INFER_OPT, false);
  set_param<bool>(MS_CTX_GRAD_FOR_SCALAR, false);
  set_param<bool>(MS_CTX_ENABLE_MINDRT, false);
  set_param<bool>(MS_CTX_ENABLE_PYNATIVE_SYNCHRONIZE, false);
  set_param<bool>(MS_CTX_ENABLE_PYNATIVE_OP_GRAPH_CACHE, true);
  set_param<bool>(MS_CTX_ENABLE_MEM_OFFLOAD, false);
  set_param<bool>(MS_CTX_ENABLE_PROF_MEM, false);
  set_param<bool>(MS_CTX_ENABLE_RECOVERY, false);
  set_param<bool>(MS_CTX_ENABLE_GE_HETEROGENOUS, false);
  set_param<bool>(MS_CTX_DISABLE_FORMAT_TRANSFORM, false);
  set_param<bool>(MS_CTX_RECOMPUTE_COMM_OVERLAP, false);
  set_param<bool>(MS_CTX_GRAD_COMM_OVERLAP, false);
  set_param<bool>(MS_CTX_ENABLE_OPT_SHARD_COMM_OPT, false);
  set_param<bool>(MS_CTX_ENABLE_TASK_OPT, false);
  set_param<bool>(MS_CTX_ENABLE_GRAD_COMM_OPT, false);
  set_param<bool>(MS_CTX_INTERLEAVED_MATMUL_COMM, false);
  set_param<bool>(MS_CTX_INTERLEAVED_LAYERNORM_COMM, false);
  set_param<bool>(MS_CTX_BIAS_ADD_COMM_SWAP, false);
  set_param<bool>(MS_CTX_ENABLE_BEGIN_END_INLINE_OPT, false);
  set_param<bool>(MS_CTX_ENABLE_CONCAT_ELIMINATE_OPT, false);
  set_param<bool>(MS_CTX_ENABLE_FUSED_CAST_ADD_OPT, false);
  set_param<bool>(MS_CTX_ENABLE_PROFILING, false);
  set_param<bool>(MS_CTX_CHECK_BPROP_FLAG, false);
  set_param<bool>(MS_CTX_CONV_ALLOW_TF32, true);
  set_param<bool>(MS_CTX_MATMUL_ALLOW_TF32, false);
  set_param<bool>(MS_CTX_NEED_CKPT, false);
  set_param<bool>(MS_CTX_RECOMPUTE_ALLGATHER_OVERLAP_FAGRAD, false);
  set_param<bool>(MS_CTX_ENABLE_FLASH_ATTENTION_LOAD_BALANCE, false);
}

void MsContext::InitStringTypeDefaultValue() {
  set_param<std::string>(MS_CTX_PYTHON_EXE_PATH, "python");
  set_param<std::string>(MS_CTX_KERNEL_BUILD_SERVER_DIR, "");
  set_param<std::string>(MS_CTX_SAVE_DUMP_PATH, ".");
  set_param<std::string>(MS_CTX_DETERMINISTIC, "OFF");
  set_param<std::string>(MS_CTX_ENV_CONFIG_PATH, "");
  set_param<std::string>(MS_CTX_TUNE_MODE, "NO_TUNE");
  set_param<std::string>(MS_CTX_AOE_TUNE_MODE, "");
  set_param<std::string>(MS_CTX_AOE_JOB_TYPE, "2");
  set_param<std::string>(MS_CTX_GRAPH_KERNEL_FLAGS, "");
  set_param<std::string>(MS_CTX_HOST_SCHEDULING_MAX_THRESHOLD, "");
  set_param<std::string>(MS_CTX_ENABLE_EXCEPTION_DUMP, "2");
  set_param<std::string>(MS_CTX_GRAPH_MEMORY_MAX_SIZE, "0");
  set_param<std::string>(MS_CTX_VARIABLE_MEMORY_MAX_SIZE, "0");
  set_param<std::string>(MS_CTX_PROFILING_OPTIONS, "training_trace");
  set_param<std::string>(MS_CTX_PRINT_FILE_PATH, "");
  set_param<std::string>(MS_CTX_CONV_FPROP_ALGO, "normal");
  set_param<std::string>(MS_CTX_CONV_DGRAD_ALGO, "normal");
  set_param<std::string>(MS_CTX_CONV_WGRAD_ALGO, "normal");
  set_param<std::string>(MS_CTX_JIT_LEVEL, "");
  set_param<std::string>(MS_CTX_PROF_MEM_OUTPUT_PATH, "");
}

void MsContext::InitDigitalTypeDefaultValue() {
  set_param<int>(MS_CTX_EXECUTION_MODE, kPynativeMode);
  set_param<int>(MS_CTX_JIT_SYNTAX_LEVEL, kLax);
  set_param<int>(MS_CTX_CUR_STEP_NUM, 0);
  set_param<int>(MS_CTX_SAVE_CKPT_STEPS, 0);
  set_param<int>(MS_CTX_LAST_TRIGGERED_STEP, 0);
  set_param<int>(MS_CTX_COMPUTE_COMMUNICATE_FUSION_LEVEL, 0);
  set_param<int>(MS_CTX_DEBUG_LEVEL, kLevelRelease);
  set_param<float>(MS_CTX_MAX_DEVICE_MEMORY, kDefaultMaxDeviceMemory);
  set_param<float>(MS_CTX_MEMPOOL_BLOCK_SIZE, kDefaultMempoolBlockSize);
  //
  uint32_t kDefaultInterOpParallelThreads = 0;
  uint32_t kDefaultRuntimeNumThreads = 30;
  uint32_t cpu_core_num = std::thread::hardware_concurrency();
  uint32_t runtime_num_threads_default = std::min(cpu_core_num, kDefaultRuntimeNumThreads);
  uint32_t inter_op_parallel_num_default = std::min(cpu_core_num, kDefaultInterOpParallelThreads);
  set_param<uint32_t>(MS_CTX_RUNTIME_NUM_THREADS, runtime_num_threads_default);
  set_param<uint32_t>(MS_CTX_INTER_OP_PARALLEL_NUM, inter_op_parallel_num_default);
  //
  set_param<uint32_t>(MS_CTX_TSD_REF, 0);
  set_param<uint32_t>(MS_CTX_GE_REF, 0);
  set_param<uint32_t>(MS_CTX_MAX_CALL_DEPTH, MAX_CALL_DEPTH_DEFAULT);
  set_param<uint32_t>(MS_CTX_OP_TIMEOUT, kOpTimeout);
}

inline void SplitString(const std::string &str, char delim, std::set<std::string> *output_list) {
  std::stringstream ss(str);
  std::string item;
  while (std::getline(ss, item, delim)) {
    if (!item.empty()) {
      output_list->emplace(item);
    }
  }
}

inline std::string SetToString(const std::set<std::string> &kernel_list) {
  std::string out = "";
  for (auto &name : kernel_list) {
    out.append(name).append(", ");
  }
  return out;
}

void MsContext::SetMsInternalEnableCustomKernelList() {
  const std::string kDefaultEnabledOpList =
    "MatMul,RmsNorm,Add,Sub,FlashAttentionScore,PagedAttention,AddRmsNorm,AddLayerNorm,InferenceMatmulSplit";
  auto internal_op_boost_env = common::GetEnv("MS_ENABLE_INTERNAL_BOOST");
  bool is_enalbe_internal_op = true;
  if (internal_op_boost_env == "off") {
    is_enalbe_internal_op = false;
  }

  ms_internal_enable_custom_kernel_list_.clear();
  if (is_enalbe_internal_op) {
    SplitString(kDefaultEnabledOpList, ',', &ms_internal_enable_custom_kernel_list_);
  }

  std::string env = common::GetEnv("MS_INTERNAL_ENABLE_CUSTOM_KERNEL_LIST");
  if (!env.empty()) {
    SplitString(env, ',', &ms_internal_enable_custom_kernel_list_);
  }

  MS_LOG(INFO) << "Enable internal kernel list: " << SetToString(ms_internal_enable_custom_kernel_list_);
}

bool MsContext::IsEnableInferBoost() {
  if (enable_infer_boost_.has_value()) {
    return enable_infer_boost_.value();
  }

  const auto &jit_config = PhaseManager::GetInstance().jit_config();
  auto iter = jit_config.find("infer_boost");
  if (iter != jit_config.end() && iter->second == "on") {
    enable_infer_boost_ = true;
    MS_LOG(INFO) << "MSContext enable ms infer boost from JitConfig";
    SetMsInternalEnableCustomKernelList();
    return enable_infer_boost_.value();
  }

  if (common::GetEnv("MS_ENABLE_INTERNAL_KERNELS") == "on") {
    enable_infer_boost_ = true;
    MS_LOG(INFO) << "MSContext enable ms infer boost from Env";
    SetMsInternalEnableCustomKernelList();
  } else {
    enable_infer_boost_ = false;
  }

  return enable_infer_boost_.value();
}

const std::set<std::string> &MsContext::ms_internal_enable_custom_kernel_list() const {
  return ms_internal_enable_custom_kernel_list_;
}

template MS_CORE_API void MsContext::CheckReadStatus<bool>(MsCtxParam, const bool &) const;
template MS_CORE_API void MsContext::CheckReadStatus<uint32_t>(MsCtxParam, const uint32_t &) const;
template MS_CORE_API void MsContext::CheckReadStatus<int>(MsCtxParam, const int &) const;
template MS_CORE_API void MsContext::CheckReadStatus<float>(MsCtxParam, const float &) const;
template MS_CORE_API void MsContext::CheckReadStatus<std::string>(MsCtxParam, const std::string &) const;
}  // namespace mindspore
