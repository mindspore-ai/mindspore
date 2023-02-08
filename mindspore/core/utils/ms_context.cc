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
#include "include/common/utils/utils.h"
#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace mindspore {
namespace {
std::map<std::string, MsBackendPolicy> kPolicyMap = {{"ge", kMsBackendGePrior},
                                                     {"vm", kMsBackendVmOnly},
                                                     {"ms", kMsBackendMsPrior},
                                                     {"ge_only", kMsBackendGeOnly},
                                                     {"vm_prior", kMsBackendVmPrior}};
}  // namespace
std::atomic<bool> thread_1_must_end(false);

MsContext::DeviceSeter MsContext::seter_ = nullptr;
MsContext::LoadPluginError MsContext::load_plugin_error_ = nullptr;
std::shared_ptr<MsContext> MsContext::inst_context_ = nullptr;

MsContext::MsContext(const std::string &policy, const std::string &target) {
#ifndef ENABLE_SECURITY
  set_param<int>(MS_CTX_SAVE_GRAPHS_FLAG, 0);
  set_param<std::string>(MS_CTX_SAVE_GRAPHS_PATH, ".");
  set_param<std::string>(MS_CTX_COMPILE_CACHE_PATH, "");
#else
  // Need set a default value for arrays even if running in the security mode.
  int_params_[MS_CTX_SAVE_GRAPHS_FLAG - MS_CTX_TYPE_BOOL_BEGIN] = 0;
  string_params_[MS_CTX_SAVE_GRAPHS_PATH - MS_CTX_TYPE_STRING_BEGIN] = ".";
#endif
  set_param<std::string>(MS_CTX_PYTHON_EXE_PATH, "python");
  set_param<std::string>(MS_CTX_KERNEL_BUILD_SERVER_DIR, "");
  set_param<bool>(MS_CTX_ENABLE_DUMP, false);
  set_param<std::string>(MS_CTX_SAVE_DUMP_PATH, ".");
  set_param<std::string>(MS_CTX_DETERMINISTIC, "OFF");
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
  string_params_[MS_CTX_DEVICE_TARGET - MS_CTX_TYPE_STRING_BEGIN] = target;
  set_param<int>(MS_CTX_EXECUTION_MODE, kPynativeMode);
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
  set_param<bool>(MS_CTX_ENABLE_MEM_OFFLOAD, false);
  set_param<bool>(MS_CTX_ENABLE_RECOVERY, false);
  set_param<bool>(MS_CTX_ENABLE_GE_HETEROGENOUS, false);
  set_param<bool>(MS_CTX_DISABLE_FORMAT_TRANSFORM, false);
  set_param<int>(MS_CTX_MEMORY_OPTIMIZE_LEVEL, kOptimizeO0);
  set_param<uint32_t>(MS_CTX_OP_TIMEOUT, kOpTimeout);

  uint32_t kDefaultInterOpParallelThreads = 0;
  uint32_t kDefaultRuntimeNumThreads = 30;
  uint32_t cpu_core_num = std::thread::hardware_concurrency();
  uint32_t runtime_num_threads_default = std::min(cpu_core_num, kDefaultRuntimeNumThreads);
  uint32_t inter_op_parallel_num_default = std::min(cpu_core_num, kDefaultInterOpParallelThreads);
  set_param<uint32_t>(MS_CTX_RUNTIME_NUM_THREADS, runtime_num_threads_default);
  set_param<uint32_t>(MS_CTX_INTER_OP_PARALLEL_NUM, inter_op_parallel_num_default);

  backend_policy_ = kPolicyMap[policy];
}

std::shared_ptr<MsContext> MsContext::GetInstance() {
  static std::once_flag inst_context_init_flag_ = {};
  std::call_once(inst_context_init_flag_, [&]() {
    if (inst_context_ == nullptr) {
      MS_LOG(DEBUG) << "Create new mindspore context";
      inst_context_ = std::make_shared<MsContext>("vm", kCPUDevice);
    }
  });

  return inst_context_;
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
    } else if (common::GetEnv(kGraphOpRun) == "1") {
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
  if (target == kAscendDevice && get_param<int>(MS_CTX_EXECUTION_MODE) != kPynativeMode &&
      common::GetEnv(kGraphOpRun) != "1") {
    MS_LOG(WARNING) << "Memory offload is not available when GRAPH_OP_RUN is not set to 1.";
    set_param(MS_CTX_ENABLE_MEM_OFFLOAD, false);
    return;
  }
  if (get_param<int>(MS_CTX_MEMORY_OPTIMIZE_LEVEL) == kOptimizeO1) {
    MS_LOG(WARNING) << "Memory offload is not available when memory_optimize_level is set to O1.";
    set_param(MS_CTX_ENABLE_MEM_OFFLOAD, false);
    return;
  }
  MS_LOG(INFO) << "Set memory pool block size to max device memory size for memory offload.";
  set_param(MS_CTX_MEMPOOL_BLOCK_SIZE, get_param<float>(MS_CTX_MAX_DEVICE_MEMORY));
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
  InitFuncMap().emplace(name, func);
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
  PluginPathMap().emplace(name, plugin_path);
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
  } else if (InitFuncMap().size() == 2) {
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

std::string MsContext::GetSaveGraphsPath() {
  std::string path = common::GetEnv("MS_DEV_SAVE_GRAPHS_PATH");
  if (!path.empty()) {
    return path;
  } else {
    return MsContext::GetInstance()->get_param<std::string>(MS_CTX_SAVE_GRAPHS_PATH);
  }
}

bool MsContext::CanDump(const int &level) {
  int save_graphs = MsContext::GetInstance()->get_param<int>(MS_CTX_SAVE_GRAPHS_FLAG);
  std::string save_env = common::GetEnv("MS_DEV_SAVE_GRAPHS");
  if (save_env.size() == 1) {
    int save_graphs_by_env = std::stoi(save_env);
    if (save_graphs_by_env < 0 || save_graphs_by_env > kFully) {
      MS_LOG(EXCEPTION) << "Dump level can only be from 0 to 3";
    }
    if (save_graphs_by_env >= level) {
      return true;
    }
  } else if (save_env.size() > 1) {
    MS_LOG(EXCEPTION) << "MS_DEV_SAVE_GRAPHS should be a single number with one digit.";
  }
  if (save_graphs >= level) {
    return true;
  }
  return false;
}
}  // namespace mindspore
