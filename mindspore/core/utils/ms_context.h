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

#ifndef MINDSPORE_CORE_UTILS_MS_CONTEXT_H_
#define MINDSPORE_CORE_UTILS_MS_CONTEXT_H_
#include <thread>
#include <memory>
#include <map>
#include <set>
#include <string>
#include <functional>
#include <mutex>
#include "utils/log_adapter.h"
#include "utils/ms_utils.h"

namespace mindspore {
enum MsBackendPolicy {
  kMsBackendGeOnly = 0,
  kMsBackendVmOnly = 1,
  kMsBackendGePrior = 2,
  kMsBackendVmPrior = 3,
  kMsBackendMsPrior = 4,
  kMsBackendUnknown = 5,
};

enum level : int {
  kIntroductory = 1,
  kAdvanced,
  kFully,
};

const int kGraphMode = 0;
const int kPynativeMode = 1;
const char kDeviceUnDefined[] = "DeviceUnDefined";
const char kCPUDevice[] = "CPU";
const char kGPUDevice[] = "GPU";
const char kAscendDevice[] = "Ascend";
const char kDavinciInferenceDevice[] = "AscendInference";
const char kDavinciMultiGraphInferenceDevice[] = "AscendMultiGraphInference";
const char kGpuInferenceDevice[] = "GpuInference";
const char kDavinciDevice[] = "Davinci";
const char KNpuLog[] = "_npu_log";
const char kTraining[] = "training";
const unsigned int MAX_CALL_DEPTH_DEFAULT = 1000;
const unsigned int kOpTimeout = 600;
const int kOptimizeO0 = 0;
const int kOptimizeO1 = 1;

const std::set<std::string> kTargetSet = {kCPUDevice, kGPUDevice, kAscendDevice, kDavinciDevice};
// The default max available device memory is 1024GB.
const float kDefaultMaxDeviceMemory = 1024;
// The default memory pool block size is 1.0G.
const float kDefaultMempoolBlockSize = 1.0;

// enum definition for MindSpore Context Parameter
enum MsCtxParam : unsigned {
  // parameter of type bool
  MS_CTX_TYPE_BOOL_BEGIN,
  MS_CTX_CHECK_BPROP_FLAG = MS_CTX_TYPE_BOOL_BEGIN,
  MS_CTX_ENABLE_DUMP,
  MS_CTX_ENABLE_DYNAMIC_MEM_POOL,
  MS_CTX_ENABLE_GPU_SUMMARY,
  MS_CTX_ENABLE_GRAPH_KERNEL,
  MS_CTX_ENABLE_HCCL,
  MS_CTX_ENABLE_LOOP_SINK,
  MS_CTX_ENABLE_PYNATIVE_HOOK,
  MS_CTX_ENABLE_PYNATIVE_INFER,
  MS_CTX_ENABLE_REDUCE_PRECISION,
  MS_CTX_ENABLE_TASK_SINK,
  MS_CTX_IR_FUSION_FLAG,
  MS_CTX_IS_MULTI_GRAPH_SINK,
  MS_CTX_IS_PYNATIVE_GE_INIT,
  MS_CTX_PRECOMPILE_ONLY,
  MS_CTX_ENABLE_PROFILING,
  MS_CTX_ENABLE_PARALLEL_SPLIT,
  MS_CTX_ENABLE_INFER_OPT,
  MS_CTX_GRAD_FOR_SCALAR,
  MS_CTX_ENABLE_MINDRT,
  MS_CTX_ENABLE_PYNATIVE_SYNCHRONIZE,
  MS_CTX_ENABLE_PYNATIVE_OP_GRAPH_CACHE,
  MS_CTX_ENABLE_MEM_OFFLOAD,
  MS_CTX_ENABLE_RECOVERY,
  MS_CTX_ENABLE_GE_HETEROGENOUS,
  MS_CTX_DISABLE_FORMAT_TRANSFORM,
  MS_CTX_TYPE_BOOL_END,

  // parameter of type int
  MS_CTX_TYPE_INT_BEGIN = MS_CTX_TYPE_BOOL_END,
  MS_CTX_EXECUTION_MODE = MS_CTX_TYPE_INT_BEGIN,
  MS_CTX_MEMORY_OPTIMIZE_LEVEL,
  MS_CTX_SAVE_GRAPHS_FLAG,
  MS_CTX_TYPE_INT_END,

  // parameter of type uint32
  MS_CTX_TYPE_UINT32_BEGIN = MS_CTX_TYPE_INT_END,
  MS_CTX_DEVICE_ID = MS_CTX_TYPE_UINT32_BEGIN,
  MS_CTX_RUNTIME_NUM_THREADS,
  MS_CTX_INTER_OP_PARALLEL_NUM,
  MS_CTX_GE_REF,
  MS_CTX_MAX_CALL_DEPTH,
  MS_CTX_TSD_REF,
  MS_CTX_OP_TIMEOUT,
  MS_CTX_TYPE_UINT32_END,

  // parameter of type float
  MS_CTX_TYPE_FLOAT_BEGIN = MS_CTX_TYPE_UINT32_END,
  MS_CTX_MAX_DEVICE_MEMORY = MS_CTX_TYPE_FLOAT_BEGIN,
  MS_CTX_MEMPOOL_BLOCK_SIZE,
  MS_CTX_TYPE_FLOAT_END,

  // parameter of type string
  MS_CTX_TYPE_STRING_BEGIN = MS_CTX_TYPE_FLOAT_END,
  MS_CTX_DEVICE_TARGET = MS_CTX_TYPE_STRING_BEGIN,
  MS_CTX_GRAPH_MEMORY_MAX_SIZE,
  MS_CTX_PRINT_FILE_PATH,
  MS_CTX_PROFILING_OPTIONS,
  MS_CTX_SAVE_DUMP_PATH,
  MS_CTX_SAVE_GRAPHS_PATH,
  MS_CTX_COMPILE_CACHE_PATH,
  MS_CTX_VARIABLE_MEMORY_MAX_SIZE,
  MS_CTX_PYTHON_EXE_PATH,
  MS_CTX_KERNEL_BUILD_SERVER_DIR,
  MS_CTX_ENV_CONFIG_PATH,
  MS_CTX_TUNE_MODE,
  MS_CTX_GRAPH_KERNEL_FLAGS,
  MS_CTX_INFER_PRECISION_MODE,  // GPU inference precision mode configured by Serving or Unify API.
  MS_CTX_DETERMINISTIC,
  MS_CTX_TYPE_STRING_END,

  // parameter numbers of each type
  NUM_BOOL_PARAMS = MS_CTX_TYPE_BOOL_END - MS_CTX_TYPE_BOOL_BEGIN,
  NUM_INT_PARAMS = MS_CTX_TYPE_INT_END - MS_CTX_TYPE_INT_BEGIN,
  NUM_UINT32_PARAMS = MS_CTX_TYPE_UINT32_END - MS_CTX_TYPE_UINT32_BEGIN,
  NUM_FLOAT_PARAMS = MS_CTX_TYPE_FLOAT_END - MS_CTX_TYPE_FLOAT_BEGIN,
  NUM_STRING_PARAMS = MS_CTX_TYPE_STRING_END - MS_CTX_TYPE_STRING_BEGIN
};

class MS_CORE_API MsContext {
 public:
  MsContext(const std::string &policy, const std::string &target);
  ~MsContext() = default;
  MsContext(const MsContext &) = delete;
  MsContext &operator=(const MsContext &) = delete;
  using DeviceSeter = void (*)(const std::string &device_target);
  using InitDeviceTargetAndPolicy = void (*)(MsContext *);
  using LoadPluginError = std::string (*)();
  using EnvFunc = std::function<void(const std::string &, const std::string &)>;  // device name, library path
  static std::shared_ptr<MsContext> GetInstance();

  void Refresh();

  bool enable_dump_ir() const;
  std::string GetSaveGraphsPath();
  bool CanDump(const int &level);
  std::string backend_policy() const;
  bool set_backend_policy(const std::string &policy);
  // _comm_helper.py will try to dlopen libhccl.so, and minddata will try to dlopen libdvpp_utils.so. if load ascend
  // plugin failed on ascend environment, loading above libraries will crush the process.
  bool IsAscendPluginLoaded() const;
  void SetDefaultDeviceTarget();
  void SetDeviceTargetFromInner(const std::string &device_target);
  void SetDeviceTargetFromUser(const std::string &device_target);
  bool IsDefaultDeviceTarget() const;

  void RegisterSetEnv(const EnvFunc &func);
  void RegisterCheckEnv(const EnvFunc &func);

  void SetEnv(const std::string &device);
  void CheckEnv(const std::string &device);

  static void device_seter(const DeviceSeter &device) { seter_ = device; }
  static void RegisterInitFunc(const std::string &name, InitDeviceTargetAndPolicy func);
  static void ResisterLoadPluginErrorFunc(LoadPluginError func);

  template <typename T>
  void set_param(MsCtxParam, const T &) {
    MS_LOG(EXCEPTION) << "Need to implement " << __FUNCTION__ << " for type " << typeid(T).name() << ".";
  }

  template <typename T>
  const T &get_param(MsCtxParam) const {
    MS_LOG(EXCEPTION) << "Need to implement " << __FUNCTION__ << " for type " << typeid(T).name() << ".";
  }

  template <typename T>
  void increase_param(MsCtxParam) {
    MS_LOG(EXCEPTION) << "Need to implement " << __FUNCTION__ << " for type " << typeid(T).name() << ".";
  }

  template <typename T>
  void decrease_param(MsCtxParam) {
    MS_LOG(EXCEPTION) << "Need to implement " << __FUNCTION__ << " for type " << typeid(T).name() << ".";
  }

 private:
  void RefreshExecutionMode();
  void RefreshMemoryOffload();

 private:
  static DeviceSeter seter_;
  static std::shared_ptr<MsContext> inst_context_;
  static LoadPluginError load_plugin_error_;

  bool bool_params_[MsCtxParam::NUM_BOOL_PARAMS];
  int int_params_[MsCtxParam::NUM_INT_PARAMS];
  uint32_t uint32_params_[MsCtxParam::NUM_UINT32_PARAMS];
  float float_params_[MsCtxParam::NUM_FLOAT_PARAMS];
  std::string string_params_[MsCtxParam::NUM_STRING_PARAMS];
  MsBackendPolicy backend_policy_;
  bool default_device_target_ = true;

  EnvFunc set_env_ = nullptr;
  EnvFunc check_env_ = nullptr;

  static std::map<std::string, InitDeviceTargetAndPolicy> &InitFuncMap();
  static std::map<std::string, std::string> &PluginPathMap();
};

// set method implementation for type bool/int/uint32_t/float/std::string
template <>
inline void MsContext::set_param<bool>(MsCtxParam param, const bool &value) {
#ifdef ENABLE_SECURITY
  if (param == MS_CTX_SAVE_GRAPHS_FLAG) {
    MS_EXCEPTION(ValueError) << "The save_graphs is not supported, please without '-s on' and recompile source.";
  }
#endif
  bool_params_[param - MS_CTX_TYPE_BOOL_BEGIN] = value;
}

template <>
inline void MsContext::set_param<int>(MsCtxParam param, const int &value) {
  int_params_[param - MS_CTX_TYPE_INT_BEGIN] = value;
}

template <>
inline void MsContext::set_param<uint32_t>(MsCtxParam param, const uint32_t &value) {
  uint32_params_[param - MS_CTX_TYPE_UINT32_BEGIN] = value;
}

template <>
inline void MsContext::set_param<float>(MsCtxParam param, const float &value) {
  float_params_[param - MS_CTX_TYPE_FLOAT_BEGIN] = value;
}

template <>
inline void MsContext::set_param<std::string>(MsCtxParam param, const std::string &value) {
#ifdef ENABLE_SECURITY
  if (param == MS_CTX_SAVE_GRAPHS_PATH) {
    MS_EXCEPTION(ValueError) << "The save_graphs is not supported, please without '-s on' and recompile source.";
  }
#endif
  if (param == MS_CTX_DEVICE_TARGET) {
    SetDeviceTargetFromUser(value);
  } else {
    string_params_[param - MS_CTX_TYPE_STRING_BEGIN] = value;
  }
}

// get method implementation for type bool/int/uint32_t/float/std::string
template <>
inline const bool &MsContext::get_param<bool>(MsCtxParam param) const {
  return bool_params_[param - MS_CTX_TYPE_BOOL_BEGIN];
}

template <>
inline const int &MsContext::get_param<int>(MsCtxParam param) const {
  return int_params_[param - MS_CTX_TYPE_INT_BEGIN];
}

template <>
inline const uint32_t &MsContext::get_param<uint32_t>(MsCtxParam param) const {
  return uint32_params_[param - MS_CTX_TYPE_UINT32_BEGIN];
}

template <>
inline const float &MsContext::get_param<float>(MsCtxParam param) const {
  return float_params_[param - MS_CTX_TYPE_FLOAT_BEGIN];
}

template <>
inline const std::string &MsContext::get_param<std::string>(MsCtxParam param) const {
  return string_params_[param - MS_CTX_TYPE_STRING_BEGIN];
}

// increate method implementation for type uint32_t
template <>
inline void MsContext::increase_param<uint32_t>(MsCtxParam param) {
  uint32_params_[param - MS_CTX_TYPE_UINT32_BEGIN]++;
}

// decreate method implementation for type uint32_t
template <>
inline void MsContext::decrease_param<uint32_t>(MsCtxParam param) {
  uint32_params_[param - MS_CTX_TYPE_UINT32_BEGIN]--;
}

#define MSCONTEXT_REGISTER_INIT_FUNC(name, func)                          \
  class name##InitFuncRegister {                                          \
   public:                                                                \
    name##InitFuncRegister() { MsContext::RegisterInitFunc(name, func); } \
  } g_##name##_init_func_register;
}  // namespace mindspore

#endif  // MINDSPORE_CORE_UTILS_MS_CONTEXT_H_
