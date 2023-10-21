/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/ascend/hal/common/ascend_utils.h"
#include <vector>
#include <string>
#include <map>
#include "common/util/error_manager/error_manager.h"
#include "include/common/utils/anfalgo.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "runtime/device/ms_device_shape_transfer.h"
#include "utils/ms_context.h"
#include "utils/dlopen_macro.h"
#include "runtime/dev.h"
#include "runtime/config.h"
#include "acl/error_codes/rt_error_codes.h"
#ifdef ASCEND_910
#define EXPECT_ASCEND_VERSION "ascend910"
#elif defined(ASCEND_910B)
#define EXPECT_ASCEND_VERSION "ascend910b"
#endif

namespace mindspore {
namespace device {
namespace ascend {
namespace {
const std::map<uint32_t, std::string> error_msg = {
  {ACL_RT_SUCCESS, "success"},
  {ACL_ERROR_RT_PARAM_INVALID, "param invalid"},
  {ACL_ERROR_RT_INVALID_DEVICEID, "invalid device id"},
  {ACL_ERROR_RT_CONTEXT_NULL, "current context null"},
  {ACL_ERROR_RT_STREAM_CONTEXT, "stream not in current context"},
  {ACL_ERROR_RT_MODEL_CONTEXT, "model not in current context"},
  {ACL_ERROR_RT_STREAM_MODEL, "stream not in model"},
  {ACL_ERROR_RT_EVENT_TIMESTAMP_INVALID, "event timestamp invalid"},
  {ACL_ERROR_RT_EVENT_TIMESTAMP_REVERSAL, " event timestamp reversal"},
  {ACL_ERROR_RT_ADDR_UNALIGNED, "memory address unaligned"},
  {ACL_ERROR_RT_FILE_OPEN, "open file failed"},
  {ACL_ERROR_RT_FILE_WRITE, "write file failed"},
  {ACL_ERROR_RT_STREAM_SUBSCRIBE, "error subscribe stream"},
  {ACL_ERROR_RT_THREAD_SUBSCRIBE, "error subscribe thread"},
  {ACL_ERROR_RT_GROUP_NOT_SET, "group not set"},
  {ACL_ERROR_RT_GROUP_NOT_CREATE, "group not create"},
  {ACL_ERROR_RT_STREAM_NO_CB_REG, "callback not register to stream"},
  {ACL_ERROR_RT_INVALID_MEMORY_TYPE, "invalid memory type"},
  {ACL_ERROR_RT_INVALID_HANDLE, "invalid handle"},
  {ACL_ERROR_RT_INVALID_MALLOC_TYPE, "invalid malloc type"},
  {ACL_ERROR_RT_FEATURE_NOT_SUPPORT, "feature not support"},
  {ACL_ERROR_RT_MEMORY_ALLOCATION, "memory allocation error"},
  {ACL_ERROR_RT_MEMORY_FREE, "memory free error"},
  {ACL_ERROR_RT_AICORE_OVER_FLOW, "aicore over flow"},
  {ACL_ERROR_RT_NO_DEVICE, "no device"},
  {ACL_ERROR_RT_RESOURCE_ALLOC_FAIL, "resource alloc fail"},
  {ACL_ERROR_RT_NO_PERMISSION, "no permission"},
  {ACL_ERROR_RT_NO_EVENT_RESOURCE, "no event resource"},
  {ACL_ERROR_RT_NO_STREAM_RESOURCE, "no stream resource"},
  {ACL_ERROR_RT_NO_NOTIFY_RESOURCE, "no notify resource"},
  {ACL_ERROR_RT_NO_MODEL_RESOURCE, "no model resource"},
  {ACL_ERROR_RT_INTERNAL_ERROR, "runtime internal error"},
  {ACL_ERROR_RT_TS_ERROR, "ts internal error"},
  {ACL_ERROR_RT_STREAM_TASK_FULL, "task full in stream"},
  {ACL_ERROR_RT_STREAM_TASK_EMPTY, " task empty in stream"},
  {ACL_ERROR_RT_STREAM_NOT_COMPLETE, "stream not complete"},
  {ACL_ERROR_RT_END_OF_SEQUENCE, "end of sequence"},
  {ACL_ERROR_RT_EVENT_NOT_COMPLETE, "event not complete"},
  {ACL_ERROR_RT_CONTEXT_RELEASE_ERROR, "context release error"},
  {ACL_ERROR_RT_SOC_VERSION, "soc version error"},
  {ACL_ERROR_RT_TASK_TYPE_NOT_SUPPORT, "task type not support"},
  {ACL_ERROR_RT_LOST_HEARTBEAT, "ts lost heartbeat"},
  {ACL_ERROR_RT_MODEL_EXECUTE, " model execute failed"},
  {ACL_ERROR_RT_REPORT_TIMEOUT, "report timeout"},
  {ACL_ERROR_RT_SYS_DMA, "sys dma error"},
  {ACL_ERROR_RT_AICORE_TIMEOUT, "aicore timeout"},
  {ACL_ERROR_RT_AICORE_EXCEPTION, "aicore exception"},
  {ACL_ERROR_RT_AICORE_TRAP_EXCEPTION, " aicore trap exception"},
  {ACL_ERROR_RT_AICPU_TIMEOUT, " aicpu timeout"},
  {ACL_ERROR_RT_AICPU_EXCEPTION, "aicpu exception"},
  {ACL_ERROR_RT_AICPU_DATADUMP_RSP_ERR, " aicpu datadump response error"},
  {ACL_ERROR_RT_AICPU_MODEL_RSP_ERR, "aicpu model operate response error"},
  {ACL_ERROR_RT_PROFILING_ERROR, "profiling error"},
  {ACL_ERROR_RT_IPC_ERROR, "ipc error"},
  {ACL_ERROR_RT_MODEL_ABORT_NORMAL, "model abort normal"},
  {ACL_ERROR_RT_KERNEL_UNREGISTERING, "kernel unregistering"},
  {ACL_ERROR_RT_RINGBUFFER_NOT_INIT, "ringbuffer not init"},
  {ACL_ERROR_RT_RINGBUFFER_NO_DATA, "ringbuffer no data"},
  {ACL_ERROR_RT_KERNEL_LOOKUP, "kernel lookup error"},
  {ACL_ERROR_RT_KERNEL_DUPLICATE, "kernel register duplicate"},
  {ACL_ERROR_RT_DEBUG_REGISTER_FAIL, "debug register failed"},
  {ACL_ERROR_RT_DEBUG_UNREGISTER_FAIL, "debug unregister failed"},
  {ACL_ERROR_RT_LABEL_CONTEXT, "label not in current context"},
  {ACL_ERROR_RT_PROGRAM_USE_OUT, "program register num use out"},
  {ACL_ERROR_RT_DEV_SETUP_ERROR, "device setup error"},
  {ACL_ERROR_RT_DRV_INTERNAL_ERROR, "drv internal error"},
};

constexpr auto kUnknowErrorString = "Unknown error occurred";
}  // namespace

error_message::Context ErrorManagerAdapter::context_;
std::mutex ErrorManagerAdapter::initialized_mutex_;
bool ErrorManagerAdapter::initialized_ = false;
std::vector<std::string> ErrorManagerAdapter::traceback_;

bool ErrorManagerAdapter::Init() {
  std::unique_lock<std::mutex> lock(initialized_mutex_);
  if (initialized_) {
    MS_LOG(DEBUG) << "Ascend error manager has been initialized.";
    return true;
  }
  const auto error_manager_init_ret = ErrorManager::GetInstance().Init();
  if (error_manager_init_ret != 0) {
    MS_LOG(WARNING) << "Init ascend error manager failed, some ascend error log may be left out.";
    return false;
  }
  ErrorManager::GetInstance().GenWorkStreamIdDefault();
  context_ = ErrorManager::GetInstance().GetErrorManagerContext();
  MS_LOG(DEBUG) << "Initialize ascend error manager successfully. Work stream id: " << context_.work_stream_id;
  initialized_ = true;
  LogWriter::SetMessageHandler(&MessageHandler);
  return true;
}

void ErrorManagerAdapter::BindToCurrentThread() {
  if (initialized_) {
    ErrorManager::GetInstance().SetErrorContext(context_);
  }
}

std::string ErrorManagerAdapter::GetErrorMessage(bool add_title) {
  const string &error_message = ErrorManager::GetInstance().GetErrorMessage();
  if (error_message.empty() || error_message.find(kUnknowErrorString) != string::npos) {
    return "";
  }
  if (add_title) {
    return "#umsg#Ascend Error Message:#umsg#" + error_message +
           "\n(Please search \"Ascend Error Message\" at https://www.mindspore.cn for error code description)";
  }
  return error_message;
}

std::string ErrorManagerAdapter::GetWarningMessage(bool add_title) {
  const string &warning_message = ErrorManager::GetInstance().GetWarningMessage();
  if (warning_message.empty()) {
    return "";
  }
  if (add_title) {
    return "#umsg#Ascend Warning Message:#umsg#" + warning_message;
  }
  return warning_message;
}

void ErrorManagerAdapter::MessageHandler(std::ostringstream *oss) {
  const auto &error_message = GetErrorMessage(true);
  if (!error_message.empty()) {
    (void)traceback_.emplace_back(error_message);
  }
  const auto &warning_message = GetWarningMessage(true);
  if (!warning_message.empty()) {
    (void)traceback_.emplace_back(warning_message);
  }
  for (const auto &message : traceback_) {
    *oss << message;
  }
}

bool IsGraphMode() {
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  return context->get_param<int>(MS_CTX_EXECUTION_MODE) == kGraphMode;
}

bool IsDynamicShapeGraph(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  std::vector<AnfNodePtr> node_list = TopoSort(func_graph->get_return());
  return std::any_of(node_list.begin(), node_list.end(),
                     [](const AnfNodePtr &node) { return common::AnfAlgo::IsDynamicShape(node); });
}

std::string GetSocVersion() {
  // Get default soc version.
  static std::string version;
  if (version.empty()) {
    const int kSocVersionLen = 50;
    char soc_version[kSocVersionLen] = {0};
    auto ret = rtGetSocVersion(soc_version, kSocVersionLen);
    if (ret != RT_ERROR_NONE) {
      MS_LOG(EXCEPTION) << "GetSocVersion failed.";
    }
    version = soc_version;
  }
  return version;
}

std::string GetAscendPath() {
  Dl_info info;
  if (dladdr(reinterpret_cast<void *>(rtGetSocVersion), &info) == 0) {
    MS_LOG(INFO) << "Get dladdr failed, skip.";
    return "";
  }
  auto path_tmp = std::string(info.dli_fname);
  const std::string kLib64 = "lib64";
  auto pos = path_tmp.find(kLib64);
  if (pos == std::string::npos) {
    MS_EXCEPTION(ValueError) << "Get ascend path failed, please check the run package.";
  }
  return path_tmp.substr(0, pos);
}

std::string GetAICoreNumber() {
  constexpr int32_t kModelTypeAiCore = 4;  // enum DEV_MODULE_TYPE { MODULE_TYPE_AICORE = 4 }
  constexpr int32_t kInfoTypeCoreNum = 3;  // enum DEV_INFO_TYPE { INFO_TYPE_CORE_NUM = 3 }
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  uint32_t device_id = context_ptr->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  int64_t aicore_number = 0;
  auto rt_ret = rtGetDeviceInfo(device_id, kModelTypeAiCore, kInfoTypeCoreNum, &aicore_number);
  if (rt_ret != RT_ERROR_NONE) {
    MS_LOG(WARNING) << "Get aicore number for device " << device_id
                    << " failed, will compile tbe op with empty core_num.";
    return "";
  }
  MS_LOG(DEBUG) << "AiCore number of device " << device_id << " is " << aicore_number;
  return std::to_string(aicore_number);
}

std::string GetErrorMsg(uint32_t rt_error_code) {
  auto find_iter = error_msg.find(rt_error_code);
  if (find_iter == error_msg.end()) {
    return "Return error code unknown, ret code: " + std::to_string(rt_error_code);
  }
  return find_iter->second;
}

#if defined(ASCEND_910) || defined(ASCEND_910B)
constexpr auto k910AscendVersion = "Ascend910";
constexpr auto k910BAscendVersion = "ascend910b";
const std::map<std::string, std::string> kAscendSocVersions = {
  {"Ascend910A", "ascend910"},    {"Ascend910B", "ascend910"},    {"Ascend910PremiumA", "ascend910"},
  {"Ascend910ProA", "ascend910"}, {"Ascend910ProB", "ascend910"}, {"Ascend910B1", "ascend910b"},
  {"Ascend910B2", "ascend910b"},  {"Ascend910B3", "ascend910b"},  {"Ascend910B4", "ascend910b"}};

// for unify 1980 and 1980b, when the function throw exception, it means the 910b soc version is not available.
const bool SelectAscendPlugin = []() -> bool {
  // for 1951, if is_heterogenous, return true
  int32_t is_heterogenous = 0;
  (void)rtGetIsHeterogenous(&is_heterogenous);
  if (is_heterogenous == 1) {
    if (std::string(EXPECT_ASCEND_VERSION) == k910BAscendVersion) {
      exit(0);
    } else {
      return true;
    }
  }
  std::string soc_version = GetSocVersion();
  // if soc_version belongs to 310 or 710, return true
  if (soc_version.find(k910AscendVersion) == std::string::npos) {
    return true;
  }
  auto iter = kAscendSocVersions.find(soc_version);
  if (iter == kAscendSocVersions.end()) {
    exit(0);
  }
  if (iter->second != std::string(EXPECT_ASCEND_VERSION)) {
    exit(0);
  }
  if (iter->second == k910BAscendVersion) {
    common::SetEnv("MS_ENABLE_GE", "1");
    auto format_mode = common::GetEnv("MS_ENABLE_FORMAT_MODE");
    if (format_mode.empty()) {
      common::SetEnv("MS_ENABLE_FORMAT_MODE", "1");
    }
    auto force_acl = common::GetEnv("MS_DEV_FORCE_ACL");
    auto disable_ref = common::GetEnv("MS_DISABLE_REF_MODE");
    // MS_DEV_FORCE_ACL 1: ACL with special format, 2: ACL with default format.
    if (force_acl.empty() && disable_ref != "1") {
      common::SetEnv("MS_DEV_FORCE_ACL", "1");
    }
  }
  return true;
}();
#endif
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
