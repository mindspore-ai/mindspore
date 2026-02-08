/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include "plugin/ascend/res_manager/hal_manager/ascend_err_manager.h"

#include <map>
#include "acl/acl_rt.h"
#include "acl/error_codes/rt_error_codes.h"
#include "utils/log_adapter.h"
#include "plugin/ascend/res_manager/symbol_interface/acl_base_symbol.h"
#include "plugin/ascend/res_manager/symbol_interface/acl_rt_symbol.h"
#include "plugin/ascend/res_manager/symbol_interface/acl_symbol.h"
#include "plugin/ascend/res_manager/symbol_interface/symbol_utils.h"
#include "utils/ms_exception.h"

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

std::mutex ErrorManagerAdapter::initialized_mutex_;
bool ErrorManagerAdapter::initialized_ = false;

bool ErrorManagerAdapter::Init() {
  std::unique_lock<std::mutex> lock(initialized_mutex_);
  if (initialized_) {
    MS_LOG(DEBUG) << "Ascend error manager has been initialized.";
    return true;
  }
  LogWriter::SetMessageHandler(&MessageHandler);

  auto rt_ret = CALL_ASCEND_API(aclrtSetExceptionInfoCallback, TaskExceptionCallback);
  if (rt_ret != ACL_SUCCESS) {
    MS_LOG(EXCEPTION) << "Reg SetTaskFailCallback failed, error: " << rt_ret;
  }
  initialized_ = true;
  return true;
}

bool ErrorManagerAdapter::Finalize() {
  std::unique_lock<std::mutex> lock(initialized_mutex_);
  if (!initialized_) {
    return true;
  }
  (void)CALL_ASCEND_API(aclrtSetExceptionInfoCallback, nullptr);
  return true;
}

void ErrorManagerAdapter::TaskExceptionCallback(aclrtExceptionInfo *task_fail_info) {
  if (task_fail_info == nullptr) {
    MS_LOG(ERROR) << "Execute TaskFailCallback failed. task_fail_info is nullptr";
    return;
  }
  auto task_id = CALL_ASCEND_API(aclrtGetTaskIdFromExceptionInfo, task_fail_info);
  auto stream_id = CALL_ASCEND_API(aclrtGetStreamIdFromExceptionInfo, task_fail_info);
  auto error_code = CALL_ASCEND_API(aclrtGetErrorCodeFromExceptionInfo, task_fail_info);
  auto device_id = CALL_ASCEND_API(aclrtGetDeviceIdFromExceptionInfo, task_fail_info);
  auto tid = CALL_ASCEND_API(aclrtGetThreadIdFromExceptionInfo, task_fail_info);
  static auto fail_cb =
    GET_COMMON_CALLBACK(RunFailCallback, void, const char *, int, const char *, const std::string &, bool);
  if (fail_cb != nullptr) {
    fail_cb(FILE_NAME, __LINE__, __FUNCTION__, "Run task failed", false);
  }
  MS_LOG(ERROR) << "Run Task failed, task_id: " << task_id << ", stream_id: " << stream_id << ", tid: " << tid
                << ", device_id: " << device_id << ", retcode: " << error_code << " ("
                << GetErrorMsgFromErrorCode(error_code) << ")";
}

std::string ErrorManagerAdapter::GetErrorMessage(bool add_title) {
  int32_t device_id;
  if (CALL_ASCEND_API(aclrtGetDevice, &device_id) != ACL_SUCCESS) {
    MS_LOG(INFO) << "The device is not set yet, no need to fetch error from device.";
    return "";
  }
  const char *message = CALL_ASCEND_API(aclGetRecentErrMsg);
  const std::string error_message = message == nullptr ? "" : message;
  if (error_message.empty() || error_message.find(kUnknowErrorString) != std::string::npos) {
    return "";
  }
  if (add_title) {
    return "#umsg#Ascend Error Message:#umsg#" + error_message +
           "\n(Please search \"CANN Common Error Analysis\" at https://www.mindspore.cn/en for error code description)";
  }
  return error_message;
}

void ErrorManagerAdapter::MessageHandler(std::ostringstream *oss) {
  const auto &error_message = GetErrorMessage(true);
  if (!error_message.empty()) {
    *oss << error_message;
  }
}

std::string ErrorManagerAdapter::GetErrorMsgFromErrorCode(uint32_t rt_error_code) {
  auto find_iter = error_msg.find(rt_error_code);
  if (find_iter == error_msg.end()) {
    return "Return error code unknown, ret code: " + std::to_string(rt_error_code);
  }
  return find_iter->second;
}
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
