/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#include <string>
#include "transform/symbol/symbol_utils.h"
#include "transform/symbol/acl_rt_symbol.h"

namespace mindspore {
namespace transform {
aclrtCreateContextFunObj aclrtCreateContext_ = nullptr;
aclrtCreateEventFunObj aclrtCreateEvent_ = nullptr;
aclrtCreateEventWithFlagFunObj aclrtCreateEventWithFlag_ = nullptr;
aclrtCreateStreamWithConfigFunObj aclrtCreateStreamWithConfig_ = nullptr;
aclrtDestroyContextFunObj aclrtDestroyContext_ = nullptr;
aclrtDestroyEventFunObj aclrtDestroyEvent_ = nullptr;
aclrtDestroyStreamFunObj aclrtDestroyStream_ = nullptr;
aclrtEventElapsedTimeFunObj aclrtEventElapsedTime_ = nullptr;
aclrtFreeFunObj aclrtFree_ = nullptr;
aclrtFreeHostFunObj aclrtFreeHost_ = nullptr;
aclrtGetCurrentContextFunObj aclrtGetCurrentContext_ = nullptr;
aclrtGetDeviceFunObj aclrtGetDevice_ = nullptr;
aclrtGetDeviceCountFunObj aclrtGetDeviceCount_ = nullptr;
aclrtGetDeviceIdFromExceptionInfoFunObj aclrtGetDeviceIdFromExceptionInfo_ = nullptr;
aclrtGetErrorCodeFromExceptionInfoFunObj aclrtGetErrorCodeFromExceptionInfo_ = nullptr;
aclrtGetMemInfoFunObj aclrtGetMemInfo_ = nullptr;
aclrtGetRunModeFunObj aclrtGetRunMode_ = nullptr;
aclrtGetStreamIdFromExceptionInfoFunObj aclrtGetStreamIdFromExceptionInfo_ = nullptr;
aclrtGetTaskIdFromExceptionInfoFunObj aclrtGetTaskIdFromExceptionInfo_ = nullptr;
aclrtGetThreadIdFromExceptionInfoFunObj aclrtGetThreadIdFromExceptionInfo_ = nullptr;
aclrtLaunchCallbackFunObj aclrtLaunchCallback_ = nullptr;
aclrtMallocFunObj aclrtMalloc_ = nullptr;
aclrtMallocHostFunObj aclrtMallocHost_ = nullptr;
aclrtMemcpyFunObj aclrtMemcpy_ = nullptr;
aclrtMemcpyAsyncFunObj aclrtMemcpyAsync_ = nullptr;
aclrtMemsetFunObj aclrtMemset_ = nullptr;
aclrtProcessReportFunObj aclrtProcessReport_ = nullptr;
aclrtQueryEventStatusFunObj aclrtQueryEventStatus_ = nullptr;
aclrtRecordEventFunObj aclrtRecordEvent_ = nullptr;
aclrtResetDeviceFunObj aclrtResetDevice_ = nullptr;
aclrtResetEventFunObj aclrtResetEvent_ = nullptr;
aclrtSetCurrentContextFunObj aclrtSetCurrentContext_ = nullptr;
aclrtSetDeviceFunObj aclrtSetDevice_ = nullptr;
aclrtSetDeviceSatModeFunObj aclrtSetDeviceSatMode_ = nullptr;
aclrtSetExceptionInfoCallbackFunObj aclrtSetExceptionInfoCallback_ = nullptr;
aclrtSetOpExecuteTimeOutFunObj aclrtSetOpExecuteTimeOut_ = nullptr;
aclrtSetOpWaitTimeoutFunObj aclrtSetOpWaitTimeout_ = nullptr;
aclrtSetStreamFailureModeFunObj aclrtSetStreamFailureMode_ = nullptr;
aclrtStreamQueryFunObj aclrtStreamQuery_ = nullptr;
aclrtStreamWaitEventFunObj aclrtStreamWaitEvent_ = nullptr;
aclrtSubscribeReportFunObj aclrtSubscribeReport_ = nullptr;
aclrtSynchronizeEventFunObj aclrtSynchronizeEvent_ = nullptr;
aclrtSynchronizeStreamFunObj aclrtSynchronizeStream_ = nullptr;
aclrtSynchronizeStreamWithTimeoutFunObj aclrtSynchronizeStreamWithTimeout_ = nullptr;

void LoadAclRtApiSymbol(const std::string &ascend_path) {
  std::string aclrt_plugin_path = "lib64/libascendcl.so";
  auto handler = GetLibHandler(ascend_path + aclrt_plugin_path);
  if (handler == nullptr) {
    MS_LOG(WARNING) << "Dlopen " << aclrt_plugin_path << " failed!" << dlerror();
    return;
  }
  aclrtCreateContext_ = DlsymAscendFuncObj(aclrtCreateContext, handler);
  aclrtCreateEvent_ = DlsymAscendFuncObj(aclrtCreateEvent, handler);
  aclrtCreateEventWithFlag_ = DlsymAscendFuncObj(aclrtCreateEventWithFlag, handler);
  aclrtCreateStreamWithConfig_ = DlsymAscendFuncObj(aclrtCreateStreamWithConfig, handler);
  aclrtDestroyContext_ = DlsymAscendFuncObj(aclrtDestroyContext, handler);
  aclrtDestroyEvent_ = DlsymAscendFuncObj(aclrtDestroyEvent, handler);
  aclrtDestroyStream_ = DlsymAscendFuncObj(aclrtDestroyStream, handler);
  aclrtEventElapsedTime_ = DlsymAscendFuncObj(aclrtEventElapsedTime, handler);
  aclrtFree_ = DlsymAscendFuncObj(aclrtFree, handler);
  aclrtFreeHost_ = DlsymAscendFuncObj(aclrtFreeHost, handler);
  aclrtGetCurrentContext_ = DlsymAscendFuncObj(aclrtGetCurrentContext, handler);
  aclrtGetDevice_ = DlsymAscendFuncObj(aclrtGetDevice, handler);
  aclrtGetDeviceCount_ = DlsymAscendFuncObj(aclrtGetDeviceCount, handler);
  aclrtGetDeviceIdFromExceptionInfo_ = DlsymAscendFuncObj(aclrtGetDeviceIdFromExceptionInfo, handler);
  aclrtGetErrorCodeFromExceptionInfo_ = DlsymAscendFuncObj(aclrtGetErrorCodeFromExceptionInfo, handler);
  aclrtGetMemInfo_ = DlsymAscendFuncObj(aclrtGetMemInfo, handler);
  aclrtGetRunMode_ = DlsymAscendFuncObj(aclrtGetRunMode, handler);
  aclrtGetStreamIdFromExceptionInfo_ = DlsymAscendFuncObj(aclrtGetStreamIdFromExceptionInfo, handler);
  aclrtGetTaskIdFromExceptionInfo_ = DlsymAscendFuncObj(aclrtGetTaskIdFromExceptionInfo, handler);
  aclrtGetThreadIdFromExceptionInfo_ = DlsymAscendFuncObj(aclrtGetThreadIdFromExceptionInfo, handler);
  aclrtLaunchCallback_ = DlsymAscendFuncObj(aclrtLaunchCallback, handler);
  aclrtMalloc_ = DlsymAscendFuncObj(aclrtMalloc, handler);
  aclrtMallocHost_ = DlsymAscendFuncObj(aclrtMallocHost, handler);
  aclrtMemcpy_ = DlsymAscendFuncObj(aclrtMemcpy, handler);
  aclrtMemcpyAsync_ = DlsymAscendFuncObj(aclrtMemcpyAsync, handler);
  aclrtMemset_ = DlsymAscendFuncObj(aclrtMemset, handler);
  aclrtProcessReport_ = DlsymAscendFuncObj(aclrtProcessReport, handler);
  aclrtQueryEventStatus_ = DlsymAscendFuncObj(aclrtQueryEventStatus, handler);
  aclrtRecordEvent_ = DlsymAscendFuncObj(aclrtRecordEvent, handler);
  aclrtResetDevice_ = DlsymAscendFuncObj(aclrtResetDevice, handler);
  aclrtResetEvent_ = DlsymAscendFuncObj(aclrtResetEvent, handler);
  aclrtSetCurrentContext_ = DlsymAscendFuncObj(aclrtSetCurrentContext, handler);
  aclrtSetDevice_ = DlsymAscendFuncObj(aclrtSetDevice, handler);
  aclrtSetDeviceSatMode_ = DlsymAscendFuncObj(aclrtSetDeviceSatMode, handler);
  aclrtSetExceptionInfoCallback_ = DlsymAscendFuncObj(aclrtSetExceptionInfoCallback, handler);
  aclrtSetOpExecuteTimeOut_ = DlsymAscendFuncObj(aclrtSetOpExecuteTimeOut, handler);
  aclrtSetOpWaitTimeout_ = DlsymAscendFuncObj(aclrtSetOpWaitTimeout, handler);
  aclrtSetStreamFailureMode_ = DlsymAscendFuncObj(aclrtSetStreamFailureMode, handler);
  aclrtStreamQuery_ = DlsymAscendFuncObj(aclrtStreamQuery, handler);
  aclrtStreamWaitEvent_ = DlsymAscendFuncObj(aclrtStreamWaitEvent, handler);
  aclrtSubscribeReport_ = DlsymAscendFuncObj(aclrtSubscribeReport, handler);
  aclrtSynchronizeEvent_ = DlsymAscendFuncObj(aclrtSynchronizeEvent, handler);
  aclrtSynchronizeStream_ = DlsymAscendFuncObj(aclrtSynchronizeStream, handler);
  aclrtSynchronizeStreamWithTimeout_ = DlsymAscendFuncObj(aclrtSynchronizeStreamWithTimeout, handler);
  MS_LOG(INFO) << "Load acl rt api success!";
}

}  // namespace transform
}  // namespace mindspore
