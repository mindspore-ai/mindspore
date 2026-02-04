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
#include "acl_rt_symbol.h"
#include <string>
#include "symbol_utils.h"

namespace mindspore::device::ascend {
aclrtCreateContextFunObj aclrtCreateContext_ = nullptr;
aclrtCreateEventWithFlagFunObj aclrtCreateEventWithFlag_ = nullptr;
aclrtCreateEventExWithFlagFunObj aclrtCreateEventExWithFlag_ = nullptr;
aclrtCreateStreamFunObj aclrtCreateStream_ = nullptr;
aclrtCreateStreamWithConfigFunObj aclrtCreateStreamWithConfig_ = nullptr;
aclrtGetResInCurrentThreadFunObj aclrtGetResInCurrentThread_ = nullptr;
aclrtDestroyContextFunObj aclrtDestroyContext_ = nullptr;
aclrtDestroyEventFunObj aclrtDestroyEvent_ = nullptr;
aclrtDestroyStreamFunObj aclrtDestroyStream_ = nullptr;
aclrtDestroyStreamForceFunObj aclrtDestroyStreamForce_ = nullptr;
aclrtGetDeviceResLimitFunObj aclrtGetDeviceResLimit_ = nullptr;
aclrtSetDeviceResLimitFunObj aclrtSetDeviceResLimit_ = nullptr;
aclrtGetStreamResLimitFunObj aclrtGetStreamResLimit_ = nullptr;
aclrtSetStreamResLimitFunObj aclrtSetStreamResLimit_ = nullptr;
aclrtResetStreamResLimitFunObj aclrtResetStreamResLimit_ = nullptr;
aclrtUseStreamResInCurrentThreadFunObj aclrtUseStreamResInCurrentThread_ = nullptr;
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
aclrtMallocAlign32FunObj aclrtMallocAlign32_ = nullptr;
aclrtMallocHostFunObj aclrtMallocHost_ = nullptr;
aclrtMemcpyFunObj aclrtMemcpy_ = nullptr;
aclrtMemcpyAsyncFunObj aclrtMemcpyAsync_ = nullptr;
aclrtMemsetFunObj aclrtMemset_ = nullptr;
aclrtMemsetAsyncFunObj aclrtMemsetAsync_ = nullptr;
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
aclrtSynchronizeDeviceWithTimeoutFunObj aclrtSynchronizeDeviceWithTimeout_ = nullptr;
aclrtUnmapMemFunObj aclrtUnmapMem_ = nullptr;
aclrtReserveMemAddressFunObj aclrtReserveMemAddress_ = nullptr;
aclrtMallocPhysicalFunObj aclrtMallocPhysical_ = nullptr;
aclrtMapMemFunObj aclrtMapMem_ = nullptr;
aclrtFreePhysicalFunObj aclrtFreePhysical_ = nullptr;
aclrtReleaseMemAddressFunObj aclrtReleaseMemAddress_ = nullptr;
aclrtCtxSetSysParamOptFunObj aclrtCtxSetSysParamOpt_ = nullptr;
aclrtGetMemUceInfoFunObj aclrtGetMemUceInfo_ = nullptr;
aclrtDeviceTaskAbortFunObj aclrtDeviceTaskAbort_ = nullptr;
aclrtDeviceCanAccessPeerFunObj aclrtDeviceCanAccessPeer_ = nullptr;
aclrtDeviceEnablePeerAccessFunObj aclrtDeviceEnablePeerAccess_ = nullptr;
aclrtMemUceRepairFunObj aclrtMemUceRepair_ = nullptr;
aclrtEventGetTimestampFunObj aclrtEventGetTimestamp_ = nullptr;
aclrtDeviceGetBareTgidFunObj aclrtDeviceGetBareTgid_ = nullptr;
aclrtMemExportToShareableHandleFunObj aclrtMemExportToShareableHandle_ = nullptr;
aclrtMemSetPidToShareableHandleFunObj aclrtMemSetPidToShareableHandle_ = nullptr;
aclrtMemImportFromShareableHandleFunObj aclrtMemImportFromShareableHandle_ = nullptr;
aclrtGetLastErrorFunObj aclrtGetLastError_ = nullptr;

void LoadAclRtApiSymbol(const std::string &ascend_path) {
  std::string aclrt_plugin_path = ascend_path + "lib64/libascendcl.so";
  auto handler = GetLibHandler(aclrt_plugin_path);
  if (handler == nullptr) {
    MS_LOG(WARNING) << "Dlopen " << aclrt_plugin_path << " failed!" << dlerror();
    return;
  }
  aclrtCreateContext_ = DlsymAscendFuncObj(aclrtCreateContext, handler);
  aclrtCreateEventWithFlag_ = DlsymAscendFuncObj(aclrtCreateEventWithFlag, handler);
  aclrtCreateEventExWithFlag_ = DlsymAscendFuncObj(aclrtCreateEventExWithFlag, handler);
  aclrtCreateStream_ = DlsymAscendFuncObj(aclrtCreateStream, handler);
  aclrtCreateStreamWithConfig_ = DlsymAscendFuncObj(aclrtCreateStreamWithConfig, handler);
  aclrtGetDeviceResLimit_ = DlsymAscendFuncObj(aclrtGetDeviceResLimit, handler);
  aclrtSetDeviceResLimit_ = DlsymAscendFuncObj(aclrtSetDeviceResLimit, handler);
  aclrtGetStreamResLimit_ = DlsymAscendFuncObj(aclrtGetStreamResLimit, handler);
  aclrtSetStreamResLimit_ = DlsymAscendFuncObj(aclrtSetStreamResLimit, handler);
  aclrtResetStreamResLimit_ = DlsymAscendFuncObj(aclrtResetStreamResLimit, handler);
  aclrtUseStreamResInCurrentThread_ = DlsymAscendFuncObj(aclrtUseStreamResInCurrentThread, handler);
  aclrtGetResInCurrentThread_ = DlsymAscendFuncObj(aclrtGetResInCurrentThread, handler);
  aclrtDestroyContext_ = DlsymAscendFuncObj(aclrtDestroyContext, handler);
  aclrtDestroyEvent_ = DlsymAscendFuncObj(aclrtDestroyEvent, handler);
  aclrtDestroyStream_ = DlsymAscendFuncObj(aclrtDestroyStream, handler);
  aclrtDestroyStreamForce_ = DlsymAscendFuncObj(aclrtDestroyStreamForce, handler);
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
  aclrtMallocAlign32_ = DlsymAscendFuncObj(aclrtMallocAlign32, handler);
  aclrtMallocHost_ = DlsymAscendFuncObj(aclrtMallocHost, handler);
  aclrtMemcpy_ = DlsymAscendFuncObj(aclrtMemcpy, handler);
  aclrtMemcpyAsync_ = DlsymAscendFuncObj(aclrtMemcpyAsync, handler);
  aclrtMemset_ = DlsymAscendFuncObj(aclrtMemset, handler);
  aclrtMemsetAsync_ = DlsymAscendFuncObj(aclrtMemsetAsync, handler);
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
  aclrtSynchronizeDeviceWithTimeout_ = DlsymAscendFuncObj(aclrtSynchronizeDeviceWithTimeout, handler);
  aclrtUnmapMem_ = DlsymAscendFuncObj(aclrtUnmapMem, handler);
  aclrtReserveMemAddress_ = DlsymAscendFuncObj(aclrtReserveMemAddress, handler);
  aclrtMallocPhysical_ = DlsymAscendFuncObj(aclrtMallocPhysical, handler);
  aclrtMapMem_ = DlsymAscendFuncObj(aclrtMapMem, handler);
  aclrtFreePhysical_ = DlsymAscendFuncObj(aclrtFreePhysical, handler);
  aclrtReleaseMemAddress_ = DlsymAscendFuncObj(aclrtReleaseMemAddress, handler);
  aclrtCtxSetSysParamOpt_ = DlsymAscendFuncObj(aclrtCtxSetSysParamOpt, handler);
  aclrtGetMemUceInfo_ = DlsymAscendFuncObj(aclrtGetMemUceInfo, handler);
  aclrtDeviceTaskAbort_ = DlsymAscendFuncObj(aclrtDeviceTaskAbort, handler);
  aclrtDeviceCanAccessPeer_ = DlsymAscendFuncObj(aclrtDeviceCanAccessPeer, handler);
  aclrtDeviceEnablePeerAccess_ = DlsymAscendFuncObj(aclrtDeviceEnablePeerAccess, handler);
  aclrtMemUceRepair_ = DlsymAscendFuncObj(aclrtMemUceRepair, handler);
  aclrtEventGetTimestamp_ = DlsymAscendFuncObj(aclrtEventGetTimestamp, handler);
  aclrtDeviceGetBareTgid_ = DlsymAscendFuncObj(aclrtDeviceGetBareTgid, handler);
  aclrtMemExportToShareableHandle_ = DlsymAscendFuncObj(aclrtMemExportToShareableHandle, handler);
  aclrtMemSetPidToShareableHandle_ = DlsymAscendFuncObj(aclrtMemSetPidToShareableHandle, handler);
  aclrtMemImportFromShareableHandle_ = DlsymAscendFuncObj(aclrtMemImportFromShareableHandle, handler);
  aclrtGetLastError_ = DlsymAscendFuncObj(aclrtGetLastError, handler);
  MS_LOG(INFO) << "Load acl rt api success!";
}

void LoadSimulationRtApi() {
  ASSIGN_SIMU(aclrtCreateContext);
  ASSIGN_SIMU(aclrtCreateEventWithFlag);
  ASSIGN_SIMU(aclrtCreateEventExWithFlag);
  ASSIGN_SIMU(aclrtCreateStream);
  ASSIGN_SIMU(aclrtGetResInCurrentThread);
  ASSIGN_SIMU(aclrtCreateStreamWithConfig);
  ASSIGN_SIMU(aclrtGetDeviceResLimit);
  ASSIGN_SIMU(aclrtSetDeviceResLimit);
  ASSIGN_SIMU(aclrtSetStreamResLimit);
  ASSIGN_SIMU(aclrtSetStreamResLimit);
  ASSIGN_SIMU(aclrtResetStreamResLimit);
  ASSIGN_SIMU(aclrtDestroyContext);
  ASSIGN_SIMU(aclrtDestroyEvent);
  ASSIGN_SIMU(aclrtDestroyStream);
  ASSIGN_SIMU(aclrtDestroyStreamForce);
  ASSIGN_SIMU(aclrtEventElapsedTime);
  ASSIGN_SIMU(aclrtFree);
  ASSIGN_SIMU(aclrtFreeHost);
  ASSIGN_SIMU(aclrtGetCurrentContext);
  ASSIGN_SIMU(aclrtGetDevice);
  ASSIGN_SIMU(aclrtGetDeviceCount);
  ASSIGN_SIMU(aclrtGetDeviceIdFromExceptionInfo);
  ASSIGN_SIMU(aclrtGetErrorCodeFromExceptionInfo);
  ASSIGN_SIMU(aclrtGetMemInfo);
  ASSIGN_SIMU(aclrtGetRunMode);
  ASSIGN_SIMU(aclrtGetStreamIdFromExceptionInfo);
  ASSIGN_SIMU(aclrtGetTaskIdFromExceptionInfo);
  ASSIGN_SIMU(aclrtGetThreadIdFromExceptionInfo);
  ASSIGN_SIMU(aclrtLaunchCallback);
  ASSIGN_SIMU(aclrtMalloc);
  ASSIGN_SIMU(aclrtMallocAlign32);
  ASSIGN_SIMU(aclrtMallocHost);
  ASSIGN_SIMU(aclrtMemcpy);
  ASSIGN_SIMU(aclrtMemcpyAsync);
  ASSIGN_SIMU(aclrtMemset);
  ASSIGN_SIMU(aclrtMemsetAsync);
  ASSIGN_SIMU(aclrtProcessReport);
  ASSIGN_SIMU(aclrtQueryEventStatus);
  ASSIGN_SIMU(aclrtRecordEvent);
  ASSIGN_SIMU(aclrtResetDevice);
  ASSIGN_SIMU(aclrtResetEvent);
  ASSIGN_SIMU(aclrtSetCurrentContext);
  ASSIGN_SIMU(aclrtSetDevice);
  ASSIGN_SIMU(aclrtSetDeviceSatMode);
  ASSIGN_SIMU(aclrtSetExceptionInfoCallback);
  ASSIGN_SIMU(aclrtSetOpExecuteTimeOut);
  ASSIGN_SIMU(aclrtSetOpWaitTimeout);
  ASSIGN_SIMU(aclrtSetStreamFailureMode);
  ASSIGN_SIMU(aclrtStreamQuery);
  ASSIGN_SIMU(aclrtStreamWaitEvent);
  ASSIGN_SIMU(aclrtSubscribeReport);
  ASSIGN_SIMU(aclrtSynchronizeEvent);
  ASSIGN_SIMU(aclrtSynchronizeStream);
  ASSIGN_SIMU(aclrtSynchronizeStreamWithTimeout);
  ASSIGN_SIMU(aclrtSynchronizeDeviceWithTimeout);
  ASSIGN_SIMU(aclrtUnmapMem);
  ASSIGN_SIMU(aclrtReserveMemAddress);
  ASSIGN_SIMU(aclrtMallocPhysical);
  ASSIGN_SIMU(aclrtMapMem);
  ASSIGN_SIMU(aclrtFreePhysical);
  ASSIGN_SIMU(aclrtReleaseMemAddress);
  ASSIGN_SIMU(aclrtCtxSetSysParamOpt);
  ASSIGN_SIMU(aclrtGetMemUceInfo);
  ASSIGN_SIMU(aclrtDeviceTaskAbort);
  ASSIGN_SIMU(aclrtDeviceCanAccessPeer);
  ASSIGN_SIMU(aclrtDeviceEnablePeerAccess);
  ASSIGN_SIMU(aclrtMemUceRepair);
  ASSIGN_SIMU(aclrtEventGetTimestamp);
  ASSIGN_SIMU(aclrtDeviceGetBareTgid);
  ASSIGN_SIMU(aclrtMemExportToShareableHandle);
  ASSIGN_SIMU(aclrtMemSetPidToShareableHandle);
  ASSIGN_SIMU(aclrtMemImportFromShareableHandle);
  ASSIGN_SIMU(aclrtGetLastError);
}
}  // namespace mindspore::device::ascend
