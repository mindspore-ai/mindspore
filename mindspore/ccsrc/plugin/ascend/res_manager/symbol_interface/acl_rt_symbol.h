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
#ifndef MINDSPORE_CCSRC_TRANSFORM_SYMBOL_ACL_RT_SYMBOL_H_
#define MINDSPORE_CCSRC_TRANSFORM_SYMBOL_ACL_RT_SYMBOL_H_
#include <string>
#include "acl/acl_rt.h"
#include "utils/dlopen_macro.h"

namespace mindspore::device::ascend {
ORIGIN_METHOD_WITH_SIMU_CREATE(aclrtCreateContext, aclError, aclrtContext *, int32_t)
ORIGIN_METHOD_WITH_SIMU_CREATE(aclrtCreateEventWithFlag, aclError, aclrtEvent *, uint32_t)
ORIGIN_METHOD_WITH_SIMU_CREATE(aclrtCreateEventExWithFlag, aclError, aclrtEvent *, uint32_t)
ORIGIN_METHOD_WITH_SIMU_CREATE(aclrtCreateStream, aclError, aclrtStream *)
ORIGIN_METHOD_WITH_SIMU_CREATE(aclrtCreateStreamWithConfig, aclError, aclrtStream *, uint32_t, uint32_t)

// Compilation Macro compatible with older CANN versions. enable adaptation later according to requirements.
#ifndef CUSTOM_ASCEND_OP
ORIGIN_METHOD_WITH_SIMU(aclrtGetDeviceResLimit, aclError, int32_t, aclrtDevResLimitType, uint32_t *)
ORIGIN_METHOD_WITH_SIMU(aclrtSetDeviceResLimit, aclError, int32_t, aclrtDevResLimitType, uint32_t)
ORIGIN_METHOD_WITH_SIMU(aclrtGetStreamResLimit, aclError, aclrtStream, aclrtDevResLimitType, uint32_t *)
ORIGIN_METHOD_WITH_SIMU(aclrtSetStreamResLimit, aclError, aclrtStream, aclrtDevResLimitType, uint32_t)
ORIGIN_METHOD_WITH_SIMU(aclrtResetStreamResLimit, aclError, aclrtStream)
ORIGIN_METHOD_WITH_SIMU(aclrtUseStreamResInCurrentThread, aclError, aclrtStream)
ORIGIN_METHOD_WITH_SIMU(aclrtGetResInCurrentThread, aclError, aclrtDevResLimitType, uint32_t *)
#endif  // CUSTOM_ASCEND_OP

ORIGIN_METHOD_WITH_SIMU(aclrtDestroyContext, aclError, aclrtContext)
ORIGIN_METHOD_WITH_SIMU(aclrtDestroyEvent, aclError, aclrtEvent)
ORIGIN_METHOD_WITH_SIMU(aclrtDestroyStream, aclError, aclrtStream)
ORIGIN_METHOD_WITH_SIMU(aclrtDestroyStreamForce, aclError, aclrtStream)
ORIGIN_METHOD_WITH_SIMU_CREATE(aclrtEventElapsedTime, aclError, float *, aclrtEvent, aclrtEvent)
ORIGIN_METHOD_WITH_SIMU(aclrtFree, aclError, void *)
ORIGIN_METHOD_WITH_SIMU(aclrtFreeHost, aclError, void *)
ORIGIN_METHOD_WITH_SIMU_CREATE(aclrtGetCurrentContext, aclError, aclrtContext *)
ORIGIN_METHOD_WITH_SIMU_CREATE(aclrtGetDevice, aclError, int32_t *)
ORIGIN_METHOD_WITH_SIMU_CREATE(aclrtGetDeviceCount, aclError, uint32_t *)
ORIGIN_METHOD_WITH_SIMU(aclrtGetDeviceIdFromExceptionInfo, uint32_t, const aclrtExceptionInfo *)
ORIGIN_METHOD_WITH_SIMU(aclrtGetErrorCodeFromExceptionInfo, uint32_t, const aclrtExceptionInfo *)
ORIGIN_METHOD_WITH_SIMU(aclrtGetMemInfo, aclError, aclrtMemAttr, size_t *, size_t *)
ORIGIN_METHOD_WITH_SIMU_CREATE(aclrtGetRunMode, aclError, aclrtRunMode *)
ORIGIN_METHOD_WITH_SIMU(aclrtGetStreamIdFromExceptionInfo, uint32_t, const aclrtExceptionInfo *)
ORIGIN_METHOD_WITH_SIMU(aclrtGetTaskIdFromExceptionInfo, uint32_t, const aclrtExceptionInfo *)
ORIGIN_METHOD_WITH_SIMU(aclrtGetThreadIdFromExceptionInfo, uint32_t, const aclrtExceptionInfo *)
ORIGIN_METHOD_WITH_SIMU(aclrtLaunchCallback, aclError, aclrtCallback, void *, aclrtCallbackBlockType, aclrtStream)
ORIGIN_METHOD_WITH_SIMU_CREATE(aclrtMalloc, aclError, void **, size_t, aclrtMemMallocPolicy)
ORIGIN_METHOD_WITH_SIMU_CREATE(aclrtMallocAlign32, aclError, void **, size_t, aclrtMemMallocPolicy)
ORIGIN_METHOD_WITH_SIMU_CREATE(aclrtMallocHost, aclError, void **, size_t)
ORIGIN_METHOD_WITH_SIMU(aclrtMemcpy, aclError, void *, size_t, const void *, size_t, aclrtMemcpyKind)
ORIGIN_METHOD_WITH_SIMU(aclrtMemcpyAsync, aclError, void *, size_t, const void *, size_t, aclrtMemcpyKind, aclrtStream)
ORIGIN_METHOD_WITH_SIMU(aclrtMemset, aclError, void *, size_t, int32_t, size_t)
ORIGIN_METHOD_WITH_SIMU(aclrtMemsetAsync, aclError, void *, size_t, int32_t, size_t, aclrtStream)
ORIGIN_METHOD_WITH_SIMU(aclrtProcessReport, aclError, int32_t)
ORIGIN_METHOD_WITH_SIMU(aclrtQueryEventStatus, aclError, aclrtEvent, aclrtEventRecordedStatus *)
ORIGIN_METHOD_WITH_SIMU(aclrtRecordEvent, aclError, aclrtEvent, aclrtStream)
ORIGIN_METHOD_WITH_SIMU(aclrtResetDevice, aclError, int32_t)
ORIGIN_METHOD_WITH_SIMU(aclrtResetEvent, aclError, aclrtEvent, aclrtStream)
ORIGIN_METHOD_WITH_SIMU(aclrtSetCurrentContext, aclError, aclrtContext)
ORIGIN_METHOD_WITH_SIMU(aclrtSetDevice, aclError, int32_t)
ORIGIN_METHOD_WITH_SIMU(aclrtSetDeviceSatMode, aclError, aclrtFloatOverflowMode)
ORIGIN_METHOD_WITH_SIMU(aclrtSetExceptionInfoCallback, aclError, aclrtExceptionInfoCallback)
ORIGIN_METHOD_WITH_SIMU(aclrtSetOpExecuteTimeOut, aclError, uint32_t)
ORIGIN_METHOD_WITH_SIMU(aclrtSetOpWaitTimeout, aclError, uint32_t)
ORIGIN_METHOD_WITH_SIMU(aclrtSetStreamFailureMode, aclError, aclrtStream, uint64_t)
ORIGIN_METHOD_WITH_SIMU(aclrtStreamQuery, aclError, aclrtStream, aclrtStreamStatus *)
ORIGIN_METHOD_WITH_SIMU(aclrtStreamWaitEvent, aclError, aclrtStream, aclrtEvent)
ORIGIN_METHOD_WITH_SIMU(aclrtSubscribeReport, aclError, uint64_t, aclrtStream)
ORIGIN_METHOD_WITH_SIMU(aclrtSynchronizeEvent, aclError, aclrtEvent)
ORIGIN_METHOD_WITH_SIMU(aclrtSynchronizeStream, aclError, aclrtStream)
ORIGIN_METHOD_WITH_SIMU(aclrtSynchronizeStreamWithTimeout, aclError, aclrtStream, int32_t)
ORIGIN_METHOD_WITH_SIMU(aclrtSynchronizeDeviceWithTimeout, aclError, int32_t)
ORIGIN_METHOD_WITH_SIMU(aclrtUnmapMem, aclError, void *)
ORIGIN_METHOD_WITH_SIMU(aclrtReserveMemAddress, aclError, void **, size_t, size_t, void *, uint64_t)
ORIGIN_METHOD_WITH_SIMU(aclrtMallocPhysical, aclError, aclrtDrvMemHandle *, size_t, const aclrtPhysicalMemProp *,
                        uint64_t)
ORIGIN_METHOD_WITH_SIMU(aclrtMapMem, aclError, void *, size_t, size_t, aclrtDrvMemHandle, uint64_t)
ORIGIN_METHOD_WITH_SIMU(aclrtFreePhysical, aclError, aclrtDrvMemHandle)
ORIGIN_METHOD_WITH_SIMU(aclrtReleaseMemAddress, aclError, void *)
ORIGIN_METHOD_WITH_SIMU(aclrtCtxSetSysParamOpt, aclError, aclSysParamOpt, int64_t)
ORIGIN_METHOD_WITH_SIMU(aclrtGetMemUceInfo, aclError, int32_t, aclrtMemUceInfo *, size_t, size_t *)
ORIGIN_METHOD_WITH_SIMU(aclrtDeviceTaskAbort, aclError, int32_t, uint32_t)
ORIGIN_METHOD_WITH_SIMU(aclrtDeviceCanAccessPeer, aclError, int32_t *, int32_t, int32_t)
ORIGIN_METHOD_WITH_SIMU(aclrtDeviceEnablePeerAccess, aclError, int32_t, uint32_t)
ORIGIN_METHOD_WITH_SIMU(aclrtMemUceRepair, aclError, int32_t, aclrtMemUceInfo *, size_t)
ORIGIN_METHOD_WITH_SIMU(aclrtEventGetTimestamp, aclError, aclrtEvent, uint64_t *)
ORIGIN_METHOD_WITH_SIMU(aclrtDeviceGetBareTgid, aclError, int32_t *)
ORIGIN_METHOD_WITH_SIMU(aclrtMemExportToShareableHandle, aclError, aclrtDrvMemHandle, aclrtMemHandleType, uint64_t,
                        uint64_t *)
ORIGIN_METHOD_WITH_SIMU(aclrtMemSetPidToShareableHandle, aclError, uint64_t, int32_t *, size_t)
ORIGIN_METHOD_WITH_SIMU(aclrtMemImportFromShareableHandle, aclError, uint64_t, int32_t, aclrtDrvMemHandle *)
ORIGIN_METHOD_WITH_SIMU(aclrtGetLastError, aclError, aclrtLastErrLevel)

void LoadAclRtApiSymbol(const std::string &ascend_path);
void LoadSimulationRtApi();
}  // namespace mindspore::device::ascend

#endif  // MINDSPORE_CCSRC_TRANSFORM_SYMBOL_ACL_RT_SYMBOL_H_
