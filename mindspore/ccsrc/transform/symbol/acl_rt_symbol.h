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

namespace mindspore {
namespace transform {

ORIGIN_METHOD(aclrtCreateContext, aclError, aclrtContext *, int32_t)
ORIGIN_METHOD(aclrtCreateEvent, aclError, aclrtEvent *)
ORIGIN_METHOD(aclrtCreateEventWithFlag, aclError, aclrtEvent *, uint32_t)
ORIGIN_METHOD(aclrtCreateStreamWithConfig, aclError, aclrtStream *, uint32_t, uint32_t)
ORIGIN_METHOD(aclrtDestroyContext, aclError, aclrtContext)
ORIGIN_METHOD(aclrtDestroyEvent, aclError, aclrtEvent)
ORIGIN_METHOD(aclrtDestroyStream, aclError, aclrtStream)
ORIGIN_METHOD(aclrtEventElapsedTime, aclError, float *, aclrtEvent, aclrtEvent)
ORIGIN_METHOD(aclrtFree, aclError, void *)
ORIGIN_METHOD(aclrtFreeHost, aclError, void *)
ORIGIN_METHOD(aclrtGetCurrentContext, aclError, aclrtContext *)
ORIGIN_METHOD(aclrtGetDeviceCount, aclError, uint32_t *)
ORIGIN_METHOD(aclrtGetDeviceIdFromExceptionInfo, uint32_t, const aclrtExceptionInfo *)
ORIGIN_METHOD(aclrtGetErrorCodeFromExceptionInfo, uint32_t, const aclrtExceptionInfo *)
ORIGIN_METHOD(aclrtGetMemInfo, aclError, aclrtMemAttr, size_t *, size_t *)
ORIGIN_METHOD(aclrtGetRunMode, aclError, aclrtRunMode *)
ORIGIN_METHOD(aclrtGetStreamIdFromExceptionInfo, uint32_t, const aclrtExceptionInfo *)
ORIGIN_METHOD(aclrtGetTaskIdFromExceptionInfo, uint32_t, const aclrtExceptionInfo *)
ORIGIN_METHOD(aclrtGetThreadIdFromExceptionInfo, uint32_t, const aclrtExceptionInfo *)
ORIGIN_METHOD(aclrtLaunchCallback, aclError, aclrtCallback, void *, aclrtCallbackBlockType, aclrtStream)
ORIGIN_METHOD(aclrtMalloc, aclError, void **, size_t, aclrtMemMallocPolicy)
ORIGIN_METHOD(aclrtMallocHost, aclError, void **, size_t)
ORIGIN_METHOD(aclrtMemcpy, aclError, void *, size_t, const void *, size_t, aclrtMemcpyKind)
ORIGIN_METHOD(aclrtMemcpyAsync, aclError, void *, size_t, const void *, size_t, aclrtMemcpyKind, aclrtStream)
ORIGIN_METHOD(aclrtMemset, aclError, void *, size_t, int32_t, size_t)
ORIGIN_METHOD(aclrtProcessReport, aclError, int32_t)
ORIGIN_METHOD(aclrtQueryEventStatus, aclError, aclrtEvent, aclrtEventRecordedStatus *)
ORIGIN_METHOD(aclrtRecordEvent, aclError, aclrtEvent, aclrtStream)
ORIGIN_METHOD(aclrtResetDevice, aclError, int32_t)
ORIGIN_METHOD(aclrtResetEvent, aclError, aclrtEvent, aclrtStream)
ORIGIN_METHOD(aclrtSetCurrentContext, aclError, aclrtContext)
ORIGIN_METHOD(aclrtSetDevice, aclError, int32_t)
ORIGIN_METHOD(aclrtSetDeviceSatMode, aclError, aclrtFloatOverflowMode)
ORIGIN_METHOD(aclrtSetExceptionInfoCallback, aclError, aclrtExceptionInfoCallback)
ORIGIN_METHOD(aclrtSetOpExecuteTimeOut, aclError, uint32_t)
ORIGIN_METHOD(aclrtSetOpWaitTimeout, aclError, uint32_t)
ORIGIN_METHOD(aclrtSetStreamFailureMode, aclError, aclrtStream, uint64_t)
ORIGIN_METHOD(aclrtStreamQuery, aclError, aclrtStream, aclrtStreamStatus *)
ORIGIN_METHOD(aclrtStreamWaitEvent, aclError, aclrtStream, aclrtEvent)
ORIGIN_METHOD(aclrtSubscribeReport, aclError, uint64_t, aclrtStream)
ORIGIN_METHOD(aclrtSynchronizeEvent, aclError, aclrtEvent)
ORIGIN_METHOD(aclrtSynchronizeStream, aclError, aclrtStream)
ORIGIN_METHOD(aclrtSynchronizeStreamWithTimeout, aclError, aclrtStream, int32_t)

extern aclrtCreateContextFunObj aclrtCreateContext_;
extern aclrtCreateEventFunObj aclrtCreateEvent_;
extern aclrtCreateEventWithFlagFunObj aclrtCreateEventWithFlag_;
extern aclrtCreateStreamWithConfigFunObj aclrtCreateStreamWithConfig_;
extern aclrtDestroyContextFunObj aclrtDestroyContext_;
extern aclrtDestroyEventFunObj aclrtDestroyEvent_;
extern aclrtDestroyStreamFunObj aclrtDestroyStream_;
extern aclrtEventElapsedTimeFunObj aclrtEventElapsedTime_;
extern aclrtFreeFunObj aclrtFree_;
extern aclrtFreeHostFunObj aclrtFreeHost_;
extern aclrtGetCurrentContextFunObj aclrtGetCurrentContext_;
extern aclrtGetDeviceCountFunObj aclrtGetDeviceCount_;
extern aclrtGetDeviceIdFromExceptionInfoFunObj aclrtGetDeviceIdFromExceptionInfo_;
extern aclrtGetErrorCodeFromExceptionInfoFunObj aclrtGetErrorCodeFromExceptionInfo_;
extern aclrtGetMemInfoFunObj aclrtGetMemInfo_;
extern aclrtGetRunModeFunObj aclrtGetRunMode_;
extern aclrtGetStreamIdFromExceptionInfoFunObj aclrtGetStreamIdFromExceptionInfo_;
extern aclrtGetTaskIdFromExceptionInfoFunObj aclrtGetTaskIdFromExceptionInfo_;
extern aclrtGetThreadIdFromExceptionInfoFunObj aclrtGetThreadIdFromExceptionInfo_;
extern aclrtLaunchCallbackFunObj aclrtLaunchCallback_;
extern aclrtMallocFunObj aclrtMalloc_;
extern aclrtMallocHostFunObj aclrtMallocHost_;
extern aclrtMemcpyFunObj aclrtMemcpy_;
extern aclrtMemcpyAsyncFunObj aclrtMemcpyAsync_;
extern aclrtMemsetFunObj aclrtMemset_;
extern aclrtProcessReportFunObj aclrtProcessReport_;
extern aclrtQueryEventStatusFunObj aclrtQueryEventStatus_;
extern aclrtRecordEventFunObj aclrtRecordEvent_;
extern aclrtResetDeviceFunObj aclrtResetDevice_;
extern aclrtResetEventFunObj aclrtResetEvent_;
extern aclrtSetCurrentContextFunObj aclrtSetCurrentContext_;
extern aclrtSetDeviceFunObj aclrtSetDevice_;
extern aclrtSetDeviceSatModeFunObj aclrtSetDeviceSatMode_;
extern aclrtSetExceptionInfoCallbackFunObj aclrtSetExceptionInfoCallback_;
extern aclrtSetOpExecuteTimeOutFunObj aclrtSetOpExecuteTimeOut_;
extern aclrtSetOpWaitTimeoutFunObj aclrtSetOpWaitTimeout_;
extern aclrtSetStreamFailureModeFunObj aclrtSetStreamFailureMode_;
extern aclrtStreamQueryFunObj aclrtStreamQuery_;
extern aclrtStreamWaitEventFunObj aclrtStreamWaitEvent_;
extern aclrtSubscribeReportFunObj aclrtSubscribeReport_;
extern aclrtSynchronizeEventFunObj aclrtSynchronizeEvent_;
extern aclrtSynchronizeStreamFunObj aclrtSynchronizeStream_;
extern aclrtSynchronizeStreamWithTimeoutFunObj aclrtSynchronizeStreamWithTimeout_;

void LoadAclRtApiSymbol(const std::string &ascend_path);
}  // namespace transform
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_TRANSFORM_SYMBOL_ACL_RT_SYMBOL_H_
