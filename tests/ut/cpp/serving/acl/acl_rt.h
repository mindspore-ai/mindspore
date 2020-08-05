/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef ACL_STUB_INC_ACL_RT_H
#define ACL_STUB_INC_ACL_RT_H
#include "acl_base.h"

typedef enum aclrtRunMode {
  ACL_DEVICE,
  ACL_HOST,
} aclrtRunMode;

typedef enum aclrtTsId {
  ACL_TS_ID_AICORE,
  ACL_TS_ID_AIVECTOR,
  ACL_TS_ID_RESERVED,
} aclrtTsId;

typedef enum aclrtEventStatus {
  ACL_EVENT_STATUS_COMPLETE,
  ACL_EVENT_STATUS_NOT_READY,
  ACL_EVENT_STATUS_RESERVED,
} aclrtEventStatus;

typedef enum aclrtCallbackBlockType {
  ACL_CALLBACK_NO_BLOCK,
  ACL_CALLBACK_BLOCK,
} aclrtCallbackBlockType;

typedef enum aclrtMemcpyKind {
  ACL_MEMCPY_HOST_TO_HOST,
  ACL_MEMCPY_HOST_TO_DEVICE,
  ACL_MEMCPY_DEVICE_TO_HOST,
  ACL_MEMCPY_DEVICE_TO_DEVICE,
} aclrtMemcpyKind;

typedef enum aclrtMemMallocPolicy {
  ACL_MEM_MALLOC_HUGE_FIRST,
  ACL_MEM_MALLOC_HUGE_ONLY,
  ACL_MEM_MALLOC_NORMAL_ONLY,
} aclrtMemMallocPolicy;

typedef struct rtExceptionInfo aclrtExceptionInfo;
typedef void (*aclrtCallback)(void *userData);
typedef void (*aclrtExceptionInfoCallback)(aclrtExceptionInfo *exceptionInfo);

aclError aclrtCreateContext(aclrtContext *context, int32_t deviceId);
aclError aclrtDestroyContext(aclrtContext context);
aclError aclrtSetCurrentContext(aclrtContext context);
aclError aclrtGetCurrentContext(aclrtContext *context);
aclError aclrtSetDevice(int32_t deviceId);
aclError aclrtResetDevice(int32_t deviceId);
aclError aclrtGetDevice(int32_t *deviceId);
aclError aclrtGetRunMode(aclrtRunMode *runMode);
aclError aclrtSynchronizeDevice(void);
aclError aclrtSetTsDevice(aclrtTsId tsId);
aclError aclrtGetDeviceCount(uint32_t *count);

aclError aclrtMalloc(void **devPtr, size_t size, aclrtMemMallocPolicy policy);
aclError aclrtFree(void *devPtr);

aclError aclrtMallocHost(void **hostPtr, size_t size);
aclError aclrtFreeHost(void *hostPtr);

aclError aclrtMemcpy(void *dst, size_t destMax, const void *src, size_t count, aclrtMemcpyKind kind);
aclError aclrtMemset(void *devPtr, size_t maxCount, int32_t value, size_t count);
aclError aclrtMemcpyAsync(void *dst, size_t destMax, const void *src, size_t count, aclrtMemcpyKind kind,
                          aclrtStream stream);
aclError aclrtMemsetAsync(void *devPtr, size_t maxCount, int32_t value, size_t count, aclrtStream stream);

aclError aclrtCreateStream(aclrtStream *stream);
aclError aclrtDestroyStream(aclrtStream stream);
aclError aclrtSynchronizeStream(aclrtStream stream);
aclError aclrtStreamWaitEvent(aclrtStream stream, aclrtEvent event);

#endif