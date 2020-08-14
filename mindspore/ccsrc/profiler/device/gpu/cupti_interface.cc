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
#include <cupti.h>
#include <dlfcn.h>
#include "utils/log_adapter.h"
#include "profiler/device/gpu/cupti_interface.h"

namespace mindspore {
namespace profiler {
namespace gpu {

inline void *LoadLibrary(const char *name) {
  auto handle = dlopen(name, RTLD_LAZY | RTLD_LOCAL);
  if (handle == nullptr) {
    MS_LOG(EXCEPTION) << "Load lib " << name << " Please check whether configured the path of CUPTI to LD_LIBRARY_PATH";
  }
  return handle;
}

inline void *GetCUPTIHandle() {
  static void *handle = LoadLibrary("libcupti.so");
  return handle;
}

inline void *GetCUPTIFunc(const char *name) {
  static void *handle = GetCUPTIHandle();
  void *func = dlsym(handle, name);
  if (func == nullptr) {
    MS_LOG(EXCEPTION) << "Load func " << name << " failed, make sure you have implied it!";
  }
  return func;
}

typedef CUptiResult (*CuptiSubscribeFunc)(CUpti_SubscriberHandle *subscriber, CUpti_CallbackFunc callback,
                                          void *userdata);
typedef CUptiResult (*CuptiEnableDomainFunc)(uint32_t enable, CUpti_SubscriberHandle subscriber,
                                             CUpti_CallbackDomain domain);
typedef CUptiResult (*CuptiActivityEnableFunc)(CUpti_ActivityKind kind);
typedef CUptiResult (*CuptiActivityRegisterCallbacksFunc)(CUpti_BuffersCallbackRequestFunc funcBufferRequested,
                                                          CUpti_BuffersCallbackCompleteFunc funcBufferCompleted);
typedef CUptiResult (*CuptiUnsubscribeFunc)(CUpti_SubscriberHandle subscriber);
typedef CUptiResult (*CuptiActivityFlushAllFunc)(uint32_t flag);
typedef CUptiResult (*CuptiActivityDisableFunc)(CUpti_ActivityKind kind);
typedef CUptiResult (*CuptiActivityGetNextRecordFunc)(uint8_t *buffer, size_t validBufferSizeBytes,
                                                      CUpti_Activity **record);
typedef CUptiResult (*CuptiActivityGetNumDroppedRecordsFunc)(CUcontext context, uint32_t streamId, size_t *dropped);
typedef CUptiResult (*CuptiGetTimestampFunc)(uint64_t *timestamp);
typedef CUptiResult (*CuptiGetResultStringFunc)(CUptiResult result, const char **str);
typedef CUptiResult (*CuptiGetStreamIdFunc)(CUcontext context, CUstream stream, uint32_t *streamId);
typedef CUptiResult (*CuptiGetDeviceIdFunc)(CUcontext context, uint32_t *deviceId);

CUptiResult CuptiSubscribe(CUpti_SubscriberHandle *subscriber, CUpti_CallbackFunc callback, void *userdata) {
  static auto func_ptr = reinterpret_cast<CuptiSubscribeFunc>(GetCUPTIFunc("cuptiSubscribe"));
  return func_ptr(subscriber, callback, userdata);
}

CUptiResult CuptiEnableDomain(uint32_t enable, CUpti_SubscriberHandle subscriber, CUpti_CallbackDomain domain) {
  static auto func_ptr = reinterpret_cast<CuptiEnableDomainFunc>(GetCUPTIFunc("cuptiEnableDomain"));
  return func_ptr(enable, subscriber, domain);
}

CUptiResult CuptiActivityEnable(CUpti_ActivityKind kind) {
  static auto func_ptr = reinterpret_cast<CuptiActivityEnableFunc>(GetCUPTIFunc("cuptiActivityEnable"));
  return func_ptr(kind);
}

CUptiResult CuptiActivityRegisterCallbacks(CUpti_BuffersCallbackRequestFunc funcBufferRequested,
                                           CUpti_BuffersCallbackCompleteFunc funcBufferCompleted) {
  static auto func_ptr =
    reinterpret_cast<CuptiActivityRegisterCallbacksFunc>(GetCUPTIFunc("cuptiActivityRegisterCallbacks"));
  return func_ptr(funcBufferRequested, funcBufferCompleted);
}

CUptiResult CuptiUnsubscribe(CUpti_SubscriberHandle subscriber) {
  static auto func_ptr = reinterpret_cast<CuptiUnsubscribeFunc>(GetCUPTIFunc("cuptiUnsubscribe"));
  return func_ptr(subscriber);
}

CUptiResult CuptiActivityFlushAll(uint32_t flag) {
  static auto func_ptr = reinterpret_cast<CuptiActivityFlushAllFunc>(GetCUPTIFunc("cuptiActivityFlushAll"));
  return func_ptr(flag);
}

CUptiResult CuptiActivityDisable(CUpti_ActivityKind kind) {
  static auto func_ptr = reinterpret_cast<CuptiActivityDisableFunc>(GetCUPTIFunc("cuptiActivityDisable"));
  return func_ptr(kind);
}

CUptiResult CuptiActivityGetNextRecord(uint8_t *buffer, size_t validBufferSizeBytes, CUpti_Activity **record) {
  static auto func_ptr = reinterpret_cast<CuptiActivityGetNextRecordFunc>(GetCUPTIFunc("cuptiActivityGetNextRecord"));
  return func_ptr(buffer, validBufferSizeBytes, record);
}

CUptiResult CuptiActivityGetNumDroppedRecords(CUcontext context, uint32_t streamId, size_t *dropped) {
  static auto func_ptr =
    reinterpret_cast<CuptiActivityGetNumDroppedRecordsFunc>(GetCUPTIFunc("cuptiActivityGetNumDroppedRecords"));
  return func_ptr(context, streamId, dropped);
}

CUptiResult CuptiGetTimestamp(uint64_t *timestamp) {
  static auto func_ptr = reinterpret_cast<CuptiGetTimestampFunc>(GetCUPTIFunc("cuptiGetTimestamp"));
  return func_ptr(timestamp);
}

CUptiResult CuptiGetResultString(CUptiResult result, const char **str) {
  static auto func_ptr = reinterpret_cast<CuptiGetResultStringFunc>(GetCUPTIFunc("cuptiGetResultString"));
  return func_ptr(result, str);
}

CUptiResult CuptiGetStreamId(CUcontext context, CUstream stream, uint32_t *streamId) {
  static auto func_ptr = reinterpret_cast<CuptiGetStreamIdFunc>(GetCUPTIFunc("cuptiGetStreamId"));
  return func_ptr(context, stream, streamId);
}

CUptiResult CuptiGetDeviceId(CUcontext context, uint32_t *deviceId) {
  static auto func_ptr = reinterpret_cast<CuptiGetDeviceIdFunc>(GetCUPTIFunc("cuptiSubscribe"));
  return func_ptr(context, deviceId);
}
}  // namespace gpu
}  // namespace profiler
}  // namespace mindspore
