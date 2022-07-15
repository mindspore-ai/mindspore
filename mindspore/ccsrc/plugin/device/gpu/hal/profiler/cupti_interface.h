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
#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_HAL_PROFILER_CUPTI_INTERFACE_H
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_HAL_PROFILER_CUPTI_INTERFACE_H
#ifndef FUNC_EXPORT
#define FUNC_EXPORT __attribute__((visibility("default")))
#endif
namespace mindspore {
namespace profiler {
namespace gpu {
CUptiResult CuptiSubscribe(CUpti_SubscriberHandle *subscriber, CUpti_CallbackFunc callback, void *userdata);
CUptiResult CuptiEnableDomain(uint32_t enable, CUpti_SubscriberHandle subscriber, CUpti_CallbackDomain domain);
CUptiResult CuptiGetStreamId(CUcontext context, CUstream stream, uint32_t *streamId);
CUptiResult CuptiGetDeviceId(CUcontext context, uint32_t *deviceId);

CUptiResult CuptiActivityEnable(CUpti_ActivityKind kind);
CUptiResult CuptiActivityRegisterCallbacks(CUpti_BuffersCallbackRequestFunc funcBufferRequested,
                                           CUpti_BuffersCallbackCompleteFunc funcBufferCompleted);
CUptiResult CuptiUnsubscribe(CUpti_SubscriberHandle subscriber);
CUptiResult CuptiActivityFlushAll(uint32_t flag);
CUptiResult CuptiActivityDisable(CUpti_ActivityKind kind);
CUptiResult CuptiActivityGetNextRecord(uint8_t *buffer, size_t validBufferSizeBytes, CUpti_Activity **record);
CUptiResult CuptiActivityGetNumDroppedRecords(CUcontext context, uint32_t streamId, size_t *dropped);
CUptiResult CuptiGetTimestamp(uint64_t *timestamp);
CUptiResult CuptiGetResultString(CUptiResult result, const char **str);
CUptiResult CuptiFinalize();
}  // namespace gpu
}  // namespace profiler
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_HAL_PROFILER_CUPTI_INTERFACE_H
