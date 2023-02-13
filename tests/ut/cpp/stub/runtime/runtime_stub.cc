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
#include "runtime/base.h"
#include "runtime/context.h"
#include "runtime/dev.h"
#include "runtime/event.h"
#include "runtime/kernel.h"
#include "runtime/mem.h"
#include "runtime/rt_model.h"
#include "runtime/stream.h"
#include "toolchain/adx_datadump_server.h"
#include "toolchain/plog.h"

rtError_t rtEventSynchronize(rtEvent_t event) { return RT_ERROR_NONE; }

rtError_t rtEventCreateWithFlag(rtEvent_t *event, uint32_t flag) { return RT_ERROR_NONE; }

rtError_t rtEventElapsedTime(float *time, rtEvent_t start, rtEvent_t end) { return RT_ERROR_NONE; }

rtError_t rtMalloc(void **devPtr, uint64_t size, rtMemType_t type, const uint16_t moduleId) { return RT_ERROR_NONE; }

rtError_t rtMemcpy(void *dst, uint64_t destMax, const void *src, uint64_t count, rtMemcpyKind_t kind) {
  return RT_ERROR_NONE;
}

rtError_t rtMemset(void *devPtr, uint64_t destMax, uint32_t value, uint64_t count) { return RT_ERROR_NONE; }

rtError_t rtGetDeviceCount(int32_t *count) { return RT_ERROR_NONE; }

rtError_t rtSetDevice(int32_t device) { return RT_ERROR_NONE; }

rtError_t rtDeviceReset(int32_t device) { return RT_ERROR_NONE; }

rtError_t rtCtxCreate(rtContext_t *ctx, uint32_t flags, int32_t device) { return RT_ERROR_NONE; }

rtError_t rtCtxSetCurrent(rtContext_t ctx) { return RT_ERROR_NONE; }

rtError_t rtCtxDestroy(rtContext_t ctx) { return RT_ERROR_NONE; }

rtError_t rtStreamCreate(rtStream_t *stream, int32_t priority) { return RT_ERROR_NONE; }

rtError_t rtStreamDestroy(rtStream_t stream) { return RT_ERROR_NONE; }

rtError_t rtDevBinaryRegister(const rtDevBinary_t *bin, void **handle) { return RT_ERROR_NONE; }

rtError_t rtFunctionRegister(void *binHandle, const void *stubFunc, const char *stubName, const void *devFunc,
                             uint32_t funcMode) {
  return RT_ERROR_NONE;
}

RTS_API rtError_t rtSetSocVersion(const char *version) { return RT_ERROR_NONE; }

rtError_t rtGetSocVersion(char *version, const uint32_t maxLen) { return RT_ERROR_NONE; }

rtError_t rtKernelLaunch(const void *stubFunc, uint32_t blockDim, void *args, uint32_t argsSize, rtSmDesc_t *smDesc,
                         rtStream_t stream) {
  return RT_ERROR_NONE;
}

rtError_t rtStreamSynchronize(rtStream_t stream) { return RT_ERROR_NONE; }

rtError_t rtKernelLaunchEx(void *args, uint32_t argsSize, uint32_t flags, rtStream_t stream_) { return RT_ERROR_NONE; }

rtError_t rtFree(void *devPtr) {
  delete[] reinterpret_cast<uint8_t *>(devPtr);
  return RT_ERROR_NONE;
}

rtError_t rtEndGraphEx(rtModel_t model, rtStream_t stream, uint32_t flags) { return RT_ERROR_NONE; }

rtError_t rtModelExecute(rtModel_t model, rtStream_t stream, uint32_t flag) { return RT_ERROR_NONE; }

rtError_t rtMemAllocManaged(void **ptr, uint64_t size, uint32_t flag) { return RT_ERROR_NONE; }

rtError_t rtMemcpyAsync(void *dst, uint64_t destMax, const void *src, uint64_t count, rtMemcpyKind_t kind,
                        rtStream_t stream) {
  return RT_ERROR_NONE;
}

rtError_t rtLabelSwitch(void *ptr, rtCondition_t condition, uint32_t value, rtLabel_t trueLabel, rtStream_t stream) {
  return RT_ERROR_NONE;
}

rtError_t rtStreamSwitch(void *ptr, rtCondition_t condition, int64_t value, rtStream_t true_stream, rtStream_t stream) {
  return RT_ERROR_NONE;
}
rtError_t rtStreamSwitchEx(void *ptr, rtCondition_t condition, void *value_ptr, rtStream_t true_stream,
                           rtStream_t stream, rtSwitchDataType_t dataType) {
  return RT_ERROR_NONE;
}

rtError_t rtKernelFusionEnd(rtStream_t stream) { return RT_ERROR_NONE; }

rtError_t rtKernelFusionStart(rtStream_t stream) { return RT_ERROR_NONE; }

rtError_t rtStreamWaitEvent(rtStream_t stream, rtEvent_t event) { return RT_ERROR_NONE; }

rtError_t rtEventReset(rtEvent_t event, rtStream_t stream) { return RT_ERROR_NONE; }

rtError_t rtEventRecord(rtEvent_t event, rtStream_t stream) { return RT_ERROR_NONE; }

rtError_t rtLabelGoto(rtLabel_t label, rtStream_t stream) { return RT_ERROR_NONE; }

rtError_t rtLabelSet(rtLabel_t label, rtStream_t stream) { return RT_ERROR_NONE; }

rtError_t rtStreamActive(rtStream_t active_stream, rtStream_t stream) { return RT_ERROR_NONE; }

rtError_t rtModelUnbindStream(rtModel_t model, rtStream_t stream) { return RT_ERROR_NONE; }

rtError_t rtLabelDestroy(rtLabel_t label) { return RT_ERROR_NONE; }

rtError_t rtModelDestroy(rtModel_t model) { return RT_ERROR_NONE; }

rtError_t rtEventDestroy(rtEvent_t event) { return RT_ERROR_NONE; }

rtError_t rtMemFreeManaged(void *ptr) { return RT_ERROR_NONE; }

rtError_t rtModelCreate(rtModel_t *model, uint32_t flag) { return RT_ERROR_NONE; }

rtError_t rtModelBindStream(rtModel_t model, rtStream_t stream, uint32_t flag) { return RT_ERROR_NONE; }

rtError_t rtStreamCreateWithFlags(rtStream_t *stream, int32_t priority, uint32_t flags) { return RT_ERROR_NONE; }

rtError_t rtEventCreate(rtEvent_t *event) { return RT_ERROR_NONE; }

rtError_t rtLabelCreate(rtLabel_t *label) { return RT_ERROR_NONE; }

rtError_t rtModelLoadComplete(rtModel_t model) { return RT_ERROR_NONE; }

rtError_t rtCtxGetCurrent(rtContext_t *ctx) { return RT_ERROR_NONE; }

rtError_t rtGetStreamId(rtStream_t stream, int32_t *streamId) { return RT_ERROR_NONE; }

rtError_t rtGetFunctionByName(const char *stubName, void **stubFunc) { return RT_ERROR_NONE; }

rtError_t rtSetTaskGenCallback(rtTaskGenCallback callback) { return RT_ERROR_NONE; }

RTS_API rtError_t rtProfilerStart(uint64_t profConfig, int32_t numsDev, uint32_t *deviceList) { return RT_ERROR_NONE; }

RTS_API rtError_t rtProfilerStop(uint64_t profConfig, int32_t numsDev, uint32_t *deviceList) { return RT_ERROR_NONE; }

int AdxDataDumpServerInit() { return 0; }

int AdxDataDumpServerUnInit() { return 0; }

RTS_API rtError_t rtGetTaskIdAndStreamID(uint32_t *taskid, uint32_t *streamid) { return RT_ERROR_NONE; }

RTS_API rtError_t rtSetTaskFailCallback(rtTaskFailCallback callback) { return RT_ERROR_NONE; }

RTS_API rtError_t rtRegDeviceStateCallback(const char *regName, rtDeviceStateCallback callback) {
  return RT_ERROR_NONE;
}

RTS_API rtError_t rtSetMsprofReporterCallback(MsprofReporterCallback callback) { return RT_ERROR_NONE; }

RTS_API rtError_t rtRegTaskFailCallbackByModule(const char *moduleName, rtTaskFailCallback callback) {
  return RT_ERROR_NONE;
}

RTS_API rtError_t rtRegisterAllKernel(const rtDevBinary_t *bin, void **module) { return RT_ERROR_NONE; }

RTS_API rtError_t rtDevBinaryUnRegister(void *handle) { return RT_ERROR_NONE; }

RTS_API rtError_t rtMemsetAsync(void *ptr, uint64_t destMax, uint32_t value, uint64_t count, rtStream_t stream) {
  return RT_ERROR_NONE;
}

RTS_API rtError_t rtLabelListCpy(rtLabel_t *label, uint32_t labelNumber, void *dst, uint32_t dstMax) {
  return RT_ERROR_NONE;
}

RTS_API rtError_t rtModelGetTaskId(rtModel_t model, uint32_t *taskid, uint32_t *streamid) { return RT_ERROR_NONE; }

RTS_API rtError_t rtModelGetId(rtModel_t model, uint32_t *modelId) { return RT_ERROR_NONE; }

RTS_API rtError_t rtLabelCreateEx(rtLabel_t *label, rtStream_t stream) { return RT_ERROR_NONE; }

RTS_API rtError_t rtLabelCreateExV2(rtLabel_t *lbl, rtModel_t mdl, rtStream_t stm) { return RT_ERROR_NONE; }

RTS_API rtError_t rtCpuKernelLaunchWithFlag(const void *soName, const void *kernelName, uint32_t blockDim,
                                            const rtArgsEx_t *argsInfo, rtSmDesc_t *smDesc, rtStream_t stm,
                                            uint32_t flags) {
  return RT_ERROR_NONE;
}

RTS_API rtError_t rtLabelSwitchByIndex(void *ptr, uint32_t max, void *labelInfoPtr, rtStream_t stream) {
  return RT_ERROR_NONE;
}

RTS_API rtError_t rtProfilerTrace(uint64_t id, bool notify, uint32_t flags, rtStream_t stream) { return RT_ERROR_NONE; }

RTS_API rtError_t rtProfilerTraceEx(uint64_t id, uint64_t modelId, uint16_t tagId, rtStream_t stream) {
  return RT_ERROR_NONE;
}

RTS_API rtError_t rtKernelLaunchWithFlag(const void *stubFunc, uint32_t blockDim, rtArgsEx_t *argsInfo,
                                         rtSmDesc_t *smDesc, rtStream_t stm, uint32_t flags) {
  return RT_ERROR_NONE;
}

RTS_API rtError_t rtMemGetInfoEx(rtMemInfoType_t memInfoType, size_t *free, size_t *total) { return RT_ERROR_NONE; }

RTS_API rtError_t rtProfRegisterCtrlCallback(uint32_t moduleId, rtProfCtrlHandle callback) { return RT_ERROR_NONE; }

RTS_API rtError_t rtGetRtCapability(rtFeatureType_t, int32_t, int64_t *) { return RT_ERROR_NONE; }

RTS_API rtError_t rtKernelLaunchWithHandle(void *hdl, const uint64_t tilingKey, uint32_t blockDim, rtArgsEx_t *argsInfo,
                                           rtSmDesc_t *smDesc, rtStream_t stm, const void *kernelInfo) {
  return RT_ERROR_NONE;
}

RTS_API rtError_t rtGetEventID(rtEvent_t event, uint32_t *event_id) {
  *event_id = 0;
  return RT_ERROR_NONE;
}

RTS_API rtError_t rtGetDeviceInfo(uint32_t deviceId, int32_t moduleType, int32_t infoType, int64_t *val) {
  return RT_ERROR_NONE;
}

int DlogReportInitialize(void) { return 0; }

int DlogReportFinalize(void) { return 0; }
