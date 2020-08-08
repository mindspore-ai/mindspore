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

#include "acl_stub.h"
#include <iostream>

AclDataBuffer *g_acl_data_buffer = nullptr;
AclEnv *g_acl_env = nullptr;
AclDataSet *g_acl_dataset = nullptr;
AclModel *g_acl_model = nullptr;
AclModelDesc *g_acl_model_desc = nullptr;
AclDeviceContextStream *g_acl_device_context_stream = nullptr;
AclMemory *g_acl_memory = nullptr;
AclDvppPicDesc *g_acl_dvpp_pic_desc = nullptr;
AclDvppRoiConfig *g_acl_dvpp_roi_config = nullptr;
AclDvppResizeConfig *g_acl_dvpp_resize_config = nullptr;
AclDvppChannelDesc *g_acl_dvpp_channel_desc = nullptr;
AclDvppProcess *g_acl_dvpp_process = nullptr;
AclRunMode *g_acl_run_mode = nullptr;
AclJpegLib *g_acl_jpeg_lib = nullptr;

aclDataBuffer *aclCreateDataBuffer(void *data, size_t size) {
  return g_acl_data_buffer->aclCreateDataBuffer(data, size);
}

aclError aclDestroyDataBuffer(const aclDataBuffer *dataBuffer) {
  return g_acl_data_buffer->aclDestroyDataBuffer(dataBuffer);
}

void *aclGetDataBufferAddr(const aclDataBuffer *dataBuffer) {
  return g_acl_data_buffer->aclGetDataBufferAddr(dataBuffer);
}

uint32_t aclGetDataBufferSize(const aclDataBuffer *dataBuffer) {
  return g_acl_data_buffer->aclGetDataBufferSize(dataBuffer);
}

size_t aclDataTypeSize(aclDataType dataType) {
  std::unordered_map<aclDataType, size_t> dataTypeMap = {
    {ACL_FLOAT16, 2}, {ACL_FLOAT, 4}, {ACL_DOUBLE, 8}, {ACL_INT8, 1},   {ACL_INT16, 2},  {ACL_INT32, 4},
    {ACL_INT64, 8},   {ACL_UINT8, 1}, {ACL_UINT16, 2}, {ACL_UINT32, 4}, {ACL_UINT64, 8}, {ACL_BOOL, 1},
  };
  auto it = dataTypeMap.find(dataType);
  if (it == dataTypeMap.end()) {
    return 0;
  } else {
    return it->second;
  }
}

void aclAppLog(aclLogLevel logLevel, const char *func, const char *file, uint32_t line, const char *fmt, ...) {
  if (logLevel == ACL_ERROR) {
    // std::cout << file << ":" << line << "," << func << ": " << fmt << std::endl;
  }
}

aclError aclInit(const char *configPath) { return g_acl_env->aclInit(configPath); }

aclError aclFinalize() { return g_acl_env->aclFinalize(); }

// dataset
aclmdlDataset *aclmdlCreateDataset() { return g_acl_dataset->aclmdlCreateDataset(); }

aclError aclmdlDestroyDataset(const aclmdlDataset *dataSet) { return g_acl_dataset->aclmdlDestroyDataset(dataSet); }

aclError aclmdlAddDatasetBuffer(aclmdlDataset *dataSet, aclDataBuffer *dataBuffer) {
  return g_acl_dataset->aclmdlAddDatasetBuffer(dataSet, dataBuffer);
}

size_t aclmdlGetDatasetNumBuffers(const aclmdlDataset *dataSet) {
  return g_acl_dataset->aclmdlGetDatasetNumBuffers(dataSet);
}

aclDataBuffer *aclmdlGetDatasetBuffer(const aclmdlDataset *dataSet, size_t index) {
  return g_acl_dataset->aclmdlGetDatasetBuffer(dataSet, index);
}

// model
aclError aclmdlLoadFromFile(const char *modelPath, uint32_t *modelId) {
  return g_acl_model->aclmdlLoadFromFile(modelPath, modelId);
}

aclError aclmdlLoadFromMem(const void *model, size_t modelSize, uint32_t *modelId) {
  return g_acl_model->aclmdlLoadFromMem(model, modelSize, modelId);
}

aclError aclmdlLoadFromFileWithMem(const char *modelPath, uint32_t *modelId, void *workPtr, size_t workSize,
                                   void *weightPtr, size_t weightSize) {
  return g_acl_model->aclmdlLoadFromFileWithMem(modelPath, modelId, workPtr, workSize, weightPtr, weightSize);
}

aclError aclmdlLoadFromMemWithMem(const void *model, size_t modelSize, uint32_t *modelId, void *workPtr,
                                  size_t workSize, void *weightPtr, size_t weightSize) {
  return g_acl_model->aclmdlLoadFromMemWithMem(model, modelSize, modelId, workPtr, workSize, weightPtr, weightSize);
}

aclError aclmdlExecute(uint32_t modelId, const aclmdlDataset *input, aclmdlDataset *output) {
  return g_acl_model->aclmdlExecute(modelId, input, output);
}

aclError aclmdlExecuteAsync(uint32_t modelId, const aclmdlDataset *input, aclmdlDataset *output, aclrtStream stream) {
  return g_acl_model->aclmdlExecuteAsync(modelId, input, output, stream);
}

aclError aclmdlUnload(uint32_t modelId) { return g_acl_model->aclmdlUnload(modelId); }

// model desc
aclmdlDesc *aclmdlCreateDesc() { return g_acl_model_desc->aclmdlCreateDesc(); }

aclError aclmdlDestroyDesc(aclmdlDesc *modelDesc) { return g_acl_model_desc->aclmdlDestroyDesc(modelDesc); }

aclError aclmdlGetDesc(aclmdlDesc *modelDesc, uint32_t modelId) {
  return g_acl_model_desc->aclmdlGetDesc(modelDesc, modelId);
}

size_t aclmdlGetNumInputs(aclmdlDesc *modelDesc) { return g_acl_model_desc->aclmdlGetNumInputs(modelDesc); }

size_t aclmdlGetNumOutputs(aclmdlDesc *modelDesc) { return g_acl_model_desc->aclmdlGetNumOutputs(modelDesc); }

size_t aclmdlGetInputSizeByIndex(aclmdlDesc *modelDesc, size_t index) {
  return g_acl_model_desc->aclmdlGetInputSizeByIndex(modelDesc, index);
}

size_t aclmdlGetOutputSizeByIndex(aclmdlDesc *modelDesc, size_t index) {
  return g_acl_model_desc->aclmdlGetOutputSizeByIndex(modelDesc, index);
}

aclError aclmdlGetInputDims(const aclmdlDesc *modelDesc, size_t index, aclmdlIODims *dims) {
  return g_acl_model_desc->aclmdlGetInputDims(modelDesc, index, dims);
}

aclError aclmdlGetOutputDims(const aclmdlDesc *modelDesc, size_t index, aclmdlIODims *dims) {
  return g_acl_model_desc->aclmdlGetOutputDims(modelDesc, index, dims);
}

aclError aclmdlGetCurOutputDims(const aclmdlDesc *modelDesc, size_t index, aclmdlIODims *dims) {
  return g_acl_model_desc->aclmdlGetCurOutputDims(modelDesc, index, dims);
}

aclFormat aclmdlGetInputFormat(const aclmdlDesc *modelDesc, size_t index) {
  return g_acl_model_desc->aclmdlGetInputFormat(modelDesc, index);
}

aclFormat aclmdlGetOutputFormat(const aclmdlDesc *modelDesc, size_t index) {
  return g_acl_model_desc->aclmdlGetOutputFormat(modelDesc, index);
}

aclDataType aclmdlGetInputDataType(const aclmdlDesc *modelDesc, size_t index) {
  return g_acl_model_desc->aclmdlGetInputDataType(modelDesc, index);
}

aclDataType aclmdlGetOutputDataType(const aclmdlDesc *modelDesc, size_t index) {
  return g_acl_model_desc->aclmdlGetOutputDataType(modelDesc, index);
}

// device, context, stream

aclError aclrtCreateContext(aclrtContext *context, int32_t deviceId) {
  return g_acl_device_context_stream->aclrtCreateContext(context, deviceId);
}

aclError aclrtDestroyContext(aclrtContext context) { return g_acl_device_context_stream->aclrtDestroyContext(context); }

aclError aclrtSetCurrentContext(aclrtContext context) {
  return g_acl_device_context_stream->aclrtSetCurrentContext(context);
}

aclError aclrtSetDevice(int32_t deviceId) { return g_acl_device_context_stream->aclrtSetDevice(deviceId); }

aclError aclrtResetDevice(int32_t deviceId) { return g_acl_device_context_stream->aclrtResetDevice(deviceId); }

aclError aclrtGetRunMode(aclrtRunMode *runMode) { return g_acl_run_mode->aclrtGetRunMode(runMode); }

aclError aclrtCreateStream(aclrtStream *stream) { return g_acl_device_context_stream->aclrtCreateStream(stream); }

aclError aclrtDestroyStream(aclrtStream stream) { return g_acl_device_context_stream->aclrtDestroyStream(stream); }

aclError aclrtSynchronizeStream(aclrtStream stream) {
  return g_acl_device_context_stream->aclrtSynchronizeStream(stream);
}

// memory
aclError acldvppMalloc(void **devPtr, size_t size) { return g_acl_memory->acldvppMalloc(devPtr, size); }
aclError acldvppFree(void *devPtr) { return g_acl_memory->acldvppFree(devPtr); }

aclError aclrtMalloc(void **devPtr, size_t size, aclrtMemMallocPolicy policy) {
  return g_acl_memory->aclrtMalloc(devPtr, size, policy);
}

aclError aclrtFree(void *devPtr) { return g_acl_memory->aclrtFree(devPtr); }

aclError aclrtMallocHost(void **hostPtr, size_t size) { return g_acl_memory->aclrtMallocHost(hostPtr, size); }

aclError aclrtFreeHost(void *hostPtr) { return g_acl_memory->aclrtFreeHost(hostPtr); }

aclError aclrtMemcpy(void *dst, size_t destMax, const void *src, size_t count, aclrtMemcpyKind kind) {
  return g_acl_memory->aclrtMemcpy(dst, destMax, src, count, kind);
}

acldvppPicDesc *acldvppCreatePicDesc() { return g_acl_dvpp_pic_desc->acldvppCreatePicDesc(); }
aclError acldvppDestroyPicDesc(acldvppPicDesc *picDesc) { return g_acl_dvpp_pic_desc->acldvppDestroyPicDesc(picDesc); }

aclError acldvppSetPicDescSize(acldvppPicDesc *picDesc, uint32_t size) {
  return g_acl_dvpp_pic_desc->acldvppSetPicDescSize(picDesc, size);
}

aclError acldvppSetPicDescFormat(acldvppPicDesc *picDesc, acldvppPixelFormat format) {
  return g_acl_dvpp_pic_desc->acldvppSetPicDescFormat(picDesc, format);
}

aclError acldvppSetPicDescWidth(acldvppPicDesc *picDesc, uint32_t width) {
  return g_acl_dvpp_pic_desc->acldvppSetPicDescWidth(picDesc, width);
}

aclError acldvppSetPicDescHeight(acldvppPicDesc *picDesc, uint32_t height) {
  return g_acl_dvpp_pic_desc->acldvppSetPicDescHeight(picDesc, height);
}

aclError acldvppSetPicDescData(acldvppPicDesc *picDesc, void *dataDev) {
  return g_acl_dvpp_pic_desc->acldvppSetPicDescData(picDesc, dataDev);
}

aclError acldvppSetPicDescWidthStride(acldvppPicDesc *picDesc, uint32_t widthStride) {
  return g_acl_dvpp_pic_desc->acldvppSetPicDescWidthStride(picDesc, widthStride);
}

aclError acldvppSetPicDescHeightStride(acldvppPicDesc *picDesc, uint32_t heightStride) {
  return g_acl_dvpp_pic_desc->acldvppSetPicDescHeightStride(picDesc, heightStride);
}

acldvppRoiConfig *acldvppCreateRoiConfig(uint32_t left, uint32_t right, uint32_t top, uint32_t bottom) {
  return g_acl_dvpp_roi_config->acldvppCreateRoiConfig(left, right, top, bottom);
}

aclError acldvppDestroyRoiConfig(acldvppRoiConfig *roiConfig) {
  return g_acl_dvpp_roi_config->acldvppDestroyRoiConfig(roiConfig);
}

aclError acldvppSetRoiConfig(acldvppRoiConfig *roiConfig, uint32_t left, uint32_t right, uint32_t top,
                             uint32_t bottom) {
  return g_acl_dvpp_roi_config->acldvppSetRoiConfig(roiConfig, left, right, top, bottom);
}

acldvppResizeConfig *acldvppCreateResizeConfig() { return g_acl_dvpp_resize_config->acldvppCreateResizeConfig(); }

aclError acldvppDestroyResizeConfig(acldvppResizeConfig *resizeConfig) {
  return g_acl_dvpp_resize_config->acldvppDestroyResizeConfig(resizeConfig);
}

aclError acldvppCreateChannel(acldvppChannelDesc *channelDesc) {
  return g_acl_dvpp_channel_desc->acldvppCreateChannel(channelDesc);
}

aclError acldvppDestroyChannel(acldvppChannelDesc *channelDesc) {
  return g_acl_dvpp_channel_desc->acldvppDestroyChannel(channelDesc);
}

acldvppChannelDesc *acldvppCreateChannelDesc() { return g_acl_dvpp_channel_desc->acldvppCreateChannelDesc(); }

aclError acldvppDestroyChannelDesc(acldvppChannelDesc *channelDesc) {
  return g_acl_dvpp_channel_desc->acldvppDestroyChannelDesc(channelDesc);
}

aclError acldvppVpcResizeAsync(acldvppChannelDesc *channelDesc, acldvppPicDesc *inputDesc, acldvppPicDesc *outputDesc,
                               acldvppResizeConfig *resizeConfig, aclrtStream stream) {
  return g_acl_dvpp_process->acldvppVpcResizeAsync(channelDesc, inputDesc, outputDesc, resizeConfig, stream);
}

aclError acldvppVpcCropAsync(acldvppChannelDesc *channelDesc, acldvppPicDesc *inputDesc, acldvppPicDesc *outputDesc,
                             acldvppRoiConfig *cropArea, aclrtStream stream) {
  return g_acl_dvpp_process->acldvppVpcCropAsync(channelDesc, inputDesc, outputDesc, cropArea, stream);
}

aclError acldvppVpcCropAndPasteAsync(acldvppChannelDesc *channelDesc, acldvppPicDesc *inputDesc,
                                     acldvppPicDesc *outputDesc, acldvppRoiConfig *cropArea,
                                     acldvppRoiConfig *pasteArea, aclrtStream stream) {
  return g_acl_dvpp_process->acldvppVpcCropAndPasteAsync(channelDesc, inputDesc, outputDesc, cropArea, pasteArea,
                                                         stream);
}

aclError acldvppVpcBatchCropAsync(acldvppChannelDesc *channelDesc, acldvppBatchPicDesc *srcBatchDesc, uint32_t *roiNums,
                                  uint32_t size, acldvppBatchPicDesc *dstBatchDesc, acldvppRoiConfig *cropAreas[],
                                  aclrtStream stream) {
  return g_acl_dvpp_process->acldvppVpcBatchCropAsync(channelDesc, srcBatchDesc, roiNums, size, dstBatchDesc, cropAreas,
                                                      stream);
}

aclError acldvppJpegDecodeAsync(acldvppChannelDesc *channelDesc, const void *data, uint32_t size,
                                acldvppPicDesc *outputDesc, aclrtStream stream) {
  return g_acl_dvpp_process->acldvppJpegDecodeAsync(channelDesc, data, size, outputDesc, stream);
}

// jpeg lib
void jpeg_CreateDecompress(j_decompress_ptr cinfo, int version, size_t structsize) {
  g_acl_jpeg_lib->jpeg_CreateDecompress(cinfo, version, structsize);
}

void jpeg_mem_src(j_decompress_ptr cinfo, const unsigned char *inbuffer, unsigned long insize) {
  g_acl_jpeg_lib->jpeg_mem_src(cinfo, inbuffer, insize);
}

int jpeg_read_header(j_decompress_ptr cinfo, boolean require_image) {
  return g_acl_jpeg_lib->jpeg_read_header(cinfo, require_image);
}

void jpeg_destroy_decompress(j_decompress_ptr cinfo) { g_acl_jpeg_lib->jpeg_destroy_decompress(cinfo); }

struct jpeg_error_mgr *jpeg_std_error(struct jpeg_error_mgr *err) {
  return err;
}