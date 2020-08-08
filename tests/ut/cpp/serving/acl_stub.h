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

#ifndef MINDSPORE_ACL_STUB_H
#define MINDSPORE_ACL_STUB_H

#include "acl/acl_base.h"
#include "acl/acl.h"
#include "acl/acl_mdl.h"
#include "acl/acl_rt.h"
#include "acl/ops/acl_dvpp.h"
#include <algorithm>
#include <vector>
#include <string>
#include <unordered_map>
#include <map>
#include <functional>
#include <cstring>
#include "jpeglib.h"

struct aclDataBuffer {
  void *data = nullptr;
  size_t size = 0;
};

struct aclmdlDataset {
  std::vector<aclDataBuffer *> data_buffers;
};

struct aclTensorDesc {};

struct AclTensorDesc {
  std::vector<int64_t> dims;
  aclDataType data_type = ACL_DT_UNDEFINED;
  size_t size = 0;
};

struct aclmdlDesc {
  std::vector<AclTensorDesc> inputs;
  std::vector<AclTensorDesc> outputs;
};

struct acldvppPicDesc {
  uint32_t size = 0;
  acldvppPixelFormat format = PIXEL_FORMAT_YUV_400;
  uint32_t width = 0;
  uint32_t height = 0;
  void *dataDev = nullptr;
  uint32_t widthStride = 0;
  uint32_t heightStride = 0;
};

struct acldvppRoiConfig {
  uint32_t left = 0;
  uint32_t right = 0;
  uint32_t top = 0;
  uint32_t bottom = 0;
};

struct acldvppResizeConfig {
  uint32_t id;
};

struct acldvppChannelDesc {
  bool channel_valid_flag = false;
};

class AclModel;
extern AclModel *g_acl_model;

template <class Type>
aclError AclItemOnDestroy(
  std::vector<Type> &live, std::vector<Type> &destroy, const Type *destroy_item,
  std::function<void(Type &list_item)> func_release = [](Type &list_item) {}) {
  for (auto it = live.begin(); it != live.end(); it++) {
    if (&(*it) == destroy_item) {
      func_release(*it);
      destroy.push_back(*it);
      live.erase(it);
      return ACL_ERROR_NONE;
    }
  }
  return 1;
}

template <class PtType, typename std::enable_if<std::is_pointer<PtType>::value, int>::type = 0>
class ResourceBase {
 public:
  using Type = typename std::remove_pointer<PtType>::type;
  ResourceBase() = default;
  virtual ~ResourceBase() { Clear(); }
  void Clear() {
    for (auto item : resource_live_) {
      delete item;
    }
    resource_live_.clear();
    resource_destroy_.clear();
  }
  template <class... Args>
  Type *OnCreate(Args &&... args) {
    auto item = new Type(std::forward<Args>(args)...);
    resource_live_.push_back(item);
    return item;
  }
  aclError OnDestroy(
    const Type *item, std::function<void(Type &list_item)> func_release = [](Type &list_item) {}) {
    auto it = std::find(resource_live_.begin(), resource_live_.end(), item);
    if (it == resource_live_.end()) {
      return 1;
    }
    func_release(**it);                // Type&
    resource_destroy_.push_back(*it);  // Type*
    resource_live_.erase(it);
    delete item;
    return ACL_ERROR_NONE;
  }
  size_t LiveSize() const { return resource_live_.size(); }
  bool Check() const { return resource_live_.empty(); }
  std::vector<Type *> resource_live_;
  std::vector<Type *> resource_destroy_;
};

class AclDataBuffer {
 public:
  AclDataBuffer() {}
  virtual ~AclDataBuffer() { Clear(); }
  virtual void Clear() { data_buffer_.Clear(); }
  bool Check() { return data_buffer_.Check(); }

  virtual aclDataBuffer *aclCreateDataBuffer(void *data, size_t size) {
    aclDataBuffer data_buffer;
    data_buffer.data = data;
    data_buffer.size = size;
    return data_buffer_.OnCreate(data_buffer);
  }

  virtual aclError aclDestroyDataBuffer(const aclDataBuffer *dataBuffer) { return data_buffer_.OnDestroy(dataBuffer); }

  virtual void *aclGetDataBufferAddr(const aclDataBuffer *dataBuffer) {
    if (dataBuffer == nullptr) {
      return nullptr;
    }
    return dataBuffer->data;
  }

  virtual uint32_t aclGetDataBufferSize(const aclDataBuffer *dataBuffer) {
    if (dataBuffer == nullptr) {
      return 0;
    }
    return dataBuffer->size;
  }
  ResourceBase<aclDataBuffer *> data_buffer_;
};

class AclDataSet {
 public:
  AclDataSet() {}
  virtual ~AclDataSet() { Clear(); }
  virtual void Clear() { dataset_.Clear(); }
  bool Check() { return dataset_.Check(); }

 public:
  virtual aclmdlDataset *aclmdlCreateDataset() { return dataset_.OnCreate(); }
  virtual aclError aclmdlDestroyDataset(const aclmdlDataset *dataSet) { return dataset_.OnDestroy(dataSet); }
  virtual aclError aclmdlAddDatasetBuffer(aclmdlDataset *dataSet, aclDataBuffer *dataBuffer) {
    if (dataSet == nullptr) {
      return 1;
    }
    dataSet->data_buffers.push_back(dataBuffer);
    return ACL_ERROR_NONE;
  }
  virtual size_t aclmdlGetDatasetNumBuffers(const aclmdlDataset *dataSet) {
    if (dataSet == nullptr) {
      return 0;
    }
    return dataSet->data_buffers.size();
  }
  virtual aclDataBuffer *aclmdlGetDatasetBuffer(const aclmdlDataset *dataSet, size_t index) {
    if (dataSet == nullptr || index >= dataSet->data_buffers.size()) {
      return nullptr;
    }
    return dataSet->data_buffers[index];
  }
  ResourceBase<aclmdlDataset *> dataset_;
};

class AclEnv {
 public:
  virtual aclError aclInit(const char *configPath) {
    is_init = true;
    return ACL_ERROR_NONE;
  }
  virtual aclError aclFinalize() {
    is_init = false;
    return ACL_ERROR_NONE;
  }
  bool Check() { return is_init == false; }
  bool is_init = false;
};

class AclModel {
 public:
  bool Check() { return model_live_.empty(); }
  virtual aclError aclmdlLoadFromFile(const char *modelPath, uint32_t *modelId) {
    model_live_.push_back(cur_max_model_id_);
    *modelId = cur_max_model_id_;
    cur_max_model_id_++;
    return ACL_ERROR_NONE;
  }

  virtual aclError aclmdlLoadFromMem(const void *model, size_t modelSize, uint32_t *modelId) {
    return aclmdlLoadFromFile("fake_path", modelId);
  }

  virtual aclError aclmdlLoadFromFileWithMem(const char *modelPath, uint32_t *modelId, void *workPtr, size_t workSize,
                                             void *weightPtr, size_t weightSize) {
    return aclmdlLoadFromFile(modelPath, modelId);
  }

  virtual aclError aclmdlLoadFromMemWithMem(const void *model, size_t modelSize, uint32_t *modelId, void *workPtr,
                                            size_t workSize, void *weightPtr, size_t weightSize) {
    return aclmdlLoadFromMem(model, modelSize, modelId);
  }

  virtual aclError aclmdlExecute(uint32_t modelId, const aclmdlDataset *input, aclmdlDataset *output) {
    if (std::find(model_live_.begin(), model_live_.end(), modelId) == model_live_.end()) {
      return 1;
    }
    if (input == nullptr || output == nullptr) {
      return false;
    }
    // auto& model_desc = model_live_[modelId];
    return ACL_ERROR_NONE;
  }

  virtual aclError aclmdlExecuteAsync(uint32_t modelId, const aclmdlDataset *input, aclmdlDataset *output,
                                      aclrtStream stream) {
    return ACL_ERROR_NONE;
  }

  virtual aclError aclmdlUnload(uint32_t modelId) {
    auto it = std::find(model_live_.begin(), model_live_.end(), modelId);
    if (it == model_live_.end()) {
      return 1;
    }
    model_live_.erase(it);
    model_destroy_.push_back(modelId);
    return ACL_ERROR_NONE;
  }
  uint32_t cur_max_model_id_ = 0;
  std::vector<uint32_t> model_live_;
  std::vector<uint32_t> model_destroy_;
};

class AclModelDesc {
 public:
  AclModelDesc() {}
  virtual ~AclModelDesc() { Clear(); }
  virtual void Clear() { model_desc_.Clear(); }
  bool Check() { return model_desc_.Check(); }

 public:
  virtual aclmdlDesc *aclmdlCreateDesc() { return model_desc_.OnCreate(); }
  aclError aclmdlDestroyDesc(aclmdlDesc *modelDesc) { return model_desc_.OnDestroy(modelDesc); }

  aclError aclmdlGetDesc(aclmdlDesc *modelDesc, uint32_t modelId) {
    auto &model_live = g_acl_model->model_live_;
    auto it = std::find(model_live.begin(), model_live.end(), modelId);
    if (it == model_live.end()) {
      return 1;
    }
    return ACL_ERROR_NONE;
  }

  size_t aclmdlGetNumInputs(aclmdlDesc *modelDesc) { return modelDesc->inputs.size(); }

  size_t aclmdlGetNumOutputs(aclmdlDesc *modelDesc) { return modelDesc->outputs.size(); }

  size_t aclmdlGetInputSizeByIndex(aclmdlDesc *modelDesc, size_t index) { return modelDesc->inputs[index].size; }

  size_t aclmdlGetOutputSizeByIndex(aclmdlDesc *modelDesc, size_t index) { return modelDesc->outputs[index].size; }

  aclError aclmdlGetInputDims(const aclmdlDesc *modelDesc, size_t index, aclmdlIODims *dims) {
    auto &input = modelDesc->inputs[index];
    dims->dimCount = input.dims.size();
    for (size_t i = 0; i < dims->dimCount; i++) {
      dims->dims[i] = input.dims[i];
    }
    return ACL_ERROR_NONE;
  }

  aclError aclmdlGetOutputDims(const aclmdlDesc *modelDesc, size_t index, aclmdlIODims *dims) {
    auto &input = modelDesc->outputs[index];
    dims->dimCount = input.dims.size();
    for (size_t i = 0; i < dims->dimCount; i++) {
      dims->dims[i] = input.dims[i];
    }
    return ACL_ERROR_NONE;
  }

  aclError aclmdlGetCurOutputDims(const aclmdlDesc *modelDesc, size_t index, aclmdlIODims *dims) {
    return aclmdlGetOutputDims(modelDesc, index, dims);
  }

  aclFormat aclmdlGetInputFormat(const aclmdlDesc *modelDesc, size_t index) { return ACL_FORMAT_NCHW; }
  aclFormat aclmdlGetOutputFormat(const aclmdlDesc *modelDesc, size_t index) { return ACL_FORMAT_NCHW; }

  aclDataType aclmdlGetInputDataType(const aclmdlDesc *modelDesc, size_t index) {
    return modelDesc->inputs[index].data_type;
  }

  aclDataType aclmdlGetOutputDataType(const aclmdlDesc *modelDesc, size_t index) {
    return modelDesc->outputs[index].data_type;
  }

  ResourceBase<aclmdlDesc *> model_desc_;
};

class AclRunMode {
 public:
  virtual aclError aclrtGetRunMode(aclrtRunMode *runMode) {
    *runMode = aclrtRunMode::ACL_HOST;
    return ACL_ERROR_NONE;
  }
};

class AclDeviceContextStream {
 public:
  AclDeviceContextStream() {}
  ~AclDeviceContextStream() { Clear(); }
  virtual void Clear() {
    for (auto context : context_live_) {
      delete (int *)context;
    }
    context_live_.clear();
    context_destroy_.clear();
    device_id_live_.clear();
    device_id_destroy_.clear();
    for (auto item : stream_live_) {
      delete (int *)item;
    }
    stream_live_.clear();
    stream_destroy_.clear();
  }
  bool Check() { return context_live_.empty() && device_id_live_.empty() && stream_live_.empty(); }
  virtual aclError aclrtCreateContext(aclrtContext *context, int32_t deviceId) {
    context_live_.push_back(new int());
    *context = context_live_.back();
    return ACL_ERROR_NONE;
  }
  virtual aclError aclrtDestroyContext(aclrtContext context) {
    for (auto it = context_live_.begin(); it != context_live_.end(); ++it) {
      if (*it == context) {
        context_live_.erase(it);
        context_destroy_.push_back(context);
        delete (int *)context;
        return ACL_ERROR_NONE;
      }
    }
    return 1;
  }
  aclError aclrtSetCurrentContext(aclrtContext context) { return ACL_ERROR_NONE; }
  aclError aclrtGetCurrentContext(aclrtContext *context) { return ACL_ERROR_NONE; }
  virtual aclError aclrtSetDevice(int32_t deviceId) {
    device_id_live_.push_back(deviceId);
    return ACL_ERROR_NONE;
  }
  virtual aclError aclrtResetDevice(int32_t deviceId) {
    for (auto it = device_id_live_.begin(); it != device_id_live_.end(); ++it) {
      if (*it == deviceId) {
        device_id_live_.erase(it);
        device_id_destroy_.push_back(deviceId);
        return ACL_ERROR_NONE;
      }
    }
    return 1;
  }
  aclError aclrtGetDevice(int32_t *deviceId) {
    *deviceId = 0;
    return ACL_ERROR_NONE;
  }
  aclError aclrtSynchronizeDevice(void) { return ACL_ERROR_NONE; }
  aclError aclrtSetTsDevice(aclrtTsId tsId) { return ACL_ERROR_NONE; }
  aclError aclrtGetDeviceCount(uint32_t *count) {
    *count = 1;
    return ACL_ERROR_NONE;
  }
  virtual aclError aclrtCreateStream(aclrtStream *stream) {
    stream_live_.push_back(new int());
    *stream = stream_live_.back();
    return ACL_ERROR_NONE;
  }
  virtual aclError aclrtDestroyStream(aclrtStream stream) {
    for (auto it = stream_live_.begin(); it != context_live_.end(); ++it) {
      if (*it == stream) {
        stream_live_.erase(it);
        stream_destroy_.push_back(stream);
        delete (int *)stream;
        return ACL_ERROR_NONE;
      }
    }
    return 1;
  }
  aclError aclrtSynchronizeStream(aclrtStream stream) {
    for (auto it = stream_live_.begin(); it != context_live_.end(); ++it) {
      if (*it == stream) {
        return ACL_ERROR_NONE;
      }
    }
    return 1;
  }
  std::vector<int32_t> device_id_live_;
  std::vector<int32_t> device_id_destroy_;
  std::vector<aclrtContext> context_live_;
  std::vector<aclrtContext> context_destroy_;
  std::vector<aclrtStream> stream_live_;
  std::vector<aclrtStream> stream_destroy_;
};

class AclMemory {
 public:
  AclMemory() {}
  ~AclMemory() { Clear(); }
  void Clear() {
    for (auto item : device_buffer_live_) {
      delete[] item;
    }
    for (auto item : host_buffer_live_) {
      delete[] item;
    }
    for (auto item : dvpp_buffer_live_) {
      delete[] item;
    }
    device_buffer_live_.clear();
    device_buffer_destroy_.clear();
    host_buffer_live_.clear();
    host_buffer_destroy_.clear();
    dvpp_buffer_live_.clear();
    dvpp_buffer_destroy_.clear();
  }
  bool Check() { return device_buffer_live_.empty() && host_buffer_live_.empty() && dvpp_buffer_live_.empty(); }
  virtual aclError aclrtMalloc(void **devPtr, size_t size, aclrtMemMallocPolicy policy) {
    auto buffer = new uint8_t[size];
    *devPtr = buffer;
    device_buffer_live_.push_back(buffer);
    memory_len_[buffer] = size;
    return ACL_ERROR_NONE;
  }
  aclError aclrtFree(void *devPtr) {
    auto it = std::find(device_buffer_live_.begin(), device_buffer_live_.end(), devPtr);
    if (it != device_buffer_live_.end()) {
      delete[](*it);
      device_buffer_live_.erase(it);
      device_buffer_destroy_.push_back(*it);
      return ACL_ERROR_NONE;
    }
    return 1;
  }

  virtual aclError aclrtMallocHost(void **hostPtr, size_t size) {
    auto buffer = new uint8_t[size];
    *hostPtr = buffer;
    host_buffer_live_.push_back(buffer);
    memory_len_[buffer] = size;
    return ACL_ERROR_NONE;
  }

  aclError aclrtFreeHost(void *hostPtr) {
    auto it = std::find(host_buffer_live_.begin(), host_buffer_live_.end(), hostPtr);
    if (it != host_buffer_live_.end()) {
      delete[](*it);
      host_buffer_live_.erase(it);
      host_buffer_destroy_.push_back(*it);
      return ACL_ERROR_NONE;
    }
    return 1;
  }

  aclError aclrtMemcpy(void *dst, size_t destMax, const void *src, size_t count, aclrtMemcpyKind kind) {
    auto is_device_memory = [this](const void *memory, uint32_t use_size) {
      for (auto it = device_buffer_live_.begin(); it != device_buffer_live_.end(); it++) {
        auto size = memory_len_[*it];
        if (memory >= *it && static_cast<const uint8_t *>(memory) + use_size <= (*it) + size) {
          return true;
        }
      }
      for (auto it = dvpp_buffer_live_.begin(); it != dvpp_buffer_live_.end(); it++) {
        auto size = memory_len_[*it];
        if (memory >= *it && static_cast<const uint8_t *>(memory) + use_size <= (*it) + size) {
          return true;
        }
      }
      return false;
    };
    if (kind == ACL_MEMCPY_HOST_TO_HOST) {
      if (is_device_memory(dst, destMax) || is_device_memory(src, count)) {
        return 1;
      }
    } else if (kind == ACL_MEMCPY_HOST_TO_DEVICE) {
      if (!is_device_memory(dst, destMax) || is_device_memory(src, count)) {
        return 1;
      }
    } else if (kind == ACL_MEMCPY_DEVICE_TO_HOST) {
      if (is_device_memory(dst, destMax) || !is_device_memory(src, count)) {
        return 1;
      }
    } else if (kind == ACL_MEMCPY_DEVICE_TO_DEVICE) {
      if (!is_device_memory(dst, destMax) || !is_device_memory(src, count)) {
        return 1;
      }
    } else {
      return 1;
    }
    memcpy(dst, src, count);
    return ACL_ERROR_NONE;
  }

  virtual aclError acldvppMalloc(void **devPtr, size_t size) {
    auto buffer = new uint8_t[size];
    *devPtr = buffer;
    dvpp_buffer_live_.push_back(buffer);
    memory_len_[buffer] = size;
    return ACL_ERROR_NONE;
  }
  aclError acldvppFree(void *devPtr) {
    auto it = std::find(dvpp_buffer_live_.begin(), dvpp_buffer_live_.end(), devPtr);
    if (it != dvpp_buffer_live_.end()) {
      delete[](*it);
      dvpp_buffer_live_.erase(it);
      dvpp_buffer_destroy_.push_back(*it);
      return ACL_ERROR_NONE;
    }
    return 1;
  }

  std::vector<uint8_t *> device_buffer_live_;
  std::vector<uint8_t *> device_buffer_destroy_;
  std::vector<uint8_t *> host_buffer_live_;
  std::vector<uint8_t *> host_buffer_destroy_;
  std::vector<uint8_t *> dvpp_buffer_live_;
  std::vector<uint8_t *> dvpp_buffer_destroy_;
  std::map<uint8_t *, uint32_t> memory_len_;
};

class AclDvppPicDesc {
 public:
  bool Check() { return pic_desc_.Check(); }
  acldvppPicDesc *acldvppCreatePicDesc() { return pic_desc_.OnCreate(); }

  aclError acldvppDestroyPicDesc(acldvppPicDesc *picDesc) { return pic_desc_.OnDestroy(picDesc); }

  aclError acldvppSetPicDescSize(acldvppPicDesc *picDesc, uint32_t size) {
    picDesc->size = size;
    return ACL_ERROR_NONE;
  }

  aclError acldvppSetPicDescFormat(acldvppPicDesc *picDesc, acldvppPixelFormat format) {
    picDesc->format = format;
    return ACL_ERROR_NONE;
  }

  aclError acldvppSetPicDescWidth(acldvppPicDesc *picDesc, uint32_t width) {
    picDesc->width = width;
    return ACL_ERROR_NONE;
  }

  aclError acldvppSetPicDescHeight(acldvppPicDesc *picDesc, uint32_t height) {
    picDesc->height = height;
    return ACL_ERROR_NONE;
  }

  aclError acldvppSetPicDescData(acldvppPicDesc *picDesc, void *dataDev) {
    picDesc->dataDev = dataDev;
    return ACL_ERROR_NONE;
  }

  aclError acldvppSetPicDescWidthStride(acldvppPicDesc *picDesc, uint32_t widthStride) {
    picDesc->widthStride = widthStride;
    return ACL_ERROR_NONE;
  }

  aclError acldvppSetPicDescHeightStride(acldvppPicDesc *picDesc, uint32_t heightStride) {
    picDesc->heightStride = heightStride;
    return ACL_ERROR_NONE;
  }
  ResourceBase<acldvppPicDesc *> pic_desc_;
};

class AclDvppRoiConfig {
 public:
  bool Check() { return roi_config_.Check(); }
  acldvppRoiConfig *acldvppCreateRoiConfig(uint32_t left, uint32_t right, uint32_t top, uint32_t bottom) {
    return roi_config_.OnCreate(acldvppRoiConfig{.left = left, .right = right, .top = top, .bottom = bottom});
  }

  aclError acldvppDestroyRoiConfig(acldvppRoiConfig *roiConfig) { return roi_config_.OnDestroy(roiConfig); }

  aclError acldvppSetRoiConfig(acldvppRoiConfig *roiConfig, uint32_t left, uint32_t right, uint32_t top,
                               uint32_t bottom) {
    roiConfig->left = left;
    roiConfig->right = right;
    roiConfig->top = top;
    roiConfig->bottom = bottom;
    return ACL_ERROR_NONE;
  }
  ResourceBase<acldvppRoiConfig *> roi_config_;
};

class AclDvppResizeConfig {
 public:
  bool Check() { return resize_config_.Check(); }
  acldvppResizeConfig *acldvppCreateResizeConfig() { return resize_config_.OnCreate(acldvppResizeConfig{}); }

  aclError acldvppDestroyResizeConfig(acldvppResizeConfig *resizeConfig) {
    return resize_config_.OnDestroy(resizeConfig);
  }
  ResourceBase<acldvppResizeConfig *> resize_config_;
};

class AclDvppChannelDesc {
 public:
  bool Check() { return channel_desc_.Check(); }
  aclError acldvppCreateChannel(acldvppChannelDesc *channelDesc) {
    channelDesc->channel_valid_flag = true;
    return ACL_ERROR_NONE;
  }
  aclError acldvppDestroyChannel(acldvppChannelDesc *channelDesc) {
    channelDesc->channel_valid_flag = false;
    return ACL_ERROR_NONE;
  }
  acldvppChannelDesc *acldvppCreateChannelDesc() { return channel_desc_.OnCreate(); }
  aclError acldvppDestroyChannelDesc(acldvppChannelDesc *channelDesc) {
    if (channelDesc->channel_valid_flag) {
      return 1;
    }
    return channel_desc_.OnDestroy(channelDesc);
  }
  ResourceBase<acldvppChannelDesc *> channel_desc_;
};

class AclDvppProcess {
 public:
  bool Check() { return true; }
  virtual aclError acldvppVpcResizeAsync(acldvppChannelDesc *channelDesc, acldvppPicDesc *inputDesc,
                                         acldvppPicDesc *outputDesc, acldvppResizeConfig *resizeConfig,
                                         aclrtStream stream) {
    resize_call_times_++;
    if (channelDesc == nullptr || inputDesc == nullptr || outputDesc == nullptr || resizeConfig == nullptr ||
        stream == nullptr) {
      return 1;
    }
    if (CheckPicDesc(inputDesc) != ACL_ERROR_NONE) {
      return 1;
    }
    if (CheckPicDesc(outputDesc) != ACL_ERROR_NONE) {
      return 1;
    }
    return ACL_ERROR_NONE;
  }

  virtual aclError acldvppVpcCropAsync(acldvppChannelDesc *channelDesc, acldvppPicDesc *inputDesc,
                                       acldvppPicDesc *outputDesc, acldvppRoiConfig *cropArea, aclrtStream stream) {
    crop_call_times_++;
    if (channelDesc == nullptr || inputDesc == nullptr || outputDesc == nullptr || cropArea == nullptr ||
        stream == nullptr) {
      return 1;
    }
    if (CheckPicDesc(inputDesc) != ACL_ERROR_NONE) {
      return 1;
    }
    if (CheckPicDesc(outputDesc) != ACL_ERROR_NONE) {
      return 1;
    }
    if (CheckCropArea(cropArea) != ACL_ERROR_NONE) {
      return 1;
    }
    return ACL_ERROR_NONE;
  }

  virtual aclError acldvppVpcCropAndPasteAsync(acldvppChannelDesc *channelDesc, acldvppPicDesc *inputDesc,
                                               acldvppPicDesc *outputDesc, acldvppRoiConfig *cropArea,
                                               acldvppRoiConfig *pasteArea, aclrtStream stream) {
    crop_paste_call_times_++;
    if (channelDesc == nullptr || inputDesc == nullptr || outputDesc == nullptr || cropArea == nullptr ||
        pasteArea == nullptr || stream == nullptr) {
      return 1;
    }
    if (CheckPicDesc(inputDesc) != ACL_ERROR_NONE) {
      return 1;
    }
    if (CheckPicDesc(outputDesc) != ACL_ERROR_NONE) {
      return 1;
    }
    if (CheckCropArea(cropArea) != ACL_ERROR_NONE) {
      return 1;
    }
    if (CheckCropArea(pasteArea) != ACL_ERROR_NONE) {
      return 1;
    }
    return ACL_ERROR_NONE;
  }

  aclError acldvppVpcBatchCropAsync(acldvppChannelDesc *channelDesc, acldvppBatchPicDesc *srcBatchDesc,
                                    uint32_t *roiNums, uint32_t size, acldvppBatchPicDesc *dstBatchDesc,
                                    acldvppRoiConfig *cropAreas[], aclrtStream stream) {
    return ACL_ERROR_NONE;
  }

  virtual aclError acldvppJpegDecodeAsync(acldvppChannelDesc *channelDesc, const void *data, uint32_t size,
                                          acldvppPicDesc *outputDesc, aclrtStream stream) {
    decode_call_times_++;
    if (channelDesc == nullptr || data == nullptr || size == 0 || outputDesc == nullptr || stream == nullptr) {
      return 1;
    }
    if (outputDesc->widthStride % 128 != 0) {
      return 1;
    }
    if (outputDesc->heightStride % 16 != 0) {
      return 1;
    }
    if (outputDesc->widthStride < 32 || outputDesc->widthStride > 8192) {
      return 1;
    }
    if (outputDesc->heightStride < 32 || outputDesc->heightStride > 8192) {
      return 1;
    }
    if (CheckPicDesc(outputDesc) != ACL_ERROR_NONE) {
      return 1;
    }
    return ACL_ERROR_NONE;
  }
  aclError CheckCropArea(acldvppRoiConfig *crop_area) {
    if (crop_area->left % 2 != 0 || crop_area->top % 2 != 0) {
      return 1;
    }
    if (crop_area->right % 2 != 1 || crop_area->bottom % 2 != 1) {
      return 1;
    }
    auto crop_width = crop_area->right - crop_area->left + 1;
    if (crop_width < 10 || crop_width > 4096) {
      return 1;
    }
    auto crop_heigth = crop_area->bottom - crop_area->top + 1;
    if (crop_heigth < 6 || crop_heigth > 4096) {
      return 1;
    }
    return ACL_ERROR_NONE;
  }
  aclError CheckPicDesc(acldvppPicDesc *pic_desc) {
    if (pic_desc->width == 0 || pic_desc->height == 0) {
      return 1;
    }
    if (pic_desc->widthStride % 16 != 0 || pic_desc->widthStride < pic_desc->width) {
      return 1;
    }
    if (pic_desc->heightStride % 2 != 0 || pic_desc->heightStride < pic_desc->height) {
      return 1;
    }
    if (pic_desc->widthStride < 32 || pic_desc->widthStride > 4096) {
      return 1;
    }
    if (pic_desc->heightStride < 6 || pic_desc->heightStride > 4096) {
      return 1;
    }
    if (pic_desc->dataDev == nullptr) {
      return 1;
    }
    auto size = pic_desc->size;
    auto ele_cnt = pic_desc->widthStride * pic_desc->heightStride;
    switch (pic_desc->format) {
      case PIXEL_FORMAT_YUV_SEMIPLANAR_420:
      case PIXEL_FORMAT_YVU_SEMIPLANAR_420:
        if (ele_cnt * 3 / 2 != size) {
          return 1;
        }
        break;
      case PIXEL_FORMAT_YUV_SEMIPLANAR_422:
      case PIXEL_FORMAT_YVU_SEMIPLANAR_422:
        if (ele_cnt * 2 != size) {
          return 1;
        }
        break;
      case PIXEL_FORMAT_YUV_SEMIPLANAR_444:
      case PIXEL_FORMAT_YVU_SEMIPLANAR_444:
        if (ele_cnt * 3 != size) {
          return 1;
        }
        break;
      default:
        return 1;
    }
    return ACL_ERROR_NONE;
  }
  uint32_t decode_call_times_ = 0;
  uint32_t resize_call_times_ = 0;
  uint32_t crop_call_times_ = 0;
  uint32_t crop_paste_call_times_ = 0;
};

class AclJpegLib {
 public:
  bool Check() { return jpeg_live_.empty(); }
  AclJpegLib(uint32_t width, uint32_t height) : image_width_(width), image_height_(height) {}

  void jpeg_CreateDecompress(j_decompress_ptr cinfo, int version, size_t structsize) { jpeg_live_.push_back(cinfo); }
  void jpeg_mem_src(j_decompress_ptr cinfo, const unsigned char *inbuffer, unsigned long insize) {}
  int jpeg_read_header(j_decompress_ptr cinfo, boolean require_image) {
    static JHUFF_TBL tal;
    cinfo->image_width = image_width_;
    cinfo->image_height = image_height_;
    cinfo->jpeg_color_space = color_space_;
    for (int i = 0; i < NUM_HUFF_TBLS; i++) {
      cinfo->ac_huff_tbl_ptrs[i] = &tal;
      cinfo->dc_huff_tbl_ptrs[i] = &tal;
    }
    return 0;
  }
  void jpeg_destroy_decompress(j_decompress_ptr cinfo) {
    auto it = std::find(jpeg_live_.begin(), jpeg_live_.end(), cinfo);
    if (it != jpeg_live_.end()) {
      jpeg_live_.erase(it);
    }
  }
  uint32_t image_width_;
  uint32_t image_height_;
  J_COLOR_SPACE color_space_ = JCS_YCbCr;
  std::vector<j_decompress_ptr> jpeg_live_;
};

extern AclDataBuffer *g_acl_data_buffer;
extern AclEnv *g_acl_env;
extern AclDataSet *g_acl_dataset;
extern AclModelDesc *g_acl_model_desc;
extern AclDeviceContextStream *g_acl_device_context_stream;
extern AclMemory *g_acl_memory;
extern AclDvppPicDesc *g_acl_dvpp_pic_desc;
extern AclDvppRoiConfig *g_acl_dvpp_roi_config;
extern AclDvppResizeConfig *g_acl_dvpp_resize_config;
extern AclDvppChannelDesc *g_acl_dvpp_channel_desc;
extern AclDvppProcess *g_acl_dvpp_process;
extern AclRunMode *g_acl_run_mode;
extern AclJpegLib *g_acl_jpeg_lib;

#endif  // MINDSPORE_ACL_STUB_H
