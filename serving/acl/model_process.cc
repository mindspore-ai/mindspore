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

#include "serving/acl/model_process.h"
#include <algorithm>
#include <unordered_map>

#include "include/infer_log.h"

namespace mindspore {
namespace inference {

bool ModelProcess::LoadModelFromFile(const std::string &file_name, uint32_t &model_id) {
  aclError acl_ret = aclmdlLoadFromFile(file_name.c_str(), &model_id);
  if (acl_ret != ACL_ERROR_NONE) {
    MSI_LOG_ERROR << "Read model file failed, file name is " << file_name;
    return false;
  }
  MSI_LOG_INFO << "Load model success " << file_name;

  model_desc_ = aclmdlCreateDesc();
  acl_ret = aclmdlGetDesc(model_desc_, model_id);
  if (acl_ret != ACL_ERROR_NONE) {
    MSI_LOG_ERROR << "Read model desc failed";
    return false;
  }
  bool ret = InitInputsBuffer();
  if (!ret) {
    MSI_LOG_ERROR << "Create input buffer failed";
    return false;
  }
  ret = InitOutputsBuffer();
  if (!ret) {
    MSI_LOG_ERROR << "Create output buffer failed";
    return false;
  }
  model_id_ = model_id;
  return true;
}

bool ModelProcess::InitInputsBuffer() {
  aclError ret;
  inputs_ = aclmdlCreateDataset();
  if (inputs_ == nullptr) {
    MSI_LOG_ERROR << "Create input dataset failed";
    return false;
  }
  size_t input_size = aclmdlGetNumInputs(model_desc_);

  for (size_t i = 0; i < input_size; ++i) {
    auto buffer_size = aclmdlGetInputSizeByIndex(model_desc_, i);
    void *data_mem_buffer = nullptr;
    if (!is_run_on_device_) {  // need to copy input/output to/from device
      ret = aclrtMalloc(&data_mem_buffer, buffer_size, ACL_MEM_MALLOC_NORMAL_ONLY);
      if (ret != ACL_ERROR_NONE) {
        MSI_LOG_ERROR << "Malloc device input buffer faild , input size " << buffer_size;
        return false;
      }
    }

    aclmdlIODims dims;
    ret = aclmdlGetInputDims(model_desc_, i, &dims);
    if (ret != ACL_ERROR_NONE) {
      MSI_LOG_ERROR << "Get input shape failed";
      return false;
    }
    aclDataType dataType = aclmdlGetInputDataType(model_desc_, i);
    std::vector<int64_t> shape(dims.dims, dims.dims + dims.dimCount);
    input_infos_.emplace_back(AclTensorInfo{data_mem_buffer, buffer_size, dataType, shape});
  }
  MSI_LOG_INFO << "Create model inputs success";
  return true;
}

bool ModelProcess::CreateDataBuffer(void *&data_mem_buffer, size_t buffer_size, aclmdlDataset *dataset) {
  aclError ret;
  auto free_data_buffer = [this](void *dataMemBuffer) {
    if (!is_run_on_device_) {
      aclrtFree(dataMemBuffer);
    } else {
      aclrtFreeHost(dataMemBuffer);
    }
  };
  if (!is_run_on_device_) {
    ret = aclrtMalloc(&data_mem_buffer, buffer_size, ACL_MEM_MALLOC_NORMAL_ONLY);
    if (ret != ACL_ERROR_NONE) {
      MSI_LOG_ERROR << "Malloc device buffer faild , buffer size " << buffer_size;
      return false;
    }
  } else {
    ret = aclrtMallocHost(&data_mem_buffer, buffer_size);
    if (ret != ACL_ERROR_NONE) {
      MSI_LOG_ERROR << "Malloc device buffer faild , buffer size " << buffer_size;
      return false;
    }
  }

  auto data_buffer = aclCreateDataBuffer(data_mem_buffer, buffer_size);
  if (data_buffer == nullptr) {
    MSI_LOG_ERROR << "Create Data Buffer failed";
    free_data_buffer(data_mem_buffer);
    return false;
  }
  ret = aclmdlAddDatasetBuffer(dataset, data_buffer);
  if (ret != ACL_ERROR_NONE) {
    MSI_LOG_ERROR << "add data buffer failed";
    free_data_buffer(data_mem_buffer);
    aclDestroyDataBuffer(data_buffer);
    return false;
  }
  return true;
}

bool ModelProcess::InitOutputsBuffer() {
  aclError ret;
  outputs_ = aclmdlCreateDataset();
  if (outputs_ == nullptr) {
    MSI_LOG_ERROR << "Create input dataset failed";
    return false;
  }
  size_t output_size = aclmdlGetNumOutputs(model_desc_);
  for (size_t i = 0; i < output_size; ++i) {
    auto buffer_size = aclmdlGetOutputSizeByIndex(model_desc_, i);

    void *data_mem_buffer = nullptr;
    if (CreateDataBuffer(data_mem_buffer, buffer_size, outputs_) != true) {
      MSI_LOG_ERROR << "add output data buffer failed, buffer size " << buffer_size;
      return false;
    }
    aclmdlIODims dims;
    ret = aclmdlGetOutputDims(model_desc_, i, &dims);
    if (ret != ACL_ERROR_NONE) {
      MSI_LOG_ERROR << "Get input shape failed";
      return false;
    }
    aclDataType dataType = aclmdlGetOutputDataType(model_desc_, i);
    std::vector<int64_t> shape(dims.dims, dims.dims + dims.dimCount);
    output_infos_.emplace_back(AclTensorInfo{data_mem_buffer, buffer_size, dataType, shape});
  }
  MSI_LOG_INFO << "Create model output success";
  return true;
}

void ModelProcess::DestroyInputsDataset() {
  if (inputs_ == nullptr) {
    return;
  }
  for (size_t i = 0; i < aclmdlGetDatasetNumBuffers(inputs_); i++) {
    auto dataBuffer = aclmdlGetDatasetBuffer(inputs_, i);
    aclDestroyDataBuffer(dataBuffer);
  }
  aclmdlDestroyDataset(inputs_);
  inputs_ = nullptr;
}

void ModelProcess::DestroyInputsDataMem() {
  if (!is_run_on_device_) {
    for (const auto &item : input_infos_) {
      aclrtFree(item.device_data);
    }
  }
  input_infos_.clear();
}

void ModelProcess::DestroyInputsBuffer() {
  DestroyInputsDataset();
  DestroyInputsDataMem();
}

void ModelProcess::DestroyOutputsBuffer() {
  if (outputs_ == nullptr) {
    return;
  }
  for (size_t i = 0; i < aclmdlGetDatasetNumBuffers(outputs_); i++) {
    auto dataBuffer = aclmdlGetDatasetBuffer(outputs_, i);
    auto data = aclGetDataBufferAddr(dataBuffer);
    if (!is_run_on_device_) {
      aclrtFree(data);
    } else {
      aclrtFreeHost(data);
    }
    aclDestroyDataBuffer(dataBuffer);
  }
  aclmdlDestroyDataset(outputs_);
  outputs_ = nullptr;
  output_infos_.clear();
}

void ModelProcess::UnLoad() {
  auto ret = aclmdlUnload(model_id_);
  if (ret != ACL_ERROR_NONE) {
    MSI_LOG_ERROR << "Unload model failed";
  }
  if (model_desc_ != nullptr) {
    aclmdlDestroyDesc(model_desc_);
    model_desc_ = nullptr;
  }
  DestroyInputsBuffer();
  DestroyOutputsBuffer();
  MSI_LOG_INFO << "End unload model " << model_id_;
}

bool ModelProcess::CheckAndInitInput(const RequestBase &request) {
  aclError ret;
  inputs_ = aclmdlCreateDataset();
  // check inputs
  if (request.size() != input_infos_.size()) {
    MSI_LOG_ERROR << "inputs count not match, required count " << input_infos_.size() << ", given count "
                  << request.size();
    return false;
  }
  for (size_t i = 0; i < input_infos_.size(); i++) {
    if (request[i] == nullptr) {
      MSI_LOG_ERROR << "input " << i << " cannot be null";
      return false;
    }
    if (request[i]->data_size() != input_infos_[i].buffer_size) {
      MSI_LOG_ERROR << "input " << i << " data size not match, required size " << input_infos_[i].buffer_size
                    << ", given count " << request[i]->data_size();
      return false;
    }
  }
  // copy inputs
  for (size_t i = 0; i < input_infos_.size(); i++) {
    void *input_buffer = nullptr;
    auto &info = input_infos_[i];
    const void *data = request[i]->data();
    if (!is_run_on_device_) {
      ret = aclrtMemcpy(info.device_data, info.buffer_size, data, request[i]->data_size(), ACL_MEMCPY_HOST_TO_DEVICE);
      if (ret != ACL_ERROR_NONE) {
        MSI_LOG_ERROR << "memcpy input " << i << " data to device failed, buffer size " << request[i]->data_size();
        return false;
      }
      input_buffer = info.device_data;
    } else {
      input_buffer = const_cast<void *>(data);
    }
    auto data_buffer = aclCreateDataBuffer(input_buffer, info.buffer_size);
    if (data_buffer == nullptr) {
      MSI_LOG_ERROR << "Create Data Buffer failed";
      return false;
    }
    ret = aclmdlAddDatasetBuffer(inputs_, data_buffer);
    if (ret != ACL_ERROR_NONE) {
      MSI_LOG_ERROR << "add data buffer failed";
      aclDestroyDataBuffer(data_buffer);
      return false;
    }
  }
  return true;
}

bool ModelProcess::BuildOutputs(ReplyBase &reply) {
  aclError ret;
  // copy outputs
  reply.clear();

  std::unordered_map<aclDataType, inference::DataType> dataTypeMap = {
    {ACL_FLOAT16, inference::kMSI_Float16}, {ACL_FLOAT, inference::kMSI_Float32}, {ACL_DOUBLE, inference::kMSI_Float64},
    {ACL_INT8, inference::kMSI_Int8},       {ACL_INT16, inference::kMSI_Int16},   {ACL_INT32, inference::kMSI_Int32},
    {ACL_INT64, inference::kMSI_Int64},     {ACL_UINT8, inference::kMSI_Uint8},   {ACL_UINT16, inference::kMSI_Uint16},
    {ACL_UINT32, inference::kMSI_Uint32},   {ACL_UINT64, inference::kMSI_Uint64}, {ACL_BOOL, inference::kMSI_Bool},
  };
  auto trans_to_serving_type = [&dataTypeMap](aclDataType data_type) {
    auto it = dataTypeMap.find(data_type);
    if (it == dataTypeMap.end()) {
      return inference::kMSI_Unknown;
    } else {
      return it->second;
    }
  };
  for (size_t i = 0; i < output_infos_.size(); i++) {
    auto &info = output_infos_[i];
    auto output = reply.add();
    if (output == nullptr) {
      MSI_LOG_ERROR << "add new output failed";
      return false;
    }
    output->set_data_type(trans_to_serving_type(info.data_type));
    output->set_shape(info.dims);
    if (!output->resize_data(info.buffer_size)) {
      MSI_LOG_ERROR << "new output data buffer failed, data size " << info.buffer_size;
      return false;
    }
    if (!is_run_on_device_) {
      ret = aclrtMemcpy(output->mutable_data(), output->data_size(), info.device_data, info.buffer_size,
                        ACL_MEMCPY_DEVICE_TO_HOST);
      if (ret != ACL_ERROR_NONE) {
        MSI_LOG_ERROR << "Memcpy output " << i << " to host failed, memory size " << info.buffer_size;
        return false;
      }
    } else {
      ret = aclrtMemcpy(output->mutable_data(), output->data_size(), info.device_data, info.buffer_size,
                        ACL_MEMCPY_HOST_TO_HOST);
      if (ret != ACL_ERROR_NONE) {
        MSI_LOG_ERROR << "Memcpy output " << i << " to host failed, memory size " << info.buffer_size;
        return false;
      }
    }
  }
  return true;
}

bool ModelProcess::Execute(const RequestBase &request, ReplyBase &reply) {
  aclError acl_ret;
  if (CheckAndInitInput(request) != true) {
    MSI_LOG_ERROR << "check or init input failed";
    DestroyInputsDataset();
    return false;
  }
  acl_ret = aclmdlExecute(model_id_, inputs_, outputs_);
  DestroyInputsDataset();
  if (acl_ret != ACL_ERROR_NONE) {
    MSI_LOG_ERROR << "Execute Model Failed";
    return false;
  }
  bool ret = BuildOutputs(reply);
  if (!ret) {
    MSI_LOG_ERROR << "Build outputs faield";
    return false;
  }
  MSI_LOG_INFO << "excute model success";
  return true;
}

}  // namespace inference
}  // namespace mindspore
