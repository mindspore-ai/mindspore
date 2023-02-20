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

#include "cxx_api/graph/acl/model_process.h"
#include <sys/time.h>
#include <algorithm>
#include <map>
#include <memory>
#include <utility>

#include "mindspore/core/utils/convert_utils_base.h"
#include "acl/acl_rt.h"
#include "include/api/data_type.h"
#include "utils/log_adapter.h"

namespace mindspore {
static DataType TransToApiType(aclDataType data_type) {
  static const std::map<aclDataType, enum DataType> data_type_map = {
    {ACL_FLOAT16, DataType::kNumberTypeFloat16}, {ACL_FLOAT, DataType::kNumberTypeFloat32},
    {ACL_DOUBLE, DataType::kNumberTypeFloat64},  {ACL_INT8, DataType::kNumberTypeInt8},
    {ACL_INT16, DataType::kNumberTypeInt16},     {ACL_INT32, DataType::kNumberTypeInt32},
    {ACL_INT64, DataType::kNumberTypeInt64},     {ACL_UINT8, DataType::kNumberTypeUInt8},
    {ACL_UINT16, DataType::kNumberTypeUInt16},   {ACL_UINT32, DataType::kNumberTypeUInt32},
    {ACL_UINT64, DataType::kNumberTypeUInt64},   {ACL_BOOL, DataType::kNumberTypeBool},
  };
  auto it = data_type_map.find(data_type);
  if (it == data_type_map.end()) {
    return DataType::kTypeUnknown;
  } else {
    return it->second;
  }
}

template <class T>
inline static void ClearIfNotNull(T *vec) {
  if (vec != nullptr) {
    vec->clear();
  }
}

template <class T, class U = std::vector<T>>
inline static void PushbackIfNotNull(U *vec, T &&item) {
  if (vec != nullptr) {
    vec->emplace_back(item);
  }
}

static void ConstructTensorDesc(const std::vector<AclTensorInfo> &acl_tensor_list, std::vector<std::string> *names,
                                std::vector<std::vector<int64_t>> *shapes, std::vector<enum DataType> *data_types,
                                std::vector<size_t> *mem_sizes) {
  ClearIfNotNull(names);
  ClearIfNotNull(shapes);
  ClearIfNotNull(data_types);
  ClearIfNotNull(mem_sizes);
  for (size_t i = 0; i < acl_tensor_list.size(); ++i) {
    const auto &info = acl_tensor_list[i];
    PushbackIfNotNull(names, info.name);
    PushbackIfNotNull(shapes, info.dims);
    PushbackIfNotNull(data_types, TransToApiType(info.data_type));
    PushbackIfNotNull(mem_sizes, info.buffer_size);
  }
}

static std::string ShapeToString(const std::vector<int64_t> &shape) {
  std::string result = "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    result += std::to_string(shape[i]);
    if (i + 1 < shape.size()) {
      result += ", ";
    }
  }
  result += "]";
  return result;
}

Status ModelProcess::ConstructTensors(const std::vector<AclTensorInfo> &acl_tensor_list,
                                      std::vector<MSTensor> *tensor_list) const {
  MS_EXCEPTION_IF_NULL(tensor_list);
  std::vector<std::string> names;
  std::vector<std::vector<int64_t>> shapes;
  std::vector<enum DataType> data_types;
  std::vector<size_t> mem_sizes;

  ConstructTensorDesc(acl_tensor_list, &names, &shapes, &data_types, &mem_sizes);
  tensor_list->clear();
  if (names.size() != acl_tensor_list.size() || shapes.size() != acl_tensor_list.size() ||
      data_types.size() != acl_tensor_list.size() || mem_sizes.size() != acl_tensor_list.size()) {
    MS_LOG(ERROR) << "Inner error, size do not match: names size " << names.size() << " shapes size " << shapes.size()
                  << " data types size " << data_types.size() << " mem sizes size " << mem_sizes.size()
                  << " acl_tensor_list size " << acl_tensor_list.size();
    return kMCFailed;
  }

  aclrtMemcpyKind kind = is_run_on_device_ ? ACL_MEMCPY_HOST_TO_HOST : ACL_MEMCPY_DEVICE_TO_HOST;
  for (size_t i = 0; i < acl_tensor_list.size(); ++i) {
    tensor_list->emplace_back(names[i], data_types[i], shapes[i], nullptr, mem_sizes[i]);
    if (acl_tensor_list[i].cur_device_data == nullptr) {
      // when run on device, cur_device_data is nullptr before first execute
      continue;
    }
    auto ret = aclrtMemcpy((*tensor_list)[i].MutableData(), (*tensor_list)[i].DataSize(),
                           acl_tensor_list[i].cur_device_data, acl_tensor_list[i].buffer_size, kind);
    if (ret != ACL_ERROR_NONE) {
      MS_LOG(ERROR) << "Memcpy input " << i << " from " << (is_run_on_device_ ? "host" : "device")
                    << " to host failed, memory size " << acl_tensor_list[i].buffer_size;
      return kMCFailed;
    }
  }

  return kSuccess;
}

Status ModelProcess::PreInitModelResource() {
  model_desc_ = aclmdlCreateDesc();
  aclError acl_ret = aclmdlGetDesc(model_desc_, model_id_);
  if (acl_ret != ACL_ERROR_NONE) {
    MS_LOG(ERROR) << "Read model desc failed";
    return kMCDeviceError;
  }
  Status ret = InitInputsBuffer();
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "Create input buffer failed";
    return ret;
  }
  ret = InitOutputsBuffer();
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "Create output buffer failed";
    return ret;
  }
  return kSuccess;
}

Status ModelProcess::InitInputsBuffer() {
  aclError ret;
  size_t input_size = aclmdlGetNumInputs(model_desc_);
  MS_LOG(INFO) << "input_size = " << input_size;
  for (size_t i = 0; i < input_size; ++i) {
    auto buffer_size = aclmdlGetInputSizeByIndex(model_desc_, i);
    void *data_mem_buffer = nullptr;
    if (!is_run_on_device_) {  // need to copy input/output to/from device
      ret = aclrtMalloc(&data_mem_buffer, buffer_size, ACL_MEM_MALLOC_NORMAL_ONLY);
      if (ret != ACL_ERROR_NONE) {
        MS_LOG(ERROR) << "Malloc device input buffer failed , input size " << buffer_size;
        return kMCDeviceError;
      }
    }

    aclmdlIODims dims;
    ret = aclmdlGetInputDims(model_desc_, i, &dims);
    if (ret != ACL_ERROR_NONE) {
      MS_LOG(ERROR) << "Get input shape failed";
      if (!is_run_on_device_) {
        (void)aclrtFree(data_mem_buffer);
      }
      return kMCDeviceError;
    }
    aclDataType data_type = aclmdlGetInputDataType(model_desc_, i);
    std::vector<int64_t> shape(dims.dims, dims.dims + dims.dimCount);
    const char *input_name_char = aclmdlGetInputNameByIndex(model_desc_, i);
    std::string input_name = (input_name_char != nullptr) ? input_name_char : std::string();
    if (input_name.empty()) {
      MS_LOG(WARNING) << "Get name of input " << i << " failed.";
    }
    MS_LOG(INFO) << "Name of input " << i << " is " << input_name;
    input_infos_.emplace_back(
      AclTensorInfo{data_mem_buffer, data_mem_buffer, buffer_size, data_type, shape, input_name});
  }
  MS_LOG(INFO) << "Create model inputs success";
  return kSuccess;
}

Status ModelProcess::CreateDataBuffer(void **data_mem_buffer, size_t buffer_size, aclmdlDataset *dataset) const {
  MS_EXCEPTION_IF_NULL(data_mem_buffer);
  aclError ret;
  auto free_data_buffer = [this](void *dataMemBuffer) {
    if (!is_run_on_device_) {
      (void)aclrtFree(dataMemBuffer);
    } else {
      (void)aclrtFreeHost(dataMemBuffer);
    }
  };

  if (!is_run_on_device_) {
    ret = aclrtMalloc(data_mem_buffer, buffer_size, ACL_MEM_MALLOC_NORMAL_ONLY);
    if (ret != ACL_ERROR_NONE) {
      MS_LOG(ERROR) << "Malloc device buffer failed , buffer size " << buffer_size;
      return kMCDeviceError;
    }
  } else {
    ret = aclrtMallocHost(data_mem_buffer, buffer_size);
    if (ret != ACL_ERROR_NONE) {
      MS_LOG(ERROR) << "Malloc device buffer failed , buffer size " << buffer_size;
      return kMCDeviceError;
    }
  }

  auto data_buffer = aclCreateDataBuffer(*data_mem_buffer, buffer_size);
  if (data_buffer == nullptr) {
    MS_LOG(ERROR) << "Create Data Buffer failed";
    free_data_buffer(*data_mem_buffer);
    return kMCDeviceError;
  }
  ret = aclmdlAddDatasetBuffer(dataset, data_buffer);
  if (ret != ACL_ERROR_NONE) {
    MS_LOG(ERROR) << "add data buffer failed";
    free_data_buffer(*data_mem_buffer);
    (void)aclDestroyDataBuffer(data_buffer);
    return kMCDeviceError;
  }
  return kSuccess;
}

Status ModelProcess::InitOutputsBuffer() {
  aclError ret;
  outputs_ = aclmdlCreateDataset();
  if (outputs_ == nullptr) {
    MS_LOG(ERROR) << "Create input dataset failed";
    return kMCDeviceError;
  }
  size_t output_size = aclmdlGetNumOutputs(model_desc_);
  MS_LOG(INFO) << "output_size = " << output_size;
  for (size_t i = 0; i < output_size; ++i) {
    auto buffer_size = aclmdlGetOutputSizeByIndex(model_desc_, i);

    void *data_mem_buffer = nullptr;
    if (CreateDataBuffer(&data_mem_buffer, buffer_size, outputs_) != kSuccess) {
      MS_LOG(ERROR) << "add output data buffer failed, buffer size " << buffer_size;
      return kMCDeviceError;
    }
    aclmdlIODims dims;
    ret = aclmdlGetOutputDims(model_desc_, i, &dims);
    if (ret != ACL_ERROR_NONE) {
      MS_LOG(ERROR) << "Get input shape failed";
      if (!is_run_on_device_) {
        (void)aclrtFree(data_mem_buffer);
      } else {
        (void)aclrtFreeHost(data_mem_buffer);
      }
      return kMCDeviceError;
    }
    aclDataType data_type = aclmdlGetOutputDataType(model_desc_, i);
    std::vector<int64_t> shape(dims.dims, dims.dims + dims.dimCount);
    const char *output_name_char = aclmdlGetOutputNameByIndex(model_desc_, i);
    std::string output_name = (output_name_char != nullptr) ? output_name_char : std::string();
    if (output_name.empty()) {
      MS_LOG(WARNING) << "Get name of output " << i << " failed.";
    }
    MS_LOG(INFO) << "Name of input " << i << " is " << output_name;
    output_infos_.emplace_back(
      AclTensorInfo{data_mem_buffer, data_mem_buffer, buffer_size, data_type, shape, output_name});
  }
  MS_LOG(INFO) << "Create model output success";
  return kSuccess;
}

void ModelProcess::DestroyInputsDataset() {
  if (inputs_ == nullptr) {
    return;
  }
  for (size_t i = 0; i < aclmdlGetDatasetNumBuffers(inputs_); i++) {
    auto dataBuffer = aclmdlGetDatasetBuffer(inputs_, i);
    (void)aclDestroyDataBuffer(dataBuffer);
  }
  (void)aclmdlDestroyDataset(inputs_);
  inputs_ = nullptr;
}

void ModelProcess::DestroyInputsDataMem() {
  if (!is_run_on_device_) {
    for (const auto &item : input_infos_) {
      (void)aclrtFree(item.device_data);
    }
  }
  input_infos_.clear();
}

void ModelProcess::DestroyInputsBuffer() {
  DestroyInputsDataMem();
  DestroyInputsDataset();
}

void ModelProcess::DestroyOutputsBuffer() {
  for (const auto &item : output_infos_) {
    if (!is_run_on_device_) {
      (void)aclrtFree(item.device_data);
    } else {
      (void)aclrtFreeHost(item.device_data);
    }
  }
  output_infos_.clear();

  if (outputs_ == nullptr) {
    return;
  }
  for (size_t i = 0; i < aclmdlGetDatasetNumBuffers(outputs_); i++) {
    auto dataBuffer = aclmdlGetDatasetBuffer(outputs_, i);
    (void)aclDestroyDataBuffer(dataBuffer);
  }
  (void)aclmdlDestroyDataset(outputs_);
  outputs_ = nullptr;
}

Status ModelProcess::UnLoad() {
  auto ret = aclmdlUnload(model_id_);
  if (ret != ACL_ERROR_NONE) {
    MS_LOG(ERROR) << "Unload model failed";
    return kMCDeviceError;
  }
  if (model_desc_ != nullptr) {
    ret = aclmdlDestroyDesc(model_desc_);
    if (ret != ACL_ERROR_NONE) {
      MS_LOG(ERROR) << "Unload model failed";
      return kMCDeviceError;
    }
    model_desc_ = nullptr;
  }
  DestroyInputsBuffer();
  DestroyOutputsBuffer();
  MS_LOG(INFO) << "End unload model " << model_id_;
  return kSuccess;
}

size_t ModelProcess::GetDynamicDims(const std::vector<AclTensorInfo> &inputs) const {
  size_t max_num = 0;
  for (auto input : inputs) {
    size_t cur_num = LongToSize(std::count(input.dims.begin(), input.dims.end(), -1));
    if (cur_num > max_num) {
      max_num = cur_num;
    }
  }
  return max_num;
}

Status ModelProcess::SetBatchSize(const std::vector<MSTensor> &inputs) {
  size_t index;
  aclError ret;
  for (size_t i = 0; i < inputs.size(); i++) {
    input_infos_[i].buffer_size = inputs[i].DataSize();
  }
  auto *p = static_cast<const float *>(inputs[inputs.size() - 1].Data().get());
  MS_EXCEPTION_IF_NULL(p);
  size_t dynamicBatchSize = FloatToSize(p[0]);
  ret = aclmdlGetInputIndexByName(model_desc_, ACL_DYNAMIC_TENSOR_NAME, &index);
  if (ret != ACL_ERROR_NONE) {
    MS_LOG(ERROR) << "get index failed";
    return kMCDeviceError;
  }
  ret = aclmdlSetDynamicBatchSize(model_id_, inputs_, index, dynamicBatchSize);
  if (ret != ACL_ERROR_NONE) {
    MS_LOG(ERROR) << "dynamic batch set failed, modelId is " << model_id_;
    return kMCDeviceError;
  }
  return kSuccess;
}

Status ModelProcess::CheckAndInitInput(const std::vector<MSTensor> &inputs) {
  aclError ret;
  inputs_ = aclmdlCreateDataset();
  constexpr size_t dynamic_batch_size = 1;
  constexpr size_t dynamic_image_size = 2;
  size_t dynamic_nums = GetDynamicDims(input_infos_);
  // check inputs
  if (inputs.size() != input_infos_.size()) {
    MS_LOG(ERROR) << "Inputs count not match, required count " << input_infos_.size() << ", given count "
                  << inputs.size();
    return kMCInvalidInput;
  }
  if (dynamic_nums == 0) {
    for (size_t i = 0; i < input_infos_.size(); ++i) {
      if (inputs[i].Shape() != input_infos_[i].dims) {
        MS_LOG(INFO) << "Note: input " << i << " shape not match, required " << ShapeToString(input_infos_[i].dims)
                     << ", given " << ShapeToString(inputs[i].Shape());
      }
      if (inputs[i].DataType() != TransToApiType(input_infos_[i].data_type)) {
        MS_LOG(INFO) << "Note: input " << i << " data type not match, required "
                     << TransToApiType(input_infos_[i].data_type) << ", given " << inputs[i].DataType();
      }
      if (inputs[i].DataSize() != input_infos_[i].buffer_size) {
        MS_LOG(ERROR) << "Input " << i << " data size not match, required size " << input_infos_[i].buffer_size
                      << ", given count " << inputs[i].DataSize();
        return kMCInvalidInput;
      }
    }
  }
  // copy inputs
  for (size_t i = 0; i < input_infos_.size(); ++i) {
    auto &info = input_infos_[i];
    auto input = inputs[i];
    void *data = input.MutableData();
    void *input_buffer = nullptr;
    if (!is_run_on_device_) {
      if (input.IsDevice()) {
        info.cur_device_data = data;
        input_buffer = info.cur_device_data;
      } else {
        info.cur_device_data = info.device_data;
        ret = aclrtMemcpy(info.cur_device_data, info.buffer_size, data, input.DataSize(), ACL_MEMCPY_HOST_TO_DEVICE);
        if (ret != ACL_ERROR_NONE) {
          MS_LOG(ERROR) << "Acl memcpy input " << i << " data to device failed, buffer size " << input.DataSize();
          return kMCDeviceError;
        }
        input_buffer = info.cur_device_data;
      }
    } else {
      input_buffer = data;
    }
    auto data_buffer = aclCreateDataBuffer(input_buffer, info.buffer_size);
    if (data_buffer == nullptr) {
      MS_LOG(ERROR) << "Create Data Buffer failed";
      return kMCDeviceError;
    }
    ret = aclmdlAddDatasetBuffer(inputs_, data_buffer);
    if (ret != ACL_ERROR_NONE) {
      MS_LOG(ERROR) << "add data buffer failed";
      (void)aclDestroyDataBuffer(data_buffer);
      return kMCDeviceError;
    }
  }
  if (dynamic_nums == dynamic_batch_size) {
    if (SetBatchSize(inputs) != kSuccess) {
      MS_LOG(ERROR) << "failed to convert dynamic batch size";
      return kMCDeviceError;
    }
    if (ResetOutputSize() != kSuccess) {
      MS_LOG(ERROR) << "reset output size failed";
      return kMCDeviceError;
    }
  } else if (dynamic_nums == dynamic_image_size) {
    MS_LOG(ERROR) << "only dynamic batch size is supported";
    return kMCInvalidInput;
  }
  return kSuccess;
}

Status ModelProcess::ResetOutputSize() {
  aclDataType output_type;
  aclError ret;
  size_t output_size = aclmdlGetNumOutputs(model_desc_);
  for (size_t index = 0; index < output_size; index++) {
    int64_t dims = 1;
    struct aclmdlIODims output_dims;
    ret = aclmdlGetCurOutputDims(model_desc_, index, &output_dims);
    if (ret != ACL_ERROR_NONE) {
      MS_LOG(ERROR) << "get output dim error.";
      return kMCDeviceError;
    }
    for (size_t i = 0; i < output_dims.dimCount; i++) {
      dims *= output_dims.dims[i];
    }
    output_type = aclmdlGetOutputDataType(model_desc_, index);
    output_infos_[index].buffer_size = LongToSize(dims) * aclDataTypeSize(output_type);
  }
  return kSuccess;
}

Status ModelProcess::PredictFromHost(const std::vector<MSTensor> &inputs, std::vector<MSTensor> *outputs) {
  MS_EXCEPTION_IF_NULL(outputs);
  aclError acl_ret;
  Status ret = CheckAndInitInput(inputs);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "check or init input failed";
    DestroyInputsDataset();
    return ret;  // forward status error
  }

  struct timeval start_time;
  struct timeval end_time;
  (void)gettimeofday(&start_time, nullptr);
  acl_ret = aclmdlExecute(model_id_, inputs_, outputs_);
  (void)gettimeofday(&end_time, nullptr);
  constexpr uint64_t kUSecondInSecond = 1000000;
  uint64_t cost =
    (kUSecondInSecond * static_cast<uint64_t>(end_time.tv_sec) + static_cast<uint64_t>(end_time.tv_usec)) -
    (kUSecondInSecond * static_cast<uint64_t>(start_time.tv_sec) + static_cast<uint64_t>(start_time.tv_usec));
  MS_LOG(INFO) << "Model execute in " << cost << " us";

  DestroyInputsDataset();
  if (acl_ret != ACL_ERROR_NONE) {
    MS_LOG(ERROR) << "Execute Model Failed";
    return kMCDeviceError;
  }
  ret = BuildOutputs(outputs);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "Build outputs failed";
    return ret;
  }
  MS_LOG(INFO) << "Execute model success";
  return kSuccess;
}

Status ModelProcess::BuildOutputs(std::vector<MSTensor> *outputs) {
  MS_EXCEPTION_IF_NULL(outputs);
  // copy outputs
  outputs->clear();
  auto inner_outputs = GetOutputs();
  if (inner_outputs.size() != output_infos_.size()) {
    MS_LOG(ERROR) << "Invalid inner outputs size " << inner_outputs.size() << " do not match device output infos size "
                  << output_infos_.size();
    return kMCFailed;
  }
  (*outputs) = inner_outputs;
  return kSuccess;
}

std::vector<MSTensor> ModelProcess::GetInputs() {
  Status ret = ConstructTensors(input_infos_, &input_tensors_);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "ConstructTensors failed.";
    input_tensors_.clear();
  }

  return input_tensors_;
}

std::vector<MSTensor> ModelProcess::GetOutputs() {
  Status ret = ConstructTensors(output_infos_, &output_tensors_);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "ConstructTensors failed.";
    output_tensors_.clear();
  }

  return output_tensors_;
}
}  // namespace mindspore
