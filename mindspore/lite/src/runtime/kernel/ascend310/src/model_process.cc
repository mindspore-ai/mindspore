/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "src/runtime/kernel/ascend310/src/model_process.h"
#include <utility>
#include <algorithm>
#include <map>
#include "common/log_adapter.h"

namespace mindspore::kernel {
namespace acl {
namespace {
constexpr size_t kDynamicBatchSize = 1;
constexpr size_t kDynamicImageSize = 2;
}  // namespace
static DataType TransToDataType(aclDataType data_type) {
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
    return DataType::kNumberTypeEnd;
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

static STATUS ConstructTensorDesc(const std::vector<AclTensorInfo> &acl_tensor_list, std::vector<std::string> *names,
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
    PushbackIfNotNull(data_types, TransToDataType(info.data_type));
    PushbackIfNotNull(mem_sizes, info.buffer_size);
  }

  if (names->size() != acl_tensor_list.size() || shapes->size() != acl_tensor_list.size() ||
      data_types->size() != acl_tensor_list.size() || mem_sizes->size() != acl_tensor_list.size()) {
    MS_LOG(ERROR) << "Inner error, size do not match: names size " << names->size() << " shapes size " << shapes->size()
                  << " data types size " << data_types->size() << " mem sizes size " << mem_sizes->size()
                  << " acl_tensor_list size " << acl_tensor_list.size();
    return lite::RET_ERROR;
  }

  return lite::RET_OK;
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

STATUS ModelProcess::PreInitModelResource() {
  model_desc_ = aclmdlCreateDesc();
  aclError acl_ret = aclmdlGetDesc(model_desc_, model_id_);
  if (acl_ret != ACL_ERROR_NONE) {
    MS_LOG(ERROR) << "Read model desc failed, ret = " << acl_ret;
    return lite::RET_ERROR;
  }
  STATUS ret = InitInputsBuffer();
  if (ret != lite::RET_OK) {
    MS_LOG(ERROR) << "Create input buffer failed.";
    return ret;
  }
  ret = InitOutputsBuffer();
  if (ret != lite::RET_OK) {
    MS_LOG(ERROR) << "Create output buffer failed.";
    return ret;
  }
  return lite::RET_OK;
}

STATUS ModelProcess::InitInputsBuffer() {
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
        return lite::RET_ERROR;
      }
    }

    aclmdlIODims dims;
    ret = aclmdlGetInputDims(model_desc_, i, &dims);
    if (ret != ACL_ERROR_NONE) {
      MS_LOG(ERROR) << "Get input shape failed, ret = " << ret;
      if (!is_run_on_device_) {
        aclrtFree(data_mem_buffer);
      }
      return lite::RET_ERROR;
    }
    aclDataType data_type = aclmdlGetInputDataType(model_desc_, i);
    std::vector<int64_t> shape(dims.dims, dims.dims + dims.dimCount);
    std::string input_name = aclmdlGetInputNameByIndex(model_desc_, i);
    if (input_name.empty()) {
      MS_LOG(WARNING) << "Get name of input " << i << " failed.";
    }
    MS_LOG(INFO) << "Name of input " << i << " is " << input_name;
    input_infos_.emplace_back(
      AclTensorInfo{data_mem_buffer, data_mem_buffer, buffer_size, data_type, shape, input_name});
  }
  MS_LOG(INFO) << "Create model inputs success";
  return lite::RET_OK;
}

STATUS ModelProcess::CreateDataBuffer(void **data_mem_buffer, size_t buffer_size, aclmdlDataset *dataset) {
  if (data_mem_buffer == nullptr) {
    MS_LOG(ERROR) << "Data mem buffer is nullptr.";
    return lite::RET_ERROR;
  }
  aclError ret;
  auto free_data_buffer = [this](void *dataMemBuffer) {
    if (!is_run_on_device_) {
      aclrtFree(dataMemBuffer);
    } else {
      aclrtFreeHost(dataMemBuffer);
    }
  };

  if (!is_run_on_device_) {
    ret = aclrtMalloc(data_mem_buffer, buffer_size, ACL_MEM_MALLOC_NORMAL_ONLY);
    if (ret != ACL_ERROR_NONE) {
      MS_LOG(ERROR) << "Malloc device buffer failed , buffer size " << buffer_size;
      return lite::RET_ERROR;
    }
  } else {
    ret = aclrtMallocHost(data_mem_buffer, buffer_size);
    if (ret != ACL_ERROR_NONE) {
      MS_LOG(ERROR) << "Malloc host buffer failed , buffer size " << buffer_size;
      return lite::RET_ERROR;
    }
  }

  auto data_buffer = aclCreateDataBuffer(*data_mem_buffer, buffer_size);
  if (data_buffer == nullptr) {
    MS_LOG(ERROR) << "Create Data Buffer failed";
    free_data_buffer(*data_mem_buffer);
    return lite::RET_ERROR;
  }
  ret = aclmdlAddDatasetBuffer(dataset, data_buffer);
  if (ret != ACL_ERROR_NONE) {
    MS_LOG(ERROR) << "add data buffer failed";
    free_data_buffer(*data_mem_buffer);
    aclDestroyDataBuffer(data_buffer);
    return lite::RET_ERROR;
  }
  return lite::RET_OK;
}

STATUS ModelProcess::InitOutputsBuffer() {
  aclError ret;
  outputs_ = aclmdlCreateDataset();
  if (outputs_ == nullptr) {
    MS_LOG(ERROR) << "Create input dataset failed";
    return lite::RET_ERROR;
  }
  size_t output_size = aclmdlGetNumOutputs(model_desc_);
  MS_LOG(INFO) << "Output_size = " << output_size;
  for (size_t i = 0; i < output_size; ++i) {
    auto buffer_size = aclmdlGetOutputSizeByIndex(model_desc_, i);

    void *data_mem_buffer = nullptr;
    if (CreateDataBuffer(&data_mem_buffer, buffer_size, outputs_) != lite::RET_OK) {
      MS_LOG(ERROR) << "Add output data buffer failed, buffer size " << buffer_size;
      return lite::RET_ERROR;
    }
    aclmdlIODims dims;
    ret = aclmdlGetOutputDims(model_desc_, i, &dims);
    if (ret != ACL_ERROR_NONE) {
      MS_LOG(ERROR) << "Get input shape failed";
      if (!is_run_on_device_) {
        aclrtFree(data_mem_buffer);
      } else {
        aclrtFreeHost(data_mem_buffer);
      }
      return lite::RET_OK;
    }
    aclFormat format = aclmdlGetOutputFormat(model_desc_, i);
    if (format != aclFormat::ACL_FORMAT_NCHW && format != aclFormat::ACL_FORMAT_ND) {
      MS_LOG(WARNING) << "The output format of om should be nchw, but now is " << format;
    }
    aclDataType data_type = aclmdlGetOutputDataType(model_desc_, i);
    std::vector<int64_t> shape(dims.dims, dims.dims + dims.dimCount);
    std::string output_name = aclmdlGetOutputNameByIndex(model_desc_, i);
    if (output_name.empty()) {
      MS_LOG(WARNING) << "Get name of output " << i << " failed.";
    }
    MS_LOG(INFO) << "Name of input " << i << " is " << output_name;
    output_infos_.emplace_back(
      AclTensorInfo{data_mem_buffer, data_mem_buffer, buffer_size, data_type, shape, output_name});
  }
  MS_LOG(INFO) << "Create model output success.";
  return lite::RET_OK;
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
  DestroyInputsDataMem();
  DestroyInputsDataset();
}

void ModelProcess::DestroyOutputsBuffer() {
  for (const auto &item : output_infos_) {
    if (!is_run_on_device_) {
      aclrtFree(item.device_data);
    } else {
      aclrtFreeHost(item.device_data);
    }
  }
  output_infos_.clear();

  if (outputs_ == nullptr) {
    return;
  }
  for (size_t i = 0; i < aclmdlGetDatasetNumBuffers(outputs_); i++) {
    auto dataBuffer = aclmdlGetDatasetBuffer(outputs_, i);
    aclDestroyDataBuffer(dataBuffer);
  }
  aclmdlDestroyDataset(outputs_);
  outputs_ = nullptr;
}

STATUS ModelProcess::UnLoad() {
  auto ret = aclmdlUnload(model_id_);
  if (ret != ACL_ERROR_NONE) {
    MS_LOG(ERROR) << "Unload model failed, ret = " << ret;
    return lite::RET_ERROR;
  }
  if (model_desc_ != nullptr) {
    ret = aclmdlDestroyDesc(model_desc_);
    if (ret != ACL_ERROR_NONE) {
      MS_LOG(ERROR) << "Unload model failed, ret = " << ret;
      return lite::RET_ERROR;
    }
    model_desc_ = nullptr;
  }
  DestroyInputsBuffer();
  DestroyOutputsBuffer();
  MS_LOG(INFO) << "End unload model " << model_id_;
  return lite::RET_OK;
}

size_t ModelProcess::GetDynamicDims(const std::vector<AclTensorInfo> &inputs) {
  size_t max_num = 0;
  for (auto input : inputs) {
    size_t cur_num = std::count(input.dims.begin(), input.dims.end(), -1);
    if (cur_num > max_num) {
      max_num = cur_num;
    }
  }
  return max_num;
}

STATUS ModelProcess::SetBatchSize(const std::vector<mindspore::MSTensor> &inputs) {
  size_t index;
  aclError ret;
  for (size_t i = 0; i < inputs.size(); i++) {
    input_infos_[i].buffer_size = inputs[i].DataSize();
  }
  auto *p = reinterpret_cast<const float *>(inputs[inputs.size() - 1].Data().get());
  if (p == nullptr) {
    MS_LOG(ERROR) << "Pointer is nullptr.";
    return lite::RET_OK;
  }
  auto dynamicBatchSize = p[0];
  ret = aclmdlGetInputIndexByName(model_desc_, ACL_DYNAMIC_TENSOR_NAME, &index);
  if (ret != ACL_ERROR_NONE) {
    MS_LOG(ERROR) << "Get index failed";
    return lite::RET_ERROR;
  }
  ret = aclmdlSetDynamicBatchSize(model_id_, inputs_, index, dynamicBatchSize);
  if (ret != ACL_ERROR_NONE) {
    MS_LOG(ERROR) << "Set dynamic batch size failed, model_id is " << model_id_;
    return lite::RET_ERROR;
  }
  return lite::RET_OK;
}

STATUS ModelProcess::CheckTensorByTensorInfo(const std::vector<mindspore::MSTensor> &tensor,
                                             const std::vector<AclTensorInfo> &tensor_info, size_t dynamic_nums) {
  if (dynamic_nums == 0) {
    for (size_t i = 0; i < tensor_info.size(); ++i) {
      if (tensor[i].Shape() != tensor_info[i].dims) {
        MS_LOG(ERROR) << "Note: input " << i << " shape not match, required " << ShapeToString(tensor_info[i].dims)
                      << ", given " << ShapeToString(tensor[i].Shape());
        return lite::RET_ERROR;
      }
      if (tensor[i].DataType() != TransToDataType(tensor_info[i].data_type)) {
        MS_LOG(ERROR) << "Note: input " << i << " data type not match, required "
                      << static_cast<int>(TransToDataType(tensor_info[i].data_type)) << ", given "
                      << static_cast<int>(tensor[i].DataType());
        return lite::RET_ERROR;
      }
      if (tensor[i].DataSize() != tensor_info[i].buffer_size) {
        MS_LOG(ERROR) << "Input " << i << " data size not match, required size " << tensor_info[i].buffer_size
                      << ", given count " << tensor[i].DataSize();
        return lite::RET_ERROR;
      }
    }
  }
  return lite::RET_OK;
}

STATUS ModelProcess::ProcDynamicShape(const std::vector<mindspore::MSTensor> &inputs, size_t dynamic_nums) {
  if (dynamic_nums == kDynamicBatchSize) {
    if (SetBatchSize(inputs) != lite::RET_OK) {
      MS_LOG(ERROR) << "Failed to convert dynamic batch size";
      return lite::RET_ERROR;
    }
    if (ResetOutputSize() != lite::RET_OK) {
      MS_LOG(ERROR) << "Reset output size failed";
      return lite::RET_ERROR;
    }
  } else if (dynamic_nums == kDynamicImageSize) {
    MS_LOG(ERROR) << "Only dynamic batch size is supported";
    return lite::RET_ERROR;
  }
  return lite::RET_OK;
}

STATUS ModelProcess::CheckAndInitInput(const std::vector<mindspore::MSTensor> &inputs) {
  aclError ret;
  inputs_ = aclmdlCreateDataset();
  size_t dynamic_nums = GetDynamicDims(input_infos_);
  // check inputs
  if (CheckTensorByTensorInfo(inputs, input_infos_, dynamic_nums) != lite::RET_OK) {
    MS_LOG(ERROR) << "Check input tensor failed.";
    return lite::RET_ERROR;
  }
  // copy inputs
  for (size_t i = 0; i < input_infos_.size(); ++i) {
    auto &info = input_infos_[i];
    auto input = inputs[i];
    void *data = input.MutableData();
    void *input_buffer = nullptr;
    if (!is_run_on_device_) {
      info.cur_device_data = info.device_data;
      ret = aclrtMemcpy(info.cur_device_data, info.buffer_size, data, input.DataSize(), ACL_MEMCPY_HOST_TO_DEVICE);
      if (ret != ACL_ERROR_NONE) {
        MS_LOG(ERROR) << "Acl memcpy input " << i << " data to device failed, buffer size " << input.DataSize();
        return lite::RET_ERROR;
      }
      input_buffer = info.cur_device_data;
    } else {
      input_buffer = data;
    }
    auto data_buffer = aclCreateDataBuffer(input_buffer, info.buffer_size);
    if (data_buffer == nullptr) {
      MS_LOG(ERROR) << "Create Data Buffer failed";
      return lite::RET_ERROR;
    }
    ret = aclmdlAddDatasetBuffer(inputs_, data_buffer);
    if (ret != ACL_ERROR_NONE) {
      MS_LOG(ERROR) << "Add data buffer failed";
      aclDestroyDataBuffer(data_buffer);
      return lite::RET_ERROR;
    }
  }
  if (ProcDynamicShape(inputs, dynamic_nums) != lite::RET_OK) {
    MS_LOG(ERROR) << "Proc input dynamic shape failed.";
    return lite::RET_ERROR;
  }
  return lite::RET_OK;
}

STATUS ModelProcess::ResetOutputSize() {
  aclDataType output_type;
  aclError ret;
  size_t output_size = aclmdlGetNumOutputs(model_desc_);
  for (size_t index = 0; index < output_size; index++) {
    size_t dims = 1;
    struct aclmdlIODims output_dims;
    ret = aclmdlGetCurOutputDims(model_desc_, index, &output_dims);
    if (ret != ACL_ERROR_NONE) {
      MS_LOG(ERROR) << "get output dim error.";
      return lite::RET_ERROR;
    }
    for (size_t i = 0; i < output_dims.dimCount; i++) {
      dims *= output_dims.dims[i];
    }
    output_type = aclmdlGetOutputDataType(model_desc_, index);
    output_infos_[index].buffer_size = dims * aclDataTypeSize(output_type);
  }
  return lite::RET_OK;
}

STATUS ModelProcess::SortTensorInfoByName(const std::vector<mindspore::MSTensor> &tensor,
                                          std::vector<AclTensorInfo> *tensor_info) {
  if (tensor_info == nullptr) {
    MS_LOG(ERROR) << "Tensor info is nullptr.";
    return lite::RET_ERROR;
  }
  if (tensor.size() != tensor_info->size()) {
    MS_LOG(ERROR) << "Actual tensor count not match, required count " << tensor_info->size() << ", given count "
                  << tensor.size();
    return lite::RET_ERROR;
  }
  size_t size = tensor.size();
  for (size_t i = 0; i < size; i++) {
    std::string name = tensor[i].Name();
    size_t j;
    for (j = 0; j < size; j++) {
      if (name.find((*tensor_info)[j].name) != std::string::npos) {
        std::swap((*tensor_info)[i], (*tensor_info)[j]);
        break;
      }
    }
    if (j == size) {
      MS_LOG(ERROR) << "Input[" << i << "] " << name << " can't be found in acl om.";
      return lite::RET_ERROR;
    }
  }
  return lite::RET_OK;
}

STATUS ModelProcess::PredictFromHost(const std::vector<mindspore::MSTensor> &inputs,
                                     std::vector<mindspore::MSTensor> *outputs) {
  if (SortTensorInfoByName(inputs, &input_infos_) != lite::RET_OK) {
    MS_LOG(ERROR) << "Sort input tensor info failed.";
    return lite::RET_ERROR;
  }
  STATUS ret = CheckAndInitInput(inputs);
  if (ret != lite::RET_OK) {
    MS_LOG(ERROR) << "Check or init input failed";
    DestroyInputsDataset();
    return ret;  // forward status error
  }
  aclError acl_ret = aclmdlExecute(model_id_, inputs_, outputs_);
  DestroyInputsDataset();
  if (acl_ret != ACL_ERROR_NONE) {
    MS_LOG(ERROR) << "Execute Model Failed, ret = " << acl_ret;
    return lite::RET_ERROR;
  }
  ret = GetOutputs(outputs);
  if (ret != lite::RET_OK) {
    MS_LOG(ERROR) << "Build outputs failed";
    return ret;
  }
  MS_LOG(INFO) << "Execute model success";
  return lite::RET_OK;
}

STATUS ModelProcess::GetOutputs(std::vector<mindspore::MSTensor> *outputs) {
  if (outputs == nullptr) {
    MS_LOG(ERROR) << "Ms tensor output is nullptr.";
    return lite::RET_ERROR;
  }

  if (ConstructTensor(outputs) != lite::RET_OK) {
    MS_LOG(ERROR) << "Construct ms tensor failed.";
    return lite::RET_ERROR;
  }
  return lite::RET_OK;
}

STATUS ModelProcess::ConstructTensor(std::vector<mindspore::MSTensor> *outputs) {
  if (outputs == nullptr) {
    MS_LOG(ERROR) << "Ms tensor output is nullptr.";
    return lite::RET_ERROR;
  }
  if (outputs->size() != output_infos_.size()) {
    MS_LOG(ERROR) << "Actual tensor count not match, required count " << output_infos_.size() << ", given count "
                  << outputs->size();
    return lite::RET_ERROR;
  }
  std::vector<std::string> names;
  std::vector<std::vector<int64_t>> shapes;
  std::vector<enum DataType> data_types;
  std::vector<size_t> mem_sizes;
  if (ConstructTensorDesc(output_infos_, &names, &shapes, &data_types, &mem_sizes) != lite::RET_OK) {
    MS_LOG(ERROR) << "Construct tensor desc failed.";
    return lite::RET_ERROR;
  }
  // set output info and malloc data size
  for (size_t i = 0; i < output_infos_.size(); ++i) {
    std::string lite_output_name = (*outputs)[i].Name();
    if (lite_output_name != names[i]) {
      MS_LOG(INFO) << "Lite output name: " << lite_output_name << "; Om output name: " << names[i];
    }
    (*outputs)[i].SetFormat(Format::NCHW);
    (*outputs)[i].SetDataType(data_types[i]);
    (*outputs)[i].SetShape(shapes[i]);
    (*outputs)[i].MutableData();
    if ((*outputs)[i].DataSize() != mem_sizes[i]) {
      MS_LOG(ERROR) << "Ms tensor size " << (*outputs)[i].DataSize() << " not match acl tensor size " << mem_sizes[i];
      return lite::RET_ERROR;
    }
  }
  aclrtMemcpyKind kind = is_run_on_device_ ? ACL_MEMCPY_HOST_TO_HOST : ACL_MEMCPY_DEVICE_TO_HOST;
  for (size_t i = 0; i < output_infos_.size(); ++i) {
    if (output_infos_[i].cur_device_data == nullptr) {
      // when run on device, cur_device_data is nullptr before first execute
      continue;
    }
    auto ret = aclrtMemcpy((*outputs)[i].MutableData(), (*outputs)[i].DataSize(), output_infos_[i].cur_device_data,
                           output_infos_[i].buffer_size, kind);
    if (ret != ACL_ERROR_NONE) {
      MS_LOG(ERROR) << "Memcpy input " << i << " from " << (is_run_on_device_ ? "host" : "device")
                    << " to host failed, memory size " << output_infos_[i].buffer_size;
      return lite::RET_ERROR;
    }
  }
  return lite::RET_OK;
}
}  // namespace acl
}  // namespace mindspore::kernel
