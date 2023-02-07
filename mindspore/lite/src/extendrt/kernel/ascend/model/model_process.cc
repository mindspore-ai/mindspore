/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include "extendrt/kernel/ascend/model/model_process.h"
#include <sys/time.h>
#include <utility>
#include <algorithm>
#include <map>
#include "common/log_adapter.h"
#include "src/common/utils.h"
#include "src/common/log_util.h"

namespace mindspore::kernel {
namespace acl {
namespace {
constexpr size_t kBatchSizeNum = 1;
constexpr size_t kImageSizeHwNum = 2;
}  // namespace
static TypeId TransToDataType(aclDataType data_type) {
  static const std::map<aclDataType, enum TypeId> data_type_map = {
    {ACL_FLOAT16, TypeId::kNumberTypeFloat16}, {ACL_FLOAT, TypeId::kNumberTypeFloat32},
    {ACL_DOUBLE, TypeId::kNumberTypeFloat64},  {ACL_INT8, TypeId::kNumberTypeInt8},
    {ACL_INT16, TypeId::kNumberTypeInt16},     {ACL_INT32, TypeId::kNumberTypeInt32},
    {ACL_INT64, TypeId::kNumberTypeInt64},     {ACL_UINT8, TypeId::kNumberTypeUInt8},
    {ACL_UINT16, TypeId::kNumberTypeUInt16},   {ACL_UINT32, TypeId::kNumberTypeUInt32},
    {ACL_UINT64, TypeId::kNumberTypeUInt64},   {ACL_BOOL, TypeId::kNumberTypeBool},
  };
  auto it = data_type_map.find(data_type);
  if (it == data_type_map.end()) {
    return TypeId::kNumberTypeEnd;
  } else {
    return it->second;
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

bool ModelProcess::PreInitModelResource() {
  model_desc_ = aclmdlCreateDesc();
  aclError acl_ret = aclmdlGetDesc(model_desc_, model_id_);
  if (acl_ret != ACL_ERROR_NONE) {
    MS_LOG(ERROR) << "Read model desc failed, ret = " << acl_ret;
    return false;
  }
  if (!InitInputsBuffer()) {
    MS_LOG(ERROR) << "Create input buffer failed.";
    return false;
  }
  if (!InitOutputsBuffer()) {
    MS_LOG(ERROR) << "Create output buffer failed.";
    return false;
  }
  dynamic_shape_options_.batch_size = GetDynamicBatch();
  dynamic_shape_options_.image_size = GetDynamicImage();
  data_input_num_ = input_infos_.size();
  if (IsDynamicShape() && data_input_num_ > 0) {
    data_input_num_ -= 1;
  }

  dynamic_shape_options_.input_format = GetInputFormat();
  dynamic_shape_options_.input_shapes = GetInputShape();

  if (!dyn_shape_proc_.Init(dynamic_shape_options_)) {
    MS_LOG(ERROR) << "Init DynShapeProcess failed.";
    return false;
  }
  return true;
}

std::set<uint64_t> ModelProcess::GetDynamicBatch() {
  if (model_desc_ == nullptr) {
    MS_LOG(ERROR) << " Model desc is nullptr.";
    return std::set<uint64_t>();
  }
  aclmdlBatch dynamic_batch;
  if (aclmdlGetDynamicBatch(model_desc_, &dynamic_batch) != ACL_SUCCESS) {
    MS_LOG(ERROR) << "Failed to get dynamic batch.";
    return std::set<uint64_t>();
  }
  size_t batch_count = dynamic_batch.batchCount;
  if (batch_count > ACL_MAX_BATCH_NUM) {
    MS_LOG(ERROR) << "Real batch count " << batch_count << " is larger than max " << ACL_MAX_BATCH_NUM;
    return std::set<uint64_t>();
  }
  std::set<uint64_t> batch;
  for (size_t i = 0; i < dynamic_batch.batchCount; ++i) {
    batch.insert(dynamic_batch.batch[i]);
  }
  return batch;
}

std::set<std::pair<uint64_t, uint64_t>> ModelProcess::GetDynamicImage() {
  if (model_desc_ == nullptr) {
    MS_LOG(ERROR) << " Model desc is nullptr.";
    return std::set<std::pair<uint64_t, uint64_t>>();
  }
  aclmdlHW dynamic_hw;
  if (aclmdlGetDynamicHW(model_desc_, 0, &dynamic_hw) != ACL_SUCCESS) {
    MS_LOG(ERROR) << "Failed to get dynamic hw.";
    return std::set<std::pair<uint64_t, uint64_t>>();
  }
  size_t hw_count = dynamic_hw.hwCount;
  if (hw_count > ACL_MAX_HW_NUM) {
    MS_LOG(ERROR) << "Real hw count " << hw_count << " is larger than max " << ACL_MAX_HW_NUM;
    return std::set<std::pair<uint64_t, uint64_t>>();
  }
  std::set<std::pair<uint64_t, uint64_t>> image;
  for (size_t i = 0; i < dynamic_hw.hwCount; ++i) {
    image.insert(std::pair<uint64_t, uint64_t>(dynamic_hw.hw[i][0], dynamic_hw.hw[i][1]));
  }
  return image;
}

std::vector<Format> ModelProcess::GetInputFormat() {
  if (model_desc_ == nullptr) {
    MS_LOG(ERROR) << " Model desc is nullptr.";
    return std::vector<Format>();
  }
  std::vector<Format> input_formats;
  static const std::map<aclFormat, enum Format> acl_format_map = {
    {ACL_FORMAT_NCHW, NCHW}, {ACL_FORMAT_NHWC, NHWC}, {ACL_FORMAT_ND, NCHW}};
  for (size_t i = 0; i < data_input_num_; ++i) {
    aclFormat format = aclmdlGetInputFormat(model_desc_, i);
    auto iter = acl_format_map.find(format);
    if (iter != acl_format_map.end()) {
      input_formats.emplace_back(iter->second);
    }
    MS_LOG(DEBUG) << "Format of Input " << i << " is " << static_cast<int32_t>(format);
  }
  return input_formats;
}

std::vector<Format> ModelProcess::GetOutputFormat() {
  if (model_desc_ == nullptr) {
    MS_LOG(ERROR) << " Model desc is nullptr.";
    return std::vector<Format>();
  }
  std::vector<Format> output_formats;
  static const std::map<aclFormat, enum Format> acl_format_map = {
    {ACL_FORMAT_NCHW, NCHW}, {ACL_FORMAT_NHWC, NHWC}, {ACL_FORMAT_ND, NCHW}};
  for (size_t i = 0; i < output_infos_.size(); ++i) {
    aclFormat format = aclmdlGetOutputFormat(model_desc_, i);
    auto iter = acl_format_map.find(format);
    if (iter != acl_format_map.end()) {
      output_formats.emplace_back(iter->second);
    }
    MS_LOG(DEBUG) << "Format of Output " << i << " is " << static_cast<int32_t>(format);
  }
  return output_formats;
}

const std::vector<TypeId> ModelProcess::GetOutputDataType() {
  std::vector<TypeId> data_types;
  for (size_t i = 0; i < output_infos_.size(); ++i) {
    TypeId data_type = TransToDataType(output_infos_[i].data_type);
    data_types.emplace_back(data_type);
  }
  return data_types;
}

const std::vector<ShapeVector> ModelProcess::GetOutputShape() {
  std::vector<ShapeVector> shapes;
  for (size_t i = 0; i < output_infos_.size(); ++i) {
    shapes.emplace_back(output_infos_[i].dims);
  }
  return shapes;
}

const std::vector<ShapeVector> ModelProcess::GetInputShape() {
  std::vector<ShapeVector> shapes;
  for (size_t i = 0; i < data_input_num_; ++i) {
    shapes.push_back(input_infos_[i].dims);
  }
  return shapes;
}

const std::vector<TypeId> ModelProcess::GetInputDataType() {
  std::vector<TypeId> data_types;
  for (size_t i = 0; i < data_input_num_; ++i) {
    TypeId data_type = TransToDataType(input_infos_[i].data_type);
    data_types.emplace_back(data_type);
  }
  return data_types;
}

bool ModelProcess::InitInputsBuffer() {
  aclError ret;
  inputs_ = aclmdlCreateDataset();
  if (inputs_ == nullptr) {
    MS_LOG(ERROR) << "Create input dataset failed";
    return false;
  }
  size_t input_size = aclmdlGetNumInputs(model_desc_);
  MS_LOG(INFO) << "input_size = " << input_size;
  for (size_t i = 0; i < input_size; ++i) {
    aclmdlIODims dims;
    // To get correct dims with static AIPP configured, same result as aclmdlGetInputDims without static AIPP
    ret = aclmdlGetInputDimsV2(model_desc_, i, &dims);
    if (ret != ACL_ERROR_NONE) {
      MS_LOG(ERROR) << "Get input shape failed, ret = " << ret;
      return false;
    }
    auto buffer_size = aclmdlGetInputSizeByIndex(model_desc_, i);
    void *data_mem_buffer = nullptr;
    if (!CreateDataBuffer(&data_mem_buffer, buffer_size, inputs_)) {
      MS_LOG(ERROR) << "Add input data buffer failed, buffer size " << buffer_size;
      return false;
    }
    aclDataType data_type = aclmdlGetInputDataType(model_desc_, i);
    std::vector<int64_t> shape(dims.dims, dims.dims + dims.dimCount);
    std::string input_name = aclmdlGetInputNameByIndex(model_desc_, i);
    if (input_name.empty()) {
      MS_LOG(WARNING) << "Get name of input " << i << " failed.";
    }
    MS_LOG(INFO) << "Name of input " << i << " is " << input_name;
    input_infos_.emplace_back(
      AclTensorInfo{data_mem_buffer, data_mem_buffer, buffer_size, buffer_size, data_type, shape, input_name});
  }
  MS_LOG(INFO) << "Create model inputs success";
  return true;
}

bool ModelProcess::InitOutputsBuffer() {
  aclError ret;
  outputs_ = aclmdlCreateDataset();
  if (outputs_ == nullptr) {
    MS_LOG(ERROR) << "Create output dataset failed";
    return false;
  }
  size_t output_size = aclmdlGetNumOutputs(model_desc_);
  MS_LOG(INFO) << "Output_size = " << output_size;
  for (size_t i = 0; i < output_size; ++i) {
    aclmdlIODims dims;
    ret = aclmdlGetOutputDims(model_desc_, i, &dims);
    if (ret != ACL_ERROR_NONE) {
      MS_LOG(ERROR) << "Get output shape failed";
      return true;
    }
    auto buffer_size = aclmdlGetOutputSizeByIndex(model_desc_, i);
    void *data_mem_buffer = nullptr;
    if (!CreateDataBuffer(&data_mem_buffer, buffer_size, outputs_)) {
      MS_LOG(ERROR) << "Add output data buffer failed, buffer size " << buffer_size;
      return false;
    }
    aclFormat format = aclmdlGetOutputFormat(model_desc_, i);
    MS_LOG(DEBUG) << "The output format of om is " << format;
    aclDataType data_type = aclmdlGetOutputDataType(model_desc_, i);
    std::vector<int64_t> shape(dims.dims, dims.dims + dims.dimCount);
    std::string output_name = aclmdlGetOutputNameByIndex(model_desc_, i);
    if (output_name.empty()) {
      MS_LOG(WARNING) << "Get name of output " << i << " failed.";
    }
    MS_LOG(INFO) << "Name of om output " << i << " is " << output_name << "Buffer size " << buffer_size;
    output_infos_.emplace_back(
      AclTensorInfo{data_mem_buffer, data_mem_buffer, buffer_size, buffer_size, data_type, shape, output_name});
  }
  MS_LOG(INFO) << "Create model output success.";
  return true;
}

bool ModelProcess::CreateDataBuffer(void **data_mem_buffer, size_t buffer_size, aclmdlDataset *dataset) {
  if (data_mem_buffer == nullptr) {
    MS_LOG(ERROR) << "Data mem buffer is nullptr.";
    return false;
  }
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
      return false;
    }
  } else {
    ret = aclrtMallocHost(data_mem_buffer, buffer_size);
    if (ret != ACL_ERROR_NONE) {
      MS_LOG(ERROR) << "Malloc host buffer failed , buffer size " << buffer_size;
      return false;
    }
  }

  auto data_buffer = aclCreateDataBuffer(*data_mem_buffer, buffer_size);
  if (data_buffer == nullptr) {
    MS_LOG(ERROR) << "Create Data Buffer failed";
    free_data_buffer(*data_mem_buffer);
    return false;
  }
  ret = aclmdlAddDatasetBuffer(dataset, data_buffer);
  if (ret != ACL_ERROR_NONE) {
    MS_LOG(ERROR) << "add data buffer failed";
    free_data_buffer(*data_mem_buffer);
    aclDestroyDataBuffer(data_buffer);
    return false;
  }
  return true;
}

void ModelProcess::DestroyInputsBuffer() {
  for (const auto &item : input_infos_) {
    if (!is_run_on_device_) {
      aclrtFree(item.device_data);
    } else {
      aclrtFreeHost(item.device_data);
    }
  }
  input_infos_.clear();

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

bool ModelProcess::Load(const void *om_data, size_t om_data_size) {
  if (loaded_) {
    MS_LOG(INFO) << "Model has been loaded";
    return true;
  }
  MS_LOG(INFO) << "Start load model model.";
  // model load model
  auto acl_ret = aclmdlLoadFromMem(om_data, om_data_size, &model_id_);
  if (acl_ret != ACL_ERROR_NONE) {
    MS_LOG(ERROR) << "Call aclmdlLoadFromMem failed, ret = " << acl_ret;
    return false;
  }
  // model init model resource
  if (!PreInitModelResource()) {
    (void)aclmdlUnload(model_id_);
    MS_LOG(ERROR) << "Pre init model resource failed.";
    return false;
  }
  loaded_ = true;
  MS_LOG(INFO) << "Load model model success.";
  return true;
}

bool ModelProcess::UnLoad() {
  if (!loaded_) {
    MS_LOG(INFO) << "Model has not been loaded or has been unloaded";
    return true;
  }
  loaded_ = false;
  auto ret = aclmdlUnload(model_id_);
  if (ret != ACL_ERROR_NONE) {
    MS_LOG(ERROR) << "Unload model failed, ret = " << ret;
    return false;
  }
  if (model_desc_ != nullptr) {
    ret = aclmdlDestroyDesc(model_desc_);
    if (ret != ACL_ERROR_NONE) {
      MS_LOG(ERROR) << "Unload model failed, ret = " << ret;
      return false;
    }
    model_desc_ = nullptr;
  }
  DestroyInputsBuffer();
  DestroyOutputsBuffer();
  MS_LOG(INFO) << "End unload model " << model_id_;
  return true;
}

bool ModelProcess::IsDynamicShape() { return IsDynamicBatchSize() || IsDynamicImageSize(); }

bool ModelProcess::IsDynamicBatchSize() { return !dynamic_shape_options_.batch_size.empty(); }

bool ModelProcess::IsDynamicImageSize() { return !dynamic_shape_options_.image_size.empty(); }

bool ModelProcess::ResetInputSize(const std::vector<ShapeVector> &new_shapes) {
  for (size_t index = 0; index < new_shapes.size(); index++) {
    std::vector<int64_t> shape = new_shapes[index];
    size_t elem_count = 1;
    for (size_t i = 0; i < shape.size(); i++) {
      if (shape[i] < 0) {
        elem_count = 0;
        break;
      }
      elem_count *= shape[i];
    }
    auto data_type = aclmdlGetInputDataType(model_desc_, index);
    input_infos_[index].dims = shape;
    input_infos_[index].buffer_size = elem_count * aclDataTypeSize(data_type);
  }
  return true;
}

bool ModelProcess::ResetOutputSize() {
  if (model_desc_ == nullptr) {
    MS_LOG(ERROR) << " Model desc is nullptr.";
    return false;
  }
  aclDataType data_type;
  aclError ret;
  size_t output_size = aclmdlGetNumOutputs(model_desc_);
  for (size_t index = 0; index < output_size; index++) {
    struct aclmdlIODims dims;
    ret = aclmdlGetCurOutputDims(model_desc_, index, &dims);
    if (ret != ACL_ERROR_NONE) {
      MS_LOG(ERROR) << "get output dim error.";
      return false;
    }
    std::vector<int64_t> shape(dims.dims, dims.dims + dims.dimCount);
    size_t elem_count = 1;
    for (size_t i = 0; i < dims.dimCount; i++) {
      if (dims.dims[i] < 0) {
        elem_count = 0;
        break;
      }
      elem_count *= dims.dims[i];
    }
    data_type = aclmdlGetOutputDataType(model_desc_, index);
    output_infos_[index].dims = shape;
    output_infos_[index].buffer_size = elem_count * aclDataTypeSize(data_type);
  }
  return true;
}

bool ModelProcess::Resize(const std::vector<ShapeVector> &new_shapes) {
  if (!loaded_) {
    MS_LOG(ERROR) << "Model has not been loaded";
    return false;
  }
  auto input_shapes = GetInputShape();
  if (input_shapes.size() != new_shapes.size()) {
    MS_LOG(ERROR) << "Invalid new input size " << new_shapes.size() << ", expect input size " << input_shapes.size();
    return false;
  }
  bool input_shape_changed = false;
  for (size_t i = 0; i < new_shapes.size(); i++) {
    auto new_shape = new_shapes[i];
    if (std::any_of(new_shape.begin(), new_shape.end(), [](auto dim) { return dim < 0; })) {
      MS_LOG(ERROR) << "New shape of input " << i << " cannot be dynamic, new shape: " << new_shape;
      return false;
    }
    if (input_shapes[i] != new_shape) {
      input_shape_changed = true;
    }
  }
  if (!input_shape_changed) {
    return true;
  }
  if (!IsDynamicShape()) {
    MS_LOG(ERROR) << "Not support dynamic input";
    return false;
  }
  if (model_desc_ == nullptr || inputs_ == nullptr) {
    MS_LOG(ERROR) << "Model is not inited";
    return false;
  }
  size_t index;
  auto ret = aclmdlGetInputIndexByName(model_desc_, ACL_DYNAMIC_TENSOR_NAME, &index);
  if (ret != ACL_ERROR_NONE) {
    MS_LOG(ERROR) << "Get index of dynamic tensor failed";
    return false;
  }
  if (IsDynamicBatchSize()) {
    int32_t batch_size = 0;
    if (!dyn_shape_proc_.CheckAndGetBatchSize(new_shapes, &batch_size)) {
      MS_LOG(ERROR) << "Failed to check batch size";
      return false;
    }
    MS_LOG(INFO) << "Set Batch size(" << batch_size << ") of input " << index << ".";
    ret = aclmdlSetDynamicBatchSize(model_id_, inputs_, index, batch_size);
    if (ret != ACL_ERROR_NONE) {
      MS_LOG(ERROR) << "Set dynamic batch size failed, model_id is " << model_id_;
      return false;
    }
  } else if (IsDynamicImageSize()) {
    int32_t height = 0;
    int32_t width = 0;
    if (!dyn_shape_proc_.CheckAndGetImageSize(new_shapes, &height, &width)) {
      MS_LOG(ERROR) << "Failed to check image size";
      return false;
    }
    MS_LOG(INFO) << "Set Image size(" << height << "," << width << ") of input " << index << ".";
    ret = aclmdlSetDynamicHWSize(model_id_, inputs_, index, height, width);
    if (ret != ACL_ERROR_NONE) {
      MS_LOG(ERROR) << "Set dynamic batch size failed, model_id is " << model_id_;
      return false;
    }
  } else {
    MS_LOG(ERROR) << "Not support dynamic input";
    return false;
  }
  if (!ResetInputSize(new_shapes)) {
    MS_LOG(ERROR) << "Reset input size failed";
    return false;
  }
  if (!ResetOutputSize()) {
    MS_LOG(ERROR) << "Reset output size failed";
    return false;
  }
  return true;
}

bool ModelProcess::CheckInputTensors(const std::vector<KernelTensorPtr> &input_tensors) {
  if (data_input_num_ != input_tensors.size()) {
    MS_LOG(ERROR) << "Expect input size to be " << data_input_num_ << ", but got " << input_tensors.size();
    return false;
  }
  for (size_t i = 0; i < input_tensors.size(); ++i) {
    auto &tensor = input_tensors[i];
    auto &info = input_infos_[i];
    if (tensor->GetShapeVector() != info.dims) {
      MS_LOG(WARNING) << "Note: input " << i << " shape not match, required " << ShapeToString(info.dims) << ", given "
                      << ShapeToString(tensor->GetShapeVector()) << "."
                      << "Please check input shape has been modified by DVPP method.";
    }
    if (tensor->GetDtype() != TransToDataType(info.data_type)) {
      MS_LOG(ERROR) << "Note: input " << i << " data type not match, required "
                    << static_cast<int>(TransToDataType(info.data_type)) << ", given "
                    << static_cast<int>(tensor->GetDtype());
      return false;
    }
    auto device_data = tensor->GetData();
    auto host_data = tensor->GetHostData();
    if (device_data != nullptr && device_data->addr != nullptr) {
      if (device_data->size != info.buffer_size) {
        MS_LOG(ERROR) << "Input " << i << " data size not match, required size " << info.buffer_size << ", given count "
                      << device_data->size;
        return false;
      }
    } else if (host_data != nullptr && host_data->addr != nullptr) {
      if (host_data->size != info.buffer_size) {
        MS_LOG(ERROR) << "Input " << i << " data size not match, required size " << info.buffer_size << ", given count "
                      << host_data->size;
        return false;
      }
    } else {
      MS_LOG(ERROR) << "Failed to get data from input " << i;
      return false;
    }
  }
  return true;
}

bool ModelProcess::CheckOutputTensors(const std::vector<KernelTensorPtr> &outputs) {
  if (outputs.size() != output_infos_.size()) {
    MS_LOG(ERROR) << "Actual tensor count not match, required count " << output_infos_.size() << ", given count "
                  << outputs.size();
    return false;
  }
  for (size_t i = 0; i < outputs.size(); ++i) {
    auto &tensor = outputs[i];
    auto &info = output_infos_[i];
    if (tensor->GetShapeVector() != info.dims) {
      MS_LOG(WARNING) << "Note: output " << i << " shape not match, required " << ShapeToString(info.dims) << ", given "
                      << ShapeToString(tensor->GetShapeVector()) << "."
                      << "Please check output shape.";
    }
    if (tensor->GetDtype() != TransToDataType(info.data_type)) {
      MS_LOG(ERROR) << "Note: output " << i << " data type not match, required "
                    << static_cast<int>(TransToDataType(info.data_type)) << ", given "
                    << static_cast<int>(tensor->GetDtype());
      return false;
    }
    auto device_data = tensor->GetData();
    auto host_data = tensor->GetHostData();
    if (device_data != nullptr && device_data->addr != nullptr) {
      if (device_data->size != info.buffer_size) {
        MS_LOG(ERROR) << "Output " << i << " device data size not match, required size " << info.buffer_size
                      << ", given count " << tensor->GetData()->size;
        return false;
      }
    } else if (host_data != nullptr && host_data->addr != nullptr) {
      if (host_data->size != info.buffer_size) {
        MS_LOG(ERROR) << "Output " << i << " host data size not match, required size " << info.buffer_size
                      << ", given count " << tensor->GetData()->size;
        return false;
      }
    } else {
      MS_LOG(ERROR) << "Failed to get data from output " << i;
      return false;
    }
  }
  return true;
}

bool ModelProcess::CheckAndInitInput(const std::vector<KernelTensorPtr> &inputs) {
  // check inputs
  if (!CheckInputTensors(inputs)) {
    MS_LOG(ERROR) << "Check input tensor failed.";
    return false;
  }
  aclError ret;
  // copy inputs
  for (size_t i = 0; i < inputs.size(); ++i) {
    auto &info = input_infos_[i];
    auto input = inputs[i];
    void *input_buffer = nullptr;
    auto device_data = input->GetData();
    auto host_data = input->GetHostData();
    if (device_data && device_data->addr) {
      input_buffer = device_data->addr;
    } else {
      auto data = host_data->addr;
      auto size = host_data->size;
      if (!is_run_on_device_) {
        ret = aclrtMemcpy(info.device_data, info.buffer_size, data, size, ACL_MEMCPY_HOST_TO_DEVICE);
        if (ret != ACL_ERROR_NONE) {
          MS_LOG(ERROR) << "Acl memcpy input " << i << " data to device failed, src input size: " << size
                        << ", dst device buffer size: " << info.buffer_size;
          return false;
        }
        input_buffer = info.device_data;
      } else {
        input_buffer = data;
      }
    }
    auto data_buffer = aclmdlGetDatasetBuffer(inputs_, i);
    if (data_buffer == nullptr) {
      MS_LOG(ERROR) << "Failed to get dataset buffer of input " << i;
      return false;
    }
    ret = aclUpdateDataBuffer(data_buffer, input_buffer, info.buffer_size);
    if (ret != ACL_ERROR_NONE) {
      MS_LOG(ERROR) << "Failed to update Data Buffer of input " << i << ", buffer size: " << info.buffer_size
                    << ", input shape: " << input->GetShapeVector();
      return false;
    }
  }
  return true;
}

bool ModelProcess::CheckAndInitOutput(const std::vector<KernelTensorPtr> &outputs) {
  // check outputs
  if (!CheckOutputTensors(outputs)) {
    MS_LOG(ERROR) << "Check output tensor failed.";
    return false;
  }
  aclError ret;
  // copy inputs
  for (size_t i = 0; i < outputs.size(); ++i) {
    auto &info = output_infos_[i];
    auto output = outputs[i];
    void *output_buffer = nullptr;
    auto device_data = output->GetData();
    auto host_data = output->GetHostData();
    if (device_data && device_data->addr) {
      output_buffer = device_data->addr;
    } else if (host_data && host_data->addr && is_run_on_device_) {
      output_buffer = host_data->addr;
    } else {
      output_buffer = info.device_data;
    }
    auto data_buffer = aclmdlGetDatasetBuffer(outputs_, i);
    if (data_buffer == nullptr) {
      MS_LOG(ERROR) << "Failed to get dataset buffer of output " << i;
      return false;
    }
    ret = aclUpdateDataBuffer(data_buffer, output_buffer, info.buffer_size);
    if (ret != ACL_ERROR_NONE) {
      MS_LOG(ERROR) << "Failed to update Data Buffer of output " << i << ", buffer size: " << info.buffer_size
                    << ", output shape: " << output->GetShapeVector();
      return false;
    }
  }
  return true;
}

bool ModelProcess::PredictFromHost(const std::vector<KernelTensorPtr> &inputs,
                                   const std::vector<KernelTensorPtr> &outputs) {
  if (!loaded_) {
    MS_LOG(ERROR) << "Model has not been loaded";
    return false;
  }
  if (!CheckAndInitInput(inputs)) {
    MS_LOG(ERROR) << "Check or init input failed";
    return false;
  }
  if (!CheckAndInitOutput(outputs)) {
    MS_LOG(ERROR) << "Check output tensor failed";
    return false;
  }

  aclError acl_ret;
  auto env = std::getenv("GLOG_v");
  if (env != nullptr && env[0] == '1') {
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
  } else {
    acl_ret = aclmdlExecute(model_id_, inputs_, outputs_);
  }
  if (acl_ret != ACL_ERROR_NONE) {
    MS_LOG(ERROR) << "Execute Model Failed, ret = " << acl_ret;
    return false;
  }
  if (!GetOutputs(outputs)) {
    MS_LOG(ERROR) << "Build outputs failed";
    return false;
  }
  MS_LOG(INFO) << "Execute model success";
  return true;
}

bool ModelProcess::GetOutputs(const std::vector<KernelTensorPtr> &outputs) {
  aclrtMemcpyKind kind = is_run_on_device_ ? ACL_MEMCPY_HOST_TO_HOST : ACL_MEMCPY_DEVICE_TO_HOST;
  for (size_t i = 0; i < output_infos_.size(); ++i) {
    auto &output = outputs[i];
    auto &output_info = output_infos_[i];
    if (output_info.cur_device_data == nullptr) {
      MS_LOG(WARNING) << "Output device add is nullptr.";
      continue;
    }
    auto host_data = output->GetHostData();
    if (host_data && host_data->addr && !is_run_on_device_) {
      if (host_data->size != output_info.buffer_size) {
        MS_LOG(ERROR) << "Specified output host data size " << host_data->size << " != execute output data size "
                      << output_info.buffer_size << ", output shape: " << output_info.dims;
        return false;
      }
      auto ret =
        aclrtMemcpy(host_data->addr, host_data->size, output_info.cur_device_data, output_info.buffer_size, kind);
      if (ret != ACL_ERROR_NONE) {
        MS_LOG(ERROR) << "Memcpy input " << i << " from " << (is_run_on_device_ ? "host" : "device")
                      << " to host failed, memory size " << output_info.buffer_size;
        return false;
      }
    }
  }
  return true;
}
}  // namespace acl
}  // namespace mindspore::kernel
