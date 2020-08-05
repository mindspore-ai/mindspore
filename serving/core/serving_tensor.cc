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

#include "core/serving_tensor.h"
#include <vector>
#include <unordered_map>
#include <string>
#include <algorithm>
#include "include/infer_log.h"

using std::string;
using std::unordered_map;
using std::vector;

namespace mindspore {
namespace serving {

using inference::DataType;
using inference::InferTensorBase;

const size_t kMaxShapeElementCount = INT32_MAX;
const size_t kMaxDataBufferSize = UINT32_MAX;

ServingTensor::ServingTensor(ms_serving::Tensor &other) : tensor_(other) {}

ServingTensor::~ServingTensor() {}

DataType ServingTensor::data_type() const {
  const std::unordered_map<ms_serving::DataType, inference::DataType> type2id_map{
    {ms_serving::MS_UNKNOWN, inference::kMSI_Unknown}, {ms_serving::MS_BOOL, inference::kMSI_Bool},
    {ms_serving::MS_INT8, inference::kMSI_Int8},       {ms_serving::MS_UINT8, inference::kMSI_Uint8},
    {ms_serving::MS_INT16, inference::kMSI_Int16},     {ms_serving::MS_UINT16, inference::kMSI_Uint16},
    {ms_serving::MS_INT32, inference::kMSI_Int32},     {ms_serving::MS_UINT32, inference::kMSI_Uint32},
    {ms_serving::MS_INT64, inference::kMSI_Int64},     {ms_serving::MS_UINT64, inference::kMSI_Uint64},
    {ms_serving::MS_FLOAT16, inference::kMSI_Float16}, {ms_serving::MS_FLOAT32, inference::kMSI_Float32},
    {ms_serving::MS_FLOAT64, inference::kMSI_Float64},
  };
  auto it = type2id_map.find(tensor_.tensor_type());
  if (it == type2id_map.end()) {
    MSI_LOG_WARNING << "failed to get data type, undefined data type " << tensor_.tensor_type();
    return inference::kMSI_Unknown;
  } else {
    return it->second;
  }
}

void ServingTensor::set_data_type(DataType data_type) {
  const std::unordered_map<inference::DataType, ms_serving::DataType> id2type_map{
    {inference::kMSI_Unknown, ms_serving::MS_UNKNOWN}, {inference::kMSI_Bool, ms_serving::MS_BOOL},
    {inference::kMSI_Float64, ms_serving::MS_FLOAT64}, {inference::kMSI_Int8, ms_serving::MS_INT8},
    {inference::kMSI_Uint8, ms_serving::MS_UINT8},     {inference::kMSI_Int16, ms_serving::MS_INT16},
    {inference::kMSI_Uint16, ms_serving::MS_UINT16},   {inference::kMSI_Int32, ms_serving::MS_INT32},
    {inference::kMSI_Uint32, ms_serving::MS_UINT32},   {inference::kMSI_Int64, ms_serving::MS_INT64},
    {inference::kMSI_Uint64, ms_serving::MS_UINT64},   {inference::kMSI_Float16, ms_serving::MS_FLOAT16},
    {inference::kMSI_Float32, ms_serving::MS_FLOAT32},
  };
  auto it = id2type_map.find(data_type);
  if (it == id2type_map.end()) {
    MSI_LOG_WARNING << "failed to set data type, undefined data type " << data_type;
    tensor_.set_tensor_type(ms_serving::MS_UNKNOWN);
  } else {
    tensor_.set_tensor_type(it->second);
  }
}

std::vector<int64_t> ServingTensor::shape() const {
  std::vector<int64_t> result;
  auto dims = tensor_.tensor_shape().dims();
  std::transform(dims.begin(), dims.end(), std::back_inserter(result), [](const int64_t dim) { return dim; });
  return result;
}

void ServingTensor::set_shape(const std::vector<int64_t> &shape) {
  auto tensor_shape = tensor_.mutable_tensor_shape();
  tensor_shape->Clear();
  size_t element_count = 1;
  for (auto dim : shape) {
    if (dim <= 0 || element_count > kMaxShapeElementCount / dim) {
      MSI_LOG_ERROR << "failed to set shape, invalid dim num " << dim;
      tensor_shape->Clear();
      return;
    }
    element_count *= dim;
    tensor_shape->add_dims(dim);
  }
}

bool ServingTensor::resize_data(size_t data_len) {
  string *buffer = tensor_.mutable_data();
  if (buffer == nullptr) {
    MSI_LOG_ERROR << "invalid buffer data";
    return false;
  }
  buffer->resize(data_len);
  return true;
}

size_t ServingTensor::data_size() const { return tensor_.data().size(); }

void *ServingTensor::mutable_data() { return const_cast<char *>(tensor_.mutable_data()->data()); }

const void *ServingTensor::data() const { return tensor_.data().data(); }

ServingRequest::ServingRequest(const ms_serving::PredictRequest &request) : request_(request) {
  auto &data = request_.data();
  std::transform(data.begin(), data.end(), std::back_inserter(cache_),
                 [](const ms_serving::Tensor &item) { return ServingTensor(const_cast<ms_serving::Tensor &>(item)); });
}

size_t ServingRequest::size() const { return cache_.size(); }

const InferTensorBase *ServingRequest::operator[](size_t index) const {
  if (index >= cache_.size()) {
    MSI_LOG_ERROR << "visit invalid index " << index << " total size " << cache_.size();
    return nullptr;
  }
  return &(cache_[index]);
}

ServingImages::ServingImages(const ms_serving::Images &images) : images_(images) {}

size_t ServingImages::batch_size() const { return images_.images_size(); }

bool ServingImages::get(size_t index, const void *&pic_buffer, uint32_t &pic_size) const {
  if (index >= static_cast<size_t>(images_.images_size())) {
    MSI_LOG_ERROR << "visit invalid index " << index << " total size " << images_.images_size();
    return false;
  }
  pic_buffer = images_.images(index).data();
  pic_size = images_.images(index).size();
  return true;
}

size_t ServingImages::input_index() const { return static_cast<size_t>(images_.input_index()); }

size_t ServingReply::size() const { return cache_.size(); }

InferTensorBase *ServingReply::operator[](size_t index) {
  if (index >= cache_.size()) {
    MSI_LOG_ERROR << "visit invalid index " << index << " total size " << cache_.size();
    return nullptr;
  }
  return &(cache_[index]);
}

const InferTensorBase *ServingReply::operator[](size_t index) const {
  if (index >= cache_.size()) {
    MSI_LOG_ERROR << "visit invalid index " << index << " total size " << cache_.size();
    return nullptr;
  }
  return &(cache_[index]);
}

InferTensorBase *ServingReply::add() {
  auto new_item = reply_.add_result();
  if (new_item == nullptr) {
    MSI_LOG_ERROR << "add new item failed, current total size " << cache_.size();
    return nullptr;
  }
  cache_.push_back(ServingTensor(*new_item));
  return &(cache_.back());
}

void ServingReply::clear() { reply_.mutable_result()->Clear(); }

ServingImagesRequest::ServingImagesRequest(const ms_serving::PredictRequest &request) : request_(request) {
  auto &images_inputs = request_.images();
  std::transform(images_inputs.begin(), images_inputs.end(), std::back_inserter(cache_),
                 [](const ms_serving::Images &item) { return ServingImages(const_cast<ms_serving::Images &>(item)); });
}

size_t ServingImagesRequest::size() const { return cache_.size(); }

const inference::InferImagesBase *ServingImagesRequest::operator[](size_t index) const {
  if (index >= cache_.size()) {
    MSI_LOG_ERROR << "visit invalid index " << index << " total size " << cache_.size();
    return nullptr;
  }
  return &(cache_[index]);
}

}  // namespace serving
}  // namespace mindspore
