/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "ir/map_tensor.h"
#include <utility>
#include <vector>
#include <algorithm>
#include "abstract/abstract_value.h"
#include "ir/tensor.h"
#include "utils/log_adapter.h"
#include "utils/ms_utils_secure.h"
#include "runtime/device/hash_table.h"

namespace mindspore {
using device::HashTable;
using DataLenPair = std::pair<void *, size_t>;
namespace tensor {
using tensor::Tensor;
using tensor::TensorPtr;

static ShapeVector ConcatShape(const ShapeVector &a, const ShapeVector &b) {
  ShapeVector result_shape = a;
  result_shape.insert(result_shape.end(), b.cbegin(), b.cend());
  return result_shape;
}

std::size_t MapTensor::hash() const { return static_cast<std::size_t>(tid()); }

bool MapTensor::operator==(const MapTensor &other) const { return this == &other; }

abstract::AbstractBasePtr MapTensor::ToAbstract() {
  if (param_info_ != nullptr) {
    // For parameter, a broaden abstract is created with ref_key set.
    ValuePtr ref_key = std::make_shared<RefKey>(param_info_->name());
    return std::make_shared<abstract::AbstractMapTensor>(shared_from_base<MapTensor>(), ref_key);
  } else {
    // For value, an abstract is created with value set.
    return std::make_shared<abstract::AbstractMapTensor>(shared_from_base<MapTensor>());
  }
}

std::string MapTensor::ToString() const {
  auto key_dtype = KeyDtype();
  auto value_dtype = ValueDtype();
  return "MapTensor(key_dtype=" + (key_dtype == nullptr ? "<null>" : key_dtype->ToString()) +
         ", value_dtype=" + (value_dtype == nullptr ? "<null>" : value_dtype->ToString()) +
         ", value_shape=" + tensor::ShapeToString(shape_) +
         ", default_value=" + (default_value_ == nullptr ? "<null>" : default_value_->ToString()) + ")";
}

TensorPtr MapTensor::Get(const TensorPtr &key_tensor) {
  MS_EXCEPTION_IF_NULL(key_tensor);
  // Check input.
  if (key_tensor->shape().size() != 1) {
    MS_LOG(EXCEPTION) << "Invalid key tensor shape: " << tensor::ShapeToString(key_tensor->shape());
  }
  // Result shape = key_tensor.shape + value_shape.
  ShapeVector result_shape = ConcatShape(key_tensor->shape(), value_shape());
  // Make the result tensor.
  TensorPtr result_tensor = std::make_shared<Tensor>(value_dtype(), result_shape);
  // Note: this is the fake implementation that fill result tensor with zeros.
  size_t nbytes = static_cast<size_t>(result_tensor->data().nbytes());
  auto data_ptr = static_cast<uint8_t *>(result_tensor->data_c());
  (void)std::fill(data_ptr, data_ptr + nbytes, 0);
  return result_tensor;
}

void MapTensor::Put(const TensorPtr &key_tensor, const TensorPtr &value_tensor) {
  MS_EXCEPTION_IF_NULL(key_tensor);
  MS_EXCEPTION_IF_NULL(value_tensor);
}

void MapTensor::Erase(const TensorPtr &key_tensor) { MS_EXCEPTION_IF_NULL(key_tensor); }

void MapTensor::Update(const MapTensor::ExportData &data) {
  MS_EXCEPTION_IF_NULL(data.key_tensor);
  MS_EXCEPTION_IF_NULL(data.value_tensor);
  MS_EXCEPTION_IF_NULL(data.status_tensor);
  key_tensor_ = data.key_tensor;
  value_tensor_ = data.value_tensor;
  status_tensor_ = data.status_tensor;
}

MapTensor::ExportData MapTensor::ExportDataFromDevice(DeviceSyncPtr device_sync) {
  auto user_data = device_sync->user_data();
  MS_EXCEPTION_IF_NULL(user_data);
  if (key_dtype() == TypeId::kNumberTypeInt32 && value_dtype() == TypeId::kNumberTypeFloat32) {
    const auto &user_data_data = user_data->get<HashTable<int, float>>(kUserDataData);
    MS_EXCEPTION_IF_NULL(user_data_data);
    size_t data_size = user_data_data->size();
    // key length
    auto key_length = data_size * abstract::TypeIdSize(key_dtype());
    // value length
    std::vector<size_t> value_shape_tmp{data_size};
    (void)std::transform(value_shape().cbegin(), value_shape().cend(), std::back_inserter(value_shape_tmp), IntToSize);
    auto value_length = abstract::ShapeSize(value_shape_tmp) * abstract::TypeIdSize(value_dtype());
    // status length：status shape is same as the shape of key
    auto status_length = data_size * abstract::TypeIdSize(kNumberTypeInt);

    ShapeVector new_shape{SizeToInt(data_size)};
    key_tensor()->set_shape(new_shape);
    status_tensor()->set_shape(new_shape);
    (void)std::copy(value_shape().cbegin(), value_shape().cend(), std::back_inserter(new_shape));
    value_tensor()->set_shape(new_shape);
    status_tensor()->set_data_type(kNumberTypeInt);
    DataLenPair keys_pair = {key_tensor()->data_c(), key_length};
    DataLenPair values_pair = {value_tensor()->data_c(), value_length};
    DataLenPair status_pair = {status_tensor()->data_c(), status_length};
    user_data_data->Export(keys_pair, values_pair, status_pair);
    return {key_tensor(), value_tensor(), status_tensor()};
  }
  MS_LOG(EXCEPTION) << "Invalid Map Tensor:" << ToString();
}

MapTensor::ExportData MapTensor::Export(bool full) {
  MS_LOG(DEBUG) << (full ? "Full" : "Incremental") << " export MapTensor";

  // Check device
  DeviceSyncPtr device_sync = device_address();
  if (device_sync != nullptr) {
    return ExportDataFromDevice(device_sync);
  }
  // Note: this is fake implementation.
  ShapeVector key_shape = {1};
  ShapeVector values_shape = ConcatShape(ShapeVector{1}, value_shape());
  auto key_tensor = std::make_shared<Tensor>(key_dtype(), key_shape);
  auto value_tensor = std::make_shared<Tensor>(value_dtype(), values_shape);
  auto status_tensor = std::make_shared<Tensor>(kNumberTypeInt, key_shape);
  return {key_tensor, value_tensor, status_tensor};
}
}  // namespace tensor
}  // namespace mindspore
