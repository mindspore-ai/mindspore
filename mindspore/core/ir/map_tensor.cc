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
#include "abstract/abstract_value.h"
#include "ir/tensor.h"
#include "utils/log_adapter.h"
#include "utils/ms_utils_secure.h"

namespace mindspore {
using tensor::Tensor;
using tensor::TensorPtr;

static ShapeVector ConcatShape(const ShapeVector &a, const ShapeVector &b) {
  ShapeVector result_shape = a;
  result_shape.insert(result_shape.end(), b.begin(), b.end());
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

TensorPtr MapTensor::Get(const TensorPtr &key_tensor, const TensorPtr &default_value) {
  MS_EXCEPTION_IF_NULL(key_tensor);
  MS_EXCEPTION_IF_NULL(default_value);
  // Check input.
  if (key_tensor->shape().size() != 1) {
    MS_LOG(EXCEPTION) << "Invalid key tensor shape: " << tensor::ShapeToString(key_tensor->shape());
  }
  // Result shape = key_tensor.shape + value_shape.
  ShapeVector result_shape = ConcatShape(key_tensor->shape(), value_shape());
  // Make the result tensor.
  TensorPtr result_tensor = std::make_shared<Tensor>(value_dtype(), result_shape);
  // Note: this is the fake implementation that fill result tensor by copy default values.
  const size_t num_of_rows = static_cast<size_t>(result_shape[0]);
  const size_t default_value_bytes = static_cast<size_t>(default_value->data().nbytes());
  const uint8_t *default_value_data = static_cast<const uint8_t *>(default_value->data_c());
  auto data_ptr = static_cast<uint8_t *>(result_tensor->data_c());
  for (size_t i = 0; i < num_of_rows; ++i) {
    auto ret = common::huge_memcpy(data_ptr, default_value_bytes, default_value_data, default_value_bytes);
    if (ret != EOK) {
      MS_LOG(EXCEPTION) << "Copy tensor data failed!";
    }
    data_ptr += default_value_bytes;
  }
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
}

MapTensor::ExportData MapTensor::Export(bool full) {
  MS_LOG(DEBUG) << (full ? "Full" : "Incremental") << " export MapTensor";
  // Note: this is fake implementation.
  ShapeVector key_shape = {1};
  ShapeVector values_shape = ConcatShape(ShapeVector{1}, value_shape());
  auto key_tensor = std::make_shared<Tensor>(key_dtype(), key_shape);
  auto value_tensor = std::make_shared<Tensor>(value_dtype(), values_shape);
  auto status_tensor = std::make_shared<Tensor>(kNumberTypeUInt8, key_shape);
  return {key_tensor, value_tensor, status_tensor};
}
}  // namespace mindspore
