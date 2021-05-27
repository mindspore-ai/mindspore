/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#include "ir/meta_tensor.h"

#include <functional>
#include <numeric>
#include <vector>
#include <sstream>
#include <string>

namespace mindspore {
namespace tensor {
// MetaTensor has default type_id_ which is TypeId::kTypeUnknown.
MetaTensor::MetaTensor() : data_type_(TypeId::kTypeUnknown) {}

MetaTensor::MetaTensor(const TypeId data_type, const ShapeVector &shape) : data_type_(data_type), shape_(shape) {}

MetaTensor::MetaTensor(const TypePtr &type_ptr, const ShapeVector &shape) {
  TypeId data_type = TypeId::kTypeUnknown;
  if (type_ptr != nullptr) {
    data_type = type_ptr->type_id();
  }
  data_type_ = data_type;
  shape_ = shape;
}

MetaTensor::MetaTensor(const MetaTensor &meta_tensor)
    : Value(meta_tensor), data_type_(meta_tensor.data_type()), shape_(meta_tensor.shape()) {}

MetaTensor &MetaTensor::operator=(const MetaTensor &meta_tensor) {
  if (&meta_tensor == this) {
    return *this;
  }

  data_type_ = meta_tensor.data_type();
  shape_ = meta_tensor.shape();
  device_info_ = meta_tensor.device_info();

  return *this;
}

bool MetaTensor::operator==(const MetaTensor &meta_tensor) const {
  return data_type_ == meta_tensor.data_type() && shape_ == meta_tensor.shape();
}

// Get the size of a given dimension by its index number.
// The given index number should be in [0, shape_.size()).
// param index Dimension index number.
// return The size of the dimension if succeed, or -1 if failed.
int64_t MetaTensor::DimensionSize(const size_t index) const {
  int64_t dim_size = -1;
  if (index < shape_.size()) {
    dim_size = shape_[index];
  } else {
    MS_LOG(ERROR) << "Dimension index is wrong: " << index;
  }
  return dim_size;
}

int MetaTensor::ElementsNum() const { return std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies<int>()); }

TypePtr MetaTensor::SetDtype(const TypePtr type_ptr) {
  if (type_ptr == nullptr) {
    MS_LOG(ERROR) << "Dtype to be set is nullptr.";
    return nullptr;
  }
  (void)set_data_type(type_ptr->type_id());
  return type_ptr;
}

void MetaTensor::SetDeviceInfo(const std::string &format, const TypePtr &data_type, const std::string &host_format) {
  DeviceInfo info(format, data_type, host_format);
  set_device_info(info);
}

std::string MetaTensor::ToString() const {
  std::ostringstream buf;
  buf << "MetaTensor(shape=[" << shape() << "]";
  if (is_parameter_) {
    buf << ", name=" << param_info_->name();
  }
  buf << ")";
  return buf.str();
}

std::string MetaTensor::DumpText() const {
  std::ostringstream oss;
  oss << type_name() << "(" << SizeToInt(data_type_) << ")[";
  for (size_t i = 0; i < shape_.size(); ++i) {
    oss << (i > 0 ? ", " : "") << shape_[i];
  }
  oss << "]";
  return oss.str();
}
}  // namespace tensor
}  // namespace mindspore
