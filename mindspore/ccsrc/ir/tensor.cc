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

#include "ir/tensor.h"

#include <atomic>
#include <functional>
#include <numeric>
#include <vector>
#include <sstream>
#include <string>
#include <utility>

#include "device/device_address.h"
#include "pipeline/static_analysis/abstract_value.h"

namespace mindspore {
namespace tensor {

using Bool = unsigned char;

static std::string MakeId() {
  // Use atomic to make id generator thread safe.
  static std::atomic<uint64_t> last_id{1};
  return std::to_string(last_id.fetch_add(1, std::memory_order_relaxed));
}

static TypeId TypeIdOf(const TypePtr &data_type, TypeId defaultTypeId) {
  return data_type ? data_type->type_id() : defaultTypeId;
}

static size_t SizeOf(const std::vector<int> &shape) {
  return std::accumulate(shape.begin(), shape.end(), size_t(1), std::multiplies<size_t>());
}

template <typename T>
std::vector<T> CopyData(const std::vector<int> &shape, void *data, TypeId data_type) {
  const size_t count = SizeOf(shape);
  switch (data_type) {
    case kNumberTypeBool: {
      auto buf = static_cast<Bool *>(data);
      return std::vector<T>(buf, buf + count);
    }
    case kNumberTypeUInt8: {
      auto buf = static_cast<uint8_t *>(data);
      return std::vector<T>(buf, buf + count);
    }
    case kNumberTypeInt8: {
      auto buf = static_cast<int8_t *>(data);
      return std::vector<T>(buf, buf + count);
    }
    case kNumberTypeInt16: {
      auto buf = static_cast<int16_t *>(data);
      return std::vector<T>(buf, buf + count);
    }
    case kNumberTypeInt32: {
      auto buf = static_cast<int32_t *>(data);
      return std::vector<T>(buf, buf + count);
    }
    case kNumberTypeInt64: {
      auto buf = static_cast<int64_t *>(data);
      return std::vector<T>(buf, buf + count);
    }
    case kNumberTypeUInt16: {
      auto buf = static_cast<uint16_t *>(data);
      return std::vector<T>(buf, buf + count);
    }
    case kNumberTypeUInt32: {
      auto buf = static_cast<uint32_t *>(data);
      return std::vector<T>(buf, buf + count);
    }
    case kNumberTypeUInt64: {
      auto buf = static_cast<uint64_t *>(data);
      return std::vector<T>(buf, buf + count);
    }
    case kNumberTypeFloat16: {
      auto buf = static_cast<float16 *>(data);
      return std::vector<T>(buf, buf + count);
    }
    case kNumberTypeFloat32: {
      const float *buf = static_cast<float *>(data);
      return std::vector<T>(buf, buf + count);
    }
    case kNumberTypeFloat64: {
      auto buf = static_cast<double *>(data);
      return std::vector<T>(buf, buf + count);
    }
    default:
      break;
  }
  MS_LOG(EXCEPTION) << "Cannot construct Tensor because of unsupported data type: " << data_type << ".";
}

// Convert to bool is not allowed.
template <>
std::vector<Bool> CopyData<Bool>(const std::vector<int> &shape, void *data, TypeId data_type) {
  MS_LOG(EXCEPTION) << "Cannot convert from " << TypeIdLabel(data_type) << " to " << TypeIdLabel(kNumberTypeBool)
                    << ".";
  return {};
}

template <typename T>
std::vector<T> CopyData(const std::vector<int> &shape, void *data, size_t data_len) {
  size_t size = SizeOf(shape);
  if (size * sizeof(T) != data_len) {
    MS_LOG(EXCEPTION) << "Incorrect tensor input data length  " << data_len << ", expect " << size * sizeof(T)
                      << " item size " << sizeof(T);
  }
  auto buf = static_cast<T *>(data);
  return {buf, buf + size};
}

// Tensor data implementation.
template <typename T>
class TensorDataImpl : public TensorData {
 public:
  explicit TensorDataImpl(const std::vector<int> &shape) : shape_(shape), data_(SizeOf(shape)) {}

  TensorDataImpl(const std::vector<int> &shape, void *data, size_t data_len)
      : shape_(shape), data_(CopyData<T>(shape, data, data_len)) {}

  TensorDataImpl(const std::vector<int> &shape, void *data, TypeId data_type)
      : shape_(shape), data_(CopyData<T>(shape, data, data_type)) {}

  template <typename InputIt>
  TensorDataImpl(const std::vector<int> &shape, InputIt first, InputIt last) : shape_(shape), data_(first, last) {}

  template <typename Scalar>
  TensorDataImpl(const std::vector<int> &shape, Scalar scalar) : shape_(shape), data_({static_cast<T>(scalar)}) {}

  ssize_t size() const override { return data_.size(); }

  ssize_t itemsize() const override { return static_cast<ssize_t>(sizeof(T)); }

  ssize_t nbytes() const override { return size() * itemsize(); }

  ssize_t ndim() const override { return static_cast<ssize_t>(shape_.size()); }

  void *data() override {
    static std::vector<T> empty_data(1);
    if (data_.empty()) {
      // Prevent null pointer for empty data.
      return empty_data.data();
    }
    return data_.data();
  }

  bool equals(const TensorData &other) const override {
    auto ptr = dynamic_cast<const TensorDataImpl<T> *>(&other);
    if (ptr) {
      return (ptr == this) || ((shape_ == ptr->shape_) && (data_ == ptr->data_));
    }
    return false;
  }

  std::string ToString() const override {
    std::ostringstream ss;
    ss << '[';
    for (auto value : data_) {
      ss << value << ',';
    }
    ss << ']';
    return ss.str();
  }

 private:
  std::vector<int> shape_;
  std::vector<T> data_;
};

template <typename... Args>
TensorDataPtr MakeTensorData(TypeId data_type, const std::vector<int> &shape, Args... args) {
  switch (data_type) {
    case kNumberTypeBool:
      // std::vector<bool> is a specialization of std::vector,
      // it may use single bit instead of sizeof(bool) bytes,
      // so we use std::vector<Bool> for bool tensors.
      return std::make_shared<TensorDataImpl<Bool>>(shape, args...);
    case kNumberTypeUInt8:
      return std::make_shared<TensorDataImpl<uint8_t>>(shape, args...);
    case kNumberTypeInt8:
      return std::make_shared<TensorDataImpl<int8_t>>(shape, args...);
    case kNumberTypeInt16:
      return std::make_shared<TensorDataImpl<int16_t>>(shape, args...);
    case kNumberTypeInt32:
      return std::make_shared<TensorDataImpl<int32_t>>(shape, args...);
    case kNumberTypeInt64:
      return std::make_shared<TensorDataImpl<int64_t>>(shape, args...);
    case kNumberTypeUInt16:
      return std::make_shared<TensorDataImpl<uint16_t>>(shape, args...);
    case kNumberTypeUInt32:
      return std::make_shared<TensorDataImpl<uint32_t>>(shape, args...);
    case kNumberTypeUInt64:
      return std::make_shared<TensorDataImpl<uint64_t>>(shape, args...);
    case kNumberTypeFloat16:
      return std::make_shared<TensorDataImpl<float16>>(shape, args...);
    case kNumberTypeFloat32:
      return std::make_shared<TensorDataImpl<float>>(shape, args...);
    case kNumberTypeFloat64:
      return std::make_shared<TensorDataImpl<double>>(shape, args...);
    default:
      break;
  }
  MS_LOG(EXCEPTION) << "Cannot construct Tensor because of unsupported data type: " << data_type << ".";
}

Tensor::Tensor(const Tensor &tensor)
    : MetaTensor(tensor),
      init_flag_(tensor.init_flag_),
      data_(tensor.data_),
      dirty_(tensor.dirty_),
      id_(tensor.id_),
      device_address_(tensor.device_address_) {}

Tensor::Tensor(const Tensor &tensor, TypeId data_type)
    : MetaTensor(data_type, tensor.shape_),
      init_flag_(tensor.init_flag_),
      data_(MakeTensorData(data_type, tensor.shape_, tensor.data_->data(), tensor.data_type_)),
      dirty_(tensor.dirty_),
      id_(tensor.id_),
      device_address_(tensor.device_address_) {}

Tensor::Tensor(TypeId data_type, const std::vector<int> &shape, TensorDataPtr data)
    : MetaTensor(data_type, shape), data_(std::move(data)), id_(MakeId()) {}

Tensor::Tensor(TypeId data_type, const std::vector<int> &shape)
    : Tensor(data_type, shape, MakeTensorData(data_type, shape)) {}

Tensor::Tensor(TypeId data_type, const std::vector<int> &shape, void *data, size_t data_len)
    : Tensor(data_type, shape, MakeTensorData(data_type, shape, data, data_len)) {}

Tensor::Tensor(TypeId data_type, const std::vector<int> &shape, void *data, TypeId src_data_type)
    : Tensor(data_type, shape, MakeTensorData(data_type, shape, data, src_data_type)) {}

Tensor::Tensor(const std::vector<int64_t> &input, const TypePtr &data_type)
    : MetaTensor(TypeIdOf(data_type, kNumberTypeInt32), {static_cast<int>(input.size())}),
      data_(MakeTensorData(data_type_, shape_, input.begin(), input.end())),
      id_(MakeId()) {}

Tensor::Tensor(const std::vector<double> &input, const TypePtr &data_type)
    : MetaTensor(TypeIdOf(data_type, kNumberTypeFloat32), {static_cast<int>(input.size())}),
      data_(MakeTensorData(data_type_, shape_, input.begin(), input.end())),
      id_(MakeId()) {}

Tensor::Tensor(int64_t input, const TypePtr &data_type)
    : MetaTensor(TypeIdOf(data_type, kNumberTypeInt32), {}),
      data_(MakeTensorData(data_type_, {}, input)),
      id_(MakeId()) {}

Tensor::Tensor(double input, const TypePtr &data_type)
    : MetaTensor(TypeIdOf(data_type, kNumberTypeFloat32), {}),
      data_(MakeTensorData(data_type_, {}, input)),
      id_(MakeId()) {}

bool Tensor::operator==(const Tensor &tensor) const {
  return (&tensor == this || (MetaTensor::operator==(tensor) && data_ == tensor.data_));
}

bool Tensor::ValueEqual(const Tensor &tensor) const {
  return (&tensor == this || (MetaTensor::operator==(tensor) && data_->equals(*tensor.data_)));
}

abstract::AbstractBasePtr Tensor::ToAbstract() {
  auto tens = shared_from_base<Tensor>();
  auto dtype = tens->Dtype();
  if (!IsSubType(dtype, kNumber)) {
    MS_LOG(EXCEPTION) << "Expect tensor type kNumber but got: " << dtype->ToString() << ".";
  }
  auto tensor_shape = tens->shape();
  auto abs_tensor = std::make_shared<abstract::AbstractTensor>(dtype, tensor_shape);
  abs_tensor->set_value(shared_from_base<Tensor>());
  return abs_tensor;
}

std::string Tensor::GetShapeAndDataTypeInfo() const {
  std::ostringstream buf;
  buf << "Tensor shape:[" << shape() << "]" << this->Dtype()->ToString();
  return buf.str();
}

std::string Tensor::ToString() const {
  const int small_tensor_size = 30;
  std::ostringstream buf;
  buf << "Tensor shape:[" << shape() << "]" << this->Dtype()->ToString();
  // only print small tensor
  if (DataSize() < small_tensor_size) {
    buf << "val:" << data().ToString();
  }
  return buf.str();
}

std::string Tensor::ToStringRepr() const {
  std::ostringstream buf;
  auto type_ptr = this->Dtype();
  MS_EXCEPTION_IF_NULL(type_ptr);
  buf << "Tensor shape:[" << shape() << "]" << type_ptr->ToString();
  buf << "\nval:" << data().ToString();
  return buf.str();
}

void Tensor::data_sync() const {
  if (device_address_ != nullptr) {
    if (!device_address_->SyncDeviceToHost(shape(), static_cast<size_t>(data().nbytes()), data_type(), data_c())) {
      MS_LOG(EXCEPTION) << "SyncDeviceToHost when asnumpy.";
    }
  }
}

TypeId Tensor::set_data_type(const TypeId data_type) {
  if (data_type != data_type_) {
    data_ = MakeTensorData(data_type, shape_, data_->data(), data_type_);
    return MetaTensor::set_data_type(data_type);
  }
  return data_type;
}
}  // namespace tensor

namespace inference {
MSTensor *MSTensor::CreateTensor(TypeId data_type, const std::vector<int> &shape) {
  return new Tensor(data_type, shape);
}

Tensor::Tensor(TypeId data_type, const std::vector<int> &shape) {
  this->tensor_impl_ = std::make_shared<tensor::Tensor>(data_type, shape);
}

Tensor::Tensor(std::shared_ptr<tensor::Tensor> tensor_ptr) { this->tensor_impl_ = std::move(tensor_ptr); }

TypeId Tensor::data_type() const {
  MS_ASSERT(this->tensor_impl_ != nullptr);
  return this->tensor_impl_->data_type();
}

TypeId Tensor::set_data_type(TypeId data_type) {
  MS_ASSERT(this->tensor_impl_ != nullptr);
  return this->tensor_impl_->set_data_type(data_type);
}

std::vector<int> Tensor::shape() const {
  MS_ASSERT(this->tensor_impl_ != nullptr);
  return this->tensor_impl_->shape();
}

size_t Tensor::set_shape(const std::vector<int> &shape) {
  MS_ASSERT(this->tensor_impl_ != nullptr);
  return this->tensor_impl_->set_shape(shape);
}

int Tensor::DimensionSize(size_t index) const {
  MS_ASSERT(this->tensor_impl_ != nullptr);
  return this->tensor_impl_->DimensionSize(index);
}

int Tensor::ElementsNum() const {
  MS_ASSERT(this->tensor_impl_ != nullptr);
  return this->tensor_impl_->ElementsNum();
}

std::size_t Tensor::hash() const {
  MS_ASSERT(this->tensor_impl_ != nullptr);
  return this->tensor_impl_->hash();
}

std::shared_ptr<tensor::Tensor> Tensor::tensor() const {
  MS_ASSERT(this->tensor_impl_ != nullptr);
  return this->tensor_impl_;
}

size_t Tensor::Size() const {
  MS_ASSERT(this->tensor_impl_ != nullptr);
  return this->tensor_impl_->data().nbytes();
}

void *Tensor::MutableData() const {
  MS_ASSERT(this->tensor_impl_ != nullptr);
  return this->tensor_impl_->data_c();
}

}  // namespace inference
}  // namespace mindspore
