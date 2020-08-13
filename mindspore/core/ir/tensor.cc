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
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <iomanip>
#include <algorithm>
#include <type_traits>
#include <typeinfo>

#include "abstract/abstract_value.h"

namespace mindspore {
namespace tensor {
constexpr auto kEllipsis = "...";
constexpr auto kThreshold = 6;

constexpr auto kThreshold1DFloat = kThreshold * 2;
constexpr auto kThreshold1DInt = kThreshold * 4;
constexpr auto kThreshold1DBool = kThreshold * 2;

static std::string MakeId() {
  // Use atomic to make id generator thread safe.
  static std::atomic<uint64_t> last_id{1};
  return "T" + std::to_string(last_id.fetch_add(1, std::memory_order_relaxed));
}

static TypeId TypeIdOf(const TypePtr &data_type, TypeId defaultTypeId) {
  return data_type ? data_type->type_id() : defaultTypeId;
}

static size_t SizeOf(const std::vector<int> &shape) {
  return std::accumulate(shape.begin(), shape.end(), size_t(1), std::multiplies<size_t>());
}

template <typename T, typename U>
std::unique_ptr<T[]> NewData(const U *input, size_t size) {
  if (input == nullptr || size == 0) {
    return nullptr;
  }
  auto data = std::make_unique<T[]>(size);
  if constexpr (!std::is_same<T, U>::value && (std::is_same<T, float16>::value || std::is_same<U, float16>::value)) {
    // Because float16 do not support implicit cast from/to other types,
    // We can not use std::copy() on array of float16, use a loop here.
    for (size_t i = 0; i < size; ++i) {
      data[i] = static_cast<T>(input[i]);
    }
  } else {
    // otherwise, use std::copy for better performance.
    std::copy(input, input + size, data.get());
  }
  return data;
}

template <typename T, typename Scalar>
std::unique_ptr<T[]> NewData(Scalar scalar) {
  auto data = std::make_unique<T[]>(1);
  data[0] = static_cast<T>(scalar);
  return data;
}

template <typename T>
std::unique_ptr<T[]> CopyData(const std::vector<int> &shape, void *const data, TypeId data_type) {
  const size_t size = SizeOf(shape);
  switch (data_type) {
    case kNumberTypeBool:
    case kNumberTypeUInt8: {
      auto buf = static_cast<uint8_t *>(data);
      return NewData<T>(buf, size);
    }
    case kNumberTypeInt8: {
      auto buf = static_cast<int8_t *>(data);
      return NewData<T>(buf, size);
    }
    case kNumberTypeInt16: {
      auto buf = static_cast<int16_t *>(data);
      return NewData<T>(buf, size);
    }
    case kNumberTypeInt32: {
      auto buf = static_cast<int32_t *>(data);
      return NewData<T>(buf, size);
    }
    case kNumberTypeInt64: {
      auto buf = static_cast<int64_t *>(data);
      return NewData<T>(buf, size);
    }
    case kNumberTypeUInt16: {
      auto buf = static_cast<uint16_t *>(data);
      return NewData<T>(buf, size);
    }
    case kNumberTypeUInt32: {
      auto buf = static_cast<uint32_t *>(data);
      return NewData<T>(buf, size);
    }
    case kNumberTypeUInt64: {
      auto buf = static_cast<uint64_t *>(data);
      return NewData<T>(buf, size);
    }
    case kNumberTypeFloat16: {
      auto buf = static_cast<float16 *>(data);
      return NewData<T>(buf, size);
    }
    case kNumberTypeFloat32: {
      auto buf = static_cast<float *>(data);
      return NewData<T>(buf, size);
    }
    case kNumberTypeFloat64: {
      auto buf = static_cast<double *>(data);
      return NewData<T>(buf, size);
    }
    default:
      break;
  }
  MS_LOG(EXCEPTION) << "Cannot construct Tensor because of unsupported data type: " << data_type << ".";
}

template <typename T>
std::unique_ptr<T[]> CopyData(const std::vector<int> &shape, void *const data, size_t data_len) {
  size_t size = SizeOf(shape);
  if (size * sizeof(T) != data_len) {
    MS_LOG(EXCEPTION) << "Incorrect tensor input data length  " << data_len << ", expect " << size * sizeof(T)
                      << " item size " << sizeof(T);
  }
  auto buf = static_cast<T *>(data);
  return NewData<T>(buf, size);
}

// Tensor data implementation.
template <typename T>
class TensorDataImpl : public TensorData {
 public:
  explicit TensorDataImpl(const std::vector<int> &shape) : ndim_(shape.size()), data_size_(SizeOf(shape)) {}
  ~TensorDataImpl() = default;

  TensorDataImpl(const std::vector<int> &shape, void *data, size_t data_len)
      : ndim_(shape.size()), data_size_(SizeOf(shape)), data_(CopyData<T>(shape, data, data_len)) {}

  TensorDataImpl(const std::vector<int> &shape, void *data, TypeId data_type)
      : ndim_(shape.size()), data_size_(SizeOf(shape)), data_(CopyData<T>(shape, data, data_type)) {}

  template <typename U>
  TensorDataImpl(const std::vector<int> &shape, const U *input, size_t size)
      : ndim_(shape.size()), data_size_(SizeOf(shape)), data_(NewData<T>(input, size)) {}

  template <typename Scalar>
  TensorDataImpl(const std::vector<int> &shape, Scalar scalar)
      : ndim_(shape.size()), data_size_(SizeOf(shape)), data_(NewData<T>(scalar)) {}

  ssize_t size() const override { return static_cast<ssize_t>(data_size_); }

  ssize_t itemsize() const override { return static_cast<ssize_t>(sizeof(T)); }

  ssize_t nbytes() const override { return size() * itemsize(); }

  ssize_t ndim() const override { return static_cast<ssize_t>(ndim_); }

  void *data() override {
    if (data_ == nullptr) {
      // Lazy allocation.
      data_ = std::make_unique<T[]>(data_size_);
    }
    return data_.get();
  }

  bool equals(const TensorData &other) const override {
    auto ptr = dynamic_cast<const TensorDataImpl<T> *>(&other);
    if (ptr == nullptr) {
      return false;
    }
    if (ptr == this) {
      return true;
    }
    if (data_ == nullptr || ptr->data_ == nullptr) {
      return false;
    }
    return (ndim_ == ptr->ndim_) && (data_size_ == ptr->data_size_) &&
           std::equal(data_.get(), data_.get() + data_size_, ptr->data_.get());
  }

  std::string ToString(const TypeId type, const std::vector<int> &shape) const override {
    constexpr auto valid =
      std::is_same<T, Bool>::value || std::is_same<T, uint8_t>::value || std::is_same<T, int8_t>::value ||
      std::is_same<T, int16_t>::value || std::is_same<T, int32_t>::value || std::is_same<T, int64_t>::value ||
      std::is_same<T, uint16_t>::value || std::is_same<T, uint32_t>::value || std::is_same<T, uint64_t>::value ||
      std::is_same<T, float16>::value || std::is_same<T, float>::value || std::is_same<T, double>::value;
    static_assert(valid, "Type is invalid");
    if (data_size_ == 0) {
      return "";
    }
    if (data_ == nullptr) {
      return "<uninitialized>";
    }

    std::ostringstream ss;
    if (data_size_ == 1 && ndim_ == 0) {  // Scalar
      OutputDataString(ss, type, 0, 0, 1);
      return ss.str();
    }
    ssize_t cursor = 0;
    SummaryStringRecursive(ss, type, shape, &cursor, 0);
    return ss.str();
  }

 private:
  void OutputDataString(std::ostringstream &ss, const TypeId type, ssize_t cursor, ssize_t start, ssize_t end) const {
    bool isScalar = ndim_ == 0 && end - start == 1;
    int linefeedThreshold;
    constexpr auto isFloat =
      std::is_same<T, float16>::value || std::is_same<T, float>::value || std::is_same<T, double>::value;
    for (ssize_t i = start; i < end && (cursor + i) < static_cast<ssize_t>(data_size_); i++) {
      const auto value = data_[cursor + i];
      if constexpr (isFloat) {
        if (isScalar) {
          ss << value;
        } else {
          if (std::is_same<T, float16>::value) {
            ss << std::setw(11) << std::setprecision(4) << std::setiosflags(std::ios::scientific | std::ios::right)
               << value;
          } else {
            ss << std::setw(15) << std::setprecision(8) << std::setiosflags(std::ios::scientific | std::ios::right)
               << value;
          }
        }
        linefeedThreshold = kThreshold1DFloat;
      } else if (type == kNumberTypeBool) {
        if (isScalar) {
          ss << (value == 0 ? "False" : "True");
        } else {
          ss << std::setw(5) << std::setiosflags(std::ios::right) << (value == 0 ? "False" : "True");
        }
        linefeedThreshold = kThreshold1DBool;
      } else {
        constexpr auto isSigned = std::is_same<T, int8_t>::value || std::is_same<T, int16_t>::value ||
                                  std::is_same<T, int32_t>::value || std::is_same<T, int64_t>::value;
        if constexpr (isSigned) {
          if (!isScalar && static_cast<int64_t>(value) >= 0) {
            ss << ' ';
          }
        }

        // Set width and indent for different int type.
        //
        //   int8/uint8 width:   3
        //   int16/uint16 width: 5
        //   int32/uint32 width: 10
        //   int64/uint64 width: NOT SET
        if constexpr (std::is_same<T, int8_t>::value) {
          ss << std::setw(3) << std::setiosflags(std::ios::right) << static_cast<int16_t>(value);
        } else if constexpr (std::is_same<T, uint8_t>::value) {
          ss << std::setw(3) << std::setiosflags(std::ios::right) << static_cast<uint16_t>(value);
        } else if constexpr (std::is_same<T, int16_t>::value || std::is_same<T, uint16_t>::value) {
          ss << std::setw(5) << std::setiosflags(std::ios::right) << value;
        } else if constexpr (std::is_same<T, int32_t>::value || std::is_same<T, uint32_t>::value) {
          ss << std::setw(10) << std::setiosflags(std::ios::right) << value;
        } else {
          ss << value;
        }
        linefeedThreshold = kThreshold1DInt;
      }
      if (!isScalar && i != end - 1) {
        ss << ' ';
      }
      if (!isScalar && ndim_ == 1 && (i + 1) % linefeedThreshold == 0) {
        // Add a line feed every {threshold of type} for 1D tensor.
        ss << '\n' << ' ';
      }
    }
  }

  void SummaryStringRecursive(std::ostringstream &ss, const TypeId type, const std::vector<int> &shape, ssize_t *cursor,
                              ssize_t depth) const {
    if (depth >= static_cast<ssize_t>(ndim_)) {
      return;
    }
    ss << '[';
    if (depth == static_cast<ssize_t>(ndim_) - 1) {  // Bottom dimension
      ssize_t num = shape[depth];
      if (num > kThreshold && ndim_ > 1) {
        OutputDataString(ss, type, *cursor, 0, kThreshold / 2);
        ss << ' ' << kEllipsis << ' ';
        OutputDataString(ss, type, *cursor, num - kThreshold / 2, num);
      } else {
        OutputDataString(ss, type, *cursor, 0, num);
      }
      *cursor += num;
    } else {  // Middle dimension
      ssize_t num = shape[depth];
      // Handle the first half.
      for (ssize_t i = 0; i < std::min(static_cast<ssize_t>(kThreshold / 2), num); i++) {
        if (i > 0) {
          ss << '\n';
          ss << std::setw(depth + 1) << ' ';  // Add the indent.
        }
        SummaryStringRecursive(ss, type, shape, cursor, depth + 1);
      }
      // Handle the ignored part.
      if (num > kThreshold) {
        ss << '\n';
        ss << std::setw(depth + 1) << ' ';  // Add the indent.
        ss << kEllipsis;
        // Ignored at this layer.
        ssize_t ignored = shape[depth + 1];
        for (ssize_t i = depth + 2; i < static_cast<ssize_t>(ndim_); i++) {
          ignored *= shape[i];
        }
        // Multiple with ignored layers number.
        ignored *= num - kThreshold;

        *cursor += ignored;
      }
      // Handle the second half.
      if (num > kThreshold / 2) {
        for (ssize_t i = num - kThreshold / 2; i < num; i++) {
          ss << '\n';
          ss << std::setw(depth + 1) << ' ';  // Add the indent.
          SummaryStringRecursive(ss, type, shape, cursor, depth + 1);
        }
      }
    }
    ss << ']';
  }

  size_t ndim_{0};
  size_t data_size_{0};
  std::unique_ptr<T[]> data_;
};

template <typename... Args>
TensorDataPtr MakeTensorData(TypeId data_type, const std::vector<int> &shape, const Args... args) {
  switch (data_type) {
    case kNumberTypeBool:
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
      device_sync_(tensor.device_sync_) {}

Tensor::Tensor(const Tensor &tensor, TypeId data_type)
    : MetaTensor(data_type, tensor.shape_),
      init_flag_(tensor.init_flag_),
      data_(MakeTensorData(data_type, tensor.shape_, tensor.data_->data(), tensor.data_type_)),
      dirty_(tensor.dirty_),
      id_(tensor.id_),
      device_sync_(tensor.device_sync_) {}

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
      data_(MakeTensorData(data_type_, shape_, input.data(), input.size())),
      id_(MakeId()) {}

Tensor::Tensor(const std::vector<double> &input, const TypePtr &data_type)
    : MetaTensor(TypeIdOf(data_type, kNumberTypeFloat32), {static_cast<int>(input.size())}),
      data_(MakeTensorData(data_type_, shape_, input.data(), input.size())),
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
// assgin value to this tensor
Tensor &Tensor::AssignValue(const Tensor &tensor) {
  if (this != &tensor) {
    MetaTensor::operator=(tensor);
    dirty_ = tensor.dirty_;
    device_sync_ = tensor.device_sync_;
    data_ = tensor.data_;
    id_ = tensor.id_;
  }
  return *this;
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
  data_sync();
  buf << "Tensor shape:[" << shape() << "]" << this->Dtype()->ToString();
  // only print small tensor
  if (DataSize() < small_tensor_size) {
    buf << ", value:" << data().ToString(data_type_, shape());
  }
  return buf.str();
}

std::string Tensor::ToStringRepr() const {
  std::ostringstream buf;
  auto type_ptr = this->Dtype();
  MS_EXCEPTION_IF_NULL(type_ptr);
  data_sync();
  buf << "Tensor shape:[" << shape() << "]" << type_ptr->ToString();
  buf << "\nvalue:" << data().ToString(data_type_, shape());
  return buf.str();
}

void Tensor::data_sync() const {
  if (device_sync_ != nullptr) {
    if (!device_sync_->SyncDeviceToHost(shape(), static_cast<size_t>(data().nbytes()), data_type(), data_c())) {
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
}  // namespace mindspore
