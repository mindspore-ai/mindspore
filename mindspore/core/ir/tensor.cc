/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include <cstdint>
#include <iomanip>
#include <functional>
#include <memory>
#include <utility>
#include <algorithm>
#include <type_traits>
#include <map>
#include <vector>
#include "mindapi/base/type_id.h"
#include "abstract/utils.h"
#include "abstract/abstract_value.h"
#include "base/complex_storage.h"
#include "utils/log_adapter.h"
#include "mindspore/ccsrc/include/common/utils/convert_utils.h"
#include "utils/ms_utils_secure.h"
#include "utils/shape_utils.h"
#include "utils/ordered_set.h"

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

std::string ShapeToString(const ShapeVector &shape) {
  std::string str = "[";
  const size_t count = shape.size();
  for (size_t i = 0; i < count; ++i) {
    if (i > 0) {
      str.append(", ");
    }
    str.append(std::to_string(shape[i]));
  }
  return str.append("]");
}

inline static void CopyTensorData(const TensorDataPtr &dest, const TensorDataPtr &src) {
  auto dest_bytes = dest->nbytes();
  auto src_bytes = src->nbytes();
  auto err = common::huge_memcpy(static_cast<uint8_t *>(dest->data()), dest_bytes,
                                 static_cast<const uint8_t *>(src->const_data()), src_bytes);
  if (err != EOK) {
    MS_LOG(EXCEPTION) << "Copy tensor data failed! bytes: " << dest_bytes << "/" << src_bytes << ".";
  }
}

template <typename T, typename U>
std::unique_ptr<T[]> NewData(const U *input, size_t size) {
  if (input == nullptr || size == 0) {
    return nullptr;
  }
  if (size > INT32_MAX) {
    MS_LOG(WARNING) << "Try to alloca a large memory, size is:" << size * sizeof(T);
  }

  auto data = std::make_unique<T[]>(size);
  if constexpr (!std::is_same<T, U>::value &&
                (std::is_same<T, float16>::value || std::is_same<U, float16>::value ||
                 std::is_same<T, ComplexStorage<float>>::value || std::is_same<U, ComplexStorage<float>>::value ||
                 std::is_same<T, ComplexStorage<double>>::value || std::is_same<U, ComplexStorage<double>>::value)) {
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
std::unique_ptr<T[]> CopyData(const ShapeVector &shape, void *const data, TypeId data_type) {
  const size_t size = SizeOf(shape);
  switch (data_type) {
    case kNumberTypeBool: {
      auto buf = static_cast<bool *>(data);
      return NewData<T>(buf, size);
    }
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
    case kNumberTypeComplex64: {
      auto buf = static_cast<ComplexStorage<float> *>(data);
      return NewData<T>(buf, size);
    }
    case kNumberTypeComplex128: {
      auto buf = static_cast<ComplexStorage<double> *>(data);
      return NewData<T>(buf, size);
    }
    case kObjectTypeString: {
      auto buf = static_cast<uint8_t *>(data);
      return NewData<T>(buf, size);
    }
    default:
      break;
  }
  MS_LOG(EXCEPTION) << "Cannot construct Tensor because of unsupported data type: " << data_type << ".";
}

template <typename T>
std::unique_ptr<T[]> CopyData(const ShapeVector &shape, void *const data, size_t data_len) {
  size_t size = SizeOf(shape);
  if (size * sizeof(T) != data_len) {
    MS_LOG(EXCEPTION) << "Incorrect tensor input data length " << data_len << ", expect " << size * sizeof(T)
                      << " item size " << sizeof(T);
  }
  auto buf = static_cast<T *>(data);
  return NewData<T>(buf, size);
}

// TensorStringifier provide methods to convert tensor data to its string representation.
template <typename T>
class TensorStringifier {
 public:
  TensorStringifier(const T *data, size_t data_size, size_t ndim) : data_(data), data_size_(data_size), ndim_(ndim) {}
  ~TensorStringifier() = default;

  std::string ToString(TypeId, const ShapeVector &shape, bool use_comma) const {
    constexpr auto valid =
      std::is_same<T, bool>::value || std::is_same<T, uint8_t>::value || std::is_same<T, int8_t>::value ||
      std::is_same<T, int16_t>::value || std::is_same<T, int32_t>::value || std::is_same<T, int64_t>::value ||
      std::is_same<T, uint16_t>::value || std::is_same<T, uint32_t>::value || std::is_same<T, uint64_t>::value ||
      std::is_same<T, float16>::value || std::is_same<T, float>::value || std::is_same<T, double>::value ||
      std::is_same<T, ComplexStorage<float>>::value || std::is_same<T, ComplexStorage<double>>::value;
    static_assert(valid, "Type is invalid");
    if (data_size_ == 0) {
      return "";
    }
    if (data_ == nullptr) {
      return "<uninitialized>";
    }

    std::ostringstream ss;
    if (data_size_ == 1 && ndim_ == 0) {  // Scalar
      int max = 0;
      OutputDataString(ss, 0, 0, 1, false, &max);
      return ss.str();
    }

    int num_width = 0;
    ssize_t cursor = 0;
    SummaryStringRecursive(ss, shape, &cursor, 0, use_comma, &num_width);
    return ProcessPlaceholder(ss, num_width);
  }

 private:
  static void OutputFloatDataString(std::ostringstream &ss, bool isScalar, const T &value) {
    if (isScalar) {
      ss << value;
    } else {
      // The placeholder of float16 is fixed at 11, while float/double is fixed at 15.
      const int width = std::is_same<T, float16>::value ? 11 : 15;
      // The printing precision of float16 is fixed at 4, while float/double is fixed at 8.
      const int precision = std::is_same<T, float16>::value ? 4 : 8;
      ss << std::setw(width) << std::setprecision(precision) << std::setiosflags(std::ios::scientific | std::ios::right)
         << value;
    }
  }

  static void OutputBoolDataString(std::ostringstream &ss, bool isScalar, const T &value) {
    if (isScalar) {
      ss << (value ? "True" : "False");
    } else {
      constexpr int bool_max_width = sizeof("False") - 1;
      ss << std::setw(bool_max_width) << std::setiosflags(std::ios::right) << (value ? "True" : "False");
    }
  }

  static void OutputOtherDataString(std::ostringstream &ss, bool isScalar, const T &value, int *max_width) {
    std::ostringstream value_ss;
    if constexpr (std::is_same<T, uint8_t>::value) {
      value_ss << static_cast<uint16_t>(value);
    } else if constexpr (std::is_same<T, int8_t>::value) {
      value_ss << static_cast<int16_t>(value);
    } else {
      value_ss << value;
    }
    auto value_str = value_ss.str();
    if (!isScalar) {
      const int width = static_cast<int>(value_str.size());
      *max_width = std::max(*max_width, width);
      // Add a padding string before the number, such as "###123", for subsequent replacement.
      std::string pad(width, '#');
      ss << pad;
    }
    ss << value_str;
  }

  static std::string ProcessPlaceholder(const std::ostringstream &ss, int max_width) {
    std::string str = ss.str();
    if constexpr (std::is_same<T, bool>::value || std::is_same<T, float16>::value || std::is_same<T, float>::value ||
                  std::is_same<T, double>::value) {
      return str;
    }
    // Replace # with placeholder.
    size_t index = str.find('#');
    while (index != std::string::npos) {
      size_t pos = index;
      while (str[pos] == '#') {
        pos++;
      }
      size_t len = pos - index;
      std::string space(max_width - SizeToInt(len), ' ');
      str = str.replace(index, len, space);
      index = str.find('#', index);
    }
    return str;
  }

  void OutputDataString(std::ostringstream &ss, ssize_t cursor, ssize_t start, ssize_t end, bool use_comma,
                        int *max_width) const {
    const bool isScalar = ndim_ == 0 && end - start == 1;
    constexpr auto isBool = std::is_same<T, bool>::value;
    constexpr auto isFloat =
      std::is_same<T, float16>::value || std::is_same<T, float>::value || std::is_same<T, double>::value;
    constexpr auto isComplex =
      std::is_same<T, ComplexStorage<float>>::value || std::is_same<T, ComplexStorage<double>>::value;
    constexpr int linefeedThreshold = isFloat ? kThreshold1DFloat : (isBool ? kThreshold1DBool : kThreshold1DInt);
    for (ssize_t i = start; i < end && (cursor + i) < static_cast<ssize_t>(data_size_); i++) {
      const auto value = data_[cursor + i];
      if constexpr (isComplex) {
        ss << value;
      } else if constexpr (isFloat) {
        OutputFloatDataString(ss, isScalar, value);
      } else if (isBool) {
        OutputBoolDataString(ss, isScalar, value);
      } else {
        OutputOtherDataString(ss, isScalar, value, max_width);
      }
      if (!isScalar && i != end - 1) {
        if (use_comma) {
          ss << ',';
        }
        ss << ' ';
      }
      if (!isScalar && ndim_ == 1 && (i + 1) % linefeedThreshold == 0) {
        // Add a line feed every {threshold of type} for 1D tensor.
        ss << '\n' << ' ';
      }
    }
  }

  void SummaryStringRecursive(std::ostringstream &ss, const ShapeVector &shape, ssize_t *cursor, ssize_t depth,
                              bool use_comma, int *max_width) const {
    if (depth >= static_cast<ssize_t>(ndim_)) {
      return;
    }
    ss << '[';
    if (depth == static_cast<ssize_t>(ndim_) - 1) {  // Bottom dimension
      ssize_t num = shape[depth];
      if (num > kThreshold && ndim_ > 1) {
        OutputDataString(ss, *cursor, 0, kThreshold >> 1, use_comma, max_width);
        ss << ' ' << kEllipsis << ' ';
        OutputDataString(ss, *cursor, num - (kThreshold >> 1), num, use_comma, max_width);
      } else {
        OutputDataString(ss, *cursor, 0, num, use_comma, max_width);
      }
      *cursor += num;
    } else {  // Middle dimension
      ssize_t num = shape[depth];
      // Handle the first half.
      for (ssize_t i = 0; i < std::min(static_cast<ssize_t>(kThreshold >> 1), num); i++) {
        if (i > 0) {
          if (use_comma) {
            ss << ',';
          }
          ss << '\n';
          ss << std::setw(depth + 1) << ' ';  // Add the indent.
        }
        SummaryStringRecursive(ss, shape, cursor, depth + 1, use_comma, max_width);
      }
      // Handle the ignored part.
      if (num > kThreshold) {
        if (use_comma) {
          ss << ',';
        }
        ss << '\n';
        ss << std::setw(depth + 1) << ' ';  // Add the indent.
        ss << kEllipsis;
        // Ignored at this layer.
        ssize_t ignored = shape[depth + 1];
        const size_t offset = 2;
        for (ssize_t i = depth + offset; i < static_cast<ssize_t>(ndim_); i++) {
          ignored *= shape[i];
        }
        // Multiple with ignored layers number.
        ignored *= (num - kThreshold);
        *cursor += ignored;
      }
      // Handle the second half.
      if (num > (kThreshold >> 1)) {
        ssize_t iter_times =
          std::min(static_cast<ssize_t>(num - (kThreshold >> 1)), static_cast<ssize_t>(kThreshold >> 1));
        for (ssize_t i = 0; i < iter_times; i++) {
          if (use_comma && (i != 0 || num <= kThreshold)) {  // Not just after ignored part || Not handle ignored part
            ss << ',';
          }
          ss << '\n';
          ss << std::setw(depth + 1) << ' ';  // Add the indent.
          SummaryStringRecursive(ss, shape, cursor, depth + 1, use_comma, max_width);
        }
      }
    }
    ss << ']';
  }

  const T *data_;
  const size_t data_size_;
  const size_t ndim_;
};

// Tensor data implementation.
template <typename T>
class TensorDataImpl : public TensorData {
 public:
  explicit TensorDataImpl(const ShapeVector &shape) : ndim_(shape.size()), data_size_(SizeOf(shape)) {}
  ~TensorDataImpl() override = default;

  TensorDataImpl(const ShapeVector &shape, void *data, size_t data_len)
      : ndim_(shape.size()), data_size_(SizeOf(shape)), data_(CopyData<T>(shape, data, data_len)) {}

  TensorDataImpl(const ShapeVector &shape, void *data, TypeId data_type)
      : ndim_(shape.size()), data_size_(SizeOf(shape)), data_(CopyData<T>(shape, data, data_type)) {}

  template <typename U>
  TensorDataImpl(const ShapeVector &shape, const U *input, size_t size)
      : ndim_(shape.size()), data_size_(SizeOf(shape)), data_(NewData<T>(input, size)) {}

  template <typename Scalar>
  TensorDataImpl(const ShapeVector &shape, Scalar scalar)
      : ndim_(shape.size()), data_size_(SizeOf(shape)), data_(NewData<T>(scalar)) {}

  ssize_t size() const override { return static_cast<ssize_t>(data_size_); }

  ssize_t itemsize() const override { return static_cast<ssize_t>(sizeof(T)); }

  ssize_t nbytes() const override { return size() * itemsize(); }

  ssize_t ndim() const override { return static_cast<ssize_t>(ndim_); }

  bool is_sub_data() const override { return false; }

  bool has_sub_data() const override { return false; }

  void *data() override {
    if (data_ == nullptr) {
      if (data_size_ > INT32_MAX) {
        MS_LOG(WARNING) << "Try to alloca a large memory, size is:" << data_size_ * sizeof(T);
      }
      // Lazy allocation.
      data_ = std::make_unique<T[]>(data_size_);
    }
    return data_.get();
  }

  const void *const_data() const override {
    // May return nullptr if data not initialized.
    return data_.get();
  }

  virtual bool equals(const TensorDataImpl<T> &other) const {
    auto ptr = &other;
    if (ptr == this) {
      return true;
    }
    if (data_ == nullptr || ptr->data_ == nullptr) {
      return false;
    }
    return (ndim_ == ptr->ndim_) && (data_size_ == ptr->data_size_) &&
           std::equal(data_.get(), data_.get() + data_size_, ptr->data_.get());
  }

  bool equals(const TensorData &other) const override {
    // Not same type, compare data byte by byte.
    return TensorData::equals(other);
  }

  std::string ToString(TypeId type, const ShapeVector &shape, bool use_comma) const override {
    TensorStringifier<T> stringifier{data_.get(), data_size_, ndim_};
    return stringifier.ToString(type, shape, use_comma);
  }

 private:
  size_t ndim_{0};
  size_t data_size_{0};
  std::unique_ptr<T[]> data_;
};

// Tensor chunk data.
template <typename T>
class TensorChunkData : public TensorDataImpl<T> {
 public:
  explicit TensorChunkData(size_t size) : TensorDataImpl<T>(ShapeVector{static_cast<int64_t>(size)}) {}

  ~TensorChunkData() override = default;

  bool has_sub_data() const override { return true; }
};

// Tensor compression data.
template <typename T>
class CompressionTensorData : public TensorDataImpl<T> {
 public:
  explicit CompressionTensorData(size_t size) : TensorDataImpl<T>(ShapeVector{static_cast<int64_t>(size)}) {}

  ~CompressionTensorData() override = default;
};

// TensorSubData is the base class to provide tensor data as a segment from an owner tensor data.
class TensorSubData : public TensorData {
 public:
  TensorSubData(const TensorPtr &data_owner, size_t offset, size_t data_size, size_t ndim)
      : data_owner_(data_owner), data_offset_(offset), data_size_(data_size), ndim_(ndim) {}

  ~TensorSubData() override = default;

  ssize_t size() const override { return static_cast<ssize_t>(data_size_); }

  ssize_t nbytes() const override { return size() * itemsize(); }

  ssize_t ndim() const override { return static_cast<ssize_t>(ndim_); }

  bool is_sub_data() const override { return true; }

  bool has_sub_data() const override { return false; }

  void *data() override {
    // Set data initialized if data() is called.
    data_initialized_ = true;
    auto start = static_cast<uint8_t *>(data_owner_->data().data());
    return static_cast<void *>(start + data_offset_);
  }

  const void *const_data() const override {
    if (!data_initialized_) {
      // Return nullptr if data not initialized.
      return nullptr;
    }
    auto start = static_cast<uint8_t *>(data_owner_->data().data());
    return static_cast<void *>(start + data_offset_);
  }

  // Get the owner Tensor.
  const TensorPtr &GetOwner() const { return data_owner_; }

  // Data offset in bytes.
  size_t data_offset() const { return data_offset_; }

 protected:
  const TensorPtr data_owner_;
  size_t data_offset_{0};
  size_t data_size_{0};
  size_t ndim_{0};
  bool data_initialized_{false};
};

// TensorSubDataImpl implements methods that rely on T.
template <typename T>
class TensorSubDataImpl : public TensorSubData {
 public:
  TensorSubDataImpl(const TensorPtr &data_owner, size_t offset, size_t data_size, size_t ndim)
      : TensorSubData(data_owner, offset, data_size, ndim) {}

  ~TensorSubDataImpl() override = default;

  ssize_t itemsize() const override { return static_cast<ssize_t>(sizeof(T)); }

  std::string ToString(TypeId type, const ShapeVector &shape, bool use_comma) const override {
    TensorStringifier<T> stringifier{static_cast<const T *>(const_data()), data_size_, ndim_};
    return stringifier.ToString(type, shape, use_comma);
  }
};

template <template <class> class ImplClass = TensorDataImpl, typename... Args>
TensorDataPtr MakeTensorData(TypeId data_type, Args &&... args) {
  switch (data_type) {
    case kNumberTypeBool:
      return std::make_shared<ImplClass<bool>>(std::forward<Args>(args)...);
    case kNumberTypeUInt8:
      return std::make_shared<ImplClass<uint8_t>>(std::forward<Args>(args)...);
    case kNumberTypeInt8:
      return std::make_shared<ImplClass<int8_t>>(std::forward<Args>(args)...);
    case kNumberTypeInt16:
      return std::make_shared<ImplClass<int16_t>>(std::forward<Args>(args)...);
    case kNumberTypeInt:
    case kNumberTypeInt32:
      return std::make_shared<ImplClass<int32_t>>(std::forward<Args>(args)...);
    case kNumberTypeInt64:
      return std::make_shared<ImplClass<int64_t>>(std::forward<Args>(args)...);
    case kNumberTypeUInt16:
      return std::make_shared<ImplClass<uint16_t>>(std::forward<Args>(args)...);
    case kNumberTypeUInt32:
      return std::make_shared<ImplClass<uint32_t>>(std::forward<Args>(args)...);
    case kNumberTypeUInt64:
      return std::make_shared<ImplClass<uint64_t>>(std::forward<Args>(args)...);
    case kNumberTypeFloat16:
      return std::make_shared<ImplClass<float16>>(std::forward<Args>(args)...);
    case kNumberTypeFloat:
      return std::make_shared<ImplClass<float>>(std::forward<Args>(args)...);
    case kNumberTypeFloat32:
      return std::make_shared<ImplClass<float>>(std::forward<Args>(args)...);
    case kNumberTypeFloat64:
      return std::make_shared<ImplClass<double>>(std::forward<Args>(args)...);
    case kNumberTypeComplex64:
      return std::make_shared<ImplClass<ComplexStorage<float>>>(std::forward<Args>(args)...);
    case kNumberTypeComplex128:
      return std::make_shared<ImplClass<ComplexStorage<double>>>(std::forward<Args>(args)...);
    case kObjectTypeString:
      return std::make_shared<ImplClass<uint8_t>>(std::forward<Args>(args)...);
    case kObjectTypeTensorType:
    case kObjectTypeMapTensorType:
      return std::make_shared<ImplClass<int>>(std::forward<Args>(args)...);
    default:
      break;
  }
  MS_LOG(ERROR) << "Cannot construct Tensor because of unsupported data type: " << data_type << ".";
  return nullptr;
}

TensorDataPtr MakeTensorSubData(const TensorPtr &owner, size_t offset, const TensorDataPtr &data) {
  if (data->nbytes() == 0) {
    MS_LOG(EXCEPTION) << "Tensor data size is 0.";
  }
  auto sub_data = MakeTensorData<TensorSubDataImpl>(owner->data_type(), owner, offset, data->size(), data->ndim());
  // If tensor data is initialized, copy it.
  if (data->const_data() != nullptr) {
    CopyTensorData(sub_data, data);
  }
  return sub_data;
}

Tensor::Tensor(const Tensor &tensor)
    : MetaTensor(tensor),
      init_flag_(tensor.init_flag_),
      is_forward_output_(tensor.is_forward_output_),
      data_(tensor.data_),
      id_(tensor.id_),
      event_(tensor.event_),
      need_wait_(tensor.need_wait_),
      sync_status_(tensor.sync_status_),
      device_sync_(tensor.device_sync_),
      need_release_device_mem_(tensor.need_release_device_mem_),
      cache_enable_(tensor.cache_enable_),
      base_shape_ptr_(tensor.base_shape_ptr_),
      cache_tensor_ptr_(tensor.cache_tensor_ptr_),
      hashmap_tensor_ptr_(tensor.hashmap_tensor_ptr_),
      padding_type_(tensor.padding_type()),
      device_event_(tensor.device_event_),
      lazy_callback_(tensor.lazy_callback_),
      user_data_(tensor.user_data_),
      compression_type_(tensor.compression_type_),
      tensor_name_(tensor.tensor_name_) {}

Tensor::Tensor(const Tensor &tensor, TypeId data_type)
    : MetaTensor(data_type, tensor.shape_),
      init_flag_(tensor.init_flag_),
      is_forward_output_(tensor.is_forward_output_),
      data_(MakeTensorData(data_type, tensor.shape_, tensor.data_->data(), tensor.data_type_)),
      id_(tensor.data_type_ != data_type ? MakeId() : tensor.id_),
      event_(tensor.event_),
      need_wait_(tensor.need_wait_),
      sync_status_(tensor.sync_status_),
      device_sync_(tensor.device_sync_),
      need_release_device_mem_(tensor.need_release_device_mem_),
      cache_enable_(tensor.cache_enable_),
      base_shape_ptr_(tensor.base_shape_ptr_),
      cache_tensor_ptr_(tensor.cache_tensor_ptr_),
      hashmap_tensor_ptr_(tensor.hashmap_tensor_ptr_),
      padding_type_(tensor.padding_type()),
      device_event_(tensor.device_event_),
      lazy_callback_(tensor.lazy_callback_),
      user_data_(tensor.user_data_),
      compression_type_(tensor.compression_type_),
      tensor_name_(tensor.tensor_name_) {}

Tensor::Tensor(TypeId data_type, const ShapeVector &shape, TensorDataPtr data)
    : MetaTensor(data_type, shape), data_(std::move(data)), id_(MakeId()) {}

Tensor::Tensor(TypeId data_type, const ShapeVector &shape)
    : Tensor(data_type, shape, MakeTensorData(data_type, shape)) {}

Tensor::Tensor(TypeId data_type, const ShapeVector &shape, void *data, size_t data_len)
    : Tensor(data_type, shape, MakeTensorData(data_type, shape, data, data_len)) {}

Tensor::Tensor(TypeId data_type, const ShapeVector &shape, void *data, TypeId src_data_type)
    : Tensor(data_type, shape, MakeTensorData(data_type, shape, data, src_data_type)) {}

Tensor::Tensor(const std::vector<int64_t> &input, const TypePtr &data_type)
    : MetaTensor(TypeIdOf(data_type, kNumberTypeInt64), {static_cast<int>(input.size())}),
      data_(MakeTensorData(data_type_, shape_, input.data(), input.size())),
      id_(MakeId()) {}

Tensor::Tensor(const std::vector<int32_t> &input, const TypePtr &data_type)
    : MetaTensor(TypeIdOf(data_type, kNumberTypeInt32), {static_cast<int>(input.size())}),
      data_(MakeTensorData(data_type_, shape_, input.data(), input.size())),
      id_(MakeId()) {}

Tensor::Tensor(const std::vector<double> &input, const TypePtr &data_type)
    : MetaTensor(TypeIdOf(data_type, kNumberTypeFloat32), {static_cast<int>(input.size())}),
      data_(MakeTensorData(data_type_, shape_, input.data(), input.size())),
      id_(MakeId()) {}

Tensor::Tensor(int64_t input, const TypePtr &data_type)
    : MetaTensor(TypeIdOf(data_type, kNumberTypeInt64), {}),
      data_(MakeTensorData(data_type_, ShapeVector{}, input)),
      id_(MakeId()) {}

Tensor::Tensor(int32_t input, const TypePtr &data_type)
    : MetaTensor(TypeIdOf(data_type, kNumberTypeInt32), {}),
      data_(MakeTensorData(data_type_, ShapeVector{}, input)),
      id_(MakeId()) {}

Tensor::Tensor(int16_t input, const TypePtr &data_type)
    : MetaTensor(TypeIdOf(data_type, kNumberTypeInt16), {}),
      data_(MakeTensorData(data_type_, ShapeVector{}, input)),
      id_(MakeId()) {}

Tensor::Tensor(int8_t input, const TypePtr &data_type)
    : MetaTensor(TypeIdOf(data_type, kNumberTypeInt8), {}),
      data_(MakeTensorData(data_type_, ShapeVector{}, input)),
      id_(MakeId()) {}

Tensor::Tensor(double input, const TypePtr &data_type)
    : MetaTensor(TypeIdOf(data_type, kNumberTypeFloat32), {}),
      data_(MakeTensorData(data_type_, ShapeVector{}, input)),
      id_(MakeId()) {}

Tensor::Tensor(float input, const TypePtr &data_type)
    : MetaTensor(TypeIdOf(data_type, kNumberTypeFloat32), {}),
      data_(MakeTensorData(data_type_, ShapeVector{}, input)),
      id_(MakeId()) {}

Tensor::Tensor(float16 input, const TypePtr &data_type)
    : MetaTensor(TypeIdOf(data_type, kNumberTypeFloat16), {}),
      data_(MakeTensorData(data_type_, ShapeVector{}, input)),
      id_(MakeId()) {}

Tensor::Tensor(uint64_t input, const TypePtr &data_type)
    : MetaTensor(TypeIdOf(data_type, kNumberTypeUInt64), {}),
      data_(MakeTensorData(data_type_, ShapeVector{}, input)),
      id_(MakeId()) {}

Tensor::Tensor(uint32_t input, const TypePtr &data_type)
    : MetaTensor(TypeIdOf(data_type, kNumberTypeUInt32), {}),
      data_(MakeTensorData(data_type_, ShapeVector{}, input)),
      id_(MakeId()) {}

Tensor::Tensor(uint16_t input, const TypePtr &data_type)
    : MetaTensor(TypeIdOf(data_type, kNumberTypeUInt16), {}),
      data_(MakeTensorData(data_type_, ShapeVector{}, input)),
      id_(MakeId()) {}

Tensor::Tensor(uint8_t input, const TypePtr &data_type)
    : MetaTensor(TypeIdOf(data_type, kNumberTypeUInt8), {}),
      data_(MakeTensorData(data_type_, ShapeVector{}, input)),
      id_(MakeId()) {}

Tensor::Tensor(bool input, const TypePtr &data_type)
    : MetaTensor(TypeIdOf(data_type, kNumberTypeBool), {}),
      data_(MakeTensorData(data_type_, ShapeVector{}, input)),
      id_(MakeId()) {}

Tensor::Tensor(TypeId data_type, size_t data_size)
    : Tensor(data_type, ShapeVector{static_cast<int64_t>(data_size)},
             MakeTensorData<TensorChunkData>(data_type, data_size)) {}

Tensor::Tensor(TypeId origin_data_type, const ShapeVector &shape, size_t compression_data_size,
               TensorCompressionType compression_type)
    : Tensor(origin_data_type, shape, MakeTensorData<CompressionTensorData>(kNumberTypeInt8, compression_data_size)) {
  compression_type_ = compression_type;
}

bool Tensor::operator==(const Tensor &tensor) const {
  return (&tensor == this || (MetaTensor::operator==(tensor) && data_ == tensor.data_));
}

bool Tensor::ValueEqual(const Tensor &tensor) const {
  if (is_parameter_ != tensor.is_parameter_) {
    return false;
  }
  if (is_parameter_ && param_info_->name() != tensor.param_info_->name()) {
    return false;
  }
  return (&tensor == this || (MetaTensor::operator==(tensor) && data_->equals(*tensor.data_)));
}

void Tensor::ExecuteLazyTask() const {
  if (lazy_callback_ != nullptr) {
    lazy_callback_();
  }
}

// Assign value to this tensor.
Tensor &Tensor::AssignValue(const Tensor &tensor) {
  if (this != &tensor) {
    lazy_callback_ = tensor.lazy_callback_;
    ExecuteLazyTask();
    MetaTensor::operator=(tensor);
    device_sync_ = tensor.device_sync_;
    need_release_device_mem_ = tensor.need_release_device_mem_;
    is_forward_output_ = tensor.is_forward_output_;
    if (data_->is_sub_data()) {
      // If tensor data is sub data, we should keep data
      // memory address unchange and copy data to it.
      CopyTensorData(data_, tensor.data_);
    } else {
      data_ = tensor.data_;
    }
    if (!is_parameter_) {
      id_ = tensor.id_;
    }
    event_ = tensor.event_;
    need_wait_ = tensor.need_wait_;
    sync_status_ = tensor.sync_status_;
    padding_type_ = tensor.padding_type_;
    device_event_ = tensor.device_event_;
  }
  return *this;
}

abstract::AbstractBasePtr Tensor::ToAbstract() {
  auto tens = shared_from_base<Tensor>();
  auto dtype = tens->Dtype();
  if (!IsSubType(dtype, kNumber) && !IsSubType(dtype, kString) && !IsSubType(dtype, kTensorType)) {
    MS_LOG(EXCEPTION) << "Expect tensor type kNumber or kString or kTensor but got: " << dtype->ToString() << ".";
  }
  abstract::AbstractTensorPtr abs_tensor = nullptr;
  if (base_shape_ptr_ == nullptr) {
    auto tensor_shape = tens->shape();
    abs_tensor = std::make_shared<abstract::AbstractTensor>(dtype, tensor_shape);
  } else {
    abs_tensor = std::make_shared<abstract::AbstractTensor>(dtype, base_shape_ptr_);
  }
  // if is parameter always no value.
  if (is_parameter_) {
    auto param_name = param_info_->name();
    auto ref_key = std::make_shared<RefKey>(param_name);
    abs_tensor = std::make_shared<abstract::AbstractRefTensor>(abs_tensor, ref_key);
  } else {
    abs_tensor->set_value(shared_from_base<Tensor>());
  }
  return abs_tensor;
}

std::string Tensor::GetShapeAndDataTypeInfo() const {
  std::ostringstream buf;
  buf << "Tensor shape:[" << shape() << "]" << this->Dtype()->ToString();
  return buf.str();
}

std::string Tensor::ToStringInternal(size_t limit_size) const {
  std::ostringstream buf;
  auto dtype = Dtype();
  MS_EXCEPTION_IF_NULL(dtype);
  buf << "Tensor(shape=" << ShapeToString(shape_) << ", dtype=" << dtype->ToString() << ", value=";
  if (limit_size == 0 || DataSize() < limit_size) {
    // Only print data for small tensor.
    buf << ((data().ndim() > 1) ? "\n" : "") << data().ToString(data_type_, shape_, false);
  } else {
    buf << "[...]";
  }
  if (is_parameter_) {
    buf << ", name=" << param_info_->name();
  }
  buf << ")";
  return buf.str();
}

std::string Tensor::ToString() const {
  constexpr size_t small_tensor_size = 30;
  return ToStringInternal(small_tensor_size);
}

std::string Tensor::ToStringNoLimit() const { return ToStringInternal(0); }

std::string Tensor::ToStringRepr() const {
  std::ostringstream buf;
  auto dtype = Dtype();
  MS_EXCEPTION_IF_NULL(dtype);
  buf << "Tensor(shape=" << ShapeToString(shape_) << ", dtype=" << dtype->ToString()
      << ", value=" << ((data().ndim() > 1) ? '\n' : ' ') << data().ToString(data_type_, shape_, true) << ')';
  return buf.str();
}

void Tensor::data_sync(bool need_wait) const {
  if (need_wait) {
    ExecuteLazyTask();
    Wait();
  }
  if (device_sync_ == nullptr) {
    return;
  }

  if (data_->is_sub_data()) {
    return;
  }

  std::vector<size_t> shape_tmp;
  (void)std::transform(shape().begin(), shape().end(), std::back_inserter(shape_tmp), IntToSize);
  auto size = abstract::ShapeSize(shape_tmp) * abstract::TypeIdSize(data_type());
  auto address = device_sync_;
  if (size != 0 && !address->SyncDeviceToHost(shape(), size, data_type(), data_c())) {
    MS_LOG(EXCEPTION) << "SyncDeviceToHost failed.";
  }
  sync_status_ = kNeedSyncHostToDevice;
}

void Tensor::data_sync_directly(const DeviceSync *const device_sync, bool need_wait) const {
  if (need_wait) {
    ExecuteLazyTask();
    Wait();
  }
  if (device_sync == nullptr) {
    return;
  }

  if (data_->is_sub_data()) {
    return;
  }

  std::vector<size_t> shape_tmp;
  (void)std::transform(shape().begin(), shape().end(), std::back_inserter(shape_tmp), IntToSize);
  auto size = abstract::ShapeSize(shape_tmp) * abstract::TypeIdSize(data_type());
  if (size != 0 && !device_sync->SyncDeviceToHost(shape(), size, data_type(), data_c())) {
    MS_LOG(EXCEPTION) << "SyncDeviceToHost failed.";
  }
  sync_status_ = kNeedSyncHostToDevice;
}

TypeId Tensor::set_data_type(TypeId data_type) {
  if (data_type != data_type_) {
    data_ = MakeTensorData(data_type, shape_, data_->data(), data_type_);
    return MetaTensor::set_data_type(data_type);
  }
  return data_type;
}

size_t Tensor::set_shape(const ShapeVector &shape) {
  if (DataSize() != SizeOf(shape)) {
    data_ = MakeTensorData(data_type_, shape);
  }
  return MetaTensor::set_shape(shape);
}

std::pair<void *, size_t> Tensor::GetChunkOffset() const {
  // Get sub-data.
  auto sub_data = std::dynamic_pointer_cast<TensorSubData>(data_ptr());
  if (sub_data == nullptr) {
    return {nullptr, 0};
  }
  // Get owner tensor from sub-data.
  auto owner_tensor = sub_data->GetOwner();
  MS_EXCEPTION_IF_NULL(owner_tensor);
  return {owner_tensor->data_c(), sub_data->data_offset()};
}

// TensorChunk holds info for a chunk.
struct TensorChunk {
  size_t size{0};                  // chunk size in the number of elements.
  size_t bytes{0};                 // chunk size in bytes.
  std::vector<TensorPtr> tensors;  // tensors belong to this chunk.
};

static TypeId normalize_type(TypeId type_id) {
  if (type_id == kNumberTypeFloat) {
    // kNumberTypeFloat is an alias of kNumberTypeFloat32.
    return kNumberTypeFloat32;
  }
  return type_id;
}

static std::map<TypeId, std::vector<TensorChunk>> GroupingTensors(const TensorPtrList &tensors, size_t fusion_size) {
  // Use std::map to keep order by type id.
  std::map<TypeId, std::vector<TensorChunk>> group_info;
  for (auto &tensor : tensors) {
    MS_EXCEPTION_IF_NULL(tensor);
    auto tensor_bytes = static_cast<size_t>(tensor->data().nbytes());
    if ((fusion_size != 0) && (tensor_bytes > fusion_size)) {
      MS_LOG(EXCEPTION) << "Fusion size " << fusion_size << " is too small for a tensor size " << tensor_bytes << ".";
    }
    auto &chunks = group_info[normalize_type(tensor->data_type())];
    if (chunks.empty()) {
      (void)chunks.emplace_back();
    }
    if ((fusion_size != 0) && (chunks.back().bytes + tensor_bytes > fusion_size)) {
      (void)chunks.emplace_back();
    }
    auto &chunk = chunks.back();
    chunk.size += tensor->DataSize();
    chunk.bytes += tensor_bytes;
    (void)chunk.tensors.emplace_back(tensor);
  }
  return group_info;
}

TensorPtrList Tensor::FlattenTensors(const TensorPtrList &tensors, size_t fusion_size) {
  // Result tensor list.
  TensorPtrList result_list;
  // Grouping tensors by data type and fusion size.
  auto group_info = GroupingTensors(tensors, fusion_size);
  // Create chunk tensors and copy data to them.
  for (auto &type_group : group_info) {
    auto chunk_dtype = normalize_type(type_group.first);
    for (auto &chunk : type_group.second) {
      // Create chunk thensor as a lazy initialized tensor, the tensor data
      // will be allocated when we begin to copy small tensors data into it.
      auto chunk_tensor = std::make_shared<Tensor>(chunk_dtype, chunk.size);
      // Reset and copy tensors data.
      size_t offset = 0;
      for (auto &tensor : chunk.tensors) {
        auto sub_data = MakeTensorSubData(chunk_tensor, offset, tensor->data_ptr());
        offset += static_cast<size_t>(sub_data->nbytes());
        tensor->data_ = sub_data;
      }
      // Save chunk tensor to result list.
      (void)result_list.emplace_back(std::move(chunk_tensor));
    }
  }
  return result_list;
}

bool Tensor::IsFlattened(const TensorPtrList &tensors) {
  // Tensor data is flattened if all tensors data are TensorSubData.
  return std::all_of(tensors.begin(), tensors.end(), [](const TensorPtr &tensor) {
    MS_EXCEPTION_IF_NULL(tensor);
    auto data_ptr = tensor->data_ptr().get();
    return dynamic_cast<TensorSubData *>(data_ptr) != nullptr;
  });
}

TensorPtrList Tensor::GetFlattenedTensors(const TensorPtrList &tensors) {
  // Use std::map to keep order by type id.
  std::map<TypeId, OrderedSet<TensorPtr>> chunk_map;
  for (auto &tensor : tensors) {
    // Get sub-data.
    auto sub_data = std::dynamic_pointer_cast<TensorSubData>(tensor->data_ptr());
    if (sub_data == nullptr) {
      MS_LOG(WARNING) << "Tensors are not flattened.";
      return {};
    }
    // Get owner tensor from sub-data.
    auto owner_tensor = sub_data->GetOwner();
    MS_EXCEPTION_IF_NULL(owner_tensor);
    // Add as chunk tensor by its data type.
    auto chunk_dtype = normalize_type(tensor->data_type());
    chunk_map[chunk_dtype].add(owner_tensor);
  }
  // Generate result tensor list.
  TensorPtrList result_tensors;
  for (auto &entry : chunk_map) {
    auto &chunk_tensors = entry.second;
    (void)result_tensors.insert(result_tensors.end(), chunk_tensors.begin(), chunk_tensors.end());
  }
  return result_tensors;
}

size_t Tensor::GetFusionSize(const TensorPtrList &flat_tensors) {
  size_t fusion_size = 0;
  std::map<TypeId, size_t> type_groups;
  for (auto &tensor : flat_tensors) {
    MS_EXCEPTION_IF_NULL(tensor);
    auto tensor_bytes = static_cast<size_t>(tensor->data().nbytes());
    if (tensor_bytes > fusion_size) {
      fusion_size = tensor_bytes;
    }
    ++type_groups[tensor->data_type()];
  }
  const bool only_one_chunk_for_each_type =
    std::all_of(type_groups.begin(), type_groups.end(), [](auto const &e) { return e.second == 1; });
  if (only_one_chunk_for_each_type) {
    return 0;
  }
  return fusion_size;
}

bool Tensor::is_persistent_data() const { return this->data().is_persistent_data(); }

CSRTensor::CSRTensor(const TensorPtr indptr, const TensorPtr indices, const TensorPtr values, const ShapeVector &shape)
    : MetaSparseTensor(values->data_type(), shape), indptr_(indptr), indices_(indices), values_(values) {}

std::string CSRTensor::ToString() const {
  std::ostringstream buf;
  MS_EXCEPTION_IF_NULL(values_);
  MS_EXCEPTION_IF_NULL(indices_);
  MS_EXCEPTION_IF_NULL(indptr_);
  auto dtype = values_->Dtype();
  values_->data_sync(true);
  indices_->data_sync(true);
  indptr_->data_sync(true);
  buf << "CSRTensor(shape=" << ShapeToString(shape_) << ", dtype=" << dtype->ToString() << ", indptr=";
  buf << indptr_->ToString() << ", indices=" << indices_->ToString() << ", values=";
  buf << values_->ToString() << ")";
  return buf.str();
}

abstract::AbstractBasePtr CSRTensor::ToAbstract() {
  auto dtype = values_->Dtype();
  if (!IsSubType(dtype, kNumber) && !IsSubType(dtype, kString) && !IsSubType(dtype, kTensorType)) {
    MS_LOG(EXCEPTION) << "Expect tensor type kNumber or kString or kTensor but got: " << dtype->ToString() << ".";
  }

  auto indptr = indptr_->ToAbstract()->cast<abstract::AbstractTensorPtr>();
  auto indices = indices_->ToAbstract()->cast<abstract::AbstractTensorPtr>();
  auto values = values_->ToAbstract()->cast<abstract::AbstractTensorPtr>();
  std::vector<abstract::AbstractBasePtr> abstract_shape;
  (void)std::transform(
    shape_.begin(), shape_.end(), std::back_inserter(abstract_shape),
    [](auto shp) -> abstract::AbstractScalarPtr { return std::make_shared<abstract::AbstractScalar>(shp); });
  auto shape = std::make_shared<abstract::AbstractTuple>(abstract_shape);
  AbstractBasePtrList element_list{indptr, indices, values, shape};

  return std::make_shared<abstract::AbstractCSRTensor>(element_list);
}

const size_t CSRTensor::GetSizeAt(size_t index) const {
  if (index == kIndptrIdx) {
    MS_EXCEPTION_IF_NULL(indptr_);
    return indptr_->data().nbytes();
  } else if (index == kIndicesIdx) {
    MS_EXCEPTION_IF_NULL(indices_);
    return indices_->data().nbytes();
  } else if (index == kValuesIdx) {
    MS_EXCEPTION_IF_NULL(values_);
    return values_->data().nbytes();
  } else if (index >= kIndicesIdx && index < kShapeIdx + shape().size()) {
    return sizeof(int64_t);
  }
  MS_LOG(EXCEPTION) << "Invalid index: " << index << " for CSRTensor: " << ToString();
}

TensorPtr CSRTensor::GetTensorAt(size_t index) const {
  if (index == kIndptrIdx) {
    MS_EXCEPTION_IF_NULL(indptr_);
    return indptr_;
  } else if (index == kIndicesIdx) {
    MS_EXCEPTION_IF_NULL(indices_);
    return indices_;
  } else if (index == kValuesIdx) {
    MS_EXCEPTION_IF_NULL(values_);
    return values_;
  } else if (index >= kShapeIdx && index < kShapeIdx + shape().size()) {
    return std::make_shared<tensor::Tensor>(shape_[index - kShapeIdx], TypeIdToType(kNumberTypeInt64));
  }
  MS_LOG(EXCEPTION) << "Invalid index: " << index << " for CSRTensor: " << ToString();
}

TensorPtr COOTensor::GetTensorAt(size_t index) const {
  if (index == kIndicesIdx) {
    MS_EXCEPTION_IF_NULL(indices_);
    return indices_;
  } else if (index == kValuesIdx) {
    MS_EXCEPTION_IF_NULL(values_);
    return values_;
  } else if (index >= kShapeIdx && index < kShapeIdx + shape().size()) {
    return std::make_shared<tensor::Tensor>(shape_[index - kShapeIdx], TypeIdToType(kNumberTypeInt64));
  }
  MS_LOG(EXCEPTION) << "Invalid index: " << index << " for COOTensor: " << ToString();
}

std::string COOTensor::ToString() const {
  std::ostringstream buf;
  MS_EXCEPTION_IF_NULL(indices_);
  MS_EXCEPTION_IF_NULL(values_);
  indices_->data_sync(true);
  values_->data_sync(true);
  auto dtype = values_->Dtype();
  buf << "COOTensor(shape=" << ShapeToString(shape_) << ", dtype=" << dtype->ToString()
      << ", indices=" << indices_->ToString() << ", values=" << values_->ToString() << ")";
  return buf.str();
}

abstract::AbstractBasePtr COOTensor::ToAbstract() {
  MS_EXCEPTION_IF_NULL(values_);
  auto dtype = values_->Dtype();
  if (!IsSubType(dtype, kNumber) && !IsSubType(dtype, kString) && !IsSubType(dtype, kTensorType)) {
    MS_LOG(EXCEPTION) << "Expect tensor type kNumber or kString or kTensor but got: " << dtype->ToString() << ".";
  }
  MS_EXCEPTION_IF_NULL(indices_);
  MS_EXCEPTION_IF_NULL(indices_->ToAbstract());
  MS_EXCEPTION_IF_NULL(values_->ToAbstract());
  auto indices = indices_->ToAbstract()->cast<abstract::AbstractTensorPtr>();
  auto values = values_->ToAbstract()->cast<abstract::AbstractTensorPtr>();
  std::vector<abstract::AbstractBasePtr> abstract_shape;
  (void)std::transform(
    shape_.begin(), shape_.end(), std::back_inserter(abstract_shape),
    [](auto shp) -> abstract::AbstractScalarPtr { return std::make_shared<abstract::AbstractScalar>(shp); });
  auto shape = std::make_shared<abstract::AbstractTuple>(abstract_shape);
  AbstractBasePtrList element_list{indices, values, shape};

  return std::make_shared<abstract::AbstractCOOTensor>(element_list);
}

std::string RowTensor::ToString() const {
  std::ostringstream buf;
  MS_EXCEPTION_IF_NULL(indices_);
  MS_EXCEPTION_IF_NULL(values_);
  auto dtype = values_->Dtype();
  buf << "RowTensor(shape=" << ShapeToString(shape_) << ", dtype=" << dtype->ToString()
      << ", indices=" << indices_->ToString() << ", values=" << values_->ToString() << ")";
  return buf.str();
}

abstract::AbstractBasePtr RowTensor::ToAbstract() {
  auto dtype = values_->Dtype();
  if (!IsSubType(dtype, kNumber) && !IsSubType(dtype, kString) && !IsSubType(dtype, kTensorType)) {
    MS_LOG(EXCEPTION) << "Expect tensor type kNumber or kString or kTensor but got: " << dtype->ToString() << ".";
  }
  auto abs_sparse_tensor = std::make_shared<abstract::AbstractRowTensor>(dtype, shape_);
  MS_EXCEPTION_IF_NULL(indices_);
  MS_EXCEPTION_IF_NULL(indices_->ToAbstract());
  MS_EXCEPTION_IF_NULL(values_->ToAbstract());
  abs_sparse_tensor->set_indices(indices_->ToAbstract()->cast<abstract::AbstractTensorPtr>());
  abs_sparse_tensor->set_values(values_->ToAbstract()->cast<abstract::AbstractTensorPtr>());

  std::vector<abstract::AbstractBasePtr> abstract_shape;
  (void)std::transform(
    shape_.begin(), shape_.end(), std::back_inserter(abstract_shape),
    [](auto shp) -> abstract::AbstractScalarPtr { return std::make_shared<abstract::AbstractScalar>(shp); });
  abs_sparse_tensor->set_dense_shape(std::make_shared<abstract::AbstractTuple>(abstract_shape));

  return abs_sparse_tensor;
}
}  // namespace tensor
}  // namespace mindspore
