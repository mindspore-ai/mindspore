/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_IR_TENSOR_DATA_H_
#define MINDSPORE_CORE_IR_TENSOR_DATA_H_

#include <memory>
#include <string>
#include <iomanip>
#include <algorithm>
#include <utility>
#include <type_traits>
#include <complex>
#include "mindapi/base/macros.h"
#include "utils/os.h"
#include "base/complex_storage.h"
#include "utils/convert_utils_base.h"
#include "utils/system/env.h"
#include "utils/temp_file_manager.h"
#include "utils/shape_utils.h"
#include "base/float8_e5m2.h"
#include "base/float8_e4m3fn.h"
#include "base/hifloat8.h"

namespace mindspore::tensor {
// Tensor data interface.
class MS_CORE_API TensorData {
 public:
  /// \brief Virtual destructor is required for base classes.
  virtual ~TensorData() = default;

  /// \brief Get total number of elements.
  ///
  /// \return Total number of elements.
  virtual ssize_t size() const = 0;

  /// \brief Get byte size of a single element.
  ///
  /// \return Byte size of a single element.
  virtual ssize_t itemsize() const = 0;

  /// \brief Get total number of bytes.
  ///
  /// \return Total number of bytes.
  virtual ssize_t nbytes() const = 0;

  /// \brief Get number of dimensions.
  ///
  /// \return Number of dimensions.
  virtual ssize_t ndim() const = 0;

  /// \brief Get data pointer.
  ///
  /// \return Data pointer.
  virtual void *data() = 0;

  /// \brief Get const data pointer.
  ///
  /// \return Const data pointer.
  virtual void *const_data() const = 0;

  /// \brief Get whether this tensor data is from numpy.
  ///
  /// \return Whether this tensor data is from numpy.
  virtual bool is_from_numpy() const { return false; }

  /// \brief Whether the data are equal.
  ///
  /// \param[in] other Another TensorData.
  /// \return Ture if the two data are equal, otherwise false.
  virtual bool equals(const TensorData &other) const {
    if (this == &other) {
      return true;
    }
    // By default, compare data byte by byte.
    auto this_data = static_cast<const uint8_t *>(const_data());
    auto other_data = static_cast<const uint8_t *>(other.const_data());
    if (this_data == nullptr || other_data == nullptr) {
      // null means data not initialized, compare uninitialized data always return false.
      return false;
    }
    return (this_data == other_data) || (ndim() == other.ndim() && nbytes() == other.nbytes() &&
                                         std::equal(this_data, this_data + nbytes(), other_data));
  }

  /// \brief Get display information about this TensorData.
  ///
  /// \param[in] type The type of tensor data.
  /// \param[in] shape The shape of tensor data.
  /// \param[in] use_comma Whether to use comma.
  /// \return The display information.
  virtual std::string ToString(TypeId type, const ShapeVector &shape, bool use_comma) const = 0;

  /// \brief Set data saved file path.
  ///
  /// \param[in] data file path.
  /// \return Void.
  virtual void set_file_path(const std::string &path) {
    MS_LOG(INFO) << "Call default set file path, and do nothing with " << path << ".";
  }

  /// \brief Get data saved file path.
  ///
  /// \return data file path.
  virtual const std::string file_path() const { return ""; }
};

using TensorDataPtr = std::shared_ptr<TensorData>;

constexpr auto kEllipsis = "...";
constexpr auto kThreshold = 6;
constexpr auto kThreshold1D = 1000;

constexpr auto kThreshold1DFloat = kThreshold * 2;
constexpr auto kThreshold1DInt = kThreshold * 4;
constexpr auto kThreshold1DBool = kThreshold * 2;

template <typename T>
inline constexpr bool IsNonTrivialCastType = false;

template <>
inline constexpr bool IsNonTrivialCastType<float16> = true;
template <>
inline constexpr bool IsNonTrivialCastType<float8_e4m3fn> = true;
template <>
inline constexpr bool IsNonTrivialCastType<float8_e5m2> = true;
template <>
inline constexpr bool IsNonTrivialCastType<hifloat8> = true;
template <>
inline constexpr bool IsNonTrivialCastType<bfloat16> = true;
template <>
inline constexpr bool IsNonTrivialCastType<ComplexStorage<float>> = true;
template <>
inline constexpr bool IsNonTrivialCastType<ComplexStorage<double>> = true;

template <typename T, typename U>
std::unique_ptr<T[]> NewData(const U *input, size_t size) {
  if (input == nullptr || size == 0) {
    return nullptr;
  }
  if (size > INT32_MAX) {
    MS_LOG(WARNING) << "Try to alloca a large memory, size is:" << size * sizeof(T);
  }

  auto data = std::make_unique<T[]>(size);
  if constexpr (!std::is_same_v<T, U> && (IsNonTrivialCastType<T> || IsNonTrivialCastType<U>)) {
    // Because float16 and bfloat16 do not support implicit cast from/to other types,
    // We can not use std::copy() on array of float16 and bfloat16, use a loop here.
    for (size_t i = 0; i < size; ++i) {
      data[i] = static_cast<T>(input[i]);
    }
  } else {
    // otherwise, use std::copy for better performance.
    std::copy(input, input + size, data.get());
  }
  return data;
}

template <typename T, typename U>
void TransDataType(const U *input, T *output, size_t size) {
  if (input == nullptr || output == nullptr || size == 0) {
    return;
  }
  if (size > INT32_MAX) {
    MS_LOG(WARNING) << "Try to alloca a large memory, size is:" << size * sizeof(T);
  }

  if constexpr (!std::is_same_v<T, U> && (IsNonTrivialCastType<T> || IsNonTrivialCastType<U>)) {
    // Because float16 and bfloat16 do not support implicit cast from/to other types,
    // We can not use std::copy() on array of float16 and bfloat16, use a loop here.
    for (size_t i = 0; i < size; ++i) {
      output[i] = static_cast<T>(input[i]);
    }
  } else {
    // otherwise, use std::copy for better performance.
    std::copy(input, input + size, output);
  }
}

template <typename T, typename Scalar>
std::unique_ptr<T[]> NewData(Scalar scalar) {
  auto data = std::make_unique<T[]>(1);
  if constexpr ((std::is_same_v<Scalar, std::complex<float>> || std::is_same_v<Scalar, std::complex<double>>) &&
                !std::is_same_v<T, std::complex<float>> && !std::is_same_v<T, std::complex<double>>) {
    data[0] = static_cast<T>(scalar.real());
  } else {
    data[0] = static_cast<T>(scalar);
  }
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
    case kNumberTypeInt4: {
      auto buf = static_cast<int8_t *>(data);
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
    case kNumberTypeFloat8E4M3FN: {
      auto buf = static_cast<float8_e4m3fn *>(data);
      return NewData<T>(buf, size);
    }
    case kNumberTypeFloat8E5M2: {
      auto buf = static_cast<float8_e5m2 *>(data);
      return NewData<T>(buf, size);
    }
    case kNumberTypeHiFloat8: {
      auto buf = static_cast<hifloat8 *>(data);
      return NewData<T>(buf, size);
    }
#ifndef KERNEL_EXECUTOR_ANDROID
    case kNumberTypeBFloat16: {
      auto buf = static_cast<bfloat16 *>(data);
      return NewData<T>(buf, size);
    }
#endif
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
      std::is_same<T, float8_e4m3fn>::value || std::is_same<T, float8_e5m2>::value ||
      std::is_same<T, hifloat8>::value || std::is_same<T, ComplexStorage<float>>::value ||
      std::is_same<T, ComplexStorage<double>>::value || std::is_same<T, bfloat16>::value;
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
      if (!isScalar && ndim_ == 1 && end - start > (kThreshold >> 1) && (i + 1) % linefeedThreshold == 0) {
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
      if ((num > kThreshold && ndim_ > 1) || (num > kThreshold1D && ndim_ == 1)) {
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
  ~TensorDataImpl() override {
    try {
      RemoveOffloadFile();
    } catch (const std::exception &e) {
      MS_LOG(ERROR) << "Exception occurred when cleaning tensor. Error info " << e.what();
    } catch (...) {
      MS_LOG(ERROR) << "Exception occurred when cleaning tensor.";
    }
  }

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

  TensorDataImpl(const ShapeVector &shape, bool ref_mem, void *data) : ndim_(shape.size()), data_size_(SizeOf(shape)) {
    if (!ref_mem) {
      MS_LOG(ERROR) << "For Tensor Ref Data, ref_mem must be true, but got false";
    }
    ref_mem_ = true;
    external_data_ = static_cast<T *>(data);
    data_.reset(nullptr);
  }

  ssize_t size() const override { return static_cast<ssize_t>(data_size_); }

  ssize_t itemsize() const override { return static_cast<ssize_t>(sizeof(T)); }

  ssize_t nbytes() const override { return size() * itemsize(); }

  ssize_t ndim() const override { return static_cast<ssize_t>(ndim_); }

  void *data() override {
    if (ref_mem_) {
      return external_data_;
    }
    if (data_ != nullptr) {
      return data_.get();
    }

    if (data_size_ > INT32_MAX) {
      MS_LOG(WARNING) << "Try to alloca a large memory, size is:" << data_size_ * sizeof(T);
    }
    // Lazy allocation.
    data_ = std::make_unique<T[]>(data_size_);

    // Load data from file
    if (!file_path_.empty()) {
      auto fs = mindspore::system::Env::GetFileSystem();
      MS_EXCEPTION_IF_NULL(fs);
      if (fs->FileExist(file_path_)) {
        auto file = fs->CreateWriteFile(file_path_, "r+");
        MS_EXCEPTION_IF_NULL(file);
        bool success = file->PRead(data_.get(), data_size_ * sizeof(T), 0);
        if (!success) {
          MS_LOG(WARNING) << "Tensor load data from file: " << file_path_ << " failed!";
        }
        if (!file->Close()) {
          MS_LOG(WARNING) << "Close tensor file: " << file_path_ << " failed!";
        }
      } else {
        MS_LOG(WARNING) << "Invalid tensor file path: " << file_path_;
      }
    }
    return data_.get();
  }

  void set_file_path(const std::string &file_path) override { file_path_ = file_path; }

  const std::string file_path() const override { return file_path_; }

  void *const_data() const override {
    // May return nullptr if data not initialized.
    if (ref_mem_) {
      return external_data_;
    }
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
    if (ref_mem_) {
      return "";
    }
    TensorStringifier<T> stringifier{data_.get(), data_size_, ndim_};
    return stringifier.ToString(type, shape, use_comma);
  }

 private:
  void RemoveOffloadFile() {
    if (!file_path_.empty()) {
      TempFileManager::GetInstance().RemoveFile(file_path_);
      TempFileManager::GetInstance().UnRegister(file_path_);
      file_path_ = "";
    }
  }

  size_t ndim_{0};
  size_t data_size_{0};
  std::unique_ptr<T[]> data_;
  std::string file_path_{""};
  T *external_data_{nullptr};
  bool ref_mem_{false};
};

std::string GetTensorDataString(TypeId data_type, const ShapeVector &shape, void *data, size_t size, size_t ndim,
                                bool use_comma);

template <template <class> class ImplClass = TensorDataImpl, typename... Args>
TensorDataPtr MakeTensorData(TypeId data_type, Args &&...args) {
  switch (data_type) {
    case kNumberTypeBool:
      return std::make_shared<ImplClass<bool>>(std::forward<Args>(args)...);
    case kNumberTypeUInt8:
      return std::make_shared<ImplClass<uint8_t>>(std::forward<Args>(args)...);
    case kNumberTypeInt4:
      return std::make_shared<ImplClass<int8_t>>(std::forward<Args>(args)...);
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
    case kNumberTypeFloat8E4M3FN:
      return std::make_shared<ImplClass<float8_e4m3fn>>(std::forward<Args>(args)...);
    case kNumberTypeFloat8E5M2:
      return std::make_shared<ImplClass<float8_e5m2>>(std::forward<Args>(args)...);
    case kNumberTypeHiFloat8:
      return std::make_shared<ImplClass<hifloat8>>(std::forward<Args>(args)...);
#ifndef KERNEL_EXECUTOR_ANDROID
    case kNumberTypeBFloat16:
      return std::make_shared<ImplClass<bfloat16>>(std::forward<Args>(args)...);
#endif
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
  MS_LOG(ERROR) << "Cannot construct Tensor because of unsupported data type: " << TypeIdToString(data_type) << ".";
  return nullptr;
}
}  // namespace mindspore::tensor
#endif  // MINDSPORE_CORE_IR_TENSOR_DATA_H_
