/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#ifndef DATASET_CORE_DATA_TYPE_H_
#define DATASET_CORE_DATA_TYPE_H_

#include <opencv2/core/hal/interface.h>

#include <string>

#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"

#include "dataset/core/constants.h"
#include "dataset/core/pybind_support.h"

namespace py = pybind11;
namespace mindspore {
namespace dataset {

// Class that represents basic data types in DataEngine.
class DataType {
 public:
  enum Type : uint8_t {
    DE_UNKNOWN = 0,
    DE_BOOL,
    DE_INT8,
    DE_UINT8,
    DE_INT16,
    DE_UINT16,
    DE_INT32,
    DE_UINT32,
    DE_INT64,
    DE_UINT64,
    DE_FLOAT16,
    DE_FLOAT32,
    DE_FLOAT64,
    DE_STRING,
    NUM_OF_TYPES
  };

  inline static constexpr uint8_t SIZE_IN_BYTES[] = {0,   // DE_UNKNOWN
                                                     1,   // DE_BOOL
                                                     1,   // DE_INT8
                                                     1,   // DE_UINT8
                                                     2,   // DE_INT16
                                                     2,   // DE_UINT16
                                                     4,   // DE_INT32
                                                     4,   // DE_UINT32
                                                     8,   // DE_INT64
                                                     8,   // DE_UINT64
                                                     2,   // DE_FLOAT16
                                                     4,   // DE_FLOAT32
                                                     8,   // DE_FLOAT64
                                                     0};  // DE_STRING

  inline static const char *TO_STRINGS[] = {"unknown", "bool",  "int8",   "uint8",   "int16",   "uint16",  "int32",
                                            "uint32",  "int64", "uint64", "float16", "float32", "float64", "string"};

  inline static const char *PYBIND_TYPES[] = {"object", "bool",  "int8",   "uint8",   "int16",   "uint16", "int32",
                                              "uint32", "int64", "uint64", "float16", "float32", "double", "bytes"};

  inline static const std::string PYBIND_FORMAT_DESCRIPTOR[] = {"",                                        // DE_UNKNOWN
                                                                py::format_descriptor<bool>::format(),     // DE_BOOL
                                                                py::format_descriptor<int8_t>::format(),   // DE_INT8
                                                                py::format_descriptor<uint8_t>::format(),  // DE_UINT8
                                                                py::format_descriptor<int16_t>::format(),  // DE_INT16
                                                                py::format_descriptor<uint16_t>::format(),  // DE_UINT16
                                                                py::format_descriptor<int32_t>::format(),   // DE_INT32
                                                                py::format_descriptor<uint32_t>::format(),  // DE_UINT32
                                                                py::format_descriptor<int64_t>::format(),   // DE_INT64
                                                                py::format_descriptor<uint64_t>::format(),  // DE_UINT64
                                                                "e",                                      // DE_FLOAT16
                                                                py::format_descriptor<float>::format(),   // DE_FLOAT32
                                                                py::format_descriptor<double>::format(),  // DE_FLOAT64
                                                                "S"};                                     // DE_STRING

  inline static constexpr uint8_t CV_TYPES[] = {kCVInvalidType,   // DE_UNKNOWN
                                                CV_8U,            // DE_BOOL
                                                CV_8S,            // DE_INT8
                                                CV_8U,            // DE_UINT8
                                                CV_16S,           // DE_INT16
                                                CV_16U,           // DE_UINT16
                                                CV_32S,           // DE_INT32
                                                kCVInvalidType,   // DE_UINT32
                                                kCVInvalidType,   // DE_INT64
                                                kCVInvalidType,   // DE_UINT64
                                                CV_16F,           // DE_FLOAT16
                                                CV_32F,           // DE_FLOAT32
                                                CV_64F,           // DE_FLOAT64
                                                kCVInvalidType};  // DE_STRING

  // No arg constructor to create an unknown shape
  DataType() : type_(DE_UNKNOWN) {}

  // Create a type from a given string
  // @param type_str
  explicit DataType(const std::string &type_str);

  // Default destructor
  ~DataType() = default;

  // Create a type from a given enum
  // @param d
  constexpr explicit DataType(Type d) : type_(d) {}

  constexpr bool operator==(const DataType a) const { return type_ == a.type_; }

  constexpr bool operator==(const Type a) const { return type_ == a; }

  constexpr bool operator!=(const DataType a) const { return type_ != a.type_; }

  constexpr bool operator!=(const Type a) const { return type_ != a; }

  // Disable this usage `if(d)` where d is of type DataType
  // @return
  operator bool() = delete;

  // To be used in Switch/case
  // @return
  operator Type() const { return type_; }

  // The number of bytes needed to store one value of this type
  // @return
  uint8_t SizeInBytes() const;

  // Convert from DataType to OpenCV type
  // @return
  uint8_t AsCVType() const;

  // Convert from OpenCV type to DataType
  // @param cv_type
  // @return
  static DataType FromCVType(int cv_type);

  // Returns a string representation of the type
  // @return
  std::string ToString() const;

  // returns true if the template type is the same as the Tensor type_
  // @tparam T
  // @return true or false
  template <typename T>
  bool IsCompatible() const;

  // returns true if the template type is the same as the Tensor type_
  // @tparam T
  // @return true or false
  template <typename T>
  bool IsLooselyCompatible() const;

  // << Stream output operator overload
  // @notes This allows you to print the info using stream operators
  // @param out - reference to the output stream being overloaded
  // @param rO - reference to the DataType to display
  // @return - the output stream must be returned
  friend std::ostream &operator<<(std::ostream &out, const DataType &so) {
    out << so.ToString();
    return out;
  }

  // Convert from DataType to Pybind type
  // @return
  py::dtype AsNumpyType() const;

  // Convert from NP type to DataType
  // @param type
  // @return
  static DataType FromNpType(const py::dtype &type);

  // Convert from NP array to DataType
  // @param py array
  // @return
  static DataType FromNpArray(const py::array &arr);

  // Get the buffer string format of the current type. Used in pybind buffer protocol.
  // @return
  std::string GetPybindFormat() const;

  bool IsSignedInt() const {
    return type_ == DataType::DE_INT8 || type_ == DataType::DE_INT16 || type_ == DataType::DE_INT32 ||
           type_ == DataType::DE_INT64;
  }

  bool IsUnsignedInt() const {
    return type_ == DataType::DE_UINT8 || type_ == DataType::DE_UINT16 || type_ == DataType::DE_UINT32 ||
           type_ == DataType::DE_UINT64;
  }

  bool IsInt() const { return IsSignedInt() || IsUnsignedInt(); }

  bool IsFloat() const {
    return type_ == DataType::DE_FLOAT16 || type_ == DataType::DE_FLOAT32 || type_ == DataType::DE_FLOAT64;
  }

  bool IsBool() const { return type_ == DataType::DE_BOOL; }

  bool IsNumeric() const { return type_ != DataType::DE_STRING; }

  Type value() const { return type_; }

 private:
  Type type_;
};

template <>
inline bool DataType::IsCompatible<bool>() const {
  return type_ == DataType::DE_BOOL;
}

template <>
inline bool DataType::IsCompatible<double>() const {
  return type_ == DataType::DE_FLOAT64;
}

template <>
inline bool DataType::IsCompatible<float>() const {
  return type_ == DataType::DE_FLOAT32;
}

template <>
inline bool DataType::IsCompatible<float16>() const {
  return type_ == DataType::DE_FLOAT16;
}

template <>
inline bool DataType::IsCompatible<int64_t>() const {
  return type_ == DataType::DE_INT64;
}

template <>
inline bool DataType::IsCompatible<uint64_t>() const {
  return type_ == DataType::DE_UINT64;
}

template <>
inline bool DataType::IsCompatible<int32_t>() const {
  return type_ == DataType::DE_INT32;
}

template <>
inline bool DataType::IsCompatible<uint32_t>() const {
  return type_ == DataType::DE_UINT32;
}

template <>
inline bool DataType::IsCompatible<int16_t>() const {
  return type_ == DataType::DE_INT16;
}

template <>
inline bool DataType::IsCompatible<uint16_t>() const {
  return type_ == DataType::DE_UINT16;
}

template <>
inline bool DataType::IsCompatible<int8_t>() const {
  return type_ == DataType::DE_INT8;
}

template <>
inline bool DataType::IsCompatible<uint8_t>() const {
  return type_ == DataType::DE_UINT8;
}

template <>
inline bool DataType::IsCompatible<std::string_view>() const {
  return type_ == DataType::DE_STRING;
}

template <>
inline bool DataType::IsLooselyCompatible<bool>() const {
  return type_ == DataType::DE_BOOL;
}

template <>
inline bool DataType::IsLooselyCompatible<double>() const {
  return type_ == DataType::DE_FLOAT64 || type_ == DataType::DE_FLOAT32;
}

template <>
inline bool DataType::IsLooselyCompatible<float>() const {
  return type_ == DataType::DE_FLOAT32;
}

template <>
inline bool DataType::IsLooselyCompatible<float16>() const {
  return type_ == DataType::DE_FLOAT16;
}

template <>
inline bool DataType::IsLooselyCompatible<int64_t>() const {
  return type_ == DataType::DE_INT64 || type_ == DataType::DE_INT32 || type_ == DataType::DE_INT16 ||
         type_ == DataType::DE_INT8;
}

template <>
inline bool DataType::IsLooselyCompatible<uint64_t>() const {
  return type_ == DataType::DE_UINT64 || type_ == DataType::DE_UINT32 || type_ == DataType::DE_UINT16 ||
         type_ == DataType::DE_UINT8;
}

template <>
inline bool DataType::IsLooselyCompatible<int32_t>() const {
  return type_ == DataType::DE_INT32 || type_ == DataType::DE_INT16 || type_ == DataType::DE_INT8;
}

template <>
inline bool DataType::IsLooselyCompatible<uint32_t>() const {
  return type_ == DataType::DE_UINT32 || type_ == DataType::DE_UINT16 || type_ == DataType::DE_UINT8;
}

template <>
inline bool DataType::IsLooselyCompatible<int16_t>() const {
  return type_ == DataType::DE_INT16 || type_ == DataType::DE_INT8;
}

template <>
inline bool DataType::IsLooselyCompatible<uint16_t>() const {
  return type_ == DataType::DE_UINT16 || type_ == DataType::DE_UINT8;
}

template <>
inline bool DataType::IsLooselyCompatible<int8_t>() const {
  return type_ == DataType::DE_INT8;
}

template <>
inline bool DataType::IsLooselyCompatible<uint8_t>() const {
  return type_ == DataType::DE_UINT8;
}
}  // namespace dataset
}  // namespace mindspore
#endif  // DATASET_CORE_DATA_TYPE_H_
