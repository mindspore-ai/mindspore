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

  struct TypeInfo {
    const char *name_;                          // name to be represent the type while printing
    const uint8_t sizeInBytes_;                 // number of bytes needed for this type
    const char *pybindType_;                    //  Python matching type, used in get_output_types
    const std::string pybindFormatDescriptor_;  // pybind format used for numpy types
    const uint8_t cvType_;                      // OpenCv matching type
  };

  static inline const TypeInfo kTypeInfo[] = {
    // name, sizeInBytes, pybindTypem formatDescriptor, openCV
    {"unknown", 0, "object", "", kCVInvalidType},                                        // DE_UNKNOWN
    {"bool", 1, "bool", py::format_descriptor<bool>::format(), CV_8U},                   // DE_BOOL
    {"int8", 1, "int8", py::format_descriptor<int8_t>::format(), CV_8S},                 // DE_INT8
    {"uint8", 1, "uint8", py::format_descriptor<uint8_t>::format(), CV_8U},              // DE_UINT8
    {"int16", 2, "int16", py::format_descriptor<int16_t>::format(), CV_16S},             // DE_INT16
    {"uint16", 2, "uint16", py::format_descriptor<uint16_t>::format(), CV_16U},          // DE_UINT16
    {"int32", 4, "int32", py::format_descriptor<int32_t>::format(), CV_32S},             // DE_INT32
    {"uint32", 4, "uint32", py::format_descriptor<uint32_t>::format(), kCVInvalidType},  // DE_UINT32
    {"int64", 8, "int64", py::format_descriptor<int64_t>::format(), kCVInvalidType},     // DE_INT64
    {"uint64", 8, "uint64", py::format_descriptor<uint64_t>::format(), kCVInvalidType},  // DE_UINT64
    {"float16", 2, "float16", "e", CV_16F},                                              // DE_FLOAT16
    {"float32", 4, "float32", py::format_descriptor<float>::format(), CV_32F},           // DE_FLOAT32
    {"float64", 8, "double", py::format_descriptor<double>::format(), CV_64F},           // DE_FLOAT64
    {"string", 0, "bytes", "S", kCVInvalidType}                                          // DE_STRING
  };

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
