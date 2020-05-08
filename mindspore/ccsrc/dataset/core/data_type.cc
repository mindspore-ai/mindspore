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
#include "dataset/core/data_type.h"

#include "utils/log_adapter.h"

#include "dataset/core/pybind_support.h"
#include "dataset/util/de_error.h"

namespace mindspore {
namespace dataset {

uint8_t DataType::SizeInBytes() const {
  if (type_ < DataType::NUM_OF_TYPES)
    return SIZE_IN_BYTES[type_];
  else
    return 0;
}

py::dtype DataType::AsNumpyType() const {
  if (type_ < DataType::NUM_OF_TYPES)
    return py::dtype(PYBIND_TYPES[type_]);
  else
    return py::dtype("unknown");
}

uint8_t DataType::AsCVType() const {
  uint8_t res = kCVInvalidType;
  if (type_ < DataType::NUM_OF_TYPES) {
    res = CV_TYPES[type_];
  }

  if (res == kCVInvalidType) {
    MS_LOG(ERROR) << "Cannot convert to OpenCV type. Return invalid type!";
  }

  return res;
}  // namespace dataset

DataType DataType::FromCVType(int cv_type) {
  auto depth = static_cast<uchar>(cv_type) & static_cast<uchar>(CV_MAT_DEPTH_MASK);
  switch (depth) {
    case CV_8S:
      return DataType(DataType::DE_INT8);
    case CV_8U:
      return DataType(DataType::DE_UINT8);
    case CV_16S:
      return DataType(DataType::DE_INT16);
    case CV_16U:
      return DataType(DataType::DE_UINT16);
    case CV_32S:
      return DataType(DataType::DE_INT32);
    case CV_16F:
      return DataType(DataType::DE_FLOAT16);
    case CV_32F:
      return DataType(DataType::DE_FLOAT32);
    case CV_64F:
      return DataType(DataType::DE_FLOAT64);
    default:
      MS_LOG(ERROR) << "Cannot convert from OpenCV type, unknown CV type. Unknown data type is returned!";
      return DataType(DataType::DE_UNKNOWN);
  }
}

DataType::DataType(const std::string &type_str) {
  if (type_str == "bool")
    type_ = DE_BOOL;
  else if (type_str == "int8")
    type_ = DE_INT8;
  else if (type_str == "uint8")
    type_ = DE_UINT8;
  else if (type_str == "int16")
    type_ = DE_INT16;
  else if (type_str == "uint16")
    type_ = DE_UINT16;
  else if (type_str == "int32")
    type_ = DE_INT32;
  else if (type_str == "uint32")
    type_ = DE_UINT32;
  else if (type_str == "int64")
    type_ = DE_INT64;
  else if (type_str == "uint64")
    type_ = DE_UINT64;
  else if (type_str == "float16")
    type_ = DE_FLOAT16;
  else if (type_str == "float32")
    type_ = DE_FLOAT32;
  else if (type_str == "float64")
    type_ = DE_FLOAT64;
  else if (type_str == "string")
    type_ = DE_STRING;
  else
    type_ = DE_UNKNOWN;
}

std::string DataType::ToString() const {
  if (type_ < DataType::NUM_OF_TYPES)
    return TO_STRINGS[type_];
  else
    return "unknown";
}

DataType DataType::FromNpArray(const py::array &arr) {
  if (py::isinstance<py::array_t<bool>>(arr)) {
    return DataType(DataType::DE_BOOL);
  } else if (py::isinstance<py::array_t<std::int8_t>>(arr)) {
    return DataType(DataType::DE_INT8);
  } else if (py::isinstance<py::array_t<std::uint8_t>>(arr)) {
    return DataType(DataType::DE_UINT8);
  } else if (py::isinstance<py::array_t<std::int16_t>>(arr)) {
    return DataType(DataType::DE_INT16);
  } else if (py::isinstance<py::array_t<std::uint16_t>>(arr)) {
    return DataType(DataType::DE_UINT16);
  } else if (py::isinstance<py::array_t<std::int32_t>>(arr)) {
    return DataType(DataType::DE_INT32);
  } else if (py::isinstance<py::array_t<std::uint32_t>>(arr)) {
    return DataType(DataType::DE_UINT32);
  } else if (py::isinstance<py::array_t<std::int64_t>>(arr)) {
    return DataType(DataType::DE_INT64);
  } else if (py::isinstance<py::array_t<std::uint64_t>>(arr)) {
    return DataType(DataType::DE_UINT64);
  } else if (py::isinstance<py::array_t<float16>>(arr)) {
    return DataType(DataType::DE_FLOAT16);
  } else if (py::isinstance<py::array_t<std::float_t>>(arr)) {
    return DataType(DataType::DE_FLOAT32);
  } else if (py::isinstance<py::array_t<std::double_t>>(arr)) {
    return DataType(DataType::DE_FLOAT64);
  } else if (arr.dtype().kind() == 'S') {
    return DataType(DataType::DE_STRING);
  } else {
    MS_LOG(ERROR) << "Cannot convert from numpy type. Unknown data type is returned!";
    return DataType(DataType::DE_UNKNOWN);
  }
}

std::string DataType::GetPybindFormat() const {
  std::string res;
  if (type_ < DataType::NUM_OF_TYPES) {
    res = PYBIND_FORMAT_DESCRIPTOR[type_];
  }

  if (res.empty()) {
    MS_LOG(ERROR) << "Cannot convert from data type to pybind format descriptor!";
  }
  return res;
}

}  // namespace dataset
}  // namespace mindspore
