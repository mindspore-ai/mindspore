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

#include <opencv2/core/hal/interface.h>

#include "utils/log_adapter.h"

#include "dataset/core/constants.h"
#include "dataset/core/pybind_support.h"
#include "dataset/util/de_error.h"

namespace mindspore {
namespace dataset {
uint8_t DataType::SizeInBytes() const {
  switch (type_) {
    case DataType::DE_BOOL:
    case DataType::DE_INT8:
    case DataType::DE_UINT8:
      return 1;
    case DataType::DE_INT16:
    case DataType::DE_UINT16:
    case DataType::DE_FLOAT16:
      return 2;
    case DataType::DE_INT32:
    case DataType::DE_UINT32:
    case DataType::DE_FLOAT32:
      return 4;
    case DataType::DE_INT64:
    case DataType::DE_UINT64:
    case DataType::DE_FLOAT64:
      return 8;
    default:
      return 0;
  }
}

py::dtype DataType::AsNumpyType() const {
  std::string s;
  switch (type_) {
    case DataType::DE_BOOL:
      s = "bool";
      break;
    case DataType::DE_INT8:
      s = "int8";
      break;
    case DataType::DE_UINT8:
      s = "uint8";
      break;
    case DataType::DE_INT16:
      s = "int16";
      break;
    case DataType::DE_UINT16:
      s = "uint16";
      break;
    case DataType::DE_INT32:
      s = "int32";
      break;
    case DataType::DE_UINT32:
      s = "uint32";
      break;
    case DataType::DE_INT64:
      s = "int64";
      break;
    case DataType::DE_UINT64:
      s = "uint64";
      break;
    case DataType::DE_FLOAT16:
      s = "float16";
      break;
    case DataType::DE_FLOAT32:
      s = "float32";
      break;
    case DataType::DE_FLOAT64:
      s = "double";
      break;
    case DataType::DE_UNKNOWN:
      s = "unknown";
      break;
    default:
      s = "unknown";
      break;
  }
  return py::dtype(s);
}

uint8_t DataType::AsCVType() const {
  switch (type_) {
    case DataType::DE_BOOL:
      return CV_8U;
    case DataType::DE_INT8:
      return CV_8S;
    case DataType::DE_UINT8:
      return CV_8U;
    case DataType::DE_INT16:
      return CV_16S;
    case DataType::DE_UINT16:
      return CV_16U;
    case DataType::DE_INT32:
      return CV_32S;
    case DataType::DE_FLOAT16:
      return CV_16F;
    case DataType::DE_FLOAT32:
      return CV_32F;
    case DataType::DE_FLOAT64:
      return CV_64F;
    case DataType::DE_UINT32:
    case DataType::DE_INT64:
    case DataType::DE_UINT64:
    default:
      MS_LOG(ERROR) << "Cannot convert to OpenCV type. Return invalid type!";
      return kCVInvalidType;
  }
}

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
  else
    type_ = DE_UNKNOWN;
}

std::string DataType::ToString() const {
  switch (type_) {
    case DataType::DE_BOOL:
      return "bool";
    case DataType::DE_INT8:
      return "int8";
    case DataType::DE_UINT8:
      return "uint8";
    case DataType::DE_INT16:
      return "int16";
    case DataType::DE_UINT16:
      return "uint16";
    case DataType::DE_INT32:
      return "int32";
    case DataType::DE_UINT32:
      return "uint32";
    case DataType::DE_INT64:
      return "int64";
    case DataType::DE_UINT64:
      return "uint64";
    case DataType::DE_FLOAT16:
      return "float16";
    case DataType::DE_FLOAT32:
      return "float32";
    case DataType::DE_FLOAT64:
      return "float64";
    case DataType::DE_UNKNOWN:
      return "unknown";
    default:
      return "unknown";
  }
}

DataType DataType::FromNpType(const py::dtype &type) {
  if (type.is(py::dtype("bool"))) {
    return DataType(DataType::DE_BOOL);
  } else if (type.is(py::dtype("int8"))) {
    return DataType(DataType::DE_INT8);
  } else if (type.is(py::dtype("uint8"))) {
    return DataType(DataType::DE_UINT8);
  } else if (type.is(py::dtype("int16"))) {
    return DataType(DataType::DE_INT16);
  } else if (type.is(py::dtype("uint16"))) {
    return DataType(DataType::DE_UINT16);
  } else if (type.is(py::dtype("int32"))) {
    return DataType(DataType::DE_INT32);
  } else if (type.is(py::dtype("uint32"))) {
    return DataType(DataType::DE_UINT32);
  } else if (type.is(py::dtype("int64"))) {
    return DataType(DataType::DE_INT64);
  } else if (type.is(py::dtype("uint64"))) {
    return DataType(DataType::DE_UINT64);
  } else if (type.is(py::dtype("float16"))) {
    return DataType(DataType::DE_FLOAT16);
  } else if (type.is(py::dtype("float32"))) {
    return DataType(DataType::DE_FLOAT32);
  } else if (type.is(py::dtype("double"))) {
    return DataType(DataType::DE_FLOAT64);
  } else {
    MS_LOG(ERROR) << "Cannot convert from numpy type. Unknown data type is returned!";
    return DataType(DataType::DE_UNKNOWN);
  }
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
  } else {
    MS_LOG(ERROR) << "Cannot convert from numpy type. Unknown data type is returned!";
    return DataType(DataType::DE_UNKNOWN);
  }
}

std::string DataType::GetPybindFormat() const {
  switch (type_) {
    case DataType::DE_BOOL:
      return py::format_descriptor<bool>::format();
    case DataType::DE_INT8:
      return py::format_descriptor<int8_t>::format();
    case DataType::DE_UINT8:
      return py::format_descriptor<uint8_t>::format();
    case DataType::DE_INT16:
      return py::format_descriptor<int16_t>::format();
    case DataType::DE_UINT16:
      return py::format_descriptor<uint16_t>::format();
    case DataType::DE_INT32:
      return py::format_descriptor<int32_t>::format();
    case DataType::DE_UINT32:
      return py::format_descriptor<uint32_t>::format();
    case DataType::DE_INT64:
      return py::format_descriptor<int64_t>::format();
    case DataType::DE_UINT64:
      return py::format_descriptor<uint64_t>::format();
    case DataType::DE_FLOAT16:
      // Eigen 3.3.7 doesn't support py::format_descriptor<Eigen::half>::format()
      return "e";
    case DataType::DE_FLOAT32:
      return py::format_descriptor<float>::format();
    case DataType::DE_FLOAT64:
      return py::format_descriptor<double>::format();
    default:
      MS_LOG(ERROR) << "Cannot convert from data type to pybind format descriptor!";
      return "";
  }
}
}  // namespace dataset
}  // namespace mindspore
