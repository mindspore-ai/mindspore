/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_INCLUDE_COMMON_PYNATIVE_PARSER_H
#define MINDSPORE_CCSRC_INCLUDE_COMMON_PYNATIVE_PARSER_H

#include <deque>
#include <memory>
#include <utility>
#include <vector>
#include <string>
#include <unordered_map>
#include <Python.h>

#include "pybind11/pybind11.h"
#include "include/utils/visible.h"
#include "ir/anf.h"
#include "ir/dtype/op_dtype.h"
#include "ir/tensor.h"

namespace py = pybind11;
namespace mindspore {
namespace py_parse {
COMMON_EXPORT ValuePtr ConvertPyObjectTensor(PyObject *obj);
COMMON_EXPORT ValuePtr ConvertTensor(const py::object &obj);
COMMON_EXPORT tensor::TensorPtr ConvertTensorValue(const py::object &obj);
COMMON_EXPORT tensor::TensorPtr ConvertPyObjectTensorValue(PyObject *obj);

using OpDefConvertFunc = ValuePtr (*)(const py::object &);
using ConverterMap = std::unordered_map<int32_t, OpDefConvertFunc>;
COMMON_EXPORT const ConverterMap &GetConverters();
COMMON_EXPORT void ReportGetConverterError(int32_t dtype);
COMMON_EXPORT OpDefConvertFunc GetConverterByType(int32_t dtype);

constexpr int32_t kTypeShiftBits = 16;
constexpr auto kDstMask = (1 << kTypeShiftBits) - 1;
inline int32_t CombineTypesForTypeCast(const mindspore::ops::OP_DTYPE &src, const mindspore::ops::OP_DTYPE &dst) {
  return static_cast<int32_t>((static_cast<uint32_t>(src) << kTypeShiftBits) | static_cast<uint32_t>(dst));
}

template <typename TS, typename TD, OpDefConvertFunc func>
ValuePtr ConvertSequence(const py::object &obj) {
  if (!py::isinstance<TS>(obj)) {
    return nullptr;
  }
  auto seq = obj.cast<TS>();
  std::vector<ValuePtr> value_list;
  for (size_t it = 0; it < seq.size(); ++it) {
    auto out = func(seq[it]);
    if (out == nullptr) {
      return nullptr;
    }
    value_list.emplace_back(out);
  }
  return std::make_shared<TD>(value_list);
}

template <typename T, OpDefConvertFunc func>
ValuePtr ConvertSingleElementToSequence(const py::object &obj) {
  auto value = func(obj);
  if (value == nullptr) {
    return nullptr;
  }
  std::vector<ValuePtr> value_list{value};
  return std::make_shared<T>(std::move(value_list));
}

// convert PyObject to c++ type
COMMON_EXPORT bool ParseUtilsCheckInt(PyObject *obj, bool allow_scalar_tensor = true);
COMMON_EXPORT bool ParseUtilsCheckFloat(PyObject *obj, bool allow_scalar_tensor = true);
COMMON_EXPORT bool ParseUtilsCheckBool(PyObject *obj, bool allow_scalar_tensor = true);
COMMON_EXPORT bool ParseUtilsCheckScalar(PyObject *obj, bool allow_scalar_tensor = true);
COMMON_EXPORT bool IsGeneralizedInt(PyObject *obj);
COMMON_EXPORT std::optional<int64_t> ParseUtilsConvertInt(PyObject *obj);
COMMON_EXPORT std::optional<double> ParseUtilsConvertFloat(PyObject *obj);
COMMON_EXPORT std::optional<bool> ParseUtilsConvertBool(PyObject *obj);
COMMON_EXPORT std::optional<int64_t> ConvertGeneralizedIntToBasicInt(PyObject *obj);
}  // namespace py_parse
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_INCLUDE_COMMON_PYNATIVE_PARSER_H
