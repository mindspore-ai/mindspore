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

#include <algorithm>
#include <vector>
#include <tuple>
#include <string>
#include <memory>
#include "pybind11/pybind11.h"
#include "ops/op_def.h"
#include "include/op_enum.h"
#include "frontend/jit/ps/parse/data_converter.h"
#include "utils/anf_utils.h"
#include "include/utils/tensor_py.h"
#include "include/utils/pynative/py_parse.h"
#include "include/pynative/forward/pyboost/arg_handler_py.h"

namespace py = pybind11;

namespace mindspore {

namespace pynative {

namespace {
using OP_DTYPE = mindspore::ops::OP_DTYPE;

template <typename T, typename U>
std::shared_ptr<U> PyCast(PyObject *obj);

template <>
std::shared_ptr<Int64Imm> PyCast<int64_t, Int64Imm>(PyObject *obj) {
  return std::make_shared<Int64Imm>(static_cast<int64_t>(PyLong_AsLongLong(obj)));
}

template <typename... Args>
PyObject *PackToPyList(Args &&...args) {
  auto list = PyList_New(sizeof...(args));
  size_t i{0};
  for (auto &arg : {args...}) {
    Py_XINCREF(arg);
    PyList_SetItem(list, i++, arg);
  }
  return list;
}

}  // namespace

Int64ImmPtr ConvertInt(PyObject *obj) {
  // bool is also an instance of py::int_
  if (PyBool_Check(obj) || !PyLong_Check(obj)) {
    return nullptr;
  }
  return PyCast<int64_t, Int64Imm>(obj);
}

Int64ImmPtr ToDtypePy(const py::object &obj) {
  auto convert = ConvertInt(obj.ptr());
  if (convert != nullptr) {
    return convert;
  }
  if (py::isinstance<mindspore::Type>(obj)) {
    TypePtr type = py::cast<mindspore::TypePtr>(obj);
    return std::make_shared<Int64Imm>(static_cast<int>(type->type_id()));
  }
  return nullptr;
}

PyObject *DtypeToTypeId(const std::string &op_name, const std::string &arg_name, PyObject *obj) {
  if (obj == Py_None) {
    return Py_None;
  }
  py::object py_obj = py::reinterpret_borrow<py::object>(obj);
  if (py::isinstance<mindspore::Type>(py_obj)) {
    auto dtype = ToDtypePy(py_obj);
    return dtype ? PyLong_FromLong(dtype->value()) : Py_None;
  }
  if (obj == reinterpret_cast<PyObject *>(&PyBool_Type)) {
    auto ms_bool_type = mindspore::Bool();
    return PyLong_FromLong(ToDtypePy(py::cast(ms_bool_type))->value());
  }
  MS_LOG(EXCEPTION) << "For '" << op_name << "', the input '" << arg_name
                    << "' should be one of ['mindspore dtype', 'bool'], but got " << obj << ".";
  return Py_None;
}

PyObject *StrToEnum(const std::string &op_name, const std::string &arg_name, PyObject *obj) {
  if (obj == Py_None) {
    return Py_None;
  }
  if (!PyUnicode_Check(obj)) {
    PyObject *obj_type = PyObject_Str(PyObject_Type(obj));
    const char *type_str = PyUnicode_AsUTF8(obj_type);
    MS_LOG(EXCEPTION) << "For '" << op_name << "', the input '" << arg_name << "' should be a str, but got " << type_str
                      << ".";
    Py_DECREF(obj_type);
  }
  auto string_value = PyUnicode_AsUTF8(obj);
  auto enum_value = mindspore::ops::StringToEnumImpl(op_name, arg_name, string_value);
  return PyLong_FromLong(enum_value);
}

PyObject *ToPair(const std::string &op_name, const std::string &arg_name, PyObject *arg_val) {
  if (PyLong_Check(arg_val) || PyFloat_Check(arg_val)) {
    int value = static_cast<int>(PyLong_AsLong(arg_val));
    return PackToPyList(PyLong_FromLong(value), PyLong_FromLong(value));
  }
  if (PyList_Check(arg_val)) {
    PyObject *values = PyList_New(0);
    Py_ssize_t arg_size = PyList_Size(arg_val);
    for (Py_ssize_t i = 0; i < arg_size; ++i) {
      PyObject *item = PyList_GetItem(arg_val, i);
      PyList_Append(values, item);
    }
    return values;
  } else if (PyTuple_Check(arg_val)) {
    PyObject *values = PyList_New(0);
    Py_ssize_t arg_size = PyTuple_Size(arg_val);
    for (Py_ssize_t i = 0; i < arg_size; ++i) {
      PyObject *item = PyTuple_GetItem(arg_val, i);
      PyList_Append(values, item);
    }
    return values;
  }
  PyObject *arg_str_py = PyObject_Str(arg_val);
  const char *arg_str = PyUnicode_AsUTF8(arg_str_py);
  Py_DECREF(arg_str_py);
  MS_LOG(EXCEPTION) << "For '" << op_name << "', the value of '" << arg_name << "' is invalid: '" << arg_str << ".";
}

PyObject *To2dPaddings(const std::string &op_name, const std::string &arg_name, PyObject *pad) {
  if (PyLong_Check(pad)) {
    int value = static_cast<int>(PyLong_AsLong(pad));
    return PackToPyList(PyLong_FromLong(value), PyLong_FromLong(value));
  }
  if (PyList_Check(pad)) {
    PyObject *values = PyList_New(0);
    Py_ssize_t arg_size = PyList_Size(pad);
    for (Py_ssize_t i = 0; i < arg_size; ++i) {
      PyObject *item = PyList_GetItem(pad, i);
      PyList_Append(values, item);
    }
    return values;
  } else if (PyTuple_Check(pad)) {
    PyObject *values = PyList_New(0);
    Py_ssize_t arg_size = PyTuple_Size(pad);
    for (Py_ssize_t i = 0; i < arg_size; ++i) {
      PyObject *item = PyTuple_GetItem(pad, i);
      PyList_Append(values, item);
    }
    return values;
  }
  PyObject *pad_str_py = PyObject_Str(pad);
  const char *pad_str = PyUnicode_AsUTF8(pad_str_py);
  Py_DECREF(pad_str_py);
  MS_LOG(EXCEPTION) << "For '" << op_name << "', the value of '" << arg_name << "' is invalid: '" << pad_str << ".";
}

PyObject *ToVector(const std::string &op_name, const std::string &arg_name, PyObject *arg) {
  if (PyLong_Check(arg)) {
    int value = static_cast<int>(PyLong_AsLong(arg));
    return PackToPyList(PyLong_FromLong(value), PyLong_FromLong(value));
  }
  // compatible with input format like (N,C,H,W), only choose H and W
  int channels_num = 4;
  if (PyList_Check(arg)) {
    Py_ssize_t arg_size = PyList_Size(arg);
    if (arg_size == channels_num) {
      PyObject *item2 = PyList_GetItem(arg, 2);
      PyObject *item3 = PyList_GetItem(arg, 3);
      return PackToPyList(item2, item3);
    }
    PyObject *values = PyList_New(0);
    for (Py_ssize_t i = 0; i < arg_size; ++i) {
      PyObject *item = PyList_GetItem(arg, i);
      PyList_Append(values, item);
    }
    return values;
  } else if (PyTuple_Check(arg)) {
    Py_ssize_t arg_size = PyTuple_Size(arg);
    if (arg_size == channels_num) {
      PyObject *item2 = PyTuple_GetItem(arg, 2);
      PyObject *item3 = PyTuple_GetItem(arg, 3);
      return PackToPyList(item2, item3);
    }
    PyObject *values = PyList_New(0);
    for (Py_ssize_t i = 0; i < arg_size; ++i) {
      PyObject *item = PyTuple_GetItem(arg, i);
      PyList_Append(values, item);
    }
    return values;
  }
  PyObject *arg_str_py = PyObject_Str(arg);
  const char *arg_str = PyUnicode_AsUTF8(arg_str_py);
  Py_DECREF(arg_str_py);
  MS_LOG(EXCEPTION) << "For '" << op_name << "', the value of '" << arg_name << "' is invalid: '" << arg_str << ".";
}

PyObject *ToKernelSize(const std::string &op_name, const std::string &arg_name, PyObject *kernel_size) {
  return ToVector(op_name, arg_name, kernel_size);
}

PyObject *ToStrides(const std::string &op_name, const std::string &arg_name, PyObject *stride) {
  return ToVector(op_name, arg_name, stride);
}

PyObject *ToDilations(const std::string &op_name, const std::string &arg_name, PyObject *dilation) {
  return ToVector(op_name, arg_name, dilation);
}

PyObject *ToOutputPadding(const std::string &op_name, const std::string &arg_name, PyObject *output_padding) {
  return ToVector(op_name, arg_name, output_padding);
}

PyObject *ToRates(const std::string &op_name, const std::string &arg_name, PyObject *rates) {
  return ToVector(op_name, arg_name, rates);
}

PyObject *NormalizeIntSequence(const std::string &op_name, const std::string &arg_name, PyObject *arg) {
  py::handle arg_handle(arg);
  const bool is_list = py::isinstance<py::list>(arg_handle);
  const bool is_tuple = py::isinstance<py::tuple>(arg_handle);
  if (!is_list && !is_tuple) {
    return arg;
  }
  auto items = py::cast<std::vector<py::object>>(arg_handle);
  auto convert_type = py_parse::CombineTypesForTypeCast(ops::DT_TENSOR, ops::DT_INT);
  auto convert_func = py_parse::GetConverterByType(convert_type);
  MS_EXCEPTION_IF_NULL(convert_func);
  const auto normalize_item = [&](const py::object &item) -> py::object {
    if (py_parse::IsGeneralizedInt(item.ptr())) {
      return item;
    }
    ValuePtr value = convert_func(item);
    if (!value) {
      MS_EXCEPTION(ValueError) << "For '" << op_name << "', '" << arg_name << "' contain non-integer element: '"
                               << py::str(item).cast<std::string>() << "'.";
    }
    return py::cast(AnfUtils::GetIntValue(value));
  };

  const auto fill = [&](auto *out) {
    size_t i = 0;
    for (const auto &item : items) {
      (*out)[i] = normalize_item(item);
      i++;
    }
  };

  if (is_tuple) {
    py::tuple int_tuple(items.size());
    fill(&int_tuple);
    Py_INCREF(int_tuple.ptr());
    return int_tuple.ptr();
  }

  py::list int_list(items.size());
  fill(&int_list);
  Py_INCREF(int_list.ptr());
  return int_list.ptr();
}

PyObject *ScalarTensorToScalar(const std::string &op_name, const std::string &arg_name, PyObject *arg) {
  py::handle arg_handle(arg);
  if (!tensor::IsTensorPy(arg_handle)) {
    return arg;
  }

  auto tensor = py_parse::ConvertTensorValue(py::reinterpret_borrow<py::object>(arg_handle));
  if (tensor) {
    if (tensor->DataDim() != 0) {
      MS_EXCEPTION(ValueError) << "For '" << op_name << "', '" << arg_name << "' is not a scalar: '"
                               << py::str(arg_handle).cast<std::string>() << "'.";
    }
    auto convert_type = py_parse::CombineTypesForTypeCast(ops::DT_TENSOR, ops::DT_NUMBER);
    auto convert_func = py_parse::GetConverterByType(convert_type);
    ValuePtr value = convert_func(py::reinterpret_borrow<py::object>(arg_handle));
    if (!value) {
      MS_EXCEPTION(TypeError) << "For '" << op_name << "', '" << arg_name << "' is not a scalar: '"
                              << py::str(arg_handle).cast<std::string>() << "'.";
    }
    if (value->isa<Int64Imm>()) {
      return PyLong_FromLong(GetValue<int64_t>(value));
    } else if (value->isa<Int32Imm>()) {
      return PyLong_FromLong(GetValue<int32_t>(value));
    } else if (value->isa<FP32Imm>()) {
      return PyFloat_FromDouble(GetValue<float>(value));
    } else if (value->isa<FP64Imm>()) {
      return PyFloat_FromDouble(GetValue<double>(value));
    } else if (value->isa<BoolImm>()) {
      return PyBool_FromLong(GetValue<bool>(value));
    }
  }
  return arg;
}

PyObject *ScalarTensorToInt(const std::string &op_name, const std::string &arg_name, PyObject *arg) {
  return ScalarTensorToScalar(op_name, arg_name, arg);
}

PyObject *ScalarTensorToFloat(const std::string &op_name, const std::string &arg_name, PyObject *arg) {
  return ScalarTensorToScalar(op_name, arg_name, arg);
}
}  // namespace pynative
}  // namespace mindspore
