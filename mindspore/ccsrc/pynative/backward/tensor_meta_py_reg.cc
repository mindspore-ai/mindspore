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

#include <memory>
#include "pynative/backward/tensor_meta_py.h"
#include "include/utils/exception.h"
#include "pybind_api/pynative/pynative_api.h"
#include "ir/dtype/type.h"

namespace mindspore::pynative::autograd {
namespace py = pybind11;
namespace {
void TensorMeta_PyDealloc(PyObject *self) {
  reinterpret_cast<TensorMetaPy *>(self)->cdata.~TensorMeta();
  Py_TYPE(self)->tp_free(self);
}

PyObject *PyTensorMeta_str(PyObject *self) {
  const auto &tensor_meta = reinterpret_cast<TensorMetaPy *>(self)->cdata;
  const auto &shape = tensor_meta.shape();
  const auto &type_str = tensor_meta.dtype()->ToString();
  std::string shape_str = "shape=[";
  for (size_t i = 0; i < shape.size(); i++) {
    if (i > 0) {
      shape_str += ", ";
    }
    shape_str += std::to_string(shape[i]);
  }
  shape_str += "]";
  return PyUnicode_FromFormat("<TensorMeta, %s, %s>", shape_str.c_str(), type_str.c_str());
}

PyObject *PyTensorMeta_repr(PyObject *self) { return PyTensorMeta_str(self); }

PyObject *PyTensorMeta_get_shape(PyObject *self, void *) {
  HANDLE_MS_EXCEPTION
  const auto &tensor_meta = reinterpret_cast<TensorMetaPy *>(self)->cdata;
  const auto &shape = tensor_meta.shape();
  PyObject *output = PyTuple_New(shape.size());
  for (size_t i = 0; i < shape.size(); i++) {
    PyTuple_SetItem(output, i, PyLong_FromLongLong(shape[i]));
  }
  return output;
  HANDLE_MS_EXCEPTION_END
}

PyObject *PyTensorMeta_get_dtype(PyObject *self, void *) {
  HANDLE_MS_EXCEPTION
  const auto &tensor_meta = reinterpret_cast<TensorMetaPy *>(self)->cdata;
  const auto &dtype = tensor_meta.dtype();
  if (dtype == nullptr) {
    Py_RETURN_NONE;
  }
  py::object dtype_obj = py::cast(dtype);
  Py_INCREF(dtype_obj.ptr());
  return dtype_obj.ptr();
  HANDLE_MS_EXCEPTION_END
}

PyObject *PyTensorMeta_get_strides(PyObject *self, void *) {
  HANDLE_MS_EXCEPTION
  const auto &tensor_meta = reinterpret_cast<TensorMetaPy *>(self)->cdata;
  const auto &strides = tensor_meta.strides();
  PyObject *output = PyTuple_New(strides.size());
  for (size_t i = 0; i < strides.size(); i++) {
    PyTuple_SetItem(output, i, PyLong_FromLongLong(strides[i]));
  }
  return output;
  HANDLE_MS_EXCEPTION_END
}

PyObject *PyTensorMeta_get_storage_offset(PyObject *self, void *) {
  HANDLE_MS_EXCEPTION
  const auto &tensor_meta = reinterpret_cast<TensorMetaPy *>(self)->cdata;
  return PyLong_FromSize_t(tensor_meta.storage_offset());
  HANDLE_MS_EXCEPTION_END
}

PyGetSetDef PyTensorMeta_getseters[] = {
  {"shape", (getter)PyTensorMeta_get_shape, nullptr, "tensor shape", nullptr},
  {"dtype", (getter)PyTensorMeta_get_dtype, nullptr, "tensor dtype", nullptr},
  {"strides", (getter)PyTensorMeta_get_strides, nullptr, "tensor strides", nullptr},
  {"storage_offset", (getter)PyTensorMeta_get_storage_offset, nullptr, "tensor storage offset", nullptr},
  {nullptr}};

PyTypeObject TensorMetaPyType = {
  PyVarObject_HEAD_INIT(nullptr, 0) "TensorMeta", /* tp_name */
  sizeof(TensorMetaPy),                           /* tp_basicsize */
  0,                                              /* tp_itemsize */
  TensorMeta_PyDealloc,                           /* tp_dealloc */
  0,                                              /* tp_vectorcall_offset */
  nullptr,                                        /* tp_getattr */
  nullptr,                                        /* tp_setattr */
  nullptr,                                        /* tp_reserved */
  PyTensorMeta_repr,                              /* tp_repr */
  nullptr,                                        /* tp_as_number */
  nullptr,                                        /* tp_as_sequence */
  nullptr,                                        /* tp_as_mapping */
  nullptr,                                        /* tp_hash  */
  nullptr,                                        /* tp_call */
  PyTensorMeta_str,                               /* tp_str */
  nullptr,                                        /* tp_getattro */
  nullptr,                                        /* tp_setattro */
  nullptr,                                        /* tp_as_buffer */
  Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,       /* tp_flags */
  nullptr,                                        /* tp_doc */
  nullptr,                                        /* tp_traverse */
  nullptr,                                        /* tp_clear */
  nullptr,                                        /* tp_richcompare */
  0,                                              /* tp_weaklistoffset */
  nullptr,                                        /* tp_iter */
  nullptr,                                        /* tp_iternext */
  nullptr,                                        /* tp_methods */
  nullptr,                                        /* tp_members */
  PyTensorMeta_getseters,                         /* tp_getset */
  nullptr,                                        /* tp_base */
  nullptr,                                        /* tp_dict */
  nullptr,                                        /* tp_descr_get */
  nullptr,                                        /* tp_descr_set */
  0,                                              /* tp_dictoffset */
  nullptr,                                        /* tp_init */
  nullptr,                                        /* tp_alloc */
  nullptr,                                        /* tp_new */
};
}  // namespace

bool IsTensorMetaPy(const py::handle &obj) {
  if (!obj.check()) {
    return false;
  }
  PyObject *raw_ptr = obj.ptr();
  return PyObject_TypeCheck(raw_ptr, &TensorMetaPyType);
}

PyObject *WrapTensorMeta(const TensorMeta &tensor_meta) {
  PyTypeObject *type = &TensorMetaPyType;
  PyObject *obj = type->tp_alloc(type, 0);
  auto tm = reinterpret_cast<TensorMetaPy *>(obj);
  if (tm == nullptr) {
    PyErr_SetString(PyExc_RuntimeError, "Failed to create TensorMetaPy object");
    return nullptr;
  }
  new (&tm->cdata) TensorMeta(tensor_meta);
  return obj;
}

namespace py = pybind11;
void RegTensorMeta(py::module *m) {
  if (PyType_Ready(&TensorMetaPyType) < 0) {
    return;
  }
  Py_INCREF(&TensorMetaPyType);
  m->add_object("TensorMetaPy", reinterpret_cast<PyObject *>(&TensorMetaPyType));
}
}  // namespace mindspore::pynative::autograd
