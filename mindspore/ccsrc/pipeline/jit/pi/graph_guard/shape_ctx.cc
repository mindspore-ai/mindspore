/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include "pipeline/jit/pi/graph_guard/shape_ctx.h"
#include <algorithm>
#include <map>
#include "ir/tensor.h"

namespace py = pybind11;

namespace mindspore {
namespace pijit {

ShapeContext::ShapeContext(PyFrameObject *f, PyObject *signature)
    : frame_(f), signature_(signature), is_method_(false), applied_(false) {
  Py_XINCREF(f);
  Py_XINCREF(signature);
  if (signature != nullptr) {
    if (!PyTuple_Check(signature) && !PyList_Check(signature)) {
      auto tuple = PyTuple_New(1);
      PyTuple_SET_ITEM(tuple, 0, signature);
      Py_DECREF(signature);
      signature_ = tuple;
    } else if (!PyTuple_Check(signature)) {
      int size = PyList_Size(signature);
      auto tuple = PyTuple_New(size);
      for (int i = 0; i < size; ++i) {
        auto item = PyList_GetItem(signature, i);
        Py_XINCREF(item);
        PyTuple_SET_ITEM(tuple, i, item);
      }
      Py_DECREF(signature);
      signature_ = tuple;
    }
    int argc = frame_->f_code->co_argcount + frame_->f_code->co_kwonlyargcount;
    is_method_ = (argc == (PyTuple_GET_SIZE(signature_) + 1)) ? true : false;
    std::vector<PyObject *> locals(&(frame_->f_localsplus[is_method_ ? 1 : 0]), &(frame_->f_localsplus[argc]));
    origin_ = locals;
  }
}

ShapeContext::~ShapeContext() {
  RevertSignature();
  Py_XDECREF(frame_);
  Py_XDECREF(signature_);
}

static constexpr int64_t kDynamicDim = -2;
static constexpr int64_t kDynamicShape = -1;

static bool IsShapeUnknown(mindspore::tensor::TensorPtr tensor) {
  auto &shape = tensor->shape();
  if (std::any_of(shape.begin(), shape.end(), [](const auto &element) { return element == kDynamicShape; })) {
    return true;
  }
  if (shape.size() == 1 && shape[0] == kDynamicDim) {
    return true;
  }
  return false;
}

static bool CheckDynamicShape(mindspore::tensor::TensorPtr sig, mindspore::tensor::TensorPtr org) {
  if (sig->data_type() != org->data_type()) {
    return false;
  }
  auto &sig_shape = sig->shape();
  if (sig_shape.size() == 1 && sig_shape[0] == kDynamicDim) {
    return true;
  }
  auto &org_shape = org->shape();
  if (sig_shape.size() != org_shape.size()) {
    return false;
  }
  for (size_t i = 0; i < sig_shape.size(); ++i) {
    if (sig_shape[i] != org_shape[i] && sig_shape[i] != kDynamicShape) {
      return false;
    }
  }
  return true;
}

static bool CheckSymbolicShape(PyObject *attr, mindspore::tensor::TensorPtr org) {
  if (attr == nullptr || !PyList_Check(attr) || org == nullptr) {
    return false;
  }
  auto shape = org->shape();
  std::map<int64_t, int64_t> symbolic_shape_data;
  for (int i = 0; i < PyList_GET_SIZE(attr); ++i) {
    auto item = PyList_GetItem(attr, i);
    if (!PyDict_Check(item)) {
      continue;
    }
    auto id = PyDict_GetItemString(item, "id");
    if (id != nullptr) {
      auto idv = PyLong_AsLong(id);
      if (symbolic_shape_data.find(idv) == symbolic_shape_data.end()) {
        symbolic_shape_data[idv] = shape[i];
      } else if (symbolic_shape_data[idv] != shape[i]) {
        return false;
      }
    }
    auto min = PyDict_GetItemString(item, "min");
    if (min != nullptr && PyLong_Check(min) && PyLong_AsLong(min) > shape[i]) {
      return false;
    }
    auto max = PyDict_GetItemString(item, "max");
    if (max != nullptr && PyLong_Check(max) && PyLong_AsLong(max) < shape[i]) {
      return false;
    }
    auto d = PyDict_GetItemString(item, "divisor");
    int64_t dv = d != nullptr ? PyLong_AsLong(d) : 1;
    auto r = PyDict_GetItemString(item, "remainder");
    int64_t rv = r != nullptr ? PyLong_AsLong(r) : 0;
    if (dv > shape[i] || shape[i] % dv != rv) {
      return false;
    }
  }
  return true;
}

static bool CheckTensorValid(PyObject *sig, PyObject *org) {
  mindspore::tensor::TensorPtr psig = py::cast<mindspore::tensor::TensorPtr>(sig);
  mindspore::tensor::TensorPtr porg = py::cast<mindspore::tensor::TensorPtr>(org);
  if (IsShapeUnknown(psig) && !CheckDynamicShape(psig, porg)) {
    return false;
  }
  if (PyObject_HasAttrString(sig, "symbolic_shape")) {
    PyObject *attr = PyObject_GetAttrString(sig, "symbolic_shape");
    if (!CheckSymbolicShape(attr, porg)) {
      Py_DECREF(attr);
      return false;
    }
    Py_DECREF(attr);
  }
  return true;
}

static bool CheckItemValid(PyObject *sig, PyObject *org);
static bool CheckListValid(PyObject *sig, PyObject *org) {
  if (PyList_Size(sig) != PyList_Size(org)) {
    return false;
  }
  for (Py_ssize_t i = 0; i < PyList_Size(sig); ++i) {
    PyObject *sig_item = PyList_GetItem(sig, i);
    PyObject *org_item = PyList_GetItem(org, i);
    if (!CheckItemValid(sig_item, org_item)) {
      return false;
    }
  }
  return true;
}

static bool CheckTupleValid(PyObject *sig, PyObject *org) {
  if (PyTuple_GET_SIZE(sig) != PyTuple_GET_SIZE(org)) {
    return false;
  }
  for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(sig); ++i) {
    PyObject *sig_item = PyTuple_GET_ITEM(sig, i);
    PyObject *org_item = PyTuple_GET_ITEM(org, i);
    if (!CheckItemValid(sig_item, org_item)) {
      return false;
    }
  }
  return true;
}

static bool CheckItemValid(PyObject *sig, PyObject *org) {
  if (sig == nullptr || org == nullptr || sig == Py_None || org == Py_None) {
    return true;
  }
  if (py::isinstance<mindspore::tensor::Tensor>(sig) && py::isinstance<mindspore::tensor::Tensor>(org) &&
      !CheckTensorValid(sig, org)) {
    return false;
  }
  if (PyList_Check(sig) && PyList_Check(org) && !CheckListValid(sig, org)) {
    return false;
  }
  if (PyTuple_Check(sig) && PyTuple_Check(org) && !CheckTupleValid(sig, org)) {
    return false;
  }
  return true;
}

bool ShapeContext::CheckValid() {
  if (signature_ == nullptr) {
    return false;
  }
  int argc = frame_->f_code->co_argcount + frame_->f_code->co_kwonlyargcount;
  if ((PyTuple_GET_SIZE(signature_) + (is_method_ ? 1 : 0)) != argc) {
    return false;
  }
  for (int i = 0; i < PyTuple_GET_SIZE(signature_); ++i) {
    auto sig = PyTuple_GetItem(signature_, i);
    auto org = origin_[i];
    if (!CheckItemValid(sig, org)) {
      return false;
    }
  }
  return true;
}

void ShapeContext::ApplySignature() {
  if (applied_) {
    return;
  }
  if (!CheckValid()) {
    return;
  }
  int argc = frame_->f_code->co_argcount + frame_->f_code->co_kwonlyargcount;
  for (int i = (is_method_ ? 1 : 0), j = 0; i < argc; ++i, ++j) {
    PyObject *sig_item = PyTuple_GetItem(signature_, j);
    PyObject *org_item = frame_->f_localsplus[i];
    if (sig_item != nullptr && sig_item != Py_None && org_item != nullptr && org_item != Py_None) {
      frame_->f_localsplus[i] = sig_item;
    }
  }
  applied_ = true;
}

void ShapeContext::RevertSignature() {
  if (!applied_) {
    return;
  }
  int argc = frame_->f_code->co_argcount + frame_->f_code->co_kwonlyargcount;
  for (int i = (is_method_ ? 1 : 0), j = 0; i < argc; ++i, ++j) {
    frame_->f_localsplus[i] = origin_[j];
  }
  applied_ = false;
}

}  // namespace pijit
}  // namespace mindspore
