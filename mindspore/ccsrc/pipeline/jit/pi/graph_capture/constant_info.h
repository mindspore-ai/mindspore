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
#ifndef MINDSPORE_PI_JIT_GRAPH_CAPTURE_CONSTANT_INFO_H
#define MINDSPORE_PI_JIT_GRAPH_CAPTURE_CONSTANT_INFO_H

#include <memory>
#include <string>
#include <map>
#include "pybind11/pybind11.h"

namespace mindspore {
namespace pijit {
namespace py = pybind11;

class ValueNode;
class CallNode;

class ConstantInfo {
 public:
  ConstantInfo() : type_(), value_(), len_(-1), attrs_() {}

  void set_value(const py::object &op);
  void set_value(PyObject *op) { set_value(py::reinterpret_borrow<py::object>(op)); }
  void set_type(PyTypeObject *tp) { type_ = py::reinterpret_borrow<py::object>(reinterpret_cast<PyObject *>(tp)); }
  void set_len(Py_ssize_t s) { len_ = s; }

  PyTypeObject *type() const { return reinterpret_cast<PyTypeObject *>(type_.ptr()); }
  const py::object &value() const { return value_; }
  Py_ssize_t len() const { return len_; }

  ConstantInfo *GetAttr(const std::string &key) { return &attrs_[key]; }
  bool HasAttr(const std::string &key) const { return attrs_.find(key) != attrs_.end(); }

  std::string ToString() const;

  static void CollectConstantInfo(ValueNode *);
  static void CollectPrimitiveConstantInfo(CallNode *);
  static void CollectBuiltinFuncConstantInfo(CallNode *);

 private:
  py::object type_;
  py::object value_;
  Py_ssize_t len_;
  std::map<std::string, ConstantInfo> attrs_;
};

}  // namespace pijit
}  // namespace mindspore

#endif
