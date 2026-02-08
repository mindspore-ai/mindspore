/**
 * Copyright 2024-2025 Huawei Technologies Co., Ltd
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

#include <string>
#include <vector>
#include "pybind_api/pynative/pynative_api.h"

namespace mindspore {
py::object add_docstr(py::object obj, py::str doc_str) {
  static std::vector<std::string> all_docs;
  const std::string &doc = doc_str.cast<std::string>();
  all_docs.push_back(doc);
  const char *doc_cstr = all_docs.back().c_str();

  PyObject *obj_ptr = obj.ptr();
  PyTypeObject *type = Py_TYPE(obj_ptr);

  if (type == &PyCFunction_Type) {
    auto *func = reinterpret_cast<PyCFunctionObject *>(obj_ptr);
    if (func->m_ml->ml_doc != nullptr) {
      throw std::runtime_error(std::string("Function '") + func->m_ml->ml_name + "' already has a docstring");
    }
    func->m_ml->ml_doc = doc_cstr;
  } else if (strcmp(type->tp_name, "method_descriptor") == 0) {
    auto *m = reinterpret_cast<PyMethodDescrObject *>(obj_ptr);
    if (m->d_method->ml_doc != nullptr) {
      throw std::runtime_error(std::string("Method '") + m->d_method->ml_name + "' already has a docstring");
    }
    m->d_method->ml_doc = doc_cstr;
  } else if (strcmp(type->tp_name, "getset_descriptor") == 0) {
    auto *g = reinterpret_cast<PyGetSetDescrObject *>(obj_ptr);
    if (g->d_getset->doc != nullptr) {
      throw std::runtime_error(std::string("Attribute '") + g->d_getset->name + "' already has a docstring");
    }
    g->d_getset->doc = doc_cstr;
  } else if (PyType_Check(obj_ptr)) {
    auto *tp = reinterpret_cast<PyTypeObject *>(obj_ptr);
    if (tp->tp_doc != nullptr) {
      throw std::runtime_error(std::string("Type '") + tp->tp_name + "' already has a docstring");
    }
    tp->tp_doc = doc_cstr;
  } else {
    throw std::invalid_argument(std::string("Unsupported type '") + type->tp_name + "' for adding docstring");
  }

  Py_INCREF(obj_ptr);
  return obj;
}

void RegTensorDoc(py::module *m) { (void)m->def("_add_docstr", &add_docstr); }
}  // namespace mindspore
