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
#include "pipeline/jit/pi_jit/ms_adapter/infer.h"
#include <string>
#include "pipeline/jit/pi_jit/utils/utils.h"
#include "pipeline/jit/pi_jit/common.h"
#include "pipeline/jit/pi_jit/external.h"

namespace mindspore {
namespace jit {
namespace graph {

bool IsMSAdapterModuleType(PyTypeObject *tp, bool sub_type) {
  py::object type = Utils::GetModuleAttr("msadapter.pytorch.nn.modules", "Module");
  PyTypeObject *tar = reinterpret_cast<PyTypeObject *>(type.ptr());
  if (tar == nullptr) {
    return false;
  }
  if (tar == tp) {
    return true;
  }
  return sub_type ? PyType_IsSubtype(tp, tar) : false;
}

bool IsMSAdapterModuleForwardCall(PyFrameObject *f) {
  if (f->f_code->co_argcount == 0 || strcmp("forward", PyUnicode_AsUTF8(f->f_code->co_name))) {
    // code not named 'forward' and arguments not declare 'self'
    return false;
  }
  if (f->f_localsplus[0] == nullptr) {
    // self as function closure
    return false;
  }
  return IsMSAdapterModuleType(Py_TYPE(f->f_localsplus[0]), true);
}

std::string GetTopModule(PyFrameObject *f) {
  PyObject *name = PyObject_GetItem(f->f_globals, py::str("__name__").ptr());
  if (name == nullptr) {
    PyErr_Clear();
    return "";
  }
  if (!PyUnicode_Check(name)) {
    Py_DECREF(name);
    return "";
  }
  const char *module_name = PyUnicode_AsUTF8(name);
  const char *s = strchr(module_name, '.');
  std::string res = s ? std::string(module_name, s - module_name) : module_name;
  Py_DECREF(name);
  return res;
}

void SpecializeForMSAdapterModule(PyFrameObject *f) {
  if (!IsMSAdapterModuleForwardCall(f)) {
    return;
  }
  PyObject *code = reinterpret_cast<PyObject *>(f->f_code);
  if (getJitCompileResults(code, false) != nullptr) {
    return;
  }
  py::dict conf;
  std::string top_module = GetTopModule(f);
  if (!top_module.empty()) {
    py::list modules;
    modules.append(py::str(top_module));
    conf["allowed_inline_modules"] = modules;
  }
  MS_LOG(INFO) << "captured forward at" << std::string(py::str(reinterpret_cast<PyObject *>(f->f_code)));
  (void)pi_jit_should_compile(py::cast<py::object>(code), conf);
}

}  // namespace graph
}  // namespace jit
}  // namespace mindspore
