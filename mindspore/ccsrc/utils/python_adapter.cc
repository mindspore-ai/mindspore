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

#include "include/utils/python_adapter.h"
#include <memory>
#include <string>

namespace mindspore {
namespace python_adapter {
// Python scoped env, should only have one scoped_ instance
static std::shared_ptr<py::scoped_interpreter> scoped_ = nullptr;
// Python env: true: start process from python, false: start process from c++
static bool python_env_ = false;
static bool use_signature_in_resolve_ = true;
void ResetPythonScope() { scoped_ = nullptr; }
void set_use_signature_in_resolve(bool use_signature) noexcept { use_signature_in_resolve_ = use_signature; }
bool UseSignatureInResolve() { return use_signature_in_resolve_; }
void set_python_env_flag(bool python_env) noexcept { python_env_ = python_env; }
bool IsPythonEnv() { return python_env_; }
void SetPythonPath(const std::string &path) {
  // Load the python module path
  (void)python_adapter::set_python_scoped();
  py::module sys = py::module::import("sys");
  py::list sys_path = sys.attr("path");

  // Check the path is exist?
  bool is_exist = false;
  for (size_t i = 0; i < sys_path.size(); i++) {
    if (!py::isinstance<py::str>(sys_path[i])) {
      continue;
    }
    std::string path_str = py::cast<std::string>(sys_path[i]);
    if (path_str == path) {
      is_exist = true;
    }
  }
  if (!is_exist) {
    (void)sys_path.attr("append")(path.c_str());
  }
}

std::shared_ptr<py::scoped_interpreter> set_python_scoped() {
  // If start process from python, no need set the python scope.
  if (!python_env_) {
    if ((Py_IsInitialized() == 0) && (scoped_ == nullptr)) {
      scoped_ = std::make_shared<py::scoped_interpreter>();
    }
  }
  return scoped_;
}

// Return the module of python
py::module GetPyModule(const std::string &module) {
  if (!module.empty()) {
    return py::module::import(module.c_str());
  } else {
    return py::none();
  }
}

// Get the obj of attr
py::object GetPyObjAttr(const py::object &obj, const std::string &attr) {
  if (!attr.empty() && !py::isinstance<py::none>(obj)) {
    if (py::hasattr(obj, attr.c_str())) {
      return obj.attr(attr.c_str());
    }
    MS_LOG(DEBUG) << "Obj have not the attr: " << attr;
  }
  return py::none();
}

py::object GetPyFn(const std::string &module, const std::string &name) {
  (void)python_adapter::set_python_scoped();
  if (!module.empty() && !name.empty()) {
    py::module mod = py::module::import(module.c_str());
    py::object fn = mod.attr(name.c_str());
    return fn;
  }
  return py::none();
}
}  // namespace python_adapter
}  // namespace mindspore
