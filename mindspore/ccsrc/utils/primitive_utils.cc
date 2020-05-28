/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "utils/primitive_utils.h"
#include "pipeline/parse/python_adapter.h"
#include "utils/log_adapter.h"
#include "common/utils.h"

namespace mindspore {
py::function GetBpropFunctionByObj(py::object obj) {
  static const std::string get_bprop_fn = "get_bprop_fn";
  static const std::string ad_module = "mindspore.ops._grad";
  py::function fn = parse::python_adapter::GetPyFn(ad_module, get_bprop_fn)(obj);
  return fn;
}

py::function GetBpropFunction(std::string name) {
  auto fn = GetBpropFunctionByObj(py::str(name));
  if (fn.is_none()) {
    MS_LOG(WARNING) << "Can't find bprop function for " << name;
  }
  return fn;
}

py::function GetComputeFunction(std::string name) {
  static const std::string module = "mindspore._extends.builtin_operations";
  py::module mod = py::module::import(common::SafeCStr(module));
  if (!py::hasattr(mod, common::SafeCStr(name))) {
    PyErr_SetString(PyExc_NotImplementedError, common::SafeCStr(name));
    // If raise AttributeError, user can't understand. This case need raise NotImplementedError.
    throw py::error_already_set();
  }
  py::object fn = mod.attr(common::SafeCStr(name));
  return fn;
}
}  // namespace mindspore
