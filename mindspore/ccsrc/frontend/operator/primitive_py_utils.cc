/**
 * Copyright 2023-2025 Huawei Technologies Co., Ltd
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

#include "frontend/operator/primitive_py_utils.h"

#include <string>
#include <memory>
#include "include/utils/python_adapter.h"

namespace mindspore {
namespace prim {
py::function GetTaylorRuleFunctionByObj(const py::object &obj) {
  static const std::string get_taylor_fprop_fn = "get_taylor_fprop_fn";
  static const std::string ad_module = "mindspore.ops._grad_experimental";
  py::function fn = python_adapter::GetPyFn(ad_module, get_taylor_fprop_fn)(obj);
  return fn;
}

py::function GetTaylorRuleFunction(const std::string &name) {
  auto fn = GetTaylorRuleFunctionByObj(py::str(name));
  return fn;
}

py::function GetVmapRuleFunctionByObj(const py::object &obj, int axis_size) {
  constexpr char get_vmap_rule_fn[] = "get_vmap_rule";
  constexpr char vmap_module[] = "mindspore.ops._vmap";
  py::function fn = python_adapter::GetPyFn(vmap_module, get_vmap_rule_fn)(obj, axis_size);
  return fn;
}

py::function GetVmapRuleFunction(const std::string &name, int axis_size) {
  auto fn = GetVmapRuleFunctionByObj(py::str(name), axis_size);
  return fn;
}
}  // namespace prim
}  // namespace mindspore
