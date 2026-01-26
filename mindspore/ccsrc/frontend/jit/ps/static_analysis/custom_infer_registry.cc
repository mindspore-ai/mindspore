/**
 * Copyright 2026 Huawei Technologies Co., Ltd
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

#include "include/frontend/jit/ps/static_analysis/custom_infer_registry.h"
#include <algorithm>
#include "include/frontend/pybind_api.h"

namespace mindspore {
void OpInferRegistry::RegisterFunction(const py::object &op_name, const py::object &func_name, py::function func) {
  if (!py::isinstance<py::str>(op_name)) {
    std::string op_name_type_str = py::str(py::type::of(op_name));
    MS_LOG(EXCEPTION) << "Op name for op infer is not string, but got: " << op_name_type_str;
  }
  if (!py::isinstance<py::str>(func_name)) {
    std::string func_name_type_str = py::str(py::type::of(func_name));
    MS_LOG(EXCEPTION) << "Func name for op infer is not string, but got: " << func_name_type_str;
  }
  std::string op_name_str = py::cast<std::string>(op_name);
  std::string func_name_str = py::cast<std::string>(func_name);
  if (op_name_str.empty() || func_name_str.empty()) {
    MS_LOG(EXCEPTION) << "Op name or func name for op infer is empty";
  }
  auto it = op_infer_map_.find(op_name_str);
  if (it == op_infer_map_.end()) {
    op_infer_map_[op_name_str] = {{func_name_str}, {func}};
  } else {
    auto &func_names = it->second.func_names;
    auto it_func_list = std::find(func_names.begin(), func_names.end(), func_name_str);
    if (it_func_list != func_names.end()) {
      MS_LOG(EXCEPTION) << "Registering function: " << func_name_str << " for op: " << op_name_str
                        << " multiple times.";
    }
    it->second.funcs.emplace_back(func);
    it->second.func_names.emplace_back(func_name_str);
  }

  MS_LOG(DEBUG) << "Registered op infer function: " << func_name_str << " for op " << op_name_str;
  return;
}

std::vector<py::function> OpInferRegistry::GetFunctions(const std::string &op_name) const {
  if (op_name.empty()) {
    MS_LOG(ERROR) << "Op name for op infer is empty";
    return std::vector<py::function>();
  }
  auto it = op_infer_map_.find(op_name);
  if (it == op_infer_map_.end()) {
    MS_LOG(DEBUG) << "Op infer function not found for op: " << op_name;
    return std::vector<py::function>();
  }
  return it->second.funcs;
}

std::vector<std::string> OpInferRegistry::GetFunctionNames(const std::string &op_name) const {
  if (op_name.empty()) {
    MS_LOG(ERROR) << "Op name for op infer is empty";
    return std::vector<std::string>();
  }
  auto it = op_infer_map_.find(op_name);
  if (it == op_infer_map_.end()) {
    MS_LOG(DEBUG) << "Op infer function not found for op func: " << op_name;
    return std::vector<std::string>();
  }
  return it->second.func_names;
}

void OpInferRegistry::Clear() {
  op_infer_map_.clear();
  MS_LOG(DEBUG) << "Cleared all op infer functions";
}

void RegOpInferRegistryPy(const py::module *m) {
  py::class_<OpInferRegistry, std::shared_ptr<OpInferRegistry>>(*m, "OpInferRegistry")
    .def_static("instance", &OpInferRegistry::GetInstance)
    .def("register_function", &OpInferRegistry::RegisterFunction)
    .def("clear_funcs", &OpInferRegistry::Clear);
}

void OpWrapperRegistry::RegisterFunction(const py::object &op_name, py::function func) {
  if (!py::isinstance<py::str>(op_name)) {
    MS_LOG(EXCEPTION) << "Op name for op wrapper is not string";
  }

  std::string op_name_str = py::cast<std::string>(op_name);
  if (op_name_str.empty()) {
    MS_LOG(EXCEPTION) << "Op name for op wrapper is empty";
  }
  auto it_func_list = func_map_.find(op_name_str);
  if (it_func_list != func_map_.end()) {
    MS_LOG(EXCEPTION) << "Registering op wrapper for op: " << op_name_str << " multiple times.";
  }
  func_map_[op_name_str] = func;

  MS_LOG(DEBUG) << "Registered op wrapper function for op " << op_name_str;
  return;
}

py::function OpWrapperRegistry::GetFunction(const std::string &op_name) const {
  if (op_name.empty()) {
    MS_LOG(ERROR) << "Op name for op wrapper is empty";
    return py::none();
  }
  auto it = func_map_.find(op_name);
  if (it == func_map_.end()) {
    MS_LOG(DEBUG) << "Op wrapper function not found for op: " << op_name;
    return py::none();
  }
  return it->second;
}

void OpWrapperRegistry::Clear() {
  func_map_.clear();
  MS_LOG(DEBUG) << "Cleared all op wrapper functions";
}

void RegOpWrapperRegistryPy(const py::module *m) {
  py::class_<OpWrapperRegistry, std::shared_ptr<OpWrapperRegistry>>(*m, "OpWrapperRegistry")
    .def_static("instance", &OpWrapperRegistry::GetInstance)
    .def("register_function", &OpWrapperRegistry::RegisterFunction)
    .def("clear_funcs", &OpWrapperRegistry::Clear);
}
}  // namespace mindspore
