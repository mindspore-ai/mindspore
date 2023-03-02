/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "include/common/utils/dynamic_obfuscation/registry_opaque_predicate.h"
#include <algorithm>
#include "utils/info.h"

namespace mindspore {
namespace kernel {
CustomizedOpaquePredicate &CustomizedOpaquePredicate::GetInstance() {
  static CustomizedOpaquePredicate instance;
  return instance;
}

void CustomizedOpaquePredicate::set_func_names() {
  // get the function of get_func_names()
  py::gil_scoped_acquire gil_acquire;
  static const std::string &module_name = "mindspore.ops.operations._opaque_predicate_registry";
  static const std::string &entrance = "get_func_names";
  py::module module = py::module::import(module_name.c_str());
  py::object get_pyfunc_obj = module.attr(entrance.c_str());
  if (get_pyfunc_obj.is_none()) {
    MS_LOG(EXCEPTION) << "Cannot find a python function named " << entrance << "in module" << module_name;
  }
  py::function get_pyfunc = get_pyfunc_obj.cast<py::function>();
  py::tuple func_name_list = get_pyfunc();
  // clear old functions
  func_names_.clear();
  for (size_t i = 0; i < func_name_list.size(); i++) {
    func_names_.push_back(py::str(func_name_list[i]));
  }
  MS_LOG(DEBUG) << "Set function names finished, the number of functions is: " << func_names_.size();
}

const std::vector<std::string> CustomizedOpaquePredicate::get_func_names() {
  if (func_names_.size() == 0) {
    MS_LOG(EXCEPTION) << "The number of customized function names is zero, get function names failed.";
  }
  return func_names_;
}

py::function CustomizedOpaquePredicate::get_function() {
  py::gil_scoped_acquire gil_acquire;
  // get the function of get_opaque_predicate()
  static const std::string &module_name = "mindspore.ops.operations._opaque_predicate_registry";
  static const std::string &entrance = "get_opaque_predicate";
  py::module module = py::module::import(module_name.c_str());
  py::object get_pyfunc_obj = module.attr(entrance.c_str());
  if (get_pyfunc_obj.is_none()) {
    MS_LOG(EXCEPTION) << "Cannot find a python function named " << entrance << "in module" << module_name;
  }
  py::function get_pyfunc = get_pyfunc_obj.cast<py::function>();
  MS_LOG(DEBUG) << "The number of function is : " << func_names_.size();
  if (func_names_.size() == 0) {
    MS_EXCEPTION(ValueError) << "The customized_func is not set, please set it in load().";
  }
  std::string func_name = func_names_[0];
  MS_LOG(DEBUG) << "Get function name: " << func_name;
  func_name_code_.clear();
  std::transform(func_name.begin(), func_name.end(), std::back_inserter(func_name_code_),
                 [](const char &item) { return static_cast<int>(item); });
  if (func_name_code_.size() == 0) {
    MS_EXCEPTION(ValueError) << "Set func_name_code_ failed.";
  }
  py::object py_func_obj = get_pyfunc(py::str(func_name));
  if (py_func_obj.is_none()) {
    MS_EXCEPTION(ValueError) << "Cannot find python func with name: " << func_name;
  }
  return py_func_obj.cast<py::function>();
}

bool CustomizedOpaquePredicate::run_function(float x, float y) {
  if (Py_IsInitialized() != true) {
    MS_LOG(ERROR) << "Py_IsInitialized failed.";
    return false;
  }
  py::object customized_func = get_function();
  py::gil_scoped_acquire gil_acquire;
  int inputs_num = 2;
  py::tuple inputs(inputs_num);
  inputs[0] = py::float_(x);
  inputs[1] = py::float_(y);
  py::object result = customized_func(*inputs);
  if (result.is_none()) {
    MS_EXCEPTION(ValueError) << "Computing result of customized_func is None, please check it.";
  }
  bool bool_result = py::cast<bool>(result);
  int even_num = 2;
  if (func_name_code_[calling_count_ % func_name_code_.size()] % even_num == 0) {
    calling_count_ += 1;
    return bool_result;
  }
  calling_count_ += 1;
  return !bool_result;
}

void CustomizedOpaquePredicate::init_calling_count() {
  this->calling_count_ = 0;
  MS_LOG(INFO) << "calling_count_ has been initialized to " << calling_count_;
}
}  // namespace kernel
}  // namespace mindspore
