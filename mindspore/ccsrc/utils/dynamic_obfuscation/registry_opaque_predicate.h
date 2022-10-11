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

#ifndef MINDSPORE_REGISTRY_OPAQUE_PREDICATE_H
#define MINDSPORE_REGISTRY_OPAQUE_PREDICATE_H

#include <vector>
#include <utility>
#include <string>
#include <memory>
#include <map>
#include <mutex>
#include <unordered_map>
#include <list>

#include "pybind11/pybind11.h"
#include <Python.h>
#include "pybind11/numpy.h"
#include "include/common/visible.h"

namespace py = pybind11;
namespace mindspore {
namespace kernel {
class COMMON_EXPORT CustomizedOpaquePredicate {
 public:
  static CustomizedOpaquePredicate &GetInstance();
  void set_func_names();
  const std::vector<std::string> get_func_names();
  bool run_function(float x, float y);
  py::function get_function();
  void init_calling_count();

 private:
  CustomizedOpaquePredicate() : func_names_({}) {}
  ~CustomizedOpaquePredicate() = default;

  std::vector<std::string> func_names_;
  int calling_count_ = 0;
  std::vector<int> func_name_code_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_REGISTRY_OPAQUE_PREDICATE_H
