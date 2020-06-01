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

#ifndef MINDSPORE_CCSRC_IR_PARAM_VALUE_PY_H_
#define MINDSPORE_CCSRC_IR_PARAM_VALUE_PY_H_

#include <memory>

#include "ir/anf.h"
#include "pybind11/pybind11.h"

namespace mindspore {
namespace py = pybind11;

class ParamValuePy : public ParamValue {
 public:
  ParamValuePy() : value_(py::none()) {}
  explicit ParamValuePy(const py::object &value) : value_(value) {}
  ~ParamValuePy() override = default;

  py::object value() { return value_; }
  void set_value(const py::object &obj) { value_ = obj; }

 private:
  py::object value_;
};

using ParamValuePyPtr = std::shared_ptr<ParamValuePy>;
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_IR_PARAM_VALUE_PY_H_
