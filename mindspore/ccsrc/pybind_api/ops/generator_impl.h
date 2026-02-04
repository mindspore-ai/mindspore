/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include <mutex>
#include "ir/tensor.h"
#include "infer/ops_func_impl/generator.h"
#include "pybind11/pybind11.h"

#ifndef MINDSPORE_CCSRC_PYBINDAPI_OPS_GENERATOR_IMPL_H_
#define MINDSPORE_CCSRC_PYBINDAPI_OPS_GENERATOR_IMPL_H_

namespace py = pybind11;

namespace mindspore {
// NOLINTBEGIN
using namespace ops::generator;
// NOLINTEND
class GeneratorImpl {
 public:
  GeneratorImpl(const py::handle &seed_param, const py::handle &offset_param);
  ~GeneratorImpl() = default;
  py::object operator()(const py::handle &cmd, const py::tuple &inputs);
  py::object set_state(const py::handle &state_py);
  py::object get_state() const;
  py::object seed() const;
  py::object manual_seed(const py::handle &seed_py);
  py::object initial_seed() const;
  py::object step(const py::handle &step_py);

 private:
  tensor::TensorPtr seed_;
  tensor::TensorPtr offset_;

  param_type *seed_data_{nullptr};
  param_type *offset_data_{nullptr};

  std::mutex mutex_;
};
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PYBINDAPI_OPS_GENERATOR_IMPL_H_
