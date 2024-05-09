/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include <random>

#include "include/common/pybind_api/api_register.h"

namespace mindspore {
py::object random_seeded_generator() {
  // Reason to have this c++ api:
  // > Can't use the Generator primitive in dump mode, because it runs in pynative during MindSpore import
  // > No random number generation api in python
  py::object PyGenerator = py::module::import("mindspore.common.generator").attr("Generator");
  py::object g = PyGenerator();
  py::object seed = g.attr("_seed");  // seed parameter
  py::object seed_set_data = seed.attr("set_data");
  seed_set_data(std::random_device()());
  return g;
}

void RegRandomSeededGenerator(py::module *m) {
  (void)m->def("_random_seeded_generator", &random_seeded_generator, "Get a random seeded generator.");
}
}  // namespace mindspore
