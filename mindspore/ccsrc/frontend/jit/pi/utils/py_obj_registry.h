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
#ifndef MINDSPORE_PI_JIT_UTILS_PY_OBJ_REGISTRY_H
#define MINDSPORE_PI_JIT_UTILS_PY_OBJ_REGISTRY_H

#include "pybind11/pybind11.h"

namespace mindspore {
namespace pijit {

namespace py = pybind11;

class PyObjRegistry {
 public:
  class Data {
    friend class PyObjRegistry;

   public:
    static Data &GetInstance();
    const auto &enable_grad() { return enable_grad_; }
    const auto &set_enable_grad() { return set_enable_grad_; }

   private:
    Data() = default;

    py::object enable_grad_;
    py::object set_enable_grad_;
  };

  PyObjRegistry();
  ~PyObjRegistry();
};

}  // namespace pijit
}  // namespace mindspore

#endif  // MINDSPORE_PI_JIT_UTILS_PY_OBJ_REGISTRY_H
