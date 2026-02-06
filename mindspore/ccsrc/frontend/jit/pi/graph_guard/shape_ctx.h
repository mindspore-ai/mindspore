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
#ifndef MINDSPORE_CCSRC_FRONTEND_JIT_PI_SHAPE_CTX_H
#define MINDSPORE_CCSRC_FRONTEND_JIT_PI_SHAPE_CTX_H

#include <memory>
#include <map>
#include "pybind11/pybind11.h"
#include "frontend/jit/pi/python_adapter/py_frame.h"

namespace mindspore {
namespace pijit {
/// \brief shape context
class ShapeContext {
 public:
  ShapeContext(PyFrameWrapper f, const py::object &enable_dynamic_dict);
  ~ShapeContext();

  void ApplyEnableDynamic();
  void RevertEnableDynamic();
  void UpdateFastLocal(PyObject **fast_local, PyCodeObject *code, PyObject *arg, int index);

 private:
  PyFrameWrapper frame_;
  PyObject *enable_dynamic_dict_{nullptr};
  std::map<int, PyObject *> origin_;
  bool applied_{false};
};
using ShapeContextPtr = std::shared_ptr<ShapeContext>;
}  // namespace pijit
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_JIT_PI_SHAPE_CTX_H
