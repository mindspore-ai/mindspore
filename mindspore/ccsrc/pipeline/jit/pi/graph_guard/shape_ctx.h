/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_PI_JIT_SHAPE_CTX_H
#define MINDSPORE_PI_JIT_SHAPE_CTX_H

#include <memory>
#include <vector>
#include "pybind11/pybind11.h"

namespace mindspore {
namespace pijit {

/// \brief shape context
class ShapeContext {
 public:
  ShapeContext(PyFrameObject *f, PyObject *signature);
  virtual ~ShapeContext();

  virtual bool CheckValid();
  virtual void ApplySignature();
  virtual void RevertSignature();

 protected:
  PyFrameObject *frame_;
  PyObject *signature_;
  std::vector<PyObject *> origin_;
  bool is_method_;
  bool applied_;
};
using ShapeContextPtr = std::shared_ptr<ShapeContext>;

}  // namespace pijit
}  // namespace mindspore

#endif  // MINDSPORE_PI_JIT_SHAPE_CTX_H
