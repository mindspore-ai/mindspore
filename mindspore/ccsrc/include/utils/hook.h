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

#ifndef MINDSPORE_CORE_UTILS_HOOK_H_
#define MINDSPORE_CORE_UTILS_HOOK_H_

#include <vector>
#include <map>
#include <memory>
#include "pybind11/pybind11.h"
#include "ir/anf.h"
#include "include/utils/visible.h"

namespace mindspore {
namespace py = pybind11;

struct COMMON_EXPORT BackWardHook {
  virtual ~BackWardHook() = default;
  virtual ValuePtr operator()(const ValuePtr &grad) = 0;
};

struct COMMON_EXPORT TensorBackwardHook : public BackWardHook {
  TensorBackwardHook(uint64_t tensor_id, const py::function &obj);
  ~TensorBackwardHook() override;
  ValuePtr operator()(const ValuePtr &grad) override;
  uint64_t tensor_id_;
  py::function hook_;
};
}  // namespace mindspore

#endif  // MINDSPORE_CORE_UTILS_HOOK_H_
