/**
 * Copyright 2024-2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PYBIND_API_IR_HOOK_PY_H_
#define MINDSPORE_CCSRC_PYBIND_API_IR_HOOK_PY_H_

#include <memory>
#include "pybind11/pybind11.h"
#include "pybind11/pytypes.h"
#include "ir/tensor.h"
#include "include/utils/visible.h"
#include "include/utils/pynative/hook.h"

namespace mindspore::pynative::autograd {
namespace py = pybind11;
struct RegisterHook {
  /// \brief Register a backward hook
  ///
  /// \ void
  static uint64_t RegisterTensorBackwardHook(const tensor::TensorPtr &tensor, const py::function &hook);

  /// \brief Remove a backward hook
  ///
  /// \ void
  static void RemoveTensorBackwardHook(uint64_t handle_id);
  static py::list GetHooks(const tensor::TensorPtr &tensor);
  static unsigned RegisterCppTensorBackwardHook(const tensor::TensorPtr &tensor, const CppHookFn &hook);
  static void RemoveCppTensorBackwardHook(const tensor::TensorPtr &tensor, unsigned hook_id);

  static void ClearHookMap();
};
}  // namespace mindspore::pynative::autograd
#endif  // MINDSPORE_CCSRC_PYBIND_API_IR_HOOK_PY_H_
