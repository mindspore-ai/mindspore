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

#ifndef MINDSPORE_CCSRC_PYNATIVE_BACKWARD_HOOK_SAVED_TENSOR_HOOK_H_
#define MINDSPORE_CCSRC_PYNATIVE_BACKWARD_HOOK_SAVED_TENSOR_HOOK_H_

#include <stack>
#include <memory>
#include <utility>
#include "pybind11/pybind11.h"
#include "ir/tensor.h"

namespace py = pybind11;
namespace mindspore::pynative::autograd {
struct PySavedTensorHook {
  PySavedTensorHook(py::function pack_hook, py::function unpack_hook);
  ~PySavedTensorHook();
  void RunPackHook(const tensor::TensorPtr &tensor);
  tensor::TensorPtr RunUnpackHook();

 private:
  py::function pack_hook_;
  py::function unpack_hook_;
  py::object data_;
};

struct DefaultSavedTensorHookUtil {
  static void PushHook(const py::function &pack_hook, const py::function &unpack_hook);
  static void PopHook();
  static std::unique_ptr<PySavedTensorHook> GetTopHook();
  static std::optional<std::string> Disable(const std::string &error_msg, bool is_error_on_outer_hook);
  static void SetDisableErrorMessage(std::optional<std::string> error_msg);
  static bool IsEnabled();
  static bool IsActive();

 private:
  static std::stack<std::pair<py::function, py::function>> hook_stack_;
  static std::optional<std::string> disabled_error_message_;
};

}  // namespace mindspore::pynative::autograd

#endif  // MINDSPORE_CCSRC_PYNATIVE_BACKWARD_HOOK_SAVED_TENSOR_HOOK_H_
