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

#include "pynative/backward/hook/saved_tensor_hook.h"
#include <utility>
#include <memory>
#include "include/utils/tensor_py.h"
#include "include/utils/pynative/grad_state.h"
#include "include/runtime/pipeline/pipeline.h"

namespace mindspore::pynative::autograd {
std::stack<std::pair<py::function, py::function>> DefaultSavedTensorHookUtil::hook_stack_ = {};
std::optional<std::string> DefaultSavedTensorHookUtil::disabled_error_message_ = std::nullopt;

PySavedTensorHook::PySavedTensorHook(py::function pack_hook, py::function unpack_hook)
    : pack_hook_(std::move(pack_hook)), unpack_hook_(std::move(unpack_hook)) {}

PySavedTensorHook::~PySavedTensorHook() {
  py::gil_scoped_acquire gil;
  pack_hook_ = py::object();
  unpack_hook_ = py::object();
  data_ = py::object();
}

void PySavedTensorHook::RunPackHook(const tensor::TensorPtr &tensor) {
  py::gil_scoped_acquire gil;
  NoGradGuard no_grad;
  data_ = pack_hook_(py::reinterpret_steal<py::object>(tensor::Wrap(tensor)));
}

tensor::TensorPtr PySavedTensorHook::RunUnpackHook() {
  py::gil_scoped_acquire gil;
  const auto ret = unpack_hook_(data_);
  const auto ret_tensor = tensor::ConvertToTensor(ret);
  runtime::Pipeline::Get().WaitFrontend();
  MS_EXCEPTION_IF_NULL(ret_tensor);
  runtime::Pipeline::Get().WaitFrontend();
  return ret_tensor;
}

void DefaultSavedTensorHookUtil::PushHook(const py::function &pack_hook, const py::function &unpack_hook) {
  MS_EXCEPTION_IF_CHECK_FAIL(IsEnabled(), disabled_error_message_.value());
  (void)hook_stack_.emplace(pack_hook, unpack_hook);
}

void DefaultSavedTensorHookUtil::PopHook() {
  MS_EXCEPTION_IF_CHECK_FAIL(!hook_stack_.empty(), "Saved tensor hook stack is empty.");
  hook_stack_.pop();
}

std::unique_ptr<PySavedTensorHook> DefaultSavedTensorHookUtil::GetTopHook() {
  if (hook_stack_.empty()) {
    return nullptr;
  }
  py::gil_scoped_acquire gil;
  auto &[pack_hook, unpack_hook] = hook_stack_.top();
  return std::make_unique<PySavedTensorHook>(pack_hook, unpack_hook);
}

std::optional<std::string> DefaultSavedTensorHookUtil::Disable(const std::string &error_msg,
                                                               bool is_error_on_outer_hook) {
  MS_EXCEPTION_IF_CHECK_FAIL(hook_stack_.empty() || !is_error_on_outer_hook, error_msg);
  auto pre_error_msg = disabled_error_message_;
  disabled_error_message_ = error_msg;
  return pre_error_msg;
}

void DefaultSavedTensorHookUtil::SetDisableErrorMessage(std::optional<std::string> error_msg) {
  disabled_error_message_ = std::move(error_msg);
}

bool DefaultSavedTensorHookUtil::IsEnabled() { return !disabled_error_message_.has_value(); }

bool DefaultSavedTensorHookUtil::IsActive() { return IsEnabled() && GetTopHook() != nullptr; }
}  // namespace mindspore::pynative::autograd
