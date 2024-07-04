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
#include "pipeline/jit/pi/eval_frame_hook.h"
#include "pipeline/jit/pi/pydef.h"
#include "pipeline/jit/pi/pi_jit_config.h"
#include "pipeline/jit/pi/common.h"
#include "pipeline/jit/pi/external.h"

namespace mindspore {
namespace pijit {

PyFrameEvalHookManager::HookResult ApplyAutoJit(PyThreadState *tstate, PyFrameObject *f) {
  if (!kPIJitConfigDefault.GetBoolConfig(GraphJitConfig::kAutoJit)) {
    return {nullptr, false};
  }

  PyObject *code = reinterpret_cast<PyObject *>(f->f_code);
  auto c = getJitCompileResults(code, false);
  if (c == nullptr) {
    if (!kPIJitConfigDefault.ShouldAutoJit(f)) {
      return {nullptr, false};
    }
    (void)pi_jit_should_compile(py::cast<py::object>(code), py::dict(), py::none());
    c = getJitCompileResults(code, false);
  }
  return {CallCodeHook(tstate, f, c), true};
}

PyFrameEvalHookManager::HookResult ApplyAutoGrad(PyThreadState *tstate, PyFrameObject *f) {
  if (kPIJitConfigDefault.GetBoolConfig(GraphJitConfig::kInferOnly)) {
    return {nullptr, false};
  }
  auto ret = _PyEval_EvalFrameDefault(tstate, f, 0);
  AutoGrad(f, ret);
  return {ret, true};
}

PyFrameEvalHookManager::PyFrameEvalHookManager() : func_() {
  this->Register(ApplyAutoGrad);
  this->Register(ApplyAutoJit);
  this->Register([](PyThreadState *tstate, PyFrameObject *f) {
    auto c = CodeExtra::GetCodeExtra(f->f_code);
    return (c == nullptr) ? HookResult{_PyEval_EvalFrameDefault(tstate, f, 0), true}
                          : HookResult{CallCodeHook(tstate, f, c), true};
  });
}

PyFrameEvalHookManager *PyFrameEvalHookManager::GetInstance() {
  static PyFrameEvalHookManager instance;
  return &instance;
}

PyObject *PyFrameEvalHookManager::RunHook(PyThreadState *tstate, PyFrameObject *f) {
  for (auto iter = func_.rbegin(); iter != func_.rend(); ++iter) {
    HookResult r = (*iter)(tstate, f);
    if (r.has_result_) {
      return r.result_;
    }
  }
  return _PyEval_EvalFrameDefault(tstate, f, 0);
}

}  // namespace pijit
}  // namespace mindspore
