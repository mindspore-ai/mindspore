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
#include "pipeline/jit/pi/runtime.h"
#include "pipeline/jit/pi/external.h"

namespace mindspore {
namespace pijit {

bool ApplyAutoJit(PyThreadState *ts, PyFrameObject *f, PyObject **result) {
  if (!kPIJitConfigDefault.GetBoolConfig(GraphJitConfig::kAutoJit)) {
    return false;
  }

  PyObject *code = reinterpret_cast<PyObject *>(f->f_code);
  auto c = GetJitCompileResults(code);
  if (c == nullptr) {
    if (!kPIJitConfigDefault.ShouldAutoJit(f)) {
      return false;
    }
    (void)pi_jit_should_compile(py::cast<py::object>(code), py::dict(), py::none());
    c = GetJitCompileResults(code);
  }
  *result = CallCodeHook(ts, f, c);
  return true;
}

bool ApplyAutoGrad(PyThreadState *ts, PyFrameObject *f, PyObject **result) {
  if (kPIJitConfigDefault.GetBoolConfig(GraphJitConfig::kInferOnly)) {
    return false;
  }
  *result = _PyEval_EvalFrameDefault(ts, f, 0);
  AutoGrad(f, *result);
  return true;
}

PyFrameEvalHookManager::PyFrameEvalHookManager() : func_() {
  this->Register(ApplyAutoGrad);
  this->Register(ApplyAutoJit);
  this->Register([](PyThreadState *ts, PyFrameObject *f, PyObject **result) {
    auto c = GetJitCompileResults(f->f_code);
    *result = c != nullptr ? CallCodeHook(ts, f, c) : _PyEval_EvalFrameDefault(ts, f, 0);
    return true;
  });
}

PyFrameEvalHookManager *PyFrameEvalHookManager::GetInstance() {
  static PyFrameEvalHookManager instance;
  return &instance;
}

PyObject *PyFrameEvalHookManager::RunHook(PyThreadState *ts, PyFrameObject *f) {
  PyObject *res = nullptr;
  if (std::any_of(func_.rbegin(), func_.rend(), [&](Hook func) { return func(ts, f, &res); })) {
    return res;
  }
  return _PyEval_EvalFrameDefault(ts, f, 0);
}

}  // namespace pijit
}  // namespace mindspore
