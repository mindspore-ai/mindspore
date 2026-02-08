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
#include "frontend/jit/pi/eval_frame_hook.h"
#include "frontend/jit/pi/pi_jit_config.h"
#include "frontend/jit/pi/runtime.h"
#include "frontend/jit/pi/external.h"
#include "frontend/jit/pi/capture_context.h"

namespace mindspore {
namespace pijit {

bool ApplyAutoJit(PyThreadState *ts, PyFrameWrapper f, PyObject **result) {
  if (!kPIJitConfigDefault.GetBoolConfig(GraphJitConfig::kAutoJit)) {
    return false;
  }

  PyObject *code = reinterpret_cast<PyObject *>(f.GetCode().ptr());
  auto c = GetJitCompileResults(code);
  if (c == nullptr) {
    if (!kPIJitConfigDefault.ShouldAutoJit(f)) {
      return false;
    }
    (void)pi_jit_should_compile(py::cast<py::object>(code));
    c = GetJitCompileResults(code);
  }
  *result = CallCodeHook(ts, f, c);
  return true;
}

bool ApplyCaptureContext(PyThreadState *tstate, PyFrameWrapper ef, PyObject **result) {
  PyFrameWrapper f(ef);
  auto ctx = CaptureContext::GetInstance();
  if (!ctx->IsEnable()) {
    return false;
  }
  PyCodeObject *co = f.GetCode().ptr();
  auto c = GetJitCompileResults(co);
  if (c == nullptr) {
    if (ctx->IsSkip(f)) {
      SetJitCompileResults(co, JitCompileResults::get_skip_jcr());
      c = JitCompileResults::get_skip_jcr();
      MS_LOG(DEBUG) << "skip code " << py::str(reinterpret_cast<PyObject *>(co));
    } else {
      py::object code = py::cast<py::object>(reinterpret_cast<PyObject *>(co));
      (void)pi_jit_should_compile(code);
      c = GetJitCompileResults(co);
    }
  }
  if (c->stat() == JitCompileResults::NEVER_COMPILE) {
    *result = _PyEval_EvalFrameDefault(tstate, f.frame(), 0);
    return true;
  }
  c->set_conf(ctx->config());
  *result = CallCodeHook(tstate, f, c);
  return true;
}

PyFrameEvalHookManager::PyFrameEvalHookManager() : func_() {
  this->Register(ApplyAutoJit);
  this->Register(ApplyCaptureContext);
}

PyFrameEvalHookManager *PyFrameEvalHookManager::GetInstance() {
  static PyFrameEvalHookManager instance;
  return &instance;
}

PyObject *PyFrameEvalHookManager::RunHook(PyThreadState *ts, PyFrameWrapper f) {
  PyObject *res = nullptr;
  if (std::any_of(func_.rbegin(), func_.rend(), [&](Hook func) { return func(ts, f, &res); })) {
    return res;
  }
  return _PyEval_EvalFrameDefault(ts, f.frame(), 0);
}

}  // namespace pijit
}  // namespace mindspore
