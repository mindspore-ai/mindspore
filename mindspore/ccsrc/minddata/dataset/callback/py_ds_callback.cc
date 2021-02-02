/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "minddata/dataset/callback/callback_manager.h"
#include "minddata/dataset/callback/py_ds_callback.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {

Status PyDSCallback::DSBegin(const CallbackParam &cb_param) {
  return PyDSCallback::ExecutePyfunc(begin_func_, cb_param);
}
Status PyDSCallback::DSEpochBegin(const CallbackParam &cb_param) {
  return PyDSCallback::ExecutePyfunc(epoch_begin_func_, cb_param);
}
Status PyDSCallback::DSNStepBegin(const CallbackParam &cb_param) {
  return PyDSCallback::ExecutePyfunc(step_begin_func_, cb_param);
}
Status PyDSCallback::DSEnd(const CallbackParam &cb_param) { return PyDSCallback::ExecutePyfunc(end_func_, cb_param); }

Status PyDSCallback::DSEpochEnd(const CallbackParam &cb_param) {
  return PyDSCallback::ExecutePyfunc(epoch_end_func_, cb_param);
}
Status PyDSCallback::DSNStepEnd(const CallbackParam &cb_param) {
  return PyDSCallback::ExecutePyfunc(step_end_func_, cb_param);
}

bool PyDSCallback::IsBeginNeeded() { return begin_needed_; }
bool PyDSCallback::IsEpochBeginNeeded() { return epoch_begin_needed_; }
bool PyDSCallback::IsNStepBeginNeeded() { return step_begin_needed_; }
bool PyDSCallback::IsNStepEndNeeded() { return step_end_needed_; }
bool PyDSCallback::IsEpochEndNeeded() { return epoch_end_needed_; }
bool PyDSCallback::IsEndNeeded() { return end_needed_; }

Status PyDSCallback::ExecutePyfunc(py::function f, const CallbackParam &cb_param) {
  {
    // Acquire Python GIL
    py::gil_scoped_acquire gil_acquire;
    if (Py_IsInitialized() == 0) {
      return Status(StatusCode::kMDPythonInterpreterFailure, "Python Interpreter is finalized");
    }
    try {
      f(cb_param);
    } catch (const py::error_already_set &e) {
      return Status(StatusCode::kMDPyFuncException, e.what());
    }
  }
  return Status::OK();
}
void PyDSCallback::setBegin(py::function f) {
  begin_func_ = f;
  begin_needed_ = true;
}
void PyDSCallback::setEnd(py::function f) {
  end_func_ = f;
  end_needed_ = true;
}
void PyDSCallback::setEpochBegin(py::function f) {
  epoch_begin_func_ = f;
  epoch_begin_needed_ = true;
}
void PyDSCallback::setEpochEnd(py::function f) {
  epoch_end_func_ = f;
  epoch_end_needed_ = true;
}
void PyDSCallback::setStepBegin(py::function f) {
  step_begin_func_ = f;
  step_begin_needed_ = true;
}
void PyDSCallback::setStepEnd(py::function f) {
  step_end_func_ = f;
  step_end_needed_ = true;
}

}  // namespace dataset
}  // namespace mindspore
