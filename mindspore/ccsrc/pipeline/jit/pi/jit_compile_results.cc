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
#include "pipeline/jit/pi/jit_compile_results.h"

namespace mindspore {
namespace pijit {

static Py_tss_t *tss_ = nullptr;

void JitCompileResults::FreeCallback(void *ptr) {
  // maybe nullptr if other module use _PyEval_RequestCodeExtraIndex
  if (ptr == nullptr) {
    return;
  }
  JitCompileResults *c = reinterpret_cast<JitCompileResults *>(ptr);
  delete c;
}

Py_ssize_t JitCompileResults::InitIndex() {
  if (tss_ == nullptr) {
    tss_ = PyThread_tss_alloc();
    PyThread_tss_create(tss_);
  }

  Py_ssize_t index = reinterpret_cast<Py_ssize_t>(PyThread_tss_get(tss_));
  if (index == 0) {
    index = _PyEval_RequestCodeExtraIndex(FreeCallback);
    if (index == -1) {
      return -1;
    }
    // ensure index is not 0
    PyThread_tss_set(tss_, reinterpret_cast<void *>(index + 1));
  } else {
    index = index - 1;
  }
  return index;
}

JitCompileResults::JitCompileResults() {
  this->stat_ = JitCompileResults::NEVER_COMPILE;
  this->codehub_ = std::make_shared<OptCodeHub>();
  this->tbs_ = std::make_shared<Traceback>();
  this->conf_ = std::make_shared<GraphJitConfig>();
  this->compile_count_ = 0;
  this->break_count_ = 0;
}

JitCompileResults::~JitCompileResults() {
  for (auto &oc : this->codehub()->GetOptTarget(OptOption::CreateOptionByPoint(this))) {
    PyCodeObject *co = oc->GetPythonCode();
    if (co != nullptr && Py_REFCNT(co) != 1) {
      MS_LOG(ERROR) << "code handler not only one" << std::string(py::str(reinterpret_cast<PyObject *>(co)));
    }
  }
  this->code_ = nullptr;
  this->codehub_.reset();
}

void JitCompileResults::set_stat(JitCompileResults::State s) { this->stat_ = s; }

JitCompileResults *JitCompileResults::Create(PyCodeObject *co) {
  Py_ssize_t index = InitIndex();
  if (index == -1) {
    return nullptr;
  }
  PyObject *code = reinterpret_cast<PyObject *>(co);
  JitCompileResults *c = nullptr;
  if (!_PyCode_GetExtra(code, index, reinterpret_cast<void **>(&c))) {
    if (c != nullptr) {
      return c;
    }
    c = new JitCompileResults();
    if (!_PyCode_SetExtra(code, index, c)) {
      return c;
    }
    delete c;
  }
  PyErr_Clear();
  return nullptr;
}

}  // namespace pijit
}  // namespace mindspore
