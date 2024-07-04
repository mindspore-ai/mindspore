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
#include "pipeline/jit/pi/code_extra.h"

namespace mindspore {
namespace pijit {

CodeExtra CodeExtra::skip_;
Py_tss_t *CodeExtra::tss_ = nullptr;

void CodeExtra::FreeCallback(void *ptr) {
  // maybe nullptr if other module use _PyEval_RequestCodeExtraIndex
  if (ptr == nullptr || ptr == &skip_) {
    return;
  }
  // called after code object freed
  JitCompileResults *c = reinterpret_cast<JitCompileResults *>(ptr);

  for (auto &oc : c->codehub()->GetOptTarget(OptOption::CreateOptionByPoint(c))) {
    PyCodeObject *co = oc->GetPythonCode();
    if (co != nullptr && Py_REFCNT(co) != 1) {
      MS_LOG(ERROR) << "code handler not only one" << std::string(py::str(reinterpret_cast<PyObject *>(co)));
    }
  }
  c->code_ = nullptr;
  c->codehub_.reset();
  delete c;
}

CodeExtra::CodeExtra() {
  this->stat_ = JitCompileResults::NEVER_COMPILE;
  this->codehub_ = std::make_shared<OptCodeHub>();
  this->tbs_ = std::make_shared<Tracebackes>();
  this->conf_ = std::make_shared<GraphJitConfig>();
  this->compile_count_ = 0;
  this->break_count_ = 0;
}

void CodeExtra::set_stat(CodeExtra::State s) {
  MS_EXCEPTION_IF_CHECK_FAIL(this != &CodeExtra::skip_, "can't set stat for skip marker");
  this->stat_ = s;
}

Py_ssize_t CodeExtra::GetCodeExtraIndex() {
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

CodeExtra *CodeExtra::GetCodeExtra(PyCodeObject *code) {
  Py_ssize_t index = GetCodeExtraIndex();
  if (index == -1) {
    return nullptr;
  }
  void *ptr = nullptr;
  _PyCode_GetExtra(reinterpret_cast<PyObject *>(code), index, &ptr);
  return reinterpret_cast<JitCompileResults *>(ptr);
}

void CodeExtra::SetCodeExtra(PyCodeObject *code, CodeExtra *ptr) {
  Py_ssize_t index = GetCodeExtraIndex();
  if (index == -1) {
    return;
  }
  _PyCode_SetExtra(reinterpret_cast<PyObject *>(code), index, ptr);
  if (PyErr_Occurred()) {
    throw py::error_already_set();
  }
}

CodeExtra *CodeExtra::GetCodeExtraWithAlloc(PyObject *code, bool alloc) {
  if (PyMethod_Check(code)) {
    code = PyMethod_GET_FUNCTION(code);
  }
  if (PyFunction_Check(code)) {
    code = PyFunction_GET_CODE(code);
  }
  if (!PyCode_Check(code)) {
    return nullptr;
  }
  Py_ssize_t index = CodeExtra::GetCodeExtraIndex();
  if (index == -1) {
    return nullptr;
  }
  CodeExtra *c = nullptr;
  if (!_PyCode_GetExtra(code, index, reinterpret_cast<void **>(&c))) {
    if (!alloc) {
      return c;
    }
    if (c != nullptr && c != &CodeExtra::skip_) {
      return c;
    }
    c = new CodeExtra();
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
