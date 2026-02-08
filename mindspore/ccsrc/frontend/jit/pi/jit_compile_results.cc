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
#include "frontend/jit/pi/jit_compile_results.h"

namespace mindspore {
namespace pijit {

static Py_tss_t *tss_ = nullptr;

JitCompileResults *JitCompileResults::get_skip_jcr() {
  static JitCompileResults skip(true);
  return &skip;
}

void JitCompileResults::FreeCallback(void *ptr) {
  // maybe nullptr if other module use _PyEval_RequestCodeExtraIndex
  if (ptr == nullptr || ptr == get_skip_jcr()) {
    return;
  }
  JitCompileResults *c = reinterpret_cast<JitCompileResults *>(ptr);
  delete c;
}

Py_ssize_t JitCompileResults::InitIndex() {
  if (tss_ == nullptr) {
    MS_LOG(INFO) << "Initialize jcr key. Thread is " << std::this_thread::get_id();
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

JitCompileResults::JitCompileResults(bool skip)
    : stat_(JitCompileResults::NEVER_COMPILE), cache_(this), compile_count_(0), break_count_(0) {
  if (skip) {
    return;
  }
  this->tbs_ = std::make_shared<Traceback>();
  this->conf_ = std::make_shared<GraphJitConfig>();
}

void JitCompileResults::set_stat(JitCompileResults::State s) {
  MS_EXCEPTION_IF_CHECK_FAIL(this != get_skip_jcr(), "can't change the skip marker stat");
  this->stat_ = s;
}

JitCompileResults *JitCompileResults::Create(PyCodeObject *co) {
  Py_ssize_t index = InitIndex();
  if (index == -1) {
    return nullptr;
  }
  PyObject *code = reinterpret_cast<PyObject *>(co);
  JitCompileResults *c = nullptr;
  if (!_PyCode_GetExtra(code, index, reinterpret_cast<void **>(&c))) {
    if (c != nullptr && c != get_skip_jcr()) {
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

bool JitCompileResults::Clear(PyCodeObject *co) {
  Py_ssize_t index = InitIndex();
  if (index == -1) {
    return false;
  }
  PyObject *code = reinterpret_cast<PyObject *>(co);
  JitCompileResults *c = nullptr;
  if (!_PyCode_GetExtra(code, index, reinterpret_cast<void **>(&c))) {
    if (c == nullptr || c == get_skip_jcr()) {
      return false;
    }
    if (!_PyCode_SetExtra(code, index, nullptr)) {
      return true;
    }
    delete c;
  }
  PyErr_Clear();
  return false;
}
}  // namespace pijit
}  // namespace mindspore
