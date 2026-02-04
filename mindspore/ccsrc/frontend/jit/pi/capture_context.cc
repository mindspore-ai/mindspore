/**
 * Copyright 2024-2025 Huawei Technologies Co.,Ltd
 *
 * Licensed under the Apache License,Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "frontend/jit/pi/capture_context.h"
#include <algorithm>
#include "frontend/jit/pi/external.h"
#include "frontend/jit/pi/jit_compile_results.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace pijit {

constexpr auto kModuleName = "mindspore";
constexpr auto kName = "__name__";

CaptureContext *CaptureContext::GetInstance() {
  static CaptureContext instance;
  return &instance;
}

void CaptureContext::RegisterSkipCode(PyCodeObject *co) { SetJitCompileResults(co, JitCompileResults::get_skip_jcr()); }

bool CaptureContext::IsSkip(const PyFrameWrapper &f) const {
  PyObject *globals = f.Globals().ptr();
  PyCodeObject *co = f.GetCode().ptr();
  return IsSkip(co, globals);
}

bool CaptureContext::IsSkip(PyCodeObject *co, PyObject *globals) const {
  if (!PyDict_Check(globals)) {
    MS_LOG(DEBUG) << "skip because of unknown module dict";
    return true;
  }
  PyObject *module_name = PyDict_GetItemString(globals, kName);
  if (PyErr_Occurred() || module_name == nullptr || !PyUnicode_Check(module_name)) {
    MS_LOG(DEBUG) << "skip because of unknown module name";
    return true;
  }
  std::string name = PyUnicode_AsUTF8(module_name);
  name = name.substr(0, name.find('.'));

  if (std::string("construct") == PyUnicode_AsUTF8(co->co_name)) {
    return false;
  }

  if (IsSkipCode(co, name)) {
    return true;
  }
  if (IsSkipModule(co, name)) {
    return true;
  }
  return false;
}

bool CaptureContext::IsSkipFile(const char *file) const {
  if (file == nullptr || file[0] == '\0') {
    return true;  // unknown file
  }
  std::string path = file;
  auto f = [&path](const std::string &skip) { return path.compare(0, std::min(path.size(), skip.size()), skip) == 0; };
  auto iter = std::find_if(skip_files_.begin(), skip_files_.end(), f);
  if (iter != skip_files_.end()) {
    MS_LOG(DEBUG) << "skip because of code at file " << (*iter);
    return true;  // skip dir
  }
  return false;
}

bool CaptureContext::IsSkipModule(PyCodeObject *co, const std::string &module_name) const {
  if (IsSkipFile(PyUnicode_AsUTF8(co->co_filename))) {
    return true;
  }
  if (!use_white_list_) {
    return false;
  }
  if (module_name == kModuleName) {
    return true;  // use white list to specify captured
  }
  // only capture known modules
  if (known_modules_.find(module_name) == known_modules_.end()) {
    MS_LOG(DEBUG) << "skip because of module '" << module_name << "' not in white list";
    return true;
  }
  return false;
}

static std::set<std::string> forbidden_pattern = {
  "__getattribute__",  // +
  "__getattr__",       // +
  "__setattr__",       // +
  "__delattr__",       // +
  "__repr__",          // +
  "__hash__",          // +
  // "__call__",          // +
  "__str__",        // +
  "__lt__",         // +
  "__le__",         // +
  "__eq__",         // +
  "__ne__",         // +
  "__gt__",         // +
  "__ge__",         // +
  "__iter__",       // +
  "__next__",       // +
  "__get__",        // +
  "__set__",        // +
  "__delete__",     // +
  "__init__",       // +
  "__new__",        // +
  "__del__",        // +
  "__await__",      // +
  "__aiter__",      // +
  "__anext__",      // +
  "__add__",        // +
  "__radd__",       // +
  "__sub__",        // +
  "__rsub__",       // +
  "__mul__",        // +
  "__rmul__",       // +
  "__mod__",        // +
  "__rmod__",       // +
  "__pow__",        // +
  "__rpow__",       // +
  "__neg__",        // +
  "__pos__",        // +
  "__abs__",        // +
  "__bool__",       // +
  "__invert__",     // +
  "__lshift__",     // +
  "__rlshift__",    // +
  "__rshift__",     // +
  "__rrshift__",    // +
  "__and__",        // +
  "__rand__",       // +
  "__xor__",        // +
  "__rxor__",       // +
  "__or__",         // +
  "__ror__",        // +
  "__int__",        // +
  "__float__",      // +
  "__iadd__",       // +
  "__isub__",       // +
  "__imul__",       // +
  "__imod__",       // +
  "__ipow__",       // +
  "__ilshift__",    // +
  "__irshift__",    // +
  "__iand__",       // +
  "__ixor__",       // +
  "__ior__",        // +
  "__floordiv__",   // +
  "__rfloordiv__",  // +
  "__truediv__",    // +
  "__rtruediv__",   // +
  "__ifloordiv__",  // +
  "__itruediv__",   // +
  "__index__",      // +
  "__matmul__",     // +
  "__rmatmul__",    // +
  "__imatmul__",    // +
  "__len__",        // +
  "__getitem__",    // +
  "__setitem__",    // +
  "__delitem__",    // +
  "__contains__",   // +
};

bool CaptureContext::IsSkipCode(PyCodeObject *co, const std::string &module_name) const {
  if (!use_white_list_) {
    return false;
  }
  std::string name = PyUnicode_AsUTF8(co->co_name);
  if (forbidden_pattern.find(name) != forbidden_pattern.end()) {
    MS_LOG(DEBUG) << "skip because of forbidden_pattern '" << PyUnicode_AsUTF8(co->co_name) << "' magic method";
    return true;
  }
  return false;
}

void CaptureContext::SetContext(const py::args &va, const py::kwargs &kw) {
  // parse arguments
  struct ContextArgument {
    PyObject *fn_;
    PyObject *config_;
    PyObject *wrapper_;
    PyObject *skip_codes_;
    PyObject *skip_files_;
  };
  static const char *kws[] = {"fn", "config", "wrapper", "skip_codes", "skip_files", nullptr};
  constexpr const char fmt[] = "|OO$OOO:pi_jit_set_context";
  ContextArgument args = {nullptr, nullptr, nullptr, nullptr, nullptr};
  if (!PyArg_ParseTupleAndKeywords(va.ptr(), kw.ptr(), fmt, const_cast<char **>(kws), &args.fn_, &args.config_,
                                   &args.wrapper_, &args.skip_codes_, &args.skip_files_)) {
    throw py::error_already_set();  // arguments is invalid
  }

  // set context
  this->SetConfig(args.config_);
  this->SetWrapper(args.wrapper_);
  this->SetSkipCodes(args.skip_codes_);
  this->SetSkipFiles(args.skip_files_);
  this->SetFunction(args.fn_);

  if (args.fn_ == nullptr || wrapped_func_ == nullptr) {
    return;
  }
  auto jcr = GetJitCompileResults(args.fn_);
  if (jcr == nullptr || jcr == JitCompileResults::get_skip_jcr()) {
    pi_jit_should_compile(args.fn_);
    jcr = GetJitCompileResults(args.fn_);
  }
  if (jcr == nullptr) {
    MS_LOG(ERROR) << "mark compile failed. Thread is " << std::this_thread::get_id() << " function is "
                  << std::string(py::str(PyFunction_GET_CODE(args.fn_)));
    this->Disable();
    return;
  }
  if (config_ == nullptr) {
    config_ = jcr->conf();
  } else {
    jcr->set_conf(config_);
  }
}

void CaptureContext::SetFunction(PyObject *fn) {
  if (fn == nullptr) {
    return;
  }
  if (fn == Py_None) {
    this->Disable();
    return;
  }
  if (PyFunction_Check(fn)) {
    this->Enable(fn);
    return;
  }
  throw py::type_error("the arguments 'fn' must be function or None");
}

void CaptureContext::SetConfig(PyObject *config) {
  if (config == nullptr) {
    return;
  }
  if (PyDict_Check(config)) {
    this->set_config(std::make_shared<GraphJitConfig>(py::cast<py::object>(config)));
    return;
  }
  throw py::type_error("the arguments 'config' must be dict");
}

void CaptureContext::SetWrapper(PyObject *wrapper) {
  if (wrapper == nullptr) {
    return;
  }
  if (PyFunction_Check(wrapper)) {
    wrapper_code_ = PyFunction_GET_CODE(wrapper);
    return;
  }
  throw py::type_error("the arguments 'wrapper' must be function");
}

void CaptureContext::SetSkipCodes(PyObject *skip_codes) {
  if (skip_codes == nullptr) {
    return;
  }
  if (PyTuple_Check(skip_codes)) {
    Py_ssize_t i = 0;
    Py_ssize_t size = PyTuple_GET_SIZE(skip_codes);
    for (; i < size && PyCode_Check(PyTuple_GET_ITEM(skip_codes, i)); ++i) {
      this->RegisterSkipCode(reinterpret_cast<PyCodeObject *>(PyTuple_GET_ITEM(skip_codes, i)));
    }
    if (i == size) {
      return;
    }
  }
  throw py::type_error("the arguments 'skip_code' must be tuple of code");
}

void CaptureContext::SetSkipFiles(PyObject *skip_files) {
  if (skip_files == nullptr) {
    return;
  }
  if (PyTuple_Check(skip_files)) {
    Py_ssize_t i = 0;
    Py_ssize_t size = PyTuple_GET_SIZE(skip_files);
    MS_LOG(DEBUG) << "skip files:";
    for (; i < size && PyUnicode_Check(PyTuple_GET_ITEM(skip_files, i)); ++i) {
      auto file_pattern = PyUnicode_AsUTF8(PyTuple_GET_ITEM(skip_files, i));
      MS_LOG(DEBUG) << file_pattern;
      this->RegisterSkipFile(file_pattern);
    }
    if (i == size) {
      return;
    }
  }
  throw py::type_error("the arguments 'skip_files' must be tuple of str");
}

bool CaptureContext::IsEnable() const { return stat_ == kEnable; }

void CaptureContext::Enable(PyObject *top_function) {
  if (stat_ == kDefault) {
    stat_ = kEnable;
    wrapped_func_ = top_function;
  }
}

void CaptureContext::Disable() {
  if (stat_ == kEnable) {
    stat_ = kDefault;
    wrapped_func_ = nullptr;
    config_ = nullptr;
  }
}
}  // namespace pijit
}  // namespace mindspore
