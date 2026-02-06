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
#ifndef MINDSPORE_CCSRC_FRONTEND_JIT_PI_CAPTURE_CONTEXT_H
#define MINDSPORE_CCSRC_FRONTEND_JIT_PI_CAPTURE_CONTEXT_H

#include <memory>
#include <set>
#include <string>
#include "frontend/jit/pi/pi_jit_config.h"
#include "frontend/jit/pi/python_adapter/py_frame.h"

namespace mindspore {
namespace pijit {

class CaptureContext {
 public:
  enum Stat {
    kDefault,
    kEnable,
    kDisable,
  };

  // make a scope disable context. can't enable until scope exit
  class DisableScope {
   public:
    friend class CaptureContext;
    DisableScope() : stat_(CaptureContext::GetInstance()->stat_) { CaptureContext::GetInstance()->stat_ = kDisable; }
    ~DisableScope() { CaptureContext::GetInstance()->stat_ = stat_; }

   private:
    Stat stat_;
  };

  static CaptureContext *GetInstance();

  // getter
  const auto &config() const { return config_; }
  const auto &wrapper_code() const { return wrapper_code_; }
  const auto &wrapped_func() const { return wrapped_func_; }
  const auto &known_modules() const { return known_modules_; }

  // setter
  void set_config(const std::shared_ptr<GraphJitConfig> &c) { config_ = c; }
  void set_use_white_list(bool config) { use_white_list_ = config; }

  // helper
  bool IsEnable() const;
  void Enable(PyObject *top_function);
  void Disable();

  void AddKnownModule(const std::string &name) { known_modules_.insert(name); }

  // register the rule of skip code
  void RegisterSkipFile(const std::string &name) { skip_files_.insert(name); }
  void RegisterSkipCode(PyCodeObject *co);

  // check the code need skip
  bool IsSkip(const PyFrameWrapper &f) const;
  bool IsSkip(PyCodeObject *co, PyObject *globals) const;

  // set context attribute
  void SetContext(const py::args &va, const py::kwargs &kw);

 private:
  CaptureContext() = default;

  // parse and set arguments helper, check arguments is valid
  void SetSkipCodes(PyObject *);
  void SetSkipFiles(PyObject *);
  void SetWrapper(PyObject *);
  void SetConfig(PyObject *);
  void SetFunction(PyObject *);

  // white list check
  bool IsSkipModule(PyCodeObject *co, const std::string &module_name) const;
  bool IsSkipFile(const char *file) const;
  bool IsSkipCode(PyCodeObject *co, const std::string &module_name) const;

  // shared config at context
  std::shared_ptr<GraphJitConfig> config_;

  // skip rules
  std::set<std::string> skip_files_;
  std::set<std::string> known_modules_;

  // wrapper info
  void *wrapper_code_;
  void *wrapped_func_;

  Stat stat_;

  // config skip rule
  bool use_white_list_{true};
};
}  // namespace pijit
}  // namespace mindspore

#endif
