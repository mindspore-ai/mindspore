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
#ifndef MINDSPORE_CCSRC_FRONTEND_JIT_PI_JIT_COMPILE_RESULTS_H
#define MINDSPORE_CCSRC_FRONTEND_JIT_PI_JIT_COMPILE_RESULTS_H

#include <unordered_map>
#include <map>
#include <list>
#include <string>
#include <memory>
#include "frontend/jit/pi/python_adapter/py_frame.h"
#include "frontend/jit/pi/graph_guard/cache.h"
#include "frontend/jit/pi/pi_jit_config.h"
#include "frontend/jit/pi/utils/inline_reason.h"
#include "frontend/jit/pi/utils/stop_trace_reason.h"

namespace mindspore {
namespace pijit {

namespace py = pybind11;

// record the inline and stop trace reason and other information for each graph
class Traceback {
 public:
  struct Element {
    std::string func_name_;
    std::string changed_func_;
    int code_size_;
    bool is_graph_mode_;
  };

  Traceback() = default;
  Traceback(const std::string &raw_func_name, const std::string &raw_func_info_name, int raw_code_size)
      : raw_func_name_(raw_func_name), raw_func_info_name_(raw_func_info_name), raw_code_size_(raw_code_size) {}
  ~Traceback() { Clear(); }
  void Clear() {
    tbs_.clear();
    stop_trace_res_.clear();
  }

  std::string raw_func_name() const { return raw_func_name_; }
  void PushTbs(const Element &tb) { tbs_.push_back(tb); }
  void PushStopTraceRes(const std::string &func_name, StopTraceReason res) { stop_trace_res_.emplace(func_name, res); }
  int FindMaxNameLength(const std::list<Element> &tbs) const;
  std::string Dump(bool is_all = false) const;
  std::string DumpSummary() const;

 private:
  std::string raw_func_name_;
  std::string raw_func_info_name_;
  int raw_code_size_;
  std::list<Element> tbs_;
  // <func_name, stop_trace_reason>
  std::unordered_map<std::string, StopTraceReason> stop_trace_res_;
};

class JitCompileResults {
 public:
  static JitCompileResults *Create(PyCodeObject *co);
  static bool Clear(PyCodeObject *co);
  static Py_ssize_t InitIndex();
  static JitCompileResults *get_skip_jcr();

 private:
  static void FreeCallback(void *);

 public:
  enum State {
    NEVER_COMPILE = 0,
    GRAPH_CANDIDATE,
    GRAPH_CAPTURED,
    GRAPH_BUILDING,
    GRAPH_CALLABLE,
  };

  State stat() const { return stat_; }
  const auto &origin_frame() const { return compile_frame_; }
  const auto &codehub() { return cache_.code_hub(); }
  const auto &code() const { return cache_.code(); }
  const auto &tbs() const { return tbs_; }
  const auto &conf() const { return conf_; }
  int &compile_count() { return compile_count_; }
  int &break_count() { return break_count_; }
  const auto &cache() const { return cache_; }
  bool is_for_loop_body_wrapper() { return is_for_loop_body_wrapper_; }
  const auto &enable_dynamic_dict() const { return enable_dynamic_dict_; }

  void set_stat(State s);
  void set_origin_frame(PyFrameWrapper f) { compile_frame_ = f; }
  void set_code(const OptCodePtr &p) { cache_.set_code(p); }
  void set_tbs(const std::shared_ptr<Traceback> &t) { tbs_ = t; }
  void set_conf(const std::shared_ptr<GraphJitConfig> &c) { conf_ = c; }
  void set_is_for_loop_body_wrapper(bool is_for_loop_body_wrapper) {
    is_for_loop_body_wrapper_ = is_for_loop_body_wrapper;
  }
  void set_enable_dynamic_dict(const py::object &enable_dynamic_dict) { enable_dynamic_dict_ = enable_dynamic_dict; }

  int IncCodeCount() { return compile_count_++; }
  void ClearCache() { cache_.Clear(); }
  void CacheFailGuard(const PyFrameWrapper &f) { cache_.CollectFailGuard(f); }

 private:
  explicit JitCompileResults(bool skip = false);
  ~JitCompileResults() = default;

  PyFrameWrapper compile_frame_;
  py::object enable_dynamic_dict_{py::none()};
  State stat_;

  // compiler output
  CodeCache cache_;

  std::shared_ptr<Traceback> tbs_;
  std::shared_ptr<GraphJitConfig> conf_;
  int compile_count_;
  int break_count_;
  bool is_for_loop_body_wrapper_{false};
};

inline JitCompileResults *GetJitCompileResults(PyCodeObject *code) {
  Py_ssize_t index = JitCompileResults::InitIndex();
  if (index != -1) {
    void *ptr = nullptr;
    _PyCode_GetExtra(reinterpret_cast<PyObject *>(code), index, &ptr);
    return reinterpret_cast<JitCompileResults *>(ptr);
  }
  return nullptr;
}

inline JitCompileResults *GetJitCompileResults(const py::handle &h) {
  PyObject *code = h.ptr();
  code = PyMethod_Check(code) ? PyMethod_GET_FUNCTION(code) : code;
  code = PyFunction_Check(code) ? PyFunction_GET_CODE(code) : code;
  return PyCode_Check(code) ? GetJitCompileResults(reinterpret_cast<PyCodeObject *>(code)) : nullptr;
}

inline void SetJitCompileResults(PyCodeObject *code, JitCompileResults *ptr) {
  Py_ssize_t index = JitCompileResults::InitIndex();
  if (index != -1) {
    _PyCode_SetExtra(reinterpret_cast<PyObject *>(code), index, ptr);
  }
}

inline JitCompileResults *CreateJitCompileResults(const py::handle &h) {
  PyObject *code = h.ptr();
  code = PyMethod_Check(code) ? PyMethod_GET_FUNCTION(code) : code;
  code = PyFunction_Check(code) ? PyFunction_GET_CODE(code) : code;
  return PyCode_Check(code) ? JitCompileResults::Create(reinterpret_cast<PyCodeObject *>(code)) : nullptr;
}

}  // namespace pijit
}  // namespace mindspore

#endif
