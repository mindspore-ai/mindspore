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
#ifndef MINDSPORE_PI_JIT_CODE_EXTRA_H
#define MINDSPORE_PI_JIT_CODE_EXTRA_H

#include <unordered_map>
#include <map>
#include <list>
#include <string>
#include <memory>
#include "pybind11/pybind11.h"
#include "pipeline/jit/pi/graph_guard/cache.h"
#include "pipeline/jit/pi/pi_jit_config.h"
#include "pipeline/jit/pi/utils/utils.h"

namespace mindspore {
namespace pijit {

namespace py = pybind11;

// record the inline and stop trace reason and other information for each graph
class Tracebackes {
 public:
  struct Tracebacke {
    std::string func_name_;
    std::string changed_func_;
    int code_size_;
    bool is_graph_mode_;
  };

  struct InlineInfo {
    std::string func_name_;
    std::string inline_name_;
    std::string root_name_;
    InlineReason res = InlineReason::kInlineUnknown;
    int code_size_;
    int depth;
    int line;
  };
  Tracebackes() = default;
  Tracebackes(const std::string &raw_func_name, const std::string &raw_func_info_name, int raw_code_size)
      : raw_func_name_(raw_func_name), raw_func_info_name_(raw_func_info_name), raw_code_size_(raw_code_size) {}
  ~Tracebackes() { Clear(); }
  void Clear() {
    tbs_.clear();
    stop_trace_res_.clear();
    inline_infos_.clear();
  }

  std::string raw_func_name() const { return raw_func_name_; }
  void PushTbs(const Tracebacke &tb) { tbs_.push_back(tb); }
  void PushStopTraceRes(const std::string &func_name, StopTraceReason res) { stop_trace_res_.emplace(func_name, res); }
  void PushInlineInfo(InlineInfo info);
  void DumpInlineInfo(std::stringstream &os, const std::string &func_name) const;
  int FindMaxNameLength(const std::list<Tracebacke> &tbs) const;
  std::string Dump(bool is_all = false) const;
  std::string DumpSummary() const;
  std::string GetStopTrace() {
    std::string res;
    for (auto item : stop_trace_res_) {
      std::string item_str;
      if (res.size() == 0) {
        res += std::string("\"") + item.first + "\":" + std::to_string(SizeToInt(item.second));
      } else {
        res += std::string("\"") + item.first + "\":" + std::to_string(SizeToInt(item.second));
      }
    }
    res = std::string("{") + res + std::string("}");
    return res;
  }

 private:
  std::string raw_func_name_;
  std::string raw_func_info_name_;
  int raw_code_size_;
  std::list<Tracebacke> tbs_;
  // <func_name, stop_trace_reason>
  std::unordered_map<std::string, StopTraceReason> stop_trace_res_;
  // <root_func_name, InlineInfo>
  std::map<std::string, std::list<InlineInfo>> inline_infos_;
};

class CodeExtra {
 public:
  static CodeExtra *GetCodeExtra(PyCodeObject *co);
  static CodeExtra *GetCodeExtraWithAlloc(PyObject *code, bool alloc);
  static void SetCodeExtra(PyCodeObject *co, CodeExtra *ce);

  static CodeExtra skip_;

 private:
  static Py_ssize_t GetCodeExtraIndex();
  static void FreeCallback(void *);
  static Py_tss_t *tss_;

 public:
  enum State {
    NEVER_COMPILE = 0,
    GRAPH_CANDIDATE,
    GRAPH_CAPTURED,
    GRAPH_BUILDING,
    GRAPH_CALLABLE,
  };

  State stat() const { return stat_; }
  PyFrameObject *origin_frame() const { return reinterpret_cast<PyFrameObject *>(compile_frame_.ptr()); }
  const auto &input_signature() const { return input_signature_; }
  const auto &code() const { return code_; }
  const auto &codehub() const { return codehub_; }
  const auto &tbs() const { return tbs_; }
  const auto &conf() const { return conf_; }
  int &compile_count() { return compile_count_; }
  int &break_count() { return break_count_; }

  void set_stat(State s);
  void set_input_signature(const py::object &sig) { input_signature_ = sig; }
  void set_origin_frame(PyFrameObject *f) { compile_frame_ = py::cast<py::object>(reinterpret_cast<PyObject *>(f)); }
  void set_code(const OptCodePtr &p) { code_ = p; }
  void set_codehub(const OptCodeHubPtr &p) { codehub_ = p; }
  void set_tbs(const std::shared_ptr<Tracebackes> &t) { tbs_ = t; }
  void set_conf(const std::shared_ptr<GraphJitConfig> &c) { conf_ = c; }

  int IncCodeCount() { return compile_count_++; }

 private:
  CodeExtra();
  ~CodeExtra() = default;

  py::object compile_frame_;
  py::object input_signature_;
  State stat_;

  // compiler output
  OptCodePtr code_;
  OptCodeHubPtr codehub_;
  std::shared_ptr<Tracebackes> tbs_;
  std::shared_ptr<GraphJitConfig> conf_;
  int compile_count_;
  int break_count_;
};
using JitCompileResults = CodeExtra;

inline CodeExtra *getJitCompileResults(PyObject *func, bool alloc) {
  return CodeExtra::GetCodeExtraWithAlloc(func, alloc);
}

}  // namespace pijit
}  // namespace mindspore

#endif
