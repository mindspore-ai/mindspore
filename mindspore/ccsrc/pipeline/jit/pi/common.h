/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_PIPELINE_JIT_GRAPH_JIT_COMMON_H
#define MINDSPORE_CCSRC_PIPELINE_JIT_GRAPH_JIT_COMMON_H

#define PY_SSIZE_T_CLEAN
#include <functional>
#include <list>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>
#include "pybind11/pybind11.h"
#include "pipeline/jit/pi/pydef.h"
#include "pipeline/jit/pi/graph_guard/cache.h"
#include "pipeline/jit/pi/pi_jit_config.h"
#include "pipeline/jit/pi/utils/utils.h"

namespace mindspore {
namespace jit {
namespace graph {
namespace py = pybind11;
using NativeFunc = std::function<PyObject *(PyObject *, PyObject *)>;
using InlineInfoKey = std::tuple<std::string, std::string, int>;
using InlineInfoChain = std::pair<std::string, std::string>;

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
        res += std::string("\"") + item.first + "\":" + std::to_string(static_cast<int>(item.second));
      } else {
        res += std::string("\"") + item.first + "\":" + std::to_string(static_cast<int>(item.second));
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

// shouldn't save this object, must get it by 'getJitCompileResults'
// python call free function of this struct while the onwer (pyobject) is freed
typedef struct CodeExtra {
  // sub-graph trees
  CodeExtra *parent_;

  std::vector<CodeExtra *> children_;

  PyFrameObject *origin_frame_;  // frame object

  enum State {
    NEVER_COMPILE = 0,
    GRAPH_CANDIDATE,
    GRAPH_CAPTURED,
    GRAPH_BUILDING,
    GRAPH_CALLABLE,
  } stat;

  // compiler output
  OptCodePtr code;

  // code cache
  mindspore::jit::graph::OptCodeHubPtr codehub;

  std::shared_ptr<Tracebackes> tbs;

  std::shared_ptr<GraphJitConfig> conf;

  int IncCodeCount() { return compile_count_++; }
  int compile_count_;
  int break_count_;
} JitCompileResults;

JitCompileResults *getJitCompileResults(PyObject *code, bool alloc = true);
std::vector<py::object> PackArgs(const PyFrameObject *frame);
}  // namespace graph
}  // namespace jit
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PIPELINE_JIT_JITBYTECODE_COMMON_H
