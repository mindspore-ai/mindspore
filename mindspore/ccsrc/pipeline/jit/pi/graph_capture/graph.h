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
#ifndef MINDSPORE_CCSRC_PIPELINE_GRAPH_JIT_GRAPH_CAPTURE_GRAPH_H
#define MINDSPORE_CCSRC_PIPELINE_GRAPH_JIT_GRAPH_CAPTURE_GRAPH_H

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "pipeline/jit/pi/graph_capture/cfg.h"
#include "pipeline/jit/pi/graph_capture/loop.h"
#include "pipeline/jit/pi/graph_capture/node.h"
#include "pipeline/jit/pi/utils/allocator.h"
#include "pipeline/jit/pi/graph_guard/trace.h"

namespace mindspore {
namespace jit {
namespace graph {
class OptCode;

class FrameStates {
 public:
  ValueNode *Local(int i) const {
    MS_ASSERT((int)locals.size() > i);
    return locals[i];
  }
  void SetLocal(int i, ValueNode *v) {
    MS_ASSERT((int)locals.size() > i);
    locals[i] = v;
  }

  CellVarNode *Closure(int i) const {
    MS_ASSERT((int)cell_free.size() > i);
    return cell_free[i];
  }
  void SetClosure(int i, CellVarNode *v) {
    MS_ASSERT((int)cell_free.size() > i);
    cell_free[i] = v;
  }

  ValueNode *&Peek(int p) {
    MS_ASSERT((int)stack.size() > p);
    return stack[stack.size() - p - 1];
  }
  ValueNode *Pop() {
    MS_ASSERT(stack.size() > 0);
    auto r = stack[stack.size() - 1];
    stack.pop_back();
    return r;
  }
  void Popn(int n) {
    for (int i = 0; i < n; i++) {
      Pop();
    }
  }
  void Push(ValueNode *i) { stack.push_back(i); }

  void Rot(int i) {
    MS_ASSERT((int)stack.size() - i >= 0);
    ValueNode *v = Pop();
    stack.insert(stack.end() - i, v);
  }

  void ResizeLocal(int i) {
    MS_ASSERT((int)locals.size() <= i);
    locals.resize(i, &ValueNode::UnboundLocal);
  }
  void ResizeClosure(int i) {
    MS_ASSERT((int)cell_free.size() <= i);
    cell_free.resize(i);
  }

  ValueNode *GetCondition() { return cond.first; }
  bool ConditionIsTrue() { return cond.second; }
  void SetCondition(ValueNode *c) { cond.first = c; }
  void SetConditionIsTrue(bool c) { cond.second = c; }

  const auto &GetLocals() const { return locals; }
  const auto &GetStacks() const { return stack; }
  const auto &GetClosures() const { return cell_free; }

  void print();

 private:
  std::vector<ValueNode *> stack;
  std::vector<ValueNode *> locals;
  std::vector<CellVarNode *> cell_free;
  std::pair<ValueNode *, bool> cond;  // the condition come to this block
};

class Graph {
 public:
  Graph(PyCodeObject *co, PyObject *globals, const GraphJitConfig &conf);
  virtual ~Graph() {}

  void Init();
  const std::map<int, Instr *> &instr_map() const { return instrs_; }
  const std::vector<InstrNode *> &GetInstrs() const { return instr_nodes_; }
  void SetInstr(int bci, InstrNode *n) { instr_nodes_[bci] = (n); }
  void AddInstr(InstrNode *n) { instr_nodes_.push_back(n); }
  Block *GetBlockByBci(int bci) const;

  void SetRetVal(ValueNode *v) { ret_val_ = v; }
  ValueNode *GetRetVal() const { return ret_val_; }
  PyCodeObject *GetCodeObj() const { return reinterpret_cast<PyCodeObject *>(co_.ptr()); }
  const py::object &GetGlobals() const { return f_globals_; }

  void StopTraceAt(int bci, StopTraceReason reason) {
    MS_EXCEPTION_IF_CHECK_FAIL(bci >= 0 && bci < static_cast<int>(instr_nodes_.size()), "bci out of range !");
    stop_trace_info_ = {bci, reason};
  }
  auto GetStopTraceAt() const { return stop_trace_info_.bci == -1 ? nullptr : instr_nodes_[stop_trace_info_.bci]; }
  bool GetLoopInfo() const { return stop_trace_info_.reason == StopTraceReason::kStopTraceLoop_Unsupported; }
  StopTraceReason GetStopTraceReason() const { return stop_trace_info_.reason; }
  const FrameStates &GetLastFrame() const { return GetFrame(stop_trace_info_.bci); }
  const char *GetModuleName() const { return module_name_; }

  int GetNlocals() const { return co_.ptr() ? reinterpret_cast<PyCodeObject *>(co_.ptr())->co_nlocals + 1 : 0; }
  int GetExtraLocalIndex() const { return co_.ptr() ? reinterpret_cast<PyCodeObject *>(co_.ptr())->co_nlocals : 0; }

  int GetStackSize() const;
  auto &GetCFG() { return cfg_; }
  const GraphJitConfig &Config() const { return conf_; }

  const FrameStates &GetFrame(int bci) const;
  void SetFrame(int bci, const FrameStates &f);
  bool FindFrame(int bci) const { return frame_states_[bci].get(); }
  Allocator &allocator() { return alloc_; }
  const std::vector<LoopInfo *> &loops() const { return loops_; }
  void AddLoop(LoopInfo *loop) { loops_.emplace_back(loop); }

  // only func name
  std::string GetCodeName() const {
    PyCodeObject *c = reinterpret_cast<PyCodeObject *>(co_.ptr());
    return Utils::GetPyName(c->co_name);
  }

  std::string getCodeInfoName() const { return py::str(co_.ptr()).cast<std::string>(); }

  bool GuardValueNode(ValueNode *);
  TracePtr TraceValueNode(ValueNode *, int max_trace_depth = -1);
  int GetPruneBranchCount() const { return prune_branch_count_; }
  void SetPruneBranchCount(int count) { prune_branch_count_ = count; }
  const std::shared_ptr<OptCode> &GetGuard() { return guard_; }
  void SetGuard(const std::shared_ptr<OptCode> &guard) { guard_ = guard; }

  void print(int depth = 0) const;
  std::string DumpLoops() const;
  void Reset();

  void InstallToGlobal(const std::string &key, const py::object &value) {
    PyDict_SetItemString(f_globals_.ptr(), key.c_str(), value.ptr());
  }

 private:
  std::unique_ptr<CFG> cfg_;
  std::vector<LoopInfo *> loops_;

  std::map<int, Instr *> instrs_;
  std::map<int, Block *> bb_cache_;
  // bytecode copy, maybe changed to fit in trace
  std::vector<InstrNode *> instr_nodes_;

  // frame status
  std::vector<std::unique_ptr<FrameStates>> frame_states_;

  // return value
  ValueNode *ret_val_;

  // the traced code object
  py::object co_;

  // globals that may be used by frame when the tracer start
  py::object f_globals_;

  const char *module_name_;

  struct StopTraceInfo {
    int bci;  // trace stopped bci
    StopTraceReason reason;
  } stop_trace_info_;

  Allocator alloc_;

  const GraphJitConfig &conf_;

  std::shared_ptr<OptCode> guard_;
  int prune_branch_count_;
};
}  // namespace graph
}  // namespace jit
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PIPELINE_GRAPH_JIT_GRAPH_CAPTURE_GRAPH_H
