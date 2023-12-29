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

#define _GLIBCXX_ASSERTIONS 1

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
class GraphJitConfig;

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

  auto &GetLocals() { return locals; }
  auto &GetStacks() { return stack; }
  auto &GetClosures() { return cell_free; }

  void print() const;

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

  void SetRetVal(ValueNode *v) { ret_val_ = v; }
  ValueNode *GetRetVal() const { return ret_val_; }
  PyCodeObject *GetCodeObj() const { return reinterpret_cast<PyCodeObject *>(co_.ptr()); }
  const py::object &GetGlobals() const { return f_globals_; }

  void StopTraceAt(int bci, StopTraceReason reason) { stop_trace_info_ = {bci, reason}; }
  int GetStopTraceBci() const { return stop_trace_info_.bci; }
  StopTraceReason GetStopTraceReason() const { return stop_trace_info_.reason; }
  const char *GetModuleName() const { return module_name_; }

  auto &GetCFG() { return cfg_; }
  const auto &GetCFG() const { return cfg_; }
  const GraphJitConfig &Config() const { return conf_; }

  const FrameStates &GetFrame(int bci) const;
  void SetFrame(int bci, const FrameStates &f);
  auto &GetFrames() { return frame_states_; }
  const auto &GetFrames() const { return frame_states_; }
  Allocator &allocator() { return alloc_; }
  const std::vector<LoopInfo *> &loops() const { return loops_; }
  void AddLoop(LoopInfo *loop) { loops_.emplace_back(loop); }

  // only func name
  std::string GetCodeName() const {
    PyCodeObject *c = reinterpret_cast<PyCodeObject *>(co_.ptr());
    return Utils::GetPyName(c->co_name);
  }

  bool GuardValueNode(ValueNode *);
  TracePtr TraceValueNode(ValueNode *, int max_trace_depth = -1);
  int GetPruneBranchCount() const { return prune_branch_count_; }
  void SetPruneBranchCount(int count) { prune_branch_count_ = count; }
  const std::shared_ptr<OptCode> &GetGuard() const { return guard_; }
  void SetGuard(const std::shared_ptr<OptCode> &guard) { guard_ = guard; }

  // TODO: restore graph status at loop begin, clear trace values and operations and guards
  bool RestoreLoopStatus() { return false; }
  bool IsBreakAtLoop() const;
  const std::vector<ValueNode *> &GetTracedNodes() const { return traced_nodes_; }
  std::vector<ValueNode *> &GetTracedNodes() { return traced_nodes_; }

  void print(int depth = 0) const;
  std::string DumpLoops() const;

  void InstallToGlobal(const std::string &key, const py::object &value) {
    PyDict_SetItemString(f_globals_.ptr(), key.c_str(), value.ptr());
  }

 private:
  std::unique_ptr<CFG> cfg_;
  std::vector<LoopInfo *> loops_;

  // frame status
  std::map<int, std::unique_ptr<FrameStates>> frame_states_;
  std::vector<ValueNode *> traced_nodes_;

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
