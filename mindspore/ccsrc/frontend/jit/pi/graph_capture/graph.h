/**
 * Copyright 2023-2025 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_FRONTEND_JIT_PI_GRAPH_CAPTURE_GRAPH_H
#define MINDSPORE_CCSRC_FRONTEND_JIT_PI_GRAPH_CAPTURE_GRAPH_H

#include <exception>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "frontend/jit/pi/graph_capture/cfg.h"
#include "frontend/jit/pi/graph_capture/node.h"
#include "frontend/jit/pi/utils/allocator.h"
#include "frontend/jit/pi/graph_guard/trace.h"
#include "frontend/jit/pi/graph_guard/guard.h"
#include "frontend/jit/pi/graph_capture/side_effect.h"
#include "frontend/jit/pi/utils/stop_trace_reason.h"

namespace mindspore {
namespace pijit {

class OptCode;
class GraphJitConfig;
class FuncGraphBuilder;
class GuardBuilder;

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

  ValueNode *Peek(int p) const {
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
  int Size() { return stack.size(); }

  void Rot(int i) {
    MS_ASSERT(i >= 0 && (int)stack.size() - i >= 0);
    ValueNode *v = Pop();
    stack.insert(stack.end() - i, v);
  }

  void Swap(int i) {
    MS_ASSERT(i >= 0 && (int)stack.size() - i >= 0);
    auto top_idx = stack.size() - 1;
    std::swap(stack[top_idx], stack[top_idx - i]);
  }

  void ResizeLocal(int i) {
    MS_ASSERT((int)locals.size() <= i);
    locals.resize(i, &ValueNode::kUnboundLocal);
  }
  void ResizeClosure(int i) {
    MS_ASSERT((int)cell_free.size() <= i);
    cell_free.resize(i);
  }

  const auto &GetLocals() const { return locals; }
  const auto &GetStacks() const { return stack; }
  const auto &GetClosures() const { return cell_free; }

  auto &GetLocals() { return locals; }
  auto &GetStacks() { return stack; }
  auto &GetClosures() { return cell_free; }

  std::string ToString() const;

 private:
  std::vector<ValueNode *> stack;
  std::vector<ValueNode *> locals;
  std::vector<CellVarNode *> cell_free;
};

class Graph {
 public:
  struct BreakInfo {
    Instr *break_point_;
    ValueNode *break_point_node_;
    std::vector<int> alive_locals_;
    std::vector<ValueNode *> alive_nodes_;  // Does not include side-effect alive nodes!
    int bci_;
    StopTraceReason reason_;
  };

  struct ExpandParamInfo {
    ValueNode *node_;
    std::vector<const AObject *> elements_;
  };

  Graph(PyCodeObject *co, PyObject *globals, const GraphJitConfig &conf);
  virtual ~Graph();

  const BreakInfo &break_info() const { return break_info_; }
  void set_break_info(const BreakInfo &info) { break_info_ = info; }

  ValueNode *GetGeneratorResult() const { return generator_result_; }
  void SetGeneratorResult(ValueNode *generator_result) { generator_result_ = generator_result; }

  void SetRetVal(ValueNode *v) { ret_val_ = v; }
  ValueNode *GetRetVal() const { return ret_val_; }
  PyCodeObject *GetCodeObj() const { return reinterpret_cast<PyCodeObject *>(co_.ptr()); }
  const py::object &GetGlobals() const { return f_globals_; }

  /// @throws GraphBreakException if fullgraph=true (graph break is not allowed)
  void StopTraceAt(int bci, StopTraceReason reason, const std::vector<std::string> &hints = {});
  int GetStopTraceBci() const { return break_info_.bci_; }
  StopTraceReason GetStopTraceReason() const { return break_info_.reason_; }
  const char *GetModuleName() const { return module_name_; }

  auto &GetCFG() { return cfg_; }
  const auto &GetCFG() const { return cfg_; }
  const GraphJitConfig &Config() const { return conf_; }

  const FrameStates &GetFrame(int bci) const;
  void SetFrame(int bci, const FrameStates &f);

  Allocator &allocator() { return alloc_; }
  ValueNode *NewValueNode(AObject *, int op, int arg, const std::vector<ValueNode *> &inputs = {},
                          const std::string &name = "");

  CellVarNode *NewCellNode(AObject *, int op, int arg, const std::vector<ValueNode *> &inputs = {},
                           const std::string &name = "");

  ParamNode *NewParamNode(AObject *, int index, const std::string &name = "");
  CallNode *NewCallNode(int op, int arg, const std::vector<ValueNode *> &);

  // only func name
  std::string GetCodeName() const {
    PyCodeObject *c = reinterpret_cast<PyCodeObject *>(co_.ptr());
    if (c != nullptr && c->co_name != nullptr) {
      return py::str(c->co_name);
    }
    return "";
  }

  const std::vector<ValueNode *> &GetParameters() const { return params_; }

  void GuardParameter(ValueNode *param);
  void GuardGlobal(ValueNode *global_value);
  void GuardAttribute(ValueNode *attr_value);
  bool GuardValueNode(ValueNode *, GuardLevel level = GuardLevel::GEqual);
  bool GuardValueNodeClosure(ValueNode *, GuardLevel level = GuardLevel::GDeduce);
  bool GuardType(ValueNode *);
  bool GuardSequenceNodeLength(ValueNode *, Py_ssize_t);
  bool GuardInlinedFunc(CallNode *call_node);

  TracePtr TraceValueNode(ValueNode *, int max_trace_depth = -1);
  std::vector<TracePtr> TraceValueNodeClosure(ValueNode *, bool *ret, int max_trace_depth = -1);
  const std::shared_ptr<OptCode> &GetGuardManager() const;
  void SetGuard(const std::shared_ptr<OptCode> &guard);
  void RemoveAllGuardItems() const;
  std::map<const AObject *, ExpandParamInfo> &GetExpandParamInfo() { return expand_param_info_; }

  // (chaiyouheng): restore graph status at loop begin, clear trace values and operations and guards
  bool RestoreLoopStatus() const { return false; }
  bool IsBreakAtLoop() const;
  bool ShouldNeverCompile() const;
  const std::vector<ValueNode *> &GetTracedNodes() const { return traced_nodes_; }
  std::vector<ValueNode *> &GetTracedNodes() { return traced_nodes_; }

  std::string ToString(int depth = 0) const;

  void SetParent(Graph *parent) { parent_ = parent; }
  Graph *GetParent() const { return parent_; }

  const std::shared_ptr<SideEffect> &GetSideEffect() const { return side_effect_; }
  void SetSideEffect(const std::shared_ptr<SideEffect> &handler) { side_effect_ = handler; }
  const std::shared_ptr<SideEffectHandler> &GetSideEffectHandler() const { return side_effect_handler_; }

  std::vector<ValueNode *> CollectAliveNode(int bci, std::vector<int> *ids = nullptr) const;
  // collect alive node, clear the bit if alive local is unbound
  static std::vector<ValueNode *> CollectAliveNode(const FrameStates &, BitMap *, std::vector<int> * = nullptr);

  void FoundInnerClass() { found_inner_class = true; }

  const auto &prepare() const { return prepare_; }
  auto &prepare() { return prepare_; }
  bool PrepareParameter(ValueNode *node);

  // return true if has fail guard matched
  bool NeedSymbolic(ValueNode *node);

  void set_func_graph_builder(const std::shared_ptr<FuncGraphBuilder> &ptr) { func_graph_builder_ = ptr; }
  const auto &func_graph_builder() const { return func_graph_builder_; }

 private:
  void AddNodeInfo(ValueNode *node, AObject *obj_info, const std::string &name);
  void DumpBreakInfo(std::ostream *out) const;
  void PrintFrame(std::ostream *out, const std::string &prefix) const;

  std::unique_ptr<CFG> cfg_;

  // frame status
  std::map<int, std::unique_ptr<FrameStates>> frame_states_;
  std::vector<ValueNode *> traced_nodes_;

  std::vector<ValueNode *> params_;

  std::map<const AObject *, ExpandParamInfo> expand_param_info_;

  // return value
  ValueNode *ret_val_;

  // used to fold generator function call
  ValueNode *generator_result_;

  // the traced code object
  py::object co_;

  // globals that may be used by frame when the tracer start
  py::object f_globals_;

  const char *module_name_;

  BreakInfo break_info_;

  Allocator alloc_;

  const GraphJitConfig &conf_;

  Graph *parent_{nullptr};
  std::shared_ptr<SideEffect> side_effect_;
  std::shared_ptr<SideEffectHandler> side_effect_handler_;
  bool found_inner_class = false;

  struct PrepareInfo {
    std::vector<ValueNode *> inputs_;
    std::vector<ValueNode *> operations_;
  } prepare_;
  std::unique_ptr<GuardBuilder> guard_builder_;

  std::shared_ptr<FuncGraphBuilder> func_graph_builder_;
};

// If using @jit(fullgraph=true), will throw this exception when graph break occurs.
class GraphBreakException : public std::runtime_error {
 public:
  explicit GraphBreakException(const std::string &msg) : std::runtime_error(msg) {}
  explicit GraphBreakException(const char *msg) : std::runtime_error(msg) {}
  // Similar to py::builtin_exception::set_error(), call PyErr_SetString() to throw an exception to the Python side.
  void set_error() const;
};

// Return the file path of python code.
std::string GetFileName(const Graph *graph);

// Return a string in format: 'func_name' at "file_path:line_number"
std::string GetNameAndLocation(const Graph *graph);

CallNode *FindBreakAtCall(const Graph *graph);

// Check if the graph is break at calling subgraph.
inline bool IsBreakAtCall(Graph *graph) { return FindBreakAtCall(graph) != nullptr; }
}  // namespace pijit
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_JIT_PI_GRAPH_CAPTURE_GRAPH_H
