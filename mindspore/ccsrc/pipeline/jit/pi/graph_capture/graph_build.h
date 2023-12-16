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
#ifndef MINDSPORE_CCSRC_PIPELINE_GRAPH_JIT_GRAPH_CAPTURE_GRAPH_BUILD_H
#define MINDSPORE_CCSRC_PIPELINE_GRAPH_JIT_GRAPH_CAPTURE_GRAPH_BUILD_H

#include <vector>
#include <unordered_map>
#include <utility>
#include "pipeline/jit/pi/graph_capture/graph.h"

namespace mindspore {
namespace jit {
namespace graph {
class GraphBuilder {
 public:
  static const char *ID___self__;
  static const char *ID___globals__;
  static const char *ID___call__;
  static const char *ID_construct;

  explicit GraphBuilder(const PyFrameObject *f);
  GraphBuilder(GraphBuilder *r, GraphBuilder *p, PyCodeObject *co, PyObject *globals)
      : root_(r), parent_(p), graph_(NewGraph(co, globals)), frame_(), current_block_(nullptr) {}
  ~GraphBuilder() {
    for (auto i : graph_pool_) {
      delete i;
    }
    graph_pool_.clear();
  }

  /**
   * return false if should break graph
   */
  StopTraceReason BuildGraph(int depth = 0);
  void Reset();

  void CollectInlineInfo(CallNode *node, int depth);
  Graph *GetGraph() const { return graph_; }
  void DumpDFG();

  // NOTE: nn.Cell will return 'construct'
  static py::object FindPyFunc(AObject *vobj);
  static py::object GetFuncInfo(ValueNode *func_node);

  static bool IsByteCodeImplemented(int bytecode);

 private:
  GraphBuilder *root_;
  GraphBuilder *parent_;
  Graph *graph_;
  FrameStates frame_;
  Block *current_block_;
  int cur_bci_;

  // return true if switch to next block
  void HandleBlockSwitch();

  // loop analyze
  StopTraceReason HandleLoop();

  /**
   * Handle call node. Infer call result. Inline call node bytecode
   * \param depth Current inline depth
   * \return Ttop trace reason of sub-graph
   */
  StopTraceReason HandleCall(int depth);

  /**
   * Resolve callable object, if it's unknown object, return infer failed reason.
   * Check inline white list, infer result and not inline bytecode
   * If call a class, try to handle class
   * \param [in] call_node
   * \param [out] stop_reason
   * \return The function object of call target
   */
  py::object ResolveCallable(CallNode *call_node, StopTraceReason *stop_reason);

  /**
   * Resolve closure of function, generate cell free nodes to trace closure
   * \param func_info The function of call target
   * \param callable_node The value node of callable object
   * \param frame FrameStates to place closure node
   */
  void ResolveClosure(const py::object &func_info, ValueNode *callable_node, FrameStates *frame);

  std::pair<PyObject *, ValueNode *> SearchSelfPyObject(PyCodeObject *co);
  AObject *BuildSuperObject(PyCodeObject *co);
  /**
   * Collect parameters of call stack and set it to frame
   * \param func_info The function of call target
   * \param call_node This calling information
   * \param frame FrameStates to place parameters nodes
   * \return false if parameters is illegal
   */
  bool HandleCallParameters(const py::object &func_info, CallNode *call_node, FrameStates *frame);

  AbstractNodeList UnpackDynamicLengthTupleByBytecode(std::vector<ValueNode *> *params, AbstractNodeList tuple_unpack,
                                                      ValueNode *args_node, CallNode *call_node);

  bool UnpackExtraOper(AbstractNodeList tuple_unpack, int extra_local, AbstractNodeList *etra_oper,
                       AbstractNodeList dict_unpack, bool has_dict);
  /**
   * Unpack CALL_FUNCTION_EX parameters to stack
   * \param[in] params the call stack
   * \param[in] extra_local extra local index
   * \param[out] extra_oper unpack operations by bytecode
   * \param[out] has_kw this call has key-word arguments
   * \return false if can't generate unpack operations
   */
  bool UnpackCallExParams(std::vector<ValueNode *> *params, int extra_local, AbstractNodeList *extra_oper, bool *has_kw,
                          CallNode *call_node);

  bool UnpackCallExDict(std::vector<ValueNode *> *params, AbstractNodeList *extra_oper, CallNode *call_node);
  bool UnpackDynamicLengthDictByBytecode(std::vector<ValueNode *> *params, AbstractNodeList *extra_oper,
                                         CallNode *call_node, ValueNode *dict_node);
  // generate the general unpack operations of dict, return operations
  std::vector<AbstractNode *> GenerateDictUnpack(ValueNode *kwargs_node);

  /**
   * Pack key-word parameters, generate kwvargs value node, check kw-defaults arguments
   * \param[in] func The function of call target
   * \param[in] params This calling stack
   * \param[in] frame FrameStates to place parameters nodes
   * \param[out] extra_oper the move operations to move parameters to locals
   * \return false if parameters is illegal
   */
  bool HandleKWParams(const py::object &func, std::vector<ValueNode *> *params, FrameStates *frame,
                      AbstractNodeList *extra_oper);

  /**
   * Pack key-word parameters to dict, unpack the position arguments by key from the dict.
   * Set parameters to frame
   * \param[in] func The function of call target
   * \param[in] params This calling stack
   * \param[in] frame FrameStates to place parameters nodes
   * \param[out] dict_gen the move operations to move parameters to locals
   * \param[out] dict_op the opcode of dict generation
   * \return false if parameters is illegal
   */
  bool PackKwParams(const py::object &func, std::vector<ValueNode *> *params, FrameStates *frame,
                    AbstractNodeList *dict_gen, std::vector<ValueNode *> *kwvargs);

  bool CheckAndSetDefaultParams(const py::object &func, AbstractNodeList *extra_oper, FrameStates *frame, int pargc);

  /**
   * Use the call stack without key-word arguments to fill the frame locals
   */
  bool HandlePositionParams(const py::object &func, std::vector<ValueNode *> *params, AbstractNodeList *extra_oper,
                            FrameStates *frame);

  // build subgraph, return stop trace reason
  StopTraceReason BuildSubGraph(CallNode *call_node, int depth, const py::object &func, GraphBuilder *subgraph);

  bool ReplaceCall(CallNode *call_node, const py::object &func);

  // build abstract instance of python class
  bool HandleCallClass(CallNode *call_node);

  // return false if has unsupported bytecode
  bool DoByteCode(const Instr &instr);

  void ResetBci(int i) { cur_bci_ = i; }
  bool PruneBranch(const Instr &);

  // general value node for UNPACK_SEQUENCE, UNPACK_EX
  void GenIndexItemGeneral(ValueNode *iterable, int i, int j);

  // create and update condition node
  void CreateAndSetConditionNode(ValueNode *cond);

  // return true if not inline
  bool WhiteListFuncCheckAndInfer(CallNode *, const py::object &f);

  // return true if function is recursion call
  bool RecursionCheck(PyCodeObject *);

  // create merge node
  ValueNode *merge(ValueNode *a, ValueNode *b, Block *tar_block);
  void MergeFrameState(Block *tar_block);

  // frame operation
  ValueNode *&seek(int p) { return frame_.Peek(p); }
  void push(ValueNode *v) { frame_.Push(v); }
  ValueNode *pop() { return frame_.Pop(); }
  void popn(int n) { frame_.Popn(n); }
  ValueNode *getLocal(int i) { return frame_.Local(i); }
  void setLocal(int i, ValueNode *n) { frame_.SetLocal(i, n); }

  // pointers
  std::vector<Graph *> graph_pool_;
  ValueNode *NewValueNode(AObject *o, int op, int arg, const std::vector<ValueNode *> &p);
  ValueNode *NewValueNode(AObject *o, const Instr &, const std::vector<ValueNode *> &p);
  InstrNode *NewInstrNode(int op, int arg);
  Graph *NewGraph(PyCodeObject *co, PyObject *f_globals);

  // bytecode operations
  bool DoUnpack(const Instr &instr);
  bool DoForIter(const Instr &instr);
  bool DoJumpIf(const Instr &instr);
  bool DoJumpDirect(const Instr &instr);
  bool DoCall(const Instr &instr);
  bool DoNop(const Instr &instr);
  bool DoReturn(const Instr &instr);
  bool DoLocalAccess(const Instr &instr);
  bool DoCellAccess(const Instr &instr);
  bool DoGlobalAccess(const Instr &instr);
  bool DoAttrAccess(const Instr &instr);
  bool DoItemAccess(const Instr &instr);
  bool DoStackOp(const Instr &instr);
  bool DoLoadConst(const Instr &instr);
  bool DoListToTuple(const Instr &instr);
  bool DoGetIter(const Instr &instr);
  bool DoMakeFunction(const Instr &instr);
  bool DoUnary(const Instr &instr);
  bool DoBinary(const Instr &instr);
  bool DoCompare(const Instr &instr);
  bool DoBuildOp(const Instr &instr);
  bool DoMergeOp(const Instr &instr);
  bool DoFormatValue(const Instr &instr);
  bool DoImport(const Instr &instr);
  bool NotImplementBytecode(const Instr &instr);
  static const std::unordered_map<int, bool (GraphBuilder::*)(const Instr &)> bytecode_meth_map_;
};
}  // namespace graph
}  // namespace jit
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PIPELINE_GRAPH_JIT_GRAPH_CAPTURE_GRAPH_BUILD_H
