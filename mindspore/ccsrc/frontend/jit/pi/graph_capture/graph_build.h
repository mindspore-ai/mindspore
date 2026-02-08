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
#ifndef MINDSPORE_CCSRC_FRONTEND_JIT_PI_GRAPH_CAPTURE_GRAPH_BUILD_H
#define MINDSPORE_CCSRC_FRONTEND_JIT_PI_GRAPH_CAPTURE_GRAPH_BUILD_H

#include <vector>
#include <unordered_map>
#include <utility>
#include <memory>
#include <optional>
#include <string>
#include "frontend/jit/pi/python_adapter/py_frame.h"
#include "frontend/jit/pi/graph_capture/graph.h"
#include "frontend/jit/pi/graph_build/func_graph_builder.h"
#include "frontend/jit/pi/utils/utils.h"

namespace mindspore {
namespace pijit {
class GraphBuilder;
using GraphBuilderPtr = std::shared_ptr<GraphBuilder>;

struct TryBlock {
  int type;         /*what kind of block this is (SETUP_SETUP, SETUP_FINALLY, SETUP_EXCEPT)*/
  int bci;          /*where to jump to find handler*/
  std::string name; /*entry of block name,which equal with instr name*/
  int checkpoint;   /*the handler to be rolled back*/
  // int level;   /* value stack level to pop toe*/
  bool IsFinallyBlock; /*record current block is in exception block or finally block*/
};

const std::vector<std::string> kAstFunctionList = {
  "mindspore.ops.function.array_func", "mindspore.ops.function.nn_func", "mindspore.ops.function.math_func"};

class GraphBuilder {
 public:
  static const char *ID___self__;
  static const char *ID___globals__;
  static const char *ID___call__;
  static const char *ID_construct;
  static const char *ID_forward;

  explicit GraphBuilder(const PyFrameWrapper &f);
  GraphBuilder(GraphBuilder *r, GraphBuilder *p, PyCodeObject *co, PyObject *globals);
  explicit GraphBuilder(GraphBuilder *r)
      : root_(r), parent_(nullptr), graph_(nullptr), current_block_(nullptr), no_grad_(r->no_grad_) {}
  ~GraphBuilder() {
    for (auto i : graph_pool_) {
      delete i;
    }
    graph_pool_.clear();
  }

  StopTraceReason TraceRun();

  Graph *GetGraph() const { return graph_; }
  void DumpDFG();
  std::string FormatStackStr() const;

  // NOTE: nn.Cell will return 'construct'
  static py::object FindPyFunc(AObject *vobj);
  static py::object GetFuncInfo(ValueNode *func_node);

  // Exception
  ValueNode *&peekExc(int p) { return excFrame_.Peek(p); }
  void pushExc(ValueNode *v) { excFrame_.Push(v); }
  ValueNode *popExc() { return excFrame_.Pop(); }
  int excStackSize() { return excFrame_.Size(); }

  // TryBlockStack operation
  TryBlock &PeekStack(int p);
  void PushStack(TryBlock tb) { tryBlockStacks_.push_back(tb); }
  int StackSize() { return tryBlockStacks_.size(); }
  std::vector<TryBlock> &GetTryBlockStacks() { return tryBlockStacks_; }
  TryBlock PopStack();

  /**
   * Handle call node. Infer call result. Inline call node bytecode
   * \return Ttop trace reason of sub-graph
   */
  StopTraceReason HandleCall();

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
  void ResolveClosure(const py::object &func_info, CallNode *call_node, FrameStates *frame);

  std::pair<PyObject *, ValueNode *> SearchSelfPyObject(PyCodeObject *co);
  bool HandleSuper(const Instr &instr, AObject *super);
  AObject *BuildSuperObject(PyCodeObject *co);

  /**
   * Collect parameters of call stack and set it to frame
   * \param func_info The function of call target
   * \param call_node This calling information
   * \param frame FrameStates to place parameters nodes
   * \return false if parameters is illegal
   */
  bool HandleCallParameters(const py::object &func_info, CallNode *call_node, FrameStates *frame);

  /**
   * Unpack CALL_FUNCTION_EX parameters to stack
   * \param[in] params the call stack
   * \param[in] extra_local extra local index
   * \param[out] extra_oper unpack operations by bytecode
   * \param[out] has_kw this call has key-word arguments
   * \return false if can't generate unpack operations
   */
  bool UnpackCallExParams(std::vector<ValueNode *> *params, int extra_local, bool *has_kw, CallNode *call_node);

  bool UnpackCallExDict(std::vector<ValueNode *> *params, CallNode *call_node);

  /**
   * Pack key-word parameters, generate kwvargs value node, check kw-defaults arguments
   * \param[in] func The function of call target
   * \param[in] params This calling stack
   * \param[in] frame FrameStates to place parameters nodes
   * \param[out] extra_oper the move operations to move parameters to locals
   * \return false if parameters is illegal
   */
  bool HandleKWParams(const py::object &func, std::vector<ValueNode *> *params, FrameStates *frame);

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
                    std::vector<ValueNode *> *kwvargs);

  bool CheckAndSetDefaultParams(const py::object &func, FrameStates *frame, int pargc);

  /**
   * Use the call stack without key-word arguments to fill the frame locals
   */
  bool HandlePositionParams(const py::object &func, std::vector<ValueNode *> *params, FrameStates *frame);

  // build subgraph, return stop trace reason
  StopTraceReason BuildSubGraph(CallNode *call_node, const py::object &func, const GraphBuilderPtr &subgraph);

  ValueNode *BuildCallClassNode(CallNode *call_node);

  ValueNode *HandleCallClass(CallNode *call_node);

  bool HandleCallTensorClass(CallNode *call_node);

  // return false if has unsupported bytecode
  bool DoByteCode(const Instr &instr);

  // unpack elements
  bool UnpackElements(ValueNode *);

  // unpack elements
  bool UnpackSequenceElements(ValueNode *);

  // unpack dict items to build map inputs
  bool UnpackDict(ValueNode *map);

  // unpack BUILD_CONST_KEY_MAP to stack in (key1, value1, key2, value2, ...) format
  bool UnpackConstKeyMapToStack(ValueNode *map);
  // unpack BUILD_CONST_KEY_MAP in (key1, value1, key2, value2, ...) format
  std::vector<ValueNode *> UnpackConstKeyMap(ValueNode *map);

  // unpack object elements as LOAD_CONST
  std::vector<ValueNode *> UnpackConstObject(const py::object &);

  // return true if not inline
  bool WhiteListFuncCheckAndInfer(CallNode *, const py::object &f);

  bool DoSetItem(ValueNode *map, ValueNode *key, ValueNode *val);

  // transform dict set item to make a new dict
  ValueNode *TransformDictSetItem(ValueNode *map, ValueNode *key, ValueNode *val, bool ignore_key_error);

  // transform list set item to make a new list
  ValueNode *TransformListSetItem(ValueNode *list, ValueNode *key, ValueNode *val);

  // Helper method to extract dict keys from map object or AbstractDict
  std::vector<py::object> GetDictKeys(ValueNode *map, PyObject *map_object);

  // make a tensor copy operation
  ValueNode *MakeTensorCopy(ValueNode *tensor);

  ValueNode *ReplaceMergeOp(int opcode, const std::vector<ValueNode *> &inputs);

  // frame operation
  ValueNode *&seek(int p) { return frame_.Peek(p); }
  void push(ValueNode *v) { frame_.Push(v); }
  ValueNode *pop() { return frame_.Pop(); }
  void popn(int n) { frame_.Popn(n); }
  ValueNode *getLocal(int i) { return frame_.Local(i); }
  void setLocal(int i, ValueNode *n) { frame_.SetLocal(i, n); }

  Instr NewCallFuncInstr(int oparg);
  // pointers
  std::vector<Graph *> graph_pool_;
  ValueNode *NewValueNode(AObject *o, int op, int arg, const std::vector<ValueNode *> &p = {},
                          const std::string &name = "");
  ValueNode *NewValueNode(AObject *o, const Instr &, const std::vector<ValueNode *> &p = {});
  Graph *NewGraph(PyCodeObject *co, PyObject *f_globals);

  bool ReplaceAll(ValueNode *old_node, ValueNode *new_node, bool *referenced = nullptr);

  bool TraceRunForIterEnumerate(int jump_bci);
  bool TraceRunForIterZip(int jump_bci);
  bool TraceRunForIterDict(int jump_bci);
  bool TraceRunForIterDictItems(int jump_bci);
  bool TraceRunForIterSequence(int jump_bci, int seq_size);

  // bytecode operations
  bool TraceRunControl(const Instr &instr);
  bool ConditionJump(const Instr &instr, int *cond, int *jump_to, ValueNode **cond_node);
  bool ConditionJumpPy311(const Instr &instr, int *cond, int *jump_to, ValueNode **cond_node);
  bool TraceRunForIter(const Instr &instr);
  bool DoUnpack(const Instr &instr);
  bool DoBuildWithUnpack(const Instr &instr);
  bool DoBuildMapWithUnpack(const Instr &instr);
  ValueNode *GetBoundSelf(CallNode *call_node);
  bool DoCall(const Instr &instr);
  bool DoNop(const Instr &instr);
  bool DoReturn(const Instr &instr);
  bool DoLocalAccess(const Instr &instr);
  bool DoCellAccess(const Instr &instr);
  void HandleLoadGlobalPythonCode(const Instr &instr);
  void DoLoadGlobal(const Instr &instr);
  bool DoGlobalAccess(const Instr &instr);
  bool DoAttrAccess(const Instr &instr);
  ValueNode *HandleGetattr(ValueNode *target_node, const Instr &instr);
  bool DoGetItem(const Instr &instr);

  bool DoItemAccess(const Instr &instr);
  bool DoStackOp(const Instr &instr);
  bool DoLoadConst(const Instr &instr);
  bool DoListToTuple(const Instr &instr);
  bool DoGetIter(const Instr &instr);
  bool DoMakeFunction(const Instr &instr);
  bool DoUnary(const Instr &instr);
  bool DoBinary(const Instr &instr);
  bool DoIsOp(const Instr &instr);
  bool DoContainsOp(const Instr &instr);
  bool DoListOrTupleAdd(const Instr &instr);
  bool DoBinaryAdd(const Instr &instr);
  bool DoInplaceAdd(const Instr &instr);
  bool DoCompare(const Instr &instr);
  bool DoBuildOp(const Instr &instr);
  bool DoMergeOp(const Instr &instr);
  bool DoFormatValue(const Instr &instr);
  bool DoImport(const Instr &instr);
  bool DoSend(const Instr &instr);
  bool DoYieldValue(const Instr &instr);
  bool DoYieldFrom(const Instr &instr);
  bool DoGetYieldFromIter(const Instr &instr);
  bool DoWith(const Instr &instr);
  bool DoRaise(const Instr &instr);
  bool DoSetupFinally(const Instr &instr);
  bool DoWithCleanUpStart(const Instr &instr);
  bool DoWithCleanUpFinish(const Instr &instr);
  bool DoBeginFinally(const Instr &instr);
  bool DoPopFinally(const Instr &instr);
  bool DoEndFinally(const Instr &instr);
  bool DoCallFinally(const Instr &instr);
  bool DoSetupExc(const Instr &instr);
  bool DoPopExc(const Instr &instr);
  bool DoExcMatch(const Instr &instr);
  bool DoLoadAssertError(const Instr &instr);
  bool DoPopStack(const Instr &instr);
  bool DoRaiseVarags(const Instr &instr);
  bool DoLoadName(const Instr &instr);
  bool DoPushNull(const Instr &instr);
  bool DoBinaryOp(const Instr &instr);
  bool DoCheckExcMatch(const Instr &instr);
  bool DoPushExcInfo(const Instr &instr);

  const auto &root() const { return root_; }
  const auto &frame() const { return frame_; }

  FuncGraphBuilderPtr FGBuilder() const { return graph_->func_graph_builder(); }
  void FGAddNode(CallNode *call_node, const py::object &callable_info, const AbstractWrapperPtrList &args,
                 StopTraceReason *stop_reason);
  void FGAddNode(CallNode *call_node, const ValuePtr &callable_value, const AbstractWrapperPtrList &args,
                 StopTraceReason *stop_reason);
  AbstractWrapperPtrList HandleInputArgs(const std::vector<ValueNode *> args);

  std::vector<ValueNode *> side_effect_outputs() { return side_effect_outputs_; }
  void AddVarInput(ValueNode *node, bool is_key_word);
  void AddInput(ValueNode *node);
  void ExpandContainerParameters(ValueNode *node);

  /**
   * now, call this function only if:
   * 1. scalar of func_graph parameter,
   * 2. global variable,
   * 3. attribute of any,
   * 4. class instantiation(such as float(any), int(any)...),
   * return true if above value is not constant, pass the interpret result as mutable parameter to graph
   */
  bool Symbolic(ValueNode *node);

  BindArgumentsHelper<ValueNode *> PackInputsForFunc(const py::object &obj, int op_code,
                                                     const std::vector<ValueNode *> &inputs, PyObject *kw_names,
                                                     ValueNode *self_node = nullptr, bool eliminate_sens = false);
  GraphBuilderPtr get_prev_call_builder() const { return prev_call_builder_; }
  static const std::unordered_map<ValueNode *, ValueNode *> &GetExpandInputMap() { return expand_input_map_; }

 private:
  GraphBuilderPtr prev_call_builder_ = nullptr;
  GraphBuilder *root_;
  GraphBuilder *parent_;
  Graph *graph_;
  FrameStates frame_;
  Block *current_block_;
  int cur_bci_ = 0;
  bool no_grad_;
  // Side effect outputs of this graph (including the side effect outputs of all its sub-graphs).
  std::vector<ValueNode *> side_effect_outputs_;
  std::vector<TryBlock> tryBlockStacks_{};
  FrameStates excFrame_;
  int last_traced_line_ = -1;

  static const std::unordered_map<int, bool (GraphBuilder::*)(const Instr &)> bytecode_meth_map_;
  static std::unordered_map<ValueNode *, ValueNode *> expand_input_map_;

  bool IsTopGraph() const { return this == root_; }
  LocationPtr GetLocation(const Instr &instr) const;

  ValueNode *MakePrimCastNode(ValueNode *node, const py::handle &dst_dtype);
  bool DoMixedPrecisionLocalAccess(const Instr &instr, ValueNode *node);
  ValueNode *DoMixedPrecisionAttrAccess(const Instr &instr, ValueNode *node, ValueNode *attr);
  bool ResolveNoGrad(CallNode *call_node);
  bool ResolveEnableGrad(CallNode *call_node, AObject *callable, py::object callable_info);

  void FGAddTopInputsWithExpander();
  void FGAddTopInputs();
  bool FGAddInputs(const std::vector<ValueNode *> &args);

  std::vector<ValueNode *> GetNewArgs(CallNode *call_node, AObject *vobj = nullptr,
                                      const GraphBuilderPtr &subgraph = nullptr);

  py::object HandleConstantFoldFunc(const std::vector<py::object> &args, CallNode *call_node,
                                    StopTraceReason *stop_reason);
  py::object HandleMSCallable(CallNode *call_node, const py::object &callable_info, const py::object &original_callable,
                              StopTraceReason *stop_reason);
  // Handle python builtin type()
  void HandleCallType(CallNode *call_node, StopTraceReason *stop_reason) const;

  // Collect side effect nodes that need to be returned from current graph.
  void CollectSideEffectOutputs();
  FuncGraphPtr BuildSubFuncGraph(const GraphBuilderPtr &subgraph_builder, CallNode *call_node);
  bool FGAddOutput();
  bool FGAddSideEffectOutput();
  bool HandleSubGraphOutput(const AbstractWrapperPtr &output, const GraphBuilderPtr &subgraph_builder,
                            CallNode *call_node);

  AbstractWrapperPtr HandleGetShapeOfDynamicLengthTensor(const AbstractWrapperPtr &abstract_wrapper);
  std::pair<bool, std::vector<py::object>> GetConstantInputsObject(CallNode *call_node);
  py::object GetPyObject(ValueNode *node);

  ValueNode *HandleMakeNamedtuple(CallNode *call_node);
  AbstractWrapperPtr MakeNamedtupleInGraph(const CallNode *call_node, const AbstractNamedTuple *namedtuple_aobj);
  bool CollectNamedtupleElements(const CallNode *call_node, const AbstractNamedTuple *namedtuple_aobj,
                                 std::vector<AbstractWrapperPtr> *elems);
  ValueNode *HandleNamedtupleGetElem(const Instr &instr, ValueNode *node);

  ValueNode *BuildMultiOpValueNode(const Instr &instr, const std::vector<ValueNode *> &p, bool is_compare = false);
  AbstractWrapperPtr HandleMultiOp(const Instr &instr, const std::vector<ValueNode *> &p, bool is_compare);
  AbstractWrapperPtr HandleBuildOp(const Instr &instr, const std::vector<ValueNode *> &p);
  AbstractWrapperPtr HandleBuildStringOp(const PrimitivePtr &primitive, const AbstractWrapperPtrList &inputs_wrapper);

  bool ConvertClassType(const py::object &callable_info, CallNode *call_node, StopTraceReason *stop_reason);
  std::pair<bool, py::object> ConvertCallableObject(const py::object &callable_info) const;
  py::object ResolveCallableWithByteCode(CallNode *call_node, StopTraceReason *stop_reason);

  py::object FGAddNodeAst(CallNode *call_node, const py::object &callable_info,
                          const py::object &original_callable_info, StopTraceReason *stop_reason);
  py::object FGAddNodeTensorOverload(CallNode *call_node, const py::object &callable_info,
                                     StopTraceReason *stop_reason);
};

namespace fg_build_utils {
AbstractWrapperPtr FgTupleGetItem(const FuncGraphBuilderPtr &fg_builder, const AbstractWrapperPtr &tuple, int index);
std::optional<std::vector<AbstractWrapperPtr>> FgTupleUnpack(const FuncGraphBuilderPtr &fg_builder,
                                                             const AbstractWrapperPtr &tuple);
// Add AnfNode in parent FuncGraph to call the subgraph's FuncGraph.
AbstractWrapperPtr FgCallSubGraph(CallNode *call_node);
}  // namespace fg_build_utils
}  // namespace pijit
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_JIT_PI_GRAPH_CAPTURE_GRAPH_BUILD_H
