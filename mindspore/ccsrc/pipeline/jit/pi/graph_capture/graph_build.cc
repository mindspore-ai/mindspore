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
#include "pipeline/jit/pi/graph_capture/graph_build.h"
#include <algorithm>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include <utility>
#include <unordered_map>
#include <map>
#include "pipeline/jit/pi/common.h"
#include "pipeline/jit/pi/graph_capture/loop_unrolling.h"
#include "pipeline/jit/pi/graph_capture/special_func_infer.h"
#include "pipeline/jit/pi/graph_guard/infer.h"
#include "pipeline/jit/pi/external.h"
#include "pipeline/jit/pi/graph_build/func_graph_builder.h"
#include "pipeline/jit/pi/graph_capture/abstract_object.h"
#include "include/common/debug/anf_ir_dump.h"
#include "pipeline/jit/pi/graph_compiler/utils.h"
#include "ops/sequence_ops.h"
#include "ops/framework_ops.h"
#include "ops/structure_ops.h"

#ifndef PY_MINOR_VERSION
#define PY_MINOR_VERSION 3.7
#error "undefined PY_MINOR_VERSION"
#endif  // PY_MINOR_VERSION

#ifndef PY_MAJOR_VERSION
#define PY_MAJOR_VERSION 3.9
#error "undefined PY_MAJOR_VERSION"
#endif  // PY_MAJOR_VERSION

namespace mindspore {
namespace pijit {
extern TracePtr GetTrace(ValueNode *node, bool strict, bool print, int depth, int max_depth);

void LogGuardFailed(ValueNode *node, const GraphJitConfig &conf, const std::string &msg);
static bool GuardLoopSequence(Graph *graph, ValueNode *seq_node, Py_ssize_t seq_size = -1);

const char *GraphBuilder::ID___self__ = "__self__";
const char *GraphBuilder::ID___globals__ = "__globals__";
const char *GraphBuilder::ID___call__ = "__call__";
const char *GraphBuilder::ID_construct = "construct";

static const int infer_primitive_create = 1;
static const int infer_primitive_object = 2;
static const int infer_primitive_func = 4;
static int infer_func_count = 0;
static constexpr const char *kPIJitCopyFuncKey = ".<pijit.copy>.";

const std::unordered_map<int, bool (GraphBuilder::*)(const Instr &)> GraphBuilder::bytecode_meth_map_ = {
  {POP_TOP, &GraphBuilder::DoStackOp},
  {ROT_TWO, &GraphBuilder::DoStackOp},
  {ROT_THREE, &GraphBuilder::DoStackOp},
  {ROT_FOUR, &GraphBuilder::DoStackOp},
  {DUP_TOP, &GraphBuilder::DoStackOp},
  {DUP_TOP_TWO, &GraphBuilder::DoStackOp},
  {NOP, &GraphBuilder::DoNop},
  {EXTENDED_ARG, &GraphBuilder::DoNop},
  {RETURN_VALUE, &GraphBuilder::DoReturn},
  {UNARY_POSITIVE, &GraphBuilder::DoUnary},
  {UNARY_NEGATIVE, &GraphBuilder::DoUnary},
  {UNARY_NOT, &GraphBuilder::DoUnary},
  {UNARY_INVERT, &GraphBuilder::DoUnary},
  {BINARY_MATRIX_MULTIPLY, &GraphBuilder::DoBinary},
  {BINARY_MULTIPLY, &GraphBuilder::DoBinaryMul},
  {BINARY_MODULO, &GraphBuilder::DoBinary},
  {BINARY_POWER, &GraphBuilder::DoBinary},
  {BINARY_ADD, &GraphBuilder::DoBinaryAdd},
  {BINARY_SUBTRACT, &GraphBuilder::DoBinary},
  {BINARY_FLOOR_DIVIDE, &GraphBuilder::DoBinary},
  {BINARY_TRUE_DIVIDE, &GraphBuilder::DoBinary},
  {BINARY_LSHIFT, &GraphBuilder::DoBinary},
  {BINARY_RSHIFT, &GraphBuilder::DoBinary},
  {BINARY_AND, &GraphBuilder::DoBinary},
  {BINARY_XOR, &GraphBuilder::DoBinary},
  {BINARY_OR, &GraphBuilder::DoBinary},
  {INPLACE_MATRIX_MULTIPLY, &GraphBuilder::DoBinary},
  {INPLACE_MULTIPLY, &GraphBuilder::DoBinary},
  {INPLACE_MODULO, &GraphBuilder::DoBinary},
  {INPLACE_POWER, &GraphBuilder::DoBinary},
  {INPLACE_ADD, &GraphBuilder::DoInplaceAdd},
  {INPLACE_SUBTRACT, &GraphBuilder::DoBinary},
  {INPLACE_FLOOR_DIVIDE, &GraphBuilder::DoBinary},
  {INPLACE_TRUE_DIVIDE, &GraphBuilder::DoBinary},
  {INPLACE_LSHIFT, &GraphBuilder::DoBinary},
  {INPLACE_RSHIFT, &GraphBuilder::DoBinary},
  {INPLACE_AND, &GraphBuilder::DoBinary},
  {INPLACE_XOR, &GraphBuilder::DoBinary},
  {INPLACE_OR, &GraphBuilder::DoBinary},
  {IS_OP, &GraphBuilder::DoIsOp},
  {CONTAINS_OP, &GraphBuilder::DoIsOp},
  {BUILD_TUPLE, &GraphBuilder::DoBuildOp},
  {BUILD_LIST, &GraphBuilder::DoBuildOp},
  {BUILD_SET, &GraphBuilder::DoBuildOp},
  {BUILD_MAP, &GraphBuilder::DoBuildOp},
  {BUILD_SLICE, &GraphBuilder::DoBuildOp},
  {BUILD_CONST_KEY_MAP, &GraphBuilder::DoBuildOp},
  {BUILD_STRING, &GraphBuilder::DoBuildOp},
  {LIST_APPEND, &GraphBuilder::DoMergeOp},
  {LIST_EXTEND, &GraphBuilder::DoMergeOp},
  {DICT_MERGE, &GraphBuilder::DoMergeOp},
  {DICT_UPDATE, &GraphBuilder::DoMergeOp},
  {SET_UPDATE, &GraphBuilder::DoMergeOp},
  {SET_ADD, &GraphBuilder::DoMergeOp},
  {MAP_ADD, &GraphBuilder::DoMergeOp},
  {COMPARE_OP, &GraphBuilder::DoCompare},
  {MAKE_FUNCTION, &GraphBuilder::DoMakeFunction},
  {FORMAT_VALUE, &GraphBuilder::DoFormatValue},
  {LIST_TO_TUPLE, &GraphBuilder::DoListToTuple},
  {LOAD_CONST, &GraphBuilder::DoLoadConst},
  {IMPORT_STAR, &GraphBuilder::DoImport},
  {IMPORT_NAME, &GraphBuilder::DoImport},
  {IMPORT_FROM, &GraphBuilder::DoImport},
  {CALL_FUNCTION, &GraphBuilder::DoCall},
  {CALL_FUNCTION_KW, &GraphBuilder::DoCall},
  {CALL_FUNCTION_EX, &GraphBuilder::DoCall},
  {CALL_METHOD, &GraphBuilder::DoCall},
  {UNPACK_SEQUENCE, &GraphBuilder::DoUnpack},
  {UNPACK_EX, &GraphBuilder::DoUnpack},
  {BINARY_SUBSCR, &GraphBuilder::DoItemAccess},
  {STORE_SUBSCR, &GraphBuilder::DoItemAccess},
  {DELETE_SUBSCR, &GraphBuilder::DoItemAccess},
  {LOAD_GLOBAL, &GraphBuilder::DoGlobalAccess},
  {STORE_GLOBAL, &GraphBuilder::DoGlobalAccess},
  {DELETE_GLOBAL, &GraphBuilder::DoGlobalAccess},
  {LOAD_METHOD, &GraphBuilder::DoAttrAccess},
  {LOAD_ATTR, &GraphBuilder::DoAttrAccess},
  {STORE_ATTR, &GraphBuilder::DoAttrAccess},
  {DELETE_ATTR, &GraphBuilder::DoAttrAccess},
  {LOAD_CLOSURE, &GraphBuilder::DoCellAccess},
  {LOAD_DEREF, &GraphBuilder::DoCellAccess},
  {STORE_DEREF, &GraphBuilder::DoCellAccess},
  {DELETE_DEREF, &GraphBuilder::DoCellAccess},
  {LOAD_FAST, &GraphBuilder::DoLocalAccess},
  {STORE_FAST, &GraphBuilder::DoLocalAccess},
  {DELETE_FAST, &GraphBuilder::DoLocalAccess},
  {GET_ITER, &GraphBuilder::DoGetIter},
  {FOR_ITER, &GraphBuilder::TraceRunForIter},
  {POP_JUMP_IF_FALSE, &GraphBuilder::TraceRunControl},
  {POP_JUMP_IF_TRUE, &GraphBuilder::TraceRunControl},
  {JUMP_IF_FALSE_OR_POP, &GraphBuilder::TraceRunControl},
  {JUMP_IF_TRUE_OR_POP, &GraphBuilder::TraceRunControl},
  {JUMP_FORWARD, &GraphBuilder::TraceRunControl},
  {JUMP_ABSOLUTE, &GraphBuilder::TraceRunControl},
  {YIELD_VALUE, &GraphBuilder::DoYieldValue},
  {POP_BLOCK, &GraphBuilder::DoException},
  {SETUP_WITH, &GraphBuilder::DoException},
  {SETUP_FINALLY, &GraphBuilder::DoException},
  // not implement
  {LOAD_CLASSDEREF, &GraphBuilder::NotImplementBytecode},
  {LOAD_BUILD_CLASS, &GraphBuilder::NotImplementBytecode},
  {LOAD_ASSERTION_ERROR, &GraphBuilder::NotImplementBytecode},
  {GET_YIELD_FROM_ITER, &GraphBuilder::NotImplementBytecode},
  {GET_AWAITABLE, &GraphBuilder::NotImplementBytecode},
  {GET_AITER, &GraphBuilder::NotImplementBytecode},
  {GET_ANEXT, &GraphBuilder::NotImplementBytecode},
  {YIELD_FROM, &GraphBuilder::NotImplementBytecode},
  {PRINT_EXPR, &GraphBuilder::NotImplementBytecode},
  {POP_EXCEPT, &GraphBuilder::NotImplementBytecode},
  {WITH_EXCEPT_START, &GraphBuilder::NotImplementBytecode},
  {SETUP_ANNOTATIONS, &GraphBuilder::NotImplementBytecode},
  {SETUP_ASYNC_WITH, &GraphBuilder::NotImplementBytecode},
  {BEFORE_ASYNC_WITH, &GraphBuilder::NotImplementBytecode},
  {END_ASYNC_FOR, &GraphBuilder::NotImplementBytecode},
  {LOAD_NAME, &GraphBuilder::NotImplementBytecode},
  {STORE_NAME, &GraphBuilder::NotImplementBytecode},
  {DELETE_NAME, &GraphBuilder::NotImplementBytecode},
  {JUMP_IF_NOT_EXC_MATCH, &GraphBuilder::NotImplementBytecode},
  {RERAISE, &GraphBuilder::NotImplementBytecode},
  {RAISE_VARARGS, &GraphBuilder::NotImplementBytecode},

#if (PY_MAJOR_VERSION == 3) && (PY_MINOR_VERSION == 7)
  {BREAK_LOOP, &GraphBuilder::NotImplementBytecode},
  {WITH_CLEANUP_START, &GraphBuilder::DoException},
  {WITH_CLEANUP_FINISH, &GraphBuilder::DoException},
  {END_FINALLY, &GraphBuilder::DoException},
  {CONTINUE_LOOP, &GraphBuilder::NotImplementBytecode},
  {SETUP_LOOP, &GraphBuilder::NotImplementBytecode},
  {SETUP_EXCEPT, &GraphBuilder::DoException},
  {BUILD_LIST_UNPACK, &GraphBuilder::NotImplementBytecode},
  {BUILD_MAP_UNPACK, &GraphBuilder::NotImplementBytecode},
  {BUILD_MAP_UNPACK_WITH_CALL, &GraphBuilder::NotImplementBytecode},
  {BUILD_TUPLE_UNPACK, &GraphBuilder::NotImplementBytecode},
  {BUILD_SET_UNPACK, &GraphBuilder::NotImplementBytecode},
  {BUILD_TUPLE_UNPACK_WITH_CALL, &GraphBuilder::NotImplementBytecode},
#endif
};

bool GraphBuilder::IsByteCodeImplemented(int bytecode) {
  if (bytecode_meth_map_.find(bytecode) != bytecode_meth_map_.end()) {
    return bytecode_meth_map_.find(bytecode)->second != &GraphBuilder::NotImplementBytecode;
  }
  return false;
}

bool GraphBuilder::ReplaceAll(ValueNode *old_node, ValueNode *new_node) {
  // check reference relationship
  const auto &nodes = graph_->GetTracedNodes();
  bool find = std::any_of(nodes.begin(), nodes.end(), [&old_node](ValueNode *node) {
    if (Utils::IsGeneralNoSideEffectOp(node->GetOpcode())) {
      return false;
    }
    const auto &args = node->getInputs();
    return std::any_of(args.begin(), args.end(), [&old_node](ValueNode *i) { return i == old_node; });
  });
  if (find) {
    return false;
  }

  if (parent_ != nullptr && !parent_->ReplaceAll(old_node, new_node)) {
    return false;
  }
  const auto pred = [&old_node](ValueNode *i) { return i == old_node; };
  std::replace_if(frame_.GetLocals().begin(), frame_.GetLocals().end(), pred, new_node);
  std::replace_if(frame_.GetStacks().begin(), frame_.GetStacks().end(), pred, new_node);
  std::for_each(frame_.GetClosures().begin(), frame_.GetClosures().end(), [&old_node, &new_node](CellVarNode *i) {
    if (i->GetValue() == old_node) {
      i->SetValue(new_node);
    }
  });
  return true;
}

ValueNode *GraphBuilder::NewValueNode(AObject *o, int op, int arg, const std::vector<ValueNode *> &p,
                                      const std::string &name) {
  ValueNode *v;
  if (Utils::IsCallOp(op)) {
    v = graph_->NewCallNode(op, arg, p);
    v->SetVobj(o);
  } else {
    v = graph_->NewValueNode(o, op, arg, p, name);
  }
  return v;
}

ValueNode *GraphBuilder::NewValueNode(AObject *o, const Instr &i, const std::vector<ValueNode *> &p) {
  ValueNode *v = NewValueNode(o, i.op(), i.arg(), p, i.name());
  v->SetLineNo(i.line());
  v->set_bci(i.bci());
  if (o && o->GetType() == AObject::kTypeTensor) {
    current_block_->SetTrackResult(Block::kTrackHasTensor);
  }
  graph_->GetTracedNodes().push_back(v);
  return v;
}

Graph *GraphBuilder::NewGraph(PyCodeObject *co, PyObject *globals) {
  std::vector<Graph *> &graphs = (root_ != nullptr) ? root_->graph_pool_ : this->graph_pool_;
  if ((root_ == nullptr || root_ == this) && graph_ == nullptr) {
    JitCompileResults *jcr = getJitCompileResults(reinterpret_cast<PyObject *>(co), false);
    MS_EXCEPTION_IF_CHECK_FAIL(jcr && jcr->code != nullptr, "must be create guard code before trace start");
    graphs.push_back(new Graph(co, globals, *jcr->conf));
    graphs.back()->SetGuard(jcr->code);
  } else {
    graphs.push_back(new Graph(co, globals, root_->GetGraph()->Config()));
    graphs.back()->SetGuard(root_->GetGraph()->GetGuard());
  }
  return graphs.back();
}

static bool CheckValueValid(AObject *obj) {
  if (obj->GetType() == AObject::kTypeTensor) {
    AbstractTensor *tensor = static_cast<AbstractTensor *>(obj);
    return tensor->IsStubTensor() || CheckTensorDataInitialized(obj->GetPyObject());
  } else if (obj->GetType() == AObject::kTypeTraceNode) {
    auto py_obj = obj->GetPyObject();
    if (!py::isinstance<tensor::Tensor>(py_obj)) {
      return true;
    }
    return py_obj.cast<tensor::TensorPtr>()->data().const_data() != nullptr;
  } else {
    return true;
  }
}

int CondIsTrue(ValueNode *cond) {
  // if cond is tensor attrs, infer tensor attrs
  // if tensor is return node of cell, if tensor is return node of primitive
  // if tensor is result of math operation(+-*/...)
  AObject *cond_value = cond->GetVobj();
  int ret = -1;
  if (cond_value == nullptr || cond_value->GetPyObject().ptr() == nullptr) {
    return ret;
  }
  py::object value = cond_value->GetPyObject();
  if (CheckValueValid(cond_value)) {
    ret = PyObject_IsTrue(value.ptr());
    PyErr_Clear();
  }
  return ret;
}

static std::vector<AObject *> CollectObjects(const std::vector<ValueNode *> &nodes) {
  std::vector<AObject *> res;
  std::transform(nodes.begin(), nodes.end(), std::back_inserter(res),
                 [](const ValueNode *node) { return node->GetVobj(); });
  return res;
}

std::vector<ValueNode *> GraphBuilder::UnpackConstObject(const py::object &iterable) {
  std::vector<ValueNode *> outputs;
  std::transform(iterable.begin(), iterable.end(), std::back_inserter(outputs), [this](const auto &item) {
    return this->NewValueNode(AObject::Convert(item.ptr()), LOAD_CONST, -1, {});
  });
  return outputs;
}

bool GraphBuilder::UnpackSequenceElements(ValueNode *node) {
  py::object seq = node->GetVobj()->GetPyObject();
  if (seq.ptr() == nullptr || !PySequence_Check(seq.ptr()) || !GuardLoopSequence(this->graph_, node)) {
    return false;
  }

  Py_ssize_t size = PySequence_Size(seq.ptr());
  for (Py_ssize_t index = 0; index < size; ++index) {
    push(node);
    DoLoadConst({LOAD_CONST, -1, py::object(py::int_(index))});
    DoItemAccess({BINARY_SUBSCR, 0});
  }
  return true;
}

bool GraphBuilder::UnpackElements(ValueNode *node) {
  int opcode = node->GetOpcode();
  if (opcode == BUILD_LIST || opcode == BUILD_TUPLE) {
    std::for_each(node->getInputs().begin(), node->getInputs().end(), [this](ValueNode *i) { this->push(i); });
  } else if (node->IsConstantValue()) {
    std::vector<ValueNode *> nodes = UnpackConstObject(node->GetVobj()->GetPyObject());
    std::for_each(nodes.begin(), nodes.end(), [this](ValueNode *i) { this->push(i); });
  } else {
    return UnpackSequenceElements(node);
  }
  return true;
}

static void GenUnpackValue(const std::function<void(int, int)> &gen_item, int cnt, int cnt_after, Py_ssize_t size) {
  if (cnt_after != -1) {
    const int end_pos = size - cnt_after;
    for (int i = size; i > end_pos; --i) {
      gen_item(i - 1, -1);
    }
    gen_item(cnt, end_pos);
  }
  for (; cnt > 0; --cnt) {
    gen_item(cnt - 1, -1);
  }
}

Py_ssize_t GetUnpackSize(ValueNode *iterable, int cnt, int cnt_after) {
  int op = iterable->GetOpcode();
  Py_ssize_t total_args = cnt + cnt_after + 1;
  Py_ssize_t size;
  if (op == BUILD_LIST || op == BUILD_TUPLE) {
    size = iterable->getInputs().size();
  } else {
    AObject *seq = iterable->GetVobj();
    PyObject *o = (seq == nullptr) ? nullptr : seq->GetPyObject().ptr();
    size = (o == nullptr) ? -1 : PyObject_Size(o);
  }
  if (size == -1 || (cnt_after == -1 && cnt != size) || total_args > size + 1) {
    PyErr_Clear();
    return -1;
  }
  return size;
}

bool GraphBuilder::DoUnpack(const Instr &instr) {
  int opcode = instr.op();
  int oparg = instr.arg();
  int cnt = (opcode == UNPACK_EX) ? (oparg & 0xFF) : oparg;
  int cnt_after = (opcode == UNPACK_EX) ? (oparg >> 8) : -1;
  Py_ssize_t size = GetUnpackSize(seek(0), cnt, cnt_after);
  if (size == -1) {
    return false;
  }
  ValueNode *iterable = pop();

  size_t elements_size = frame_.GetStacks().size();
  if (!UnpackElements(iterable)) {
    return false;
  }
  elements_size = frame_.GetStacks().size() - elements_size;
  std::vector<ValueNode *> elements(frame_.GetStacks().end() - elements_size, frame_.GetStacks().end());
  popn(elements_size);

  auto gen_item = [this, &elements](int i, int j) {
    if (j == -1) {
      this->push(elements[i]);
      return;
    }
    MS_EXCEPTION_IF_CHECK_FAIL(j >= i, "check UNPACK_EX oparg");
    auto in_iter = elements.begin();
    std::for_each(in_iter + i, in_iter + j, [this](ValueNode *i) { this->push(i); });
    DoBuildOp({BUILD_LIST, j - i});
  };
  GenUnpackValue(gen_item, cnt, cnt_after, size);
  return true;
}

bool GraphBuilder::DoCall(const Instr &instr) {
  int opcode = instr.op();
  int oparg = instr.arg();
  int tmp_arg = oparg;
  std::vector<ValueNode *> params;
  switch (opcode) {
    case CALL_FUNCTION_EX:
      tmp_arg = (tmp_arg & 0x01); /* fall-through */
    case CALL_FUNCTION_KW:        /* fall-through */
      tmp_arg += 1;
    case CALL_METHOD:
    case CALL_FUNCTION:
      params = {frame_.GetStacks().end() - tmp_arg - 1, frame_.GetStacks().end()};
      opcode = (opcode == CALL_METHOD) ? CALL_FUNCTION : opcode;
      popn(tmp_arg + 1);
      push(NewValueNode(nullptr, opcode, oparg, params));
      break;
    default:
      return false;
  }
  CallNode *call_node = static_cast<CallNode *>(seek(0));
  call_node->SetVobj(AObject::MakeAObject(AObject::kTypeAnyValue));
  call_node->SetLineNo(instr.line());
  call_node->set_bci(instr.bci());
  this->graph_->GetTracedNodes().push_back(call_node);

  StopTraceReason r = HandleCall(0);
  if (r != StopTraceReason::kNonStopTrace) {
    graph_->StopTraceAt(cur_bci_, r);
    return false;
  }
  return true;
}

bool GraphBuilder::DoNop(const Instr &instr) { return true; }
bool GraphBuilder::NotImplementBytecode(const Instr &instr) { return false; }

bool GraphBuilder::DoYieldValue(const Instr &instr) {
  ValueNode *result = graph_->GetGeneratorResult();
  if (result == nullptr) {
    result = NewValueNode(nullptr, BUILD_TUPLE, 0);
    graph_->SetGeneratorResult(result);
  }
  ValueNode *value = seek(0);
  result->AddInput(value);
  return true;
}

bool GraphBuilder::DoReturn(const Instr &instr) {
  graph_->SetRetVal(pop());
  if (graph_->GetGeneratorResult() == nullptr) {
    return true;
  }
  const auto &inputs = graph_->GetGeneratorResult()->getInputs();
  std::for_each(inputs.begin(), inputs.end(), [this](ValueNode *i) { this->push(i); });
  DoBuildOp({BUILD_TUPLE, static_cast<int>(inputs.size())});
  ValueNode *new_node = pop();
  graph_->SetGeneratorResult(new_node);
  graph_->SetRetVal(new_node);
  return true;
}

bool GraphBuilder::DoLocalAccess(const Instr &instr) {
  switch (instr.op()) {
    case LOAD_FAST:
      push(getLocal(instr.arg()));
      break;
    case STORE_FAST:
      setLocal(instr.arg(), pop());
      break;
    case DELETE_FAST:
      setLocal(instr.arg(), &ValueNode::kUnboundLocal);
      break;
    default:
      return false;
  }
  return true;
}

bool GraphBuilder::DoCellAccess(const Instr &instr) {
  int opcode = instr.op();
  int oparg = instr.arg();
  ValueNode *node;
  ValueNode *value;
  PyObject *cell = frame_.Closure(oparg)->GetVobj()->GetPyObject().ptr();
  MS_EXCEPTION_IF_CHECK_FAIL(cell && PyCell_Check(cell), "must be a cell object");
  switch (opcode) {
    case LOAD_CLOSURE:
      push(frame_.Closure(oparg));
      break;
    case LOAD_DEREF:
      MS_EXCEPTION_IF_NULL(frame_.Closure(oparg)->GetValue());
      push(frame_.Closure(oparg)->GetValue());
      break;
    case STORE_DEREF:
      value = pop();
      if (value->GetOpcode() == LOAD_DEREF && frame_.Closure(oparg) == frame_.Closure(value->GetOparg())) {
        // same value, no side effect
        break;
      }
      node = NewValueNode(nullptr, instr, {value});
      frame_.Closure(oparg)->SetValue(value);
      frame_.Closure(oparg)->AddCellOper(node);
      current_block_->SetTrackResult(Block::kHasClosureSideEffect);
      break;
    case DELETE_DEREF:
      node = NewValueNode(nullptr, instr, {});
      frame_.Closure(oparg)->SetValue(&ValueNode::kUnboundLocal);
      frame_.Closure(oparg)->AddCellOper(node);
      current_block_->SetTrackResult(Block::kHasClosureSideEffect);
      break;
    default:
      return false;
  }
  return true;
}

// Parse byteCode -- SETUP_WITH
bool GraphBuilder::DoWith(const Instr &instr) {
  auto state = py::error_already_set();
  if (graph_->Config().GetBoolConfig(GraphJitConfig::kSkipException) || state.type().ptr() == PyExc_RuntimeError) {
    graph_->StopTraceAt(cur_bci_, StopTraceReason::kStopTraceSkip_Exception);
    return false;
  }
  auto node = pop();
  push(node);
  DoAttrAccess({LOAD_ATTR, 0, "__exit__"});

  push(node);
  DoAttrAccess({LOAD_ATTR, 0, "__enter__"});

  if (!DoCall({CALL_FUNCTION, 0})) {
    MS_LOG(ERROR) << "function '__enter__' runs failed here, it should be successful!";
    return false;
  }
  PushStack(TryBlock{SETUP_WITH, instr.extra_jump()->bci(), instr.bci(), false});
  cur_bci_++;
  return true;
}

bool GraphBuilder::DoException(const Instr &instr) {
  int opCode = instr.op();
  switch (opCode) {
    case SETUP_WITH:
      return DoWith(instr);
    case POP_BLOCK: {
      PopStack();
      return true;
    }
    case SETUP_FINALLY: {
      /*
        ByteCode like this in python3.9
        0 SETUP_FINALLY    xxx
        1 SETUP_FINALLY    xxx
        the first SETUP_FINALLY points to finally block, the second points to exception block
      */
      if (graph_->Config().GetBoolConfig(GraphJitConfig::kSkipException)) {
        graph_->StopTraceAt(cur_bci_, StopTraceReason::kStopTraceSkip_Exception);
        return false;
      }
      if (StackSize() == 0 || GetTryBlockStacks().back().type != SETUP_FINALLY) {
        PushStack(TryBlock{SETUP_FINALLY, instr.extra_jump()->bci(), instr.bci(), true});
      } else {
        assert(StackSize() > 0 || GetTryBlockStacks().back().type == SETUP_FINALLY);
        PushStack(TryBlock{SETUP_FINALLY, instr.extra_jump()->bci(), instr.bci(), false});
      }
      cur_bci_++;
      return true;
    }
#if (PY_MAJOR_VERSION == 3) && (PY_MINOR_VERSION == 7)
    case WITH_CLEANUP_START: {
      ValueNode *exc = seek(0);
      ValueNode *exit_func = seek(1);
      if (exc->GetVobj()->GetType() != AObject::kTypeNone) {
        return false;
      }
      if (exit_func->GetName() != "__exit__") {
        MS_LOG(ERROR) << "it should call function '__exit__' here!";
        return false;
      }
      // run exit func
      push(exc);
      push(exc);
      auto newInstr = Instr(instr.bci(), CALL_FUNCTION, 3);
      newInstr.set_line(instr.line());
      if (!DoCall(newInstr)) {
        MS_LOG(ERROR) << "function '__exit__' runs failed here, it should be successful!";
        return false;
      }
      push(exc);
      return true;
    }
    case WITH_CLEANUP_FINISH: {
      auto exc = pop();
      (void)pop();
      push(exc);
      return true;
    }
    case END_FINALLY: {
      (void)pop();
      return true;
    }
    case SETUP_EXCEPT: {
      PushStack(TryBlock{SETUP_EXCEPT, instr.extra_jump()->bci(), instr.bci(), false});
      cur_bci_++;
      return true;
    }
#endif
    default:
      break;
  }
  return false;
}

TryBlock &GraphBuilder::PeekStack(int p) {
  MS_ASSERT(tryBlockStacks_.size() > p);
  return tryBlockStacks_[tryBlockStacks_.size() - p - 1];
}

TryBlock &GraphBuilder::PopStack() {
  MS_ASSERT(tryBlockStacks_.size() > 0);
  auto &tb = tryBlockStacks_[tryBlockStacks_.size() - 1];
  tryBlockStacks_.pop_back();
  return tb;
}

bool GraphBuilder::DoGlobalAccess(const Instr &instr) {
  int opcode = instr.op();
  int oparg = instr.arg();
  switch (opcode) {
    case LOAD_GLOBAL: {
      if (!graph_->GetSideEffect()->GetGlobalList().empty()) {
        for (auto global_side_effect : graph_->GetSideEffect()->GetGlobalList()) {
          if (global_side_effect.getModule() == GetGraph()->GetModuleName() &&
              global_side_effect.getName() == instr.name()) {
            push(global_side_effect.getNode());
          }
        }
        break;
      } else {
        auto co = graph_->GetCodeObj();
        PyObject *key = PyTuple_GET_ITEM(co->co_names, oparg);
        // NOTE: will run __get__, __hash__ function
        PyObject *obj = PyObject_GetItem(graph_->GetGlobals().ptr(), key);
        if (obj == nullptr) {
          PyErr_Clear();
          obj = PyObject_GetItem(PyEval_GetBuiltins(), key);
        }
        py::object pyobj = py::reinterpret_steal<py::object>(obj);
        auto n = NewValueNode(AObject::Convert(pyobj), instr, {});
        n->SetName(PyUnicode_AsUTF8(key));
        push(n);
        break;
      }
    }
    case STORE_GLOBAL: {
      auto global_node = pop();
      GlobalSideEffectNode global_side_effect(instr.name(), global_node, GetGraph()->GetModuleName());
      graph_->GetSideEffect()->setGlobalList(global_side_effect);
      break;
    }
    case DELETE_GLOBAL: {
      GlobalSideEffectNode global_side_effect(instr.name(), nullptr, GetGraph()->GetModuleName());
      graph_->GetSideEffect()->setGlobalList(global_side_effect);
      break;
    }

    default:
      return false;
  }
  return true;
}

bool GraphBuilder::HandleSuper(const Instr &instr, AObject *super) {
  if (super->GetTypeObject() != &PySuper_Type) {
    return false;
  }
  ValueNode *self_super = SearchSelfPyObject(graph_->GetCodeObj()).second;
  if (self_super == nullptr) {
    return false;
  }
  py::object method = super->GetPyObject().attr(instr.name().c_str());
  if (!PyMethod_Check(method.ptr())) {
    return false;
  }

  // method type object
  auto mtype_obj = reinterpret_cast<PyObject *>(&PyMethod_Type);
  DoLoadConst({LOAD_CONST, -1, py::cast<py::object>(mtype_obj)});

  // function object
  PyObject *m = PyMethod_GET_FUNCTION(method.ptr());
  DoLoadConst({LOAD_CONST, -1, py::cast<py::object>(m)});

  push(self_super);

  // call method type
  return DoCall({CALL_FUNCTION, 2});
}

PyObject *SetLocalPyObject(ValueNode *node) {
  if (node == nullptr || node->GetVobj() == nullptr) {
    return NULL;
  } else {
    return node->GetVobj()->GetPyObject().ptr();
  }
}

std::pair<PyObject *, ValueNode *> GraphBuilder::SearchSelfPyObject(PyCodeObject *co) {
  if (co->co_argcount < 1) {
    return {nullptr, nullptr};
  }
  std::pair<PyObject *, ValueNode *> obj_value;
  ValueNode *value = frame_.Local(0);
  // get self or son class, eg.super(Son, self)
  PyObject *obj = SetLocalPyObject(frame_.Local(0));
  Py_ssize_t i, n;
  if (obj == NULL && co->co_cell2arg) {
    // the first argument might be a cell
    n = PyTuple_GET_SIZE(co->co_cellvars);
    for (i = 0; i < n; i++) {
      if (co->co_cell2arg[i] == 0) {
        value = frame_.Closure(i)->GetValue();
        obj = SetLocalPyObject(frame_.Closure(i));
        break;
      }
    }
  }
  obj_value = std::make_pair(obj, value);
  return obj_value;
}

bool GraphBuilder::DoAttrAccess(const Instr &instr) {
  int opcode = instr.op();
  switch (opcode) {
    case LOAD_METHOD: /* fall-through */
    case LOAD_ATTR: {
      auto o = pop();
      if (HandleSuper(instr, o->GetVobj())) {
        break;
      }
      auto attrs = o->GetAttrs();
      if (attrs.find(instr.name().c_str()) != attrs.end()) {
        push(attrs[instr.name().c_str()]);
      } else {
        auto n = NewValueNode(o->get_attr(instr.name()), instr, {o});
        push(n);
      }
      break;
    }
    case STORE_ATTR: {
      auto o = pop();
      auto v = pop();
      o->store_attr(instr.name().c_str(), v);
      NewValueNode(nullptr, instr, {v, o});
      current_block_->SetTrackResult(Block::kHasAttrSideEffect);
      break;
    }
    case DELETE_ATTR: {
      auto o = pop();
      o->del_attr(instr.name().c_str());
      NewValueNode(nullptr, instr, {o});
      current_block_->SetTrackResult(Block::kHasAttrSideEffect);
      break;
    }
    default:
      return false;
  }
  return true;
}

// for unpack call optimize
static ValueNode *TupleDictGetItem(ValueNode *container, ValueNode *index_node) {
  if (!index_node->IsConstantValue()) {
    return nullptr;
  }
  PyObject *index_object = index_node->GetVobj()->GetPyObject().ptr();
  int opcode = container->GetOpcode();
  if ((opcode == BUILD_TUPLE || opcode == BUILD_LIST) && PyLong_Check(index_object)) {
    Py_ssize_t index = PyLong_AsSsize_t(index_object);
    Py_ssize_t size = container->getInputs().size();
    if (index < -size || index >= size) {
      return nullptr;
    }
    index = index < 0 ? (size + index) : index;
    return container->input(index);
  }
  if (container->GetOpcode() == BUILD_MAP && PyUnicode_Check(index_object)) {
    std::string k = PyUnicode_AsUTF8(index_object);
    size_t element_count = container->GetOparg() << 1;
    MS_EXCEPTION_IF_CHECK_FAIL(element_count == container->getInputs().size(), "check BUILD_MAP oparg");
    for (int i = 0; i < container->GetOparg(); ++i) {
      AObject *tmp = container->input(i * 2)->GetVobj();
      PyObject *str = tmp ? tmp->GetPyObject().ptr() : nullptr;
      if (str == nullptr || !PyUnicode_Check(str) || k != PyUnicode_AsUTF8(str)) {
        continue;
      }
      return container->input((i << 1) + 1);
    }
  }
  return nullptr;
}

bool GraphBuilder::DoGetItem(const Instr &instr) {
  constexpr const char *kNameGetItem = "__getitem__";
  auto r = pop();
  auto l = pop();
  ValueNode *v = TupleDictGetItem(l, r);
  if (v != nullptr) {
    push(v);
    return true;
  }

  AObject *container = l->GetVobj();
  PyObject *op = container ? container->GetPyObject().ptr() : nullptr;
  AObject *meth = nullptr;

  bool call_getitem = op == nullptr || container->GetType() != AObject::kTypeAnyValue;
  if (!call_getitem) {
    call_getitem = PyDict_Check(op) || PyTuple_Check(op) || PyList_Check(op);
  }
  if (!call_getitem) {
    meth = container->GetAttr(kNameGetItem);
    PyObject *m = meth ? meth->GetPyObject().ptr() : nullptr;
    call_getitem = m == nullptr || !PyMethod_Check(m) || !PyFunction_Check(PyMethod_GET_FUNCTION(m));
  }
  if (call_getitem) {
    /**
     * TODO:
     * check safe callable of __getitem__ if user defined.
     */
    AObject *vo = l->binary_subscr(r);
    v = NewValueNode(vo, instr, {l, r});
    push(v);
    return true;
  }

  push(l);
  DoAttrAccess({LOAD_ATTR, 0, kNameGetItem});
  push(r);
  return DoCall({CALL_FUNCTION, 1});
}

bool GraphBuilder::DoSetItem(ValueNode *map, ValueNode *key, ValueNode *val) {
  if (map->GetOpcode() == BUILD_LIST) {
    PyObject *index_object = key->GetVobj()->GetPyObject().ptr();
    if (index_object == nullptr || !PyLong_CheckExact(index_object)) {
      return false;
    }
    Py_ssize_t index = PyLong_AsSsize_t(index_object);
    Py_ssize_t size = map->getInputs().size();
    if (index < -size || index >= size) {
      return false;
    }
    index = index < 0 ? (size + index) : index;
    std::vector<ValueNode *> inputs = map->getInputs();
    inputs[index] = val;

    std::for_each(inputs.begin(), inputs.end(), [this](ValueNode *i) { this->push(i); });
    DoBuildOp({BUILD_LIST, static_cast<int>(inputs.size())});
  } else if (map->GetOpcode() == BUILD_MAP) {
    std::vector<ValueNode *> inputs = map->getInputs();
    inputs.push_back(key);
    inputs.push_back(val);

    std::for_each(inputs.begin(), inputs.end(), [this](ValueNode *i) { this->push(i); });
    DoBuildOp({BUILD_MAP, static_cast<int>(inputs.size() / 2)});
  } else {
    return false;
  }

  if (!ReplaceAll(map, pop())) {
    graph_->GetTracedNodes().pop_back();
    return false;
  }
  return true;
}

bool GraphBuilder::DoSideEffect(const Instr &instr, const std::vector<ValueNode *> &p) {
  // only handle list dict ; after mark sideeffect, erase value node
  auto key = p[0];
  auto container = p[1];
  auto value = p[2];
  ValueNode *new_node;
  PyObject *container_object = container->GetVobj()->GetPyObject().ptr();
  if (container_object == nullptr || !PyList_Check(container_object)) {
    return false;
  }
  std::vector<ValueNode *> items;
  Py_ssize_t py_list_size = PyList_Size(container_object);
  for (Py_ssize_t i = 0; i < py_list_size; i++) {
    // BINARY_SUBSCR
    ValueNode *index_node = NewValueNode(AObject::Convert(py::int_(i)), LOAD_CONST, -1, {});
    PyObject *py_object_item = PyList_GET_ITEM(container_object, i);
    ValueNode *item_node = NewValueNode(AObject::Convert(py_object_item), BINARY_SUBSCR, 0, {container, index_node});
    items.push_back(item_node);
  }
  PyObject *key_object = key->GetVobj()->GetPyObject().ptr();
  Py_ssize_t index = PyLong_AsSsize_t(key_object);
  Py_ssize_t size = py_list_size;
  if (index < -size || index >= size) {
    return false;
  }
  index = index < 0 ? (size + index) : index;
  items[index] = value;
  AObject *object_info = AObject::BuildOperations(CollectObjects(items), BUILD_LIST);
  new_node = NewValueNode(object_info, BUILD_LIST, items.size(), items);
  if (!ReplaceAll(container, new_node)) {
    return false;
  }
  for (auto item : items) {
    graph_->GetTracedNodes().push_back(item);
  }

  graph_->GetSideEffect()->setVariableMaps(new_node, value);
  graph_->GetSideEffect()->setReplaceMaps(new_node, container);

  graph_->GetTracedNodes().push_back(new_node);
  return true;
}

bool GraphBuilder::DoItemAccess(const Instr &instr) {
  int opcode = instr.op();
  switch (opcode) {
    case BINARY_SUBSCR: {
      DoGetItem(instr);
      break;
    }
    case STORE_SUBSCR: {
      // STORE_SUBSCR: k->index v->item m-> container
      auto k = pop();
      auto m = pop();
      auto v = pop();
      if (DoSetItem(m, k, v) == true) {
        return true;
      }
      if (m->GetVobj()->GetType() == AObject::kTypeList) {
        if (k->GetOpcode() == BUILD_SLICE) {
          return false;
        }
        DoSideEffect(instr, {k, m, v});
        break;
      } else {
        NewValueNode(nullptr, instr, {v, m, k});
        current_block_->SetTrackResult(Block::kHasAttrSideEffect);
        break;
      }
    }
    case DELETE_SUBSCR: {
      auto sub = pop();  // sub
      auto obj = pop();  // obj
      obj->del_subscr(sub);
      NewValueNode(nullptr, instr, {obj, sub});
      current_block_->SetTrackResult(Block::kHasAttrSideEffect);
      break;
    }
    default:
      return false;
  }
  return true;
}

bool GraphBuilder::DoStackOp(const Instr &instr) {
  int opcode = instr.op();
  int oparg = instr.arg();
  int tmp_arg = oparg;
  switch (opcode) {
    case POP_TOP:
      pop();
      break;
    case ROT_TWO:
      tmp_arg = 1;
      /* fall-through */
    case ROT_THREE:
      tmp_arg = tmp_arg ? tmp_arg : 2;
      /* fall-through */
    case ROT_FOUR: {
      tmp_arg = tmp_arg ? tmp_arg : 3;
      frame_.Rot(tmp_arg);
      break;
    }
    case DUP_TOP_TWO:
      push(seek(1));
      push(seek(1));
      break;
    case DUP_TOP:
      push(seek(0));
      break;
    default:
      return false;
  }
  return true;
}

bool GraphBuilder::DoLoadConst(const Instr &instr) {
  auto n = NewValueNode(AObject::Convert(instr.cnst()), instr, {});
  push(n);
  return true;
}

bool GraphBuilder::DoListToTuple(const Instr &instr) {
  ValueNode *list = pop();
  if (list->GetOpcode() == BUILD_LIST) {
    std::for_each(list->getInputs().begin(), list->getInputs().end(), [this](ValueNode *i) { this->push(i); });
    return DoBuildOp({BUILD_TUPLE, static_cast<int>(list->getInputs().size())});
  }
  AObject *vo = list->GetVobj();
  if (vo && vo->GetType() == AObject::kTypeList) {
    vo = static_cast<AbstractList *>(vo)->ListToTuple();
  } else {
    vo = AObject::MakeAObject(AObject::kTypeAnyValue);
  }
  ValueNode *tuple = NewValueNode(vo, instr, {list});
  push(tuple);
  return true;
}

bool GraphBuilder::DoGetIter(const Instr &instr) {
  auto obj = pop();
  auto o = obj->GetVobj();
  auto iter = NewValueNode(o ? o->GetIter() : AObject::MakeAObject(AObject::kTypeAnyValue), instr, {obj});
  push(iter);
  iter->marker_ = 0;
  return true;
}

bool GraphBuilder::DoMakeFunction(const Instr &instr) {
  int oparg = instr.arg();
  // int cnt = __builtin_popcount(oparg & 0xf) + 2;
  int cnt = !!(oparg & 0x08) + !!(oparg & 0x04) + !!(oparg & 0x02) + !!(oparg & 0x01) + 2;
  std::vector<ValueNode *> p(frame_.GetStacks().end() - cnt, frame_.GetStacks().end());
  popn(cnt);
  AObject *f = AObject::MakeFunction(CollectObjects(p), graph_->GetGlobals(), oparg);
  ValueNode *func = NewValueNode(f, instr, p);
  push(func);
  current_block_->SetTrackResult(Block::kHasGlobalSideEffect);
  return true;
}

AObject *GraphBuilder::InferUnary(ValueNode *node, const Instr &instr) { return node->GetVobj()->Unary(instr.op()); }

bool GraphBuilder::DoUnary(const Instr &instr) {
  ValueNode *node = pop();

  AObject *object_info = InferUnary(node, instr);
  ValueNode *new_node = NewValueNode(object_info, instr, {node});
  push(new_node);
  return true;
}

bool GraphBuilder::DoIsOp(const Instr &instr) { return DoBinary(instr); }

AObject *GraphBuilder::InferBinary(ValueNode *left, ValueNode *right, const Instr &instr) {
  AObject *object_info;
  if (instr.op() == IS_OP || instr.op() == CONTAINS_OP) {
    object_info = left->GetVobj()->Binary(right->GetVobj(), instr.op());
    PyObject *object = object_info != nullptr ? object_info->GetPyObject().ptr() : nullptr;
    if (object != nullptr) {
      object_info = AObject::Convert(py::bool_((object == Py_True) ^ instr.arg()));
    }
  } else if (Utils::IsBinaryMathOp(instr.op())) {
    if (left->IsConstantValue() && right->IsConstantValue()) {
      // compute real tensor value, not infer fake value
      AbstractObject *tensor = static_cast<AbstractObject *>(left->GetVobj());
      object_info = tensor->AbstractObject::Binary(right->GetVobj(), instr.op());
    } else {
      object_info = left->GetVobj()->Binary(right->GetVobj(), instr.op());
    }
  } else {
    return AObject::MakeAObject(AObject::kTypeAnyValue);
  }
  return object_info;
}

bool GraphBuilder::DoBinary(const Instr &instr) {
  ValueNode *right = pop();
  ValueNode *left = pop();

  AObject *object_info = InferBinary(left, right, instr);
  ValueNode *new_node = NewValueNode(object_info, instr, {left, right});
  push(new_node);
  return true;
}

static bool CheckTupleListMul(ValueNode *left, ValueNode *right) {
  bool special = left->GetOpcode() == BUILD_LIST || left->GetOpcode() == BUILD_TUPLE || left->IsConstantValue();
  if (special && right->IsConstantValue()) {
    PyObject *mul = right->GetVobj()->GetPyObject().ptr();
    const int max = 2;
    return PyLong_Check(mul) && Py_ABS(Py_SIZE(mul)) < max;
  }
  return false;
}

bool GraphBuilder::DoBinaryMul(const Instr &instr) {
  if (!CheckTupleListMul(seek(1), seek(0))) {
    return DoBinary(instr);
  }

  ValueNode *right = pop();
  ValueNode *left = pop();
  int l_op = left->GetVobj()->GetType() == AObject::kTypeTuple ? BUILD_TUPLE : BUILD_LIST;

  Py_ssize_t mul = PyLong_AsSsize_t(right->GetVobj()->GetPyObject().ptr());
  for (auto i = mul; i > 0; --i) {
    UnpackElements(left);
  }
  int oparg = left->getInputs().size() * (mul < 0 ? 0 : size_t(mul));
  DoBuildOp({l_op, oparg});
  return true;
}

static bool CheckTupleListAdd(ValueNode *left, ValueNode *right) {
  // type must be same
  AObject::Type l_type = left->GetVobj()->GetType();
  AObject::Type r_type = right->GetVobj()->GetType();
  bool support = l_type == AObject::kTypeTuple || l_type == AObject::kTypeList;
  if (!support || l_type != r_type) {
    return false;
  }
  // only handle BUILD_TUPLE and BUILD_LIST
  int l_op = left->GetOpcode();
  int r_op = right->GetOpcode();
  bool special = l_op == BUILD_TUPLE || l_op == BUILD_LIST || l_op == LOAD_CONST;
  bool accept = r_op == BUILD_TUPLE || r_op == BUILD_LIST || r_op == LOAD_CONST;
  if (!special || !accept) {
    return false;
  }
  return true;
}

bool GraphBuilder::DoInplaceAdd(const Instr &instr) {
  AObject::Type l_type = seek(1)->GetVobj()->GetType();
  if (l_type == AObject::kTypeTuple) {
    return DoBinaryAdd(instr);
  }
  if (!CheckTupleListAdd(seek(1), seek(0))) {
    return DoBinary(instr);
  }

  ValueNode *right = pop();
  ValueNode *left = pop();
  int l_op = BUILD_LIST;

  int size = this->frame_.GetStacks().size();
  UnpackElements(left);
  UnpackElements(right);
  size = this->frame_.GetStacks().size() - size;
  DoBuildOp({l_op, size});

  ValueNode *new_node = pop();
  if (ReplaceAll(left, new_node)) {
    push(new_node);
    return true;
  }
  graph_->GetTracedNodes().pop_back();
  push(left);
  push(right);
  return DoBinary(instr);
}

bool GraphBuilder::DoBinaryAdd(const Instr &instr) {
  if (!CheckTupleListAdd(seek(1), seek(0))) {
    return DoBinary(instr);
  }

  ValueNode *right = pop();
  ValueNode *left = pop();
  int l_op = left->GetVobj()->GetType() == AObject::kTypeTuple ? BUILD_TUPLE : BUILD_LIST;

  int size = this->frame_.GetStacks().size();
  UnpackElements(left);
  UnpackElements(right);
  size = this->frame_.GetStacks().size() - size;
  DoBuildOp({l_op, size});
  return true;
}

bool GraphBuilder::DoCompare(const Instr &instr) {
  int oparg = instr.arg();
  auto r = pop();
  auto l = pop();

#if (PY_MAJOR_VERSION == 3 && PY_MINOR_VERSION < 9)
  AObject *tmp;
  int opcode = instr.op();
  bool invert = false;
  switch (oparg) {
    case PyCmp_IS:
    case PyCmp_IS_NOT:
      opcode = IS_OP;
    case PyCmp_IN:
    case PyCmp_NOT_IN:
      tmp = l->GetVobj() ? l->GetVobj()->Binary(r->GetVobj(), opcode == IS_OP ? IS_OP : CONTAINS_OP)
                         : AObject::MakeAObject(AObject::kTypeAnyValue);
      invert = (oparg == PyCmp_IS_NOT || oparg == PyCmp_NOT_IN);
      if (invert && tmp && tmp->GetPyObject().ptr()) {
        bool res = (tmp->GetPyObject().ptr() == Py_True) ^ invert;
        tmp = AObject::Convert(py::bool_(res));
      }
      push(NewValueNode(tmp, instr, {l, r}));
      return true;
    case PyCmp_EXC_MATCH:
      return false;
    default:
      break;
  }
#endif

  AObject *o = AObject::MakeAObject(AObject::kTypeBool);
  PyObject *left = l->GetVobj() ? l->GetVobj()->GetPyObject().ptr() : nullptr;
  PyObject *right = r->GetVobj() ? r->GetVobj()->GetPyObject().ptr() : nullptr;
  if (left && right && CheckValueValid(l->GetVobj()) && CheckValueValid(r->GetVobj())) {
    o = AObject::Convert(PyObject_RichCompare(left, right, oparg));
    PyErr_Clear();
  }
  auto v = NewValueNode(o, instr, {l, r});
  push(v);
  return true;
}

bool GraphBuilder::DoBuildOp(const Instr &instr) {
  int opcode = instr.op();
  int oparg = instr.arg();
  int tmp_arg = oparg;
  tmp_arg += opcode == BUILD_CONST_KEY_MAP;
  tmp_arg += opcode == BUILD_MAP ? tmp_arg : 0;
  std::vector<ValueNode *> p(frame_.GetStacks().end() - tmp_arg, frame_.GetStacks().end());
  popn(tmp_arg);

  ValueNode *v;
  if (opcode == BUILD_CONST_KEY_MAP) {
    PyObject *keys = p.back()->GetVobj()->GetPyObject().ptr();
    MS_EXCEPTION_IF_CHECK_FAIL(keys && PyTuple_CheckExact(keys), "error bytecode BUILD_CONST_KEY_MAP");
    Py_ssize_t size = PyTuple_GET_SIZE(keys);
    MS_EXCEPTION_IF_CHECK_FAIL(size_t(size) + 1 == p.size(), "error args BUILD_CONST_KEY_MAP");
    std::vector<ValueNode *> build_inputs;
    for (Py_ssize_t i = 0; i < size; ++i) {
      PyObject *item = PyTuple_GET_ITEM(keys, i);
      build_inputs.push_back(NewValueNode(AObject::Convert(item), LOAD_CONST, -1));
      build_inputs.push_back(p[i]);
    }
    AObject *vo = AObject::BuildOperations(CollectObjects(build_inputs), BUILD_MAP);
    v = NewValueNode(vo, instr, build_inputs);
    v->SetOpcode(BUILD_MAP);
    v->SetOparg(size);
  } else {
    AObject *vo = AObject::BuildOperations(CollectObjects(p), opcode);
    v = NewValueNode(vo, instr, p);
  }
  push(v);
  return true;
}

ValueNode *GraphBuilder::ReplaceMergeOp(int opcode, const std::vector<ValueNode *> &inputs) {
  ValueNode *origin = inputs[0];
  ValueNode *arg = inputs[1];
  ValueNode *arg2 = inputs.size() > 2 ? inputs[2] : nullptr;
  if (origin->GetOpcode() != BUILD_LIST && origin->GetOpcode() != BUILD_MAP) {
    return nullptr;
  }
  std::vector<ValueNode *> build_inputs = origin->getInputs();
  int div = 2;
  switch (opcode) {
    case LIST_APPEND:
      build_inputs.push_back(arg);
      opcode = BUILD_LIST;
      div = 1;
      break;
    case LIST_EXTEND:
      if (arg->IsConstantValue()) {
        build_inputs = UnpackConstObject(arg->GetConstantInfo()->value());
      } else if (arg->GetOpcode() == BUILD_LIST || arg->GetOpcode() == BUILD_TUPLE) {
        build_inputs.insert(build_inputs.end(), arg->getInputs().begin(), arg->getInputs().end());
      } else {
        return nullptr;
      }
      opcode = BUILD_LIST;
      div = 1;
      break;
    case DICT_MERGE:
      // NOTE: here not check duplicate key, will not exception if function call is inlined
    case DICT_UPDATE:
      if (arg->GetOpcode() != BUILD_MAP) {
        return nullptr;
      }
      build_inputs.insert(build_inputs.end(), arg->getInputs().begin(), arg->getInputs().end());
      opcode = BUILD_MAP;
      break;
    case MAP_ADD:
      build_inputs.push_back(arg);
      build_inputs.push_back(arg2);
      opcode = BUILD_MAP;
      break;
    default:
      return nullptr;
  }
  std::for_each(build_inputs.begin(), build_inputs.end(), [this](ValueNode *i) { this->push(i); });
  int oparg = build_inputs.size() / div;
  DoBuildOp({opcode, oparg});
  return pop();
}

bool GraphBuilder::DoMergeOp(const Instr &instr) {
  int opcode = instr.op();
  int oparg = instr.arg();
  int pos = oparg + (opcode == MAP_ADD);

  int index = this->frame_.GetStacks().size() - 1 - pos;
  ValueNode *container = seek(pos);
  std::vector<ValueNode *> inputs = {container, pop()};
  if (opcode == MAP_ADD) {
    inputs.insert(inputs.begin() + 1, pop());
  }

  // DICT_MERGE only generated when unpack-call in python3.9, all keys must be string
  // NOTE: DICT_MERGE opcode requires that *(stack_pointer - oparg - 2) is a function if has duplicate key
  // ...
  ValueNode *new_node = ReplaceMergeOp(opcode, inputs);
  if (new_node != nullptr) {
    this->frame_.GetStacks()[index] = new_node;
    return true;
  }

  std::vector<AObject *> args_info = {inputs[1]->GetVobj()};
  if (opcode == MAP_ADD) {
    args_info.push_back(inputs[2]->GetVobj());
  }
  AObject *vo = AObject::MergeOperations(container->GetVobj(), args_info, opcode);
  MS_EXCEPTION_IF_CHECK_FAIL(this->frame_.GetStacks()[index] == container, "stack error");
  this->frame_.GetStacks()[index] = NewValueNode(vo, instr, inputs);
  return true;
}

bool GraphBuilder::DoFormatValue(const Instr &instr) {
  int oparg = instr.arg();
  std::vector<ValueNode *> arg;
  if ((oparg & FVS_MASK) == FVS_HAVE_SPEC) {
    arg.push_back(pop());
  }
  arg.insert(arg.begin(), pop());
  auto vo = AObject::MakeAObject(AObject::kTypeString);
  auto v = NewValueNode(vo, instr, arg);
  push(v);
  return true;
}

bool GraphBuilder::DoImport(const Instr &instr) {
  int opcode = instr.op();
  switch (opcode) {
    case IMPORT_FROM: {
      // any object
      push(NewValueNode(AObject::MakeAObject(AObject::kTypeAnyValue), instr, {seek(0)}));
      break;
    }
    case IMPORT_STAR: {
      auto from = pop();
      NewValueNode(AObject::MakeAObject(AObject::kTypeAnyValue), instr, {from});
      break;
    }
    case IMPORT_NAME: {
      auto from_list = pop();
      auto level = pop();
      auto vo = AObject::MakeAObject(AObject::kTypeModule);
      auto v = NewValueNode(vo, instr, {level, from_list});
      push(v);
      break;
    }
    default:
      return false;
  }
  return true;
}

bool GraphBuilder::DoByteCode(const Instr &instr) {
  if (current_block_->is_loop_head() && !graph_->Config().GetBoolConfig(GraphJitConfig::kLoopUnrolling)) {
    graph_->StopTraceAt(cur_bci_, StopTraceReason::kStopTraceLoop_Unsupported);
    return false;
  }

  MS_EXCEPTION_IF_CHECK_FAIL(bytecode_meth_map_.find(instr.op()) != bytecode_meth_map_.end(),
                             "unknown opcode " + std::to_string(instr.op()));
  const auto func = bytecode_meth_map_.find(instr.op())->second;
  bool support = (this->*func)(instr);

  const auto &nodes = graph_->GetTracedNodes();
  for (auto i = nodes.rbegin(); i != nodes.rend() && (*i)->GetBlock() == nullptr; ++i) {
    (*i)->SetBlock(current_block_);
  }

  if (instr.op() == RETURN_VALUE) {
    return false;
  }

  if (!support) {
    if (graph_->GetStopTraceBci() == -1) {
      graph_->StopTraceAt(cur_bci_, StopTraceReason::kStopTraceByteCode_Unsupported);
    }
    return false;
  }

  if (instr.extra_jump() == nullptr) {
    ++cur_bci_;
  } else {
    bool valid = (cur_bci_ == instr.bci() + 1) || cur_bci_ == instr.extra_jump()->bci();
    MS_EXCEPTION_IF_CHECK_FAIL(valid, "error jump target");
  }
  if (cur_bci_ < current_block_->begin_ci() || cur_bci_ >= current_block_->end_ci()) {
    current_block_ = graph_->GetCFG()->GetBlockByBci(cur_bci_);
  }
  return true;
}

GraphBuilder::GraphBuilder(const PyFrameObject *f)
    : root_(this), parent_(nullptr), graph_(nullptr), current_block_(nullptr) {
  PyCodeObject *co = f->f_code;
  int argc = co->co_argcount + co->co_kwonlyargcount;
  argc += (co->co_flags & CO_VARARGS) ? 1 : 0;
  argc += (co->co_flags & CO_VARKEYWORDS) ? 1 : 0;
  int ncells = PyTuple_GET_SIZE(co->co_cellvars);
  int nfrees = PyTuple_GET_SIZE(co->co_freevars);

  graph_ = NewGraph(co, f->f_globals);

  frame_.ResizeLocal(co->co_nlocals);
  frame_.ResizeClosure(ncells + nfrees);
  for (int i = 0; i < argc; i++) {
    if (f->f_localsplus[i] == nullptr) {
      continue;
    }
    auto vo = AObject::Convert(f->f_localsplus[i]);
    ParamNode *n = graph_->allocator().NewNode<ParamNode>(vo, i);
    n->SetName(PyUnicode_AsUTF8(PyTuple_GET_ITEM(co->co_varnames, i)));
    frame_.SetLocal(i, n);
  }
  for (int i = 0; i < ncells + nfrees; i++) {
    PyObject *cell = f->f_localsplus[co->co_nlocals + i];
    PyObject *cell_contents = PyCell_GET(cell);
    AbstractNode::Type t = i < ncells ? AbstractNode::CellVar : AbstractNode::FreeVar;
    CellVarNode *n = graph_->allocator().NewNode<CellVarNode>(t);
    n->SetVobj(AObject::Convert(cell));
    n->SetIndex(i);
    n->SetGraph(graph_);
    frame_.SetClosure(i, n);
    if (i < ncells && co->co_cell2arg != nullptr && co->co_cell2arg[i] != CO_CELL_NOT_AN_ARG) {
      MS_EXCEPTION_IF_NULL(cell_contents);
      n->SetFromParam(co->co_cell2arg[i]);
    }
    if (cell_contents == nullptr) {
      n->SetValue(&ValueNode::kUnboundLocal);
    } else {
      ValueNode *param = NewValueNode(AObject::Convert(cell_contents), LOAD_DEREF, i);
      param->SetGraph(graph_);
      n->AddCellOper(param);
      n->SetValue(param);
    }
  }
}

void GraphBuilder::CollectInlineInfo(CallNode *node, int depth) {
  Graph *sub_graph = node->GetSubGraph();
  if (!sub_graph) {
    return;
  }
  std::string inline_name = "";
  int code_size = 0;
  if (sub_graph != nullptr && sub_graph->GetCodeObj() != nullptr) {
    inline_name = py::str(reinterpret_cast<PyObject *>(sub_graph->GetCodeObj())).cast<std::string>();
    code_size = static_cast<int>((PyBytes_GET_SIZE(sub_graph->GetCodeObj()->co_code)) / sizeof(_Py_CODEUNIT));
  }
  std::string func_name = graph_->GetCodeName();
  std::string root_name = root_->GetGraph()->GetCodeName();
  JitCompileResults *jcr = getJitCompileResults(reinterpret_cast<PyObject *>(root_->GetGraph()->GetCodeObj()), false);
  if (jcr && jcr->tbs && !func_name.empty()) {
    jcr->tbs->PushInlineInfo(
      {func_name, inline_name, root_name, node->GetInlineReason(), code_size, depth, node->GetLineNo()});
  }
}

void GraphBuilder::HandleLoop() {
  Block *loop_head = graph_->GetCFG()->GetBlockByBci(cur_bci_);
  if (!loop_head->is_loop_head()) {
    return;
  }
  /**
   * TODO(chaiyouheng): before trace start, unrolling loop. avoid graph status is changed while trace loop
   *       just unrolling a small loop that call nn.CellList.
   *
   * LoopUnrolling loopUnrollingExe = LoopUnrolling(*graph_);
   * (void)loopUnrollingExe.ExecuteLoopUnroll(loop_head);
   */
}

py::object GraphBuilder::FindPyFunc(AObject *vobj) {
  if (!vobj) {
    return py::cast<py::object>(nullptr);
  }

  switch (vobj->GetType()) {
    case AObject::kTypeCell:
      vobj = vobj->GetAttr(ID_construct);
      break;
    case AObject::kTypeAnyValue:
      vobj = vobj->GetAttr(ID___call__);
      break;
    case AObject::kTypeType:
      vobj = vobj->GetAttr("__init__");
      break;
    case AObject::kTypeBoundMethod:
      vobj = vobj->GetAttr("__func__");
    default:
      break;
  }
  py::object func = vobj ? vobj->GetPyObject() : py::object();

  if (func.ptr() == nullptr) {
    PyErr_Clear();
    return py::cast<py::object>(nullptr);
  }

  if (PyMethod_Check(func.ptr())) {
    func = py::reinterpret_borrow<py::object>(PyMethod_GET_FUNCTION(func.ptr()));
  }

  if (PyFunction_Check(func.ptr())) {
    return func;
  }
  return py::cast<py::object>(nullptr);
}

py::object GraphBuilder::GetFuncInfo(ValueNode *func_node) {
  AObject *vobj = func_node->GetVobj();
  if (vobj->GetType() == AObject::kTypeCFunction) {
    return py::object();
  }
  if (func_node->GetOpcode() == MAKE_FUNCTION) {
    return func_node->GetVobj()->GetPyObject();
  }
  return FindPyFunc(vobj);
}

bool GraphBuilder::WhiteListFuncCheckAndInfer(CallNode *call_node, const py::object &callable) {
  const auto &conf = call_node->GetGraph()->Config();

  bool cell_inline = conf.GetBoolConfig(GraphJitConfig::kReplaceNNCellByConstruct);
  AObject::Type vobj_type = call_node->input(0)->GetVobj()->GetType();
  if (vobj_type == AObject::kTypeCell) {
    current_block_->SetTrackResult(Block::kTrackHasOpsPrimitive);
    std::string module_name = GetTopModule(callable);
    if (!module_name.empty()) {
      kPIJitConfigDefault.AddAllowedInlineModules(module_name);
    }
  }

  // handle special function, not inline
  bool infer_primitive = conf.GetBoolConfig(GraphJitConfig::kInferPrimitive);
  int max_infer = conf.getIntConfig(GraphJitConfig::kInferPrimitiveMax);
  if (max_infer != 0 && infer_func_count >= max_infer) {
    infer_primitive = false;
  } else {
    infer_func_count++;
  }
  infer_primitive &= (conf.getIntConfig(GraphJitConfig::kInferPrimitiveMask) & infer_primitive_func) != 0;
  std::string special_func_key;
  if (IsFuncInWhiteList(callable, &special_func_key, infer_primitive)) {
    call_node->SetSubGraph(NewGraph(nullptr, nullptr));
    call_node->GetSubGraph()->SetGuard(root_->GetGraph()->GetGuard());
    if (!HandleFuncInWhiteList(special_func_key, call_node)) {
      return false;
    }
    if (call_node->GetSubGraph() == nullptr) {
      call_node->SetInlineReason(InlineReason::kInlineFuncSpecialize);
    } else {
      MS_EXCEPTION_IF_NULL(call_node->GetSubGraph()->GetRetVal());
      call_node->SetInlineReason(InlineReason::kInline);
      seek(0) = call_node->GetSubGraph()->GetRetVal();
    }
    return true;
  }

  // set node info before return
  if (vobj_type == AObject::kTypePrimitive || (vobj_type == AObject::kTypeCell && !cell_inline)) {
    call_node->SetVobj(AObject::MakeAObject(AObject::kTypeTensor));
    call_node->SetInlineReason(InlineReason::kInlineGraphSupportedByMS);
    current_block_->SetTrackResult(Block::kTrackHasOpsPrimitive);
    return true;
  }
  return false;
}

bool UnsupportedCodeTypeCheck(PyCodeObject *co) {
  if (co->co_flags & (CO_GENERATOR | CO_COROUTINE | CO_ASYNC_GENERATOR)) {
    MS_LOG(DEBUG) << "generator is unsupported";
    return true;
  }
  /**
   * skip super call
   * >>>def super_wrapper(self):
   * ...    __class__=type(self)
   * ...    def super_init(self):
   * ...        return super()
   * ...    return super_init(self)
   * >>>assert super(int, 1).__hash__() == super_wrapper(1).__hash__()
   */
  return false;
}

bool ApplyInlinePolicy(CallNode *call_node) {
  Graph *g = call_node->GetSubGraph();
  if (g == nullptr || g->GetRetVal() == nullptr) {
    return false;
  }

  PyCodeObject *co = g->GetCodeObj();
  int ncells = PyTuple_GET_SIZE(co->co_cellvars);
  int nfrees = PyTuple_GET_SIZE(co->co_freevars);

  bool is_make_func = call_node->input(0)->GetOpcode() == MAKE_FUNCTION;
  if (is_make_func) {
    // inline MAKE_FUNCTION, need eliminate cell and free variable if the function is not dead local.
    return ncells == 0;
  }

  const auto &closures = g->GetFrame(0).GetClosures();
  if (std::any_of(closures.begin(), closures.begin() + ncells, [](auto n) { return !n->GetCellOper().empty(); })) {
    return false;
  }
  if (nfrees > 0) {
    // if inline, guard free variable
    return nfrees == 1 && std::string("__class__") == PyUnicode_AsUTF8(PyTuple_GET_ITEM(co->co_freevars, 0));
  }

  auto jcr = getJitCompileResults(reinterpret_cast<PyObject *>(co), false);
  if (jcr != nullptr && jcr->break_count_ > 0) {
    return false;
  }

  if (g->GetRetVal()->GetOpcode() == MAKE_FUNCTION) {
    return false;
  }
  for (auto i : g->GetTracedNodes()) {
    int op = i->GetOpcode();
    if (op == STORE_GLOBAL || op == DELETE_GLOBAL) {
      return false;
    }
    // check MAKE_FUNCTION is alive, it is incorrect that inline the function of different module with MAKE_FUNCTION
    auto begin = i->getInputs().begin();
    if (Utils::IsCallOp(i->GetOpcode()) && static_cast<CallNode *>(i)->GetInlineReason() == InlineReason::kInline) {
      begin++;
    }
    if (std::any_of(begin, i->getInputs().end(), [](ValueNode *n) { return n->GetOpcode() == MAKE_FUNCTION; })) {
      return false;
    }
  }
  return true;
}

bool CheckSupportCreateInstance(CallNode *call_node) {
  /**
   * only support exactly type, sub-class not create
   */
  static const std::set<PyTypeObject *> support_create_instance_type = {
    &PyComplex_Type, &PyMap_Type,   &PyBaseObject_Type, &PyRange_Type, &PyZip_Type,    &PySlice_Type,
    &PyBool_Type,    &PyFloat_Type, &PyLong_Type,       &PyType_Type,  &PyMethod_Type,
  };

  AObject *cls_info = call_node->input(0)->GetVobj();
  PyTypeObject *tp = reinterpret_cast<PyTypeObject *>(static_cast<AbstractType *>(cls_info)->GetPyObject().ptr());
  if (tp == nullptr) {
    return false;
  }
  if (support_create_instance_type.find(tp) != support_create_instance_type.end()) {
    return true;
  }

  /**
   * maybe has sideeffect, limit create
   */
  static const std::set<PyTypeObject *> limit_create_instance_type = {
    &PyList_Type, &PyTuple_Type, &PySet_Type, &PyFrozenSet_Type, &PyDict_Type, &PyUnicode_Type, &PyEnum_Type,
  };
  if (call_node->getInputs().size() != 2) {
    return false;
  }
  ValueNode *iterable_node = call_node->input(1);
  AObject *first_param = iterable_node->GetVobj();
  if (first_param == nullptr) {
    return false;
  }

  if (first_param->GetType() == AObject::kTypeAnyValue) {
    if (iterable_node->GetOpcode() != CALL_FUNCTION || call_node->bci() - 1 != iterable_node->bci()) {
      return false;
    }
    /**
     * just process this case:
     *    z = list(zip(list(x), list(y)))
     *    z = list(enumerate(x))
     */
    PyTypeObject *iterable_type = first_param->GetTypeObject();
    if (iterable_type != &PyZip_Type && iterable_type != &PyEnum_Type) {
      return false;
    }
    // this case, zip object and enumerate object is dead variable
  }
  return limit_create_instance_type.find(tp) != limit_create_instance_type.end();
}

AObject *GraphBuilder::BuildSuperObject(PyCodeObject *co) {
  if (co->co_argcount == 0) {
    PyErr_SetString(PyExc_RuntimeError, "super(): no arguments");
    return nullptr;
  }

  Py_ssize_t i, n;
  // search self object
  PyObject *obj = SearchSelfPyObject(co).first;
  if (obj == NULL) {
    PyErr_SetString(PyExc_RuntimeError, "super(): arg[0] deleted");
    return nullptr;
  }

  if (co->co_freevars == NULL) {
    n = 0;
  } else {
    assert(PyTuple_Check(co->co_freevars));
    n = PyTuple_GET_SIZE(co->co_freevars);
  }

  PyTypeObject *type = NULL;
  for (i = 0; i < n; i++) {
    PyObject *name = PyTuple_GET_ITEM(co->co_freevars, i);
    assert(PyUnicode_Check(name));
    // check class id
    if (!strcmp("__class__", PyUnicode_AsUTF8(name))) {
      Py_ssize_t index = PyTuple_GET_SIZE(co->co_cellvars) + i;
      PyObject *cell = SetLocalPyObject(frame_.Closure(index));
      if (cell == NULL || !PyCell_Check(cell)) {
        PyErr_SetString(PyExc_RuntimeError, "super(): bad __class__ cell");
        return nullptr;
      }
      type = reinterpret_cast<PyTypeObject *>(PyCell_GET(cell));
      if (type == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "super(): empty __class__ cell");
        return nullptr;
      }
      if (!PyType_Check(type)) {
        PyErr_Format(PyExc_RuntimeError, "super(): __class__ is not a tyep (%s)", Py_TYPE(type)->tp_name);
        return nullptr;
      }
      break;
    }
  }
  if (type == NULL) {
    PyErr_SetString(PyExc_RuntimeError, "super(): __class__ cell not found");
    return nullptr;
  }

  py::object py_obj = py::reinterpret_borrow<py::object>(obj);
  py::object py_type = py::reinterpret_borrow<py::object>(reinterpret_cast<PyObject *>(type));
  py::tuple tuple_obj(2);
  tuple_obj[0] = py_type;
  tuple_obj[1] = py_obj;
  PyObject *ret = PyObject_Call(reinterpret_cast<PyObject *>(&PySuper_Type), tuple_obj.ptr(), nullptr);
  AObject *super_obj = AObject::Convert(ret);
  Py_DECREF(ret);
  return super_obj;
}

bool GraphBuilder::ClassInstantiationFold(CallNode *call_node, AObject::Type type) {
  const auto &params = call_node->getInputs();
  int call_op = call_node->GetOpcode();

  // list, tuple, dict fold
  std::vector<ValueNode *> inputs;
  int new_op;
  int new_arg;
  if (type == AObject::kTypeTuple || type == AObject::kTypeList) {
    if (params.size() > 1) {
      int arg_op = params[1]->GetOpcode();
      if (call_op == CALL_FUNCTION && (arg_op == BUILD_TUPLE || arg_op == BUILD_LIST)) {
        inputs = params[1]->getInputs();
      } else {
        return false;
      }
    }
    new_op = type == AObject::kTypeTuple ? BUILD_TUPLE : BUILD_LIST;
    new_arg = inputs.size();
  } else if (type == AObject::kTypeDict) {
    if (params.size() > 1) {
      ValueNode *map_node;
      if (call_op == CALL_FUNCTION && params[1]->GetOpcode() == BUILD_MAP) {
        map_node = params[1];
      } else if (call_op == CALL_FUNCTION_EX && params.size() > 2 && params[2]->GetOpcode() == BUILD_MAP) {
        map_node = params[2];
      } else {
        return false;
      }
      inputs = map_node->getInputs();
    }
    new_op = BUILD_MAP;
    new_arg = inputs.size() / 2;
  } else {
    return false;
  }

  Graph *sub_graph = NewGraph(nullptr, nullptr);
  AObject *res = AObject::BuildOperations(CollectObjects(inputs), new_op);
  ValueNode *new_node = sub_graph->NewValueNode(res, new_op, new_arg, inputs);
  sub_graph->GetTracedNodes().push_back(new_node);
  sub_graph->SetRetVal(new_node);

  call_node->SetSubGraph(sub_graph);
  call_node->SetInlineReason(InlineReason::kInline);
  seek(0) = new_node;
  return true;
}

void LogGuardFailed(ValueNode *node, const GraphJitConfig &conf, const std::string &msg) {
  if (!conf.GetBoolConfig(GraphJitConfig::kLogGraphBreak)) {
    return;
  }
  auto tr = GetTrace(node, false, true, 0, -1);
  std::stringstream s;
  s << "trace:\n" << (tr ? tr->FormatString().c_str() : "trace failed") << "\n";
  s << msg << " [" << node->ToString() << "]";
  GRAPH_JIT_LOG_F("%s", s.str().c_str());
}

bool GraphBuilder::HandleCallClass(CallNode *call_node) {
  AObject *vobj = call_node->input(0)->GetVobj();
  if (!vobj || vobj->GetType() != AObject::kTypeType) {
    return false;
  }
  AbstractType *t = static_cast<AbstractType *>(vobj);
  AObject::Type type = t->GetTypeType();
  if (!trace_flag() && ClassInstantiationFold(call_node, type)) {
    return true;
  }

  const auto &params = call_node->getInputs();
  AObject *instance = nullptr;
  bool support_create_instance = CheckSupportCreateInstance(call_node);
  bool constant = type == AObject::kTypePrimitive || type == AObject::kTypeTensor || type == AObject::kTypeStubTensor ||
                  IsMsClass(t->GetPyObject().ptr());
  // create instance
  if (support_create_instance || constant) {
    constant |= std::none_of(params.begin(), params.end(), [](ValueNode *i) { return !i->IsConstantValue(); });
    std::vector<py::object> args;
    std::transform(params.begin() + 1, params.end(), std::back_inserter(args), [](ValueNode *n) {
      AObject *i = n->GetVobj();
      return i ? i->GetPyObject() : py::object();
    });
    py::object res = t->BuildInstance(args, call_node->GetOpcode());
    instance = res.ptr() ? AObject::Convert(res) : nullptr;
  } else if (reinterpret_cast<PyTypeObject *>(vobj->GetPyObject().ptr()) == &PySuper_Type) {
    // take super ptr and compare with PySuper_Type
    instance = BuildSuperObject(graph_->GetCodeObj());
    this->graph_->GetTracedNodes().pop_back();
    if (PyErr_Occurred()) {
      throw py::error_already_set();
    }
  }

  if (constant && instance != nullptr && GuardConstCallNodeParam(call_node, call_node->GetGraph(), INT_MAX)) {
    // make instance is constant
    auto iter = this->graph_->GetTracedNodes().end() - 1;
    MS_EXCEPTION_IF_CHECK_FAIL(*iter == call_node, "CallNode must be last when build sub graph");
    *iter = NewValueNode(instance, LOAD_CONST, -1, {});
    seek(0) = *iter;
  } else if (!instance) {
    // create abstract instance
    instance = t->BuildAbstractInstance(CollectObjects({params.begin() + 1, params.end()}), call_node->GetOpcode());
  }
  call_node->SetVobj(instance);
  return instance != nullptr;
}

// NOTE: must be copy __code__, copy.deepcopy do nothing for code object
static py::object CopyPyFunc(const py::object &o) {
  MS_EXCEPTION_IF_CHECK_FAIL(PyFunction_Check(o.ptr()), "must be function");
  PyFunctionObject *func = reinterpret_cast<PyFunctionObject *>(o.ptr());
  PyCodeObject *code = reinterpret_cast<PyCodeObject *>(func->func_code);
  PyObject *new_name = PyUnicode_FromFormat("%s%U", kPIJitCopyFuncKey, code->co_name);
  PyCodeObject *new_code =
    PyCode_New(code->co_argcount, code->co_kwonlyargcount, code->co_nlocals, code->co_stacksize, code->co_flags,
               code->co_code, code->co_consts, code->co_names, code->co_varnames, code->co_freevars, code->co_cellvars,
               code->co_filename, code->co_name, code->co_firstlineno, code->co_lnotab);
  if (new_code == nullptr || new_name == nullptr) {
    throw py::error_already_set();
  }
  PyObject *new_func = PyFunction_NewWithQualName(reinterpret_cast<PyObject *>(new_code), func->func_globals, new_name);
  PyFunctionObject *new_ff = reinterpret_cast<PyFunctionObject *>(new_func);
  REPLACE_PY_MEMBER(new_ff->func_closure, func->func_closure);
  REPLACE_PY_MEMBER(new_ff->func_defaults, func->func_defaults);
  REPLACE_PY_MEMBER(new_ff->func_kwdefaults, func->func_kwdefaults);
  REPLACE_PY_MEMBER(new_ff->func_annotations, func->func_annotations);

  Py_DECREF(new_name);
  Py_DECREF(new_code);
  return py::reinterpret_steal<py::object>(new_func);
}

py::object GetPIJitCopiedFunc(const py::object &func) {
  PyObject *res = PyObject_GetAttrString(func.ptr(), kPIJitCopyFuncKey);
  if (res != nullptr) {
    return py::reinterpret_steal<py::object>(res);
  }
  PyErr_Clear();
  py::object copy = CopyPyFunc(func);
  PyObject_SetAttrString(func.ptr(), kPIJitCopyFuncKey, copy.ptr());
  (void)pi_jit_should_compile(copy, py::dict());
  return copy;
}

ValueNode *GetSelfFromMethod(ValueNode *method) {
  if (method->GetOpcode() != LOAD_ATTR) {
    return nullptr;
  }
  ValueNode *self = method->input(0);
  /**
   * TODO(chaiyouheng):
   * Check method is a generic attribute
   * descr = _PyType_Lookup(self->GetVobj()->GetTypeObject(), py::str(method->GetName()).ptr());
   * Check descr == nullptr || !PyFunction_Check(descr)
   */
  return self;
}

bool GraphBuilder::ReplaceCall(CallNode *call_node, const py::object &old_func) {
  if (call_node->GetOpcode() == CALL_FUNCTION_EX && call_node->input(1)->GetOpcode() != BUILD_TUPLE) {
    // dynamic length variable arguments, user-defined unpack sequence
    return false;
  }
  if (!graph_->GuardInlinedFunc(call_node)) {
    return false;
  }
  auto jcr = getJitCompileResults(old_func.ptr(), false);
  if (jcr != nullptr && jcr->stat != JitCompileResults::NEVER_COMPILE) {
    return true;
  }

  py::object new_func = GetPIJitCopiedFunc(old_func);

  auto &nodes = graph_->GetTracedNodes();
  MS_EXCEPTION_IF_CHECK_FAIL(nodes.back() == call_node, "CallNode must be last when build sub graph");

  ValueNode *self = nullptr;
  AObject::Type func_type = call_node->input(0)->GetVobj()->GetType();
  if (func_type == AObject::kTypeBoundMethod) {
    ValueNode *func_val = call_node->input(0);
    self = GetSelfFromMethod(func_val);
    if (self == nullptr) {
      ValueNode *node = NewValueNode(func_val->get_attr(GraphBuilder::ID___self__), LOAD_ATTR, -1, {func_val},
                                     GraphBuilder::ID___self__);
      node->SetGraph(call_node->GetGraph());
      nodes.insert(nodes.end() - 1, node);
      self = node;
    }
  } else if (func_type == AObject::kTypeCell || AObject::kTypeAnyValue) {
    self = call_node->input(0);
  } else if (func_type != AObject::kTypeFunction) {
    return false;
  }

  std::stringstream key;
  PyObject *func_name = reinterpret_cast<PyFunctionObject *>(new_func.ptr())->func_qualname;
  key << std::string(py::str(func_name)) << "." << new_func.ptr();

  // new func node
  ValueNode *func_node = this->NewValueNode(AObject::Convert(new_func), LOAD_CONST, -1, {});
  nodes.insert(nodes.end() - 1, func_node);

  // replace node
  call_node->getInputs()[0] = func_node;
  if (self == nullptr) {
    return true;
  }

  // append self to args
  if (call_node->GetOpcode() != CALL_FUNCTION_EX) {
    call_node->getInputs().insert(call_node->getInputs().begin() + 1, self);
    call_node->SetOparg(call_node->GetOparg() + 1);
    return true;
  }

  // append self to variable arguments
  ValueNode *args_node = call_node->input(1);
  std::vector<ValueNode *> inputs = args_node->getInputs();
  inputs.insert(inputs.begin(), self);
  AObject *args_info = AObject::BuildOperations(CollectObjects(inputs), BUILD_TUPLE);

  ValueNode *tuple = this->NewValueNode(args_info, BUILD_TUPLE, inputs.size(), inputs);
  tuple->set_bci(call_node->bci());
  tuple->SetLineNo(call_node->GetLineNo());
  nodes.insert(nodes.end() - 1, tuple);
  call_node->getInputs()[1] = tuple;
  return true;
}

namespace {
std::string GetFuncGraphName(const py::object &func, const MindGraphBuilderPtr &subgraph) {
  auto func_str = py::cast<std::string>(py::str(func));
  std::vector<std::string> vec;
  std::istringstream iss(func_str);
  std::string str;
  while (iss >> str) {
    (void)vec.emplace_back(str);
  }
  if (vec.size() <= 1) {
    return "";
  }
  auto func_name = vec[1];
  std::replace(func_name.begin(), func_name.end(), '.', '_');
  return func_name + "_" + std::to_string(subgraph->GetGraph()->GetCodeObj()->co_firstlineno);
}
}  // namespace

StopTraceReason MindGraphBuilder::BuildSubGraph(CallNode *call_node, int depth, const py::object &func,
                                                const GraphBuilderPtr &subgraph) {
  auto sg = std::dynamic_pointer_cast<MindGraphBuilder>(subgraph);
  InlineReason stat = InlineReason::kInline;
  bool is_make_func = call_node->input(0)->GetOpcode() == MAKE_FUNCTION;
  if (is_make_func) {
    // inline MAKE_FUNCTION, need eliminate cell and free variable if the function is not dead local.
    bool has_cell = PyTuple_GET_SIZE(sg->GetGraph()->GetCodeObj()->co_cellvars) != 0;
    stat = has_cell ? InlineReason::kInlinePolicyDisabled : stat;
  }

  auto code = sg->GetGraph()->GetGuard();
  MS_EXCEPTION_IF_NULL(code);
  code->GetGuard()->Backup();

  auto args = call_node->GetArgs();
  if (PyFunction_Check(func.ptr())) {
    args = GetNewArgs(call_node, AObject::Convert(func.ptr()));
  }

  MS_LOG(INFO) << "new subgraph->TraceRun:" << py::str(func);
  sg->FGAddInputs(args);
  auto reason = sg->TraceRun();
  MS_LOG(INFO) << "new subgraph->TraceRun end:" << py::str(func);

  call_node->SetSubGraph(sg->GetGraph());
  auto sub_ret = sg->GetGraph()->GetRetVal();
  if (sub_ret != nullptr) {
    if (sub_ret->GetVobj()->GetPyObject().ptr() == nullptr ||
        CheckConstPyObject(sub_ret->GetVobj()->GetPyObject().ptr())) {
      call_node->SetVobj(sub_ret->GetVobj());
    } else {
      sg->FGBuilder()->SetGraphName(GetFuncGraphName(func, sg));
      sg->FGAddOutput();
      if (sg->FGBuilder()->graph() == nullptr) {
        MS_LOG(ERROR) << "subgraph trace null";
        return StopTraceReason::kTrace_Fail;
      } else {
        TraceGuard trace_guard(GetLocation(call_node));
        auto res = FGBuilder()->AddNode(sg->FGBuilder()->graph(), args);
        if (res.ptr()) {
          MS_LOG(INFO) << "add fg node suc: ";
          call_node->SetVobj(AbstractTraceNode::MakeAObject(res));
        } else {
          MS_LOG(ERROR) << "add fg node fail";
          stat = InlineReason::kInlineInfer_Fail;
        }
      }
    }
    stat = is_make_func || ApplyInlinePolicy(call_node) ? stat : InlineReason::kInlinePolicyDisabled;
  } else {
    stat = InlineReason::kInlineInfer_Fail;
  }
  if (stat != InlineReason::kInline) {
    code->GetGuard()->Rollback();
    if (!is_make_func) {
      /**
       * replace function call, inline or resume capture after break graph
       * exclude make function, because of function always a new function but code is constant
       **/
      stat = ReplaceCall(call_node, func) ? stat : InlineReason::kInlinePolicyDisabled;
    }
  } else {
    if (!is_make_func) {
      // exclude make function, because of function always a new function but code is constant
      stat = graph_->GuardInlinedFunc(call_node) ? stat : InlineReason::kInlinePolicyDisabled;
    }
    if (stat != InlineReason::kInline) {
      code->GetGuard()->Rollback();
    } else {
      code->GetGuard()->Pop();
    }
  }

  // if stat == InlineReason::kInline, guard free variable
  call_node->SetInlineReason(stat);
  return reason;
}

// build sub-graph
StopTraceReason GraphBuilder::BuildSubGraph(CallNode *call_node, int depth, const py::object &func,
                                            const GraphBuilderPtr &subgraph) {
  InlineReason stat = InlineReason::kInline;
  bool is_make_func = call_node->input(0)->GetOpcode() == MAKE_FUNCTION;

  auto code = subgraph->GetGraph()->GetGuard();
  MS_EXCEPTION_IF_NULL(code);
  code->GetGuard()->Backup();

  MS_LOG(INFO) << "old subgraph->TraceRun";
  subgraph->TraceRun();

  call_node->SetSubGraph(subgraph->GetGraph());
  if (subgraph->GetGraph()->GetRetVal() != nullptr) {
    call_node->SetVobj(subgraph->GetGraph()->GetRetVal()->GetVobj());
  }
  bool gen_to_tuple = subgraph->GetGraph()->Config().GetBoolConfig(GraphJitConfig::kEnableGeneratorExpressionToTuple);
  if (!gen_to_tuple && subgraph->GetGraph()->GetGeneratorResult() != nullptr) {
    subgraph->GetGraph()->SetRetVal(nullptr);
  }

  stat = ApplyInlinePolicy(call_node) ? stat : InlineReason::kInlinePolicyDisabled;
  if (stat != InlineReason::kInline) {
    code->GetGuard()->Rollback();
    if (!is_make_func) {
      /**
       * replace function call, inline or resume capture after break graph
       * exclude make function, because of function always a new function but code is constant
       **/
      stat = ReplaceCall(call_node, func) ? stat : InlineReason::kInlinePolicyDisabled;
    }
  } else {
    if (!is_make_func) {
      // exclude make function, because of function always a new function but code is constant
      stat = graph_->GuardInlinedFunc(call_node) ? stat : InlineReason::kInlinePolicyDisabled;
    }
    if (stat != InlineReason::kInline) {
      code->GetGuard()->Rollback();
    } else {
      code->GetGuard()->Pop();
    }
  }

  // if stat == InlineReason::kInline, guard free variable
  call_node->SetInlineReason(stat);
  return StopTraceReason::kNonStopTrace;
}

bool GraphBuilder::UnpackDynamicLengthDictByBytecode(std::vector<ValueNode *> *params, CallNode *call_node,
                                                     ValueNode *dict_node) {
  // user defined mappings, dynamic length dictionary unpack
  if (dict_node->GetVobj()->GetType() != AObject::kTypeDict) {
    return false;
  }
  auto dict = static_cast<AbstractDict *>(dict_node->GetVobj());
  if (!dict->IsElementValid()) {
    return false;
  }
  /**
   * must be guard this dict length
   */
  py::dict py_dict = dict->GetPyObject();
  py::tuple keys(py_dict.size());
  PyObject *key;
  PyObject *value;
  Py_ssize_t pos = 0;
  Py_ssize_t cnt = 0;
  while (PyDict_Next(py_dict.ptr(), &pos, &key, &value)) {
    PyObject *py_key = key;
    MS_EXCEPTION_IF_CHECK_FAIL(PyUnicode_CheckExact(py_key), "key must be string");
    PyObject *py_value = value;
    ValueNode *index = NewValueNode(AObject::Convert(py_key), LOAD_CONST, -1, {});
    ValueNode *val = NewValueNode(AObject::Convert(py_value), BINARY_SUBSCR, 0, {dict_node, index});
    keys[cnt++] = py_key;
    params->push_back(val);
    call_node->AddParam(val);
  }
  ValueNode *const_keys = NewValueNode(AObject::Convert(keys), LOAD_CONST, -1, {});
  params->push_back(const_keys);
  return true;
}

bool GraphBuilder::UnpackCallExDict(std::vector<ValueNode *> *params, CallNode *call_node) {
  ValueNode *dict_node = params->back();
  params->clear();
  if (dict_node->GetOpcode() != BUILD_MAP) {
    return UnpackDynamicLengthDictByBytecode(params, call_node, dict_node);
  }
  if (dict_node->GetOparg() == 0) {
    return true;
  }
  py::tuple keys(dict_node->GetOparg());
  for (int i = 0; i < dict_node->GetOparg(); ++i) {
    AObject *k = dict_node->input(i * 2)->GetVobj();
    if (k->GetType() != AObject::kTypeString) {
      MS_LOG(DEBUG) << "for unpack-call, dict keys must be string";
      return false;
    }
    keys[i] = k->GetPyObject();
    params->push_back(dict_node->input((i << 1) + 1));
    MS_EXCEPTION_IF_CHECK_FAIL(keys[i].ptr(), "the keys of unpack-call must be a const string");
  }
  ValueNode *const_keys = this->NewValueNode(AObject::Convert(keys), LOAD_CONST, -1, {});
  params->push_back(const_keys);
  return true;
}

bool GraphBuilder::UnpackDynamicLengthTupleByBytecode(std::vector<ValueNode *> *params, ValueNode *args_node,
                                                      CallNode *call_node) {
  // user-defined sequence, dynamic length tuple unpack
  if (args_node->GetVobj() && args_node->GetVobj()->GetType() != AObject::kTypeTuple) {
    return false;
  }
  AbstractTuple *tuple = static_cast<AbstractTuple *>(args_node->GetVobj());
  if (!tuple->IsElementValid()) {
    return false;
  }
  /**
   * must be guard this tuple length
   */
  auto items = tuple->items();
  std::vector<ValueNode *> args;
  for (size_t i = 0; i < items.size(); i++) {
    ValueNode *idx_node = this->NewValueNode(AObject::Convert(py::int_(i)), LOAD_CONST, -1, {});
    auto value = this->NewValueNode(items[i], BINARY_SUBSCR, 0, {args_node, idx_node});
    args.push_back(value);

    call_node->AddParam(value);
  }
  params->insert(params->begin(), args.begin(), args.end());
  return true;
}

// unpack CALL_FUNCTION_EX parameters
// should do this when bytecode analyze ? replace origin opcode
bool GraphBuilder::UnpackCallExParams(std::vector<ValueNode *> *params, int extra_local, bool *has_kw,
                                      CallNode *call_node) {
  bool has_dict = params->size() > 1;
  ValueNode *args_node = params->operator[](0);
  if (!has_dict) {
    params->clear();
  } else if (!UnpackCallExDict(params, call_node)) {
    return false;
  }
  *has_kw = params->size();
  if (args_node->GetOpcode() != BUILD_TUPLE) {
    return UnpackDynamicLengthTupleByBytecode(params, args_node, call_node);
  }
  params->insert(params->begin(), args_node->getInputs().begin(), args_node->getInputs().end());
  return true;
}

bool GraphBuilder::PackKwParams(const py::object &func, std::vector<ValueNode *> *params, FrameStates *frame,
                                std::vector<ValueNode *> *kwvargs) {
  PyCodeObject *co = reinterpret_cast<PyCodeObject *>(PyFunction_GET_CODE(func.ptr()));
  AObject *keys_info = params->back()->GetVobj();
  if (params->back()->GetOpcode() != LOAD_CONST || keys_info->GetType() != AObject::kTypeTuple) {
    return false;  // other case
  }

#if (PY_MAJOR_VERSION == 3 && PY_MINOR_VERSION > 7)
  const int posonlyargcount = co->co_posonlyargcount;
#else
  const int posonlyargcount = 0;
#endif

  PyObject **vars = &PyTuple_GET_ITEM(co->co_varnames, 0);
  const int argc = co->co_argcount + co->co_kwonlyargcount;
  PyObject **kwnames = &PyTuple_GET_ITEM(keys_info->GetPyObject().ptr(), 0);
  const int k_cnt = PyTuple_GET_SIZE(keys_info->GetPyObject().ptr());
  // kwnames must be string
  MS_ASSERT(static_cast<AbstractTuple *>(keys_info)->GetElementType() == AObject::kTypeString);
  MS_EXCEPTION_IF_CHECK_FAIL(static_cast<int>(params->size()) > k_cnt, "check param");

  int kw_2_p_cnt = 0;

  // for each kw argument
  for (int i = k_cnt - 1; i >= 0; --i) {
    PyObject *key = kwnames[i];
    // find position and kwonly argument for key
    int pos = std::find_if(vars, vars + argc, [&key](PyObject *k) { return !PyUnicode_Compare(key, k); }) - vars;
    if (pos < posonlyargcount) {
      MS_LOG(DEBUG) << "position only argument specified by key-word";
      return false;
    }

    ValueNode *v = *(params->end() - 1 - k_cnt + i);
    // if key is position arg, store it
    if (pos < argc) {
      frame->SetLocal(pos, v);
      kw_2_p_cnt++;
      continue;
    }
    ValueNode *k = NewValueNode(AObject::Convert(key), LOAD_CONST, -1, {});

    kwvargs->push_back(k);
    kwvargs->push_back(v);
  }

  params->resize(params->size() - 1 - k_cnt);
  if (!(co->co_flags & CO_VARKEYWORDS)) {
    return kw_2_p_cnt == k_cnt;  // if not equal, too many key-word arguments
  }
  return true;
}

bool GraphBuilder::HandleKWParams(const py::object &func, std::vector<ValueNode *> *params, FrameStates *frame) {
  PyCodeObject *co = reinterpret_cast<PyCodeObject *>(PyFunction_GET_CODE(func.ptr()));
  std::vector<ValueNode *> kwvargs;
  if (!PackKwParams(func, params, frame, &kwvargs)) {
    // illegal arguments
    return false;
  }

  const int argc = co->co_argcount + co->co_kwonlyargcount;
  if (!(co->co_flags & CO_VARKEYWORDS)) {
    // kw_2_p_cnt == k_cnt, all kw arguments is positions arguments
    return true;
  }

  int kwvarg_loc = argc + ((co->co_flags & CO_VARARGS) ? 1 : 0);
  AObject *dict = AObject::BuildOperations(CollectObjects(kwvargs), BUILD_MAP);
  frame->SetLocal(kwvarg_loc, NewValueNode(dict, BUILD_MAP, kwvargs.size() / 2, kwvargs));

  static_cast<CallNode *>(seek(0))->AddParam(frame->Local(kwvarg_loc));
  return true;
}

bool GraphBuilder::CheckAndSetDefaultParams(const py::object &func, FrameStates *frame, int position_argc) {
  PyCodeObject *co = reinterpret_cast<PyCodeObject *>(PyFunction_GET_CODE(func.ptr()));
  PyObject *defs = PyFunction_GET_DEFAULTS(func.ptr());
  PyObject *kwdefs = PyFunction_GET_KW_DEFAULTS(func.ptr());

  const int argc = co->co_argcount + co->co_kwonlyargcount;
  PyObject *vars = co->co_varnames;

  int defs_off = defs ? co->co_argcount - PyTuple_GET_SIZE(defs) : INT_MAX;
  for (int i = position_argc; i < argc; ++i) {
    if (frame->Local(i) != &ValueNode::kUnboundLocal) {
      continue;
    }
    PyObject *val;
    if (i < co->co_argcount) {
      val = i < defs_off ? nullptr : PyTuple_GET_ITEM(defs, i - defs_off);
    } else {
      val = kwdefs == nullptr ? nullptr : PyDict_GetItem(kwdefs, PyTuple_GET_ITEM(vars, i));
    }
    if (val == nullptr) {
      MS_LOG(DEBUG) << "no " << (i < defs_off ? "" : "kw-") << "default parameter error";
      return false;
    }
    auto vo = AObject::Convert(val);
    ValueNode *c = NewValueNode(vo, LOAD_CONST, -1, {});
    frame->SetLocal(i, c);
  }
  return true;
}

ValueNode *GetBoundSelf(CallNode *call_node) {
  ValueNode *func_val = call_node->input(0);
  AObject *vo = func_val->GetVobj();
  Graph *graph = call_node->GetGraph();

  ValueNode *self = nullptr;
  switch (vo->GetType()) {
    case AObject::kTypeBoundMethod: {
      self = GetSelfFromMethod(func_val);
      if (self == nullptr) {
        AObject *tmp = func_val->get_attr(GraphBuilder::ID___self__);
        ValueNode *node = graph->NewValueNode(tmp, LOAD_ATTR, -1, {func_val}, GraphBuilder::ID___self__);
        node->SetGraph(call_node->GetGraph());
        call_node->AddParam(node);
        self = node;
      }
      break;
    }
    case AObject::kTypeCell: /* fallthrough */
    case AObject::kTypeAnyValue:
      self = func_val;
      break;
    case AObject::kTypeTraceNode:
    case AObject::kTypeFunction:
      break;
    default:
      MS_LOG(INTERNAL_EXCEPTION) << "unimplemented type " << vo->ToString();
      break;
  }
  return self;
}

bool GraphBuilder::HandlePositionParams(const py::object &func, std::vector<ValueNode *> *params, FrameStates *frame) {
  CallNode *call_node = reinterpret_cast<CallNode *>(seek(0));
  PyCodeObject *co = reinterpret_cast<PyCodeObject *>(PyFunction_GET_CODE(func.ptr()));
  auto vobj = trace_flag() ? AObject::Convert(func.ptr()) : call_node->input(0)->GetVobj();
  AObject::Type callable_type = vobj->GetType();

  ValueNode *self = GetBoundSelf(call_node);
  if (self != nullptr) {
    params->insert(params->begin(), self);
  }

  const int argc = co->co_argcount;
  const int has_varg = (co->co_flags & CO_VARARGS) ? 1 : 0;
  const int has_kwvarg = (co->co_flags & CO_VARKEYWORDS) ? 1 : 0;
  const int varg_loc = argc + co->co_kwonlyargcount;
  const int kwvarg_loc = argc + co->co_kwonlyargcount + has_varg;
  int pargc = params->size();
  if (pargc > argc && !has_varg) {
    MS_LOG(DEBUG) << "too many parameters";
    return false;
  }
  bool append_self_to_varg = has_varg && self && callable_type == AObject::kTypeBoundMethod && argc == 0;
  if (append_self_to_varg) {  // self is in variable arguments
    MS_LOG(INFO) << "not implement append self to variable arguments, inline failed";
    return false;
  }

  if (has_kwvarg && frame->Local(kwvarg_loc) == &ValueNode::kUnboundLocal) {
    auto vo = AObject::Convert(py::dict());
    auto m = NewValueNode(vo, BUILD_MAP, 0, {});
    call_node->AddParam(m);
    frame->SetLocal(kwvarg_loc, m);
  }

  if (has_varg) {
    int vargc = pargc > argc ? pargc - argc : 0;
    std::vector<ValueNode *> vargs(params->end() - vargc, params->end());
    params->resize(params->size() - vargc);

    auto vo = AObject::BuildOperations(CollectObjects(vargs), BUILD_TUPLE);
    ValueNode *build_tuple = NewValueNode(vo, BUILD_TUPLE, vargc, vargs);
    call_node->AddParam(build_tuple);
    frame->SetLocal(varg_loc, build_tuple);
  }

  pargc = params->size();
  for (int i = pargc - 1; i >= 0; --i) {
    if (frame->Local(i) != &ValueNode::kUnboundLocal) {
      MS_LOG(DEBUG) << "duplicate key-word parameter error";
      return false;
    }
    frame->SetLocal(i, params->back());
    params->pop_back();
  }
  return CheckAndSetDefaultParams(func, frame, pargc);
}

bool GraphBuilder::HandleCallParameters(const py::object &func_info, CallNode *call_node, FrameStates *frame) {
  if (func_info.ptr() == nullptr) {
    MS_LOG(EXCEPTION) << "HandleCallParameters with empty func_info input.";
  }
  PyCodeObject *co = reinterpret_cast<PyCodeObject *>(PyFunction_GET_CODE(func_info.ptr()));
  frame->ResizeLocal(co->co_nlocals);

  std::vector<ValueNode *> params(call_node->getInputs().begin() + 1, call_node->getInputs().end());
  int op = call_node->GetOpcode();
  bool has_kw = (op == CALL_FUNCTION_KW);
  if (op == CALL_FUNCTION_EX && !UnpackCallExParams(&params, co->co_nlocals, &has_kw, call_node)) {
    return false;  // ex_dict infer failed or user-defined sequence and map arguments
  }
  if (has_kw && !HandleKWParams(func_info, &params, frame)) {
    return false;
  }
  if (!HandlePositionParams(func_info, &params, frame)) {
    return false;
  }

  MS_EXCEPTION_IF_CHECK_FAIL(params.size() == 0, "check parameters handle");

  // after store all params
  // cell2arg
  const Py_ssize_t ncells = PyTuple_GET_SIZE(co->co_cellvars);
  const Py_ssize_t *c2a_arr = co->co_cell2arg;
  for (int i = 0; c2a_arr != nullptr && i < ncells; ++i) {
    if (c2a_arr[i] != CO_CELL_NOT_AN_ARG) {
      Py_ssize_t arg_index = c2a_arr[i];
      CellVarNode *cell_node = frame->Closure(i);
      ValueNode *arg_node = frame->Local(arg_index);
      /**
       * here not delete the local, continue with local same as closure
       * frame->SetLocal(arg_index, &ValueNode::kUnboundLocal);
       */

      PyObject *cell = cell_node->GetVobj()->GetPyObject().ptr();
      PyObject *cell_contents = arg_node->GetVobj() ? arg_node->GetVobj()->GetPyObject().inc_ref().ptr() : nullptr;
      MS_EXCEPTION_IF_CHECK_FAIL(cell && PyCell_Check(cell) && PyCell_GET(cell) == nullptr, "must be a empty closure");

      ValueNode *n = NewValueNode(nullptr, STORE_DEREF, i, {arg_node});

      cell_node->AddCellOper(n);
      cell_node->SetValue(arg_node);
      Py_XSETREF(PyCell_GET(cell), cell_contents);
      // cell variable is eliminate
      // call_node->AddParam(n);
    }
  }
  return true;
}

static void SetGradFuncInfo(mindspore::pijit::CallNode *call_node);

py::object GraphBuilder::ResolveCallable(CallNode *call_node, StopTraceReason *stop_reason) {
  AObject *callable = call_node->input(0)->GetVobj();
  py::object callable_info;
  *stop_reason = StopTraceReason::kStopTraceInfer_Fail;
  call_node->SetInlineReason(InlineReason::kInlineInfer_Fail);
  if (!callable) {
    return callable_info;
  }
  callable_info = callable->GetPyObject();
  if (callable_info.ptr() == nullptr) {
    callable_info = py::cast<py::object>(reinterpret_cast<PyObject *>(callable->GetTypeObject()));
  }

  AObject::Type callable_type = callable->GetType();
  if (callable_info.ptr() == nullptr) {
    if (callable->TestMsFlag(AObject::kMsFlagGradFunc | AObject::kMsFlagShardFunc | AObject::kMsFlagVmapFunc)) {
      SetGradFuncInfo(call_node);
      *stop_reason = StopTraceReason::kNonStopTrace;
    }
    return py::object();
  }

  *stop_reason = StopTraceReason::kNonStopTrace;
  if (callable_type == AObject::kTypeType) {
    call_node->SetInlineReason(InlineReason::kInlineFunc_ArgType_IsClass);
    HandleCallClass(call_node);
    if (static_cast<AbstractType *>(callable)->GetTypeType() == AObject::kTypeCell) {
      *stop_reason = StopTraceReason::kStopTraceInfer_Fail;
    }
    return py::object();
  }

  if (WhiteListFuncCheckAndInfer(call_node, callable_info)) {
    return py::object();
  }

  // find code object
  callable_info = GetFuncInfo(call_node->input(0));
  if (callable_info.ptr() == nullptr) {
    *stop_reason = StopTraceReason::kStopTraceFunc_Type_Unsupported;
    call_node->SetInlineReason(InlineReason::kInlineCFunction_Unsupported);
  }
  return callable_info;
}

void GraphBuilder::ResolveClosure(const py::object &func_info, ValueNode *callable_node, FrameStates *frame) {
  if (func_info.ptr() == nullptr) {
    MS_LOG(INTERNAL_EXCEPTION) << "When resolving closure, get func_info failed.";
  }
  PyCodeObject *co = reinterpret_cast<PyCodeObject *>(PyFunction_GET_CODE(func_info.ptr()));
  PyObject *closure = PyFunction_GET_CLOSURE(func_info.ptr());

  int ncells = PyTuple_GET_SIZE(co->co_cellvars);
  int nfrees = PyTuple_GET_SIZE(co->co_freevars);
  frame->ResizeClosure(ncells + nfrees);
  for (int i = 0; i < ncells; i++) {
    CellVarNode *n = graph_->allocator().NewNode<CellVarNode>(CellVarNode::CellVar);
    n->SetVobj(AObject::Convert(py::reinterpret_steal<py::object>(PyCell_New(nullptr))));
    frame->SetClosure(i, n);
  }
  // track free variable
  bool make_func = callable_node->GetOpcode() == MAKE_FUNCTION;
  for (int i = 0; i < nfrees; ++i) {
    CellVarNode *freevar = nullptr;
    if (make_func) {
      ValueNode *tuple = *(callable_node->getInputs().end() - 3);
      MS_EXCEPTION_IF_CHECK_FAIL(tuple->GetOpcode() == BUILD_TUPLE, "unknown closure source");
      freevar = reinterpret_cast<CellVarNode *>(tuple->input(i));
    } else if (closure) {
      auto v = PyTuple_GET_ITEM(closure, i);
      freevar = graph_->allocator().NewNode<CellVarNode>(CellVarNode::FreeVar);
      freevar->SetVobj(AObject::Convert(v));

      // if inline, guard free variable
      ValueNode *param = NewValueNode(AObject::Convert(PyCell_GET(v)), LOAD_DEREF, -1);
      param->SetGraph(graph_);
      freevar->SetValue(param);
    } else {
      MS_LOG(EXCEPTION) << "error no closure";
    }
    frame->SetClosure(ncells + i, freevar);
  }
}

StopTraceReason GraphBuilder::HandleCall(int depth) {
  MS_EXCEPTION_IF_CHECK_FAIL(seek(0)->GetType() == ValueNode::Call, "must be call node");
  CallNode *call_node = reinterpret_cast<CallNode *>(seek(0));
  if (depth > root_->graph_->Config().getIntConfig(GraphJitConfig::kMaxInlineDepth)) {
    call_node->SetInlineReason(InlineReason::kInlineTooDeep);
    return StopTraceReason::kNonStopTrace;
  }
  StopTraceReason stop_reason = StopTraceReason::kNonStopTrace;

  py::object callable_info = ResolveCallable(call_node, &stop_reason);
  if (callable_info.ptr() == nullptr) {
    return stop_reason;
  }
  MS_EXCEPTION_IF_CHECK_FAIL(PyFunction_Check(callable_info.ptr()), "'ResolveCallable' must be return a function");

  // unsupported check
  PyCodeObject *co = reinterpret_cast<PyCodeObject *>(PyFunction_GET_CODE(callable_info.ptr()));
  PyObject *globals = PyFunction_GET_GLOBALS(callable_info.ptr());
  auto subgraph = GraphBuilder::Creator(this->root_ ? this->root_ : this, this, co, globals, trace_flag());

  // frame build
  FrameStates *frame = &(subgraph->frame_);
  ResolveClosure(callable_info, call_node->input(0), frame);
  if (!HandleCallParameters(callable_info, call_node, frame)) {
    call_node->SetInlineReason(InlineReason::kInlineFunc_ArgHandle_Unsupported);
    return StopTraceReason::kStopTraceFunc_ArgHandle_Unsupported;
  }

  // build sub-graph
  stop_reason = BuildSubGraph(call_node, depth, callable_info, subgraph);
  CollectInlineInfo(call_node, depth);

  if (!trace_flag() && call_node->GetSubGraph() && call_node->GetInlineReason() == InlineReason::kInline) {
    MS_EXCEPTION_IF_NULL(call_node->GetSubGraph()->GetRetVal());
    seek(0) = call_node->GetSubGraph()->GetRetVal();
  }
  return stop_reason;
}

static bool GuardLoopSequence(Graph *graph, ValueNode *seq_node, Py_ssize_t seq_size) {
  // guard length
  PyObject *seq = seq_node->GetVobj()->GetPyObject().ptr();
  if (seq != nullptr && seq_size == -1) {
    seq_size = PySequence_Size(seq);
  }
  if (seq == nullptr || seq_size == -1) {
    PyErr_Clear();
    return false;
  }
  if (!graph->GuardSequenceNodeLength(seq_node, seq_size)) {
    return false;
  }
  if (!graph->GuardType(seq_node)) {
    return false;
  }
  return true;
}

bool GraphBuilder::TraceRunForIterSequence(int jump_bci) {
  // check for iter
  ValueNode *iter_node = seek(0);
  ValueNode *seq_node = iter_node->input(0);
  PyObject *seq = seq_node->GetVobj()->GetPyObject().ptr();
  if (seq == nullptr) {
    return false;  // infer failed
  }
  Py_ssize_t size = PySequence_Size(seq);
  if (size == -1) {
    PyErr_Clear();
    MS_LOG(DEBUG) << "FOR_ITER without __len__";
    return false;
  }

  int &index = iter_node->marker_;
  if (index == 0 && !GuardLoopSequence(graph_, seq_node)) {
    // loop start.
    return false;
  }

  if (index >= size) {
    pop();
    cur_bci_ = jump_bci;
    return true;
  }

  PyObject *item = PySequence_GetItem(seq, index);
  if (item == nullptr) {
    MS_LOG(ERROR) << "trace for iter got an error " << py::error_already_set().what();
    PyErr_Clear();
    return false;
  }

  py::object index_object = py::int_(index);
  ValueNode *index_node = NewValueNode(AObject::Convert(index_object), LOAD_CONST, -1, {});
  ValueNode *item_node = TupleDictGetItem(seq_node, index_node);
  if (item_node == nullptr) {
    item_node = NewValueNode(AObject::Convert(item), BINARY_SUBSCR, 0, {seq_node, index_node});
    graph_->GetTracedNodes().push_back(item_node);
  }
  Py_DECREF(item);

  index++;
  push(item_node);
  cur_bci_ = cur_bci_ + 1;
  return true;
}

static bool CheckForIterEnumerate(ValueNode *iter_node) {
  ValueNode *enumerate_node = iter_node->input(0);
  if (enumerate_node->GetOpcode() != CALL_FUNCTION || iter_node->bci() - 1 != enumerate_node->bci()) {
    // enumerate object maybe alive, shouldn't reduce it
    return false;
  }
  PyObject *enumerate = enumerate_node->GetVobj()->GetPyObject().ptr();
  if (enumerate == nullptr) {
    return false;
  }

  MS_EXCEPTION_IF_NULL(iter_node->GetGraph());

  ValueNode *iterable_node = enumerate_node->input(1);
  PyObject *iterable = iterable_node->GetVobj()->GetPyObject().ptr();
  if (iterable == nullptr || !PySequence_Check(iterable) || !GuardLoopSequence(iter_node->GetGraph(), iterable_node)) {
    // just support sequence iteration
    return false;
  }
  return true;
}

bool GraphBuilder::TraceRunForIterEnumerate(int jump_bci) {
  ValueNode *iter_node = seek(0);
  if (iter_node->marker_ == 0) {
    if (!CheckForIterEnumerate(iter_node)) {
      return false;
    }
    iter_node->marker_ = 1;
  }
  ValueNode *enumerate_node = iter_node->input(0);
  PyObject *enumerate = enumerate_node->GetVobj()->GetPyObject().ptr();
  ValueNode *iterable_node = enumerate_node->input(1);

  // reduce iterable object
  ValueNode *seq_node = iterable_node;
  PyObject *tuple = PyIter_Next(enumerate);
  if (tuple == nullptr) {
    if (PyErr_Occurred() && !PyErr_ExceptionMatches(PyExc_StopIteration)) {
      MS_LOG(ERROR) << "trace FOR_ITER got an error " << py::error_already_set().what();
      PyErr_Clear();
      return false;
    }
    PyErr_Clear();
    pop();
    cur_bci_ = jump_bci;
    return true;
  }
  PyObject *index = PyTuple_GET_ITEM(tuple, 0);
  PyObject *item = PyTuple_GET_ITEM(tuple, 1);
  ValueNode *index_node = NewValueNode(AObject::Convert(index), LOAD_CONST, -1, {});
  ValueNode *item_node = NewValueNode(AObject::Convert(item), BINARY_SUBSCR, 0, {seq_node, index_node});
  ValueNode *value_node = NewValueNode(AObject::Convert(tuple), BUILD_TUPLE, 2, {index_node, item_node});
  Py_DECREF(tuple);
  graph_->GetTracedNodes().push_back(item_node);
  graph_->GetTracedNodes().push_back(value_node);

  push(value_node);
  cur_bci_ = cur_bci_ + 1;
  return true;
}

static bool CheckForIterZip(ValueNode *iter_node) {
  ValueNode *zip_node = iter_node->input(0);
  if (zip_node->GetOpcode() != CALL_FUNCTION || iter_node->bci() - 1 != zip_node->bci()) {
    return false;
  }
  PyObject *zip = zip_node->GetVobj()->GetPyObject().ptr();
  if (zip == nullptr) {
    return false;
  }
  MS_EXCEPTION_IF_NULL(iter_node->GetGraph());
  Graph *graph = iter_node->GetGraph();

  std::vector<ValueNode *> iterable_nodes = {zip_node->getInputs().begin() + 1, zip_node->getInputs().end()};
  auto iter = std::find_if(iterable_nodes.begin(), iterable_nodes.end(), [&graph](ValueNode *iterable_node) {
    PyObject *iterable = iterable_node->GetVobj()->GetPyObject().ptr();
    return iterable == nullptr || !PySequence_Check(iterable) || !GuardLoopSequence(graph, iterable_node);
  });
  if (iter != iterable_nodes.end()) {
    return false;
  }
  return true;
}

bool GraphBuilder::TraceRunForIterZip(int jump_bci) {
  ValueNode *iter_node = seek(0);
  int *index = &iter_node->marker_;
  if ((*index) == 0) {
    if (!CheckForIterZip(iter_node)) {
      return false;
    }
  }

  ValueNode *zip_node = iter_node->input(0);
  PyObject *zip = zip_node->GetVobj()->GetPyObject().ptr();
  std::vector<ValueNode *> iterable_nodes = {zip_node->getInputs().begin() + 1, zip_node->getInputs().end()};

  // reduce iterable object
  PyObject *tuple = PyIter_Next(zip);
  py::object handle = py::reinterpret_steal<py::object>(tuple);
  if (handle.ptr() == nullptr) {
    if (PyErr_Occurred() && !PyErr_ExceptionMatches(PyExc_StopIteration)) {
      MS_LOG(ERROR) << "trace FOR_ITER got an error " << py::error_already_set().what();
      PyErr_Clear();
      return false;
    }
    PyErr_Clear();
    pop();
    cur_bci_ = jump_bci;
    return true;
  }

  std::vector<ValueNode *> inputs;
  for (size_t tuple_index = 0; tuple_index < iterable_nodes.size(); ++tuple_index) {
    PyObject *item = PyTuple_GET_ITEM(tuple, tuple_index);
    ValueNode *seq_node = iterable_nodes[tuple_index];
    ValueNode *index_node = NewValueNode(AObject::Convert(py::int_(*index)), LOAD_CONST, -1, {});
    ValueNode *item_node = NewValueNode(AObject::Convert(item), BINARY_SUBSCR, 0, {seq_node, index_node});
    inputs.push_back(item_node);
    graph_->GetTracedNodes().push_back(item_node);
  }
  ValueNode *value_node = NewValueNode(AObject::Convert(tuple), BUILD_TUPLE, inputs.size(), inputs);
  graph_->GetTracedNodes().push_back(value_node);
  push(value_node);

  (*index)++;
  cur_bci_ = cur_bci_ + 1;
  return true;
}

bool GraphBuilder::TraceRunForIter(const Instr &instr) {
  MS_EXCEPTION_IF_NULL(instr.extra_jump());

  // check for iter
  ValueNode *iter_node = seek(0);
  AObject *iterable = iter_node->getInputs().size() > 0 ? iter_node->input(0)->GetVobj() : nullptr;
  bool succ;
  if (iter_node->GetOpcode() != GET_ITER) {
    MS_LOG(DEBUG) << "FOR_ITER without GET_ITER";
    succ = false;
  } else if (iterable == nullptr) {
    succ = false;
  } else if (iterable->GetTypeObject() == &PyEnum_Type) {
    succ = TraceRunForIterEnumerate(instr.extra_jump()->bci());
  } else if (iterable->GetTypeObject() == &PyZip_Type) {
    succ = TraceRunForIterZip(instr.extra_jump()->bci());
  } else {
    succ = TraceRunForIterSequence(instr.extra_jump()->bci());
  }
  if (!succ) {
    graph_->StopTraceAt(cur_bci_, StopTraceReason::kStopTraceLoop_Unsupported);
  }
  return succ;
}

static bool IsConstantBoolValue(ValueNode *node) {
  const auto &cnst_info = node->GetConstantInfo();
  if (cnst_info == nullptr) {
    return false;
  }
  if (cnst_info->value().ptr() != nullptr) {
    return true;
  }
  PyTypeObject *tp = cnst_info->type();
  if (tp == nullptr) {
    return false;
  }
  static const std::set<PyTypeObject *> len_to_bool = {&PyTuple_Type, &PyList_Type, &PyDict_Type};
  if (len_to_bool.find(tp) != len_to_bool.end() && cnst_info->len() != -1) {
    return true;
  }
  return false;
}

bool IsSatisfyPruneLimit(int cond, Graph *graph_, ValueNode *cond_node) {
  if (cond == -1) {
    return false;
  }
  int limit_prune = graph_->Config().getIntConfig(GraphJitConfig::kMaxPruneCase);
  if (limit_prune >= 0 && limit_prune < graph_->GetPruneBranchCount()) {
    return false;
  }
  if (IsConstantBoolValue(cond_node)) {
    return true;
  }
  auto tr = graph_->TraceValueNode(cond_node);
  if (tr == nullptr) {
    return false;
  }
  PyObject *bool_value = cond_node->GetVobj()->GetPyObject().ptr();
  if (bool_value != Py_True && bool_value != Py_False) {
    bool strict = graph_->Config().GetBoolConfig(GraphJitConfig::kStrictTrace);
    auto bool_type = CreateOpTrace(reinterpret_cast<PyObject *>(&PyBool_Type), LOAD_CONST, -1, {}, "", "", strict);
    tr = CreateOpTrace(cond ? Py_True : Py_False, CALL_FUNCTION, 1, {bool_type, tr}, "", "", strict);
  } else {
    cond_node->SetConstantValue(true);
  }
  graph_->GetGuard()->GetGuard()->GuardOn(tr, GuardLevel::GId);
  return true;
}

static void LogPrunBranch(ValueNode *cond, const Instr &instr, const GraphJitConfig &conf) {
  MS_LOG(DEBUG) << "trace run prune branch failed [" << cond->ToString() << "]";
  if (conf.GetBoolConfig(GraphJitConfig::kPrintGuard)) {
    GRAPH_JIT_LOG_F("Fail to prune bytecode [%s]!\n", instr.ToString().c_str());
  } else {
    MS_LOG(DEBUG) << "Fail to prune bytecode [" << instr.ToString() << "]!\n";
  }

  if (conf.GetBoolConfig(GraphJitConfig::kLogGraphBreak)) {
    auto tr = GetTrace(cond, false, true, 0, conf.getIntConfig(GraphJitConfig::kMaxTraceDepth));
    GRAPH_JIT_LOG_F("trace:\n %s\n", tr ? tr->FormatString().c_str() : "trace failed");
    GRAPH_JIT_LOG_F("if branch prune failed, condition [%s] at [%U : %d]", cond->ToString().c_str(),
                    cond->GetGraph()->GetCodeObj()->co_filename, cond->GetLineNo());
  }
}

bool GraphBuilder::TraceRunControl(const Instr &instr) {
  MS_EXCEPTION_IF_NULL(instr.extra_jump());
  int opcode = instr.op();
  switch (opcode) {
    case JUMP_FORWARD:
    case JUMP_ABSOLUTE:
      cur_bci_ = instr.extra_jump()->bci();
      return true;
    case FOR_ITER:
      if (!TraceRunForIter(instr)) {
        return false;
      }
      return true;
    case JUMP_IF_NOT_EXC_MATCH:
    case SETUP_WITH:
    case SETUP_FINALLY:
#if (PY_MAJOR_VERSION == 3) && (PY_MINOR_VERSION == 7)
    case CONTINUE_LOOP:
    case SETUP_LOOP:
    case SETUP_EXCEPT:
#endif
      graph_->StopTraceAt(cur_bci_, StopTraceReason::kStopTraceByteCode_Unsupported);
      return false;
    default:
      break;
  }
  ValueNode *top = seek(0);
  int cond = CondIsTrue(top);
  if (!IsSatisfyPruneLimit(cond, graph_, top)) {
    LogPrunBranch(top, instr, graph_->Config());
    graph_->StopTraceAt(cur_bci_, StopTraceReason::kStopTraceIf_Unsupported);
    return false;
  }
  switch (opcode) {
    case POP_JUMP_IF_FALSE:
    case POP_JUMP_IF_TRUE:
      (void)pop();
      cur_bci_ = ((cond == 1) ^ (opcode == POP_JUMP_IF_TRUE)) ? cur_bci_ + 1 : instr.extra_jump()->bci();
      return true;
    case JUMP_IF_FALSE_OR_POP:
    case JUMP_IF_TRUE_OR_POP:
      if ((cond == 1) ^ (opcode == JUMP_IF_TRUE_OR_POP)) {
        (void)pop();
        cur_bci_ = cur_bci_ + 1;
      } else {
        cur_bci_ = instr.extra_jump()->bci();
      }
      return true;
    default:
      break;
  }
  MS_LOG(INTERNAL_EXCEPTION) << "shouldn't reach here";
  return false;
}

static void EliminateCellAccess(Graph *g) {
  PyCodeObject *co = g->GetCodeObj();
  int ncells = PyTuple_GET_SIZE(co->co_cellvars);
  if (ncells == 0) {
    return;
  }
  ValueNode *ret_node = g->GetRetVal();
  if (ret_node == nullptr) {
    return;
  }
  std::set<ValueNode *> escaped;
  auto CollectClosure = [&escaped](ValueNode *node) {
    if (node->GetOpcode() == MAKE_FUNCTION && (node->GetOparg() & 0x08)) {
      const auto &in = (*(node->getInputs().end() - 3))->getInputs();
      escaped.insert(in.begin(), in.end());
    }
  };
  for (auto i : g->GetTracedNodes()) {
    int op = i->GetOpcode();
    if (op == STORE_DEREF && i->GetOparg() < ncells) {
      // exclude STORE_DEREF
      continue;
    }
    auto begin = i->getInputs().begin();
    if (Utils::IsCallOp(op) && static_cast<CallNode *>(i)->GetInlineReason() == InlineReason::kInline) {
      begin++;
    }
    std::for_each(begin, i->getInputs().end(), CollectClosure);
  }
  CollectClosure(ret_node);
  // collect STORE_DEREF with MAKE_FUNCTION ...

  const auto &closures = g->GetFrame(0).GetClosures();
  for (int i = 0; i < ncells; ++i) {
    if (escaped.find(closures[i]) != escaped.end()) {
      continue;
    }
    for (auto node : closures[i]->GetCellOper()) {
      if (node->GetOpcode() != STORE_DEREF) {
        // closure access before assign, raise UnboundLocalError
        return;
      }
      node->SetOpcode(LOAD_CONST);
      node->SetVobj(AObject::Convert(Py_None));
      node->ClearInputs();
    }
    closures[i]->GetCellOper().clear();
  }
}

StopTraceReason GraphBuilder::TraceRun() {
  current_block_ = graph_->GetCFG()->GetFirstBB();
  cur_bci_ = 0;
  const auto &instrs = graph_->GetCFG()->instr_pool();
  while (true) {
    this->graph_->SetFrame(cur_bci_, frame_);
    MS_EXCEPTION_IF_CHECK_FAIL(static_cast<size_t>(cur_bci_) < instrs.size(), "error control flow");
    MS_EXCEPTION_IF_CHECK_FAIL(instrs[cur_bci_]->bci() == cur_bci_, "check instruction bci");
    if (!DoByteCode(*instrs[cur_bci_])) {
      break;
    }
  }
  EliminateCellAccess(this->graph_);
  return graph_->GetStopTraceReason();
}

extern void AddConfigToGuard(const GraphJitConfig &c, OptGuardPtr guard);
extern void AddGuardForParam(const PyFrameObject *f, OptGuardPtr guard, bool detach);

/**
 * Generate a graph from callable, this function will actually create python frame
 */
static std::unique_ptr<GraphBuilder> GenerateRootGraph(const py::object &callable, const py::object &args,
                                                       const py::object &kwargs, const GraphJitConfig &conf) {
  PyFrameObject *frame = Utils::PrepareFrame(callable.ptr(), args.ptr(), kwargs.ptr());
  if (frame == nullptr) {
    PyErr_Clear();
    return nullptr;
  }
  auto jcr = getJitCompileResults(reinterpret_cast<PyObject *>(frame->f_code));
  *jcr->conf = conf;
  jcr->code = jcr->codehub->AddOptTarget(OptOption::CreateOptionByPoint(jcr));

  auto res = std::make_unique<GraphBuilder>(frame);

  auto code = res->GetGraph()->GetGuard();
  AddConfigToGuard(conf, code->GetGuard());
  AddGuardForParam(frame, code->GetGuard(), conf.GetBoolConfig(GraphJitConfig::kGuardDetachObject));

  Py_DECREF(frame);
  return res;
}

/**
 * build graph and infer func result
 * it used to infer mindspore function, maybe replace with mindspore func_graph to infer.
 */
AObject *InferFuncResult(const py::object &callable, const py::object &args, const py::object &kwargs,
                         const GraphJitConfig &conf, bool clear_guard) {
  auto g = GenerateRootGraph(callable, args, kwargs, conf);
  if (g == nullptr) {
    return nullptr;
  }
  g->TraceRun();
  if (clear_guard) {
    Graph *graph = g->GetGraph();
    auto jcr = getJitCompileResults(reinterpret_cast<PyObject *>(graph->GetCodeObj()));
    jcr->codehub->DelOptTarget(OptOption::CreateOptionByPoint(jcr), graph->GetGuard());
  }

  ValueNode *res = g->GetGraph()->GetRetVal();
  if (res == nullptr) {
    return nullptr;
  }
  return res->GetVobj();
}

AObject *InferFuncResult(const py::object &func, const std::vector<AObject *> &stack_args, int opcode,
                         const GraphJitConfig &conf, bool clear_guard) {
  std::vector<py::object> args;
  std::transform(stack_args.begin(), stack_args.end(), std::back_inserter(args),
                 [](AObject *i) { return i ? i->GetPyObject() : py::object(); });
  auto pair = Utils::PackCallStackArgs(args, opcode);
  if (pair.first.ptr() == nullptr) {
    return nullptr;
  }
  return InferFuncResult(func, pair.first, pair.second, conf, clear_guard);
}

AObject *InferFuncResult(const py::object &callable, const py::object &args, const py::object &kwargs,
                         const GraphJitConfig &conf) {
  return InferFuncResult(callable, args, kwargs, conf, true);
}

static bool GetGradSens(ValueNode *grad_node) {
  AObject *grad_object = grad_node->GetVobj();
  if (grad_object->GetPyObject().ptr() != nullptr) {
    return grad_object->GetAttr("sens_param")->GetPyObject().ptr() == Py_True;
  }
  bool sens_param = false;
  AObject *cls = grad_node->getInputs().size() > 0 ? grad_node->input(0)->GetVobj() : nullptr;
  if (!(Utils::IsCallOp(grad_node->GetOpcode()) && cls != nullptr && cls->GetType() == AObject::kTypeType)) {
    return sens_param;
  }
  if (grad_node->GetOpcode() == CALL_FUNCTION && grad_node->getInputs().size() > 3) {
    AObject *tmp = grad_node->input(3)->GetVobj();
    sens_param = tmp ? tmp->GetPyObject().ptr() == Py_True : false;
  } else if (grad_node->GetOpcode() == CALL_FUNCTION_KW) {
    py::object kwnames = grad_node->getInputs().back()->GetVobj()->GetPyObject();
    PyObject **arr = &PyTuple_GET_ITEM(kwnames.ptr(), 0);
    Py_ssize_t size = PyTuple_GET_SIZE(kwnames.ptr());
    PyObject **iter = std::find_if(arr, arr + size, [](PyObject *k) {
      // find sens_param key
      return !PyUnicode_CompareWithASCIIString(k, "sens_param");
    });
    AObject *tmp = iter - arr != size ? grad_node->input(iter - arr)->GetVobj() : nullptr;
    sens_param = tmp ? tmp->GetPyObject().ptr() == Py_True : false;
  }
  return sens_param;
}

static void SetGradFuncInfo(CallNode *call_node) {
  const int flag = AObject::kMsFlagGradFunc | AObject::kMsFlagShardFunc | AObject::kMsFlagVmapFunc;
  ValueNode *grad_func_node = call_node->input(0);
  if (grad_func_node->getInputs().size() < 2) {
    grad_func_node->GetVobj()->ClearMsFlag(flag);
    return;
  }
  ValueNode *grad_node = grad_func_node->input(0);
  ValueNode *deco_func_node = grad_func_node->input(1);
  AObject *grad_object = grad_node->GetVobj();
  AObject *deco_func = deco_func_node->GetVobj();
  bool sens_param = false;
  if (grad_func_node->GetVobj()->TestMsFlag(AObject::kMsFlagGradFunc) &&
      grad_object->GetType() == AObject::kTypeMetaFuncGraph) {
    sens_param = GetGradSens(grad_node);
  }

  HandleGradFuncCall(call_node, deco_func, sens_param);

  // guard forward net for grad
  if (grad_func_node->GetVobj()->TestMsFlag(flag) && !call_node->GetGraph()->GuardValueNode(deco_func_node)) {
    grad_func_node->GetVobj()->ClearMsFlag(flag);
  }
}

void GraphBuilder::DumpDFG() { GRAPH_JIT_LOG_F("%s", graph_->ToString().c_str()); }

bool GraphBuilder::IsFuncInWhiteList(const py::object &f, std::string *special_func_key, bool bInferPrimitive) {
  if (f.ptr() == nullptr) {
    return false;
  }
  *special_func_key = GetFuncName(f);
  auto FuncWhiteListMap = GetFuncWhiteListMap();
  auto iter = FuncWhiteListMap.find(*special_func_key);
  if (iter != FuncWhiteListMap.end() && iter->second.check(f)) {
    return true;
  }
  auto fuzzmatcher = GetFuncWhiteListFuzzyMatcher();
  auto tar = std::find_if(fuzzmatcher.begin(), fuzzmatcher.end(),
                          [&f](const std::pair<CheckFunc, std::string> &i) { return i.first(f); });
  if (tar != fuzzmatcher.end()) {
    *special_func_key = tar->second;
    return true;
  }
  if (bInferPrimitive && CheckPrimitive(f)) {
    *special_func_key = GetMindsporeNamePrimitive();
    return true;
  }
  return false;
}

bool GraphBuilder::HandleFuncInWhiteList(const std::string &key, CallNode *call_node) {
  const auto &infer_func = GetFuncWhiteListMap().find(key)->second.infer;
  infer_func(call_node);

  // handle white list side-effects
  ValueNode *old_node = nullptr;
  ValueNode *new_node = nullptr;
  if (infer_func == InferListAppend && call_node->GetSubGraph() != nullptr) {
    // InferListAppend check it is valid
    old_node = call_node->input(0)->input(0);
    new_node = call_node->GetSubGraph()->GetRetVal();
  }
  if (old_node != nullptr && new_node != nullptr) {
    if (!ReplaceAll(old_node, new_node)) {
      call_node->SetSubGraph(nullptr);
      return false;
    }
    // if replace success, record sideeffect...
  }
  return true;
}

bool MindGraphBuilder::IsFuncInWhiteList(const py::object &f, std::string *special_func_key) {
  if (f.ptr() == nullptr) {
    return false;
  }
  *special_func_key = GetFuncName(f);
  auto MindFuncWhiteListMap = GetFuncWhiteListMap(true);
  auto iter = MindFuncWhiteListMap.find(*special_func_key);
  if (iter != MindFuncWhiteListMap.end() && iter->second.check(f)) {
    return true;
  }
  auto fuzzmatcher = GetFuncWhiteListFuzzyMatcher(true);
  auto tar = std::find_if(fuzzmatcher.begin(), fuzzmatcher.end(),
                          [&f](const std::pair<CheckFunc, std::string> &i) { return i.first(f); });
  if (tar != fuzzmatcher.end()) {
    *special_func_key = tar->second;
    return true;
  }
  return false;
}

bool MindGraphBuilder::HandleFuncInWhiteList(const std::string &key, CallNode *n) {
  MS_LOG(INFO) << "specialize for " << key;
  return GetFuncWhiteListMap(true).find(key)->second.infer(n);
}

LocationPtr MindGraphBuilder::GetLocation(CallNode *call_node) const {
  auto file_name = py::cast<std::string>(graph_->GetCodeObj()->co_filename);
  auto line_no = call_node->GetLineNo();
  std::vector<std::string> comments;
  return std::make_shared<Location>(file_name, line_no, 0, line_no, 0, "", std::move(comments));
}

bool MindGraphBuilder::WhiteListFuncCheckAndInfer(CallNode *call_node, const py::object &callable) {
  std::string special_func_key;
  if (IsFuncInWhiteList(callable, &special_func_key)) {
    call_node->SetSubGraph(NewGraph(nullptr, nullptr));
    call_node->GetSubGraph()->SetGuard(root_->GetGraph()->GetGuard());
    bool has_sub_graph = HandleFuncInWhiteList(special_func_key, call_node);
    if (!has_sub_graph) {
      call_node->SetInlineReason(InlineReason::kInlineFuncSpecialize);
      MS_ASSERT(!call_node->GetSubGraph());  // check infer function
      return true;
    }
    call_node->SetInlineReason(InlineReason::kInline);
    ValueNode *ret_node = call_node->GetSubGraph()->GetRetVal();
    MS_EXCEPTION_IF_CHECK_FAIL(ret_node, "infer special function failed");
    seek(0) = ret_node;
    return true;
  }
  return false;
}

void MindGraphBuilder::FGAddInputs(const std::vector<py::object> &args) {
  // Add function graph inputs.
  for (size_t i = 0; i < args.size(); ++i) {
    MS_LOG(INFO) << "try add input: " << py::str(args[i]);
    FGBuilder()->AddInput(args[i]);
    MS_LOG(INFO) << "add input suc";
  }
}

void MindGraphBuilder::FGAddOutput() {
  if (auto ret = GetGraph()->GetRetVal()) {
    MS_LOG(INFO) << ret->GetVobj()->ToString();
    auto out = ret->GetVobj()->GetPyObject();
    MS_LOG(INFO) << "try add output: " << py::str(out) << " addr:" << out.ptr();
    if (FGBuilder()->AddOutput(out)) {
      MS_LOG(INFO) << "add output succuss";
    } else {
      MS_LOG(INFO) << "add output fail";
    }
  }
}

py::object MindGraphBuilder::FGAddNode(CallNode *call_node, const py::object &callable_info,
                                       const std::vector<py::object> &args, StopTraceReason *stop_reason) {
  MS_LOG(INFO) << "try add node: " << py::str(callable_info);
  TraceGuard trace_guard(GetLocation(call_node));
  auto res = FGBuilder()->AddNode(callable_info, args);
  if (res.ptr() == nullptr) {
    MS_LOG(INFO) << "add node fail";
    *stop_reason = StopTraceReason::kTrace_Fail;
  } else {
    MS_LOG(INFO) << "add node suc";
    auto node = AbstractTraceNode::MakeAObject(res);
    MS_LOG(INFO) << py::str(node->GetPyObject());
    MS_LOG(INFO) << node->ToString();
    call_node->SetVobj(node);
    *stop_reason = StopTraceReason::kNonStopTrace;
  }
  return py::object();
}

std::vector<py::object> MindGraphBuilder::GetNewArgs(CallNode *call_node, AObject *vobj) {
  std::vector<py::object> new_args;
  vobj = vobj ? vobj : call_node->input(0)->GetVobj();
  if (vobj->GetType() == AObject::kTypeCFunction) {
    MS_LOG(INFO) << "not support cfunction";
  }
  auto new_callable_info = FindPyFunc(vobj);
  FrameStates f;
  ResolveClosure(new_callable_info, call_node->input(0), &f);
  if (!HandleCallParameters(new_callable_info, call_node, &f)) {
    MS_LOG(INFO) << "HandleCallParameters error" << std::endl;
  }
  PyCodeObject *co = reinterpret_cast<PyCodeObject *>(PyFunction_GET_CODE(new_callable_info.ptr()));
  int argc = co->co_argcount + co->co_kwonlyargcount;
  argc += (co->co_flags & CO_VARARGS) ? 1 : 0;
  argc += (co->co_flags & CO_VARKEYWORDS) ? 1 : 0;
  for (auto it = f.GetLocals().begin(); it != f.GetLocals().begin() + argc; it++) {
    std::set<AObject::Type> unsupported_parameter = {
      AObject::kTypeAnyValue,  AObject::kTypeFunction,      AObject::kTypeBoundMethod,
      AObject::kTypePrimitive, AObject::kTypeMetaFuncGraph, AObject::kTypeCell,
    };
    auto vobj = (*it)->GetVobj();
    if (vobj != nullptr) {
      auto pyobj = vobj->GetPyObject();
      if (pyobj.ptr() != nullptr) {
        if (unsupported_parameter.find(AbstractObjectBase::GetPyType(pyobj.ptr())) == unsupported_parameter.end()) {
          new_args.push_back(pyobj);
        }
      }
    }
  }
  return new_args;
}

py::object MindGraphBuilder::ResolveCallable(CallNode *call_node, StopTraceReason *stop_reason) {
  AObject *callable = call_node->input(0)->GetVobj();
  py::object callable_info;
  *stop_reason = StopTraceReason::kStopTraceInfer_Fail;
  call_node->SetInlineReason(InlineReason::kInlineInfer_Fail);
  if (!callable) {
    return callable_info;
  }
  callable_info = callable->GetPyObject();
  if (callable_info.ptr() == nullptr) {
    return py::object();
  }
  MS_LOG(INFO) << "trace_flag for: " << py::str(callable_info);
  auto args = call_node->GetArgs();
  auto method = FGBuilder()->ConvertMethod(callable_info);
  if (method.ptr() != nullptr) {
    MS_LOG(INFO) << "convert method :" << py::str(callable_info) << " to " << py::str(method);
    callable_info = method;
    if (!PyFunction_Check(callable_info.ptr())) {  // prim getnewargs here, func getnewargs in subgraph
      args = GetNewArgs(call_node);
    }
  }
  auto func = FGBuilder()->ConvertFunction(callable_info);
  if (func.ptr() != nullptr) {
    MS_LOG(INFO) << "convert function:" << py::str(callable_info) << " to " << py::str(func);
    callable_info = func;
  }
  if (FGBuilder()->CheckCallable(callable_info)) {
    if (PyFunction_Check(callable_info.ptr())) {
      args = GetNewArgs(call_node);
    }
    return FGAddNode(call_node, callable_info, args, stop_reason);
  }
  if (FGBuilder()->CanConstantFoldFunc(callable_info)) {
    MS_LOG(INFO) << "CanConstantFoldFunc for: " << py::str(callable_info);
    JustCallAndSetRes(call_node);
    *stop_reason = StopTraceReason::kNonStopTrace;
    return py::object();
  }
  if (callable_info.ptr() == nullptr) {
    callable_info = py::cast<py::object>(reinterpret_cast<PyObject *>(callable->GetTypeObject()));
  }

  AObject::Type callable_type = callable->GetType();
  if (callable_info.ptr() == nullptr) {
    if (callable->TestMsFlag(AObject::kMsFlagGradFunc | AObject::kMsFlagShardFunc | AObject::kMsFlagVmapFunc)) {
      SetGradFuncInfo(call_node);
      *stop_reason = StopTraceReason::kNonStopTrace;
    }
    return py::object();
  }

  *stop_reason = StopTraceReason::kNonStopTrace;
  if (callable_type == AObject::kTypeType) {
    call_node->SetInlineReason(InlineReason::kInlineFunc_ArgType_IsClass);
    HandleCallClass(call_node);
    if (static_cast<AbstractType *>(callable)->GetTypeType() == AObject::kTypeCell) {
      *stop_reason = StopTraceReason::kStopTraceInfer_Fail;
    }
    return py::object();
  }

  if (WhiteListFuncCheckAndInfer(call_node, callable_info)) {
    return py::object();
  }

  // find code object
  auto vobj = AObject::Convert(callable_info.ptr());
  callable_info = (vobj->GetType() == AObject::kTypeCFunction) ? py::object() : FindPyFunc(vobj);
  if (callable_info.ptr() == nullptr) {
    *stop_reason = StopTraceReason::kStopTraceFunc_Type_Unsupported;
    call_node->SetInlineReason(InlineReason::kInlineCFunction_Unsupported);
  }
  return callable_info;
}

AObject *MindGraphBuilder::HandleMultiOp(const Instr &instr, const std::vector<ValueNode *> &p, bool is_compare) {
  int opcode = instr.op();
  int oparg = instr.arg();
  std::vector<py::object> input_obj;
  for (auto input : p) {
    if (input->GetVobj() == nullptr) {
      return AObject::MakeAObject(AObject::kTypeAnyValue);
    }
    (void)input_obj.emplace_back(input->GetVobj()->GetPyObject());
  }
  const auto &op_name =
    is_compare ? pijit::GraphUtils::OpCompareArgToGraphName(oparg) : pijit::GraphUtils::OpCodeToGraphName(opcode);
  MS_LOG(DEBUG) << "operation name is " << op_name;
  if (op_name == "") {
    return AObject::MakeAObject(AObject::kTypeAnyValue);
  }
  auto node = fg_builder_->AddMultiNode(op_name, input_obj);
  if (node.ptr() == nullptr) {
    return AObject::MakeAObject(AObject::kTypeAnyValue);
  }
  return AbstractTraceNode::MakeAObject(node);
}

AObject *MindGraphBuilder::HandleBuildOp(const Instr &instr, const std::vector<ValueNode *> &p) {
  auto opcode = instr.op();
  std::vector<py::object> input_obj;
  for (auto input : p) {
    if (input->GetVobj() == nullptr) {
      return AObject::MakeAObject(AObject::kTypeAnyValue);
    }
    (void)input_obj.emplace_back(input->GetVobj()->GetPyObject());
  }
  auto primitive = pijit::GraphUtils::GetPrimitive(opcode);
  if (primitive == nullptr) {
    return AObject::MakeAObject(AObject::kTypeAnyValue);
  }
  if (primitive == prim::kPrimMakeDict) {
    if (opcode == BUILD_CONST_KEY_MAP) {
      MS_LOG(DEBUG) << "BUILD_CONST_KEY_MAP case, need to pack values.";
      std::vector<py::object> value_inputs;
      (void)std::transform(input_obj.begin(), input_obj.end() - 1, std::back_inserter(value_inputs),
                           [](const py::object &obj) { return obj; });
      auto value_node = fg_builder_->AddNode(prim::kPrimMakeTuple, value_inputs);
      input_obj = {input_obj.back(), value_node};
    } else {
      MS_LOG(DEBUG) << "BUILD_KEY_MAP case, need to pack keys and values.";
      size_t input_len = input_obj.size();
      if (input_len % 2 != 0) {
        MS_LOG(INTERNAL_EXCEPTION) << "BUILD_KEY_MAP should have even input, but got: " << input_len;
      }
      std::vector<py::object> key_obj;
      std::vector<py::object> value_obj;
      for (size_t i = 0; i < input_len / 2; ++i) {
        key_obj.push_back(input_obj[2 * i]);
        value_obj.push_back(input_obj[2 * i + 1]);
      }
      auto key_node = fg_builder_->AddNode(prim::kPrimMakeTuple, key_obj);
      auto value_node = fg_builder_->AddNode(prim::kPrimMakeTuple, value_obj);
      input_obj = {key_node, value_node};
    }
  }
  if (primitive == prim::kPrimMakeSlice) {
    constexpr size_t slice_without_step_len = 2;
    if (input_obj.size() == slice_without_step_len) {
      // Handle slice without step input scene, such as 0:2. MakeSlice can only handle slice with full inputs.
      (void)input_obj.emplace_back(py::int_(1));
    }
  }
  auto node = fg_builder_->AddNode(primitive, input_obj);
  return AbstractTraceNode::MakeAObject(node);
}

bool MindGraphBuilder::DoGetItem(const Instr &instr) {
  auto r = pop();
  auto l = pop();
  auto o = HandleMultiOp(instr, {l, r}, false);
  auto v = NewValueNode(o, instr, {l, r});
  push(v);
  return true;
}

bool MindGraphBuilder::DoUnary(const Instr &instr) {
  auto o = pop();
  auto r = HandleMultiOp(instr, {o}, false);
  auto v = NewValueNode(r, instr, {o});
  push(v);
  return true;
}

bool MindGraphBuilder::DoBinary(const Instr &instr) {
  auto r = pop();
  auto l = pop();
  auto o = HandleMultiOp(instr, {l, r}, false);
  auto v = NewValueNode(o, instr, {l, r});
  push(v);
  return true;
}

bool MindGraphBuilder::DoBinaryMul(const Instr &instr) {
  auto r = pop();
  auto l = pop();
  auto o = HandleMultiOp(instr, {l, r}, false);
  auto v = NewValueNode(o, instr, {l, r});
  push(v);
  return true;
}

bool MindGraphBuilder::DoCompare(const Instr &instr) {
  auto r = pop();
  auto l = pop();
  auto o = HandleMultiOp(instr, {l, r}, true);
  auto v = NewValueNode(o, instr, {l, r});
  push(v);
  return true;
}

bool MindGraphBuilder::DoBuildOp(const Instr &instr) {
  int opcode = instr.op();
  int oparg = instr.arg();
  int tmp_arg = oparg;
  tmp_arg += opcode == BUILD_CONST_KEY_MAP;
  tmp_arg += opcode == BUILD_MAP ? tmp_arg : 0;
  std::vector<ValueNode *> p(frame_.GetStacks().end() - tmp_arg, frame_.GetStacks().end());
  auto o = HandleBuildOp(instr, p);
  popn(tmp_arg);
  auto v = NewValueNode(o, instr, p);
  push(v);
  return true;
}

bool MindGraphBuilder::DoIsOp(const Instr &instr) { return GraphBuilder::DoBinary(instr); }
}  // namespace pijit
}  // namespace mindspore
