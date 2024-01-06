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

namespace mindspore {
namespace jit {
namespace graph {
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
  {BINARY_MULTIPLY, &GraphBuilder::DoBinary},
  {BINARY_MODULO, &GraphBuilder::DoBinary},
  {BINARY_POWER, &GraphBuilder::DoBinary},
  {BINARY_ADD, &GraphBuilder::DoBinary},
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
  {INPLACE_ADD, &GraphBuilder::DoBinary},
  {INPLACE_SUBTRACT, &GraphBuilder::DoBinary},
  {INPLACE_FLOOR_DIVIDE, &GraphBuilder::DoBinary},
  {INPLACE_TRUE_DIVIDE, &GraphBuilder::DoBinary},
  {INPLACE_LSHIFT, &GraphBuilder::DoBinary},
  {INPLACE_RSHIFT, &GraphBuilder::DoBinary},
  {INPLACE_AND, &GraphBuilder::DoBinary},
  {INPLACE_XOR, &GraphBuilder::DoBinary},
  {INPLACE_OR, &GraphBuilder::DoBinary},
  {IS_OP, &GraphBuilder::DoBinary},
  {CONTAINS_OP, &GraphBuilder::DoBinary},
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
  // not implement
  {LOAD_CLASSDEREF, &GraphBuilder::NotImplementBytecode},
  {LOAD_BUILD_CLASS, &GraphBuilder::NotImplementBytecode},
  {LOAD_ASSERTION_ERROR, &GraphBuilder::NotImplementBytecode},
  {GET_YIELD_FROM_ITER, &GraphBuilder::NotImplementBytecode},
  {GET_AWAITABLE, &GraphBuilder::NotImplementBytecode},
  {GET_AITER, &GraphBuilder::NotImplementBytecode},
  {GET_ANEXT, &GraphBuilder::NotImplementBytecode},
  {YIELD_VALUE, &GraphBuilder::NotImplementBytecode},
  {YIELD_FROM, &GraphBuilder::NotImplementBytecode},
  {PRINT_EXPR, &GraphBuilder::NotImplementBytecode},
  {POP_BLOCK, &GraphBuilder::NotImplementBytecode},
  {POP_EXCEPT, &GraphBuilder::NotImplementBytecode},
  {WITH_EXCEPT_START, &GraphBuilder::NotImplementBytecode},
  {SETUP_ANNOTATIONS, &GraphBuilder::NotImplementBytecode},
  {SETUP_ASYNC_WITH, &GraphBuilder::NotImplementBytecode},
  {BEFORE_ASYNC_WITH, &GraphBuilder::NotImplementBytecode},
  {END_ASYNC_FOR, &GraphBuilder::NotImplementBytecode},
  {LOAD_NAME, &GraphBuilder::NotImplementBytecode},
  {STORE_NAME, &GraphBuilder::NotImplementBytecode},
  {DELETE_NAME, &GraphBuilder::NotImplementBytecode},
  {SETUP_WITH, &GraphBuilder::NotImplementBytecode},
  {SETUP_FINALLY, &GraphBuilder::NotImplementBytecode},
  {JUMP_IF_NOT_EXC_MATCH, &GraphBuilder::NotImplementBytecode},
  {RERAISE, &GraphBuilder::NotImplementBytecode},
  {RAISE_VARARGS, &GraphBuilder::NotImplementBytecode},

#if (PY_MAJOR_VERSION == 3) && (PY_MINOR_VERSION == 7)
  {BREAK_LOOP, &GraphBuilder::NotImplementBytecode},
  {WITH_CLEANUP_START, &GraphBuilder::NotImplementBytecode},
  {WITH_CLEANUP_FINISH, &GraphBuilder::NotImplementBytecode},
  {END_FINALLY, &GraphBuilder::NotImplementBytecode},
  {CONTINUE_LOOP, &GraphBuilder::NotImplementBytecode},
  {SETUP_LOOP, &GraphBuilder::NotImplementBytecode},
  {SETUP_EXCEPT, &GraphBuilder::NotImplementBytecode},
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

ValueNode *GraphBuilder::NewValueNode(AObject *o, int op, int arg, const std::vector<ValueNode *> &p) {
  ValueNode *v;
  if (Utils::IsCallOp(op)) {
    v = graph_->allocator().NewNode<CallNode>(op, arg, p);
  } else {
    v = graph_->allocator().NewNode<ValueNode>(o, op, arg, p);
  }
  v->SetGraph(this->graph_);
  return v;
}

ValueNode *GraphBuilder::NewValueNode(AObject *o, const Instr &i, const std::vector<ValueNode *> &p) {
  ValueNode *v = NewValueNode(o, i.op(), i.arg(), p);
  v->SetName(i.name());
  v->SetLineNo(i.line());
  v->set_bci(i.bci());
  if (i.op() == LOAD_CONST) {
    AObject *o = AObject::Convert(i.cnst());
    v->SetVobj(o);
  }
  if (o && o->GetType() == AObject::kTypeTensor) {
    current_block_->SetTrackResult(Block::kTrackHasTensor);
  }
  graph_->GetTracedNodes().push_back(v);
  return v;
}

InstrNode *GraphBuilder::NewInstrNode(int op, int arg) {
  InstrNode *v = graph_->allocator().NewNode<InstrNode>(op, arg);
  v->SetGraph(this->graph_);
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

void GraphBuilder::GenIndexItemGeneral(ValueNode *iterable, int i, int end_pos) {
  AObject *seq = iterable->GetVobj();
  if (end_pos == -1) {
    AObject *index = AObject::Convert(py::int_(i));
    AObject *item = seq->GetItem(index);
    ValueNode *index_node = this->NewValueNode(index, LOAD_CONST, -1, {});
    ValueNode *item_node = this->NewValueNode(item, BINARY_SUBSCR, 0, {iterable, index_node});
    this->graph_->GetTracedNodes().push_back(item_node);
    item_node->set_bci(this->cur_bci_);
    this->push(item_node);
    return;
  }
  MS_EXCEPTION_IF_CHECK_FAIL(end_pos >= i, "check UNPACK_EX oparg");
  std::vector<ValueNode *> p;
  for (int k = i; k < end_pos; ++k) {
    AObject *index = AObject::Convert(py::int_(k));
    AObject *item = seq->GetItem(index);
    ValueNode *index_node = this->NewValueNode(index, LOAD_CONST, -1, {});
    ValueNode *item_node = this->NewValueNode(item, BINARY_SUBSCR, 0, {iterable, index_node});
    this->graph_->GetTracedNodes().push_back(item_node);
    item_node->set_bci(this->cur_bci_);
    p.push_back(item_node);
  }
  AObject *vo = AObject::BuildOperations(CollectObjects(p), BUILD_LIST);
  ValueNode *node = this->NewValueNode(vo, BUILD_LIST, end_pos - i, p);
  this->graph_->GetTracedNodes().push_back(node);
  this->push(node);
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

  if (iterable->GetOpcode() == BUILD_LIST || iterable->GetOpcode() == BUILD_TUPLE) {
    auto gen_item = [this, &iterable](int i, int j) {
      if (j == -1) {
        this->push(iterable->input(i));
        return;
      }
      MS_EXCEPTION_IF_CHECK_FAIL(j >= i, "check UNPACK_EX oparg");
      auto in_iter = iterable->getInputs().begin();
      std::vector<ValueNode *> p(in_iter + i, in_iter + j);
      AObject *vo = AObject::BuildOperations(CollectObjects(p), BUILD_LIST);
      ValueNode *node = this->NewValueNode(vo, BUILD_LIST, j - i, p);
      this->graph_->GetTracedNodes().push_back(node);
      this->push(node);
    };
    GenUnpackValue(gen_item, cnt, cnt_after, size);
    return true;
  }

  AObject *seq = iterable->GetVobj();
  switch ((seq == nullptr) ? AObject::kTypeAnyValue : seq->GetType()) {
    case AObject::kTypeString:
    case AObject::kTypeTuple:
    case AObject::kTypeList:
    case AObject::kTypeNNCellList:
    case AObject::kTypeTensor: {
      auto tr = this->graph_->TraceValueNode(iterable);
      if (tr == nullptr) {
        return false;
      }
      this->graph_->GetGuard()->GetGuard()->GuardOn(tr, GuardLevel::GDeduce, false);
      auto gen_item = [this, iterable](int i, int j) { this->GenIndexItemGeneral(iterable, i, j); };
      GenUnpackValue(gen_item, cnt, cnt_after, size);
      return true;
    }
    case AObject::kTypeDict:
    default:
      break;
  }
  return false;
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

bool GraphBuilder::DoReturn(const Instr &instr) {
  graph_->SetRetVal(pop());
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
      setLocal(instr.arg(), &ValueNode::UnboundLocal);
      break;
    default:
      return false;
  }
  return true;
}

bool GraphBuilder::DoCellAccess(const Instr &instr) {
  bool is_make_func = false;
  if (parent_ != nullptr) {
    // check this function is produced by MAKE_FUNCTION
    AbstractNode *n = parent_->GetGraph()->GetTracedNodes().back();
    MS_EXCEPTION_IF_CHECK_FAIL(n->GetType() == ValueNode::Call, "must be call node");
    is_make_func = static_cast<ValueNode *>(n)->input(0)->GetOpcode() == MAKE_FUNCTION;
  }
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
      frame_.Closure(oparg)->SetValue(value);
      if (is_make_func) {
        // inline MAKE_FUNCTION, closure is not sideeffect
        break;
      }
      node = NewValueNode(nullptr, instr, {value});
      frame_.Closure(oparg)->AddCellOper(node);
      current_block_->SetTrackResult(Block::kHasClosureSideEffect);
      break;
    case DELETE_DEREF:
      frame_.Closure(oparg)->SetValue(&ValueNode::UnboundLocal);
      if (is_make_func) {
        // inline MAKE_FUNCTION, closure is not sideeffect
        break;
      }
      node = NewValueNode(nullptr, instr, {});
      frame_.Closure(oparg)->AddCellOper(node);
      current_block_->SetTrackResult(Block::kHasClosureSideEffect);
      break;
    default:
      return false;
  }
  return true;
}

bool GraphBuilder::DoGlobalAccess(const Instr &instr) {
  int opcode = instr.op();
  int oparg = instr.arg();
  switch (opcode) {
    case LOAD_GLOBAL: {
      auto co = graph_->GetCodeObj();
      PyObject *key = PyTuple_GET_ITEM(co->co_names, oparg);
      // NOTE: will run __get__, __hash__ function
      PyObject *obj = PyObject_GetItem(graph_->GetGlobals().ptr(), key);
      if (obj == nullptr) {
        PyErr_Clear();
        obj = PyObject_GetItem(PyEval_GetBuiltins(), key);
        if (obj == nullptr) {
          PyErr_Clear();
        }
      }
      py::object pyobj = py::reinterpret_steal<py::object>(obj);
      auto n = NewValueNode(AObject::Convert(pyobj), instr, {});
      n->SetName(PyUnicode_AsUTF8(key));
      push(n);
      break;
    }
    case STORE_GLOBAL:
    case DELETE_GLOBAL:
      current_block_->SetTrackResult(Block::kHasGlobalSideEffect);
      return false;
    default:
      return false;
  }
  return true;
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
      AObject *super = o->GetVobj();
      if (super->GetTypeObject() == &PySuper_Type) {
        ValueNode *self_super = SearchSelfPyObject(graph_->GetCodeObj()).second;
        auto &nodes = this->graph_->GetTracedNodes();
        auto mtype_obj = reinterpret_cast<PyObject *>(&PyMethod_Type);
        py::object method = super->GetPyObject().attr(instr.name().c_str());
        if (PyMethod_Check(method.ptr())) {
          PyObject *m = PyMethod_GET_FUNCTION(method.ptr());
          AObject *m_tp = AObject::Convert(mtype_obj);
          // set node name
          std::stringstream node_name;
          node_name << (m_tp->GetTypeObject()->tp_name) << "<" << m_tp->GetPyObject().ptr() << ">";

          // new method type
          ValueNode *global_node = NewValueNode(m_tp, LOAD_GLOBAL, -1, {});
          global_node->SetName(node_name.str());
          global_node->set_bci(instr.bci());
          global_node->SetLineNo(instr.line());
          graph_->InstallToGlobal(global_node->GetName(), py::reinterpret_borrow<py::object>(mtype_obj));
          nodes.push_back(global_node);

          // new method node
          ValueNode *method_node = NewValueNode(AObject::Convert(m), LOAD_GLOBAL, -1, {});
          node_name << (instr.name().c_str()) << "<" << m << ">";
          method_node->SetName(node_name.str());
          method_node->set_bci(instr.bci());
          method_node->SetLineNo(instr.line());
          graph_->InstallToGlobal(method_node->GetName(), py::reinterpret_borrow<py::object>(m));
          nodes.push_back(method_node);

          // new func node
          py::tuple tuple_obj(2);
          tuple_obj[0] = method;
          tuple_obj[1] = self_super->GetVobj()->GetPyObject();
          PyObject *ret = PyObject_Call(mtype_obj, tuple_obj.ptr(), nullptr);
          py::object mh = py::reinterpret_steal<py::object>(ret);
          PyErr_Clear();

          AObject *mh_info = AObject::Convert(mh);
          ValueNode *func_node = NewValueNode(nullptr, CALL_FUNCTION, 2, {global_node, method_node, self_super});
          func_node->SetVobj(mh_info);
          func_node->set_bci(instr.bci());
          func_node->SetLineNo(instr.line());
          nodes.push_back(func_node);
          push(func_node);
        }
      } else {
        auto attrs = o->GetAttrs();
        if (attrs.find(instr.name().c_str()) != attrs.end()) {
          push(attrs[instr.name().c_str()]);
        } else {
          auto n = NewValueNode(nullptr, instr, {o});
          n->SetOpcode(LOAD_ATTR);
          push(n);
          n->SetVobj(o->get_attr(n->GetName()));
        }
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
static ValueNode *TupleDictItemAccess(ValueNode *container, ValueNode *index) {
  PyObject *o = index->GetVobj() ? index->GetVobj()->GetPyObject().ptr() : nullptr;
  if (o == nullptr) {
    return nullptr;
  }
  if (container->GetOpcode() == BUILD_TUPLE && PyLong_Check(o)) {
    size_t i = PyLong_AsLong(o);
    return i < container->getInputs().size() ? container->input(i) : nullptr;
  }
  if (container->GetOpcode() == BUILD_MAP && PyUnicode_Check(o)) {
    std::string k = PyUnicode_AsUTF8(o);
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

void GraphBuilder::ProcessGetItem(const Instr &instr, ValueNode *l, ValueNode *r) {
  ValueNode *v = TupleDictItemAccess(l, r);
  if (v == nullptr) {
    AObject *vo_l = l->GetVobj();
    if (vo_l && (vo_l->GetType() == AObject::kTypeAnyValue || vo_l->GetType() == AObject::kTypeCell) &&
        vo_l->GetAttr("__getitem__")->GetType() == AObject::kTypeBoundMethod) {
      PyObject *po = vo_l->GetAttr("__getitem__")->GetPyObject().ptr();
      if (vo_l->GetType() == AObject::kTypeCell) {
        PyObject *res = PyObject_CallOneArg(po, r->GetVobj()->GetPyObject().ptr());
        AObject *vo = AObject::Convert(res);
        v = NewValueNode(vo, instr, {l, r});
        push(v);
        if (res) {
          Py_DECREF(res);
        }
      } else {
        if (po && PyFunction_Check(PyMethod_GET_FUNCTION(po))) {
          ValueNode *v1 = NewValueNode(vo_l->GetAttr("__getitem__"), LOAD_ATTR, 0, {l});
          v1->SetName("__getitem__");
          graph_->GetTracedNodes().push_back(v1);
          v = NewValueNode(nullptr, CALL_FUNCTION, 2, {v1, r});
          CallNode *n = static_cast<CallNode *>(v);
          v->SetName(instr.name().c_str());
          v->SetLineNo(instr.line());
          v->set_bci(instr.bci());
          graph_->GetTracedNodes().push_back(v);
          push(v);
          StopTraceReason ret = HandleCall(0);
          if (StopTraceReason::kNonStopTrace == ret) {
            seek(0) = n->GetSubGraph() ? n->GetSubGraph()->GetRetVal() : n;
          }
        } else {
          AObject *vo = l->binary_subscr(r);
          v = NewValueNode(vo, instr, {l, r});
          push(v);
        }
      }
    } else {
      AObject *vo = l->binary_subscr(r);
      v = NewValueNode(vo, instr, {l, r});
      push(v);
    }
  } else {
    push(v);
  }
}

bool GraphBuilder::DoItemAccess(const Instr &instr) {
  int opcode = instr.op();
  switch (opcode) {
    case BINARY_SUBSCR: {
      auto r = pop();
      auto l = pop();
      ProcessGetItem(instr, l, r);
      break;
    }
    case STORE_SUBSCR: {
      auto k = pop();
      auto m = pop();
      auto v = pop();
      m->store_subscr(k, v);
      NewValueNode(nullptr, instr, {v, m, k});
      current_block_->SetTrackResult(Block::kHasAttrSideEffect);
      break;
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
  auto n = NewValueNode(nullptr, instr, {});
  push(n);
  return true;
}

bool GraphBuilder::DoListToTuple(const Instr &instr) {
  ValueNode *list = pop();
  AObject *vo = list->GetVobj();
  if (vo && vo->GetType() == AObject::kTypeList) {
    vo = static_cast<AbstractList *>(vo)->ListToTuple();
  } else {
    vo = AObject::MakeAObject(AObject::kTypeAnyValue);
  }
  ValueNode *tuple = NewValueNode(vo, instr, {list});
  if (list->GetOpcode() == BUILD_LIST) {
    tuple->getInputs() = list->getInputs();
    tuple->SetOpcode(BUILD_TUPLE);
    tuple->SetOparg(list->getInputs().size());
  }
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

bool GraphBuilder::DoUnary(const Instr &instr) {
  int opcode = instr.op();
  auto o = pop();
  auto t = o->GetVobj();
  auto r = NewValueNode(t ? t->Unary(opcode) : AObject::MakeAObject(AObject::kTypeAnyValue), instr, {o});
  push(r);
  return true;
}

bool GraphBuilder::DoBinary(const Instr &instr) {
  int opcode = instr.op();
  int oparg = instr.arg();
  auto r = pop();
  auto l = pop();
  auto o = l->GetVobj() ? l->GetVobj()->Binary(r->GetVobj(), opcode) : AObject::MakeAObject(AObject::kTypeAnyValue);
  if ((opcode == CONTAINS_OP || opcode == IS_OP) && o && o->GetPyObject().ptr()) {
    bool res = (o->GetPyObject().ptr() == Py_True) ^ oparg;
    o = AObject::Convert(py::bool_(res));
  }
  auto v = NewValueNode(o, instr, {l, r});
  push(v);
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
  AObject *vo = AObject::BuildOperations(CollectObjects(p), opcode);
  popn(tmp_arg);
  auto v = NewValueNode(vo, instr, p);
  push(v);
  return true;
}

static bool ReplaceMergeOp(ValueNode *container) {
  ValueNode *origin = container->input(0);
  ValueNode *arg = container->input(1);
  if (origin->GetOpcode() != BUILD_LIST && origin->GetOpcode() != BUILD_MAP) {
    return false;
  }
  std::vector<ValueNode *> inputs = origin->getInputs();
  constexpr const int second_arg = 2;
  switch (container->GetOpcode()) {
    case LIST_APPEND:
      inputs.push_back(arg);
      container->getInputs() = inputs;
      container->SetOpcode(BUILD_LIST);
      container->SetOparg(inputs.size());
      break;
    case LIST_EXTEND:
      if (arg->GetOpcode() == LOAD_CONST) {
        if (arg->GetVobj() != nullptr && arg->GetVobj()->GetType() == AObject::kTypeTuple) {
          for (auto item : arg->GetVobj()->GetPyObject()) {
            ValueNode *v;
            v = container->GetGraph()->allocator.NewNode<ValueNode>(
              AObject::Convert(item::ptr()), LOAD_CONST, -1, std::vector<ValueNode *>());
            v->SetGraph(container->GetGraph());
            inputs.push_back(v);
          }
          container->getInputs() = inputs;
          container->SetOpcode(BUILD_LIST);
          container->SetOparg(inputs.size());
          break;
        }
      }
      if (arg->GetOpcode() != BUILD_LIST && arg->GetOpcode() != BUILD_TUPLE) {
        return false;
      }
      inputs.insert(inputs.end(), arg->getInputs().begin(), arg->getInputs().end());
      container->getInputs() = inputs;
      container->SetOpcode(BUILD_LIST);
      container->SetOparg(inputs.size());
      break;
    case DICT_MERGE:
      // NOTE: here not check duplicate key, will not exception if function call is inlined
    case DICT_UPDATE:
      if (arg->GetOpcode() != BUILD_MAP) {
        return false;
      }
      inputs.insert(inputs.end(), arg->getInputs().begin(), arg->getInputs().end());
      container->getInputs() = inputs;
      container->SetOpcode(BUILD_MAP);
      MS_EXCEPTION_IF_CHECK_FAIL((inputs.size() & 1) == 0, "error inputs");
      container->SetOparg(inputs.size() >> 1);
      break;
    case MAP_ADD:
      inputs.push_back(container->input(1));
      inputs.push_back(container->input(second_arg));
      container->getInputs() = inputs;
      container->SetOpcode(BUILD_MAP);
      MS_EXCEPTION_IF_CHECK_FAIL((inputs.size() & 1) == 0, "error inputs");
      container->SetOparg(inputs.size() >> 1);
      break;
    default:
      break;
  }
  return true;
}

bool GraphBuilder::DoMergeOp(const Instr &instr) {
  int opcode = instr.op();
  int oparg = instr.arg();

  auto &container = seek(oparg + (opcode == MAP_ADD));
  if (opcode == MAP_ADD) {
    auto v = pop();
    auto k = pop();
    AObject *vo = AObject::MergeOperations(container->GetVobj(), {k->GetVobj(), v->GetVobj()}, opcode);
    container = NewValueNode(vo, instr, {container, k, v});
  } else {
    AObject *vo = AObject::MergeOperations(container->GetVobj(), {seek(0)->GetVobj()}, opcode);
    container = NewValueNode(vo, instr, {container, pop()});
  }
  // DICT_MERGE only generated when unpack-call in python3.9, all keys must be string
  // NOTE: DICT_MERGE opcode requires that *(stack_pointer - oparg - 2) is a function if has duplicate key
  // ...
  ReplaceMergeOp(container);
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
  MS_EXCEPTION_IF_CHECK_FAIL(bytecode_meth_map_.find(instr.op()) != bytecode_meth_map_.end(),
                             "unknown opcode " + std::to_string(instr.op()));
  const auto func = bytecode_meth_map_.find(instr.op())->second;
  bool support = (this->*func)(instr);

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
      n->SetValue(&ValueNode::UnboundLocal);
    } else {
      ValueNode *param = graph_->allocator().NewNode<ValueNode>(AObject::Convert(cell_contents), LOAD_DEREF, i);
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
    MS_LOG(DEBUG) << "generator not inline";
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
  PyObject *frees = co->co_freevars;
  const Py_ssize_t size = PyTuple_GET_SIZE(frees);
  for (Py_ssize_t i = 0; i < size; ++i) {
    const char *name = PyUnicode_AsUTF8(PyTuple_GET_ITEM(frees, i));
    if (!strcmp("__class__", name)) {
      MS_LOG(INFO) << ("unimplemented super call");
      return false;
    }
  }
  return false;
}

static bool ApplyInlinePolicy(Graph *g) {
  PyCodeObject *co = g->GetCodeObj();
  int ncells = PyTuple_GET_SIZE(co->co_cellvars);
  int nfrees = PyTuple_GET_SIZE(co->co_freevars);
  if (ncells > 0) {
    return false;
  }
  if (nfrees > 0) {
    return nfrees == 1 && std::string("__class__") == PyUnicode_AsUTF8(PyTuple_GET_ITEM(co->co_freevars, 0));
  }

  for (auto &i : g->GetCFG()->bb_pool()) {
    if (i->HasUnresolvedSideEffect()) {
      return false;
    }
    if (i->IsTrackBreak()) {
      return false;
    }
  }
  return true;
}

bool CheckSupportCreateInstance(CallNode *call_node) {
  /**
   * only support exactly type, sub-class not create
   * list, tuple, set, dict, will reduce generator, not support create
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

  const auto &params = call_node->getInputs();
  bool support_create_instance = support_create_instance_type.end() != support_create_instance_type.find(tp);
  if (tp == &PyUnicode_Type && params.size() == 1 && params[0]->GetVobj() != nullptr) {
    support_create_instance = params[0]->GetVobj()->GetType() != AObject::kTypeAnyValue;
  }
  return support_create_instance;
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

bool GraphBuilder::HandleCallClass(CallNode *call_node) {
  AObject *vobj = call_node->input(0)->GetVobj();
  if (!vobj || vobj->GetType() != AObject::kTypeType) {
    return false;
  }
  AbstractType *t = static_cast<AbstractType *>(vobj);
  const auto &params = call_node->getInputs();

  AObject *instance = nullptr;
  AObject::Type type = t->GetTypeType();
  bool support_create_instance = CheckSupportCreateInstance(call_node);
  bool constant = type == AObject::kTypePrimitive || type == AObject::kTypeTensor || type == AObject::kTypeStubTensor;
  // create instance
  if (support_create_instance || constant || IsMsClass(t->GetPyObject().ptr())) {
    std::vector<py::object> args;
    std::transform(params.begin() + 1, params.end(), std::back_inserter(args), [](ValueNode *n) {
      AObject *i = n->GetVobj();
      return i && i->GetType() != AObject::kTypeAnyValue ? i->GetPyObject() : py::object();
    });
    py::object res = t->BuildInstance(args, call_node->GetOpcode());
    instance = res.ptr() ? AObject::Convert(res) : nullptr;
  }

  auto &nodes = this->graph_->GetTracedNodes();
  PyTypeObject *super_tp = reinterpret_cast<PyTypeObject *>(static_cast<AbstractType *>(vobj)->GetPyObject().ptr());
  if (super_tp == &PySuper_Type) {
    nodes.pop_back();
  }

  // make instance is global
  if (constant && instance != nullptr) {
    // guard parameters
    bool guard_success = GuardConstCallNodeParam(call_node, call_node->GetGraph(), INT_MAX);
    if (guard_success) {
      MS_EXCEPTION_IF_CHECK_FAIL(nodes.back() == call_node, "CallNode must be last when build sub graph");

      nodes.pop_back();
      ValueNode *new_node = NewValueNode(instance, LOAD_GLOBAL, -1, {});
      std::stringstream key;
      key << (instance->GetTypeObject()->tp_name) << "<" << instance->GetPyObject().ptr() << ">";
      new_node->SetName(key.str());
      new_node->SetGraph(call_node->GetGraph());
      nodes.push_back(new_node);
      seek(0) = new_node;
      this->graph_->InstallToGlobal(new_node->GetName(), instance->GetPyObject());
    }
  }

  // take super ptr and compare with PySuper_Type
  AObject *cls_info = call_node->input(0)->GetVobj();
  PyTypeObject *tp = reinterpret_cast<PyTypeObject *>(static_cast<AbstractType *>(cls_info)->GetPyObject().ptr());
  if (tp != nullptr && tp == &PySuper_Type) {
    instance = BuildSuperObject(graph_->GetCodeObj());
  }

  if (!instance) {
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

bool GuardInlinedFunc(CallNode *call_node) {
  AObject::Type func_type = call_node->input(0)->GetVobj()->GetType();

  // guard this function
  TracePtr tr = call_node->GetGraph()->TraceValueNode(call_node->input(0));
  if (tr == nullptr) {
    return false;
  }
  PyObject *callable = call_node->input(0)->GetVobj()->GetPyObject().ptr();
  bool strict = call_node->GetGraph()->Config().GetBoolConfig(GraphJitConfig::kStrictTrace);
  if (func_type == AObject::kTypeBoundMethod) {
    PyObject *func = PyMethod_GET_FUNCTION(callable);
    tr = CreateOpTrace(func, LOAD_ATTR, 0, {tr}, "", "__func__", strict);
    tr = CreateOpTrace(PyFunction_GET_CODE(func), LOAD_ATTR, 0, {tr}, "", "__code__", strict);
    call_node->GetGraph()->GetGuard()->GetGuard()->GuardOn(tr, GuardLevel::GId);
  } else if (func_type == AObject::kTypeCell || AObject::kTypeAnyValue) {
    call_node->GetGraph()->GetGuard()->GetGuard()->GuardOn(tr, GuardLevel::GType, false);
  } else if (func_type == AObject::kTypeFunction) {
    PyObject *name = reinterpret_cast<PyFunctionObject *>(callable)->func_qualname;
    if (std::string(PyUnicode_AsUTF8(name)).find(kPIJitCopyFuncKey) != std::string::npos) {
      return true;
    }
    tr = CreateOpTrace(PyFunction_GET_CODE(callable), LOAD_ATTR, 0, {tr}, "", "__code__", strict);
    call_node->GetGraph()->GetGuard()->GetGuard()->GuardOn(tr, GuardLevel::GId);
  } else {
    return false;
  }
  return true;
}

bool GraphBuilder::ReplaceCall(CallNode *call_node, const py::object &old_func) {
  if (call_node->GetOpcode() == CALL_FUNCTION_EX && call_node->input(1)->GetOpcode() != BUILD_TUPLE) {
    // dynamic length variable arguments, user-defined unpack sequence
    return false;
  }
  if (!GuardInlinedFunc(call_node)) {
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
    MS_EXCEPTION_IF_NULL(call_node->GetSubGraph());
    const auto &f = call_node->GetSubGraph()->GetFrame(0);
    self = f.Local(0);
    if (self == &ValueNode::UnboundLocal && f.GetClosures().size() > 0) {
      self = f.Closure(0)->GetValue();
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
  ValueNode *func_node = this->NewValueNode(AObject::Convert(new_func), LOAD_GLOBAL, -1, {});
  func_node->SetName(key.str().c_str());
  this->graph_->InstallToGlobal(func_node->GetName(), new_func);
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
  ValueNode *tuple = this->NewValueNode(nullptr, BUILD_TUPLE, 0, args_node->getInputs());
  tuple->getInputs().insert(tuple->getInputs().begin(), self);
  tuple->SetOparg(args_node->GetOparg() + 1);
  tuple->set_bci(call_node->bci());
  tuple->SetLineNo(call_node->GetLineNo());
  tuple->SetVobj(AObject::BuildOperations(CollectObjects(tuple->getInputs()), tuple->GetOpcode()));
  nodes.insert(nodes.end() - 1, tuple);
  call_node->getInputs()[1] = tuple;
  return true;
}

// build sub-graph
StopTraceReason GraphBuilder::BuildSubGraph(CallNode *call_node, int depth, const py::object &func,
                                            GraphBuilder *subgraph) {
  InlineReason stat = InlineReason::kInline;
  bool is_make_func = call_node->input(0)->GetOpcode() == MAKE_FUNCTION;
  if (is_make_func) {
    // inline MAKE_FUNCTION, need eliminate cell and free variable if the function is not dead local.
    bool has_cell = PyTuple_GET_SIZE(subgraph->GetGraph()->GetCodeObj()->co_cellvars) != 0;
    stat = has_cell ? InlineReason::kInlinePolicyDisabled : stat;
  }

  auto code = subgraph->GetGraph()->GetGuard();
  MS_EXCEPTION_IF_NULL(code);
  code->GetGuard()->Backup();

  subgraph->TraceRun();

  call_node->SetSubGraph(subgraph->GetGraph());
  if (subgraph->GetGraph()->GetRetVal() != nullptr) {
    call_node->SetVobj(subgraph->GetGraph()->GetRetVal()->GetVobj());
    stat = is_make_func || ApplyInlinePolicy(subgraph->GetGraph()) ? stat : InlineReason::kInlinePolicyDisabled;
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
      stat = GuardInlinedFunc(call_node) ? stat : InlineReason::kInlinePolicyDisabled;
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
    if (frame->Local(i) != &ValueNode::UnboundLocal) {
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

static ValueNode *GetBoundSelf(CallNode *call_node) {
  ValueNode *func_val = call_node->input(0);
  AObject *vo = func_val->GetVobj();
  auto &alloc = call_node->GetGraph()->allocator();

  ValueNode *self = nullptr;
  switch (vo->GetType()) {
    case AObject::kTypeBoundMethod: {
      self = GetSelfFromMethod(func_val);
      AObject *tmp = func_val->get_attr(GraphBuilder::ID___self__);
      ValueNode *node = alloc.NewNode<ValueNode>(tmp, LOAD_ATTR, -1, std::vector<ValueNode *>({func_val}));
      node->SetName(GraphBuilder::ID___self__);
      node->SetGraph(call_node->GetGraph());
      if (self == nullptr) {
        call_node->AddParam(node);
        self = node;
      }
      break;
    }
    case AObject::kTypeCell: /* fallthrough */
    case AObject::kTypeAnyValue:
      self = func_val;
      break;
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
  AObject::Type callable_type = call_node->input(0)->GetVobj()->GetType();

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

  if (has_kwvarg && frame->Local(kwvarg_loc) == &ValueNode::UnboundLocal) {
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
    if (frame->Local(i) != &ValueNode::UnboundLocal) {
      MS_LOG(DEBUG) << "duplicate key-word parameter error";
      return false;
    }
    frame->SetLocal(i, params->back());
    params->pop_back();
  }
  return CheckAndSetDefaultParams(func, frame, pargc);
}

bool GraphBuilder::HandleCallParameters(const py::object &func_info, CallNode *call_node, FrameStates *frame) {
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
       * frame->SetLocal(arg_index, &ValueNode::UnboundLocal);
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

static void SetGradFuncInfo(mindspore::jit::graph::CallNode *call_node);

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

      // if inline, guard this variable
      ValueNode *param = graph_->allocator().NewNode<ValueNode>(AObject::Convert(PyCell_GET(v)), LOAD_DEREF, -1);
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
  if (UnsupportedCodeTypeCheck(co)) {
    call_node->SetInlineReason(InlineReason::kInlineFunc_Type_Unsupported);
    return StopTraceReason::kNonStopTrace;
  }

  PyObject *globals = PyFunction_GET_GLOBALS(callable_info.ptr());
  GraphBuilder subgraph(this->root_ ? this->root_ : this, this, co, globals);

  // frame build
  FrameStates *frame = &subgraph.frame_;
  ResolveClosure(callable_info, call_node->input(0), frame);
  if (!HandleCallParameters(callable_info, call_node, frame)) {
    call_node->SetInlineReason(InlineReason::kInlineFunc_ArgHandle_Unsupported);
    return StopTraceReason::kStopTraceFunc_ArgHandle_Unsupported;
  }

  // build sub-graph
  stop_reason = BuildSubGraph(call_node, depth, callable_info, &subgraph);
  CollectInlineInfo(call_node, depth);

  if (call_node->GetSubGraph() && call_node->GetInlineReason() == InlineReason::kInline) {
    MS_EXCEPTION_IF_NULL(call_node->GetSubGraph()->GetRetVal());
    seek(0) = call_node->GetSubGraph()->GetRetVal();
  }
  return stop_reason;
}

bool GraphBuilder::TraceRunForIter(const Instr &instr) {
  MS_EXCEPTION_IF_NULL(instr.extra_jump());
  int jump_bci = instr.extra_jump()->bci();
  // check for iter
  ValueNode *iter_node = seek(0);
  if (iter_node->GetOpcode() != GET_ITER) {
    MS_LOG(DEBUG) << "FOR_ITER without GET_ITER";
    return false;
  }
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
  if (index == 0 && size != 0) {
    // loop start. just guard type
    TracePtr tr = graph_->TraceValueNode(seq_node);
    if (tr == nullptr || !graph_->GetGuard()->GetGuard()->GuardOn(tr, GuardLevel::GDeduce, false)) {
      return false;
    }
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
  ValueNode *index_node = NewValueNode(AObject::Convert(py::int_(index)), LOAD_CONST, -1, {});
  ValueNode *item_node = NewValueNode(AObject::Convert(item), BINARY_SUBSCR, 0, {seq_node, index_node});
  Py_DECREF(item);
  graph_->GetTracedNodes().push_back(item_node);

  index++;
  push(item_node);
  cur_bci_ = cur_bci_ + 1;
  return true;
}

bool IsSatisfyPruneLimit(int cond, Graph *graph_, ValueNode *cond_node) {
  if (cond == -1) {
    return false;
  }
  int limit_prune = graph_->Config().getIntConfig(GraphJitConfig::kMaxPruneCase);
  if (limit_prune >= 0 && limit_prune < graph_->GetPruneBranchCount()) {
    return false;
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
  }
  return graph_->GetGuard()->GetGuard()->GuardOn(tr, GuardLevel::GId);
}

extern TracePtr GetTrace(ValueNode *node, bool strict, bool print, int depth, int max_depth);
static void LogPrunBranch(ValueNode *cond, const Instr &instr, const GraphJitConfig &conf) {
  MS_LOG(DEBUG) << "trace run prune branch failed [" << cond->ToString() << "]";
  if (conf.GetBoolConfig(GraphJitConfig::kPrintGuard)) {
    GRAPH_JIT_LOG_F("Fail to prune bytecode [%s]!\n", instr.ToString().c_str());
  } else {
    MS_LOG(DEBUG) << "Fail to prune bytecode [" << instr.ToString() << "]!\n";
  }

  if (conf.GetBoolConfig(GraphJitConfig::kLogGraphBreak)) {
    auto tr = GetTrace(cond, false, true, 0, conf.getIntConfig(GraphJitConfig::kMaxTraceDepth));
    GRAPH_JIT_LOG_F("trace %s", tr ? tr->ToString().c_str() : "trace failed");
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
        graph_->StopTraceAt(cur_bci_, StopTraceReason::kStopTraceLoop_Unsupported);
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

StopTraceReason GraphBuilder::TraceRun() {
  current_block_ = graph_->GetCFG()->GetFirstBB();
  cur_bci_ = 0;
  const auto &instrs = graph_->GetCFG()->instr_pool();
  while (true) {
    this->graph_->SetFrame(cur_bci_, frame_);
    MS_EXCEPTION_IF_CHECK_FAIL(static_cast<size_t>(cur_bci_) < instrs.size(), "error control flow");
    MS_EXCEPTION_IF_CHECK_FAIL(instrs[cur_bci_]->bci() == cur_bci_, "check instruction bci");
    if (instrs[cur_bci_]->op() == RETURN_VALUE) {
      graph_->SetRetVal(pop());
      break;
    }
    Block *loop_head = graph_->GetCFG()->GetBlockByBci(cur_bci_);
    if (loop_head->is_loop_head()) {
      graph_->StopTraceAt(cur_bci_, StopTraceReason::kStopTraceLoop_Unsupported);
      return StopTraceReason::kStopTraceLoop_Unsupported;
    }
    if (!DoByteCode(*instrs[cur_bci_])) {
      break;
    }
  }
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

}  // namespace graph
}  // namespace jit
}  // namespace mindspore
