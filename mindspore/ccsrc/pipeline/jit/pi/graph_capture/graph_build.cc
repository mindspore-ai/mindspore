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
static constexpr const char *kPIJitCopyFuncKey = ".<pijit>.";

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
  {GET_ITER, &GraphBuilder::DoGetIter},
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
  {FOR_ITER, &GraphBuilder::DoForIter},
  {JUMP_FORWARD, &GraphBuilder::DoJumpDirect},
  {JUMP_IF_FALSE_OR_POP, &GraphBuilder::DoJumpIf},
  {JUMP_IF_TRUE_OR_POP, &GraphBuilder::DoJumpIf},
  {POP_JUMP_IF_FALSE, &GraphBuilder::DoJumpIf},
  {POP_JUMP_IF_TRUE, &GraphBuilder::DoJumpIf},
  {JUMP_ABSOLUTE, &GraphBuilder::DoJumpDirect},
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
  v->SetName(i.name().c_str());
  v->SetLineNo(i.line());
  v->set_bci(i.bci());
  if (i.op() == LOAD_CONST) {
    AObject *o = AObject::Convert(i.cnst());
    MS_EXCEPTION_IF_CHECK_FAIL(o, "check AObject::Convert");
    v->SetVobj(o);
  }
  if (o && o->GetType() == AObject::kTypeTensor) {
    current_block_->SetTrackResult(Block::kTrackHasTensor);
  }
  current_block_->AddNode(v);
  MS_ASSERT(!this->graph_->GetInstrs()[i.bci()]);
  this->graph_->SetInstr(i.bci(), v);
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

void GraphBuilder::CreateAndSetConditionNode(ValueNode *cond) { frame_.SetCondition(cond); }

ValueNode *GraphBuilder::merge(ValueNode *a, ValueNode *b, Block *tar_block) {
  if (a == b || b == nullptr) {
    return a;
  }
  if (a == nullptr) {
    return b;
  }
  AObject *t1 = a->GetVobj();
  AObject *t2 = b->GetVobj();

  AObject *vobj = nullptr;
  if (t1 != nullptr && t2 != nullptr && t1->GetType() == t2->GetType()) {
    vobj = t1;
  }
  MergeNode *merged = nullptr;
  if (a->GetType() == ValueNode::Merged && b->GetType() == ValueNode::Merged) {
    merged = reinterpret_cast<MergeNode *>(a);
    merged->Merge(reinterpret_cast<MergeNode *>(b));
    merged->SetVobj(vobj);
    return merged;
  }
  if (a->GetType() == ValueNode::Merged) {
    merged = reinterpret_cast<MergeNode *>(a);
    merged->AddInput(b);
    merged->SetVobj(vobj);
    return merged;
  }
  if (b->GetType() == ValueNode::Merged) {
    merged = reinterpret_cast<MergeNode *>(b);
    merged->AddInput(a);
    merged->SetVobj(vobj);
    return merged;
  }
  merged = graph_->allocator().NewNode<MergeNode>();
  merged->SetBlock(tar_block);
  merged->AddInput(a);
  merged->AddInput(b);
  merged->SetVobj(vobj);
  return merged;
}

void GraphBuilder::MergeFrameState(Block *tar_block) {
  int target_bci = tar_block->begin_ci();
  if (!graph_->FindFrame(target_bci)) {
    graph_->SetFrame(target_bci, frame_);
    return;
  }
  FrameStates tar = graph_->GetFrame(target_bci);
  for (int i = tar.GetLocals().size() - 1; i >= 0; --i) {
    ValueNode *n = merge(frame_.Local(i), tar.Local(i), tar_block);
    tar.SetLocal(i, n);
  }
  for (int i = tar.GetClosures().size() - 1; i >= 0; --i) {
    ValueNode *n = merge(frame_.Closure(i)->GetValue(), tar.Closure(i)->GetValue(), tar_block);
    tar.Closure(i)->SetValue(n);
  }
  int sp = std::min(frame_.GetStacks().size(), tar.GetStacks().size());
  for (int i = 0; i < sp; ++i) {
    ValueNode *n = merge(frame_.Peek(i), tar.Peek(i), tar_block);
    tar.Peek(i) = n;
  }
  graph_->SetFrame(target_bci, tar);
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

bool isSatisfyPruneLimit(GraphJitConfig conf, int ret, Graph *graph_, ValueNode *cond) {
  int limit_prune = conf.getIntConfig(GraphJitConfig::kMaxPruneCase);
  return (limit_prune < 0 || limit_prune > graph_->GetPruneBranchCount()) && ret != -1 && graph_->GuardValueNode(cond);
}

extern TracePtr GetTrace(ValueNode *node, bool strict, bool print, int depth, int max_depth);
bool GraphBuilder::PruneBranch(const Instr &instr) {
  int opcode = instr.op();
  int oparg = instr.arg();
  const auto &conf = root_->graph_->Config();
  ValueNode *cond = frame_.GetCondition();
  int ret = CondIsTrue(cond);

  if (isSatisfyPruneLimit(conf, ret, graph_, cond)) {
    graph_->SetPruneBranchCount(graph_->GetPruneBranchCount() + 1);
    if (ret == 0 || ret == 1) {
      bool fall_through = ((ret == 0) ^ (opcode == POP_JUMP_IF_FALSE || opcode == JUMP_IF_FALSE_OR_POP));
      if (opcode == POP_JUMP_IF_FALSE || opcode == POP_JUMP_IF_TRUE) {
        graph_->SetInstr(cur_bci_, NewInstrNode(POP_TOP, 0));
      } else {  // opcode == JUMP_IF_FALSE_OR_POP || opcode == JUMP_IF_TRUE_OR_POP
        if (fall_through) {
          graph_->SetInstr(cur_bci_, NewInstrNode(POP_TOP, 0));
          pop();
        } else {
          graph_->SetInstr(cur_bci_, NewInstrNode(NOP, 0));
        }
      }

      // set next block
      Block *target_block = fall_through ? current_block_->GetFallBB() : current_block_->GetJumpBB();
      Block *prune_bb = fall_through ? current_block_->GetJumpBB() : current_block_->GetFallBB();
      // avoid being remove again after loop unrolling
      if (prune_bb != target_block && prune_bb != nullptr) {
        current_block_->RemoveEdge(prune_bb);
        graph_->GetCFG()->MarkDeadBB();
      }
      ResetBci(target_block->begin_ci() - 1);
      // set frame
      MS_ASSERT(!graph_->FindFrame(target_block->begin_ci()));
      graph_->SetFrame(target_block->begin_ci(), frame_);
      if (conf.GetBoolConfig(GraphJitConfig::kPrintGuard)) {
        GRAPH_JIT_LOG_F("Prune bytecode %d args %d in %s branch!\n", opcode, oparg, fall_through ? "if" : "else");
      } else {
        MS_LOG(DEBUG) << "Prune bytecode " << opcode << " args " << oparg << " in " << (fall_through ? "if" : "else")
                      << " branch!";
      }
      return true;
    }
  }
  PyErr_Clear();
  if (conf.GetBoolConfig(GraphJitConfig::kPrintGuard)) {
    GRAPH_JIT_LOG_F("Fail to prune bytecode %d args %d!\n", opcode, oparg);
  } else {
    MS_LOG(DEBUG) << "Fail to prune bytecode " << opcode << " args " << oparg << "!\n";
  }

  if (conf.GetBoolConfig(GraphJitConfig::kLogGraphBreak)) {
    auto tr = GetTrace(cond, false, true, 0, -1);
    GRAPH_JIT_LOG_F("trace %s", tr ? tr->ToString().c_str() : "trace failed");
    GRAPH_JIT_LOG_F("if branch prune failed, condition [%s] at [%U : %d]", cond->to_str().c_str(),
                    cond->GetGraph()->GetCodeObj()->co_filename, cond->GetLineNo());
  }
  return false;
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
    this->current_block_->AddNode(item_node);
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
    this->current_block_->AddNode(item_node);
    item_node->set_bci(this->cur_bci_);
    p.push_back(item_node);
  }
  AObject *vo = AObject::BuildOperations(CollectObjects(p), BUILD_LIST);
  ValueNode *node = this->NewValueNode(vo, BUILD_LIST, end_pos - i, p);
  this->current_block_->AddNode(node);
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
      this->current_block_->AddNode(node);
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

bool GraphBuilder::DoForIter(const Instr &instr) {
  ValueNode *iter = pop();
  ValueNode *element = NewValueNode(AObject::MakeAObject(AObject::kTypeAnyValue), instr, {iter});
  if (root_->graph_->Config().GetBoolConfig(GraphJitConfig::kPruneCase)) {
    return false;
  }
  CreateAndSetConditionNode(iter);
  push(iter);
  push(element);
  frame_.SetConditionIsTrue(true);
  MergeFrameState(current_block_->GetFallBB());
  pop();
  pop();
  frame_.SetConditionIsTrue(false);
  MergeFrameState(current_block_->GetJumpBB());
  return true;
}

bool GraphBuilder::DoJumpIf(const Instr &instr) {
  int opcode = instr.op();
  switch (opcode) {
    case JUMP_IF_FALSE_OR_POP: /* fall-through */
    case JUMP_IF_TRUE_OR_POP: {
      CreateAndSetConditionNode(seek(0));
      if (root_->graph_->Config().GetBoolConfig(GraphJitConfig::kPruneCase)) {
        if (!current_block_->is_loop_head()) {
          return PruneBranch(instr);
        }
        return false;
      }
      frame_.SetConditionIsTrue(opcode == JUMP_IF_TRUE_OR_POP);
      MergeFrameState(current_block_->GetJumpBB());
      pop();
      frame_.SetConditionIsTrue(opcode != JUMP_IF_TRUE_OR_POP);
      MergeFrameState(current_block_->GetFallBB());
      break;
    }
    case POP_JUMP_IF_FALSE: /* fall-through */
    case POP_JUMP_IF_TRUE: {
      CreateAndSetConditionNode(pop());
      if (root_->graph_->Config().GetBoolConfig(GraphJitConfig::kPruneCase)) {
        if (!current_block_->is_loop_head()) {
          return PruneBranch(instr);
        }
        return false;
      }
      frame_.SetConditionIsTrue(opcode == POP_JUMP_IF_TRUE);
      MergeFrameState(current_block_->GetJumpBB());
      frame_.SetConditionIsTrue(opcode != POP_JUMP_IF_TRUE);
      MergeFrameState(current_block_->GetFallBB());
      break;
    }
    default:
      return false;
  }
  return true;
}

bool GraphBuilder::DoJumpDirect(const Instr &instr) {
  if (root_->graph_->Config().GetBoolConfig(GraphJitConfig::kPruneCase)) {
    MS_ASSERT(!graph_->FindFrame(current_block_->GetJumpBB()->begin_ci()));
    ResetBci(current_block_->GetJumpBB()->begin_ci() - 1);
  }
  MergeFrameState(current_block_->GetJumpBB());
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
  ValueNode *call_node = seek(0);
  call_node->SetVobj(AObject::MakeAObject(AObject::kTypeAnyValue));
  this->graph_->SetInstr(instr.bci(), call_node);

  // replace CALL_FUNCTION_EX node with new opcode, guard function and parameters (tuple size and dict size)
  // UnpackCallExParams
  // append new operation nodes of parameters unpack to current block
  // this->current_block_->AddNode(i)
  // guard dynamic length tuple arguments

  call_node->SetName(instr.name().c_str());
  call_node->SetLineNo(instr.line());
  call_node->set_bci(instr.bci());
  this->current_block_->AddNode(call_node);
  seek(0) = call_node;
  return true;
}

bool GraphBuilder::DoNop(const Instr &instr) { return true; }
bool GraphBuilder::NotImplementBytecode(const Instr &instr) { return false; }

bool GraphBuilder::DoReturn(const Instr &instr) {
  ValueNode *r = pop();
  if (root_->graph_->Config().GetBoolConfig(GraphJitConfig::kPruneCase)) {
    ResetBci(graph_->GetCFG()->instr_pool().size() - 1);
  }

  if (frame_.GetCondition() == nullptr) {
    graph_->SetRetVal(r);
  } else {
    graph_->SetRetVal(merge(graph_->GetRetVal(), r, current_block_));
  }
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
      value = frame_.Closure(oparg)->GetValue();
      if (value != nullptr) {
        push(value);
        break;
      }
      value = NewValueNode(AObject::Convert(PyCell_GET(cell)), instr, {});
      push(value);
      // first LOAD_DEREF
      frame_.Closure(oparg)->SetValue(value);
      frame_.Closure(oparg)->AddCellOper(value);
      break;
    case STORE_DEREF:
      value = pop();
      node = NewValueNode(nullptr, instr, {value});
      frame_.Closure(oparg)->SetValue(value);
      frame_.Closure(oparg)->AddCellOper(node);
      current_block_->SetTrackResult(Block::kHasClosureSideEffect);
      break;
    case DELETE_DEREF:
      node = NewValueNode(nullptr, instr, {});
      frame_.Closure(oparg)->SetValue(&ValueNode::UnboundLocal);
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
        value = frame_.Closure(i);
        PyObject *cell = SetLocalPyObject(frame_.Closure(i));
        assert(PyCell_Check(cell));
        obj = PyCell_GET(cell);
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
      ValueNode *self_super = SearchSelfPyObject(graph_->GetCodeObj()).second;
      if (super->GetTypeObject() == &PySuper_Type) {
        auto &nodes = current_block_->GetNodes();
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
          graph_->InstallToGlobal(global_node->GetName(), py::reinterpret_borrow<py::object>(mtype_obj));
          nodes.push_back(global_node);

          // new method node
          ValueNode *method_node = NewValueNode(AObject::Convert(m), LOAD_GLOBAL, -1, {});
          node_name << (instr.name().c_str()) << "<" << m << ">";
          method_node->SetName(node_name.str());
          graph_->InstallToGlobal(method_node->GetName(), py::reinterpret_borrow<py::object>(m));
          nodes.push_back(method_node);

          // new func node
          py::tuple tuple_obj(2);
          tuple_obj[0] = method;
          tuple_obj[1] = self_super->GetVobj()->GetPyObject();
          PyObject *ret = PyObject_Call(mtype_obj, tuple_obj.ptr(), nullptr);
          ValueNode *func_node =
            NewValueNode(AObject::Convert(ret), CALL_FUNCTION, 2, {global_node, method_node, self_super});
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

bool GraphBuilder::DoItemAccess(const Instr &instr) {
  int opcode = instr.op();
  switch (opcode) {
    case BINARY_SUBSCR: {
      auto r = pop();
      auto l = pop();
      ValueNode *v = TupleDictItemAccess(l, r);
      if (v == nullptr) {
        v = NewValueNode(l->binary_subscr(r), instr, {l, r});
      }
      push(v);
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
    this->graph_->SetInstr(instr.bci(), nullptr);
  }
  push(tuple);
  return true;
}

bool GraphBuilder::DoGetIter(const Instr &instr) {
  auto obj = pop();
  auto o = obj->GetVobj();
  auto iter = NewValueNode(o ? o->GetIter() : AObject::MakeAObject(AObject::kTypeAnyValue), instr, {obj});
  push(iter);
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
  if (ReplaceMergeOp(container)) {
    this->graph_->SetInstr(instr.bci(), nullptr);
  }
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
  return support;
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
    AbstractNode::Type t = i < ncells ? AbstractNode::CellVar : AbstractNode::FreeVar;
    CellVarNode *n = graph_->allocator().NewNode<CellVarNode>(t);
    n->SetVobj(AObject::Convert(cell));
    n->SetIndex(i);
    n->SetGraph(graph_);
    frame_.SetClosure(i, n);
    if (i < ncells && co->co_cell2arg != nullptr && co->co_cell2arg[i] != CO_CELL_NOT_AN_ARG) {
      MS_EXCEPTION_IF_CHECK_FAIL(PyCell_GET(cell), "check frame");
      n->SetFromParam(co->co_cell2arg[i]);
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

static void FillInstrNode(Graph *g) {
  const auto &nodes = g->GetInstrs();
  const FrameStates &f = g->GetFrame(0);
  for (auto bb : *g->GetCFG()) {
    if (bb->is_dead()) {
      continue;
    }
    for (auto &instr : bb->instrs()) {
      if (nodes[instr.bci()] || instr.op() == EXTENDED_ARG || instr.op() == NOP) {
        continue;
      }
      InstrNode *n;
      if (instr.op() == LOAD_CONST) {
        n = g->allocator().NewValueNode(AObject::Convert(instr.cnst()), instr.op(), instr.arg(), {});
      } else {
        n = g->allocator().NewNode<InstrNode>(instr.op(), instr.arg());
      }
      if (Utils::IsCellAccessOp(instr.op())) {
        f.Closure(instr.arg())->AddCellOper(n);
      }
      if (instr.op() == LOAD_METHOD) {
        n->SetOpcode(LOAD_ATTR);
      }
      if (instr.op() == CALL_METHOD) {
        n->SetOpcode(CALL_FUNCTION);
      }
      g->SetInstr(instr.bci(), n);
      n->set_bci(instr.bci());
      n->SetName(instr.name().c_str());
      n->SetLineNo(instr.line());
      n->SetGraph(g);
    }
  }
  for (auto b : *g->GetCFG()) {
    InstrNode *n = nodes[b->instrs().back().bci()];
    if (n == nullptr || Utils::GetBranchDestIndex(n->GetOpcode(), n->GetOparg(), n->bci()) == -1) {
      continue;
    }
    if (b->GetJumpBB() == nullptr) {
      MS_EXCEPTION_IF_CHECK_FAIL(n->GetOpcode() == POP_JUMP_IF_FALSE || n->GetOpcode() == POP_JUMP_IF_TRUE,
                                 "check prune branch and loop unrolling");
      n->SetOpcode(POP_TOP);
      n->SetOparg(0);
      continue;
    }

    int i = b->GetJumpBB()->begin_ci();
    while (!nodes[i]) {
      ++i;
    }
    n->SetJump(nodes[i]);
  }
}

// for loop_unrolling
void GraphBuilder::Reset() {
  frame_ = graph_->GetFrame(0);
  current_block_ = nullptr;
  cur_bci_ = 0;
  graph_->Reset();
}

StopTraceReason GraphBuilder::HandleLoop() {
  Block *loop_head = current_block_;
  Graph *graph = loop_head->GetGraph();
  if (!graph->Config().GetBoolConfig(GraphJitConfig::kLoopUnrolling)) {
    return StopTraceReason::kStopTraceLoop_Unsupported;
  }

  // produce values of loop
  int end_bci = loop_head->end_ci();
  for (; cur_bci_ < end_bci; ++cur_bci_) {
    const Instr &instr = *graph_->instr_map().find(cur_bci_)->second;
    if (!DoByteCode(instr)) {
      break;
    }
    if (Utils::IsCallOp(instr.op()) && StopTraceReason::kNonStopTrace != HandleCall(0)) {
      break;
    }
  }

  // loop unrolling by loop values
  LoopUnrolling loopUnrollingExe = LoopUnrolling(*graph);
  (void)loopUnrollingExe.ExecuteLoopUnroll(loop_head);

  // clear values of loop
  loop_head->ClearTrackInfo();
  for (int bci = loop_head->begin_ci(); bci < end_bci; ++bci) {
    graph_->SetInstr(bci, nullptr);
  }

  if (!loopUnrollingExe.IsCFGChanged()) {
    return StopTraceReason::kStopTraceLoop_Unsupported;
  }
  // restart BuildGraph
  Reset();
  return StopTraceReason::kNonStopTrace;
}

void GraphBuilder::HandleBlockSwitch() {
  Block *next_block = graph_->GetBlockByBci(cur_bci_);
  if (current_block_ != next_block) {
    MS_LOG(DEBUG) << "=== block changed " << current_block_->Dump(false) << " -> " << next_block->Dump(false);
  }
  if (cur_bci_ == current_block_->end_ci() && !current_block_->GetJumpBB() && current_block_->GetFallBB()) {
    // merge to fall-through frame
    bool pruned = root_->graph_->Config().GetBoolConfig(GraphJitConfig::kPruneCase);
    bool merged = graph_->FindFrame(current_block_->GetFallBB()->begin_ci());
    if (!pruned || !merged) {
      // if prune branch just merge once
      MergeFrameState(current_block_->GetFallBB());
    }
  }
  current_block_ = next_block;
  if (current_block_->is_dead()) {
    cur_bci_ = current_block_->end_ci() - 1;
    return;
  }
  if (cur_bci_ == current_block_->begin_ci()) {
    MS_ASSERT(graph_->FindFrame(current_block_->begin_ci()));
    frame_ = graph_->GetFrame(current_block_->begin_ci());
  } else {
    graph_->SetFrame(cur_bci_, frame_);
  }
}

StopTraceReason GraphBuilder::BuildGraph(int depth) {
  graph_->SetFrame(0, frame_);
  int stop_trace_at = -1;
  StopTraceReason ret = StopTraceReason::kNonStopTrace;

  current_block_ = graph_->GetCFG()->GetFirstBB();
  cur_bci_ = 0;
  for (; cur_bci_ < static_cast<int>(graph_->GetInstrs().size()); ++cur_bci_) {
    HandleBlockSwitch();
    if (current_block_->is_dead()) {
      continue;
    }
    // check bb is loop header when bb tail
    if (current_block_->is_loop_head()) {
      ret = HandleLoop();
      if (ret == StopTraceReason::kNonStopTrace) {
        return BuildGraph(depth);
      }
      stop_trace_at = current_block_->begin_ci();
      break;
    }
    MS_EXCEPTION_IF_CHECK_FAIL(graph_->instr_map().find(cur_bci_) != graph_->instr_map().end(),
                               "not found index:" + std::to_string(cur_bci_));
    const auto &instr = *graph_->instr_map().find(cur_bci_)->second;
    MS_ASSERT(cur_bci_ == instr.bci());
    if (!DoByteCode(instr)) {
      stop_trace_at = instr.bci();
      ret = StopTraceReason::kStopTraceByteCode_Unsupported;
      break;
    }
    if (Utils::IsCallOp(instr.op())) {
      ret = HandleCall(depth);
      if (StopTraceReason::kNonStopTrace != ret) {
        stop_trace_at = instr.bci();
        break;
      }
    }
  }

  FillInstrNode(graph_);

  if (stop_trace_at == -1) {
    return StopTraceReason::kNonStopTrace;
  }
  current_block_->SetTrackResult(Block::kTrackBreak);
  graph_->StopTraceAt(stop_trace_at, ret);
  return ret;
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

bool GraphBuilder::RecursionCheck(PyCodeObject *co) {
  bool rec = co == this->GetGraph()->GetCodeObj();
  for (auto p = parent_; !rec && p; p = p->parent_) {
    rec = co == p->graph_->GetCodeObj();
  }
  return rec;
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
  if (ncells + nfrees > 0) {
    // not implement free variable merge
    return false;
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
    &PyComplex_Type, &PyMap_Type,  &PyBaseObject_Type, &PyRange_Type, &PyZip_Type,
    &PySlice_Type,   &PyBool_Type, &PyFloat_Type,      &PyLong_Type,  &PyType_Type,
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

  auto &nodes = current_block_->GetNodes();
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
static py::object CopyPyFunc(const py::object &o, const std::string &parent) {
  MS_EXCEPTION_IF_CHECK_FAIL(PyFunction_Check(o.ptr()), "must be function");
  PyFunctionObject *func = reinterpret_cast<PyFunctionObject *>(o.ptr());
  PyCodeObject *code = reinterpret_cast<PyCodeObject *>(func->func_code);
  PyObject *new_name = PyUnicode_FromFormat("%s%s%U", parent.c_str(), kPIJitCopyFuncKey, func->func_qualname);
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

bool GraphBuilder::ReplaceCall(CallNode *call_node, const py::object &new_func) {
  if (call_node->GetOpcode() == CALL_FUNCTION_EX && call_node->input(1)->GetOpcode() != BUILD_TUPLE) {
    // dynamic length variable arguments, user-defined unpack sequence
    return false;
  }

  auto &nodes = current_block_->GetNodes();
  MS_EXCEPTION_IF_CHECK_FAIL(nodes.back() == call_node, "CallNode must be last when build sub graph");

  std::stringstream key;
  key << std::string(py::str(new_func.ptr())) << "." << new_func.ptr();

  // new func node
  ValueNode *func_node = this->NewValueNode(AObject::Convert(new_func), LOAD_GLOBAL, -1, {});
  func_node->SetName(key.str().c_str());
  this->graph_->InstallToGlobal(func_node->GetName(), new_func);
  nodes.insert(nodes.end() - 1, func_node);

  ValueNode *self = nullptr;
  AObject::Type func_type = call_node->input(0)->GetVobj()->GetType();
  if (func_type == AObject::kTypeBoundMethod) {
    // new self node
    AObject *self_info = call_node->input(0)->GetVobj()->GetAttr(ID___self__);
    self = this->NewValueNode(self_info, LOAD_ATTR, -1, {call_node->input(0)});
    self->SetName(GraphBuilder::ID___self__);
    self->set_bci(call_node->bci());
    self->SetLineNo(call_node->GetLineNo());
    nodes.insert(nodes.end() - 1, self);
  } else if (func_type == AObject::kTypeCell || AObject::kTypeAnyValue) {
    self = call_node->input(0);
  }

  // replace node
  this->graph_->SetInstr(call_node->bci(), nullptr);
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
  ValueNode *tuple = this->NewValueNode(args_node->GetVobj(), BUILD_TUPLE, 0, args_node->getInputs());
  tuple->getInputs().insert(tuple->getInputs().begin(), self);
  tuple->SetOparg(args_node->GetOparg() + 1);
  tuple->set_bci(call_node->bci());
  tuple->SetLineNo(call_node->GetLineNo());
  nodes.insert(nodes.end() - 1, tuple);
  call_node->getInputs()[1] = tuple;
  return true;
}

static py::object GetPIJitCopiedFunc(const py::object &func, const std::string &parent) {
  if (!kPIJitConfigDefault.GetBoolConfig(GraphJitConfig::kCopyFuncOnlyOnceIfTraceBreak)) {
    return CopyPyFunc(func, parent);
  }
  PyObject *res = PyObject_GetAttrString(func.ptr(), kPIJitCopyFuncKey);
  if (res != nullptr) {
    return py::reinterpret_steal<py::object>(res);
  }
  PyErr_Clear();
  py::object copy = CopyPyFunc(func, parent);
  PyObject_SetAttrString(func.ptr(), kPIJitCopyFuncKey, copy.ptr());
  return copy;
}

// build sub-graph
StopTraceReason GraphBuilder::BuildSubGraph(CallNode *call_node, int depth, const py::object &func,
                                            GraphBuilder *subgraph) {
  auto code = subgraph->GetGraph()->GetGuard();
  if (code != nullptr) {
    code->GetGuard()->Backup();
  }
  StopTraceReason break_graph = subgraph->BuildGraph(depth + 1);

  if (!ApplyInlinePolicy(subgraph->GetGraph())) {
    if (code != nullptr) {
      code->GetGuard()->Rollback();
    }
    call_node->SetInlineReason(InlineReason::kInlinePolicyDisabled);
    if (subgraph->GetGraph()->GetRetVal()) {
      call_node->SetVobj(subgraph->GetGraph()->GetRetVal()->GetVobj());
    }
    if (break_graph == StopTraceReason::kNonStopTrace) {
      return StopTraceReason::kNonStopTrace;
    }
    if (getJitCompileResults(func.ptr(), false) != nullptr) {
      return StopTraceReason::kNonStopTrace;
    }
    // mark sub-routine to compile
    py::object func_copy = GetPIJitCopiedFunc(func, PyUnicode_AsUTF8(graph_->GetCodeObj()->co_name));
    (void)pi_jit_should_compile(func_copy, py::dict());
    *getJitCompileResults(func_copy.ptr(), false)->conf = root_->GetGraph()->Config();

    // replace function call
    // guard this function
    ReplaceCall(call_node, func_copy);
    return StopTraceReason::kNonStopTrace;
  }

  call_node->SetSubGraph(subgraph->GetGraph());
  if (break_graph != StopTraceReason::kNonStopTrace) {
    call_node->SetInlineReason(InlineReason::kInlinePartial);
    current_block_->SetTrackResult(Block::kTrackBreak);
    return break_graph;
  }
  call_node->SetVobj(subgraph->GetGraph()->GetRetVal()->GetVobj());
  call_node->SetInlineReason(InlineReason::kInline);
  return StopTraceReason::kNonStopTrace;
}

static bool UnpackDynamicLengthDictByBytecode() {
  // user defined mappings, dynamic length dictionary unpack
  return false;
}

bool GraphBuilder::UnpackCallExDict(std::vector<ValueNode *> *params, AbstractNodeList *extra_oper) {
  ValueNode *dict_node = params->back();
  AbstractNodeList dict_unpack;
  params->clear();
  if (dict_node->GetOpcode() != BUILD_MAP) {
    return UnpackDynamicLengthDictByBytecode();
  }
  if (dict_node->GetOparg() == 0) {
    extra_oper->push_back(this->NewInstrNode(POP_TOP, 0));
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

  // extra operations of unpack-call dict args for graph break
  for (int i = 0; i < dict_node->GetOparg(); ++i) {
    ValueNode *idx_node = this->NewValueNode(AObject::Convert(keys[i]), LOAD_CONST, -1, {});
    AObject *value = dict_node->input(i * 2 + 1)->GetVobj();
    dict_unpack.push_back(this->NewInstrNode(DUP_TOP, 0));
    dict_unpack.push_back(idx_node);
    dict_unpack.push_back(this->NewValueNode(value, BINARY_SUBSCR, 0, {dict_node, idx_node}));
    dict_unpack.push_back(this->NewInstrNode(ROT_TWO, 0));
  }
  dict_unpack.push_back(this->NewInstrNode(POP_TOP, 0));
  dict_unpack.push_back(const_keys);
  extra_oper->insert(nullptr, &dict_unpack);
  return true;
}

AbstractNodeList GraphBuilder::UnpackDynamicLengthTupleByBytecode(std::vector<ValueNode *> *params,
                                                                  AbstractNodeList tuple_unpack, ValueNode *args_node,
                                                                  CallNode *call_node) {
  // user-defined sequence, dynamic length tuple unpack
  auto items = static_cast<AbstractTuple *>(args_node->GetVobj())->items();
  for (uint i = 0; i < items.size(); i++) {
    ValueNode *idx_node = this->NewValueNode(AObject::Convert(py::int_(i)), LOAD_CONST, -1, {});
    auto value = this->NewValueNode(items[i], BINARY_SUBSCR, 0, {args_node, idx_node});
    params->insert(params->begin() + i, value);

    call_node->AddParam(value);
  }

  for (uint i = 0; i < items.size(); i++) {
    ValueNode *idx_node = this->NewValueNode(AObject::Convert(py::int_(i)), LOAD_CONST, -1, {});
    tuple_unpack.push_back(this->NewInstrNode(DUP_TOP, 0));
    tuple_unpack.push_back(idx_node);
    tuple_unpack.push_back(this->NewValueNode(items[i], BINARY_SUBSCR, 0, {args_node, idx_node}));
    tuple_unpack.push_back(this->NewInstrNode(ROT_TWO, 0));
  }
  tuple_unpack.push_back(this->NewInstrNode(POP_TOP, 0));

  return tuple_unpack;
}

bool GraphBuilder::UnpackExtraOper(AbstractNodeList tuple_unpack, int extra_local, AbstractNodeList *extra_oper,
                                   AbstractNodeList dict_unpack, bool has_dict) {
  if (has_dict) {
    extra_oper->push_back(this->NewInstrNode(STORE_FAST, extra_local));
    extra_oper->insert(nullptr, &tuple_unpack);
    extra_oper->push_back(this->NewInstrNode(LOAD_FAST, extra_local));
    extra_oper->insert(nullptr, &dict_unpack);
  } else {
    extra_oper->insert(nullptr, &tuple_unpack);
  }
  return true;
}

// unpack CALL_FUNCTION_EX parameters
// should do this when bytecode analyze ? replace origin opcode
bool GraphBuilder::UnpackCallExParams(std::vector<ValueNode *> *params, int extra_local, AbstractNodeList *extra_oper,
                                      bool *has_kw, CallNode *call_node) {
  bool has_dict = params->size() > 1;
  ValueNode *args_node = params->operator[](0);
  AbstractNodeList tuple_unpack;
  AbstractNodeList dict_unpack;
  if (!has_dict) {
    params->clear();
  } else if (!UnpackCallExDict(params, &dict_unpack)) {
    return false;
  }
  *has_kw = params->size();

  if (args_node->GetOpcode() != BUILD_TUPLE) {
    if ((args_node->GetVobj())->GetType() != AObject::kTypeTuple || args_node->GetVobj() == nullptr) {
      return false;
    }
    tuple_unpack = UnpackDynamicLengthTupleByBytecode(params, tuple_unpack, args_node, call_node);
    return UnpackExtraOper(tuple_unpack, extra_local, extra_oper, dict_unpack, has_dict);
  }

  params->insert(params->begin(), args_node->getInputs().begin(), args_node->getInputs().end());

  // extra operations of unpack-call tuple args for graph break
  for (int i = 0; i < args_node->GetOparg(); ++i) {
    ValueNode *idx_node = this->NewValueNode(AObject::Convert(py::int_(i)), LOAD_CONST, -1, {});
    tuple_unpack.push_back(this->NewInstrNode(DUP_TOP, 0));
    tuple_unpack.push_back(idx_node);
    tuple_unpack.push_back(this->NewValueNode(args_node->input(i)->GetVobj(), BINARY_SUBSCR, 0, {args_node, idx_node}));
    tuple_unpack.push_back(this->NewInstrNode(ROT_TWO, 0));
  }
  tuple_unpack.push_back(this->NewInstrNode(POP_TOP, 0));

  return UnpackExtraOper(tuple_unpack, extra_local, extra_oper, dict_unpack, has_dict);
}

bool GraphBuilder::PackKwParams(const py::object &func, std::vector<ValueNode *> *params, FrameStates *frame,
                                AbstractNodeList *dict_gen, std::vector<ValueNode *> *kwvargs) {
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

  // extra operations for generate kwvargs dictionary
  dict_gen->push_back(NewInstrNode(POP_TOP, 0));  // pop const keys
  dict_gen->push_back(NewInstrNode(BUILD_MAP, 0));

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
    dict_gen->push_back(NewInstrNode(ROT_TWO, 0));
    // if key is position arg, store it
    if (pos < argc) {
      dict_gen->push_back(NewInstrNode(STORE_FAST, pos));
      frame->SetLocal(pos, v);
      kw_2_p_cnt++;
      continue;
    }
    ValueNode *k = NewValueNode(AObject::Convert(key), LOAD_CONST, -1, {});

    dict_gen->push_back(k);
    dict_gen->push_back(NewInstrNode(ROT_TWO, 0));
    dict_gen->push_back(NewInstrNode(MAP_ADD, 1));

    kwvargs->push_back(k);
    kwvargs->push_back(v);
  }

  params->resize(params->size() - 1 - k_cnt);
  if (!(co->co_flags & CO_VARKEYWORDS)) {
    return kw_2_p_cnt == k_cnt;  // if not equal, too many key-word arguments
  }
  return true;
}

bool GraphBuilder::HandleKWParams(const py::object &func, std::vector<ValueNode *> *params, FrameStates *frame,
                                  AbstractNodeList *extra_oper) {
  PyCodeObject *co = reinterpret_cast<PyCodeObject *>(PyFunction_GET_CODE(func.ptr()));
  AbstractNodeList dict_gen;
  std::vector<ValueNode *> kwvargs;
  if (!PackKwParams(func, params, frame, &dict_gen, &kwvargs)) {
    // illegal arguments
    return false;
  }

  const int argc = co->co_argcount + co->co_kwonlyargcount;
  extra_oper->insert(nullptr, &dict_gen);
  if (!(co->co_flags & CO_VARKEYWORDS)) {
    // kw_2_p_cnt == k_cnt, all kw arguments is positions arguments
    extra_oper->push_back(NewInstrNode(POP_TOP, 0));
    return true;
  }

  int kwvarg_loc = argc + ((co->co_flags & CO_VARARGS) ? 1 : 0);
  AObject *dict = AObject::BuildOperations(CollectObjects(kwvargs), BUILD_MAP);
  frame->SetLocal(kwvarg_loc, NewValueNode(dict, BUILD_MAP, kwvargs.size() / 2, kwvargs));

  MS_ASSERT(seek(0)->GetType() == AbstractNode::Call);
  reinterpret_cast<CallNode *>(seek(0))->AddParam(frame->Local(kwvarg_loc));
  extra_oper->push_back(NewInstrNode(STORE_FAST, kwvarg_loc));
  return true;
}

bool GraphBuilder::CheckAndSetDefaultParams(const py::object &func, AbstractNodeList *extra_oper, FrameStates *frame,
                                            int position_argc) {
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
    extra_oper->push_back(c);
    extra_oper->push_back(NewInstrNode(STORE_FAST, i));
  }
  return true;
}

static ValueNode *GetBoundSelf(CallNode *call_node, AbstractNodeList *extra_oper) {
  ValueNode *func_val = call_node->input(0);
  AObject *vo = func_val->GetVobj();
  auto &alloc = call_node->GetGraph()->allocator();

  ValueNode *self = nullptr;
  AObject *tmp;
  switch (vo->GetType()) {
    case AObject::kTypeBoundMethod:
      tmp = func_val->get_attr(GraphBuilder::ID___self__);
      self = alloc.NewNode<ValueNode>(tmp, LOAD_ATTR, -1, std::vector<ValueNode *>({func_val}));
      self->SetName(GraphBuilder::ID___self__);
      self->SetGraph(call_node->GetGraph());
      call_node->AddParam(self);
      extra_oper->push_back(self);
      break;
    case AObject::kTypeCell: /* fallthrough */
    case AObject::kTypeAnyValue:
      self = func_val;
      break;
    case AObject::kTypeFunction:
      break;
    default:
      MS_LOG(EXCEPTION) << "unimplemented type " << vo->ToString();
  }
  if (self != nullptr) {
    extra_oper->push_back(alloc.NewNode<InstrNode>(STORE_FAST, 0));
  } else {
    extra_oper->push_back(alloc.NewNode<InstrNode>(POP_TOP, 0));
  }
  return self;
}

bool GraphBuilder::HandlePositionParams(const py::object &func, std::vector<ValueNode *> *params,
                                        AbstractNodeList *extra_oper, FrameStates *frame) {
  MS_ASSERT(seek(0)->GetType() == AbstractNode::Call);
  CallNode *call_node = reinterpret_cast<CallNode *>(seek(0));
  PyCodeObject *co = reinterpret_cast<PyCodeObject *>(PyFunction_GET_CODE(func.ptr()));
  AObject::Type callable_type = call_node->input(0)->GetVobj()->GetType();
  AbstractNodeList load_self;

  // here save callable object to debug
  frame->SetLocal(co->co_nlocals, call_node->input(0));
  load_self.push_back(NewInstrNode(DUP_TOP, 0));
  load_self.push_back(NewInstrNode(STORE_FAST, co->co_nlocals));
  ValueNode *self = GetBoundSelf(call_node, &load_self);
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
    extra_oper->push_back(m);
    extra_oper->push_back(NewInstrNode(STORE_FAST, kwvarg_loc));
  }

  if (has_varg) {
    int vargc = pargc > argc ? pargc - argc : 0;
    std::vector<ValueNode *> vargs(params->end() - vargc, params->end());
    params->resize(params->size() - vargc);

    auto vo = AObject::BuildOperations(CollectObjects(vargs), BUILD_TUPLE);
    ValueNode *build_tuple = NewValueNode(vo, BUILD_TUPLE, vargc, vargs);
    call_node->AddParam(build_tuple);
    frame->SetLocal(varg_loc, build_tuple);
    extra_oper->push_back(build_tuple);
    extra_oper->push_back(NewInstrNode(STORE_FAST, varg_loc));
  }

  pargc = params->size();
  for (int i = pargc - 1; i >= 0; --i) {
    if (frame->Local(i) != &ValueNode::UnboundLocal) {
      MS_LOG(DEBUG) << "duplicate key-word parameter error";
      return false;
    }
    frame->SetLocal(i, params->back());
    params->pop_back();
    extra_oper->push_back(NewInstrNode(STORE_FAST, i));
    if (i == 0 && self) {
      extra_oper->erase(extra_oper->back());
    }
  }
  if (argc > 0) {  // self is position arguments
    // top of stack is only left callable object
    extra_oper->insert(nullptr, &load_self);
  }
  return CheckAndSetDefaultParams(func, extra_oper, frame, pargc);
}

bool GraphBuilder::HandleCallParameters(const py::object &func_info, CallNode *call_node, FrameStates *frame) {
  PyCodeObject *co = reinterpret_cast<PyCodeObject *>(PyFunction_GET_CODE(func_info.ptr()));
  frame->ResizeLocal(co->co_nlocals + 1);

  AbstractNodeList extra_oper;
  std::vector<ValueNode *> params(call_node->getInputs().begin() + 1, call_node->getInputs().end());
  int op = call_node->GetOpcode();
  bool has_kw = (op == CALL_FUNCTION_KW);
  if (op == CALL_FUNCTION_EX && !UnpackCallExParams(&params, co->co_nlocals, &extra_oper, &has_kw, call_node)) {
    return false;  // ex_dict infer failed or user-defined sequence and map arguments
  }
  if (has_kw && !HandleKWParams(func_info, &params, frame, &extra_oper)) {
    return false;
  }
  if (!HandlePositionParams(func_info, &params, &extra_oper, frame)) {
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
      frame->SetLocal(arg_index, &ValueNode::UnboundLocal);

      PyObject *cell = cell_node->GetVobj()->GetPyObject().ptr();
      PyObject *cell_contents = arg_node->GetVobj() ? arg_node->GetVobj()->GetPyObject().inc_ref().ptr() : nullptr;
      MS_EXCEPTION_IF_CHECK_FAIL(cell && PyCell_Check(cell) && PyCell_GET(cell) == nullptr, "must be a empty closure");

      extra_oper.push_back(NewInstrNode(LOAD_FAST, arg_index));
      ValueNode *n = NewValueNode(nullptr, STORE_DEREF, i, {arg_node});
      extra_oper.push_back(n);

      cell_node->AddCellOper(n);
      cell_node->SetValue(arg_node);
      Py_XSETREF(PyCell_GET(cell), cell_contents);
      call_node->AddParam(n);
    }
  }
  call_node->SetExtraOper(reinterpret_cast<ValueNode *>(extra_oper.head()));
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
    return py::object();
  }

  if (WhiteListFuncCheckAndInfer(call_node, callable_info)) {
    return py::object();
  }

  // find code object
  callable_info = GetFuncInfo(call_node->input(0));
  if (callable_info.ptr() == nullptr) {
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
  if (RecursionCheck(co)) {
    call_node->SetInlineReason(InlineReason::kInlineRecurse_Unsupported);
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

  if (call_node->GetSubGraph()) {
    MS_EXCEPTION_IF_NULL(call_node->GetSubGraph()->GetRetVal());
    seek(0) = call_node->GetSubGraph()->GetRetVal();
  }
  return stop_reason;
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
  g->BuildGraph();
  if (conf.GetBoolConfig(GraphJitConfig::kPrintAfterAll)) {
    GRAPH_JIT_LOG_F("====== infer results of %s", std::string(py::str(callable.ptr())).c_str());
    g->DumpDFG();
  }

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

void GraphBuilder::DumpDFG() {
  GRAPH_JIT_LOG_F("\n*** Dump ByteCode After Data Flow Graph on [%s] ***\n", graph_->getCodeInfoName().c_str());
  graph_->print();
}
}  // namespace graph
}  // namespace jit
}  // namespace mindspore
