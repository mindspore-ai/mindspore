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
#include "pipeline/jit/pi/graph_capture/bytecode_inliner.h"
#include <set>
#include <utility>
#include <algorithm>
#include <string>
#include "pipeline/jit/pi/graph_capture/graph.h"
#include "pipeline/jit/pi/graph_capture/side_effect.h"
#include "pipeline/jit/pi/graph_guard/cache.h"
#include "pipeline/jit/pi/pi_jit_config.h"

namespace mindspore {
namespace pijit {

extern std::string PrintInstr(const std::vector<std::unique_ptr<Instr>> &list);
extern bool CheckMSConstexpr(const py::object &func);
extern bool CheckJitConstexpr(const py::object &func);
extern bool ApplyInlinePolicy(Graph *g);

void BytecodeInliner::Run() {
  if (graph_->IsBreakAtLoop() && !graph_->RestoreLoopStatus()) {
    return;
  }

  inline_partial_ = graph_->Config().GetBoolConfig(GraphJitConfig::kFeatureBreakAtInlinedFunction);
  cfg_ = std::make_unique<CFG>(nullptr);

  // collect traced nodes, inline second half bytecode
  if (graph_->GetStopTraceBci() != -1) {
    last_frame_ = std::make_unique<FrameStates>();
    ProcessGraph(graph_, 0);
  } else {
    CollectTracedNodes(graph_);
  }

  if (traced_nodes_.empty()) {
    return;
  }

  if (inline_partial_ && graph_->GetStopTraceBci() != -1) {
    InitCFG();
  }

  Rebuild();

  if (graph_->Config().GetBoolConfig(GraphJitConfig::kPrintBB)) {
    GRAPH_JIT_LOG_F("%s\n\n", cfg_->DumpBBs().c_str());
  }

  ResetGraphStat();

  if (graph_->Config().GetBoolConfig(GraphJitConfig::kLogGraphBreak)) {
    std::stringstream s;
    s << "graph new break bci is " << new_break_bci_ << " after inline";
    if (reconstructed_value_ != nullptr) {
      const auto &instr = graph_->GetCFG()->instr_pool()[new_break_bci_];
      s << ", node is reconstructed by inliner [" << reconstructed_value_->ToString() << "] -> [" << instr->ToString()
        << "]";
    }
    GRAPH_JIT_LOG_F("%s", s.str().c_str());
  }
}

void BytecodeInliner::ResetGraphStat() {
  graph_->GetFrames().swap(new_frames_);
  graph_->GetCFG().swap(cfg_);
  graph_->GetTracedNodes().swap(traced_nodes_);
  if (graph_->GetStopTraceBci() != -1) {
    graph_->StopTraceAt(new_break_bci_, graph_->GetStopTraceReason());
  }
}

void BytecodeInliner::ResetCFG(CodeGenerator *cg) {
  std::vector<std::unique_ptr<Instr>> list = cg->MoveCode();
  std::move(cfg_->instr_pool().begin(), cfg_->instr_pool().end(), std::back_inserter(list));
  cfg_->instr_pool().swap(list);
  cfg_->bb_pool().clear();
  cfg_->liveness().reset();
  cfg_->SetLocalCount(cg->GetCode().co_nlocals);
  InitCFG();
}

void BytecodeInliner::Rebuild(CodeGenerator *cg) {
  FrameStates new_f = graph_->GetFrame(0);
  int new_bci = 0;
  cg->SetGlobals(extra_globals_);
  cg->Init();
  cg->MarkAlive();
  new_frames_[0] = std::make_unique<FrameStates>(new_f);
  for (size_t index = 0; index < traced_nodes_.size(); ++index) {
    ValueNode *node = traced_nodes_[index];
    if (IsNonLocalValue(node)) {
      node->set_bci(new_bci);
      continue;
    }
    cg->BuildOper(node, index);

    // reset bci
    int last_op = cg->GetCode().co_code.back()->op();
    new_bci = cg->GetCode().co_code.size() - 1 - (last_op == POP_TOP || last_op == STORE_FAST);
    node->set_bci(new_bci);

    // reset frame status
    new_f.GetStacks() = node->getInputs();
    new_frames_[new_bci] = std::make_unique<FrameStates>(new_f);
    if (last_op == STORE_FAST) {
      int arg = cg->GetCode().co_code.back()->arg();
      new_f.ResizeLocal(std::max(new_f.GetLocals().size(), static_cast<size_t>(arg + 1)));
      new_f.SetLocal(arg, node);
    }
  }
  int nlocals = std::max(cg->GetLocalsMap().size(), new_f.GetLocals().size());
  nlocals = std::max(nlocals, cg->GetCode().co_nlocals);
  cg->SetLocalsCount(nlocals);
}

void BytecodeInliner::Rebuild() {
  NodeSet ns = {
    graph_->GetFrame(0).GetLocals(),
    std::vector<ValueNode *>(),
    traced_nodes_,
  };
  CodeGenerator cg(&ns);

  std::vector<int> alive_locals;
  if (last_frame_ != nullptr) {
    BitMap alive = inline_partial_ ? cfg_->GetLiveness()->CollectAlive(0)
                                   : graph_->GetCFG()->GetLiveness()->CollectAlive(graph_->GetStopTraceBci());
    ns.outputs = Graph::CollectAliveNode(*last_frame_, &alive, &alive_locals);
  } else {
    ns.outputs.push_back(graph_->GetRetVal());
  }
  if (graph_->Config().GetBoolConfig(GraphJitConfig::kEnableEliminateUnusedOperation)) {
    for (auto side_effect_node : graph_->GetSideEffectNodes()) {
      for (auto item : side_effect_node->getInputs()) {
        ns.outputs.push_back(item);
      }
    }
    for (auto replace_map : graph_->GetSideEffectReplacedMap()) {
      ns.outputs.push_back(replace_map.second);
      for (auto item : replace_map.second->getInputs()) {
        ns.outputs.push_back(item);
      }
    }
    // erase dead local between inline and code rebuild
    EraseDeadLocal(ns.outputs);
    EliminateClosureSideEffect();
  }
  Rebuild(&cg);
  if (last_frame_ != nullptr) {
    std::for_each(ns.outputs.begin(), ns.outputs.end(), [&cg](ValueNode *i) { cg.LoadValue(i); });
    std::for_each(alive_locals.rbegin(), alive_locals.rend(), [&cg](int i) { cg.NewInstr(STORE_FAST, i); });
    cg.SetLocalsCount(last_frame_->GetLocals().size());
  } else {
    cg.GenReturn();
  }

  if (last_frame_ != nullptr) {
    MS_EXCEPTION_IF_CHECK_FAIL(new_frames_.find(cg.GetCode().co_code.size()) == new_frames_.end(),
                               "duplicate frame status");
    new_break_bci_ = cg.GetCode().co_code.size();
    new_frames_[new_break_bci_] = std::move(last_frame_);
  }

  ResetCFG(&cg);
}

void BytecodeInliner::CollectTracedNodes(Graph *graph) {
  for (ValueNode *n : graph->GetTracedNodes()) {
    if (n->GetType() != AbstractNode::Call) {
      traced_nodes_.push_back(n);
      continue;
    }
    CallNode *call_node = static_cast<CallNode *>(n);
    if (call_node->GetSubGraph() == nullptr || call_node->GetInlineReason() != InlineReason::kInline) {
      traced_nodes_.push_back(n);
      continue;
    }
    std::copy(call_node->GetParams().begin(), call_node->GetParams().end(), std::back_inserter(traced_nodes_));
    CollectTracedNodes(call_node->GetSubGraph());
  }
  // collect side_effect_nodes // graph_ is top graph

  if (graph->GetSideEffect() != nullptr) {
    graph_->GetSideEffect()->ReprocessVariableMutationMaps();

    for (auto side_effect_item : graph->GetSideEffect()->GetSideEffectInstrs()) {
      graph_->SetSideEffectNode(side_effect_item.first);
    }
    for (auto side_effect_item : graph->GetSideEffect()->GetReplaceMaps()) {
      graph_->SetSideEffectReplacedMap(side_effect_item.first, side_effect_item.second);
    }
  }
}

void BytecodeInliner::ProcessGraph(Graph *graph, int local_off) {
  int break_bci = graph->GetStopTraceBci();
  if (break_bci == -1) {
    return;
  }

  // build last frame
  const FrameStates &f = graph->GetFrame(break_bci);
  last_frame_->GetLocals().insert(last_frame_->GetLocals().end(), f.GetLocals().begin(), f.GetLocals().end());
  last_frame_->GetStacks().insert(last_frame_->GetStacks().end(), f.GetStacks().begin(), f.GetStacks().end());

  CollectTracedNodes(graph);

  const auto &nodes = graph->GetTracedNodes();
  if (nodes.size() > 0 && nodes.back()->bci() == break_bci) {
    // break at traced value
    Reconstruct(nodes.back(), local_off + f.GetLocals().size());
    break_bci++;
  } else {
    // break at unsupported bytecode
    MS_EXCEPTION_IF_CHECK_FAIL(nodes.empty() || break_bci > nodes.back()->bci(), "check break bci");
    new_break_bci_ = 0;
  }

  std::vector<std::unique_ptr<Instr>> list = CodeGenerator::CopyInstr(graph->GetCFG()->instr_pool(), break_bci);
  if (inline_partial_) {
    FixInstr(graph, local_off, &list);
  }
  std::move(list.begin(), list.end(), std::back_inserter(cfg_->instr_pool()));
  cfg_->SetLocalCount(std::max(static_cast<size_t>(cfg_->GetLocalCount()), local_off + f.GetLocals().size()));
}

static bool EliminateSideEffect(Graph *top_graph, Graph *sub_graph) {
  /**
   * TODO:
   * kFeatureBreakAtInlinedFunc
   * eliminate untracked bytecode side effect after graph break
   * 1. eliminate MAKE_FUNCTION if it has global access and globals is not same as top func
   * 2. eliminate STORE_GLOBAL and DELETE_GLOBAL if globals is not same as top func
   * 3. eliminate closure access operations if function has cell or free variable
   */
  if (top_graph != sub_graph && sub_graph->GetFrame(0).GetClosures().size() != 0) {
    PyObject *frees = sub_graph->GetCodeObj()->co_freevars;
    if (PyTuple_GET_SIZE(frees) == 1 && std::string("__class__") == PyUnicode_AsUTF8(PyTuple_GET_ITEM(frees, 0))) {
      /**
       * BUGS: not check super call or free variable access after the break point
       **/
      return true;
    }
    return false;
  }
  /**
   * BUGS: not check MAKE_FUNCTION which has global access after the break point
   **/
  return true;
}

static bool CanIninePartial(Graph *top_graph, Graph *sub_graph) {
  if (sub_graph == nullptr) {
    return false;
  }
  if (sub_graph->IsBreakAtLoop()) {
    return false;
  }
  if (!EliminateSideEffect(top_graph, sub_graph)) {
    return false;
  }
  return ApplyInlinePolicy(sub_graph);
}

void BytecodeInliner::Reconstruct(ValueNode *node, int local_off) {
  static const std::set<int> not_value_oper = {
    STORE_DEREF,  DELETE_DEREF,  STORE_GLOBAL, DELETE_GLOBAL, STORE_ATTR, DELETE_ATTR,
    STORE_SUBSCR, DELETE_SUBSCR, IMPORT_STAR,  RAISE_VARARGS, RERAISE,
  };
  traced_nodes_.pop_back();

  Graph *graph = node->GetGraph();
  const auto &instr = graph->GetCFG()->instr_pool()[node->bci()];

  int stack_effect = PyCompile_OpcodeStackEffect(instr->op(), instr->arg());
  bool is_value = not_value_oper.find(instr->op()) == not_value_oper.end();
  MS_EXCEPTION_IF_CHECK_FAIL(stack_effect <= 0 && stack_effect != PY_INVALID_STACK_EFFECT,
                             "check break bci, too many value produced");
  last_frame_->Popn(-stack_effect + is_value);

  if (inline_partial_ && node->GetType() == AbstractNode::Call) {
    CallNode *call_node = static_cast<CallNode *>(node);
    if (CanIninePartial(this->graph_, call_node->GetSubGraph())) {
      std::copy(call_node->GetParams().begin(), call_node->GetParams().end(), std::back_inserter(traced_nodes_));
      ProcessGraph(call_node->GetSubGraph(), local_off);
      return;
    }
  }
  for (auto i : node->getInputs()) {
    last_frame_->Push(i);
  }
  MS_EXCEPTION_IF_CHECK_FAIL(cfg_->instr_pool().empty(), "just call once if graph break at traced value");
  cfg_->NewInstrNode(*instr);
  reconstructed_value_ = node;

  /**
   * TODO: if the node not match the instruction opcode, check it's sideeffect
   */
}

void BytecodeInliner::FixInstr(Graph *graph, int local_off, std::vector<std::unique_ptr<Instr>> *list) {
  if (list->empty()) {
    return;
  }
  for (const auto &i : *list) {
    if (Utils::IsLocalAccessOp(i->op())) {
      i->set_arg(i->arg() + local_off);
      continue;
    }
    if (this->graph_ != graph && i->op() == RETURN_VALUE) {
      i->set_op(JUMP_FORWARD);
      i->set_extra_jump(list->back().get());
      continue;
    }
    if (graph->GetGlobals().ptr() == this->graph_->GetGlobals().ptr()) {
      continue;
    }
    if (i->op() == LOAD_GLOBAL) {
      PyObject *value = PyObject_GetItem(graph->GetGlobals().ptr(), py::str(i->name()).ptr());
      py::object _value_handle = py::reinterpret_steal<py::object>(value);
      if (value == nullptr) {
        PyErr_Clear();
        continue;
      }
      auto tr = std::make_shared<RootTrace>(value, TraceType::Global, -1, i->name(), graph->GetModuleName());
      graph->GetGuard()->GetGuard()->GuardOn(tr, GuardLevel::GId);
      std::string key = i->name();
      MapAdd(extra_globals_, key, _value_handle, &key);
      i->set_name(key);
      continue;
    }
  }

  if (list->back()->op() != JUMP_FORWARD || list->back()->extra_jump() != list->back().get()) {
    return;
  }
  list->back()->set_extra_jump(nullptr);
  if (graph != this->graph_) {
    list->back()->set_op(NOP);
  } else {
    list->back()->set_op(RETURN_VALUE);
  }
}

/**
 * TODO:
 * unify the implementations of cfg initialization
 */
void BytecodeInliner::InitCFG() {
  const auto &list = cfg_->instr_pool();

  // reset bci, erase unused jump
  CodeGenerator::EraseUnusedInstr(&cfg_->instr_pool());

  // mark labels, ordered map
  std::map<int, Block *> blocks;
  blocks.insert({0, cfg_->NewBBAppend()});
  for (const auto &i : list) {
    size_t bci = list.size();
    if (Utils::IsNonFall(i->op())) {
      bci = i->bci() + 1;
    }
    if (i->extra_jump() != nullptr) {
      bci = i->bci() + 1;
      if (blocks.find(i->extra_jump()->bci()) == blocks.end()) {
        blocks.insert({i->extra_jump()->bci(), cfg_->NewBBAppend()});
      }
    }
    if (bci != list.size() && blocks.find(bci) == blocks.end()) {
      blocks.insert({bci, cfg_->NewBBAppend()});
    }
  }

  // link blocks, set range
  for (auto iter = blocks.begin(); iter != blocks.end();) {
    Block *cur = iter->second;
    int head = iter->first;
    int back;
    iter++;
    if (iter != blocks.end()) {
      back = iter->first;
    } else {
      back = list.size();
    }
    cur->set_begin_ci(head);
    cur->set_end_ci(back);
    const auto &instr = list[back - 1];
    if (instr->extra_jump()) {
      cur->SetJumpBB(blocks[instr->extra_jump()->bci()]);
    }
    if (!Utils::IsNonFall(instr->op())) {
      cur->SetFallBB(iter->second);
    }
  }
  cfg_->MarkDeadBB();
  cfg_->GetLiveness();
}

static bool IsEliminate(ValueNode *v) {
  int op = v->GetOpcode();
  if (Utils::IsNoSideEffectOp(op)) {
    return true;
  }
  if (Utils::IsGeneralNoSideEffectOp(op)) {
    if (Utils::IsBinaryMathOp(op) || op == COMPARE_OP) {
      return v->input(0)->GetVobj()->GetType() != AObject::kTypeAnyValue;
    }
    return true;
  }
  if (Utils::IsBinaryMathOp(op)) {
    // inplace binary
    AObject::Type t = v->input(0)->GetVobj()->GetType();
    return t != AObject::kTypeAnyValue && t != AObject::kTypeList && t != AObject::kTypeCell &&
           t != AObject::kTypeNNCellList;
  }
  if (Utils::IsCallOp(op)) {
    py::object callable = v->GetVobj()->GetPyObject();
    if (callable.ptr() == nullptr) {
      return false;
    }
    return CheckJitConstexpr(callable) || CheckMSConstexpr(callable);
  }
  if (op == GET_ITER) {
    return v->input(0)->GetVobj()->GetType() != AObject::kTypeAnyValue;
  }
  return false;
}

void BytecodeInliner::EraseDeadLocal(const std::vector<ValueNode *> &alive_nodes) {
  std::set<ValueNode *> alive;
  for (auto i : alive_nodes) {
    alive.insert(i);
  }

  // erase dead locals
  std::set<ValueNode *> used;
  do {
    used = alive;
    for (auto i : traced_nodes_) {
      for (auto j : i->getInputs()) {
        used.insert(j);
      }
    }
    auto iter = std::remove_if(traced_nodes_.begin(), traced_nodes_.end(), [&used](ValueNode *i) {
      // check it
      return used.find(i) == used.end() && IsEliminate(i);
    });
    if (iter == traced_nodes_.end()) {
      break;
    }
    traced_nodes_.erase(iter, traced_nodes_.end());
  } while (true);
}

void BytecodeInliner::EliminateClosureSideEffect() {
  PyCodeObject *co = graph_->GetCodeObj();
  int ncells = PyTuple_GET_SIZE(co->co_cellvars);
  int nfrees = PyTuple_GET_SIZE(co->co_freevars);
  if (ncells + nfrees == 0) {
    return;
  }
  std::set<InstrNode *> alive_closure_access;

  if (last_frame_ != nullptr) {
    auto iter = std::find_if(cfg_->instr_pool().begin(), cfg_->instr_pool().end(), [](const std::unique_ptr<Instr> &i) {
      return i->op() == LOAD_DEREF || (i->op() == MAKE_FUNCTION && (i->arg() & 0x08));
    });
    if (iter != cfg_->instr_pool().end()) {
      return;
    }
  }

  for (auto i : traced_nodes_) {
    if (i->GetOpcode() == MAKE_FUNCTION && (i->GetOparg() & 0x08)) {
      ValueNode *tuple = *(i->getInputs().end() - 3);
      for (auto c : tuple->getInputs()) {
        const auto &nodes = static_cast<CellVarNode *>(c)->GetCellOper();
        alive_closure_access.insert(nodes.begin(), nodes.end());
      }
    }
  }

  auto iter = std::remove_if(traced_nodes_.begin(), traced_nodes_.end(), [&alive_closure_access](ValueNode *i) {
    int op = i->GetOpcode();
    return (op == STORE_DEREF || op == DELETE_DEREF) && alive_closure_access.find(i) == alive_closure_access.end();
  });
  traced_nodes_.erase(iter, traced_nodes_.end());

  for (auto item = traced_nodes_.begin(); item != traced_nodes_.end();) {
    if ((*item)->GetOpcode() == STORE_DEREF) {
      if ((*item)->getInputs()[0]->GetOpcode() == LOAD_DEREF &&
          (*item)->getInputs()[0]->GetOparg() == (*item)->GetOparg() &&
          (*item)->getInputs()[0]->GetGraph() == (*item)->GetGraph()) {
        item = traced_nodes_.erase(item);
      } else {
        ++item;
      }
    } else {
      ++item;
    }
  }
}

}  // namespace pijit
}  // namespace mindspore
