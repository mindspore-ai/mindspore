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
#include <set>
#include "pipeline/jit/pi/graph_capture/bytecode_inliner.h"
#include "pipeline/jit/pi/graph_capture/graph.h"
#include "pipeline/jit/pi/graph_guard/cache.h"

namespace mindspore {
namespace jit {
namespace graph {

extern std::string PrintInstr(const std::vector<std::unique_ptr<Instr>> &list);
extern std::vector<ValueNode *> CollectInterpretOutputs(const FrameStates &last_frame, const BitMap &alive,
                                                        std::vector<int> *alive_locals);

void BytecodeInliner::Run() {
  if (graph_->IsBreakAtLoop() && !graph_->RestoreLoopStatus()) {
    return;
  }

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

  if (graph_->GetStopTraceBci() != -1) {
    InitCFG();
  }

  Rebuild();

  PY_PRINT_F("%s\n%s", __PRETTY_FUNCTION__, PrintInstr(cfg_->instr_pool()).c_str());

  ResetGraphStat();
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
    .inputs = graph_->GetFrame(0).GetLocals(),
    .outputs = std::vector<ValueNode *>(),
    .operations = traced_nodes_,
  };
  CodeGenerator cg(&ns);

  std::vector<int> alive_locals;
  if (last_frame_ != nullptr) {
    ns.outputs = CollectInterpretOutputs(*last_frame_, cfg_->GetLiveness()->CollectAlive(0), &alive_locals);
  } else {
    ns.outputs.push_back(graph_->GetRetVal());
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
  FixInstr(graph, local_off, &list);
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
    return false;
  }
  return true;
}

static bool CanInine(Graph *top_graph, Graph *sub_graph) {
  if (sub_graph == nullptr) {
    return false;
  }
  if (sub_graph->IsBreakAtLoop()) {
    return false;
  }
  if (!EliminateSideEffect(top_graph, sub_graph)) {
    return false;
  }
  return true;
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

  if (node->GetType() == AbstractNode::Call) {
    CallNode *call_node = static_cast<CallNode *>(node);
    if (CanInine(this->graph_, call_node->GetSubGraph())) {
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
}

void BytecodeInliner::FixInstr(Graph *graph, int local_off, std::vector<std::unique_ptr<Instr>> *list) {
  MS_EXCEPTION_IF_CHECK_FAIL(list->size() > 0 && list->back()->op() == RETURN_VALUE,
                             "check instruction list, not end with RETURN_VALUE");
  for (const auto &i : *list) {
    if (Utils::IsLocalAccessOp(i->op())) {
      i->set_arg(i->arg() + local_off);
      continue;
    }
    if (i->op() == RETURN_VALUE) {
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

  if (graph != this->graph_) {
    list->back()->set_op(NOP);
  } else {
    list->back()->set_op(RETURN_VALUE);
  }
}

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

}  // namespace graph
}  // namespace jit
}  // namespace mindspore
