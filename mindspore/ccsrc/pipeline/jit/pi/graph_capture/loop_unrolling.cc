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
#include "pipeline/jit/pi/graph_capture/loop_unrolling.h"
#include <fstream>
#include <map>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>
#include "pipeline/jit/pi/graph_capture/graph.h"
#include "pipeline/jit/pi/graph_capture/loop.h"
#include "pipeline/jit/pi/pi_jit_config.h"

namespace mindspore {
namespace jit {
namespace graph {
#define CHECK_PHASE(func, ...)                   \
  do {                                           \
    res_ = func(__VA_ARGS__);                    \
    if (!LoopUnrolling::IsloopUnorlling(res_)) { \
      return;                                    \
    }                                            \
  } while (0)

LoopUnrollingReason LoopUnrolling::ExecuteLoopUnroll(Block *header) {
  if (graph_.loops().empty()) {
    return kCanNotUnroll;
  }
  for (auto *lp : graph_.loops()) {
    if (lp->header() == header) {
      loop_ = lp;
      break;
    }
  }
  if (loop_ == nullptr) {
    return kCanNotUnroll;
  }
  Run();
  // Dump loop unrolling info
  if (graph_.Config().GetBoolConfig(GraphJitConfig::kPrintAfterAll)) {
    GRAPH_JIT_LOG_F("%s\n\n", DumpLoopUnrolling().c_str());
  }
  if (IsloopUnorlling(res_) && graph_.Config().GetBoolConfig(GraphJitConfig::kPrintBB)) {
    GRAPH_JIT_LOG_F("%s\n\n", graph_.GetCFG()->DumpBBs("after loop unrolling ").c_str());
  }
  return res_;
}

void LoopUnrolling::Run() {
  // check only one exit and one backedges in loop
  if (loop_->exits().empty() || loop_->exits().size() >= 2 || loop_->backedges().size() >= 2) {
    res_ = kCanNotSplitGoto;
    return;
  }
  // check pred of exit has only succ block
  Block *exit = *loop_->exits().begin();
  if (exit->pred_bbs().size() != 1) {
    res_ = kCanNotSplitGoto;
    return;
  }
  Block *backedge = *loop_->backedges().begin();
  if (backedge->GetFallBB() != nullptr) {
    res_ = kCanNotJumpBackedge;
    return;
  }
  // check foritem
  if (loop_->header()->instrs().front().op() == FOR_ITER) {
    CHECK_PHASE(AnalyzeForItem);
  }
  CHECK_PHASE(CheckLoopUnrollingSideeffect);
  is_cfg_changed_ = true;
  RemoveBackedge();
  CopyAndInsertBB();
  FixupInstr();
}

LoopUnrollingReason LoopUnrolling::AnalyzeForItem() {
  // find GET_ITER opcode
  MS_EXCEPTION_IF_NULL(loop_value_);
  AObject *loop_vobj = loop_value_->GetVobj();
  if (!loop_vobj) {
    return kCanNotUnroll;
  }
  PyObject *obj = loop_vobj->GetPyObject().ptr();
  // check unrolling count
  if (loop_vobj->GetType() == AObject::kTypeList || loop_vobj->GetType() == AObject::kTypeTuple) {
    AbstractTuple *list = static_cast<AbstractTuple *>(loop_vobj);
    if (!list->IsElementValid()) {
      return kCanNotUnroll;
    }
    AddLoopGurad(loop_value_);
    unrolling_count_ = list->size();
    loop_op_ = NOP;
    loop_arg_ = 0;
  } else if (loop_vobj->GetType() == AObject::kTypeNNCellList && obj != nullptr) {
    AddLoopGurad(loop_value_);
    unrolling_count_ = PyObject_Size(obj);
    loop_op_ = NOP;
    loop_arg_ = 0;
  } else {
    return kCanNotUnroll;
  }
  return kCanForItemUnroll;
}

bool LoopUnrolling::AddLoopGurad(ValueNode *value) { return graph_.GuardValueNode(value); }

LoopUnrollingReason LoopUnrolling::CheckLoopUnrollingSideeffect() {
  // check length
  if (unrolling_count_ <= 0 || unrolling_count_ > graph_.Config().getIntConfig(GraphJitConfig::kMaxLoopUnrolling)) {
    return kCanNotMaxCount;
  }
  if (loop_value_ == nullptr && loop_value_->GetVobj()) {
    return res_;
  }
  // check if loop_value is called by CFunction, e.g. list.append()
  // check side effects
  return res_;
}

void LoopUnrolling::AddLoopUnrollingInstr(Block *bb, int count) {
  bb->set_is_loop_head(false);
  const Instr &first_instr = bb->instrs().front();
  bb->RemoveInstrs();
  // remove GET_ITER and adding [count - 1] DUP_TOP
  if (loop_op_ == NOP && count == 0) {
    // GET_ITER --> DUP_TOP
    if (iter_instr_ != nullptr) {
      iter_instr_->set_op(DUP_TOP);
      iter_instr_->set_arg(0);
    }
    if (unrolling_count_ == 1) {
      Instr *instr = graph_.GetCFG()->NewInstrNode(first_instr.bci(), POP_TOP, 0, first_instr.line());
      bb->AddInstr(instr);
    }
    for (int i = 0; i < unrolling_count_ - 2; ++i) {
      Instr *instr = graph_.GetCFG()->NewInstrNode(first_instr.bci(), DUP_TOP, 0, first_instr.line());
      bb->AddInstr(instr);
    }
  }
  // get list or tuple ref
  Instr *i = graph_.GetCFG()->NewInstrNode(-1, loop_op_, loop_arg_, first_instr.line());
  bb->AddInstr(i);
  if (count == 0) {
    i->set_bci(first_instr.bci());
  }
  py::object value = py::int_(count);
  // subscript index
  i = graph_.GetCFG()->NewLoadInstrNode(-1, -1, first_instr.line(), value.ptr());
  bb->AddInstr(i);
  i = graph_.GetCFG()->NewInstrNode(-1, BINARY_SUBSCR, 0, first_instr.line());
  bb->AddInstr(i);
}

// while-do pattern loop
void LoopUnrolling::RemoveBackedge() {
  loop_->header()->SetJumpBB(nullptr);
  MS_EXCEPTION_IF_CHECK_FAIL(loop_->backedges().size() == 1, "backedges has only one block");
  Block *backedge = *loop_->backedges().begin();
  if (!backedge->instrs().empty() && &backedge->instrs().front() == &backedge->instrs().back()) {
    backedge->instrs().front().set_op(NOP);  // replace JUMP_ABSOLUTE
    backedge->instrs().front().set_arg(0);
  } else {
    backedge->RemoveInstr(&backedge->instrs().back());  // remove JUMP_ABSOLUTE
  }
  backedge->SetJumpBB(nullptr);
  backedge->SetFallBB(*loop_->exits().begin());
}

void LoopUnrolling::CopyAndInsertBB() {
  Block *exit = *loop_->exits().begin();
  MS_EXCEPTION_IF_CHECK_FAIL(exit->pred_bbs().size() == 1, "pred of exit has only succ block");
  Block *exit_pred = *exit->pred_bbs().begin();
  Block *header = loop_->header();
  Block *start = nullptr;
  Block *tail = nullptr;
  for (int i = 0; i < unrolling_count_; ++i) {
    if (i == 0) {
      AddLoopUnrollingInstr(header, i);
      continue;
    }
    std::map<int, Block *> bb_map = CopyBB();
    header = bb_map[loop_->header()->id()];
    AddLoopUnrollingInstr(header, i);
    if (i == 1) {
      start = bb_map[loop_->header()->id()];
      tail = bb_map[exit_pred->id()];
    }
    if (i > 1) {
      tail->SetFallBB(bb_map[loop_->header()->id()]);
    }
    tail = bb_map[exit_pred->id()];
    if (i == unrolling_count_ - 1) {
      exit_pred->SetFallBB(nullptr);
      exit_pred->SetFallBB(start);
      tail->SetFallBB(exit);
    }
  }
}

std::map<int, Block *> LoopUnrolling::CopyBB() {
  std::map<int, Block *> bb_map;
  for (Block *memb : loop_->loop_members()) {
    Block *new_bb = memb->Clone(graph_.GetCFG().get());
    bb_map.insert(std::make_pair(memb->id(), new_bb));
  }
  // link active bb and instr
  for (auto iter = graph_.GetCFG()->begin(loop_->header()); iter != graph_.GetCFG()->end(); ++iter) {
    Block *bb = *iter;
    if (loop_->loop_members().find(bb) == loop_->loop_members().cend()) {
      break;
    }
    Block *dst_bb = bb_map[bb->id()];
    if (dst_bb == nullptr) {
      continue;
    }
    if (bb->GetFallBB() != nullptr) {
      Block *dst_fall_bb = bb_map[bb->GetFallBB()->id()];
      dst_bb->SetFallBB(dst_fall_bb);
    }
    if (bb->GetJumpBB() != nullptr) {
      Block *dst_jump_bb = bb_map[bb->GetJumpBB()->id()];
      dst_bb->SetJumpBB(dst_jump_bb);
      // link instr jump
      dst_bb->instrs().back().set_extra_jump(&dst_jump_bb->instrs().front());
      dst_jump_bb->instrs().front().AddExtraPred(&dst_bb->instrs().back());
    }
  }
  return bb_map;
}

void LoopUnrolling::FixupInstr() {
  int head_bci = loop_->header()->instrs().front().bci();  // first instruction bci of header is computed
  int bci = head_bci;
  // fixup bci
  std::priority_queue<Block *, std::vector<Block *>, BBIdGreaterCmp> queue;
  queue.push(loop_->header());
  std::vector<bool> visited(graph_.GetCFG()->bb_pool().size(), false);
  // dfs search falled bb
  while (!queue.empty()) {
    Block *bb = queue.top();
    queue.pop();
    while (bb != nullptr) {
      if (visited[bb->id()]) {
        bb = bb->GetFallBB();
        continue;
      }
      visited[bb->id()] = true;
      for (auto &instr : bb->instrs()) {
        instr.set_bci(bci++);
      }
      if (bb->GetJumpBB() != nullptr && !visited[bb->GetJumpBB()->id()]) {
        queue.push(bb->GetJumpBB());
      }
      bb = bb->GetFallBB();
    }
  }
  // fixup jump arg
  for (Block *bb : *graph_.GetCFG()) {
    if (bb->GetJumpBB() == nullptr) {
      continue;
    }
    int jump_bci = bb->GetJumpBB()->instrs().front().bci();
    Instr &curr_instr = bb->instrs().back();
    int jump_arg = Utils::GetBranchDestArg(curr_instr.op(), jump_bci, curr_instr.bci());
    curr_instr.set_arg(jump_arg);
  }
  // fixup deaded bb bci, because BuildGraph traverse instructions in bci order
  for (const auto &bb : graph_.GetCFG()->bb_pool()) {
    if (bb && bb->is_dead() && !bb->instrs().empty() && bb->instrs().front().bci() > head_bci) {
      for (auto &instr : bb->instrs()) {
        instr.set_bci(bci++);
      }
    }
  }
}

std::string LoopUnrolling::DumpLoopUnrolling() {
  std::ostringstream os;
  os << "*** Dump info after loop unrolling on ["
     << py::str(reinterpret_cast<PyObject *>(graph_.GetCodeObj())).cast<std::string>() << "] ***\n";
  os << "loop unrolling reason: " << GetLoopUnrollingReasonDesc(res_) << '\n';
  if (loop_ != nullptr) {
    os << "loop header: " << loop_->header()->Dump(false);
  }
  if (unrolling_count_ > 0) {
    os << "loop count: " << unrolling_count_ << '\n';
  }
  os << '\n';
  return os.str();
}
}  // namespace graph
}  // namespace jit
}  // namespace mindspore
