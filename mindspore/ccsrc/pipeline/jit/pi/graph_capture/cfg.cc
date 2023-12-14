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
#include "pipeline/jit/pi/graph_capture/cfg.h"
#include <fstream>
#include <map>
#include <string>
#include <utility>
#include <vector>
#include "pipeline/jit/pi/graph_capture/node.h"
#include "pipeline/jit/pi/pi_jit_config.h"
#include "pipeline/jit/pi/utils/utils.h"

namespace mindspore {
namespace jit {
namespace graph {
std::string Instr::Dump(const std::string &prefix) const {
  std::stringstream os;
  os << prefix << " " << bci_ << ' ' << Utils::GetOpName(op_) << ' ' << arg_;
  return os.str();
}

std::string LoadConstInstr::Dump(const std::string &prefix) const {
  return this->Instr::Dump(prefix) + " " + std::string(py::str(cnst()));
}

std::string NameRelatedInstr::Dump(const std::string &prefix) const { return this->Instr::Dump(prefix) + " " + name(); }

void Block::AddNode(AbstractNode *n) {
  nodes_.push_back(n);
  n->SetGraph(this->GetGraph());
  n->SetBlock(this);
}

void Block::ClearTrackInfo() {
  nodes_.clear();  // tracked instructions. transform to use-def node
  track_result_ = (1 << kNotTrack);
}

void Block::AddSuccBB(Block *bb) {
  succ_bbs_.insert(bb);
  bb->pred_bbs_.insert(this);
}

void Block::SetFallBB(Block *arg) {
  if (arg != nullptr) {
    fall_bb_ = arg;
    AddSuccBB(arg);
  } else if (fall_bb_ != nullptr) {
    // remove fall_bb_
    succ_bbs_.erase(fall_bb_);
    fall_bb_->pred_bbs().erase(this);
    fall_bb_ = nullptr;
  }
}

void Block::SetJumpBB(Block *arg) {
  if (arg != nullptr) {
    jump_bb_ = arg;
    AddSuccBB(arg);
  } else if (jump_bb_ != nullptr) {
    // remove jump_bb_
    succ_bbs_.erase(jump_bb_);
    jump_bb_->pred_bbs().erase(this);
    jump_bb_ = nullptr;
  }
}

void Block::RemoveInstr(Instr *instr) {
  Instr *jump = instr->extra_jump();
  if (jump != nullptr) {
    auto &v = jump->extra_preds();
    v.erase(std::remove(v.begin(), v.end(), jump), v.end());
    jump->set_extra_jump(nullptr);
  }
  for (Instr *pred : instr->extra_preds()) {
    pred->set_extra_jump(nullptr);
  }
  instr->extra_preds().clear();
  instrs_.erase(instr);
}

void Block::RemoveInstrs() {
  if (instrs_.empty()) {
    return;
  }
  RemoveInstr(&instrs_.front());
  if (instrs_.empty()) {
    return;
  }
  RemoveInstr(&instrs_.back());
  instrs_.clear();
}

bool Block::RemoveEdge(Block *bb) {
  bb->pred_bbs_.erase(this);
  jump_bb_ = jump_bb_ == bb ? nullptr : jump_bb_;
  fall_bb_ = fall_bb_ == bb ? nullptr : fall_bb_;
  return succ_bbs_.erase(bb);
}

void Block::ClearOutEdges() {
  while (!succ_bbs_.empty()) {
    RemoveEdge(*succ_bbs_.begin());
  }
}

std::string Block::Dump(bool dump_instr) const {
  std::stringstream os;
  os << "Block [" << (begin_ci() * PY_BCSIZE) << ',' << (end_ci() * PY_BCSIZE) << "), (id=" << id_
     << ", is_dead=" << is_dead_ << ", is_loop_head=" << is_loop_head_ << ", preds={";
  for (Block *bb : pred_bbs_) {
    os << bb->id() << " ";
  }
  os << "}, succs={";
  for (Block *bb : succ_bbs_) {
    if (bb == jump_bb_) {
      os << "jump:";
    } else {
      os << "fall:";
    }
    os << bb->id() << " ";
  }
  os << "}";
  if (IsTrackBreak()) {
    os << " Break";
  }
  if (HasPrimitive()) {
    os << " HasPrimitive";
  }
  if (HasTensor()) {
    os << " HasTensor";
  }
  if (HasAttrSideEffect()) {
    os << " HasAttrSideEffect";
  }
  os << ")";
  if (!dump_instr) {
    return os.str();
  }
  os << "\n";
  for (const auto &instr : instrs_) {
    os << instr.Dump("    ") << "\n";
  }
  return os.str();
}

Block *Block::Clone(CFG *cfg) {
  Block *new_bb = cfg->NewBBAppend();
  new_bb->set_is_dead(is_dead_);
  new_bb->set_is_loop_head(is_loop_head_);
  // clone instr list
  for (const auto &instr : instrs_) {
    Instr *new_instr = cfg->NewInstrNode(instr);
    new_bb->AddInstr(new_instr);
  }
  // move node list to new bb
  for (AbstractNode *node : nodes_) {
    new_bb->AddNode(node);
  }
  return new_bb;
}

bool BBIdCmp::operator()(const Block *lhs, const Block *rhs) const { return (lhs->id() < rhs->id()); }

bool BBIdGreaterCmp::operator()(const Block *lhs, const Block *rhs) const { return (lhs->id() > rhs->id()); }

Block *CFG::NewBBAppend() {
  std::unique_ptr<Block> bb_node = std::make_unique<Block>();
  bb_node->set_id(bb_pool_.size());
  bb_pool_.push_back(std::move(bb_node));
  Block *bb = bb_pool_.back().get();
  return bb;
}

Instr *CFG::NewInstrNode(int bci, int op, int arg, int line) {
  Instr *i;
  if (op == LOAD_CONST) {
    PyObject *cnst = PyTuple_GET_ITEM(pycode_->co_consts, arg);
    i = NewLoadInstrNode(bci, arg, line, cnst);
  } else if (Utils::IsNameRelated(op)) {
    NameRelatedInstr *t = alloc_.NewInstr<NameRelatedInstr>(bci, op, arg, line);
    t->set_name(PyUnicode_AsUTF8(PyTuple_GET_ITEM(pycode_->co_names, arg)));
    i = t;
  } else {
    i = alloc_.NewInstr<Instr>(bci, op, arg, line);
  }
  return i;
}

LoadConstInstr *CFG::NewLoadInstrNode(int bci, int arg, int line, PyObject *cnst) {
  LoadConstInstr *l = alloc_.NewInstr<LoadConstInstr>(bci, LOAD_CONST, arg, line);
  l->set_cnst(cnst);
  return l;
}

Instr *CFG::NewInstrNode(const Instr &instr) {
  if (instr.op() == LOAD_CONST) {
    return alloc_.NewInstr<LoadConstInstr>(static_cast<const LoadConstInstr &>(instr));
  }
  if (Utils::IsNameRelated(instr.op())) {
    return alloc_.NewInstr<NameRelatedInstr>(static_cast<const NameRelatedInstr &>(instr));
  }
  return alloc_.NewInstr<Instr>(instr);
}

void CFG::GenerateCFG() {
  MS_EXCEPTION_IF_CHECK_FAIL(pycode_, "shouldn't use this function to generate empty cfg");
  bytecode_ = reinterpret_cast<_Py_CODEUNIT *>(PyBytes_AS_STRING(pycode_->co_code));
  if (!is_generated_) {
    is_generated_ = true;
    BuildInst();
    BuildBB();
    BuildCFG();
    MarkDeadBB();
  }
  if (conf_.GetBoolConfig(GraphJitConfig::kPrintBB)) {
    GRAPH_JIT_LOG_F("%s\n\n", DumpBBs().c_str());
  }
  if (conf_.GetBoolConfig(GraphJitConfig::kPrintCFG)) {
    DumpCFGGraph();
  }
}

void CFG::BuildInst() {
  int size = GetBytecodeSize();
  int exarg = 0;
  std::map<int, std::vector<Instr *>> succ_jump;
  for (int bci = 0; bci < size; ++bci) {
    int opcode = _Py_OPCODE(bytecode_[bci]);
    int oparg = (exarg << 8) | _Py_OPARG(bytecode_[bci]);
    exarg = (opcode == EXTENDED_ARG) ? oparg : 0;
    int line = PyCode_Addr2Line(pycode_, PY_BCSIZE * bci);
    Instr *instr = NewInstrNode(bci, opcode, oparg, line);
    instr->set_is_fall(!Utils::IsNonFall(opcode));
    // link instr jump relation
    if (Utils::IsRelativeJump(opcode) || Utils::IsAbsoluteJump(opcode)) {
      int dest = Utils::GetBranchDestIndex(opcode, oparg, bci);
      if (dest < bci) {
        Instr *succ = alloc_.instr_pool()[dest];
        succ->AddExtraPred(instr);
        instr->set_extra_jump(succ);
      } else {
        // record succ jump
        succ_jump[dest].push_back(instr);
      }
    }
    auto it = succ_jump.find(bci);
    if (it != succ_jump.cend()) {
      for (Instr *pred : it->second) {
        instr->AddExtraPred(pred);
        MS_EXCEPTION_IF_CHECK_FAIL(pred->extra_jump() == nullptr, "Python bytecode has at most one jump branch");
        pred->set_extra_jump(instr);
      }
    }
  }
}

void CFG::BuildBB() {
  Block *curr_bb = nullptr;
  for (auto *instr : alloc_.instr_pool()) {
    if (instr == nullptr) {
      continue;
    }
    // check start of BB
    if (curr_bb == nullptr || !instr->extra_preds().empty()) {
      curr_bb = NewBBAppend();
    }
    curr_bb->AddInstr(instr);
    // check end of BB
    if (!instr->is_fall() || instr->extra_jump() != nullptr) {
      curr_bb = nullptr;
    }
  }
}

bool CFG::BuildCFG() {
  // build target map
  std::map<const Instr *, Block *> target_instr_bb_map;
  for (const auto &unique_bb : bb_pool_) {
    Block *bb = unique_bb.get();
    if (bb->instrs().empty()) {
      continue;
    }
    const Instr *instr_head = &(bb->instrs().front());
    target_instr_bb_map[instr_head] = bb;
  }
  // link
  for (size_t i = 0; i < bb_pool_.size(); ++i) {
    Block *bb = bb_pool_[i].get();
    const Instr *instr_tail = &(bb->instrs().back());
    if (instr_tail->is_fall()) {
      if (i + 1 >= bb_pool_.size()) {
        MS_EXCEPTION_IF_CHECK_FAIL(false, "Method without return");
        return false;
      }
      Block *bb_next = bb_pool_[i + 1].get();
      bb->SetFallBB(bb_next);
    }
    if (instr_tail->extra_jump() != nullptr) {
      Instr *instr = instr_tail->extra_jump();
      const auto &it_bb = target_instr_bb_map.find(instr);
      MS_EXCEPTION_IF_CHECK_FAIL(it_bb != target_instr_bb_map.cend(), "Target BB is not found");
      Block *bb_next = it_bb->second;
      bb->SetJumpBB(bb_next);
      // If jumping to the previous bb, this previous bb is loop head
      if (instr_tail->extra_jump()->bci() < instr_tail->bci()) {
        bb_next->set_is_loop_head(true);
      }
    }
  }
  return true;
}

void CFG::MarkDeadBB() {
  if (bb_pool_.empty()) {
    return;
  }
  auto iter = this->begin();
  auto end = this->end();
  for (; iter != end; ++iter) {
  }
  std::vector<bool> map = iter.GetVisitMap();
  for (size_t id = 0; id < map.size(); ++id) {
    bb_pool_[id]->set_is_dead(!map[id]);
  }
}

// Simplified cfg
void CFG::ClearDeadBBEdges() {
  MarkDeadBB();
  for (auto &i : bb_pool_) {
    if (i->is_dead()) {
      i->ClearOutEdges();
    }
  }
}

std::unique_ptr<CFG> CFG::Clone(Allocator *new_alloc) {
  std::unique_ptr<CFG> new_cfg = std::make_unique<CFG>(pycode_, new_alloc, conf_);
  if (bb_pool_.empty()) {
    return new_cfg;
  }
  for (const auto &bb : bb_pool_) {
    (void)bb->Clone(new_cfg.get());
  }
  // link active bb and instr
  for (Block *bb : *this) {
    Block *dst_bb = new_cfg->bb_pool()[bb->id()].get();
    if (bb->GetFallBB() != nullptr) {
      Block *dst_fall_bb = new_cfg->bb_pool()[bb->GetFallBB()->id()].get();
      dst_bb->SetFallBB(dst_fall_bb);
    }
    if (bb->GetJumpBB() != nullptr) {
      Block *dst_jump_bb = new_cfg->bb_pool()[bb->GetJumpBB()->id()].get();
      dst_bb->SetJumpBB(dst_jump_bb);
      // link instr jump
      dst_bb->instrs().back().set_extra_jump(&dst_jump_bb->instrs().front());
      dst_jump_bb->instrs().front().AddExtraPred(&dst_bb->instrs().back());
    }
  }
  return new_cfg;
}

std::string CFG::DumpBBs(std::string phase) const {
  std::ostringstream os;
  os << "*** Dump BB " << phase << "on [" << py::str(reinterpret_cast<PyObject *>(pycode_)).cast<std::string>()
     << "] ***\n";
  for (const auto &bb : bb_pool_) {
    os << bb->Dump();
  }
  return os.str();
}

void CFG::DumpCFGGraph() {
  std::string file_name = Utils::GetPyName(pycode_->co_name);
  file_name = file_name + ".dot";
  std::ofstream file(file_name);
  MS_EXCEPTION_IF_CHECK_FAIL(file.is_open(), "Failed to open General CFG Graph FileName:" + file_name);
  file << "digraph {\n";
  file << "  label=\"" << file_name << "\"\n";
  file << "  labelloc=t\n";
  DumpCFGGraph(file);
  file.close();
}

void CFG::DumpCFGGraph(std::ofstream &file) {
  for (const auto &bb : bb_pool_) {
    DumpCFGGraphForBB(file, *bb);
  }
  DumpCFGGraphForEdge(file);
  file << "}\n";
}

void CFG::DumpCFGGraphForBB(std::ofstream &file, const Block &bb) const {
  file << "  BB" << bb.id() << " [shape=record,label=\"{\n";
  for (const auto &instr : bb.instrs()) {
    file << "      <instr" << instr.bci() << "> " << instr.Dump();
    if (&instr == &bb.instrs().back()) {
      file << "\n";
      break;
    } else {
      file << " |\n";
    }
  }
  file << "    }\"];\n";
}

void CFG::DumpCFGGraphForEdge(std::ofstream &file) {
  file << "  subgraph cfg_edges {\n";
  file << "    edge [color=\"#000000\",weight=0.3,len=3];\n";
  for (const auto &bb : bb_pool_) {
    const Instr &instrS = bb->instrs().back();
    for (Block *bb_next : bb->succ_bbs()) {
      const Instr &instrE = bb_next->instrs().front();
      file << "    BB" << bb->id() << ":instr" << instrS.bci() << " -> ";
      file << "BB" << bb_next->id() << ":instr" << instrE.bci() << "\n";
    }
  }
  file << "  }\n";
}

CFG::BBIterator &CFG::BBIterator::operator++() {
  if (q_.empty()) {
    return *this;
  }
  Block *bb = q_.front();
  q_.pop();
  for (Block *bb_next : bb->succ_bbs()) {
    if (visit_[bb_next->id()]) {
      continue;
    }
    q_.push(bb_next);
    visit_[bb_next->id()] = true;
  }
  return *this;
}
}  // namespace graph
}  // namespace jit
}  // namespace mindspore
