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
#ifndef MINDSPORE_PI_JIT_GRAPH_CAPTURE_CFG_H
#define MINDSPORE_PI_JIT_GRAPH_CAPTURE_CFG_H

#include <memory>
#include <set>
#include <string>
#include <queue>
#include <vector>
#include "pipeline/jit/pi/pydef.h"
#include "pipeline/jit/pi/utils/ptr_list_ref.h"
#include "pybind11/pybind11.h"
#include "pipeline/jit/pi/graph_capture/local_liveness.h"

namespace mindspore {
namespace pijit {

namespace py = pybind11;

class AbstractNode;
class Graph;

class Instr : public PtrListNodeBase<Instr> {
 public:
  Instr(const Instr &) = delete;
  Instr &operator=(const Instr &) = delete;
  Instr(int op, int arg, int bci = -1, int line = -1) : bci_(bci), op_(op), arg_(arg), line_(line) {}
  Instr(int op, int arg, const std::string &name) : Instr(op, arg) { name_ = name; }
  Instr(int op, int arg, const py::object &cnst) : Instr(op, arg) { cnst_ = cnst; }
  explicit Instr(int op) : Instr(op, 0) {}
  virtual ~Instr() = default;

  int bci() const { return bci_; }
  void set_bci(int i) { bci_ = i; }
  int op() const { return op_; }
  void set_op(int op) { op_ = op; }
  int arg() const { return arg_; }
  void set_arg(int arg) { arg_ = arg; }
  int line() const { return line_; }
  void set_line(int l) { line_ = l; }
  bool is_fall() const { return is_fall_; }
  void set_is_fall(int is_fall) { is_fall_ = is_fall; }
  const std::vector<Instr *> &extra_preds() const { return extra_preds_; }
  std::vector<Instr *> &extra_preds() { return extra_preds_; }
  Instr *extra_jump() const { return extra_jump_; }
  void set_extra_jump(Instr *j) { extra_jump_ = j; }

  const std::string &name() const { return name_; }
  void set_name(const std::string &n) { name_ = n; }
  const py::object &cnst() const { return cnst_; }
  void set_cnst(PyObject *cnst) { cnst_ = py::reinterpret_borrow<py::object>(cnst); }
  void set_cnst(const py::object &cnst) { cnst_ = cnst; }

  void AddExtraPred(Instr *instr) { extra_preds_.push_back(instr); }
  std::string Dump(const std::string &prefix = "") const;
  std::string ToString() const;

 private:
  int bci_;
  int op_;
  int arg_;
  int line_;
  std::string name_;
  py::object cnst_;

  bool is_fall_ = true;
  std::vector<Instr *> extra_preds_;
  Instr *extra_jump_ = nullptr;
};

class Block;
struct BBIdCmp {
  bool operator()(const Block *lhs, const Block *rhs) const;
};

struct BBIdGreaterCmp {
  bool operator()(const Block *lhs, const Block *rhs) const;
};

using UniqueInstr = std::unique_ptr<Instr>;
using Instrs = PtrListRef<Instr>;
class CFG;
class Block {
 public:
  enum TrackResult {
    kNotTrack,
    kTrackHasTensor,
    kTrackHasOpsPrimitive,
    kTrackBreak,
    kHasGlobalSideEffect,
    kHasAttrSideEffect,
    kHasClosureSideEffect,
  };
  Block() = default;
  ~Block() = default;
  uint32_t id() const { return id_; }
  void set_id(uint32_t arg) { id_ = arg; }
  Instrs &instrs() { return instrs_; }
  const Instrs &instrs() const { return instrs_; }
  void AddInstr(Instr *i) { instrs_.push_back(i); }
  const std::set<Block *, BBIdCmp> &pred_bbs() const { return pred_bbs_; }
  std::set<Block *, BBIdCmp> &pred_bbs() { return pred_bbs_; }
  const std::set<Block *, BBIdCmp> &succ_bbs() const { return succ_bbs_; }
  std::set<Block *, BBIdCmp> &succ_bbs() { return succ_bbs_; }
  void set_is_loop_head(bool flag) { is_loop_head_ = flag; }
  bool is_loop_head() const { return is_loop_head_; }
  void set_is_loop_body(bool flag) { is_loop_body_ = flag; }
  bool is_loop_body() const { return is_loop_body_; }
  bool is_dead() const { return is_dead_; }
  void set_is_dead(bool flag) { is_dead_ = flag; }

  std::string Dump(bool dump_instr = true) const;

  int begin_ci() const { return begin_; }
  int end_ci() const { return end_; }
  void set_begin_ci(int i) { begin_ = i; }
  void set_end_ci(int i) { end_ = i; }
  Block *GetFallBB() const { return fall_bb_; }
  Block *GetJumpBB() const { return jump_bb_; }
  void SetFallBB(Block *arg);
  void SetJumpBB(Block *arg);
  void RemoveInstr(Instr *instr);
  void RemoveInstrs();

  bool IsTrackBreak() const { return track_result_ & (1 << kTrackBreak); }
  bool HasPrimitive() const { return track_result_ & (1 << kTrackHasOpsPrimitive); }
  bool HasTensor() const { return track_result_ & (1 << kTrackHasTensor); }
  bool HasUnresolvedSideEffect() const { return track_result_ & (1 << kHasGlobalSideEffect); }
  bool HasAttrSideEffect() const { return track_result_ & (1 << kHasAttrSideEffect); }
  bool HasClosureSideEffect() const { return track_result_ & (1 << kHasClosureSideEffect); }
  void SetTrackResult(TrackResult r) { track_result_ = (track_result_ & ~(1 << kNotTrack)) | (1 << r); }

  void AddSuccBB(Block *bb);
  bool RemoveEdge(Block *bb);
  void ClearOutEdges();

  Block *Clone(CFG *cfg);

 private:
  uint32_t id_;  // start from 0
  int begin_;
  int end_;
  std::set<Block *, BBIdCmp> pred_bbs_;
  std::set<Block *, BBIdCmp> succ_bbs_;  // include fall_bb_ and jump_bb_
  Block *fall_bb_ = nullptr;
  Block *jump_bb_ = nullptr;

  bool is_loop_body_ = false;
  bool is_loop_head_ = false;
  bool is_dead_ = true;

  // TODO(chaiyouheng): remove
  Instrs instrs_;
  int track_result_ = (1 << kNotTrack);
};

class CFG {
 public:
  explicit CFG(PyCodeObject *co) : pycode_(co), nlocals_(0) {}

  // BFS Iterator
  class BBIterator {
   public:
    BBIterator() = default;
    explicit BBIterator(const CFG *c) : visit_(c->bb_pool().size(), false) {
      q_.push(c->GetFirstBB());
      visit_[c->GetFirstBB()->id()] = true;
    }

    BBIterator(const CFG *c, Block *bb) : visit_(c->bb_pool().size(), false) {
      q_.push(bb);
      visit_[bb->id()] = true;
    }

    const auto &GetVisitMap() const { return visit_; }
    Block *operator*() const { return q_.front(); }
    bool operator!=(const BBIterator &end) const { return !q_.empty(); }
    BBIterator &operator++();

    std::queue<Block *> q_;
    std::vector<bool> visit_;
  };

  BBIterator begin() const { return BBIterator(this); }
  BBIterator begin(Block *start) const { return BBIterator(this, start); }
  BBIterator end() const { return BBIterator(); }

  const std::vector<std::unique_ptr<Block>> &bb_pool() const { return bb_pool_; }
  const std::vector<std::unique_ptr<Instr>> &instr_pool() const { return instrs_; }
  const std::unique_ptr<Liveness> &liveness() const { return liveness_; }
  std::vector<std::unique_ptr<Instr>> &instr_pool() { return instrs_; }
  std::vector<std::unique_ptr<Block>> &bb_pool() { return bb_pool_; }
  std::unique_ptr<Liveness> &liveness() { return liveness_; }
  PyCodeObject *GetCodeObject() const { return pycode_; }
  int GetLocalCount() const { return nlocals_; }
  void SetLocalCount(int n) { nlocals_ = n; }
  std::string ToString() const { return DumpBBs(); }

  const Liveness *GetLiveness();

  void GenerateCFG();
  void MarkDeadBB();

  // clear dead bb's edges
  void ClearDeadBBEdges();

  Block *GetFirstBB() const { return bb_pool_.size() ? bb_pool_[0].get() : nullptr; }
  Block *GetBlockByBci(int) const;

  std::string DumpBBs(std::string phase = "") const;
  void DumpCFGGraph();
  void DumpCFGGraph(std::ofstream &file);
  void DumpCFGGraphForBB(std::ofstream &file, const Block &bb) const;
  void DumpCFGGraphForEdge(std::ofstream &file);

  Block *NewBBAppend();
  Instr *NewInstrNode(int bci, int op, int arg, int line);
  Instr *NewInstrNode(const Instr &instr);
  Instr *NewLoadInstrNode(int bci, int arg, int line, PyObject *cnst);
  std::unique_ptr<CFG> Clone();

 private:
  void BuildInst();
  void BuildBB();
  bool BuildCFG();

  PyCodeObject *const pycode_;
  std::vector<std::unique_ptr<Instr>> instrs_;
  std::vector<std::unique_ptr<Block>> bb_pool_;
  std::unique_ptr<Liveness> liveness_;
  int nlocals_;
  bool is_generated_ = false;
};
}  // namespace pijit
}  // namespace mindspore

#endif  // MINDSPORE_PI_JIT_GRAPH_CAPTURE_CFG_H
