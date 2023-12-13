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
#ifndef MINDSPORE_CCSRC_PIPELINE_GRAPH_JIT_GRAPH_CAPTURE_CFG_H
#define MINDSPORE_CCSRC_PIPELINE_GRAPH_JIT_GRAPH_CAPTURE_CFG_H

#include <memory>
#include <set>
#include <string>
#include <queue>
#include <vector>
#include "pipeline/jit/pi/pydef.h"
#include "pipeline/jit/pi/utils/allocator.h"
#include "pipeline/jit/pi/utils/ptr_list_ref.h"

namespace mindspore {
namespace jit {
namespace graph {
class AbstractNode;
class Graph;
class GraphJitConfig;

static const int PY_BCSIZE = sizeof(_Py_CODEUNIT);
class Instr : public PtrListNodeBase<Instr> {
 public:
  Instr(int bci, int op, int arg, int line) : bci_(bci), op_(op), arg_(arg), line_(line) {}
  Instr &operator=(const Instr &o) {
    this->set_bci(o.bci());
    this->set_op(o.op());
    this->set_arg(o.arg());
    this->set_line(o.line());
    this->set_name(o.name());
    this->set_cnst(o.cnst());
    return *this;
  }
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

  virtual std::string name() const { return std::string(); }
  virtual void set_name(const std::string &n) {}
  virtual PyObject *cnst() const { return nullptr; }
  virtual void set_cnst(PyObject *cnst) {}

  void AddExtraPred(Instr *instr) { extra_preds_.push_back(instr); }
  virtual std::string Dump(const std::string &prefix = "") const;

 private:
  int bci_;
  int op_;
  int arg_;
  int line_;
  bool is_fall_ = true;
  std::vector<Instr *> extra_preds_;
  Instr *extra_jump_ = nullptr;
};

class LoadConstInstr : public Instr {
 public:
  LoadConstInstr(int bci, int op, int arg, int line) : Instr(bci, LOAD_CONST, arg, line) {}
  ~LoadConstInstr() {}
  PyObject *cnst() const override { return cnst_; }
  void set_cnst(PyObject *cnst) override { cnst_ = cnst; }
  std::string Dump(const std::string &prefix = "") const override;

 private:
  PyObject *cnst_;
};

class NameRelatedInstr : public Instr {
 public:
  NameRelatedInstr(int bci, int op, int arg, int line) : Instr(bci, op, arg, line) {}
  ~NameRelatedInstr() {}
  std::string name() const override { return name_; }
  void set_name(const std::string &n) override { name_ = n; }
  std::string Dump(const std::string &prefix = "") const override;

 private:
  std::string name_;
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
  bool is_dead() const { return is_dead_; }
  void set_is_dead(bool flag) { is_dead_ = flag; }

  std::string Dump(bool dump_instr = true) const;

  int begin_ci() const { return instrs_.front().bci(); }
  int end_ci() const { return instrs_.back().bci() + 1; }
  void SetGraph(Graph *g) { graph_ = g; }
  Graph *GetGraph() const { return graph_; }
  Block *GetFallBB() const { return fall_bb_; }
  Block *GetJumpBB() const { return jump_bb_; }
  void SetFallBB(Block *arg);
  void SetJumpBB(Block *arg);
  void RemoveInstr(Instr *instr);
  void RemoveInstrs();
  const std::vector<AbstractNode *> &GetNodes() const { return nodes_; }
  std::vector<AbstractNode *> &GetNodes() { return nodes_; }
  void EraseNodesRange(int l, int r) { nodes_.erase(nodes_.begin() + l, nodes_.begin() + r); }

  bool IsTrackBreak() const { return track_result_ & (1 << kTrackBreak); }
  bool HasPrimitive() const { return track_result_ & (1 << kTrackHasOpsPrimitive); }
  bool HasTensor() const { return track_result_ & (1 << kTrackHasTensor); }
  bool HasUnresolvedSideEffect() const { return track_result_ & (1 << kHasGlobalSideEffect); }
  bool HasAttrSideEffect() const { return track_result_ & (1 << kHasAttrSideEffect); }
  bool HasClosureSideEffect() const { return track_result_ & (1 << kHasClosureSideEffect); }
  void SetTrackResult(TrackResult r) { track_result_ = (track_result_ & ~(1 << kNotTrack)) | (1 << r); }

  void AddNode(AbstractNode *n);
  void ClearTrackInfo();

  void AddSuccBB(Block *bb);
  bool RemoveEdge(Block *bb);
  void ClearOutEdges();

  Block *Clone(CFG *cfg);

 private:
  uint32_t id_;  // start from 0
  Instrs instrs_;
  std::set<Block *, BBIdCmp> pred_bbs_;
  std::set<Block *, BBIdCmp> succ_bbs_;  // include fall_bb_ and jump_bb_
  Block *fall_bb_ = nullptr;
  Block *jump_bb_ = nullptr;

  Graph *graph_ = nullptr;
  std::vector<AbstractNode *> nodes_;  // tracked instructions. transform to use-def node
  int track_result_ = (1 << kNotTrack);

  bool is_loop_head_ = false;
  bool is_dead_ = true;
};

class CFG {
 public:
  CFG(PyCodeObject *co, Allocator *alloc, const GraphJitConfig &conf)
      : pycode_(co), bytecode_(nullptr), alloc_(*alloc), conf_(conf) {}
  ~CFG() { bb_pool_.clear(); }

  // BFS Iterator
  class BBIterator {
   public:
    BBIterator() = default;
    explicit BBIterator(CFG *c) : visit_(c->bb_pool().size(), false) {
      q_.push(c->GetFirstBB());
      visit_[c->GetFirstBB()->id()] = true;
    }

    BBIterator(CFG *c, Block *bb) : visit_(c->bb_pool().size(), false) {
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

  BBIterator begin() { return BBIterator(this); }
  BBIterator begin(Block *start) { return BBIterator(this, start); }
  BBIterator end() { return BBIterator(); }

  Allocator &alloc() { return alloc_; }
  const std::vector<std::unique_ptr<Block>> &bb_pool() const { return bb_pool_; }
  const std::vector<Instr *> &instr_pool() const { return alloc_.instr_pool(); }

  int GetBytecodeSize() const { return (PyBytes_GET_SIZE(pycode_->co_code)) / PY_BCSIZE; }
  void GenerateCFG();
  void MarkDeadBB();

  // clear dead bb's edges
  void ClearDeadBBEdges();

  const GraphJitConfig &Config() const { return conf_; }
  Block *GetFirstBB() const { return bb_pool_.size() ? bb_pool_[0].get() : nullptr; }

  std::string DumpBBs(std::string phase = "") const;
  void DumpCFGGraph();
  void DumpCFGGraph(std::ofstream &file);
  void DumpCFGGraphForBB(std::ofstream &file, const Block &bb) const;
  void DumpCFGGraphForEdge(std::ofstream &file);

  Block *NewBBAppend();
  Instr *NewInstrNode(int bci, int op, int arg, int line);
  Instr *NewInstrNode(const Instr &instr);
  LoadConstInstr *NewLoadInstrNode(int bci, int arg, int line, PyObject *cnst);
  std::unique_ptr<CFG> Clone(Allocator *new_alloc);

 private:
  void BuildInst();
  void BuildBB();
  bool BuildCFG();

  bool is_generated_ = false;
  PyCodeObject *pycode_;
  const _Py_CODEUNIT *bytecode_;
  Allocator &alloc_;
  const GraphJitConfig &conf_;
  std::vector<std::unique_ptr<Block>> bb_pool_;
};
}  // namespace graph
}  // namespace jit
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PIPELINE_GRAPH_JIT_GRAPH_CAPTURE_CFG_H
