/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd

 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at

 * http://www.apache.org/licenses/LICENSE-2.0

 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*/

#ifndef MINDSPORE_CCSRC_BACKEND_COMMON_SOMAS_SOMAS_SOLVER_ALG_H_
#define MINDSPORE_CCSRC_BACKEND_COMMON_SOMAS_SOMAS_SOLVER_ALG_H_

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstring>
#include <list>
#include <memory>
#include <numeric>
#include <set>
#include <stack>
#include <utility>
#include <vector>

#include "utils/hash_map.h"
#include "backend/common/somas/somas_solver_pre.h"
#include "utils/ms_context.h"

using std::pair;
using std::set;
using std::stack;
using std::vector;

namespace mindspore {
namespace somas {
constexpr auto kDefaultAlignmentSize = 512;

class Interval {
 public:
  Interval() : m_a_(0), m_b_(0) {}
  explicit Interval(const SomasSolverTensorDescPtr &t) : m_a_(t->offset_), m_b_(t->offset_ + t->size_) {}
  Interval(const size_t &a, const size_t &b) : m_a_(a), m_b_(b) {}
  Interval(const Interval &in) {
    if (this == &in) {
      return;
    }
    m_a_ = in.m_a_;
    m_b_ = in.m_b_;
  }
  ~Interval() = default;

  bool intersect(const Interval &i) const { return (in(i.m_a_) || in(i.m_b_)); }
  bool in(const size_t &a) const { return ((a > m_a_) && (a < m_b_)); }
  Interval intersection(const Interval &i) {
    if (m_a_ < i.m_a_) {
      return Interval(m_a_, i.m_b_);
    } else {
      return Interval(i.m_a_, m_b_);
    }
  }
  void merge(const Interval &i) {
    m_a_ = std::min(m_a_, i.m_a_);
    m_b_ = std::max(m_b_, i.m_b_);
  }
  size_t &lb() { return m_a_; }
  size_t &ub() { return m_b_; }
  bool contains(size_t width) const { return (m_b_ - m_a_) >= width; }
  bool contains(const Interval &a) const { return ((a.m_a_ >= m_a_) && (a.m_b_ <= m_b_)); }
  Interval &operator=(const Interval &in) {
    if (this == &in) {
      return *this;
    }
    m_a_ = in.m_a_;
    m_b_ = in.m_b_;
    return *this;
  }

 private:
  size_t m_a_;
  size_t m_b_;
};

class BlockTensor {
 public:
  SomasSolverTensorDescPtr m_start_tensor_;
  mindspore::HashMap<
    uint32_t, std::set<pair<size_t, size_t>, bool (*)(const pair<size_t, size_t> &, const pair<size_t, size_t> &)>>
    offsets_candidates_;
  uint32_t m_current_sol_;
  bool m_bre_allocate_;
  mindspore::HashMap<uint32_t, size_t> offsets_;
  size_t m_size_;
  BlockTensor()
      : m_start_tensor_(nullptr),
        offsets_candidates_(),
        m_current_sol_(0),
        m_bre_allocate_(true),
        offsets_(),
        m_size_(0) {}
  ~BlockTensor() = default;
  BlockTensor(const BlockTensor &bt) {
    if (this == &bt) {
      return;
    }
    m_bre_allocate_ = bt.m_bre_allocate_;
    m_current_sol_ = 0;
    m_start_tensor_ = bt.m_start_tensor_;
    offsets_candidates_ = bt.offsets_candidates_;
    offsets_ = bt.offsets_;
    m_size_ = bt.m_size_;
  }

  BlockTensor &operator=(const BlockTensor &bt) {
    if (this == &bt) {
      return *this;
    }
    m_bre_allocate_ = bt.m_bre_allocate_;
    m_current_sol_ = 0;
    m_start_tensor_ = bt.m_start_tensor_;
    offsets_candidates_ = bt.offsets_candidates_;
    offsets_ = bt.offsets_;
    m_size_ = bt.m_size_;
    return *this;
  }
  bool Alone() const { return ((nullptr == m_start_tensor_->right_) && (nullptr == m_start_tensor_->left_)); }
};

struct AllocatedTensorInfo {
  size_t index_;
  size_t size_;
  size_t offset_;
  explicit AllocatedTensorInfo(const SomasSolverTensorDescPtr &tensor)
      : index_(tensor->index_), size_(tensor->size_), offset_(tensor->offset_) {}
};

class FootPrint : public std::enable_shared_from_this<FootPrint> {
 public:
  uint32_t m_solId_;

  FootPrint()
      : m_solId_(0),
        m_foot_print_next_(nullptr),
        m_offset_(0),
        m_starts_({}),
        m_alignment_(0),
        m_branching_strategy_(0),
        m_algorithm_(0) {}
  ~FootPrint() = default;
  void setAlignment(const size_t a) { m_alignment_ = a; }
  void setBranchingStrategy(uint32_t bs) { m_branching_strategy_ = bs; }
  void setCurrentSol(uint32_t solId) { m_solId_ = solId; }
  void setAlgorithm(uint32_t algorithm) { m_algorithm_ = algorithm; }
  void addElem(BlockTensor *block, const size_t &offset);
  void addTensorsInfo(BlockTensor *elemIndex);
  std::shared_ptr<FootPrint> &Next() { return m_foot_print_next_; }
  vector<BlockTensor *> &getStarts() { return m_starts_; }
  void Destroy();
  const size_t getOffset() const { return m_offset_; }
  void setOffset(const size_t &offset) { m_offset_ = offset; }
  bool findOffset(const std::vector<DynamicBitSet> *constraints, const BlockTensor &block, size_t *offset);
  bool findFirst(vector<Interval> *interval_v, const BlockTensor &block, size_t *offset);
  size_t Result();
  void printStats();

 private:
  std::shared_ptr<FootPrint> m_foot_print_next_;
  size_t m_offset_;
  vector<BlockTensor *> m_starts_;
  vector<AllocatedTensorInfo> m_tensors_info_;
  size_t m_alignment_;
  uint32_t m_branching_strategy_;
  uint32_t m_algorithm_;
};

class FastHeuristic {
 public:
  FastHeuristic() : m_alignment_(kDefaultAlignmentSize), m_tensors_allocated_(0) {}
  ~FastHeuristic() = default;

  void setAlignment(const size_t &a) { m_alignment_ = a; }
  void Destroy();
  bool Eval(vector<BlockTensor> *block_tensors_v, const std::shared_ptr<FootPrint> &foot_print,
            const std::vector<DynamicBitSet> *pConstraints);

 private:
  size_t m_alignment_;
  size_t m_tensors_allocated_;
};
}  // namespace somas
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_COMMON_SOMAS_SOMAS_SOLVER_ALG_H_
