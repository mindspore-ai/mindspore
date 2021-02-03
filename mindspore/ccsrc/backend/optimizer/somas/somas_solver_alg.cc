/**
 * Copyright 2020 Huawei Technologies Co., Ltd

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

#include "backend/optimizer/somas/somas_solver_alg.h"

#include <algorithm>
#include <stack>
#include <utility>

namespace mindspore {
namespace somas {
// offset picking heuristics
bool SmallestFit(const pair<size_t, size_t> &a, const pair<size_t, size_t> &b) {
  return a.first < b.first || (a.first == b.first && a.second < b.second);
}
bool LargestFit(const pair<size_t, size_t> &a, const pair<size_t, size_t> &b) {
  return a.first > b.first || (a.first == b.first && a.second < b.second);
}
bool BestFit(const pair<size_t, size_t> &a, const pair<size_t, size_t> &b) {
  return a.second < b.second || (a.second == b.second && a.first < b.first);
}
bool WorstFit(const pair<size_t, size_t> &a, const pair<size_t, size_t> &b) {
  return a.second > b.second || (a.second == b.second && a.first < b.first);
}
size_t SharedObjects(FootPrint *p) { return p->Next()->getOffset(); }
size_t SingleObject(FootPrint *p) { return SIZE_MAX; }

bool (*g_pBranching[kNumFittingTypes])(const pair<size_t, size_t> &a, const pair<size_t, size_t> &b) = {
  BestFit, SmallestFit
#ifdef SOMAS_DEBUG
  ,
  LargestFit, WorstFit
#endif
};
size_t (*algorithm[kNumAlgorithmTypes])(FootPrint *p) = {SharedObjects, SingleObject};

size_t FootPrint::Result() {
  std::shared_ptr<FootPrint> foot_print = shared_from_this();
  size_t upperbound = 0;
  uint32_t total_footprints = 0;
  while (NULL != foot_print) {
    foot_print->printStats();

    upperbound = foot_print->getOffset();
    foot_print = foot_print->Next();
    total_footprints++;
  }

  MS_LOG(DEBUG) << total_footprints << " footprints allocated";

  return upperbound;
}
bool FootPrint::findFirst(stack<Interval> *merged, const BlockTensor &block, size_t *offset) {
  MS_EXCEPTION_IF_NULL(merged);
  MS_EXCEPTION_IF_NULL(offset);
  bool bfound = false;
  std::set<pair<size_t, size_t>, bool (*)(const pair<size_t, size_t> &a, const pair<size_t, size_t> &b)>
    offsetcandidates(g_pBranching[m_branching_strategy_]);
  size_t gap = 0;

  Interval a;
  Interval it;

  a.ub() = algorithm[m_algorithm_](this);
  while (!(*merged).empty()) {
    it = (*merged).top();
    (*merged).pop();
    a.lb() = it.ub();
    if (a.contains(block.m_size_)) {
      gap = a.ub() - a.lb() - block.m_size_;
      offsetcandidates.emplace(pair<size_t, size_t>(a.lb(), gap));
    }
    a.ub() = it.lb();
  }
  a.lb() = m_offset_;
  gap = a.ub() - a.lb() - block.m_size_;
  if (a.contains(block.m_size_)) offsetcandidates.emplace(pair<size_t, size_t>(a.lb(), gap));

  if (offsetcandidates.size() > 0) {
    *offset = (*offsetcandidates.begin()).first;
    m_foot_print_next_->m_offset_ = std::max(m_foot_print_next_->m_offset_, *offset + block.m_size_);
    offsetcandidates.erase(offsetcandidates.begin());
    bfound = true;
  }

  return bfound;
}
void FootPrint::Merge(vector<Interval> *interval_v, stack<Interval> *s) {
  MS_EXCEPTION_IF_NULL(s);
  MS_EXCEPTION_IF_NULL(interval_v);
  sort((*interval_v).begin(), (*interval_v).end(),
       [](Interval &i1, Interval &i2) { return (i1.lb() < i2.lb()) || (i1.lb() == i2.lb() && i1.ub() < i2.ub()); });
  (*s).push((*interval_v)[0]);

  for (size_t i = 1; i < (*interval_v).size(); i++) {
    Interval &top = (*s).top();
    Interval &b = (*interval_v)[i];
    if (top.ub() < b.lb())
      (*s).push(b);

    else if (top.ub() < b.ub())
      top.ub() = b.ub();
  }

  return;
}
bool FootPrint::findOffset(const std::vector<DynamicBitSet> *constraints, const BlockTensor &block, size_t *offset) {
  MS_EXCEPTION_IF_NULL(offset);
  bool bretval = true;
  vector<Interval> l_interval;

  const size_t intervals_estimation = 1000;
  l_interval.reserve(intervals_estimation * sizeof(Interval));

  *offset = m_offset_;
  bretval = true;

  // transform constrained tensors in non eligible intervals
  if (block.Alone()) {
    if (m_algorithm_ != kSingleObject && m_starts_.size() > 0 && m_starts_[0]->Alone() &&
        (*constraints)[block.m_start_tensor_->index_].IsBitTrue(m_starts_[0]->m_start_tensor_->index_) == false) {
      return false;
    }
    for (size_t i = 0; i < m_starts_.size(); i++) {
      auto allocated_tensor = m_starts_[i]->m_start_tensor_;
      while (allocated_tensor != NULL) {
        if ((*constraints)[block.m_start_tensor_->index_].IsBitTrue(allocated_tensor->index_) == false) {
          l_interval.emplace_back(Interval(allocated_tensor));
        }
        allocated_tensor = allocated_tensor->right_;
      }
    }
  } else {
    int64_t start_offset = static_cast<int64_t>(m_offset_);
    for (size_t i = 0; i < m_starts_.size(); i++) {
      auto allocated_tensor = m_starts_[i]->m_start_tensor_;
      while (allocated_tensor != NULL) {
        int64_t allocated_offset = static_cast<int64_t>(allocated_tensor->offset_);
        int64_t allocated_size = static_cast<int64_t>(allocated_tensor->size_);
        int64_t accumulator = 0;
        for (auto block_tensor = block.m_start_tensor_; block_tensor != NULL; block_tensor = block_tensor->right_) {
          int64_t end_placement = allocated_offset + allocated_size - accumulator;
          if ((*constraints)[block_tensor->index_].IsBitTrue(allocated_tensor->index_) == false &&
              end_placement > start_offset) {
            l_interval.emplace_back(Interval(allocated_tensor));
            break;
          }
          accumulator += block_tensor->size_;
        }
        allocated_tensor = allocated_tensor->right_;
      }
    }
  }

  // merge non-eligible intervals and find a slot to allocate the tensor block
  if (!l_interval.empty()) {
    stack<Interval> l_mergedIntervals;
    Merge(&l_interval, &l_mergedIntervals);
    bretval = findFirst(&l_mergedIntervals, block, offset);
  }

  return bretval;
}
void FootPrint::addElem(BlockTensor *block, const size_t &offset) {
  if (m_foot_print_next_ == NULL) {
    m_foot_print_next_ = std::make_shared<FootPrint>();
    size_t newoffset = m_offset_ + block->m_size_;
    m_foot_print_next_->setOffset(newoffset);
    m_foot_print_next_->setAlignment(m_alignment_);
    m_foot_print_next_->m_solId_ = m_solId_;
    m_starts_.clear();
    MS_LOG(DEBUG) << "Creating footprint at offset: " << m_offset_;
  }

  addStart(block);
  size_t offset1 = offset;
  SomasSolverTensorDescPtr tensor = block->m_start_tensor_;
  MS_LOG(DEBUG) << "Allocating block: " << tensor->index_ << " in offset: " << offset;
  pair<uint32_t, size_t> sol_offset;
  sol_offset.first = block->m_current_sol_;
  sol_offset.second = offset;
  if (block->offsets_.count(sol_offset.first))
    MS_LOG(WARNING) << "Warning addElem: Offset overwritten at solution " << block->m_current_sol_ << " for block "
                    << block->m_start_tensor_->index_;
  block->offsets_.insert(sol_offset);
  while (tensor) {
    tensor->offset_ = offset1;
    offset1 += tensor->size_;

    MS_LOG(DEBUG) << tensor->index_ << " " << tensor->size_ << " " << tensor->offset_;
    tensor = tensor->right_;
  }
}
void FootPrint::printStats() {
  MS_LOG(DEBUG) << "Footprint blocks: " << m_starts_.size() << " \toffset: " << m_offset_;
}
bool FastHeuristic::Eval(vector<BlockTensor> *block_tensors_v, std::shared_ptr<FootPrint> foot_print,
                         const std::vector<DynamicBitSet> *pConstraints) {
  MS_EXCEPTION_IF_NULL(foot_print);
  auto start = std::chrono::system_clock::now();

  std::shared_ptr<FootPrint> p = foot_print;
  bool bpushed = false;
  uint32_t startscount = 0;
  size_t offset = foot_print->getOffset();
  m_tensors_allocated_ = 0;
  SomasSolverTensorDescPtr tensor = NULL;

  for (size_t i = 0; i < (*block_tensors_v).size(); i++) {
    BlockTensor &block = (*block_tensors_v)[i];
    if (block.m_bre_allocate_ == false) {
      offset = block.m_start_tensor_->offset_;
      pair<uint32_t, size_t> aux;
      aux.first = foot_print->m_solId_;
      aux.second = block.m_start_tensor_->offset_;
      if (block.offsets_.count(aux.first)) {
        MS_LOG(WARNING) << "Warning: Offset overwritten at solution " << aux.first << " for block "
                        << block.m_start_tensor_->index_;
      }
      block.offsets_.insert(aux);
      continue;
    }
    bpushed = false;
    p = foot_print;
    block.m_current_sol_ = foot_print->m_solId_;
    while (!bpushed) {
      if (p->findOffset(pConstraints, block, &offset)) {
        p->addElem(&block, offset);
        startscount++;
        tensor = block.m_start_tensor_;
        while (tensor) {
          m_tensors_allocated_++;
          tensor = tensor->right_;
        }
        bpushed = true;
        break;
      }
      // go to the next footprint slot
      if (NULL != p->Next()) {
        p = p->Next();
      } else if (bpushed == false) {  // something went wrong
        MS_LOG(WARNING) << "Could not allocate memory for tensor: " << tensor->index_;
        return false;
      }
    }
  }

  MS_LOG(DEBUG)
    << "\nElapsed time of Fast Heuristic search: "
    << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - start).count() << " ms";
  return true;
}
}  // namespace somas
}  // namespace mindspore
