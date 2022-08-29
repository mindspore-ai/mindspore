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

#include "backend/common/somas/somas_solver_alg.h"

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
size_t SingleObject(FootPrint *) { return SIZE_MAX; }

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
  while (foot_print != nullptr) {
    foot_print->printStats();

    upperbound = foot_print->getOffset();
    foot_print = foot_print->Next();
    total_footprints++;
  }

  MS_LOG(DEBUG) << total_footprints << " footprints allocated";

  return upperbound;
}

bool FootPrint::findFirst(vector<Interval> *interval_v, const BlockTensor &block, size_t *offset) {
  MS_EXCEPTION_IF_NULL(interval_v);
  MS_EXCEPTION_IF_NULL(offset);
  if (this->Next() == nullptr) {
    *offset = this->getOffset();
    return true;
  }
  sort((*interval_v).begin(), (*interval_v).end(),
       [](Interval &i1, Interval &i2) { return (i1.lb() < i2.lb()) || (i1.lb() == i2.lb() && i1.ub() < i2.ub()); });

  bool bfound = false;
  auto fit_func = g_pBranching[m_branching_strategy_];
  pair<size_t, size_t> fit_ret;
  Interval a;

  size_t block_size;
  if (block.Alone()) {
    block_size = block.m_size_;
  } else {
    block_size = block.m_start_tensor_->size_;  // consider only first tensor for contiguous block
  }

  Interval top(m_offset_, m_offset_);
  a.lb() = m_offset_;

  auto update_fit = [block_size, &fit_func](Interval a, bool &bfound, pair<size_t, size_t> &fit_ret) {
    size_t gap;
    gap = a.ub() - a.lb() - block_size;
    auto fit_update = pair<size_t, size_t>(a.lb(), gap);
    if (!bfound) {
      bfound = true;
      fit_ret = fit_update;
    } else if (fit_func(fit_update, fit_ret)) {
      fit_ret = fit_update;
    }
  };

  for (auto &b : *interval_v) {
    if (top.ub() < b.lb()) {
      a.ub() = b.lb();
      if (a.contains(block_size) && a.lb() + block.m_size_ <= algorithm[m_algorithm_](this)) {
        update_fit(a, bfound, fit_ret);
      }
      top = b;
    } else if (top.ub() < b.ub()) {
      top.ub() = b.ub();
    }
    a.lb() = top.ub();
  }

  a.ub() = algorithm[m_algorithm_](this);
  if (a.contains(block_size) && a.lb() + block.m_size_ <= algorithm[m_algorithm_](this)) {
    update_fit(a, bfound, fit_ret);
  }

  if (bfound) {
    *offset = fit_ret.first;
    m_foot_print_next_->m_offset_ = std::max(m_foot_print_next_->m_offset_, *offset + block.m_size_);
  }

  return bfound;
}

bool FootPrint::findOffset(const std::vector<DynamicBitSet> *constraints, const BlockTensor &block, size_t *offset) {
  MS_EXCEPTION_IF_NULL(offset);
  vector<Interval> l_interval;

  const size_t intervals_estimation = 1000;
  l_interval.reserve(intervals_estimation * sizeof(Interval));

  *offset = m_offset_;

  // transform constrained tensors in non eligible intervals
  if (block.Alone()) {
    if (m_algorithm_ == static_cast<uint32_t>(kManyObjects) && m_starts_.size() > 0 && m_starts_[0]->Alone() &&
        !(*constraints)[block.m_start_tensor_->index_].IsBitTrue(m_starts_[0]->m_start_tensor_->index_)) {
      return false;
    }

    for (const auto &allocated_tensor_info : m_tensors_info_) {
      if (!(*constraints)[block.m_start_tensor_->index_].IsBitTrue(allocated_tensor_info.index_)) {
        l_interval.emplace_back(allocated_tensor_info.offset_,
                                allocated_tensor_info.offset_ + allocated_tensor_info.size_);
      }
    }
  } else {
    auto start_offset = static_cast<int64_t>(m_offset_);
    int64_t accumulator = 0;
    for (auto block_tensor = block.m_start_tensor_; block_tensor != nullptr; block_tensor = block_tensor->right_) {
      for (const auto &allocated_tensor_info : m_tensors_info_) {
        auto allocated_offset = static_cast<int64_t>(allocated_tensor_info.offset_);
        auto allocated_size = static_cast<int64_t>(allocated_tensor_info.size_);
        if (!(*constraints)[block_tensor->index_].IsBitTrue(allocated_tensor_info.index_)) {
          int64_t start_first_contiguous = allocated_offset - accumulator - SizeToLong(block_tensor->size_);
          int64_t end_first_contiguous = allocated_offset - accumulator + allocated_size;
          if (start_first_contiguous > start_offset) {
            l_interval.emplace_back(start_first_contiguous, end_first_contiguous);
          } else {
            if (end_first_contiguous > start_offset) {
              l_interval.emplace_back(start_offset, end_first_contiguous);
            }
          }
        }
      }
      accumulator += SizeToLong(block_tensor->size_);
    }
  }

  // merge non-eligible intervals and find a slot to allocate the tensor block
  return findFirst(&l_interval, block, offset);
}

void FootPrint::addTensorsInfo(BlockTensor *elemIndex) {
  MS_EXCEPTION_IF_NULL(elemIndex);
  m_starts_.push_back(elemIndex);
  auto allocated_tensor = elemIndex->m_start_tensor_;
  while (allocated_tensor != nullptr) {
    m_tensors_info_.emplace_back(allocated_tensor);
    allocated_tensor = allocated_tensor->right_;
  }
}

void FootPrint::addElem(BlockTensor *block, const size_t &offset) {
  if (m_foot_print_next_ == nullptr) {
    m_foot_print_next_ = std::make_shared<FootPrint>();
    size_t newoffset = m_offset_ + block->m_size_;
    m_foot_print_next_->setOffset(newoffset);
    m_foot_print_next_->setAlignment(m_alignment_);
    m_foot_print_next_->m_solId_ = m_solId_;
    m_starts_.clear();
    m_tensors_info_.clear();
    MS_LOG(DEBUG) << "Creating footprint at offset: " << m_offset_;
  }

  size_t offset1 = offset;
  SomasSolverTensorDescPtr tensor = block->m_start_tensor_;
  MS_LOG(DEBUG) << "Allocating block: " << tensor->index_ << " in offset: " << offset;
  auto sol_id = block->m_current_sol_;
  if (block->offsets_.find(sol_id) != block->offsets_.end()) {
    MS_LOG(WARNING) << "Warning addElem: Offset overwritten at solution " << sol_id << " for block "
                    << block->m_start_tensor_->index_;
  }
  (void)block->offsets_.emplace(sol_id, offset);
  while (tensor) {
    tensor->offset_ = offset1;
    offset1 += tensor->size_;

    MS_LOG(DEBUG) << tensor->index_ << " " << tensor->size_ << " " << tensor->offset_;
    tensor = tensor->right_;
  }
  addTensorsInfo(block);
}
void FootPrint::printStats() {
  MS_LOG(DEBUG) << "Footprint blocks: " << m_starts_.size() << " \toffset: " << m_offset_;
}
bool FastHeuristic::Eval(vector<BlockTensor> *block_tensors_v, const std::shared_ptr<FootPrint> &foot_print,
                         const std::vector<DynamicBitSet> *pConstraints) {
  MS_EXCEPTION_IF_NULL(foot_print);
  auto start = std::chrono::system_clock::now();

  std::shared_ptr<FootPrint> p = foot_print;
  bool bpushed = false;
  size_t offset = foot_print->getOffset();
  m_tensors_allocated_ = 0;
  SomasSolverTensorDescPtr tensor = nullptr;

  for (auto &block : *block_tensors_v) {
    if (!block.m_bre_allocate_) {
      offset = block.m_start_tensor_->offset_;
      auto aux_id = foot_print->m_solId_;
      auto aux_offset = block.m_start_tensor_->offset_;
      if (block.offsets_.find(aux_id) != block.offsets_.end()) {
        MS_LOG(WARNING) << "Warning: Offset overwritten at solution " << aux_id << " for block "
                        << block.m_start_tensor_->index_;
      }
      (void)block.offsets_.emplace(aux_id, aux_offset);
      continue;
    }
    bpushed = false;
    p = foot_print;
    block.m_current_sol_ = foot_print->m_solId_;
    while (!bpushed) {
      if (p->findOffset(pConstraints, block, &offset)) {
        p->addElem(&block, offset);
        tensor = block.m_start_tensor_;
        while (tensor) {
          m_tensors_allocated_++;
          tensor = tensor->right_;
        }
        bpushed = true;
        break;
      }
      // go to the next footprint slot
      if (p->Next() != nullptr) {
        p = p->Next();
      } else if (bpushed == false) {  // something went wrong
        MS_LOG(WARNING) << "Internal Error: Could not allocate memory for tensor: " << tensor->index_;
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
