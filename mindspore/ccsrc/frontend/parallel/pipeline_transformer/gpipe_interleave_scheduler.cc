
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

#include <vector>
#include <string>
#include <utility>
#include <algorithm>
#include <memory>
#include "frontend/parallel/pipeline_transformer/gpipe_interleave_scheduler.h"
#include "frontend/parallel/ops_info/ops_utils.h"
#include "frontend/parallel/step_parallel_utils.h"
#include "frontend/parallel/node_check.h"
#include "mindspore/core/ops/array_ops.h"
#include "ir/anf.h"
#include "ir/graph_utils.h"

namespace mindspore {
namespace parallel {
void GpipeInterleavedScheduler::GetBackwardBorderNode(const CNodePtr &cnode) {
  auto chunk = GetValue<int64_t>(cnode->GetPrimalAttr(CHUNK));
  auto micro = GetValue<int64_t>(cnode->GetPrimalAttr(MICRO));
  Border border = {cnode, chunk, micro};
  Border border_cell = {nullptr, chunk, micro};
  if (cnode->HasPrimalAttr(kPrimalAttrForwardNodeName)) {
    if (cnode->HasPrimalAttr(PIPELINE_BEGIN)) {
      auto bwd_cell = GetCellBySend(cnode);
      MS_EXCEPTION_IF_NULL(bwd_cell);
      if (stage_ == stage_num_ - 1 && chunk == chunk_num_ - 1) {
        Border bwd_begin = {bwd_cell, chunk, micro};
        bwd_begin_.emplace_back(bwd_begin);
        border_cell.border = bwd_cell;
        bwd_cell_.emplace_back(border_cell);
      }
      bwd_end_.emplace_back(border);
    }
    if (cnode->HasPrimalAttr(PIPELINE_END)) {
      auto bwd_cell = GetCellByReceive(cnode, manager_);
      MS_EXCEPTION_IF_NULL(bwd_cell);
      if (stage_ == 0 && chunk == 0) {
        Border bwd_end = {bwd_cell, chunk, micro};
        bwd_end_.emplace_back(bwd_end);
      }
      border_cell.border = bwd_cell;
      bwd_cell_.emplace_back(border_cell);
      bwd_begin_.emplace_back(border);
    }
    if (cnode->HasPrimalAttr(PIPELINE_PARAM)) {
      bwd_params_.emplace_back(border);
    }
  }
}

void GpipeInterleavedScheduler::GetBorderNode() {
  auto all_nodes = DeepScopedGraphSearch(root_->get_return());
  for (auto &node : all_nodes) {
    if (!IsPrimitiveCNode(node, prim::kPrimSend) && !IsPrimitiveCNode(node, prim::kPrimReceive)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    auto chunk = GetValue<int64_t>(cnode->GetPrimalAttr(CHUNK));
    chunk_num_ = (chunk + 1) > chunk_num_ ? (chunk + 1) : chunk_num_;
  }
  for (auto &node : all_nodes) {
    if (!IsPrimitiveCNode(node, prim::kPrimSend) && !IsPrimitiveCNode(node, prim::kPrimReceive)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    auto chunk = GetValue<int64_t>(cnode->GetPrimalAttr(CHUNK));
    auto micro = GetValue<int64_t>(cnode->GetPrimalAttr(MICRO));
    micro_size_ = (micro + 1) > micro_size_ ? (micro + 1) : micro_size_;
    Border border = {cnode, chunk, micro};
    Border border_cell = {nullptr, chunk, micro};
    if (cnode->HasPrimalAttr(kPrimalAttrForwardNodeName)) {
      GetBackwardBorderNode(cnode);
      continue;
    }
    if (cnode->HasPrimalAttr(PIPELINE_BEGIN)) {
      auto fwd_cell = GetCellByReceive(cnode, manager_);
      MS_EXCEPTION_IF_NULL(fwd_cell);
      if (stage_ == stage_num_ - 1 && chunk == chunk_num_ - 1) {
        Border fwd_end = {fwd_cell, chunk, micro};
        fwd_end_.emplace_back(fwd_end);
      }
      border_cell.border = fwd_cell;
      fwd_cell_.emplace_back(border_cell);
      fwd_begin_.emplace_back(border);
      continue;
    }
    if (cnode->HasPrimalAttr(PIPELINE_END)) {
      auto fwd_cell = GetCellBySend(cnode);
      MS_EXCEPTION_IF_NULL(fwd_cell);
      if (stage_ == 0 && chunk == 0) {
        Border fwd_begin = {fwd_cell, chunk, micro};
        fwd_begin_.emplace_back(fwd_begin);
        border_cell.border = fwd_cell;
        fwd_cell_.emplace_back(border_cell);
      }
      fwd_end_.emplace_back(border);
      continue;
    }
    if (cnode->HasPrimalAttr(PIPELINE_PARAM)) {
      fwd_params_.emplace_back(border);
      continue;
    }
  }
}

static bool SortFuncBetweenMicro(const BorderPair &b_i, const BorderPair &b_j, bool is_backward) {
  auto micro_i = b_i.first.micro;
  auto micro_j = b_j.first.micro;
  auto chunk_i = b_i.first.chunk;
  auto chunk_j = b_j.first.chunk;
  if (chunk_i != chunk_j) {
    if (is_backward) {
      return chunk_i > chunk_j;
    }
    return chunk_i < chunk_j;
  }

  if (micro_i == micro_j) {
    MS_LOG(EXCEPTION) << "Some wrong when sorted order between micro.";
  }
  return micro_i < micro_j;
}

std::vector<BorderPair> GpipeInterleavedScheduler::SortBetweenMicro(const std::vector<Border> &borders,
                                                                    bool is_backward) {
  auto sorted_borders = SortInsideMicro(borders);
  std::sort(sorted_borders.begin(), sorted_borders.end(), [this, is_backward](BorderPair a, BorderPair b) -> bool {
    return SortFuncBetweenMicro(a, b, is_backward);
  });
  return sorted_borders;
}

void GpipeInterleavedScheduler::Reorder() {
  auto sorted_fwd_begin = SortBetweenMicro(fwd_begin_, false);
  auto sorted_fwd_end = SortBetweenMicro(fwd_end_, false);
  auto sorted_bwd_begin = SortBetweenMicro(bwd_begin_, true);
  auto sorted_bwd_end = SortBetweenMicro(bwd_end_, true);
  auto sorted_fwd_cell = SortBetweenMicro(fwd_cell_, false);
  auto sorted_bwd_cell = SortBetweenMicro(bwd_cell_, true);
  int64_t bias = 0;
  if (micro_size_ > stage_num_) {
    bias = micro_size_ - stage_num_;
  }

  // Sort forward
  for (size_t i = 0; i < LongToSize(micro_size_ * chunk_num_ - 1); ++i) {
    if (stage_ != stage_num_ - 1 || micro_size_ < stage_num_ || sorted_fwd_end[i].second.chunk == chunk_num_ - 1) {
      auto prior = sorted_fwd_end[i].second;
      auto last = sorted_fwd_begin[i + 1].first;
      ControlOrder(prior, last);
      continue;
    }
    auto prior = sorted_fwd_cell[i].second;
    auto last = sorted_fwd_begin[i + 1].first;
    ControlOrder(prior, last);
    if (bias > 0) {
      auto prior1 = sorted_fwd_cell[i + bias].second;
      auto last1 = sorted_fwd_begin[i + bias + 1].first;
      ControlOrder(prior1, last1);
      auto prior2 = sorted_fwd_begin[i + bias + 1].second;
      auto last2 = sorted_fwd_end[i].first;
      ControlOrder(prior2, last2);
      auto prior3 = sorted_fwd_end[i].second;
      auto last3 = sorted_fwd_cell[i + bias + 1].first;
      ControlOrder(prior3, last3);
    } else {
      auto prior1 = sorted_fwd_begin[i + 1].second;
      auto last1 = sorted_fwd_end[i].first;
      ControlOrder(prior1, last1);
      auto prior2 = sorted_fwd_end[i].second;
      auto last2 = sorted_fwd_cell[i + 1].first;
      ControlOrder(prior2, last2);
    }
  }

  auto prior_back = sorted_fwd_end.back().second;
  auto last_front = sorted_bwd_begin.front().first;
  ControlOrder(prior_back, last_front);

  // Sort backward
  for (size_t i = 0; i < LongToSize(micro_size_ * chunk_num_ - 1); ++i) {
    if (stage_ != 0 || micro_size_ < stage_num_ || sorted_bwd_end[i].second.chunk == 0) {
      auto prior = sorted_bwd_end[i].second;
      auto last = sorted_bwd_begin[i + 1].first;
      ControlOrder(prior, last);
      continue;
    }
    auto prior = sorted_bwd_cell[i].second;
    auto last = sorted_bwd_begin[i + 1].first;
    ControlOrder(prior, last);
    if (bias > 0) {
      auto prior1 = sorted_bwd_cell[i + bias].second;
      auto last1 = sorted_bwd_begin[i + bias + 1].first;
      ControlOrder(prior1, last1);
      auto prior2 = sorted_bwd_begin[i + bias + 1].second;
      auto last2 = sorted_bwd_end[i].first;
      ControlOrder(prior2, last2);
      auto prior3 = sorted_bwd_end[i].second;
      auto last3 = sorted_bwd_cell[i + bias + 1].first;
      ControlOrder(prior3, last3);
    } else {
      auto prior1 = sorted_bwd_begin[i + 1].second;
      auto last1 = sorted_bwd_end[i].first;
      ControlOrder(prior1, last1);
      auto prior2 = sorted_bwd_end[i].second;
      auto last2 = sorted_bwd_cell[i + 1].first;
      ControlOrder(prior2, last2);
    }
  }

  // Parameters phase
  if (!fwd_params_.empty()) {
    std::sort(fwd_params_.begin(), fwd_params_.end(), SortFuncInsideMicro);
    std::sort(bwd_params_.begin(), bwd_params_.end(), SortFuncInsideMicro);
    auto prior = fwd_params_.back();
    auto last = sorted_fwd_begin.front().first;
    ControlOrder(prior, last);
    auto prior2 = sorted_bwd_end.back().second;
    auto last2 = bwd_params_.front();
    ControlOrder(prior2, last2);
  }
}
}  // namespace parallel
}  // namespace mindspore
