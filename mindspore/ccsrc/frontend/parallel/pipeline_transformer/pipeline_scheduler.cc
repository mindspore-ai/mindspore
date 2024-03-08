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
#include "frontend/parallel/pipeline_transformer/pipeline_scheduler.h"
#include "frontend/parallel/ops_info/ops_utils.h"
#include "frontend/parallel/step_parallel_utils.h"
#include "frontend/parallel/node_check.h"
#include "mindspore/core/ops/array_ops.h"
#include "ir/anf.h"
#include "ir/graph_utils.h"

namespace mindspore {
namespace parallel {
CNodePtr GetCellByReceive(const AnfNodePtr &node, const FuncGraphManagerPtr &manager) {
  // receive->fg
  if (!IsPrimitiveCNode(node, prim::kPrimReceive)) {
    return nullptr;
  }
  auto users = manager->node_users()[node];
  auto user = users.front().first;
  while (IsPrimitiveCNode(user, prim::kPrimDepend)) {
    users = manager->node_users()[user];
    user = users.front().first;
  }
  auto fg_cnode = users.front().first->cast<CNodePtr>();
  return fg_cnode;
}

CNodePtr GetCellBySend(const AnfNodePtr &node) {
  // send->tuple_getitem->fg->slice
  if (!IsPrimitiveCNode(node, prim::kPrimSend)) {
    return nullptr;
  }
  auto cnode = node->cast<CNodePtr>();
  auto fg_node = cnode->input(1);
  while (IsPrimitiveCNode(fg_node, prim::kPrimTupleGetItem) || IsPrimitiveCNode(fg_node, prim::kPrimDepend)) {
    fg_node = fg_node->cast<CNodePtr>()->input(1);
  }
  auto fg_cnode = fg_node->cast<CNodePtr>();
  return fg_cnode;
}

void InterleavedScheduler::GetBackwardBorderNode(const CNodePtr &cnode) {
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

void InterleavedScheduler::GetBorderNode() {
  auto all_nodes = DeepScopedGraphSearch(root_->get_return());
  for (auto &node : all_nodes) {
    if (!IsPrimitiveCNode(node, prim::kPrimSend) && !IsPrimitiveCNode(node, prim::kPrimReceive)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    auto chunk = GetValue<int64_t>(cnode->GetPrimalAttr(CHUNK));
    chunk_num_ = (chunk + 1) > chunk_num_ ? (chunk + 1) : chunk_num_;
    auto micro = GetValue<int64_t>(cnode->GetPrimalAttr(MICRO));
    micro_size_ = (micro + 1) > micro_size_ ? (micro + 1) : micro_size_;
  }
  for (auto &node : all_nodes) {
    if (!IsPrimitiveCNode(node, prim::kPrimSend) && !IsPrimitiveCNode(node, prim::kPrimReceive)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    auto chunk = GetValue<int64_t>(cnode->GetPrimalAttr(CHUNK));
    auto micro = GetValue<int64_t>(cnode->GetPrimalAttr(MICRO));
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

bool SortFuncInsideMicro(const Border &b_i, const Border &b_j) {
  auto node_i = b_i.border;
  auto node_j = b_j.border;
  auto order_i = node_i->GetPrimalAttr(ORDER);
  auto order_j = node_j->GetPrimalAttr(ORDER);
  return (GetValue<int64_t>(order_i) < GetValue<int64_t>(order_j));
}

static bool SortFuncBetweenMicro(const BorderPair &b_i, const BorderPair &b_j, int64_t stage_num, bool is_backward) {
  auto micro_i = b_i.first.micro;
  auto micro_j = b_j.first.micro;
  auto chunk_i = b_i.first.chunk;
  auto chunk_j = b_j.first.chunk;
  auto loop_i = micro_i / stage_num;
  auto loop_j = micro_j / stage_num;
  if (loop_i != loop_j) {
    return loop_i < loop_j;
  }
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

void PipelineScheduler::ControlOrder(const Border &b_prior, const Border &b_last) {
  auto node_prior = b_prior.border;
  auto node_last = b_last.border;
  std::vector<AnfNodePtr> depend_input = {NewValueNode(prim::kPrimDepend), node_last->input(1), node_prior};
  auto depend_node = root_->NewCNode(depend_input);
  depend_node->set_abstract(node_last->input(1)->abstract());
  depend_node->AddPrimalAttr("pipeline_control", MakeValue(true));
  manager_->SetEdge(node_last, 1, depend_node);
}

std::vector<BorderPair> PipelineScheduler::SortInsideMicro(const std::vector<Border> &borders) {
  std::vector<BorderPair> out;
  for (int64_t chunk = 0; chunk < chunk_num_; ++chunk) {
    for (int64_t micro = 0; micro < micro_size_; ++micro) {
      auto border = SpecifiedBorder(borders, chunk, micro);
      out.emplace_back(border);
    }
  }
  return out;
}

std::vector<BorderPair> InterleavedScheduler::SortBetweenMicro(const std::vector<Border> &borders, bool is_backward) {
  auto sorted_borders = SortInsideMicro(borders);
  std::sort(sorted_borders.begin(), sorted_borders.end(), [this, is_backward](BorderPair a, BorderPair b) -> bool {
    return SortFuncBetweenMicro(a, b, this->stage_num_, is_backward);
  });
  return sorted_borders;
}

std::pair<Border, Border> PipelineScheduler::SpecifiedBorder(const std::vector<Border> &borders, int64_t chunk,
                                                             int64_t micro) {
  std::vector<Border> candidates;
  std::copy_if(borders.begin(), borders.end(), std::back_inserter(candidates),
               [&chunk, &micro](const auto &b) { return (b.chunk == chunk && b.micro == micro); });
  if (candidates.empty()) {
    MS_LOG(EXCEPTION) << "Can find border of the pipeline.";
  }
  if (candidates.size() > 1) {
    std::sort(candidates.begin(), candidates.end(), SortFuncInsideMicro);
    ControlOrder(candidates.front(), candidates.back());
  }
  return std::make_pair(candidates.front(), candidates.back());
}

void InterleavedScheduler::WarmUpPhaseReorder() {
  auto sorted_fwd_begin = SortBetweenMicro(fwd_begin_, false);
  auto sorted_fwd_end = SortBetweenMicro(fwd_end_, false);
  auto sorted_fwd_cell = SortBetweenMicro(fwd_cell_, false);
  auto sorted_bwd_begin = SortBetweenMicro(bwd_begin_, true);
  auto bias = stage_num_ * (chunk_num_ - 1) + (stage_num_ - stage_ - 1) * 2;
  // WarmUp phase
  for (size_t i = 0; i < LongToSize(bias); ++i) {
    // last stage
    if (stage_ == stage_num_ - 1) {
      auto prior = sorted_fwd_begin[i + 1].second;
      auto last = sorted_fwd_end[i].first;
      ControlOrder(prior, last);
      auto prior2 = sorted_fwd_end[i].second;
      auto last2 = sorted_fwd_cell[i + 1].first;
      ControlOrder(prior2, last2);
      auto prior3 = sorted_fwd_cell[i].second;
      auto last3 = sorted_fwd_begin[i + 1].first;
      ControlOrder(prior3, last3);
      continue;
    }
    auto prior = sorted_fwd_end[i].second;
    auto last = sorted_fwd_begin[i + 1].first;
    ControlOrder(prior, last);
  }
}

void InterleavedScheduler::ParameterReorder(const std::vector<BorderPair> &sorted_fwd_begin,
                                            const std::vector<BorderPair> &sorted_bwd_end) {
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

void InterleavedScheduler::Reorder() {
  auto sorted_fwd_begin = SortBetweenMicro(fwd_begin_, false);
  auto sorted_fwd_end = SortBetweenMicro(fwd_end_, false);
  auto sorted_bwd_begin = SortBetweenMicro(bwd_begin_, true);
  auto sorted_bwd_end = SortBetweenMicro(bwd_end_, true);
  auto sorted_fwd_cell = SortBetweenMicro(fwd_cell_, false);
  auto sorted_bwd_cell = SortBetweenMicro(bwd_cell_, true);
  auto bias = stage_num_ * (chunk_num_ - 1) + (stage_num_ - stage_ - 1) * 2;
  if (micro_size_ < stage_num_ * 2 || micro_size_ % stage_num_ != 0) {
    MS_LOG(EXCEPTION) << "For 1F1B Scheduler, MicroBatch num must be a multiple of StageNum, and must be larger or "
                         "equal than StageNum * 2, but got MicroBatch:"
                      << micro_size_ << " StageNum:" << stage_num_;
  }
  // WarmUp phase
  WarmUpPhaseReorder();

  for (size_t i = bias; i < LongToSize(micro_size_ * chunk_num_); ++i) {
    if (stage_ != stage_num_ - 1 || sorted_fwd_end[i].first.chunk != chunk_num_ - 1) {
      auto prior1 = sorted_fwd_cell[i].second;
      auto last1 = sorted_bwd_begin[i - bias].first;
      ControlOrder(prior1, last1);
      if (stage_ == stage_num_ - 1) {
        auto prior2 = sorted_bwd_cell[i - bias].second;
        auto last2 = sorted_fwd_begin[i + 1].first;
        ControlOrder(prior2, last2);
        auto prior3 = sorted_fwd_begin[i + 1].second;
        auto last3 = sorted_fwd_end[i].first;
        ControlOrder(prior3, last3);
        auto prior4 = sorted_fwd_end[i].second;
        auto last4 = sorted_fwd_cell[i + 1].first;
        ControlOrder(prior4, last4);
      } else {
        auto prior2 = sorted_bwd_cell[i - bias].second;
        auto last2 = sorted_fwd_end[i].first;
        ControlOrder(prior2, last2);
        if (i != LongToSize(micro_size_ * chunk_num_ - 1)) {
          auto prior3 = sorted_fwd_end[i].second;
          auto last3 = sorted_fwd_begin[i + 1].first;
          ControlOrder(prior3, last3);
        } else {
          auto prior3 = sorted_fwd_end[i].second;
          auto last3 = sorted_bwd_begin[i - bias + 1].first;
          ControlOrder(prior3, last3);
        }
      }
    }
    if (i == LongToSize(micro_size_ * chunk_num_ - 1)) {
      break;
    }
    if (stage_ != 0 || sorted_bwd_end[i - bias].first.chunk != 0) {
      auto prior4 = sorted_bwd_cell[i - bias].second;
      auto last4 = sorted_fwd_begin[i + 1].first;
      ControlOrder(prior4, last4);
      if (stage_ == 0) {
        auto prior5 = sorted_fwd_cell[i + 1].second;
        auto last5 = sorted_bwd_begin[i - bias + 1].first;
        ControlOrder(prior5, last5);
        auto prior6 = sorted_bwd_begin[i - bias + 1].second;
        auto last6 = sorted_bwd_end[i - bias].first;
        ControlOrder(prior6, last6);
        auto prior7 = sorted_bwd_end[i - bias].second;
        auto last7 = sorted_bwd_cell[i - bias + 1].first;
        ControlOrder(prior7, last7);
        continue;
      }
      auto prior5 = sorted_fwd_cell[i + 1].second;
      auto last5 = sorted_bwd_end[i - bias].first;
      ControlOrder(prior5, last5);
      auto prior6 = sorted_bwd_end[i - bias].second;
      auto last6 = sorted_bwd_begin[i - bias + 1].first;
      ControlOrder(prior6, last6);
    } else {
      auto prior7 = sorted_bwd_end[i - bias].second;
      auto last7 = sorted_fwd_begin[i + 1].first;
      ControlOrder(prior7, last7);
    }
  }

  for (size_t i = LongToSize(micro_size_ * chunk_num_ - bias - 1); i < LongToSize(micro_size_ * chunk_num_); ++i) {
    if (stage_ == 0 && sorted_bwd_end[i].second.chunk != 0) {
      auto prior = sorted_bwd_cell[i].second;
      auto last = sorted_bwd_begin[i + 1].first;
      ControlOrder(prior, last);
      auto prior2 = sorted_bwd_begin[i + 1].second;
      auto last2 = sorted_bwd_end[i].first;
      ControlOrder(prior2, last2);
      auto prior3 = sorted_bwd_end[i].second;
      auto last3 = sorted_bwd_cell[i + 1].first;
      ControlOrder(prior3, last3);
      continue;
    }
    if (i != LongToSize(micro_size_ * chunk_num_ - 1)) {
      auto end_prior = sorted_bwd_end[i].second;
      auto end_last = sorted_bwd_begin[i + 1].first;
      ControlOrder(end_prior, end_last);
    }
  }

  // Parameters phase
  ParameterReorder(sorted_fwd_begin, sorted_bwd_end);
}
}  // namespace parallel
}  // namespace mindspore
