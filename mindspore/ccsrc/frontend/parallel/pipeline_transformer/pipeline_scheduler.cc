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
#include "mindspore/core/ops/other_ops.h"
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
  auto cnode = node->cast<CNodePtr>();
  if (cnode->HasPrimalAttr(ORDER)) {
    auto order = cnode->GetPrimalAttr(ORDER);
    fg_cnode->AddPrimalAttr(ORDER, order);
  }
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
  if (cnode->HasPrimalAttr(ORDER)) {
    auto order = cnode->GetPrimalAttr(ORDER);
    fg_cnode->AddPrimalAttr(ORDER, order);
  }
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
  MS_EXCEPTION_IF_NULL(order_i);
  MS_EXCEPTION_IF_NULL(order_j);
  return (GetValue<int64_t>(order_i) < GetValue<int64_t>(order_j));
}

static bool SortFuncBetweenMicro(const BorderPair &b_i, const BorderPair &b_j, int64_t stage_num, bool is_backward,
                                 int64_t offset) {
  auto micro_i = b_i.first.micro;
  auto micro_j = b_j.first.micro;
  auto chunk_i = b_i.first.chunk;
  auto chunk_j = b_j.first.chunk;
  auto loop_i = (micro_i - offset) / stage_num;
  auto loop_j = (micro_j - offset) / stage_num;
  auto loop_i_offset = micro_i / (stage_num + offset);
  auto loop_j_offset = micro_j / (stage_num + offset);
  loop_i = loop_i_offset == 0 ? 0 : loop_i;
  loop_j = loop_j_offset == 0 ? 0 : loop_j;
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
    return SortFuncBetweenMicro(a, b, this->stage_num_, is_backward, this->offset_);
  });
  return sorted_borders;
}

std::pair<Border, Border> PipelineScheduler::SpecifiedBorder(const std::vector<Border> &borders, int64_t chunk,
                                                             int64_t micro) {
  std::vector<Border> candidates;
  std::copy_if(borders.begin(), borders.end(), std::back_inserter(candidates),
               [&chunk, &micro](const auto &b) { return (b.chunk == chunk && b.micro == micro); });
  if (candidates.empty()) {
    MS_LOG(EXCEPTION) << "Can not find border of the pipeline.";
  }
  if (candidates.size() > 1) {
    std::sort(candidates.begin(), candidates.end(), SortFuncInsideMicro);
    for (size_t index = 0; index < candidates.size() - 1; ++index) {
      auto prior = candidates[index];
      auto last = candidates[index + 1];
      ControlOrder(prior, last);
    }
  }
  return std::make_pair(candidates.front(), candidates.back());
}

void InterleavedScheduler::WarmUpPhaseReorder() {
  auto sorted_fwd_begin = SortBetweenMicro(fwd_begin_, false);
  auto sorted_fwd_end = SortBetweenMicro(fwd_end_, false);
  auto sorted_fwd_cell = SortBetweenMicro(fwd_cell_, false);
  auto sorted_bwd_cell = SortBetweenMicro(bwd_cell_, true);
  // WarmUp phase
  for (size_t i = 0; i < LongToSize(bias_); ++i) {
    if (i == LongToSize(micro_size_ * chunk_num_ - 1)) {
      return;
    }
    // last stage
    if (stage_ == stage_num_ - 1) {
      if (offset_ > 0) {
        auto prior = sorted_fwd_cell[i].second;
        auto last = sorted_fwd_begin[i + 1].first;
        ControlOrder(prior, last);
      }
      if (is_even_stage_) {
        if (offset_ > 0) {
          if (i + LongToSize(offset_) >= LongToSize(bias_)) {
            auto prior1 = sorted_bwd_cell[i + LongToSize(offset_) - LongToSize(bias_)].second;
            auto last1 = sorted_fwd_end[i].first;
            ControlOrder(prior1, last1);
          } else {
            auto prior1 = sorted_fwd_cell[i + LongToSize(offset_)].second;
            auto last1 = sorted_fwd_end[i].first;
            ControlOrder(prior1, last1);
          }
        }
        auto prior2 = sorted_fwd_end[i].second;
        auto last2 = sorted_fwd_begin[i + LongToSize(offset_) + 1].first;
        ControlOrder(prior2, last2);
        continue;
      }
      auto prior1 = sorted_fwd_cell[i + LongToSize(offset_)].second;
      if (i + LongToSize(offset_) >= LongToSize(bias_)) {
        prior1 = sorted_bwd_cell[i + LongToSize(offset_) - LongToSize(bias_)].second;
      }
      auto last1 = sorted_fwd_begin[i + LongToSize(offset_) + 1].first;
      ControlOrder(prior1, last1);
      auto prior2 = sorted_fwd_begin[i + LongToSize(offset_) + 1].second;
      auto last2 = sorted_fwd_end[i].first;
      ControlOrder(prior2, last2);
      auto prior3 = sorted_fwd_end[i].second;
      auto last3 = sorted_fwd_cell[i + LongToSize(offset_) + 1].first;
      ControlOrder(prior3, last3);
      continue;
    }
    if (is_even_stage_) {
      auto prior = sorted_fwd_end[i].second;
      auto last = sorted_fwd_begin[i + 1].first;
      ControlOrder(prior, last);
      continue;
    }
    auto prior = sorted_fwd_cell[i].second;
    auto last = sorted_fwd_begin[i + 1].first;
    ControlOrder(prior, last);
    auto prior1 = sorted_fwd_begin[i + 1].second;
    auto last1 = sorted_fwd_end[i].first;
    ControlOrder(prior1, last1);
    auto prior2 = sorted_fwd_end[i].second;
    auto last2 = sorted_fwd_cell[i + 1].first;
    ControlOrder(prior2, last2);
  }
}

void InterleavedScheduler::LastForwardMicroReorder() {
  auto sorted_fwd_end = SortBetweenMicro(fwd_end_, false);
  auto sorted_fwd_cell = SortBetweenMicro(fwd_cell_, false);
  auto sorted_bwd_begin = SortBetweenMicro(bwd_begin_, true);
  auto sorted_bwd_cell = SortBetweenMicro(bwd_cell_, true);
  auto sorted_bwd_end = SortBetweenMicro(bwd_end_, true);
  auto index = chunk_num_ * micro_size_ - 1 - SizeToLong(bias_);
  if (index < 0) {
    auto prior = sorted_fwd_end.back().second;
    auto last = sorted_bwd_begin.front().first;
    ControlOrder(prior, last);
    return;
  }
  if (stage_ == stage_num_ - 1) {
    auto prior = sorted_fwd_end.back().second;
    auto last = sorted_bwd_begin[index].first;
    ControlOrder(prior, last);
    return;
  }
  auto prior = sorted_bwd_cell[index].second;
  auto last = sorted_fwd_end.back().first;
  ControlOrder(prior, last);
  auto prior1 = sorted_fwd_cell.back().second;
  auto last1 = sorted_bwd_begin[index].first;
  ControlOrder(prior1, last1);
  if (stage_ == 0 && sorted_bwd_end[index].second.chunk == 0) {
    auto prior2 = sorted_fwd_end.back().second;
    auto last2 = sorted_bwd_begin[index + 1].first;
    ControlOrder(prior2, last2);
    return;
  }
  if (stage_ == 0) {
    auto loop_index = sorted_bwd_end[index].second.micro / (stage_num_ + SizeToLong(offset_));
    if (loop_index == 0) {
      auto prior2 = sorted_fwd_end.back().second;
      auto last2 = sorted_bwd_begin[index + 1].first;
      ControlOrder(prior2, last2);
    } else {
      auto prior2 = sorted_fwd_end.back().second;
      auto last2 = sorted_bwd_end[index].first;
      ControlOrder(prior2, last2);
    }
    return;
  }
  if (is_even_stage_) {
    auto prior2 = sorted_fwd_end.back().second;
    auto last2 = sorted_bwd_end[index].first;
    ControlOrder(prior2, last2);
  } else {
    auto prior2 = sorted_fwd_end.back().second;
    auto last2 = sorted_bwd_begin[index + 1].first;
    ControlOrder(prior2, last2);
  }
}

void InterleavedScheduler::EndPhaseReorder() {
  auto sorted_fwd_end = SortBetweenMicro(fwd_end_, false);
  auto sorted_bwd_begin = SortBetweenMicro(bwd_begin_, true);
  auto sorted_bwd_cell = SortBetweenMicro(bwd_cell_, true);
  auto sorted_bwd_end = SortBetweenMicro(bwd_end_, true);
  auto begin_index =
    LongToSize(chunk_num_ * micro_size_) > bias_ ? LongToSize(chunk_num_ * micro_size_ - bias_ - 1) : 0;
  for (size_t i = LongToSize(begin_index); i < LongToSize(chunk_num_ * micro_size_ - 1); ++i) {
    if (stage_ == 0) {
      auto loop_index = sorted_bwd_end[i].second.micro / (stage_num_ + SizeToLong(offset_));
      auto offset = LongToSize(offset_);
      if (loop_index != 0 || sorted_bwd_end[i].second.chunk == 0) {
        offset = 0;
      }
      if (offset > 0) {
        auto prior = sorted_bwd_cell[i].second;
        auto last = sorted_bwd_begin[i + 1].first;
        ControlOrder(prior, last);
        auto prior1 = sorted_bwd_cell[i + offset].second;
        auto last1 = sorted_bwd_end[i].first;
        ControlOrder(prior1, last1);
      }
      auto prior2 = sorted_bwd_end[i].second;
      auto last2 = sorted_bwd_begin[i + offset + 1].first;
      ControlOrder(prior2, last2);
      continue;
    }
    if (is_even_stage_ || (stage_ == stage_num_ - 1 && sorted_bwd_begin[i + 1].first.chunk == chunk_num_ - 1)) {
      auto prior = sorted_bwd_end[i].second;
      auto last = sorted_bwd_begin[i + 1].first;
      ControlOrder(prior, last);
      continue;
    }
    auto prior1 = sorted_bwd_cell[i].second;
    auto last1 = sorted_bwd_begin[i + 1].first;
    ControlOrder(prior1, last1);
    auto prior2 = sorted_bwd_begin[i + 1].second;
    auto last2 = sorted_bwd_end[i].first;
    ControlOrder(prior2, last2);
    auto prior3 = sorted_bwd_end[i].second;
    auto last3 = sorted_bwd_cell[i + 1].first;
    ControlOrder(prior3, last3);
  }
}

AbstractBasePtr InterleavedScheduler::GenerateTupleAbstract(const std::vector<AnfNodePtr> &nodes) {
  AbstractBasePtr abs;
  if (nodes.size() == 2) {
    auto cnode = nodes.back()->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    abs = cnode->abstract();
  } else {
    AbstractBasePtrList abstract_list;
    abstract_list.resize(nodes.size() - 1);
    (void)std::transform(nodes.begin() + 1, nodes.end(), abstract_list.begin(), [](const AnfNodePtr &node) {
      auto cnode = node->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(cnode);
      return cnode->abstract();
    });
    abs = std::make_shared<abstract::AbstractTuple>(abstract_list);
  }
  return abs;
}

void InterleavedScheduler::OptimizerShardCommReorder() {
  auto enable_opt_shard = ParallelContext::GetInstance()->enable_parallel_optimizer();
  if (!enable_opt_shard) {
    return;
  }
  auto sorted_fwd_begin = SortBetweenMicro(fwd_begin_, false);
  auto sorted_fwd_cell = SortBetweenMicro(fwd_cell_, false);
  std::vector<AnfNodePtr> make_tuple_inputs = {NewValueNode(prim::kPrimMakeTuple)};
  for (int64_t chunk = 1; chunk < chunk_num_; ++chunk) {
    for (const auto &border : sorted_fwd_cell) {
      if (border.first.chunk == chunk) {
        auto cnode = border.first.border;
        for (const auto &input : cnode->inputs()) {
          if (!IsPrimitiveCNode(input, prim::kPrimAllGather)) {
            continue;
          }
          make_tuple_inputs.emplace_back(input);
        }
      }
    }
  }
  if (make_tuple_inputs.size() > 1) {
    auto make_tuple = root_->NewCNode(make_tuple_inputs);
    auto abs = GenerateTupleAbstract(make_tuple_inputs);
    make_tuple->set_abstract(abs);
    auto begin_node = sorted_fwd_begin.front().first.border;
    if (begin_node->inputs().size() < 2) {
      return;
    }
    std::vector<AnfNodePtr> depend_inputs = {NewValueNode(prim::kPrimDepend), begin_node->input(1), make_tuple};
    auto depend = root_->NewCNode(depend_inputs);
    depend->set_abstract(begin_node->input(1)->abstract());
    manager_->SetEdge(begin_node, 1, depend);
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

void InterleavedScheduler::MemoryOptimizedWarmUpPhaseReorder() {
  auto sorted_fwd_begin = SortBetweenMicro(fwd_begin_, false);
  auto sorted_fwd_end = SortBetweenMicro(fwd_end_, false);
  auto sorted_fwd_cell = SortBetweenMicro(fwd_cell_, false);
  auto sorted_bwd_cell = SortBetweenMicro(bwd_cell_, true);
  for (size_t i = 0; i < LongToSize(bias_); ++i) {
    if (stage_ != 0) {
      auto prior = sorted_fwd_end[i].second;
      auto last = sorted_fwd_begin[i + 1].first;
      ControlOrder(prior, last);
      continue;
    } else {
      size_t offset = 0;
      if (sorted_fwd_begin[i + 1].first.chunk != 0) {
        offset = offset_;
      }
      auto prior = sorted_fwd_end[i].second;
      auto last = sorted_fwd_cell[i + 1].first;
      ControlOrder(prior, last);
      auto prior1 = sorted_fwd_cell[i - LongToSize(offset)].second;
      auto last1 = sorted_fwd_begin[i + 1].first;
      ControlOrder(prior1, last1);
      auto prior2 = sorted_fwd_begin[i + 1].second;
      auto last2 = sorted_fwd_end[i - LongToSize(offset)].first;
      ControlOrder(prior1, last1);
    }
  }
}

void InterleavedScheduler::MemoryOptimizedStablePhaseReorder() {
  auto sorted_fwd_begin = SortBetweenMicro(fwd_begin_, false);
  auto sorted_fwd_end = SortBetweenMicro(fwd_end_, false);
  auto sorted_bwd_begin = SortBetweenMicro(bwd_begin_, true);
  auto sorted_bwd_end = SortBetweenMicro(bwd_end_, true);
  auto sorted_fwd_cell = SortBetweenMicro(fwd_cell_, false);
  auto sorted_bwd_cell = SortBetweenMicro(bwd_cell_, true);
  for (size_t i = bias_; i < LongToSize(micro_size_ * chunk_num_); ++i) {
    if (i == LongToSize(micro_size_ * chunk_num_ - 1)) {
      if (stage_ != 0) {
        auto prior = sorted_fwd_end[i].second;
        auto last = sorted_bwd_begin[i - bias_].first;
        ControlOrder(prior, last);
      } else {
        auto prior = sorted_fwd_cell[i].second;
        auto last = sorted_bwd_begin[i - bias_].first;
        ControlOrder(prior, last);
        auto prior1 = sorted_bwd_begin[i - bias_].second;
        auto last1 = sorted_fwd_end[i].first;
        ControlOrder(prior1, last1);
        auto prior2 = sorted_fwd_end[i].second;
        auto last2 = sorted_bwd_cell[i - bias_].first;
        ControlOrder(prior2, last2);
      }
      continue;
    }
    if (stage_ != 0) {
      auto prior = sorted_bwd_end[i - bias_].second;
      auto last = sorted_fwd_begin[i + 1].first;
      ControlOrder(prior, last);
    } else {
      auto offset = offset_;
      auto loop_index_bwd = sorted_bwd_end[i - bias_].second.micro / (stage_num_ + SizeToLong(offset_));
      if (loop_index_bwd != 0) {
        offset = 0;
      }
      auto loop_index_fwd = sorted_fwd_end[i + 1].second.micro / (stage_num_ + SizeToLong(offset_));
      if (loop_index_fwd == 0) {
        auto prior1 = sorted_fwd_end[i - offset_].second;
        auto last1 = sorted_fwd_cell[i + 1 - offset_].first;
        ControlOrder(prior1, last1);
        auto prior2 = sorted_fwd_cell[i - offset_].second;
        auto last2 = sorted_fwd_begin[i + 1].first;
        ControlOrder(prior2, last2);
        auto prior3 = sorted_fwd_begin[i + 1].second;
        auto last3 = sorted_fwd_end[i - offset_].first;
        ControlOrder(prior3, last3);
      }
      if (sorted_bwd_end[i - bias_].second.chunk != 0) {
        auto prior1 = sorted_bwd_cell[i - bias_].second;
        auto last1 = sorted_fwd_cell[i + 1].first;
        ControlOrder(prior1, last1);
        if (i + 1 + offset > LongToSize(micro_size_ * chunk_num_ - 1)) {
          auto prior2 = sorted_bwd_begin[i - bias_ + 1 + offset].second;
          auto last2 = sorted_bwd_end[i - bias_].first;
          ControlOrder(prior2, last2);
        } else {
          auto prior2 = sorted_fwd_end[i + 1 + offset].second;
          auto last2 = sorted_bwd_end[i - bias_].first;
          ControlOrder(prior2, last2);
        }
        if ((i + 1 + offset <= LongToSize(micro_size_ * chunk_num_ - 1)) &&
            sorted_fwd_end[i + 1 + offset].second.chunk != chunk_num_ - 1) {
          auto prior3 = sorted_bwd_end[i - bias_].second;
          auto last3 = sorted_fwd_begin[i + 1 + LongToSize(stage_num_) + offset].first;
          ControlOrder(prior3, last3);
          auto prior4 = sorted_fwd_begin[i + 1 + LongToSize(stage_num_) + offset].second;
          auto last4 = sorted_bwd_cell[i - bias_ + 1 + offset].first;
          ControlOrder(prior4, last4);
        } else {
          auto prior3 = sorted_bwd_end[i - bias_].second;
          auto last3 = sorted_bwd_cell[i - bias_ + 1 + offset].first;
          ControlOrder(prior3, last3);
        }
      } else {
        auto prior = sorted_bwd_end[i - bias_].second;
        auto last = sorted_fwd_cell[i + 1].first;
        ControlOrder(prior, last);
      }
    }
    if (stage_ != stage_num_ - 1 || sorted_fwd_end[i].second.chunk != chunk_num_ - 1) {
      auto prior = sorted_fwd_cell[i].second;
      auto last = sorted_bwd_begin[i - bias_].first;
      ControlOrder(prior, last);
      auto prior1 = sorted_bwd_begin[i - bias_].second;
      auto last1 = sorted_fwd_end[i].first;
      ControlOrder(prior1, last1);
      auto prior2 = sorted_fwd_end[i].second;
      auto last2 = sorted_bwd_cell[i - bias_].first;
      ControlOrder(prior2, last2);
    } else {
      auto prior = sorted_fwd_end[i].second;
      auto last = sorted_bwd_begin[i - bias_].first;
      ControlOrder(prior, last);
    }
  }
}

void InterleavedScheduler::MemoryOptimizedReorder() {
  offset_ = LongToSize(micro_size_ % stage_num_);
  bias_ = LongToSize((stage_num_ + offset_) * (chunk_num_ - 1) + stage_num_ - stage_ - 1);
  auto sorted_bwd_begin = SortBetweenMicro(bwd_begin_, true);
  auto sorted_bwd_end = SortBetweenMicro(bwd_end_, true);
  auto sorted_bwd_cell = SortBetweenMicro(bwd_cell_, true);
  if (micro_size_ < stage_num_) {
    MS_LOG(EXCEPTION) << "For 1F1B Scheduler, MicroBatch num must be larger or equal than StageNum, but got MicroBatch:"
                      << micro_size_ << " StageNum:" << stage_num_;
  }
  // WarmUp phase
  MemoryOptimizedWarmUpPhaseReorder();

  // Stable phase
  MemoryOptimizedStablePhaseReorder();

  for (size_t i = LongToSize(micro_size_ * chunk_num_ - bias_ - 1); i < LongToSize(micro_size_ * chunk_num_ - 1); ++i) {
    if (stage_ != stage_num_ - 1 || sorted_bwd_begin[i + 1].first.chunk == chunk_num_ - 1) {
      auto prior = sorted_bwd_end[i].second;
      auto last = sorted_bwd_begin[i + 1].first;
      ControlOrder(prior, last);
    } else {
      auto prior = sorted_bwd_cell[i].second;
      auto last = sorted_bwd_begin[i + 1].first;
      ControlOrder(prior, last);
      auto prior1 = sorted_bwd_begin[i + 1].second;
      auto last1 = sorted_bwd_end[i].first;
      ControlOrder(prior1, last1);
      auto prior2 = sorted_bwd_end[i].second;
      auto last2 = sorted_bwd_cell[i + 1].first;
      ControlOrder(prior2, last2);
    }
  }
}

static bool EnableKbk() {
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  auto jit_level = context->get_param<std::string>(MS_CTX_JIT_LEVEL);
  return (jit_level == "O0" || jit_level == "O1");
}

void InterleavedScheduler::StablePhaseReorder() {
  auto sorted_fwd_begin = SortBetweenMicro(fwd_begin_, false);
  auto sorted_fwd_end = SortBetweenMicro(fwd_end_, false);
  auto sorted_bwd_begin = SortBetweenMicro(bwd_begin_, true);
  auto sorted_bwd_end = SortBetweenMicro(bwd_end_, true);
  auto sorted_fwd_cell = SortBetweenMicro(fwd_cell_, false);
  auto sorted_bwd_cell = SortBetweenMicro(bwd_cell_, true);
  for (size_t i = LongToSize(bias_); i < LongToSize(micro_size_ * chunk_num_ - 1); ++i) {
    if (stage_ == stage_num_ - 1 && sorted_fwd_end[i].first.chunk == chunk_num_ - 1) {
      auto prior = sorted_fwd_end[i].second;
      auto last = sorted_bwd_begin[i - LongToSize(bias_)].first;
      ControlOrder(prior, last);
      auto prior1 = sorted_bwd_cell[i - LongToSize(bias_)].second;
      auto last1 = sorted_fwd_begin[i + 1].first;
      ControlOrder(prior1, last1);
    } else {
      auto prior = sorted_fwd_cell[i].second;
      auto last = sorted_bwd_begin[i - LongToSize(bias_)].first;
      ControlOrder(prior, last);
    }
    if (is_even_stage_) {
      if (stage_ != stage_num_ - 1 || sorted_fwd_end[i].first.chunk != chunk_num_ - 1) {
        auto prior = sorted_bwd_cell[i - LongToSize(bias_)].second;
        auto last = sorted_fwd_end[i].first;
        ControlOrder(prior, last);
        auto prior1 = sorted_fwd_end[i].second;
        auto last1 = sorted_fwd_begin[i + 1].first;
        ControlOrder(prior1, last1);
      }
      if (stage_ != 0 || sorted_bwd_end[i - LongToSize(bias_)].first.chunk != 0) {
        auto loop_index = sorted_bwd_end[i - LongToSize(bias_)].first.micro / (stage_num_ + SizeToLong(offset_));
        auto offset = LongToSize(offset_);
        if (loop_index != 0 || stage_ != 0) {
          offset = 0;
        }
        if (i + offset + 1 > LongToSize(micro_size_ * chunk_num_ - 1)) {
          auto prior = sorted_bwd_cell[i + offset - LongToSize(bias_)].second;
          auto last = sorted_bwd_end[i - LongToSize(bias_)].first;
          ControlOrder(prior, last);
        } else {
          auto prior = sorted_fwd_cell[i + offset + 1].second;
          auto last = sorted_bwd_end[i - LongToSize(bias_)].first;
          ControlOrder(prior, last);
        }
        auto prior1 = sorted_bwd_end[i - LongToSize(bias_)].second;
        auto last1 = sorted_bwd_begin[i + offset + 1 - LongToSize(bias_)].first;
        ControlOrder(prior1, last1);
      }
      continue;
    }
    if (stage_ != stage_num_ - 1 || sorted_fwd_end[i].first.chunk != chunk_num_ - 1) {
      auto prior = sorted_bwd_cell[i - LongToSize(bias_)].second;
      auto last = sorted_fwd_begin[i + 1].first;
      ControlOrder(prior, last);
      auto prior1 = sorted_fwd_begin[i + 1].second;
      auto last1 = sorted_fwd_end[i].first;
      ControlOrder(prior1, last1);
      auto prior2 = sorted_fwd_end[i].second;
      auto last2 = sorted_fwd_cell[i + 1].first;
      ControlOrder(prior2, last2);
    }
    if (stage_ != stage_num_ - 1 || sorted_bwd_begin[i - LongToSize(bias_) + 1].second.chunk != chunk_num_ - 1) {
      auto prior = sorted_bwd_begin[i - LongToSize(bias_) + 1].second;
      auto last = sorted_bwd_end[i - LongToSize(bias_)].first;
      ControlOrder(prior, last);
      auto prior1 = sorted_bwd_end[i - LongToSize(bias_)].second;
      auto last1 = sorted_bwd_cell[i - LongToSize(bias_) + 1].first;
      ControlOrder(prior1, last1);
      continue;
    }
    auto prior = sorted_fwd_cell[i + 1].second;
    auto last = sorted_bwd_end[i - LongToSize(bias_)].first;
    ControlOrder(prior, last);
    auto prior1 = sorted_bwd_end[i - LongToSize(bias_)].second;
    auto last1 = sorted_bwd_cell[i - LongToSize(bias_) + 1].first;
    ControlOrder(prior1, last1);
  }
}

void InterleavedScheduler::Reorder() {
  auto enable_kbk = EnableKbk();
  auto sorted_fwd_begin = SortBetweenMicro(fwd_begin_, false);
  auto sorted_bwd_end = SortBetweenMicro(bwd_end_, true);
  if (enable_kbk) {
    MemoryOptimizedReorder();
    ParameterReorder(sorted_fwd_begin, sorted_bwd_end);
    OptimizerShardCommReorder();
    return;
  }
  offset_ = LongToSize(micro_size_ % stage_num_);
  bias_ = LongToSize((stage_num_ + SizeToLong(offset_)) * (chunk_num_ - 1) + (stage_num_ - stage_ - 1) * 2);
  is_even_stage_ = stage_ % 2 == 0;
  if (micro_size_ < stage_num_) {
    MS_LOG(EXCEPTION) << "For 1F1B Scheduler, MicroBatch num must be larger or equal than StageNum, but got MicroBatch:"
                      << micro_size_ << " StageNum:" << stage_num_;
  }
  // WarmUp phase
  WarmUpPhaseReorder();

  // Stable phase
  StablePhaseReorder();
  LastForwardMicroReorder();

  // End phase
  EndPhaseReorder();

  // Parameters phase
  ParameterReorder(sorted_fwd_begin, sorted_bwd_end);
  OptimizerShardCommReorder();
}
}  // namespace parallel
}  // namespace mindspore
