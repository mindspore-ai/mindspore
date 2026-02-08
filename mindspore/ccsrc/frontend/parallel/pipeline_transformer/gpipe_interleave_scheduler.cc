
/**
 * Copyright 2024-2025 Huawei Technologies Co., Ltd
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
#include "primitive/array_ops.h"
#include "primitive/other_ops.h"
#include "ir/anf.h"
#include "ir/graph_utils.h"
#include "primitive/auto_generate/gen_ops_primitive_a.h"
#include "primitive/auto_generate/gen_ops_primitive_d.h"
#include "primitive/auto_generate/gen_ops_primitive_m.h"

namespace mindspore {
namespace parallel {
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

void GpipeInterleavedScheduler::ForwardReorder(size_t bias, int64_t flag) {
  auto sorted_fwd_begin = SortBetweenMicro(fwd_begin_, false);
  auto sorted_fwd_end = SortBetweenMicro(fwd_end_, false);
  auto sorted_fwd_cell = SortBetweenMicro(fwd_cell_, false);
  // Sort forward
  for (size_t i = 0; i < LongToSize(micro_size_ * chunk_num_ - 1); ++i) {
    if (stage_ != stage_num_ - 1 || micro_size_ < stage_num_ || sorted_fwd_end[i].second.chunk == chunk_num_ - 1) {
      if (flag != 0 && stage_ != stage_num_ - 1) {
        auto prior = sorted_fwd_cell[i].second;
        auto last = sorted_fwd_begin[i + 1].first;
        ControlOrder(prior, last);
        auto prior1 = sorted_fwd_begin[i + 1].second;
        auto last1 = sorted_fwd_end[i].first;
        ControlOrder(prior1, last1);
        auto prior2 = sorted_fwd_end[i].second;
        auto last2 = sorted_fwd_cell[i + 1].first;
        ControlOrder(prior2, last2);
        continue;
      }
      auto prior = sorted_fwd_end[i].second;
      auto last = sorted_fwd_begin[i + 1].first;
      ControlOrder(prior, last);
      continue;
    }
    if (bias > 0) {
      auto prior = sorted_fwd_cell[i].second;
      auto last = sorted_fwd_begin[i + 1].first;
      ControlOrder(prior, last);
    }
    if (flag != 0) {
      auto prior1 = sorted_fwd_cell[i + bias].second;
      auto last1 = sorted_fwd_begin[i + bias + 1].first;
      ControlOrder(prior1, last1);
      auto prior2 = sorted_fwd_begin[i + bias + 1].second;
      auto last2 = sorted_fwd_end[i].first;
      ControlOrder(prior2, last2);
      auto prior3 = sorted_fwd_end[i].second;
      auto last3 = sorted_fwd_cell[i + bias + 1].first;
      ControlOrder(prior3, last3);
      continue;
    }
    auto prior1 = sorted_fwd_cell[i + bias].second;
    auto last1 = sorted_fwd_end[i].first;
    ControlOrder(prior1, last1);
    auto prior2 = sorted_fwd_end[i].second;
    auto last2 = sorted_fwd_begin[i + bias + 1].first;
    ControlOrder(prior2, last2);
  }
}

void GpipeInterleavedScheduler::Reorder() {
  auto sorted_fwd_begin = SortBetweenMicro(fwd_begin_, false);
  auto sorted_fwd_end = SortBetweenMicro(fwd_end_, false);
  auto sorted_bwd_begin = SortBetweenMicro(bwd_begin_, true);
  auto sorted_bwd_end = SortBetweenMicro(bwd_end_, true);
  auto sorted_bwd_cell = SortBetweenMicro(bwd_cell_, true);
  size_t bias = 0;
  if (micro_size_ > stage_num_) {
    bias = LongToSize(micro_size_ - stage_num_);
  }

  // Sort forward
  int64_t flag = stage_ % 2;
  ForwardReorder(bias, flag);

  auto prior_back = sorted_fwd_end.back().second;
  auto last_front = sorted_bwd_begin.front().first;
  ControlOrder(prior_back, last_front);

  // Sort backward
  for (size_t i = 0; i < LongToSize(micro_size_ * chunk_num_ - 1); ++i) {
    if (stage_ != 0 || micro_size_ < stage_num_ || sorted_bwd_end[i].second.chunk == 0) {
      if (flag == 0 || (stage_ == stage_num_ - 1 && sorted_bwd_begin[i + 1].first.chunk == chunk_num_ - 1) ||
          micro_size_ < stage_num_) {
        auto prior = sorted_bwd_end[i].second;
        auto last = sorted_bwd_begin[i + 1].first;
        ControlOrder(prior, last);
        continue;
      }
      auto prior = sorted_bwd_cell[i].second;
      auto last = sorted_bwd_begin[i + 1].first;
      ControlOrder(prior, last);
      auto prior1 = sorted_bwd_begin[i + 1].second;
      auto last1 = sorted_bwd_end[i].first;
      ControlOrder(prior1, last1);
      auto prior2 = sorted_bwd_end[i].second;
      auto last2 = sorted_bwd_cell[i + 1].first;
      ControlOrder(prior2, last2);
      continue;
    }
    if (bias > 0) {
      auto prior = sorted_bwd_cell[i].second;
      auto last = sorted_bwd_begin[i + 1].first;
      ControlOrder(prior, last);
    }
    if (flag != 0) {
      auto prior1 = sorted_bwd_cell[i + bias].second;
      auto last1 = sorted_bwd_begin[i + bias + 1].first;
      ControlOrder(prior1, last1);
      auto prior2 = sorted_bwd_begin[i + bias + 1].second;
      auto last2 = sorted_bwd_end[i].first;
      ControlOrder(prior2, last2);
      auto prior3 = sorted_bwd_end[i].second;
      auto last3 = sorted_bwd_cell[i + bias + 1].first;
      ControlOrder(prior3, last3);
      continue;
    }
    auto prior1 = sorted_bwd_cell[i + bias].second;
    auto last1 = sorted_bwd_end[i].first;
    ControlOrder(prior1, last1);
    auto prior2 = sorted_bwd_end[i].second;
    auto last2 = sorted_bwd_begin[i + bias + 1].first;
    ControlOrder(prior2, last2);
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

  OptimizerShardCommReorder();
}

AbstractBasePtr GpipeInterleavedScheduler::GenerateTupleAbstract(const std::vector<AnfNodePtr> &nodes) {
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

void GpipeInterleavedScheduler::OptimizerShardCommReorder() {
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
SchedulerRegisterAction PipelineSchedulerGpipe(parallel::kPipelineGpipe, [](const FuncGraphManagerPtr &manager,
                                                                            const FuncGraphPtr &root, int64_t stage,
                                                                            int64_t stage_num) {
  return std::make_shared<parallel::GpipeInterleavedScheduler>(manager, root, stage, stage_num);
});
}  // namespace parallel
}  // namespace mindspore
