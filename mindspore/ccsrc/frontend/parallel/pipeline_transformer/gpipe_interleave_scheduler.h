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

#ifndef MINDSPORE_CCSRC_FRONTEND_PARALLEL_PIPELINE_TRANSFORMER_GPIPE_INTERLEAVE_SCHEDULER_H_
#define MINDSPORE_CCSRC_FRONTEND_PARALLEL_PIPELINE_TRANSFORMER_GPIPE_INTERLEAVE_SCHEDULER_H_

#include <set>
#include <utility>
#include <string>
#include <memory>
#include <vector>
#include "ir/value.h"
#include "ir/graph_utils.h"
#include "base/base.h"
#include "utils/hash_map.h"
#include "frontend/parallel/pipeline_transformer/pipeline_scheduler.h"

namespace mindspore {
namespace parallel {
class GpipeInterleavedScheduler : public PipelineScheduler {
 public:
  GpipeInterleavedScheduler(const FuncGraphManagerPtr &manager, const FuncGraphPtr &root, int64_t stage,
                            int64_t stage_num)
      : PipelineScheduler(manager, root, stage, stage_num) {}
  virtual ~GpipeInterleavedScheduler() = default;

  void GetBorderNode() override;
  void Reorder() override;

 private:
  std::vector<BorderPair> SortBetweenMicro(const std::vector<Border> &borders, bool is_backward);
  void GetBackwardBorderNode(const CNodePtr &cnode);
  void ForwardReorder(int64_t bias, int64_t flag);
  AbstractBasePtr GenerateTupleAbstract(const std::vector<AnfNodePtr> &nodes);
  void OptimizerShardCommReorder();
  std::vector<Border> fwd_begin_;
  std::vector<Border> fwd_end_;
  std::vector<Border> bwd_begin_;
  std::vector<Border> bwd_end_;
  std::vector<Border> fwd_cell_;
  std::vector<Border> bwd_cell_;
  std::vector<Border> fwd_params_;
  std::vector<Border> bwd_params_;
};
}  // namespace parallel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_PIPELINE_TRANSFORMER_GPIPE_INTERLEAVE_SCHEDULER_H_
