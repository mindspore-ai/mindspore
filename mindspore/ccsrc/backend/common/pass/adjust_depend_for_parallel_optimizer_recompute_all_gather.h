/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_PASS_ADJUST_DEPEND_FOR_PARALLEL_OPTIMIZER_RECOMPUTE_ALL_GATHER_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_PASS_ADJUST_DEPEND_FOR_PARALLEL_OPTIMIZER_RECOMPUTE_ALL_GATHER_H_
#include <vector>
#include <string>
#include <memory>

#include "include/backend/optimizer/pass.h"
#include "ir/func_graph.h"
#include "ir/anf.h"
#include "include/backend/optimizer/helper.h"
#include "include/backend/optimizer/optimizer.h"

namespace mindspore {
namespace opt {
class BACKEND_EXPORT AdjustDependForParallelOptimizerRecomputeAllGather : public Pass {
 public:
  AdjustDependForParallelOptimizerRecomputeAllGather()
      : Pass("adjust_depend_for_parallel_optimizer_recompute_all_gather") {}
  ~AdjustDependForParallelOptimizerRecomputeAllGather() override = default;
  bool Run(const FuncGraphPtr &graph) override;

 private:
  bool AdjustAllgatherDepend(const FuncGraphPtr &graph,
                             const std::vector<AnfNodePtr> &parallel_optimizer_recompute_allgathers) const;
  void IncreaseAllgatherFusionId(const std::vector<AnfNodePtr> &parallel_optimizer_recompute_allgathers,
                                 const std::vector<AnfNodePtr> &parallel_optimizer_recompute_first_fusion_allgathers,
                                 int64_t unrecompute_max_fusion_id, int64_t recompute_min_fusion_id) const;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_PASS_ADJUST_DEPEND_FOR_PARALLEL_OPTIMIZER_RECOMPUTE_ALL_GATHER_H_
