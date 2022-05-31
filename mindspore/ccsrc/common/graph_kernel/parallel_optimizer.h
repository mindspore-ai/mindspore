/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_PARALLEL_OPTIMIZER_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_PARALLEL_OPTIMIZER_H_

#include "backend/common/optimizer/pass.h"

namespace mindspore::graphkernel {
/**
 * @brief Parallel Optimizer If Order is Meaningless
 * @example
 *   %1 = UpdateState(...)
 *   %2 = AdamWeightDecay(..., %1)
 *   %3 = UpdateState(%1, %2)
 *   %4 = AdamWeightDecay(..., %3)
 *   %5 = UpdateState(%3, %4)
 *   ---------->
 *   %1 = UpdateState(...)
 *   %2 = AdamWeightDecay(..., %1)
 *   %3 = AdamWeightDecay(..., %1)
 *   %4 = UpdateState(%1, %2, %3)
 */
class ParallelOptimizer : public opt::Pass {
 public:
  explicit ParallelOptimizer(const size_t &max_n = 7) : Pass("parallel_optimizer"), max_parallel_num_(max_n) {}
  ~ParallelOptimizer() override = default;
  bool Run(const FuncGraphPtr &func_graph) override;

 private:
  size_t max_parallel_num_;
};
}  // namespace mindspore::graphkernel
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_PARALLEL_OPTIMIZER_H_
