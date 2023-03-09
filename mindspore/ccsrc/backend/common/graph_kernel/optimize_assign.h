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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_OPTIMIZE_ASSIGN_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_OPTIMIZE_ASSIGN_H_

#include <memory>
#include "include/backend/optimizer/optimizer.h"
#include "include/backend/kernel_graph.h"

namespace mindspore::graphkernel {
class OptimizeAssign : public opt::Pass {
 public:
  OptimizeAssign() : Pass("optimize_assign") {}
  ~OptimizeAssign() override = default;
  bool Run(const FuncGraphPtr &func_graph) override;
};
using OptimizeAssignPtr = std::shared_ptr<OptimizeAssign>;
}  // namespace mindspore::graphkernel
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_OPTIMIZE_ASSIGN_H_
