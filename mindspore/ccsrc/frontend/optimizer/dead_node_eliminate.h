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
#ifndef MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_DEAD_NODE_ELIMINATE_H
#define MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_DEAD_NODE_ELIMINATE_H
#include "ir/anf.h"
#include "ir/manager.h"
#include "frontend/optimizer/optimizer.h"
namespace mindspore::opt {
bool EliminateDeadNode(const FuncGraphPtr &func_graph);
class EliminateDeadNodePass {
 public:
  EliminateDeadNodePass() = default;
  ~EliminateDeadNodePass() = default;
  bool operator()(const FuncGraphPtr &func_graph, const OptimizerPtr &optimizer) {
    static const auto eliminate_unused_element = common::GetEnv("MS_DEV_ENABLE_DDE");
    static const auto enable_eliminate_unused_element = (eliminate_unused_element == "1");
    if (enable_eliminate_unused_element) {
      return false;
    }

    static bool enable_closure = common::GetEnv("MS_DEV_ENABLE_CLOSURE") == "1";
    MS_LOG(INFO) << "Closure enable:" << enable_closure;
    if (!enable_closure) {
      return false;
    }

    auto change = EliminateDeadNode(func_graph);
    return change;
  }
};
}  // namespace mindspore::opt
#endif  // MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_DEAD_NODE_ELIMINATE_H
