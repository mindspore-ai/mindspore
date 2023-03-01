/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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
#include "extendrt/graph_executor/factory.h"
#include <functional>
#include <memory>

namespace mindspore {
GraphExecutorRegistry &GraphExecutorRegistry::GetInstance() {
  static GraphExecutorRegistry instance;
  return instance;
}

void GraphExecutorRegistry::RegExecutor(const mindspore::GraphExecutorType &type, const GraphExecutorRegFunc &creator) {
  graph_executor_map_[type] = creator;
}

std::shared_ptr<infer::abstract::Executor> GraphExecutorRegistry::GetExecutor(
  const mindspore::GraphExecutorType &type, const std::string &name,
  std::shared_ptr<infer::abstract::ExecutionPlan> execution_plan) {
  auto it = graph_executor_map_.find(type);
  if (it == graph_executor_map_.end()) {
    return nullptr;
  }
  return it->second(name, execution_plan);
}
}  // namespace mindspore
