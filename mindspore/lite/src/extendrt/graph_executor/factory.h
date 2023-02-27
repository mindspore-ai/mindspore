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
#ifndef MINDSPORE_LITE_SRC_EXTENDRT_GRAPH_EXECUTOR_FACTORY_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_GRAPH_EXECUTOR_FACTORY_H_

#include <functional>
#include <memory>
#include <string>

#include "extendrt/graph_executor/type.h"
#include "infer/executor.h"
#include "infer/execution_plan.h"

namespace mindspore {
using GraphExecutorRegFunc = std::function<std::shared_ptr<infer::abstract::Executor>(
  const std::string &name, std::shared_ptr<infer::abstract::ExecutionPlan> execution_plan)>;

class GraphExecutorRegistry {
 public:
  GraphExecutorRegistry() = default;
  virtual ~GraphExecutorRegistry() = default;

  static GraphExecutorRegistry &GetInstance();

  void RegExecutor(const GraphExecutorType &type, const GraphExecutorRegFunc &creator);

  std::shared_ptr<infer::abstract::Executor> GetExecutor(
    const mindspore::GraphExecutorType &type, const std::string &name,
    std::shared_ptr<infer::abstract::ExecutionPlan> execution_plan);

 private:
  mindspore::HashMap<GraphExecutorType, GraphExecutorRegFunc> graph_executor_map_;
};

class GraphExecutorRegistrar {
 public:
  GraphExecutorRegistrar(const mindspore::GraphExecutorType &type, const GraphExecutorRegFunc &creator) {
    GraphExecutorRegistry::GetInstance().RegExecutor(type, creator);
  }
  ~GraphExecutorRegistrar() = default;
};

#define REG_GRAPH_EXECUTOR(type, creator) static GraphExecutorRegistrar g_##type##GraphExecutor(type, creator);
}  // namespace mindspore

#endif  // MINDSPORE_LITE_SRC_EXTENDRT_GRAPH_EXECUTOR_FACTORY_H_
