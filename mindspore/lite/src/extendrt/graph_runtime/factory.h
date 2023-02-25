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
#ifndef MINDSPORE_LITE_SRC_EXTENDRT_GRAPH_RUNTIME_FACTORY_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_GRAPH_RUNTIME_FACTORY_H_

#include <functional>
#include <memory>

#include "extendrt/graph_runtime/type.h"
#include "infer/graph_runtime.h"

namespace mindspore {
using GraphRuntime = infer::abstract::GraphRuntime;
using GraphRuntimeRegFunc = std::function<std::shared_ptr<GraphRuntime>()>;

class GraphRuntimRegistry {
 public:
  GraphRuntimRegistry() = default;
  virtual ~GraphRuntimRegistry() = default;

  static GraphRuntimRegistry &GetInstance();

  void RegRuntime(const GraphRuntimeType &type, const GraphRuntimeRegFunc &creator);

  std::shared_ptr<GraphRuntime> GetRuntime(const mindspore::GraphRuntimeType &type);

 private:
  mindspore::HashMap<GraphRuntimeType, GraphRuntimeRegFunc> graph_runtime_map_;
};

class GraphRuntimeRegistrar {
 public:
  GraphRuntimeRegistrar(const mindspore::GraphRuntimeType &type, const GraphRuntimeRegFunc &creator) {
    GraphRuntimRegistry::GetInstance().RegRuntime(type, creator);
  }
  ~GraphRuntimeRegistrar() = default;
};

#define REG_GRAPH_RUNTIME(type, creator) static GraphRuntimeRegistrar g_##type##GraphRuntime(type, creator);
}  // namespace mindspore

#endif  // MINDSPORE_LITE_SRC_EXTENDRT_GRAPH_RUNTIME_FACTORY_H_
