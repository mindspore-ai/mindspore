/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_LITE_SRC_EXTENDRT_GRAPH_PARTITIONER_FACTORY_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_GRAPH_PARTITIONER_FACTORY_H_

#include <functional>
#include <memory>

#include "extendrt/graph_partitioner/type.h"
#include "infer/graph_partitioner.h"

namespace mindspore {
using GraphPartitionerRegFunc = std::function<std::shared_ptr<GraphPartitioner>()>;

class GraphPartitionerRegistry {
 public:
  GraphPartitionerRegistry() = default;
  virtual ~GraphPartitionerRegistry() = default;

  static GraphPartitionerRegistry &GetInstance();

  void RegPartitioner(const GraphPartitionerType &type, const GraphPartitionerRegFunc &creator);

  std::shared_ptr<GraphPartitioner> GetPartitioner(const mindspore::GraphPartitionerType &type);

 private:
  mindspore::HashMap<GraphPartitionerType, GraphPartitionerRegFunc> graph_partitioner_map_;
};

class GraphPartitionerRegistrar {
 public:
  GraphPartitionerRegistrar(const mindspore::GraphPartitionerType &type, const GraphPartitionerRegFunc &creator) {
    GraphPartitionerRegistry::GetInstance().RegPartitioner(type, creator);
  }
  ~GraphPartitionerRegistrar() = default;
};

#define REG_GRAPH_PARTITIONER(type, creator) static GraphPartitionerRegistrar g_##type##Partitioner(type, creator);
}  // namespace mindspore

#endif  // MINDSPORE_LITE_SRC_EXTENDRT_GRAPH_PARTITIONER_FACTORY_H_
