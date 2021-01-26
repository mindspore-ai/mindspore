/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_CXX_API_GRAPH_GRAPH_DATA_H
#define MINDSPORE_CCSRC_CXX_API_GRAPH_GRAPH_DATA_H

#include <string>
#include <vector>
#include <map>
#include <memory>
#include "include/api/graph.h"
#include "include/api/types.h"
#include "ir/func_graph.h"

namespace mindspore {
class Graph::GraphData {
 public:
  GraphData();

  explicit GraphData(const FuncGraphPtr &func_graph, enum ModelType model_type = kMindIR);

  GraphData(Buffer om_data, enum ModelType model_type);

  ~GraphData();

  enum ModelType ModelType() const { return model_type_; }

  FuncGraphPtr GetFuncGraph() const;

  Buffer GetOMData() const;

 private:
  FuncGraphPtr func_graph_;
  Buffer om_data_;
  enum ModelType model_type_;
};
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_CXX_API_GRAPH_GRAPH_DATA_H
