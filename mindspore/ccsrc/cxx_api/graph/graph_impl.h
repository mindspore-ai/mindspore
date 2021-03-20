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
#ifndef MINDSPORE_CCSRC_CXX_API_GRAPH_GRAPH_IMPL_H
#define MINDSPORE_CCSRC_CXX_API_GRAPH_GRAPH_IMPL_H
#include <functional>
#include <map>
#include <string>
#include <vector>
#include <memory>
#include <utility>
#include "include/api/cell.h"
#include "include/api/graph.h"
#include "cxx_api/graph/graph_data.h"
#include "utils/utils.h"

namespace mindspore {
class GraphCell::GraphImpl {
 public:
  GraphImpl() : graph_(nullptr) {}
  virtual ~GraphImpl() = default;

  std::shared_ptr<Graph::GraphData> &MutableGraphData() const { return graph_->graph_data_; }
  void SetGraph(const std::shared_ptr<Graph> &graph) { graph_ = graph; }

  virtual Status Run(const std::vector<MSTensor> &inputs, std::vector<MSTensor> *outputs) = 0;
  virtual Status Load(uint32_t device_id) = 0;

  virtual std::vector<MSTensor> GetInputs() = 0;
  virtual std::vector<MSTensor> GetOutputs() = 0;

 protected:
  std::shared_ptr<Graph> graph_;
};
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_CXX_API_GRAPH_GRAPH_IMPL_H
