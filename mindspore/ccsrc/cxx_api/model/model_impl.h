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
#ifndef MINDSPORE_CCSRC_CXX_API_MODEL_MODEL_IMPL_H
#define MINDSPORE_CCSRC_CXX_API_MODEL_MODEL_IMPL_H
#include <functional>
#include <map>
#include <string>
#include <vector>
#include <memory>
#include <utility>
#include "include/api/model.h"
#include "include/api/graph.h"
#include "cxx_api/graph/graph_data.h"
#include "utils/utils.h"
#include "ir/func_graph.h"

namespace mindspore::api {
class ModelImpl {
 public:
  ModelImpl() = default;
  virtual ~ModelImpl() = default;

  virtual Status Build(const std::map<std::string, std::string> &options) = 0;

  virtual Status Train(const DataSet &dataset, std::map<std::string, Buffer> *outputs) = 0;
  virtual Status Eval(const DataSet &dataset, std::map<std::string, Buffer> *outputs) = 0;
  virtual Status Predict(const std::vector<Buffer> &inputs, std::vector<Buffer> *outputs) = 0;

  virtual Status GetInputsInfo(std::vector<std::string> *names, std::vector<std::vector<int64_t>> *shapes,
                               std::vector<DataType> *data_types, std::vector<size_t> *mem_sizes) const = 0;
  virtual Status GetOutputsInfo(std::vector<std::string> *names, std::vector<std::vector<int64_t>> *shapes,
                                std::vector<DataType> *data_types, std::vector<size_t> *mem_sizes) const = 0;

 protected:
  Status Load(const std::shared_ptr<GraphCell> &graph_cell) {
    MS_EXCEPTION_IF_NULL(graph_cell);
    return graph_cell->Load();
  }

  FuncGraphPtr GetFuncGraph() const {
    if (graph_->ModelType() != ModelType::kMindIR) {
      return nullptr;
    }

    auto graph_data = graph_->graph_data_;
    MS_EXCEPTION_IF_NULL(graph_data);
    return graph_data->GetFuncGraph();
  }

  std::shared_ptr<Graph> graph_;

 private:
  friend class Model;
  void SetGraph(const std::shared_ptr<Graph> &graph) { graph_ = graph; }
};
}  // namespace mindspore::api

#endif  // MINDSPORE_CCSRC_CXX_API_MODEL_MODEL_IMPL_H
