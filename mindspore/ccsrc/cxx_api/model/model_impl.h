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
#include "include/api/context.h"
#include "include/api/model.h"
#include "include/api/graph.h"
#include "cxx_api/graph/graph_data.h"
#include "utils/utils.h"
#include "ir/func_graph.h"

namespace mindspore {
class ModelImpl {
 public:
  ModelImpl() = default;
  virtual ~ModelImpl() = default;

  virtual Status Build() = 0;
  virtual Status Resize(const std::vector<MSTensor> &inputs, const std::vector<std::vector<int64_t>> &dims) = 0;

  virtual Status Predict(const std::vector<MSTensor> &inputs, std::vector<MSTensor> *outputs) = 0;

  virtual std::vector<MSTensor> GetInputs() = 0;
  virtual std::vector<MSTensor> GetOutputs() = 0;

 protected:
  Status Load(const std::shared_ptr<GraphCell> &graph_cell, uint32_t device_id) {
    MS_EXCEPTION_IF_NULL(graph_cell);
    return graph_cell->Load(device_id);
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
  std::shared_ptr<Context> model_context_;

 private:
  friend class Model;
  void SetGraph(const std::shared_ptr<Graph> &graph) { graph_ = graph; }
  void SetContext(const std::shared_ptr<Context> &model_context) {
    if (model_context != nullptr) {
      model_context_ = std::make_shared<Context>(*model_context);
    }
  }
};
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_CXX_API_MODEL_MODEL_IMPL_H
