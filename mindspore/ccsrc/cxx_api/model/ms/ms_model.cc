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

#include "cxx_api/model/ms/ms_model.h"
#include <memory>
#include "utils/ms_context.h"
#include "cxx_api/factory.h"

namespace mindspore {
namespace api {
API_FACTORY_REG(ModelImpl, Ascend910, MsModel);
API_FACTORY_REG(ModelImpl, GPU, MsModel);

Status MsModel::Build(const std::map<std::string, std::string> &) {
  MS_LOG(INFO) << "Start build model.";
  MS_EXCEPTION_IF_NULL(graph_);

  auto func_graph = ModelImpl::GetFuncGraph();
  MS_EXCEPTION_IF_NULL(func_graph);

  auto graph = std::make_shared<Graph>(std::make_shared<Graph::GraphData>(func_graph, ModelType::kMindIR));
  MS_EXCEPTION_IF_NULL(graph);
  auto graph_cell = std::make_shared<GraphCell>(graph);
  MS_EXCEPTION_IF_NULL(graph_cell);
  auto ret = ModelImpl::Load(graph_cell);
  if (ret != SUCCESS) {
    MS_LOG(ERROR) << "Load failed.";
    return ret;
  }

  // save result
  graph_cell_ = graph_cell;
  MS_LOG(INFO) << "Build model success.";
  return SUCCESS;
}

Status MsModel::Train(const DataSet &, std::map<std::string, Buffer> *) {
  MS_LOG(ERROR) << "Unsupported feature.";
  return FAILED;
}

Status MsModel::Eval(const DataSet &, std::map<std::string, Buffer> *) {
  MS_LOG(ERROR) << "Unsupported feature.";
  return FAILED;
}

Status MsModel::Predict(const std::vector<Buffer> &inputs, std::vector<Buffer> *outputs) {
  MS_EXCEPTION_IF_NULL(outputs);
  if (graph_ == nullptr) {
    MS_LOG(ERROR) << "Invalid data, graph_ is null.";
    return FAILED;
  }

  if (graph_cell_ == nullptr) {
    MS_LOG(INFO) << "Model has not been built, it will be built with default options";
    Status ret = Build({});
    if (ret != SUCCESS) {
      MS_LOG(ERROR) << "Build model failed.";
      return FAILED;
    }
  }

  MS_EXCEPTION_IF_NULL(graph_cell_);
  Status ret = graph_cell_->Run(inputs, outputs);
  if (ret != SUCCESS) {
    MS_LOG(ERROR) << "Run graph failed.";
    return FAILED;
  }

  return SUCCESS;
}

Status MsModel::GetInputsInfo(std::vector<std::string> *names, std::vector<std::vector<int64_t>> *shapes,
                              std::vector<DataType> *data_types, std::vector<size_t> *mem_sizes) const {
  MS_EXCEPTION_IF_NULL(graph_cell_);
  return graph_cell_->GetInputsInfo(names, shapes, data_types, mem_sizes);
}

Status MsModel::GetOutputsInfo(std::vector<std::string> *names, std::vector<std::vector<int64_t>> *shapes,
                               std::vector<DataType> *data_types, std::vector<size_t> *mem_sizes) const {
  MS_EXCEPTION_IF_NULL(graph_cell_);
  return graph_cell_->GetOutputsInfo(names, shapes, data_types, mem_sizes);
}
}  // namespace api
}  // namespace mindspore
