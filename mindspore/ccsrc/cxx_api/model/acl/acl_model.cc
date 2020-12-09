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

#include "cxx_api/model/acl/acl_model.h"
#include <memory>
#include "cxx_api/factory.h"
#include "cxx_api/python_utils.h"

namespace mindspore::api {
API_FACTORY_REG(ModelImpl, Ascend310, AclModel);

Status AclModel::Build(const std::map<std::string, std::string> &options_map) {
  MS_LOG(INFO) << "Start build model.";
  MS_EXCEPTION_IF_NULL(graph_);
  std::unique_ptr<AclModelOptions> options = std::make_unique<AclModelOptions>(options_map);
  std::string options_str = GenerateOptionsStr(options_map);
  MS_EXCEPTION_IF_NULL(options);
  if (graph_cell_ != nullptr && options_str == options_str_) {
    MS_LOG(INFO) << "This model has been built, skip.";
    return SUCCESS;
  }

  if (graph_cell_ == nullptr && graph_->ModelType() == ModelType::kOM) {
    graph_cell_ = std::make_shared<GraphCell>(graph_);
    MS_EXCEPTION_IF_NULL(graph_cell_);
    if (!options_map.empty()) {
      MS_LOG(WARNING) << "All build options will be ignored.";
    }
    return SUCCESS;
  }

  auto func_graph = ModelImpl::GetFuncGraph();
  MS_EXCEPTION_IF_NULL(func_graph);
  model_converter_.set_options(options.get());
  auto om_data = model_converter_.LoadMindIR(func_graph);
  if (om_data.Data() == nullptr || om_data.DataSize() == 0) {
    MS_LOG(ERROR) << "Load MindIR failed.";
    return FAILED;
  }

  auto graph = std::make_shared<Graph>(std::make_shared<Graph::GraphData>(om_data, ModelType::kOM));
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
  options_ = std::move(options);
  options_str_ = options_str;
  MS_LOG(INFO) << "Build model success.";
  return SUCCESS;
}

Status AclModel::Train(const DataSet &, std::map<std::string, Buffer> *) {
  MS_LOG(ERROR) << "Unsupported feature.";
  return FAILED;
}

Status AclModel::Eval(const DataSet &, std::map<std::string, Buffer> *) {
  MS_LOG(ERROR) << "Unsupported feature.";
  return FAILED;
}

Status AclModel::Predict(const std::vector<Buffer> &inputs, std::vector<Buffer> *outputs) {
  MS_EXCEPTION_IF_NULL(outputs);
  if (graph_ == nullptr) {
    MS_LOG(ERROR) << "Invalid data, graph_ is null.";
    return FAILED;
  }

  if (graph_cell_ == nullptr) {
    MS_LOG(WARNING) << "Model has not been built, it will be built with default options";
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

Status AclModel::GetInputsInfo(std::vector<std::string> *names, std::vector<std::vector<int64_t>> *shapes,
                               std::vector<DataType> *data_types, std::vector<size_t> *mem_sizes) const {
  MS_EXCEPTION_IF_NULL(graph_cell_);
  return graph_cell_->GetInputsInfo(names, shapes, data_types, mem_sizes);
}

Status AclModel::GetOutputsInfo(std::vector<std::string> *names, std::vector<std::vector<int64_t>> *shapes,
                                std::vector<DataType> *data_types, std::vector<size_t> *mem_sizes) const {
  MS_EXCEPTION_IF_NULL(graph_cell_);
  return graph_cell_->GetOutputsInfo(names, shapes, data_types, mem_sizes);
}

std::string AclModel::GenerateOptionsStr(const std::map<std::string, std::string> &options) {
  std::string ret;
  for (auto &[key, value] : options) {
    ret += key + "^" + value + "^^";
  }
  return ret;
}
}  // namespace mindspore::api
