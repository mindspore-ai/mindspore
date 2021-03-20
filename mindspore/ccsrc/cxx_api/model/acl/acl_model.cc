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
#include "include/api/context.h"
#include "cxx_api/factory.h"
#include "cxx_api/graph/acl/acl_env_guard.h"

namespace mindspore {
API_FACTORY_REG(ModelImpl, Ascend310, AclModel);

Status AclModel::Build() {
  MS_LOG(INFO) << "Start build model.";
  MS_EXCEPTION_IF_NULL(graph_);

  if (graph_cell_ != nullptr) {
    MS_LOG(INFO) << "This model has been built, skip.";
    return kSuccess;
  }

  if (graph_cell_ == nullptr && graph_->ModelType() == ModelType::kOM) {
    MS_LOG(INFO) << "Note: Load om model and all build options will be ignored.";
    graph_cell_ = std::make_shared<GraphCell>(graph_);
    MS_EXCEPTION_IF_NULL(graph_cell_);
    return kSuccess;
  }

  std::unique_ptr<AclModelOptions> options = std::make_unique<AclModelOptions>(model_context_);
  MS_EXCEPTION_IF_NULL(options);
  std::string dump_cfg = options->GetDumpCfgPath();
  if (!dump_cfg.empty()) {
    MS_LOG(INFO) << "Options dump config file path " << dump_cfg;
    (void)AclEnvGuard::GetAclEnv(dump_cfg);
  }
  std::string options_key = options->GenAclOptionsKey();
  std::shared_ptr<Graph> graph;
  if (auto iter = dynamic_size_graph_map_.find(options_key); iter != dynamic_size_graph_map_.end()) {
    MS_LOG(INFO) << "This options has been built, read cache.";
    graph = iter->second;
  } else {
    auto func_graph = ModelImpl::GetFuncGraph();
    auto inputs = func_graph->parameters();
    std::vector<std::string> input_names;
    for (auto node : inputs) {
      auto para = node->cast<ParameterPtr>();
      MS_EXCEPTION_IF_NULL(para);
      std::string name = para->name();
      for (auto pos = name.find(':'); pos != std::string::npos; pos = name.find(':')) {
        name = name.substr(0, pos) + "_" + name.substr(pos + 1);
        MS_LOG(INFO) << name;
      }
      para->set_name(name);
      input_names.push_back(name);
    }
    options->RenameInput(input_names);
    MS_EXCEPTION_IF_NULL(func_graph);
    model_converter_.set_options(options.get());
    auto om_data = model_converter_.LoadMindIR(func_graph);
    if (om_data.Data() == nullptr || om_data.DataSize() == 0) {
      MS_LOG(ERROR) << "Load MindIR failed.";
      return kMCFailed;
    }
    graph = std::make_shared<Graph>(std::make_shared<Graph::GraphData>(om_data, ModelType::kOM));
    dynamic_size_graph_map_[options_key] = graph;
  }

  MS_EXCEPTION_IF_NULL(graph);
  auto graph_cell = std::make_shared<GraphCell>(graph);
  MS_EXCEPTION_IF_NULL(graph_cell);
  auto ret = ModelImpl::Load(graph_cell, options->GetDeviceID());
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "Load failed.";
    return ret;
  }

  // save result
  graph_cell_ = graph_cell;
  options_ = std::move(options);
  MS_LOG(INFO) << "Build model success.";
  return kSuccess;
}

Status AclModel::Resize(const std::vector<MSTensor> &inputs, const std::vector<std::vector<int64_t>> &dims) {
  MS_LOG(INFO) << "Start to resize model.";
  MS_EXCEPTION_IF_NULL(graph_);
  if (graph_->ModelType() == ModelType::kOM) {
    MS_LOG(ERROR) << "OM model is not supported to resize model.";
    return kMCFailed;
  }

  auto origin_inputs = GetInputs();
  if (inputs.size() != origin_inputs.size()) {
    MS_LOG(ERROR) << "Invalid inputs size " << inputs.size() << " not match model inputs size " << origin_inputs.size();
    return kMCInvalidInput;
  }

  if (inputs.size() != dims.size()) {
    MS_LOG(ERROR) << "Invalid dims size " << dims.size() << " not match inputs size " << inputs.size();
    return kMCInvalidInput;
  }

  if (model_context_ == nullptr) {
    model_context_ = std::make_shared<Context>();
    model_context_->MutableDeviceInfo().emplace_back(std::make_shared<Ascend310DeviceInfo>());
  }

  std::string input_shape_option;
  for (size_t i = 0; i < inputs.size(); ++i) {
    if (inputs[i].Name() != origin_inputs[i].Name()) {
      MS_LOG(ERROR) << "Invalid inputs " << i << " name " << inputs[i].Name() << " not match model input name "
                    << origin_inputs[i].Name();
      return kMCInvalidInput;
    }
    input_shape_option += inputs[i].Name() + ":";
    for (size_t j = 0; j < dims[i].size(); ++j) {
      input_shape_option += std::to_string(dims[i][j]);
      if (j + 1 < dims[i].size()) {
        input_shape_option += ",";
      }
    }
    if (i + 1 < inputs.size()) {
      input_shape_option += ";";
    }
  }
  MS_LOG(INFO) << "Set input size option is " << input_shape_option;
  auto &device_infos = model_context_->MutableDeviceInfo();
  if (device_infos.size() != 1) {
    MS_LOG(ERROR) << "Invalid model context, only single device info is supported.";
    return kMCInvalidArgs;
  }
  auto ascend310_info = device_infos[0]->Cast<Ascend310DeviceInfo>();
  MS_EXCEPTION_IF_NULL(ascend310_info);
  ascend310_info->SetInputShape(input_shape_option);
  auto graph_cell_bak = std::move(graph_cell_);
  auto ret = Build();
  if (ret != kSuccess) {
    MS_LOG(INFO) << "Resize build failed.";
    graph_cell_ = std::move(graph_cell_bak);
    return ret;
  }
  MS_LOG(INFO) << "Resize success.";
  return kSuccess;
}

Status AclModel::Predict(const std::vector<MSTensor> &inputs, std::vector<MSTensor> *outputs) {
  MS_EXCEPTION_IF_NULL(outputs);
  if (graph_ == nullptr) {
    MS_LOG(ERROR) << "Invalid data, graph_ is null.";
    return kMCFailed;
  }

  if (graph_cell_ == nullptr) {
    MS_LOG(WARNING) << "Model has not been built, it will be built with default options";
    Status ret = Build();
    if (ret != kSuccess) {
      MS_LOG(ERROR) << "Build model failed.";
      return ret;
    }
  }

  MS_EXCEPTION_IF_NULL(graph_cell_);
  Status ret = graph_cell_->Run(inputs, outputs);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "Run graph failed.";
    return ret;
  }

  return kSuccess;
}

std::vector<MSTensor> AclModel::GetInputs() {
  MS_EXCEPTION_IF_NULL(graph_cell_);
  return graph_cell_->GetInputs();
}

std::vector<MSTensor> AclModel::GetOutputs() {
  MS_EXCEPTION_IF_NULL(graph_cell_);
  return graph_cell_->GetOutputs();
}
}  // namespace mindspore
