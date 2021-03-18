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
#include "include/api/context.h"
#include "utils/ms_context.h"
#include "cxx_api/factory.h"

namespace mindspore {
// mindspore-serving check current package for version check with ModelImpl factory.
#if ENABLE_D
API_FACTORY_REG(ModelImpl, Ascend910, MsModel);
#elif ENABLE_GPU
API_FACTORY_REG(ModelImpl, GPU, MsModel);
#endif

static std::string GenerateShapeKey(const std::vector<std::vector<int64_t>> &dims) {
  std::string shape_key;
  for (size_t i = 0; i < dims.size(); ++i) {
    shape_key += std::to_string(i) + ":";
    for (size_t j = 0; j < dims[i].size(); ++j) {
      shape_key += std::to_string(dims[i][j]);
      if (j + 1 < dims[i].size()) {
        shape_key += ",";
      }
    }
    if (i + 1 < dims.size()) {
      shape_key += ";";
    }
  }
  return shape_key;
}

std::shared_ptr<GraphCell> MsModel::GenerateGraphCell(const std::vector<std::vector<int64_t>> &dims) {
  std::string shape_key = GenerateShapeKey(dims);
  if (auto iter = dynamic_size_graph_map_.find(shape_key); iter != dynamic_size_graph_map_.end()) {
    MS_LOG(INFO) << "This options has been built, read cache.";
    return iter->second;
  }

  auto func_graph = ModelImpl::GetFuncGraph();
  MS_EXCEPTION_IF_NULL(func_graph);

  const auto &inputs = func_graph->parameters();
  if (dims.size() != inputs.size()) {
    MS_LOG(ERROR) << "Invalid dims size " << dims.size() << " not match model inputs size " << inputs.size();
    return nullptr;
  }
  for (size_t i = 0; i < dims.size(); ++i) {
    const auto &param = inputs[i];
    auto shape_ptr = std::dynamic_pointer_cast<abstract::Shape>(param->Shape());
    if (shape_ptr == nullptr) {
      MS_LOG(ERROR) << "Inputs " << i << " is not supported to resize, debug string: " << param->DebugString();
      return nullptr;
    }
    shape_ptr->shape() = dims[i];
  }

  auto graph = std::make_shared<Graph>(std::make_shared<Graph::GraphData>(func_graph, ModelType::kMindIR));
  MS_EXCEPTION_IF_NULL(graph);
  auto graph_cell = std::make_shared<GraphCell>(graph);
  MS_EXCEPTION_IF_NULL(graph_cell);
  auto ret = ModelImpl::Load(graph_cell, GetDeviceID());
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "Load failed.";
    return nullptr;
  }
  dynamic_size_graph_map_[shape_key] = graph_cell;
  return graph_cell;
}

Status MsModel::Build() {
  MS_LOG(INFO) << "Start build model.";
  MS_EXCEPTION_IF_NULL(graph_);

  if (graph_cell_ != nullptr) {
    MS_LOG(INFO) << "This model has been built, skip.";
    return kSuccess;
  }

  auto func_graph = ModelImpl::GetFuncGraph();
  MS_EXCEPTION_IF_NULL(func_graph);

  auto graph = std::make_shared<Graph>(std::make_shared<Graph::GraphData>(func_graph, ModelType::kMindIR));
  MS_EXCEPTION_IF_NULL(graph);
  auto graph_cell = std::make_shared<GraphCell>(graph);
  MS_EXCEPTION_IF_NULL(graph_cell);
  auto ret = ModelImpl::Load(graph_cell, GetDeviceID());
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "Load failed.";
    return ret;
  }

  // save result
  graph_cell_ = graph_cell;
  MS_LOG(INFO) << "Build model success.";
  return kSuccess;
}

Status MsModel::Resize(const std::vector<MSTensor> &inputs, const std::vector<std::vector<int64_t>> &dims) {
  MS_LOG(INFO) << "Start to resize model";
  auto origin_inputs = GetInputs();
  if (inputs.size() != origin_inputs.size()) {
    MS_LOG(ERROR) << "Invalid inputs size " << inputs.size() << " not match model inputs size " << origin_inputs.size();
    return kMCInvalidInput;
  }

  if (inputs.size() != dims.size()) {
    MS_LOG(ERROR) << "Invalid dims size " << dims.size() << " not match inputs size " << inputs.size();
    return kMCInvalidInput;
  }

  auto graph_cell = GenerateGraphCell(dims);
  if (graph_cell == nullptr) {
    MS_LOG(ERROR) << "GenerateGraphCell failed.";
    return kMCFailed;
  }

  MS_LOG(INFO) << "Resize model success.";
  graph_cell_ = std::move(graph_cell);
  return kSuccess;
}

Status MsModel::Predict(const std::vector<MSTensor> &inputs, std::vector<MSTensor> *outputs) {
  MS_EXCEPTION_IF_NULL(outputs);
  if (graph_ == nullptr) {
    MS_LOG(ERROR) << "Invalid data, graph_ is null.";
    return kMCFailed;
  }

  if (graph_cell_ == nullptr) {
    MS_LOG(INFO) << "Model has not been built, it will be built with default options";
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

std::vector<MSTensor> MsModel::GetInputs() {
  MS_EXCEPTION_IF_NULL(graph_cell_);
  return graph_cell_->GetInputs();
}

std::vector<MSTensor> MsModel::GetOutputs() {
  MS_EXCEPTION_IF_NULL(graph_cell_);
  return graph_cell_->GetOutputs();
}

uint32_t MsModel::GetDeviceID() const {
  if (model_context_ == nullptr) {
    return 0;
  }

  auto &device_infos = model_context_->MutableDeviceInfo();
  if (device_infos.size() != 1) {
    return 0;
  }

  auto ascend910_info = device_infos[0]->Cast<Ascend910DeviceInfo>();
  if (ascend910_info != nullptr) {
    return ascend910_info->GetDeviceID();
  }

  auto gpu_info = device_infos[0]->Cast<NvidiaGPUDeviceInfo>();
  if (gpu_info != nullptr) {
    return gpu_info->GetDeviceID();
  }

  return 0;
}
}  // namespace mindspore
