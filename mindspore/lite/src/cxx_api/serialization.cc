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

#include "include/api/serialization.h"
#include <algorithm>
#include <queue>
#include <set>
#include "include/api/graph.h"
#include "include/api/lite_context.h"
#include "include/api/types.h"
#include "include/model.h"
#include "include/ms_tensor.h"
#include "src/cxx_api/graph/graph_data.h"
#include "src/common/log_adapter.h"

namespace mindspore {

Graph Serialization::LoadModel(const void *model_data, size_t data_size, ModelType model_type) {
  if (model_type != kMindIR) {
    MS_LOG(ERROR) << "Unsupported IR.";
    return Graph(nullptr);
  }
  auto model = std::shared_ptr<lite::Model>(lite::Model::Import(static_cast<const char *>(model_data), data_size));
  if (model == nullptr) {
    MS_LOG(ERROR) << "New model failed.";
    return Graph(nullptr);
  }
  auto graph_data = std::shared_ptr<Graph::GraphData>(new (std::nothrow) Graph::GraphData(model));
  if (graph_data == nullptr) {
    MS_LOG(ERROR) << "New graph data failed.";
    return Graph(nullptr);
  }
  Graph graph = Graph(graph_data);
  return graph;
}

Graph Serialization::LoadModel(const std::string &file, ModelType model_type) {
  MS_LOG(ERROR) << "Unsupported Feature.";
  return Graph(nullptr);
}

Status Serialization::LoadCheckPoint(const std::string &ckpt_file, std::map<std::string, Buffer> *parameters) {
  MS_LOG(ERROR) << "Unsupported feature.";
  return kMEFailed;
}

Status Serialization::SetParameters(const std::map<std::string, Buffer> &parameters, Model *model) {
  MS_LOG(ERROR) << "Unsupported feature.";
  return kMEFailed;
}

Status Serialization::ExportModel(const Model &model, ModelType model_type, Buffer *model_data) {
  MS_LOG(ERROR) << "Unsupported feature.";
  return kMEFailed;
}

Status Serialization::ExportModel(const Model &model, ModelType model_type, const std::string &model_file) {
  MS_LOG(ERROR) << "Unsupported feature.";
  return kMEFailed;
}
}  // namespace mindspore
