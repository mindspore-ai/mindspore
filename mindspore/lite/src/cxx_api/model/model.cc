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
#include "include/api/model.h"
#include "include/api/lite_context.h"
#include "src/cxx_api/model/model_impl.h"
#include "src/common/log_adapter.h"

namespace mindspore {

Status Model::Build() {
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Model implement is null.";
    return kLiteNullptr;
  }
  return impl_->Build();
}

Status Model::Resize(const std::vector<MSTensor> &inputs, const std::vector<std::vector<int64_t>> &dims) {
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Model implement is null.";
    return kLiteNullptr;
  }
  return impl_->Resize(inputs, dims);
}

Status Model::Predict(const std::vector<MSTensor> &inputs, std::vector<MSTensor> *outputs) {
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Model implement is null.";
    return kLiteNullptr;
  }
  return impl_->Predict(inputs, outputs);
}

Model::Model(const GraphCell &graph, const std::shared_ptr<Context> &model_context) {
  impl_ = std::shared_ptr<ModelImpl>(new (std::nothrow) ModelImpl());
  if (impl_ == nullptr || graph.GetGraph() == nullptr) {
    MS_LOG(ERROR) << "Invalid graph.";
  } else if (model_context == nullptr) {
    MS_LOG(ERROR) << "Invalid context.";
  } else {
    auto new_graph_cell = std::shared_ptr<GraphCell>(new (std::nothrow) GraphCell(graph));
    if (new_graph_cell != nullptr) {
      impl_->SetContext(model_context);
      impl_->SetGraphCell(new_graph_cell);
    } else {
      MS_LOG(ERROR) << "New graphcell failed.";
    }
  }
}

Model::Model(const std::vector<Output> &network, const std::shared_ptr<Context> &model_context) {
  MS_LOG(ERROR) << "Unsupported feature.";
}

Model::~Model() {}

bool Model::CheckModelSupport(const std::string &device_type, ModelType) {
  MS_LOG(ERROR) << "Unsupported feature.";
  return false;
}

std::vector<MSTensor> Model::GetInputs() {
  std::vector<MSTensor> empty;
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Model implement is null.";
    return empty;
  }
  return impl_->GetInputs();
}

std::vector<MSTensor> Model::GetOutputs() {
  std::vector<MSTensor> empty;
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Model implement is null.";
    return empty;
  }
  return impl_->GetOutputs();
}

}  // namespace mindspore
