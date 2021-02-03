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
#include "include/api/context.h"
#include "cxx_api/model/model_impl.h"
#include "cxx_api/factory.h"
#include "utils/utils.h"

namespace mindspore {
namespace {
const std::map<std::string, std::set<ModelType>> kSupportedModelMap = {
  {kDeviceTypeAscend310, {kOM, kMindIR}},
  {kDeviceTypeAscend910, {kMindIR}},
  {kDeviceTypeGPU, {kMindIR}},
};
}
Status Model::Build() {
  MS_EXCEPTION_IF_NULL(impl_);
  return impl_->Build();
}

Status Model::Resize(const std::vector<MSTensor> &inputs, const std::vector<std::vector<int64_t>> &dims) {
  MS_EXCEPTION_IF_NULL(impl_);
  return impl_->Resize(inputs, dims);
}

Status Model::Predict(const std::vector<MSTensor> &inputs, std::vector<MSTensor> *outputs) {
  MS_EXCEPTION_IF_NULL(impl_);
  return impl_->Predict(inputs, outputs);
}

std::vector<MSTensor> Model::GetInputs() {
  MS_EXCEPTION_IF_NULL(impl_);
  return impl_->GetInputs();
}

std::vector<MSTensor> Model::GetOutputs() {
  MS_EXCEPTION_IF_NULL(impl_);
  return impl_->GetOutputs();
}

Model::Model(const GraphCell &graph_cell, const std::shared_ptr<Context> &model_context)
    : impl_(Factory<ModelImpl>::Instance().Create(mindspore::GlobalContext::GetGlobalDeviceTarget())) {
  if (impl_ == nullptr) {
    MS_LOG(EXCEPTION) << "Create session type " << mindspore::GlobalContext::GetGlobalDeviceTarget() << " failed";
  }
  MS_EXCEPTION_IF_NULL(graph_cell.GetGraph());
  impl_->SetGraph(std::make_shared<Graph>(*graph_cell.GetGraph()));
  impl_->SetContext(model_context);
}

Model::Model(const std::vector<Output> &network, const std::shared_ptr<Context> &model_context) {
  MS_LOG(EXCEPTION) << "Unsupported feature.";
}

Model::~Model() {}

bool Model::CheckModelSupport(const std::string &device_type, ModelType model_type) {
  if (!Factory<ModelImpl>::Instance().CheckModelSupport(device_type)) {
    return false;
  }

  auto first_iter = kSupportedModelMap.find(device_type);
  if (first_iter == kSupportedModelMap.end()) {
    return false;
  }

  auto secend_iter = first_iter->second.find(model_type);
  if (secend_iter == first_iter->second.end()) {
    return false;
  }

  return true;
}
}  // namespace mindspore
