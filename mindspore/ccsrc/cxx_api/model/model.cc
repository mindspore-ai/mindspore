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

namespace mindspore::api {
Status Model::Build(const std::map<std::string, std::string> &options) {
  MS_EXCEPTION_IF_NULL(impl_);
  return impl_->Build(options);
}

Status Model::Train(const DataSet &dataset, bool data_sink, std::map<std::string, Buffer> *outputs) {
  MS_EXCEPTION_IF_NULL(impl_);
  return impl_->Train(dataset, outputs);
}

Status Model::Eval(const DataSet &dataset, bool data_sink, std::map<std::string, Buffer> *outputs) {
  MS_EXCEPTION_IF_NULL(impl_);
  return impl_->Eval(dataset, outputs);
}

Status Model::Predict(const std::vector<Buffer> &inputs, std::vector<Buffer> *outputs) {
  MS_EXCEPTION_IF_NULL(impl_);
  return impl_->Predict(inputs, outputs);
}

Status Model::GetInputsInfo(std::vector<std::string> *names, std::vector<std::vector<int64_t>> *shapes,
                            std::vector<DataType> *data_types, std::vector<size_t> *mem_sizes) const {
  MS_EXCEPTION_IF_NULL(impl_);
  return impl_->GetInputsInfo(names, shapes, data_types, mem_sizes);
}

Status Model::GetOutputsInfo(std::vector<std::string> *names, std::vector<std::vector<int64_t>> *shapes,
                             std::vector<DataType> *data_types, std::vector<size_t> *mem_sizes) const {
  MS_EXCEPTION_IF_NULL(impl_);
  return impl_->GetOutputsInfo(names, shapes, data_types, mem_sizes);
}

Model::Model(const GraphCell &graph_cell)
    : impl_(Factory<ModelImpl>::Instance().Create(Context::Instance().GetDeviceTarget())) {
  if (impl_ == nullptr) {
    MS_LOG(EXCEPTION) << "Create session type " << Context::Instance().GetDeviceTarget() << " failed";
  }
  MS_EXCEPTION_IF_NULL(graph_cell.GetGraph());
  impl_->SetGraph(std::make_shared<Graph>(*graph_cell.GetGraph()));
}

Model::Model(const std::vector<Output> &network) { MS_LOG(EXCEPTION) << "Unsupported feature."; }

Model::~Model() {}

bool Model::CheckModelSupport(const std::string &device_type, ModelType) {
  return Factory<ModelImpl>::Instance().CheckModelSupport(device_type);
}

}  // namespace mindspore::api
