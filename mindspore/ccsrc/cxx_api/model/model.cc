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
#include "cxx_api/model/model_impl.h"
#include "utils/utils.h"

namespace mindspore::api {
Status Model::LoadModel(const Buffer &model_data, ModelType type, const std::map<std::string, std::string> &options) {
  MS_EXCEPTION_IF_NULL(impl_);
  return impl_->LoadModel(model_data, type, options);
}

Status Model::LoadModel(const std::string &file_name, ModelType type,
                        const std::map<std::string, std::string> &options) {
  MS_EXCEPTION_IF_NULL(impl_);
  return impl_->LoadModel(file_name, type, options);
}

Status Model::UnloadModel() {
  MS_EXCEPTION_IF_NULL(impl_);
  return impl_->UnloadModel();
}

Status Model::Train(const DataSet &dataset, std::map<std::string, Buffer> *outputs) {
  MS_EXCEPTION_IF_NULL(impl_);
  return impl_->Train(dataset, outputs);
}

Status Model::Eval(const DataSet &dataset, std::map<std::string, Buffer> *outputs) {
  MS_EXCEPTION_IF_NULL(impl_);
  return impl_->Eval(dataset, outputs);
}

Status Model::Predict(const std::map<std::string, Buffer> &inputs, std::map<std::string, Buffer> *outputs) {
  MS_EXCEPTION_IF_NULL(impl_);
  return impl_->Predict(inputs, outputs);
}

Status Model::Predict(const std::vector<Buffer> &inputs, std::map<std::string, Buffer> *outputs) {
  std::vector<Tensor> tensor_list;
  auto ret = GetInputsInfo(&tensor_list);
  if (ret != SUCCESS) {
    MS_LOG(ERROR) << "GetInputsInfo failed.";
    return ret;
  }

  if (inputs.size() != tensor_list.size()) {
    MS_LOG(ERROR) << "Model need " << tensor_list.size() << " inputs, but given " << inputs.size();
    return FAILED;
  }

  std::map<std::string, Buffer> inputs_with_map;
  for (size_t i = 0; i < tensor_list.size(); ++i) {
    inputs_with_map.emplace(tensor_list[i].Name(), inputs[i]);
  }

  return Predict(inputs_with_map, outputs);
}

Status Model::GetInputsInfo(std::vector<Tensor> *tensor_list) const {
  MS_EXCEPTION_IF_NULL(impl_);
  return impl_->GetInputsInfo(tensor_list);
}

Status Model::GetOutputsInfo(std::vector<Tensor> *tensor_list) const {
  MS_EXCEPTION_IF_NULL(impl_);
  return impl_->GetOutputsInfo(tensor_list);
}

Model::Model(const std::string &device_type, uint32_t device_id)
    : impl_(ModelFactory::Instance().Create(device_type, device_id)) {
  if (impl_ == nullptr) {
    MS_LOG(EXCEPTION) << "Create session type " << device_type << " failed";
  }
}

Model::Model(NetWork network, const std::string &device_type, uint32_t device_id) {
  // todo
  if (impl_ == nullptr) {
    MS_LOG(EXCEPTION) << "Create session type " << device_type << " failed";
  }
}

Model::~Model() {}
}  // namespace mindspore::api
