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
#include "include/common/utils/utils.h"

namespace mindspore {
Status Model::Build(GraphCell graph_cell, const std::shared_ptr<Context> &model_context,
                    const std::shared_ptr<TrainCfg> &) {
  if (graph_cell.GetGraph() == nullptr) {
    MS_LOG(ERROR) << "Invalid graph input.";
    return kMCInvalidInput;
  }

  if (model_context == nullptr) {
    MS_LOG(ERROR) << "Invalid model context.";
    return kMCInvalidInput;
  }
  auto &device_info = model_context->MutableDeviceInfo();
  if (device_info.size() != 1) {
    MS_LOG(ERROR) << "Invalid model context, only single device info is supported.";
    return kMCInvalidInput;
  }

  auto device_target = device_info[0]->GetDeviceType();
  impl_ = ModelImplFactory::Instance().Create(device_target);
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Create session type " << device_target << " failed";
    return kMEFailed;
  }

  g_device_target = device_target;

  impl_->SetGraph(std::make_shared<Graph>(*graph_cell.GetGraph()));
  impl_->SetContext(model_context);

  return impl_->Build();
}

Status Model::Build(const std::vector<char> &, ModelType, const std::shared_ptr<Context> &, const Key &,
                    const std::vector<char> &, const std::vector<char> &) {
  MS_LOG(ERROR) << "Unsupported Feature.";
  return kMCFailed;
}

Status Model::Build(const std::vector<char> &, ModelType, const std::shared_ptr<Context> &) {
  MS_LOG(ERROR) << "Unsupported Feature.";
  return kMCFailed;
}

Status Model::Build(const void * /* model_data */, size_t /* data_size */, ModelType /* model_type */,
                    const std::shared_ptr<Context> & /* model_context */, const Key & /* dec_key */,
                    const std::vector<char> & /* dec_mode */, const std::vector<char> & /* cropto_lib_path */) {
  MS_LOG(ERROR) << "Unsupported Feature.";
  return kMCFailed;
}

Status Model::Resize(const std::vector<MSTensor> &inputs, const std::vector<std::vector<int64_t>> &dims) {
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Failed because this model has not been built.";
    return kMCFailed;
  }
  return impl_->Resize(inputs, dims);
}

Status Model::Predict(const std::vector<MSTensor> &inputs, std::vector<MSTensor> *outputs,
                      const MSKernelCallBack & /* before */, const MSKernelCallBack & /* after */) {
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Failed because this model has not been built.";
    return kMCFailed;
  }
  return impl_->Predict(inputs, outputs);
}

Status Model::PredictWithPreprocess(const std::vector<std::vector<MSTensor>> &inputs, std::vector<MSTensor> *outputs,
                                    const MSKernelCallBack & /* before */, const MSKernelCallBack & /* after */) {
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Failed because this model has not been built.";
    return kMCFailed;
  }
  return impl_->PredictWithPreprocess(inputs, outputs);
}

Status Model::Preprocess(const std::vector<std::vector<MSTensor>> &inputs, std::vector<MSTensor> *outputs) {
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Failed because this model has not been built.";
    return kMCFailed;
  }
  return impl_->Preprocess(inputs, outputs);
}

bool Model::HasPreprocess() {
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Failed because this model has not been built.";
    return false;
  }
  return impl_->HasPreprocess();
}

std::vector<MSTensor> Model::GetInputs() {
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Failed because this model has not been built.";
    return {};
  }
  return impl_->GetInputs();
}

std::vector<MSTensor> Model::GetOutputs() {
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Failed because this model has not been built.";
    return {};
  }
  return impl_->GetOutputs();
}

MSTensor Model::GetInputByTensorName(const std::vector<char> &tensor_name) {
  std::string tensor_name_str = CharToString(tensor_name);
  auto inputs = GetInputs();
  for (auto in : inputs) {
    if (in.Name() == tensor_name_str) {
      return in;
    }
  }

  return MSTensor(nullptr);
}

std::vector<std::vector<char>> Model::GetOutputTensorNamesChar() {
  std::vector<std::vector<char>> ret;
  auto outputs = GetOutputs();
  std::transform(outputs.begin(), outputs.end(), std::back_inserter(ret),
                 [](const MSTensor &item) -> std::vector<char> { return StringToChar(item.Name()); });
  return ret;
}

MSTensor Model::GetOutputByTensorName(const std::vector<char> &tensor_name) {
  std::string tensor_name_str = CharToString(tensor_name);
  auto outputs = GetOutputs();
  for (auto out : outputs) {
    if (out.Name() == tensor_name_str) {
      return out;
    }
  }

  return MSTensor(nullptr);
}

std::vector<MSTensor> Model::GetOutputsByNodeName(const std::vector<char> &node_name) {
  return std::vector<MSTensor>{GetOutputByTensorName(node_name)};
}

Model::Model() : impl_(nullptr) {}
Model::~Model() {}

bool Model::CheckModelSupport(enum DeviceType device_type, ModelType model_type) {
  auto check_model = ModelImplFactory::Instance().Create(device_type);
  if (check_model == nullptr) {
    return false;
  }
  return check_model->CheckModelSupport(model_type);
}

Status Model::LoadConfig(const std::vector<char> & /* config_path */) {
  MS_LOG(ERROR) << "Unsupported Feature.";
  return kMCFailed;
}

#ifdef _MSC_VER
Status Model::UpdateConfig(const std::vector<char> &section,
                           const std::pair<std::vector<char>, std::vector<char>> &config) {
  MS_LOG(ERROR) << "Model::UpdateConfig Unsupported on msvc.";
  return kMCFailed;
}

#endif
}  // namespace mindspore
