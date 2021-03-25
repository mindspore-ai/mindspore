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
const std::map<enum DeviceType, std::set<ModelType>> kSupportedModelMap = {
  {kAscend310, {kOM, kMindIR}},
  {kAscend910, {kMindIR}},
  {kNvidiaGPU, {kMindIR}},
};

std::string GetDeviceTypeString(enum DeviceType type) {
  static const std::map<enum DeviceType, std::string> kDeviceTypeStrs = {
    {kCPU, "CPU"},           {kMaliGPU, "MaliGPU"},     {kNvidiaGPU, "GPU"},
    {kKirinNPU, "KirinGPU"}, {kAscend910, "Ascend910"}, {kAscend310, "Ascend310"},
  };
  auto iter = kDeviceTypeStrs.find(type);
  if (iter != kDeviceTypeStrs.end()) {
    return iter->second;
  }

  return "InvalidDeviceType" + std::to_string(type);
}
}  // namespace
Status Model::Build(GraphCell graph_cell, const std::shared_ptr<Context> &model_context) {
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

  std::string device_target = GetDeviceTypeString(device_info[0]->GetDeviceType());
  impl_ = Factory<ModelImpl>::Instance().Create(device_target);
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Create session type " << device_target << " failed";
    return kMEFailed;
  }

  g_device_target = device_target;

  impl_->SetGraph(std::make_shared<Graph>(*graph_cell.GetGraph()));
  impl_->SetContext(model_context);

  return impl_->Build();
}

Status Model::Resize(const std::vector<MSTensor> &inputs, const std::vector<std::vector<int64_t>> &dims) {
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Failed because this model has not been built.";
    return kMCFailed;
  }
  return impl_->Resize(inputs, dims);
}

Status Model::Predict(const std::vector<MSTensor> &inputs, std::vector<MSTensor> *outputs) {
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Failed because this model has not been built.";
    return kMCFailed;
  }
  return impl_->Predict(inputs, outputs);
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
                 [](MSTensor item) -> std::vector<char> { return StringToChar(item.Name()); });
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

Model::Model() : impl_(nullptr) {}
Model::~Model() {}

bool Model::CheckModelSupport(enum DeviceType device_type, ModelType model_type) {
  std::string device_type_str = GetDeviceTypeString(device_type);
  if (!Factory<ModelImpl>::Instance().CheckModelSupport(device_type_str)) {
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
