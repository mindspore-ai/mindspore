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

#include "src/cxx_api/model/model_impl.h"
#include <memory>
#include <unordered_map>
#include <algorithm>
#include "include/api/types.h"
#include "include/api/lite_context.h"
#include "include/lite_session.h"
#include "include/context.h"
#include "src/lite_model.h"
#include "src/runtime/allocator.h"
#include "src/cxx_api/utils.h"
#include "src/cxx_api/graph/graph_data.h"
#include "src/cxx_api/tensor/tensor_impl.h"
#include "src/common/log_adapter.h"

namespace mindspore {
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;

Status ModelImpl::Build() {
  MS_LOG(DEBUG) << "Start build model.";
  if (session_ != nullptr) {
    MS_LOG(DEBUG) << "Model has been already built.";
    return kSuccess;
  }
  if (graph_cell_ == nullptr || graph_cell_->GetGraph() == nullptr || graph_cell_->GetGraph()->graph_data_ == nullptr) {
    MS_LOG(ERROR) << "Graph cell is invalid.";
    return kLiteNullptr;
  }
  auto model = graph_cell_->GetGraph()->graph_data_->lite_model();
  if (model == nullptr) {
    MS_LOG(ERROR) << "Lite model is nullptr.";
    return kLiteNullptr;
  }
  if (model->buf == nullptr) {
    MS_LOG(ERROR) << "Lite model has been freed.";
    return kLiteError;
  }
  if (context_ == nullptr) {
    MS_LOG(ERROR) << "Invalid context.";
    return kLiteNullptr;
  }
  lite::Context model_context;
  model_context.allocator = Context::GetAllocator(context_);
  if (model_context.allocator == nullptr) {
    model_context.allocator = lite::Allocator::Create();
    if (model_context.allocator == nullptr) {
      MS_LOG(ERROR) << "Create Allocator failed.";
      return kLiteNullptr;
    }
    MS_LOG(DEBUG) << "Set new allocator.";
    Context::SetAllocator(context_, model_context.allocator);
  }
  model_context.vendor_name_ = Context::GetVendorName(context_);
  model_context.thread_num_ = Context::GetThreadNum(context_);
  model_context.device_list_.clear();
  if (Context::IfCPUEnabled(context_) && Context::IfGPUEnabled(context_) && Context::IfNPUEnabled(context_)) {
    MS_LOG(ERROR) << "CPU/GPU/NPU cannot be enabled at the same time.";
    return kLiteInputParamInvalid;
  }
  if (!Context::IfCPUEnabled(context_)) {
    MS_LOG(INFO) << "CPU is forced to be enabled.";
  }
  lite::DeviceInfo cpu_info = {
    .cpu_device_info_ = {Context::IfCPUFp16Enabled(context_), Context::GetCPUBindMode(context_)}};
  model_context.device_list_.push_back({lite::DT_CPU, cpu_info});
  if (Context::IfGPUEnabled(context_)) {
    lite::DeviceInfo gpu_info = {.gpu_device_info_ = {Context::IfGPUFp16Enabled(context_)}};
    model_context.device_list_.push_back({lite::DT_GPU, gpu_info});
  }
  if (Context::IfNPUEnabled(context_)) {
    lite::DeviceInfo npu_info = {.npu_device_info_ = {Context::GetNPUFrequency(context_)}};
    model_context.device_list_.push_back({lite::DT_NPU, npu_info});
  }
  auto session = std::shared_ptr<session::LiteSession>(session::LiteSession::CreateSession(&model_context));
  if (session == nullptr) {
    MS_LOG(ERROR) << "Allocate session failed.";
    return kLiteNullptr;
  }
  auto ret = session->CompileGraph(model.get());
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Build model failed.";
    return static_cast<StatusCode>(ret);
  }
  session_.swap(session);
  model->Free();
  MS_LOG(DEBUG) << "Build model success.";
  return kSuccess;
}

Status ModelImpl::Predict(const std::vector<MSTensor> &inputs, std::vector<MSTensor> *outputs) {
  if (session_ == nullptr) {
    MS_LOG(ERROR) << "Run graph failed.";
    return kLiteError;
  }
  auto input_tensors = session_->GetInputs();
  if (input_tensors.empty()) {
    MS_LOG(ERROR) << "Failed to get input tensor.";
    return kLiteError;
  }
  if (input_tensors.size() != inputs.size()) {
    MS_LOG(ERROR) << "Wrong input size.";
    return kLiteError;
  }
  std::vector<void *> old_data;
  for (size_t i = 0; i < inputs.size(); i++) {
    auto input = input_tensors.at(i);
    auto user_input = inputs.at(i);
    if (user_input.Name() != input->tensor_name()) {
      MS_LOG(WARNING) << "Tensor " << user_input.Name() << " has a different name from input" << input->tensor_name()
                      << ".";
    }
    old_data.push_back(input->MutableData());
    if (user_input.MutableData() != input->MutableData()) {
      if (input->Size() != user_input.DataSize()) {
        for (size_t j = 0; j < old_data.size(); j++) {
          input_tensors.at(j)->set_data(old_data.at(j));
        }
        MS_LOG(ERROR) << "Tensor " << user_input.Name() << " has wrong data size.";
        return kLiteInputTensorError;
      }
      if (user_input.impl_->need_copy()) {
        ::memcpy(input->MutableData(), user_input.MutableData(), input->Size());
      } else {
        input->set_data(user_input.MutableData());
      }
    }
  }
  auto ret = session_->RunGraph();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Run graph failed.";
    return static_cast<StatusCode>(ret);
  }
  MS_LOG(DEBUG) << "Run graph success.";
  for (size_t i = 0; i < old_data.size(); i++) {
    input_tensors.at(i)->set_data(old_data.at(i));
  }
  auto res = GetOutputs();
  if (res.empty()) {
    MS_LOG(DEBUG) << "Empty outputs.";
    return kLiteError;
  }
  outputs->clear();
  outputs->insert(outputs->end(), res.begin(), res.end());
  return kSuccess;
}

std::vector<MSTensor> ModelImpl::GetInputs() {
  std::vector<MSTensor> empty;
  if (session_ == nullptr) {
    MS_LOG(ERROR) << "Session is null.";
    return empty;
  }
  std::vector<MSTensor> res;
  auto inputs = session_->GetInputs();
  if (inputs.empty()) {
    MS_LOG(ERROR) << "The inputs of model is null.";
    return empty;
  }
  res.resize(inputs.size());
  for (size_t i = 0; i < inputs.size(); i++) {
    auto impl = std::shared_ptr<MSTensor::Impl>(new (std::nothrow) MSTensor::Impl(inputs[i]));
    if (impl == nullptr) {
      MS_LOG(ERROR) << "Create tensor failed.";
      return empty;
    }
    auto tensor = MSTensor(impl);
    if (tensor == nullptr) {
      MS_LOG(ERROR) << "Create tensor failed.";
      return empty;
    }
    res[i] = tensor;
  }
  return res;
}

std::vector<MSTensor> ModelImpl::GetOutputs() {
  std::vector<MSTensor> empty;
  if (session_ == nullptr) {
    MS_LOG(ERROR) << "Session is null.";
    return empty;
  }
  std::vector<MSTensor> res;
  auto names = session_->GetOutputTensorNames();
  if (names.empty()) {
    MS_LOG(ERROR) << "The names of model is null.";
    return empty;
  }
  auto outputs = session_->GetOutputs();
  if (outputs.empty()) {
    MS_LOG(ERROR) << "The outputs of model is null.";
    return empty;
  }
  if (names.size() != outputs.size()) {
    MS_LOG(ERROR) << "The size of outputs dose not match the size of names.";
    return empty;
  }
  res.resize(names.size());
  for (size_t i = 0; i < names.size(); i++) {
    auto impl = std::shared_ptr<MSTensor::Impl>(new (std::nothrow) MSTensor::Impl(outputs[names[i]]));
    if (impl == nullptr) {
      MS_LOG(ERROR) << "Create tensor failed.";
      return empty;
    }
    auto tensor = MSTensor(impl);
    if (tensor == nullptr) {
      MS_LOG(ERROR) << "Create tensor failed.";
      return empty;
    }
    res[i] = tensor;
  }
  return res;
}

Status ModelImpl::Resize(const std::vector<MSTensor> &inputs, const std::vector<std::vector<int64_t>> &dims) {
  if (session_ == nullptr) {
    MS_LOG(ERROR) << "Session is null.";
    return kLiteNullptr;
  }
  if (inputs.empty()) {
    MS_LOG(ERROR) << "Inputs is null.";
    return kLiteInputParamInvalid;
  }
  if (dims.empty()) {
    MS_LOG(ERROR) << "Dims is null.";
    return kLiteInputParamInvalid;
  }
  if (inputs.size() != dims.size()) {
    MS_LOG(ERROR) << "The size of inputs does not match the size of dims.";
    return kLiteInputParamInvalid;
  }
  auto model_inputs = session_->GetInputs();
  if (model_inputs.empty()) {
    MS_LOG(ERROR) << "The inputs of model is null.";
    return kLiteParamInvalid;
  }
  if (inputs.size() != model_inputs.size()) {
    MS_LOG(ERROR) << "The size of inputs is incorrect.";
    return kLiteInputParamInvalid;
  }
  std::vector<tensor::MSTensor *> inner_input;
  inner_input.resize(inputs.size());
  std::vector<std::vector<int32_t>> truncated_shape;
  truncated_shape.resize(inputs.size());
  for (size_t i = 0; i < inputs.size(); i++) {
    auto input = inputs[i];
    if (input.impl_ == nullptr || input.impl_->lite_tensor() == nullptr) {
      MS_LOG(ERROR) << "Input tensor " << input.Name() << " is null.";
      return kLiteInputTensorError;
    }
    inner_input[i] = input.impl_->lite_tensor();
    std::vector<int32_t> shape = TruncateShape(dims[i], inner_input[i]->data_type(), inner_input[i]->Size(), false);
    if (shape.empty() && !(dims[i].empty())) {
      MS_LOG(ERROR) << "Input dims[" << i << "] is invalid.";
      return kLiteParamInvalid;
    }
    truncated_shape[i] = shape;
  }
  auto ret = session_->Resize(inner_input, truncated_shape);
  return static_cast<StatusCode>(ret);
}

}  // namespace mindspore
