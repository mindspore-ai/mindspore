/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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

#include <algorithm>
#include <functional>
#include <string>
#include <vector>

#include "src/extendrt/session/lite_infer_session.h"

#include "extendrt/mock/lite_runtime/converters.h"
#include "extendrt/session/factory.h"
#include "extendrt/utils/runtime_utils.h"
#include "extendrt/utils/tensor_utils.h"

namespace mindspore {
const size_t tensor_max_size = 0x1000000;

Status LiteInferSession::Init(const std::shared_ptr<Context> &context) {
  MS_LOG(INFO) << "LiteInferSession::Init";
  context_ = context;
  lite_session_ = CreateLiteSession(ContextUtils::Convert(context_.get()));
  MS_EXCEPTION_IF_NULL(lite_session_);
  return kSuccess;
}

Status LiteInferSession::CompileGraph(FuncGraphPtr graph, const void *data, size_t size) {
  // Lite infer session do not use graph, just use data and size
  MS_LOG(INFO) << "LiteInferSession::CompileGraph";
  MS_EXCEPTION_IF_NULL(data);
  MS_EXCEPTION_IF_ZERO("size", size);
  lite_session_ = CreateLiteSession(ContextUtils::Convert(context_.get()));
  MS_EXCEPTION_IF_NULL(lite_session_);

  auto ret = lite_session_->LoadModelAndCompileByBuf(static_cast<const char *>(data), kMindIR, size);
  if (ret != RET_OK) {
    MS_LOG(EXCEPTION) << "load model and compile failed";
  }

  return kSuccess;
}

void LiteInferSession::ResetTensorData(std::vector<void *> old_data, const std::vector<lite::Tensor *> &tensors) {
  for (size_t j = 0; j < old_data.size(); j++) {
    tensors.at(j)->set_data(old_data.at(j));
  }
}

std::vector<MSTensor> LiteInferSession::GetLiteSessionOutputs() {
  std::vector<MSTensor> empty;
  if (lite_session_ == nullptr) {
    MS_LOG(ERROR) << "Session is null.";
    return empty;
  }
  std::vector<MSTensor> res;
  auto names = lite_session_->GetOutputTensorNames();
  if (names.empty()) {
    MS_LOG(ERROR) << "The output tensor name of this model is null.";
    return empty;
  }
  auto outputs = lite_session_->GetOutputs();
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
    auto impl = std::make_shared<LiteTensorImpl>(outputs[names[i]]);
    if (impl == nullptr || impl->lite_tensor() == nullptr) {
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

std::vector<int32_t> LiteInferSession::TruncateShape(const std::vector<int64_t> &shape, TypeId type, size_t data_len,
                                                     bool verify_size) {
  std::vector<int32_t> empty;
  if (shape.empty()) {
    return empty;
  }
  std::vector<int32_t> truncated_shape;
  truncated_shape.resize(shape.size());
  size_t element_size = lite::DataTypeSize(type);
  for (size_t i = 0; i < shape.size(); i++) {
    auto dim = shape[i];
    if (dim < 0 || dim > INT_MAX || (dim != 0 && element_size > INT_MAX / static_cast<size_t>(dim))) {
      MS_LOG(ERROR) << "Invalid shape!dim: " << dim << ", element_size: " << element_size;
      return empty;
    } else {
      element_size *= static_cast<size_t>(dim);
      truncated_shape[i] = static_cast<int32_t>(dim);
    }
  }
  if (verify_size) {
    if (element_size != data_len) {
      MS_LOG(ERROR) << "Invalid data size!element_size: " << element_size << ", data_len: " << data_len;
      return empty;
    }
  }
  return truncated_shape;
}

Status LiteInferSession::RunGraph(const std::vector<tensor::Tensor> &inputs, std::vector<tensor::Tensor> *outputs) {
  MS_LOG(INFO) << "SingleOpInferSession::RunGraph with input and outputs";
  MS_EXCEPTION_IF_NULL(outputs);
  MS_EXCEPTION_IF_NULL(lite_session_);

  auto input_tensors = lite_session_->GetInputs();
  if (input_tensors.empty()) {
    MS_LOG(EXCEPTION) << "Failed to get input tensor.";
  }
  if (input_tensors.size() != inputs.size()) {
    MS_LOG(EXCEPTION) << "Wrong input size.";
  }
  std::vector<void *> old_data;
  for (size_t i = 0; i < inputs.size(); i++) {
    auto input = input_tensors.at(i);
    auto user_input = &inputs[i];
    if (user_input->data_type() != input->data_type()) {
      ResetTensorData(old_data, input_tensors);
      MS_LOG(EXCEPTION) << "Tensor " << user_input->id() << " has a different data type from input"
                        << input->tensor_name() << ".";
    }
    if (user_input->data_c() == nullptr) {
      ResetTensorData(old_data, input_tensors);
      MS_LOG(EXCEPTION) << "Tensor " << user_input->id() << " has no data.";
    }
    old_data.push_back(input->data());
    if (input->data_type() == kObjectTypeString) {
#ifndef STRING_KERNEL_CLIP
      std::vector<int32_t> shape =
        TruncateShape(user_input->shape_c(), input->data_type(), user_input->DataSize(), false);
      if (shape.empty() && !(user_input->shape_c().empty())) {
        ResetTensorData(old_data, input_tensors);
        MS_LOG(EXCEPTION) << "Input dims of tensor " << user_input->id() << " is invalid.";
      }
      input->set_shape(shape);
      input->set_data(user_input->data_c());
#else
      MS_LOG(ERROR) << unsupport_string_tensor_log;
      return kLiteError;
#endif
    } else {
      if (user_input->data_c() != input->data()) {
        if (input->Size() != user_input->Size()) {
          ResetTensorData(old_data, input_tensors);
#ifndef ENABLE_LITE_ACL
          MS_LOG(EXCEPTION) << "Tensor " << user_input->id() << " has wrong data size.";
#else
          MS_LOG(WARNING) << "Please check tensor " << user_input->id()
                          << " has been modified data size by DVPP method.";
          std::vector<int> truncate_shape = {static_cast<int>(user_input->DataSize())};
          input->set_shape(truncate_shape);
#endif
        }
        input->set_data(user_input->data_c());
      }
    }
  }
  auto ret = static_cast<StatusCode>(lite_session_->RunGraph());
  ResetTensorData(old_data, input_tensors);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "Run graph failed.";
    return ret;
  }
  MS_LOG(DEBUG) << "Run graph success.";
  auto res = GetLiteSessionOutputs();
  if (res.empty()) {
    MS_LOG(DEBUG) << "Empty outputs.";
    return kLiteError;
  }
  outputs->clear();
  *outputs = TensorUtils::MSTensorToTensor(res);
  return kSuccess;
}

std::vector<MutableTensorImplPtr> LiteInferSession::GetOutputs() {
  auto outputs = lite_session_->GetOutputs();
  std::vector<MutableTensorImplPtr> output_tensors;
  for (auto &iter : outputs) {
    auto output = iter.second;
    auto impl = std::make_shared<LiteTensorImpl>(output);
    output_tensors.emplace_back(impl);
  }
  return output_tensors;
}

std::vector<MutableTensorImplPtr> LiteInferSession::GetInputs() {
  auto inputs = lite_session_->GetInputs();
  std::vector<MutableTensorImplPtr> input_tensors;
  for (auto &input : inputs) {
    auto impl = std::make_shared<LiteTensorImpl>(input);
    input_tensors.emplace_back(impl);
  }
  return input_tensors;
}

std::vector<std::string> LiteInferSession::GetOutputNames() {
  auto outputs = lite_session_->GetOutputs();
  std::vector<std::string> output_names;
  std::transform(outputs.begin(), outputs.end(), std::back_inserter(output_names),
                 [](auto iter) { return iter.first; });
  return output_names;
}

std::vector<std::string> LiteInferSession::GetInputNames() { return ConvertToTensorNames(lite_session_->GetInputs()); }
MutableTensorImplPtr LiteInferSession::GetOutputByTensorName(const std::string &name) {
  auto outputs = lite_session_->GetOutputs();
  for (auto &iter : outputs) {
    auto output = iter.second;
    if (output->tensor_name() == name) {
      return std::make_shared<LiteTensorImpl>(output);
    }
  }
  return nullptr;
}

MutableTensorImplPtr LiteInferSession::GetInputByTensorName(const std::string &name) {
  auto inputs = lite_session_->GetInputs();
  for (auto &input : inputs) {
    if (input->tensor_name() == name) {
      return std::make_shared<LiteTensorImpl>(input);
    }
  }
  return nullptr;
}

std::shared_ptr<lite::LiteSession> LiteInferSession::CreateLiteSession(
  const std::shared_ptr<lite::InnerContext> &context) {
  auto session = std::make_shared<lite::LiteSession>();
  if (session == nullptr) {
    MS_LOG(ERROR) << "create session failed";
    return nullptr;
  }

  auto ret = session->Init(context);
  if (ret != mindspore::lite::RET_OK) {
    MS_LOG(ERROR) << "init session failed";
    return nullptr;
  }
  return session;
}

std::vector<std::string> LiteInferSession::ConvertToTensorNames(
  const std::vector<mindspore::lite::Tensor *> &lite_tensors) {
  std::vector<std::string> tensor_names;
  std::transform(lite_tensors.begin(), lite_tensors.end(), std::back_inserter(tensor_names),
                 [](mindspore::lite::Tensor *lite_tensor) {
                   MS_EXCEPTION_IF_NULL(lite_tensor);
                   return lite_tensor->tensor_name();
                 });
  return tensor_names;
}

std::vector<tensor::TensorPtr> LiteInferSession::ConvertToTensors(
  const std::vector<mindspore::lite::Tensor *> &lite_tensors) {
  std::vector<tensor::TensorPtr> tensors;
  for (auto lite_tensor : lite_tensors) {
    auto type_id = lite_tensor->data_type();
    auto shape = lite_tensor->shape();
    ShapeVector shape_vec;
    std::transform(shape.begin(), shape.end(), std::back_inserter(shape_vec),
                   [](int s) { return static_cast<int64_t>(s); });
    auto data = lite_tensor->data();
    auto data_size = lite_tensor->Size();
    auto tensor_ptr = std::make_shared<mindspore::tensor::Tensor>(type_id, shape_vec, data, data_size);
    tensors.emplace_back(tensor_ptr);
  }
  return tensors;
}

static std::shared_ptr<InferSession> LiteInferSessionCreator(const std::shared_ptr<Context> &ctx,
                                                             const ConfigInfos &config_infos) {
  auto session = std::make_shared<LiteInferSession>();
  session->Init(ctx);
  return session;
}
REG_SESSION(kLiteInferSession, LiteInferSessionCreator);
}  // namespace mindspore
