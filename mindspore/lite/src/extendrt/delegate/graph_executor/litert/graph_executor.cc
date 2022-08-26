/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#include <vector>
#include <string>
#include <memory>
#include <map>
#include <algorithm>

#include "tools/converter/converter.h"
#include "src/litert/lite_model.h"
#include "schema/inner/model_generated.h"
#include "include/errorcode.h"
#include "flatbuffers/flatbuffers.h"

#include "extendrt/delegate/graph_executor/litert/converters.h"
#include "extendrt/delegate/graph_executor/litert/graph_executor.h"
#include "extendrt/delegate/graph_executor/factory.h"

#include "tools/common/meta_graph_serializer.h"
#include "extendrt/utils/tensor_utils.h"
#include "backend/common/session/kernel_graph.h"

namespace mindspore {
namespace {
const int64_t kBufferSize = 1024;
}
const char litert_provider[] = "litert";

LiteRTGraphExecutor::LiteRTGraphExecutor(const std::shared_ptr<mindspore::Context> &context) : context_(context) {
  lite_session_ = CreateLiteSession(ContextUtils::Convert(context_.get()));
}

bool LiteRTGraphExecutor::CompileGraph(const FuncGraphPtr &graph, const std::map<string, string> &compile_options) {
  MS_EXCEPTION_IF_NULL(graph);
  if (graph->isa<mindspore::session::KernelGraph>()) {
    MS_LOG(WARNING) << "LiteRTGraphExecutor not support kernel garph, please pass func graph instead";
    return false;
  }
  auto converter = std::make_shared<mindspore::lite::ConverterImpl>();
  auto param = std::make_shared<ConverterPara>();
  param->fmk_type = converter::kFmkTypeMs;
  auto mutable_graph = std::const_pointer_cast<FuncGraph>(graph);
  schema::MetaGraphT *meta_graph_t = nullptr;
  converter->Convert(param, &meta_graph_t, mutable_graph);
  flatbuffers::FlatBufferBuilder builder(kBufferSize);
  size_t data_size;
  auto buffer = lite::MetaGraphSerializer::GetMetaGraphPackedBuff(&builder, *meta_graph_t, &data_size);
  auto buf = malloc(data_size);
  memcpy(buf, buffer, data_size);
  int ret = lite_session_->LoadModelAndCompileByBuf(reinterpret_cast<char *>(buf), kMindIR_Lite, data_size);
  if (ret != lite::RET_OK) {
    MS_LOG(ERROR) << "Load model by meta graph failed";
    return false;
  }
  return true;
}

bool LiteRTGraphExecutor::RunGraph(const FuncGraphPtr &graph, const std::vector<tensor::Tensor> &inputs,
                                   std::vector<tensor::Tensor> *outputs,
                                   const std::map<string, string> &compile_options) {
  MS_LOG(INFO) << "LiteRTGraphExecutor::RunGraph with input and outputs";
  MS_EXCEPTION_IF_NULL(outputs);
  MS_EXCEPTION_IF_NULL(lite_session_);

  auto input_tensors = lite_session_->GetInputs();
  if (input_tensors.empty()) {
    MS_LOG(EXCEPTION) << "Failed to get input tensor.";
  }
  if (input_tensors.size() != inputs.size()) {
    MS_LOG(EXCEPTION) << "Wrong input size.";
  }

  std::vector<std::vector<int>> user_shapes;
  std::transform(inputs.begin(), inputs.end(), std::back_inserter(user_shapes), [](auto &input) {
    auto user_shape = input.shape_c();
    std::vector<int> shape;
    std::transform(user_shape.begin(), user_shape.end(), std::back_inserter(shape),
                   [](auto s) { return static_cast<int>(s); });
    return shape;
  });
  auto ret = lite_session_->Resize(input_tensors, user_shapes);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "Run graph failed. resize input error with " << ret;
    return false;
  }
  std::vector<void *> old_data;
  for (size_t i = 0; i < inputs.size(); i++) {
    auto input = input_tensors.at(i);
    auto &user_input = inputs.at(i);
    if (user_input.data_type() != input->data_type()) {
      ResetTensorData(old_data, input_tensors);
      MS_LOG(EXCEPTION) << "Tensor " << user_input.id() << " has a different data type from input"
                        << input->tensor_name() << ".";
    }
    if (user_input.data_c() == nullptr) {
      ResetTensorData(old_data, input_tensors);
      MS_LOG(EXCEPTION) << "Tensor " << user_input.id() << " has no data.";
    }
    old_data.push_back(input->data());
    if (input->data_type() == kObjectTypeString) {
#ifndef STRING_KERNEL_CLIP
      std::vector<int32_t> shape =
        TruncateShape(user_input.shape_c(), input->data_type(), user_input.DataSize(), false);
      if (shape.empty() && !(user_input.shape_c().empty())) {
        ResetTensorData(old_data, input_tensors);
        MS_LOG(EXCEPTION) << "Input dims of tensor " << user_input.id() << " is invalid.";
      }
      input->set_shape(shape);
      input->set_data(user_input.data_c());
#else
      MS_LOG(ERROR) << unsupport_string_tensor_log;
      return kLiteError;
#endif
    } else {
      if (user_input.data_c() != input->data()) {
        if (input->Size() != user_input.Size()) {
          ResetTensorData(old_data, input_tensors);
#ifndef ENABLE_LITE_ACL
          MS_LOG(EXCEPTION) << "Tensor " << user_input.id() << " has wrong data size.";
#else
          MS_LOG(WARNING) << "Please check tensor " << user_input.id()
                          << " has been modified data size by DVPP method.";
          std::vector<int> truncate_shape = {static_cast<int>(user_input.DataSize())};
          input->set_shape(truncate_shape);
#endif
        }
        input->set_data(user_input.data_c());
      }
    }
  }
  ret = lite_session_->RunGraph();
  ResetTensorData(old_data, input_tensors);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "Run graph failed.";
    return false;
  }
  MS_LOG(DEBUG) << "Run graph success.";
  auto res = GetLiteSessionOutputs();
  if (res.empty()) {
    MS_LOG(DEBUG) << "Empty outputs.";
    return false;
  }
  outputs->clear();
  *outputs = TensorUtils::TensorPtrToTensor(TensorUtils::MSTensorToTensorPtr(res));
  return true;
}

bool LiteRTGraphExecutor::Resize(const std::vector<tensor::Tensor> &inputs,
                                 const std::vector<std::vector<int64_t>> &dims) {
  auto input_tensors = lite_session_->GetInputs();
  if (input_tensors.empty()) {
    MS_LOG(EXCEPTION) << "Failed to get input tensor.";
  }
  if (input_tensors.size() != inputs.size()) {
    MS_LOG(EXCEPTION) << "Wrong input size.";
  }
  std::vector<std::vector<int>> user_shapes;
  std::transform(inputs.begin(), inputs.end(), std::back_inserter(user_shapes), [](auto &input) {
    auto user_shape = input.shape_c();
    std::vector<int> shape;
    std::transform(user_shape.begin(), user_shape.end(), std::back_inserter(shape),
                   [](auto s) { return static_cast<int>(s); });
    return shape;
  });
  auto ret = lite_session_->Resize(input_tensors, user_shapes);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "lite session resize failed";
    return false;
  }
  return true;
}

std::vector<tensor::TensorPtr> LiteRTGraphExecutor::GetInputs() {
  auto input_tensors = GetLiteSessionInputs();
  return TensorUtils::MSTensorToTensorPtr(input_tensors);
}

void LiteRTGraphExecutor::ResetTensorData(std::vector<void *> old_data, const std::vector<lite::Tensor *> &tensors) {
  for (size_t j = 0; j < old_data.size(); j++) {
    tensors.at(j)->set_data(old_data.at(j));
  }
}

std::vector<MSTensor> LiteRTGraphExecutor::GetLiteSessionInputs() {
  std::vector<MSTensor> empty;
  if (lite_session_ == nullptr) {
    MS_LOG(ERROR) << "Session is null.";
    return empty;
  }
  auto inputs = lite_session_->GetInputs();
  if (inputs.empty()) {
    MS_LOG(ERROR) << "The input of model is empty";
    return empty;
  }
  std::vector<MSTensor> input_tensors;
  input_tensors.resize(inputs.size());
  for (size_t i = 0; i < input_tensors.size(); ++i) {
    auto impl = std::make_shared<LiteTensorImpl>(inputs[i]);
    if (impl == nullptr || impl->lite_tensor() == nullptr) {
      MS_LOG(ERROR) << "Create tensor failed";
      return empty;
    }
    auto tensor = MSTensor(impl);
    if (tensor == nullptr) {
      MS_LOG(ERROR) << "Create tensor failed";
      return empty;
    }
    input_tensors[i] = tensor;
  }
  return input_tensors;
}

std::vector<MSTensor> LiteRTGraphExecutor::GetLiteSessionOutputs() {
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

std::vector<int32_t> LiteRTGraphExecutor::TruncateShape(const std::vector<int64_t> &shape, TypeId type, size_t data_len,
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

std::shared_ptr<lite::LiteSession> LiteRTGraphExecutor::CreateLiteSession(lite::InnerContext *context) {
  auto session = std::make_shared<lite::LiteSession>();
  if (session == nullptr) {
    MS_LOG(ERROR) << "create session failed";
    delete context;
    return nullptr;
  }

  auto ret = session->Init(context);
  if (ret != mindspore::lite::RET_OK) {
    MS_LOG(ERROR) << "init session failed";
    return nullptr;
  }
  return session;
}

static std::shared_ptr<device::GraphExecutor> LiteRTGraphExecutorCreator(
  const std::shared_ptr<mindspore::DelegateConfig> &config) {
  MS_EXCEPTION_IF_NULL(config);
  return std::make_shared<LiteRTGraphExecutor>(config->GetContext());
}

REG_GRAPH_EXECUTOR(kCPU, litert_provider, LiteRTGraphExecutorCreator);
}  // namespace mindspore
