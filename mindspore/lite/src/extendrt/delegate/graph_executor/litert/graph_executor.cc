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

#include "extendrt/delegate/graph_executor/litert/graph_executor.h"
#include "tools/converter/converter_metagraph.h"
#include "src/litert/lite_model.h"
#include "src/litert/cpu_info.h"
#include "include/errorcode.h"
#include "flatbuffers/flatbuffers.h"
#include "extendrt/mock/lite_runtime/converters.h"
#include "extendrt/delegate/factory.h"

#include "tools/common/meta_graph_serializer.h"
#include "extendrt/utils/tensor_utils.h"
#include "backend/common/session/kernel_graph.h"
#include "src/common/helper/external_tensor/memory_helper.h"
#include "src/litert/kernel_exec.h"
#include "src/extendrt/delegate/graph_executor/litert/func_graph_reuse_manager.h"

namespace mindspore {
namespace {
// leave 200MB for the model struct to make sure the model will not large than 2GB
const size_t kOnlineExtractDataSize = 1800 * 1024 * 1024;
const int64_t kBufferSize = 1024;

Status LiteTensorToMSTensor(lite::Tensor *srcTensor, MSTensor *dstTensor, bool fromSession) {
  auto impl = std::make_shared<LiteTensorImpl>(srcTensor);
  if (impl == nullptr || impl->lite_tensor() == nullptr) {
    MS_LOG(ERROR) << "Create tensor failed.";
    return kLiteError;
  }
  impl->set_from_session(fromSession);
  auto tensor = MSTensor(impl);
  if (tensor == nullptr) {
    MS_LOG(ERROR) << "Create tensor failed.";
    return kLiteError;
  }
  *dstTensor = tensor;
  return kSuccess;
}

std::vector<MSTensor> LiteTensorsToMSTensors(const std::vector<mindspore::lite::Tensor *> &srcTensors,
                                             bool fromSession) {
  std::vector<MSTensor> dstTensors;
  dstTensors.reserve(srcTensors.size());
  for (auto inTensor : srcTensors) {
    MSTensor tensor;
    auto status = LiteTensorToMSTensor(inTensor, &tensor, fromSession);
    if (status != kSuccess) {
      return {};
    }
    dstTensors.emplace_back(tensor);
  }
  return dstTensors;
}
}  // namespace
const char litert_provider[] = "litert";

LiteRTGraphExecutor::LiteRTGraphExecutor(const std::shared_ptr<mindspore::Context> &context,
                                         const ConfigInfos &config_infos)
    : context_(context), config_infos_(config_infos) {
  lite_session_ = CreateLiteSession(ContextUtils::Convert(context_.get()), config_infos_);
}

bool LiteRTGraphExecutor::CompileGraph(const FuncGraphPtr &graph, const std::map<string, string> &compile_options) {
  MS_EXCEPTION_IF_NULL(graph);
  if (graph->isa<mindspore::session::KernelGraph>()) {
    MS_LOG(INFO) << "LiteRTGraphExecutor not support kernel garph, please pass func graph instead";
    return false;
  }

  if (!PlatformInstructionSetSupportCheck()) {
    MS_LOG(ERROR) << "The platform exist don't support's instruction.";
    return false;
  }
  size_t data_size;
  auto pair_result = FuncGraphReuseManager::GetInstance()->GetFbModelBuf(&data_size, &is_shared_fb_buf_, config_infos_);
  fb_model_buf_ = pair_result.first;
  helpers_ = pair_result.second;
  schema::MetaGraphT *meta_graph = nullptr;
  if (fb_model_buf_ == nullptr) {
    auto param = std::make_shared<ConverterPara>();
    param->fmk_type = converter::kFmkTypeMs;
    param->export_mindir = kMindIR;
    auto mutable_graph = std::const_pointer_cast<FuncGraph>(graph);
    meta_graph = lite::ConverterToMetaGraph::Build(param, mutable_graph);
    if (meta_graph == nullptr) {
      MS_LOG(ERROR) << "func graph convert to meta graph failed.";
      return false;
    }
    if (this->IsNeedExtractTensorData(meta_graph)) {
      if (!this->ExtractTensorData(meta_graph)) {
        MS_LOG(ERROR) << "Compile Large Graph failed, extract tensor data error.";
        return false;
      }
    }
    flatbuffers::FlatBufferBuilder builder(kBufferSize);
    auto buffer = lite::MetaGraphSerializer::GetMetaGraphPackedBuff(&builder, *meta_graph, &data_size);
    fb_model_buf_ = malloc(data_size);
    memcpy(fb_model_buf_, buffer, data_size);
    FuncGraphReuseManager::GetInstance()->StoreFbModelBuf(fb_model_buf_, data_size, helpers_, config_infos_);
  } else {
    MS_LOG(INFO) << "the graph is the same as the last time. We do not need to convert, and we can directly use the "
                    "cached model buf.";
  }
  if (lite_session_ == nullptr) {
    MS_LOG(ERROR) << "lite session is nullptr.";
    return false;
  }
  int ret = lite_session_->LoadModelAndCompileByBuf(reinterpret_cast<char *>(fb_model_buf_), kMindIR_Lite, data_size,
                                                    helpers_.get());
  delete meta_graph;
  meta_graph = nullptr;
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
      input->set_data(user_input.data_c(), false);
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
        input->set_data(user_input.data_c(), false);
      }
    }
  }
  lite::KernelCallBack before_call_back = nullptr;
  lite::KernelCallBack after_call_back = nullptr;
  if (before_ != nullptr) {
    before_call_back = [&](const std::vector<mindspore::lite::Tensor *> &before_inputs,
                           const std::vector<mindspore::lite::Tensor *> &before_outputs,
                           const MSCallBackParam &call_param) {
      std::vector<MSTensor> inputs = LiteTensorsToMSTensors(before_inputs, true);
      std::vector<MSTensor> outputs = LiteTensorsToMSTensors(before_outputs, true);
      return before_(inputs, outputs, call_param);
    };
  }

  if (after_ != nullptr) {
    after_call_back = [&](const std::vector<mindspore::lite::Tensor *> &before_inputs,
                          const std::vector<mindspore::lite::Tensor *> &before_outputs,
                          const MSCallBackParam &call_param) {
      std::vector<MSTensor> inputs = LiteTensorsToMSTensors(before_inputs, true);
      std::vector<MSTensor> outputs = LiteTensorsToMSTensors(before_outputs, true);
      return after_(inputs, outputs, call_param);
    };
  }
  auto ret = lite_session_->RunGraph(before_call_back, after_call_back);
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
  *outputs = TensorUtils::MSTensorToTensor(res);
  return true;
}

bool LiteRTGraphExecutor::Resize(const FuncGraphPtr &, const std::vector<tensor::Tensor> &inputs,
                                 const std::vector<std::vector<int64_t>> &dims) {
  auto input_tensors = lite_session_->GetInputs();
  if (input_tensors.empty()) {
    MS_LOG(EXCEPTION) << "Failed to get input tensor.";
  }
  if (input_tensors.size() != inputs.size()) {
    MS_LOG(EXCEPTION) << "Wrong input size.";
  }
  std::vector<std::vector<int>> user_shapes;
  std::transform(dims.begin(), dims.end(), std::back_inserter(user_shapes), [](auto &input) {
    std::vector<int> shape;
    std::transform(input.begin(), input.end(), std::back_inserter(shape), [](auto s) { return static_cast<int>(s); });
    return shape;
  });
  auto ret = lite_session_->Resize(input_tensors, user_shapes);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "lite session resize failed";
    return false;
  }
  return true;
}

std::vector<tensor::Tensor> LiteRTGraphExecutor::GetInputInfos(const FuncGraphPtr &) {
  if (lite_session_ == nullptr) {
    MS_LOG(ERROR) << "Session is null.";
    return {};
  }
  auto inputs = lite_session_->GetInputs();
  std::vector<tensor::Tensor> input_tensors;
  for (size_t i = 0; i < inputs.size(); ++i) {
    auto type_id = inputs[i]->data_type();
    auto shape = inputs[i]->shape();
    std::vector<int64_t> lite_shape;
    std::transform(shape.begin(), shape.end(), std::back_inserter(lite_shape),
                   [](int c) { return static_cast<int64_t>(c); });
    input_tensors.push_back(tensor::Tensor(type_id, lite_shape));
  }
  return input_tensors;
}

std::vector<tensor::Tensor> LiteRTGraphExecutor::GetOutputInfos(const FuncGraphPtr &) {
  auto outputs = GetLiteSessionOutputs();
  std::vector<tensor::Tensor> output_tensors;
  for (size_t i = 0; i < outputs.size(); ++i) {
    auto type_id = static_cast<enum TypeId>(outputs[i].DataType());
    output_tensors.push_back(tensor::Tensor(type_id, outputs[i].Shape()));
  }
  return output_tensors;
}

void LiteRTGraphExecutor::ResetTensorData(std::vector<void *> old_data, const std::vector<lite::Tensor *> &tensors) {
  for (size_t j = 0; j < old_data.size(); j++) {
    tensors.at(j)->set_data(old_data.at(j));
  }
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

std::shared_ptr<lite::LiteSession> LiteRTGraphExecutor::CreateLiteSession(
  const std::shared_ptr<lite::InnerContext> &context, const ConfigInfos &config_infos) {
  auto session = std::make_shared<lite::LiteSession>();
  if (session == nullptr) {
    MS_LOG(ERROR) << "create session failed";
    return nullptr;
  }
  session->SetConfigInfo(&config_infos);

  session->SetKeepModelBuf(true);
  auto ret = session->Init(context);
  if (ret != mindspore::lite::RET_OK) {
    MS_LOG(ERROR) << "init session failed";
    return nullptr;
  }
  return session;
}

bool LiteRTGraphExecutor::ExtractTensorData(mindspore::schema::MetaGraphT *meta_graph_t) {
  MS_EXCEPTION_IF_NULL(meta_graph_t);
  helpers_ = std::make_shared<mindspore::infer::helper::InferHelpers>();
  if (helpers_ == nullptr) {
    MS_LOG(ERROR) << "Create InferHelpers failed.";
    return false;
  }
  auto tensor_helper = new (std::nothrow) mindspore::infer::helper::MemoryExternalTensorHelper();
  if (tensor_helper == nullptr) {
    MS_LOG(ERROR) << "Create Memory External TensorHelper failed.";
    return false;
  }
  int64_t cur_offset = 0;
  size_t size = 0;
  uint8_t *data = nullptr;
  for (const auto &tensor : meta_graph_t->allTensors) {
    if (tensor->nodeType == mindspore::lite::NodeType_CNode) {
      continue;
    }
    if (tensor->dataType == kObjectTypeTensorType) {  // not support control-flow now
      continue;
    }
    auto *external_data_t = new (std::nothrow) schema::ExternalDataT;
    if (external_data_t == nullptr) {
      MS_LOG(ERROR) << "Create ExternalDataT failed";
      return false;
    }
    data = tensor->data.data();
    size = tensor->data.size();
    external_data_t->location = "MEM: " + tensor->name;
    external_data_t->offset = cur_offset;
    external_data_t->length = static_cast<int64_t>(size);
    if (data != nullptr && size > 0) {
      std::stringstream oss;
      oss << std::hash<char>()(data[0]);
      external_data_t->checkSum = oss.str();
      cur_offset += static_cast<int64_t>(size);
      flatbuffers::FlatBufferBuilder builder(kBufferSize);
      auto offset = mindspore::schema::ExternalData::Pack(builder, external_data_t);
      builder.Finish(offset);
      auto external_data = flatbuffers::GetRoot<mindspore::schema::ExternalData>(builder.GetBufferPointer());
      tensor_helper->SetExternalTensorData(external_data, static_cast<void *>(data));
    }
    tensor->data.clear();
    tensor->externalData.emplace_back(external_data_t);
  }
  helpers_->SetExternalTensorHelper(tensor_helper);
  return true;
}

bool LiteRTGraphExecutor::IsNeedExtractTensorData(mindspore::schema::MetaGraphT *meta_graph_t) {
  MS_EXCEPTION_IF_NULL(meta_graph_t);
  size_t size = 0;
  for (auto &tensor : meta_graph_t->allTensors) {
    size += tensor->data.size();
  }
  if (size >= kOnlineExtractDataSize) {
    return true;
  }
  return false;
}

static std::shared_ptr<device::GraphExecutor> LiteRTGraphExecutorCreator(const std::shared_ptr<Context> &ctx,
                                                                         const ConfigInfos &config_infos) {
  return std::make_shared<LiteRTGraphExecutor>(ctx, config_infos);
}

REG_DELEGATE(kCPU, litert_provider, LiteRTGraphExecutorCreator);
}  // namespace mindspore
