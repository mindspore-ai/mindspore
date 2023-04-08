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

#include "extendrt/session/delegate_session.h"
#include <vector>
#include <string>
#include <mutex>
#include <memory>
#include "extendrt/utils/tensor_utils.h"
#include "src/extendrt/utils/kernel_build_utils.h"
#include "extendrt/delegate/factory.h"
#include "extendrt/session/factory.h"
#include "extendrt/utils/tensor_default_impl.h"
#include "src/extendrt/delegate/graph_executor/litert/func_graph_reuse_manager.h"
#include "src/extendrt/delegate/plugin/ascend_ge_executor_plugin.h"
#include "extendrt/utils/func_graph_utils.h"

namespace mindspore {
namespace {
constexpr auto kAscendProviderGe = "ge";
std::mutex kernel_graph_mutex;
std::mutex g_build_graph_mutex;
}  // namespace
GraphSinkSession::~GraphSinkSession() {
  graph_executor_ = nullptr;
  if (is_use_ascend_ge_) {
    lite::AscendGeExecutorPlugin::GetInstance().DestroyGeContext();
  }
}

Status GraphSinkSession::GeDeviceContextInit() {
  return lite::AscendGeExecutorPlugin::GetInstance().InitializeGeContext();
}

Status GraphSinkSession::Init(const std::shared_ptr<Context> &context) {
  MS_LOG(INFO) << "GraphSinkSession::Init";
  if (graph_executor_ == nullptr) {
    MS_LOG(ERROR) << "GraphSinkSession::Init failed, graph executor is nullptr.";
    return kLiteUninitializedObj;
  }
  auto device_list = context->MutableDeviceInfo();
  for (const auto &device_info : device_list) {
    if (device_info == nullptr) {
      MS_LOG(ERROR) << "GraphSinkSession::Init failed, device info is nullptr.";
      return kLiteUninitializedObj;
    }
    if (device_info->GetDeviceType() == DeviceType::kAscend && device_info->GetProvider() == kAscendProviderGe) {
      MS_LOG(INFO) << "GraphSinkSession::Init ascend helper";
      is_use_ascend_ge_ = true;
      GeDeviceContextInit();
      break;
    }
  }
  context_ = context;
  return kSuccess;
}

Status GraphSinkSession::CompileGraph(FuncGraphPtr graph, const void *data, size_t size, uint32_t *graph_id) {
  MS_LOG(INFO) << "GraphSinkSession::CompileGraph";
  // This lock can be removed when LiteRT supports concurrent multithreading compilation.
  std::lock_guard<std::mutex> lock(g_build_graph_mutex);
  // kernel graph will be removed from GraphSinkSession, and this code will be moved to TensorRT plugin
  if (context_ && !context_->MutableDeviceInfo().empty()) {
    auto device_info = context_->MutableDeviceInfo()[0];
    if (device_info && device_info->GetDeviceType() == DeviceType::kAscend &&
        device_info->GetProvider() == kAscendProviderGe) {
      lite::AscendGeExecutorPlugin::GetInstance().AdaptGraph(graph);
    }
  }
  DelegateGraphInfo graph_info;
  auto status = InitGraphInputsOutputs(graph, &graph_info);
  if (!status.IsOk()) {
    MS_LOG(ERROR) << "Failed to get inputs and outputs info from graph";
    return status;
  }
  auto ret = graph_executor_->CompileGraph(graph, options_, graph_id);
  if (!ret) {
    MS_LOG(ERROR) << "GraphSinkSession::CompileGraph compile graph failed";
    return kCoreFailed;
  }
  status = UpdateGraphInputsOutputs(*graph_id, &graph_info);
  if (!status.IsOk()) {
    MS_LOG(ERROR) << "Failed to update inputs and outputs info from graph executor";
    return status;
  }
  graph_infos_[*graph_id] = graph_info;
  return kSuccess;
}

Status GraphSinkSession::InitGraphInputsOutputs(const FuncGraphPtr &graph, DelegateGraphInfo *graph_info_ptr) {
  auto &info = *graph_info_ptr;
  std::vector<tensor::TensorPtr> graph_inputs, graph_outputs;
  {
    std::unique_lock<std::mutex> l(kernel_graph_mutex);
    FuncGraphReuseManager::GetInstance()->GetInOut(config_infos_, &graph_inputs, &graph_outputs, &info.input_names,
                                                   &info.output_names);
    if (graph_inputs.empty() || graph_outputs.empty() || info.input_names.empty() || info.output_names.empty()) {
      FuncGraphUtils::GetFuncGraphInputsInfo(graph, &graph_inputs, &info.input_names);
      FuncGraphUtils::GetFuncGraphOutputsInfo(graph, &graph_outputs, &info.output_names);
      FuncGraphReuseManager::GetInstance()->StoreInOut(config_infos_, graph_inputs, graph_outputs, info.input_names,
                                                       info.output_names);
    } else {
      MS_LOG(INFO) << "the input and output are the same as the last time. We do not need to construct, and we can "
                      "directly use the cached input and output info.";
    }
  }
  if (graph_inputs.size() != info.input_names.size()) {
    MS_LOG(ERROR) << "Graph input size " << graph_inputs.size() << " != input names size " << info.input_names.size();
    return kCoreFailed;
  }
  if (graph_outputs.size() != info.output_names.size()) {
    MS_LOG(ERROR) << "Graph output size " << graph_outputs.size() << " != output names size "
                  << info.output_names.size();
    return kCoreFailed;
  }
  info.inputs.clear();
  for (size_t i = 0; i < info.input_names.size(); i++) {
    auto &input = graph_inputs[i];
    auto data_type = static_cast<enum DataType>(input->data_type());
    auto impl = std::make_shared<TensorDefaultImpl>(info.input_names[i], data_type, input->shape_c());
    info.inputs.push_back(impl);
  }
  info.outputs.clear();
  for (size_t i = 0; i < info.output_names.size(); i++) {
    auto &output = graph_outputs[i];
    auto data_type = static_cast<enum DataType>(output->data_type());
    auto impl = std::make_shared<TensorDefaultImpl>(info.output_names[i], data_type, output->shape_c());
    info.outputs.push_back(impl);
  }
  return kSuccess;
}

Status GraphSinkSession::UpdateGraphInputsOutputs(uint32_t graph_id, DelegateGraphInfo *graph_info_ptr) {
  auto &info = *graph_info_ptr;
  auto new_inputs = graph_executor_->GetInputInfos(graph_id);
  if (!new_inputs.empty()) {
    if (new_inputs.size() != info.input_names.size()) {
      MS_LOG(ERROR) << "Input count " << new_inputs.size() << " get from executor != input names count "
                    << info.input_names.size();
      return kCoreFailed;
    }
    info.inputs.clear();
    for (size_t i = 0; i < info.input_names.size(); i++) {
      auto &input = new_inputs[i];
      auto data_type = static_cast<enum DataType>(input.data_type());
      auto impl = std::make_shared<TensorDefaultImpl>(info.input_names[i], data_type, input.shape_c());
      info.inputs.push_back(impl);
    }
  }
  auto new_outputs = graph_executor_->GetOutputInfos(graph_id);
  if (!new_outputs.empty()) {
    if (new_outputs.size() != info.output_names.size()) {
      MS_LOG(ERROR) << "Output count " << new_outputs.size() << " get from executor != output names count "
                    << info.output_names.size();
      return kCoreFailed;
    }
    info.outputs.clear();
    for (size_t i = 0; i < info.output_names.size(); i++) {
      auto &output = new_outputs[i];
      auto data_type = static_cast<enum DataType>(output.data_type());
      auto impl = std::make_shared<TensorDefaultImpl>(info.output_names[i], data_type, output.shape_c());
      info.outputs.push_back(impl);
    }
  }
  return kSuccess;
}

Status GraphSinkSession::RunGraph(uint32_t graph_id, const std::vector<tensor::Tensor> &inputs,
                                  std::vector<tensor::Tensor> *outputs, const MSKernelCallBack &before,
                                  const MSKernelCallBack &after) {
  MS_LOG(INFO) << "GraphSinkSession::RunGraph";
  MS_EXCEPTION_IF_NULL(outputs);
  graph_executor_->SetBefore(before);
  graph_executor_->SetAfter(after);
  bool ret = graph_executor_->RunGraph(graph_id, inputs, outputs, options_);
  if (!ret) {
    MS_LOG(ERROR) << "GraphSinkSession::RunGraph run graph failed";
    return kCoreFailed;
  }
  return kSuccess;
}

Status GraphSinkSession::RunGraph(uint32_t graph_id, const std::vector<tensor::Tensor> &inputs,
                                  std::vector<tensor::Tensor> *outputs) {
  return RunGraph(graph_id, inputs, outputs, nullptr, nullptr);
}

Status GraphSinkSession::Resize(uint32_t graph_id, const std::vector<tensor::Tensor> &inputs,
                                const std::vector<std::vector<int64_t>> &new_shapes) {
  MS_LOG(INFO) << "GraphSinkSession::Resize";
  MS_EXCEPTION_IF_NULL(graph_executor_);
  auto info_it = graph_infos_.find(graph_id);
  if (info_it == graph_infos_.end()) {
    MS_LOG(ERROR) << "Failed to find graph id " << graph_id;
    return kCoreFailed;
  }
  auto &info = info_it->second;
  auto ret = graph_executor_->Resize(graph_id, inputs, new_shapes);
  if (!ret) {
    return kCoreFailed;
  }
  auto new_outputs = graph_executor_->GetOutputInfos(graph_id);
  if (new_outputs.empty()) {
    return kSuccess;
  }
  if (new_outputs.size() != info.outputs.size()) {
    MS_LOG(ERROR) << "Output count " << new_outputs.size() << " get from executor != last output count "
                  << info.outputs.size();
    return kCoreFailed;
  }
  for (size_t i = 0; i < new_shapes.size(); i++) {
    auto &input_shape = new_shapes[i];
    info.inputs[i]->SetShape(input_shape);
    info.inputs[i]->SetData(nullptr, false);  // reset data
  }
  for (size_t i = 0; i < info.outputs.size(); i++) {
    auto &output = new_outputs[i];
    info.outputs[i]->SetShape(output.shape_c());
    info.outputs[i]->SetData(nullptr, false);  // reset data
  }
  return kSuccess;
}
std::vector<MutableTensorImplPtr> GraphSinkSession::GetOutputs(uint32_t graph_id) {
  auto info_it = graph_infos_.find(graph_id);
  if (info_it == graph_infos_.end()) {
    MS_LOG(ERROR) << "Failed to find graph id " << graph_id;
    return {};
  }
  auto &info = info_it->second;
  return info.outputs;
}
std::vector<MutableTensorImplPtr> GraphSinkSession::GetInputs(uint32_t graph_id) {
  auto info_it = graph_infos_.find(graph_id);
  if (info_it == graph_infos_.end()) {
    MS_LOG(ERROR) << "Failed to find graph id " << graph_id;
    return {};
  }
  auto &info = info_it->second;
  return info.inputs;
}
std::vector<std::string> GraphSinkSession::GetOutputNames(uint32_t graph_id) {
  auto info_it = graph_infos_.find(graph_id);
  if (info_it == graph_infos_.end()) {
    MS_LOG(ERROR) << "Failed to find graph id " << graph_id;
    return {};
  }
  auto &info = info_it->second;
  return info.output_names;
}
std::vector<std::string> GraphSinkSession::GetInputNames(uint32_t graph_id) {
  auto info_it = graph_infos_.find(graph_id);
  if (info_it == graph_infos_.end()) {
    MS_LOG(ERROR) << "Failed to find graph id " << graph_id;
    return {};
  }
  auto &info = info_it->second;
  return info.input_names;
}
MutableTensorImplPtr GraphSinkSession::GetOutputByTensorName(uint32_t graph_id, const std::string &tensorName) {
  auto info_it = graph_infos_.find(graph_id);
  if (info_it == graph_infos_.end()) {
    MS_LOG(ERROR) << "Failed to find graph id " << graph_id;
    return {};
  }
  auto &info = info_it->second;
  for (size_t i = 0; i < info.output_names.size(); i++) {
    if (info.output_names[i] == tensorName) {
      return info.outputs[i];
    }
  }
  return nullptr;
}
MutableTensorImplPtr GraphSinkSession::GetInputByTensorName(uint32_t graph_id, const std::string &name) {
  auto info_it = graph_infos_.find(graph_id);
  if (info_it == graph_infos_.end()) {
    MS_LOG(ERROR) << "Failed to find graph id " << graph_id;
    return {};
  }
  auto &info = info_it->second;
  for (size_t i = 0; i < info.input_names.size(); i++) {
    if (info.input_names[i] == name) {
      return info.inputs[i];
    }
  }
  return nullptr;
}
static std::shared_ptr<InferSession> DelegateSessionCreator(const std::shared_ptr<Context> &ctx,
                                                            const ConfigInfos &config_infos) {
  auto &device_contexts = ctx->MutableDeviceInfo();
  if (device_contexts.empty()) {
    return nullptr;
  }
  auto device_type = device_contexts.at(0)->GetDeviceType();
  auto provider = device_contexts.at(0)->GetProvider();

  auto delegate = DelegateRegistry::GetInstance().GetDelegate(device_type, provider, ctx, config_infos);
  if (delegate == nullptr) {
    return nullptr;
  }
  auto session = std::make_shared<GraphSinkSession>(delegate);
  if (provider != kAscendProviderGe) {
    session->Init(ctx);
  }
  session->SetConfigInfo(config_infos);
  return session;
}
REG_SESSION(kDelegateSession, DelegateSessionCreator);
}  // namespace mindspore
