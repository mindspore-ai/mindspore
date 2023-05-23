/**
 * Copyright 2019-2023 Huawei Technologies Co., Ltd
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
#include <utility>
#include "extendrt/utils/tensor_utils.h"
#include "src/extendrt/utils/kernel_build_utils.h"
#include "extendrt/delegate/factory.h"
#include "extendrt/session/factory.h"
#include "extendrt/utils/tensor_default_impl.h"
#include "src/litert/cxx_api/tensor/tensor_impl.h"
#include "src/extendrt/delegate/graph_executor/litert/func_graph_reuse_manager.h"
#include "src/extendrt/delegate/plugin/ascend_ge_executor_plugin.h"
#include "extendrt/utils/func_graph_utils.h"
#include "common/common.h"

namespace mindspore {
namespace {
constexpr auto kDataFlowGraphType = "data_flow";
constexpr auto kIsAdapted = "is_adapted";

std::mutex kernel_graph_mutex;
std::mutex g_build_graph_mutex;
}  // namespace

GraphSinkSession::~GraphSinkSession() = default;

Status GraphSinkSession::Init(const std::shared_ptr<Context> &context, const ConfigInfos &config_info) {
  MS_LOG(INFO) << "GraphSinkSession::Init";
  if (graph_executor_ == nullptr) {
    MS_LOG(ERROR) << "GraphSinkSession::Init failed, graph executor is nullptr.";
    return kLiteUninitializedObj;
  }
  context_ = context;
  config_infos_ = config_info;
  return kSuccess;
}

Status GraphSinkSession::CompileGraph(const void *model_data, size_t data_size, uint32_t *graph_id) {
  MS_LOG(INFO) << "GraphSinkSession::CompileGraph";
  // This lock can be removed when LiteRT supports concurrent multithreading compilation.
  std::lock_guard<std::mutex> lock(g_build_graph_mutex);
  auto ret = graph_executor_->CompileGraph(model_data, data_size, options_, graph_id);
  if (!ret) {
    MS_LOG(ERROR) << "GraphSinkSession::CompileGraph compile graph failed";
    return kCoreFailed;
  }
  DelegateGraphInfo graph_info;
  auto status = InitGraphInfo(*graph_id, &graph_info);
  if (!status.IsOk()) {
    MS_LOG(ERROR) << "Failed to get inputs and outputs info from graph";
    return status;
  }
  graph_infos_[*graph_id] = graph_info;
  return kSuccess;
}

Status GraphSinkSession::CompileGraph(FuncGraphPtr graph, const void *data, size_t size, uint32_t *graph_id) {
  MS_LOG(INFO) << "GraphSinkSession::CompileGraph";
  // This lock can be removed when LiteRT supports concurrent multithreading compilation.
  std::lock_guard<std::mutex> lock(g_build_graph_mutex);
  // kernel graph will be removed from GraphSinkSession, and this code will be moved to TensorRT plugin
  auto func_type = graph->get_attr(kAttrFuncType);
  is_data_flow_graph_ = func_type != nullptr && GetValue<std::string>(func_type) == kDataFlowGraphType;
  if (context_ && !context_->MutableDeviceInfo().empty()) {
    auto device_info = context_->MutableDeviceInfo()[0];
    bool is_ge_backend = device_info && device_info->GetDeviceType() == DeviceType::kAscend &&
                         device_info->GetProvider() == lite::kAscendProviderGe;
    bool is_adapted = graph->has_attr(kIsAdapted);  // The funcgraph will only adapted once while running parallel.
    if (is_ge_backend && !is_adapted && !is_data_flow_graph_) {
      lite::AscendGeExecutorPlugin::GetInstance().AdaptGraph(graph);
      graph->set_attr(kIsAdapted, MakeValue(true));
    }
  }
  DelegateGraphInfo graph_info;
  auto ret = graph_executor_->CompileGraph(graph, options_, graph_id);
  if (!ret) {
    MS_LOG(ERROR) << "GraphSinkSession::CompileGraph compile graph failed";
    return kCoreFailed;
  }
  auto status = !is_data_flow_graph_ ? InitGraphInfo(*graph_id, &graph_info) : kSuccess;
  if (!status.IsOk()) {
    MS_LOG(ERROR) << "Failed to update inputs and outputs info from graph executor";
    return status;
  }
  graph_infos_[*graph_id] = graph_info;
  return kSuccess;
}

Status GraphSinkSession::InitGraphInfo(uint32_t graph_id, DelegateGraphInfo *graph_info_ptr) {
  auto &info = *graph_info_ptr;

  auto new_inputs = graph_executor_->GetInputInfos(graph_id);
  if (new_inputs.empty()) {
    MS_LOG(ERROR) << "Input is empty.";
    return kCoreFailed;
  }
  info.inputs.clear();
  info.input_names.clear();
  for (size_t i = 0; i < new_inputs.size(); i++) {
    auto impl = std::make_shared<LiteTensorImpl>(new_inputs[i]);
    if (impl == nullptr) {
      MS_LOG(ERROR) << "Create tensor failed.";
      return kCoreFailed;
    }
    info.input_names.push_back(impl->Name());
    info.inputs.push_back(impl);
  }

  auto new_outputs = graph_executor_->GetOutputInfos(graph_id);
  if (new_outputs.empty()) {
    MS_LOG(ERROR) << "Output is empty.";
    return kCoreFailed;
  }

  info.outputs.clear();
  info.output_names.clear();
  for (size_t i = 0; i < new_outputs.size(); i++) {
    auto impl = std::make_shared<LiteTensorImpl>(new_outputs[i]);
    if (impl == nullptr) {
      MS_LOG(ERROR) << "Create tensor failed.";
      return kCoreFailed;
    }
    info.output_names.push_back(impl->Name());
    info.outputs.push_back(impl);
  }
  return kSuccess;
}

Status GraphSinkSession::RunGraph(uint32_t graph_id, const std::vector<lite::Tensor *> &inputs,
                                  std::vector<lite::Tensor *> *outputs, const MSKernelCallBack &before,
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
  if (is_data_flow_graph_) {
    DelegateGraphInfo graph_info;
    InitGraphInfo(graph_id, &graph_info);
    graph_infos_[graph_id] = graph_info;
  }
  return kSuccess;
}

Status GraphSinkSession::RunGraph(uint32_t graph_id, const std::vector<lite::Tensor *> &inputs,
                                  std::vector<lite::Tensor *> *outputs) {
  return RunGraph(graph_id, inputs, outputs, nullptr, nullptr);
}

Status GraphSinkSession::Resize(uint32_t graph_id, const std::vector<lite::Tensor *> &inputs,
                                const std::vector<std::vector<int64_t>> &new_shapes) {
  MS_LOG(INFO) << "GraphSinkSession::Resize";
  MS_EXCEPTION_IF_NULL(graph_executor_);
  auto ret = graph_executor_->Resize(graph_id, inputs, new_shapes);
  if (!ret) {
    return kCoreFailed;
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
  auto ret = session->Init(ctx, config_infos);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "Init session failed.";
    return nullptr;
  }
  return session;
}
REG_SESSION(kDelegateSession, DelegateSessionCreator);
}  // namespace mindspore
