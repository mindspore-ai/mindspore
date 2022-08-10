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
#include <memory>
#include "extendrt/utils/tensor_utils.h"
#include "src/extendrt/utils/kernel_build_utils.h"
#include "extendrt/delegate/factory.h"
#include "extendrt/session/factory.h"
namespace mindspore {
Status GraphSinkSession::Init(const std::shared_ptr<Context> &context) {
  MS_LOG(INFO) << "GraphSinkSession::Init";
  kernel_graph_utils_ = std::make_shared<mindspore::KernelGraphUtils>();
  return kSuccess;
}
Status GraphSinkSession::CompileGraph(FuncGraphPtr graph, const void *data, size_t size) {
  MS_LOG(INFO) << "GraphSinkSession::CompileGraph";
  std::vector<KernelGraphPtr> all_out_graph;
  kernel_graph_ = kernel_graph_utils_->ConstructKernelGraph(graph, &all_out_graph, mindspore::device::DeviceType::kCPU);
  MS_EXCEPTION_IF_NULL(kernel_graph_);
  auto &kernel_nodes = kernel_graph_->execution_order();
  for (const auto &kernel_node : kernel_nodes) {
    mindspore::infer::SetKernelInfo(kernel_node);
  }
  if (graph_executor_->CompileGraph(kernel_graph_, options_)) {
    kernel_graph_utils_->GetModelInputsInfo(kernel_graph_->graph_id(), &inputs_, &input_names_);
    kernel_graph_utils_->GetModelOutputsInfo(kernel_graph_->graph_id(), &outputs_, &output_names_);
    return kSuccess;
  }
  return kCoreFailed;
}

Status GraphSinkSession::RunGraph(const std::vector<tensor::Tensor> &inputs, std::vector<tensor::Tensor> *outputs) {
  return kSuccess;
}
std::vector<MutableTensorImplPtr> GraphSinkSession::GetOutputs() { return {}; }
std::vector<MutableTensorImplPtr> GraphSinkSession::GetInputs() { return {}; }
std::vector<std::string> GraphSinkSession::GetOutputNames() { return std::vector<std::string>(); }
std::vector<std::string> GraphSinkSession::GetInputNames() { return std::vector<std::string>(); }
MutableTensorImplPtr GraphSinkSession::GetOutputByTensorName(const std::string &tensorName) { return nullptr; }
MutableTensorImplPtr GraphSinkSession::GetInputByTensorName(const std::string &name) { return nullptr; }
static std::shared_ptr<InferSession> DelegateSessionCreator(const std::shared_ptr<Context> &ctx) {
  auto &device_contexts = ctx->MutableDeviceInfo();
  if (device_contexts.empty()) {
    return nullptr;
  }
  auto device_type = device_contexts.at(0)->GetDeviceType();
  auto provider = device_contexts.at(0)->GetProvider();

  auto delegate = DelegateRegistry::GetInstance().GetDelegate(device_type, provider, ctx);
  auto session = std::make_shared<GraphSinkSession>(delegate);
  session->Init(ctx);
  return session;
}
REG_SESSION(kDelegateSession, DelegateSessionCreator);
}  // namespace mindspore
