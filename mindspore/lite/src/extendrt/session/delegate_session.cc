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
#include "extendrt/session/optimizer/tensorrt_optimizer.h"
#include "src/extendrt/delegate/graph_executor/litert/func_graph_reuse_manager.h"
#include "src/extendrt/delegate/plugin/ascend_ge_executor_plugin.h"

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
  kernel_graph_utils_ = std::make_shared<mindspore::KernelGraphUtils>();
  context_ = context;
  return kSuccess;
}

Status GraphSinkSession::CompileGraph(FuncGraphPtr graph, const void *data, size_t size) {
  MS_LOG(INFO) << "GraphSinkSession::CompileGraph";
  // This lock can be removed when LiteRT supports concurrent multithreading compilation.
  std::lock_guard<std::mutex> lock(g_build_graph_mutex);
  // kernel graph will be removed from GraphSinkSession, and this code will be moved to TensorRT plugin
  if (context_ && !context_->MutableDeviceInfo().empty()) {
    auto device_info = context_->MutableDeviceInfo()[0];
    if (device_info && device_info->GetDeviceType() == DeviceType::kGPU && device_info->GetProvider() == "tensorrt") {
      TensorRtOptimizer optimizer;
      optimizer.RunOptimizer(graph);
    }
    if (device_info && device_info->GetDeviceType() == DeviceType::kAscend &&
        device_info->GetProvider() == kAscendProviderGe) {
      lite::AscendGeExecutorPlugin::GetInstance().AdaptGraph(graph);
    }
  }
  func_graph_ = graph;
  std::vector<KernelGraphPtr> all_out_graph;
  {
    std::unique_lock<std::mutex> l(kernel_graph_mutex);
    kernel_graph_ = FuncGraphReuseManager::GetInstance()->GetKernelGraph(config_infos_);
    if (kernel_graph_ == nullptr) {
      kernel_graph_ =
        kernel_graph_utils_->ConstructKernelGraph(graph, &all_out_graph, mindspore::device::DeviceType::kCPU);
      MS_EXCEPTION_IF_NULL(kernel_graph_);
      auto &kernel_nodes = kernel_graph_->execution_order();
      for (const auto &kernel_node : kernel_nodes) {
        mindspore::infer::SetKernelInfo(kernel_node);
      }
      FuncGraphReuseManager::GetInstance()->StoreKernelGraph(config_infos_, kernel_graph_);
    } else {
      MS_LOG(INFO) << "the kernel graph is the same as the last time. We do not need to construct, and we can directly "
                      "use the cached kernel graph.";
    }
  }
  bool ret = true;
  if (is_use_kernel_graph_) {
    if (!graph_executor_->CompileGraph(kernel_graph_, options_)) {
      is_use_kernel_graph_ = false;
      ret = graph_executor_->CompileGraph(func_graph_, options_);
    }
  } else {
    ret = graph_executor_->CompileGraph(func_graph_, options_);
  }
  if (!ret) {
    MS_LOG(ERROR) << "GraphSinkSession::CompileGraph compile graph failed";
    return kCoreFailed;
  }
  return InitGraphInputsOutputs();
}

Status GraphSinkSession::InitGraphInputsOutputs() {
  std::vector<tensor::TensorPtr> graph_inputs, graph_outputs;
  {
    std::unique_lock<std::mutex> l(kernel_graph_mutex);
    FuncGraphReuseManager::GetInstance()->GetInOut(config_infos_, &graph_inputs, &graph_outputs, &input_names_,
                                                   &output_names_);
    if (graph_inputs.empty() || graph_outputs.empty() || input_names_.empty() || output_names_.empty()) {
      kernel_graph_utils_->GetModelInputsInfo(kernel_graph_->graph_id(), &graph_inputs, &input_names_);
      kernel_graph_utils_->GetModelOutputsInfo(kernel_graph_->graph_id(), &graph_outputs, &output_names_);
      FuncGraphReuseManager::GetInstance()->StoreInOut(config_infos_, graph_inputs, graph_outputs, input_names_,
                                                       output_names_);
    } else {
      MS_LOG(INFO) << "the input and output are the same as the last time. We do not need to construct, and we can "
                      "directly use the cached input and output info.";
    }
  }
  if (graph_inputs.size() != input_names_.size()) {
    MS_LOG(ERROR) << "Graph input size " << graph_inputs.size() << " != input names size " << input_names_.size();
    return kCoreFailed;
  }
  if (graph_outputs.size() != output_names_.size()) {
    MS_LOG(ERROR) << "Graph output size " << graph_outputs.size() << " != output names size " << output_names_.size();
    return kCoreFailed;
  }
  inputs_.clear();
  auto new_inputs = graph_executor_->GetInputInfos(kernel_graph_);
  if (new_inputs.empty()) {
    for (size_t i = 0; i < input_names_.size(); i++) {
      auto &input = graph_inputs[i];
      auto data_type = static_cast<enum DataType>(input->data_type());
      auto impl = std::make_shared<TensorDefaultImpl>(input_names_[i], data_type, input->shape_c());
      inputs_.push_back(impl);
    }
  } else {
    if (new_inputs.size() != input_names_.size()) {
      MS_LOG(ERROR) << "Input count " << new_inputs.size() << " get from executor != input names count "
                    << input_names_.size();
      return kCoreFailed;
    }
    for (size_t i = 0; i < input_names_.size(); i++) {
      auto &input = new_inputs[i];
      auto data_type = static_cast<enum DataType>(input.data_type());
      auto impl = std::make_shared<TensorDefaultImpl>(input_names_[i], data_type, input.shape_c());
      inputs_.push_back(impl);
    }
  }
  outputs_.clear();
  auto new_outputs = graph_executor_->GetOutputInfos(kernel_graph_);
  if (new_outputs.empty()) {
    for (size_t i = 0; i < output_names_.size(); i++) {
      auto &output = graph_outputs[i];
      auto data_type = static_cast<enum DataType>(output->data_type());
      auto impl = std::make_shared<TensorDefaultImpl>(output_names_[i], data_type, output->shape_c());
      outputs_.push_back(impl);
    }
  } else {
    if (new_outputs.size() != output_names_.size()) {
      MS_LOG(ERROR) << "Output count " << new_outputs.size() << " get from executor != output names count "
                    << output_names_.size();
      return kCoreFailed;
    }
    for (size_t i = 0; i < output_names_.size(); i++) {
      auto &output = new_outputs[i];
      auto data_type = static_cast<enum DataType>(output.data_type());
      auto impl = std::make_shared<TensorDefaultImpl>(output_names_[i], data_type, output.shape_c());
      outputs_.push_back(impl);
    }
  }
  return kSuccess;
}

Status GraphSinkSession::RunGraph(const std::vector<tensor::Tensor> &inputs, std::vector<tensor::Tensor> *outputs,
                                  const MSKernelCallBack &before, const MSKernelCallBack &after) {
  MS_LOG(INFO) << "GraphSinkSession::RunGraph";
  MS_EXCEPTION_IF_NULL(outputs);
  graph_executor_->SetBefore(before);
  graph_executor_->SetAfter(after);
  bool ret = true;
  if (is_use_kernel_graph_) {
    ret = graph_executor_->RunGraph(kernel_graph_, inputs, outputs, options_);
  } else {
    ret = graph_executor_->RunGraph(func_graph_, inputs, outputs, options_);
  }
  if (!ret) {
    MS_LOG(ERROR) << "GraphSinkSession::RunGraph run graph failed";
    return kCoreFailed;
  }
  return kSuccess;
}

Status GraphSinkSession::RunGraph(const std::vector<tensor::Tensor> &inputs, std::vector<tensor::Tensor> *outputs) {
  return RunGraph(inputs, outputs, nullptr, nullptr);
}

Status GraphSinkSession::Resize(const std::vector<tensor::Tensor> &inputs,
                                const std::vector<std::vector<int64_t>> &new_shapes) {
  MS_LOG(INFO) << "GraphSinkSession::Resize";
  MS_EXCEPTION_IF_NULL(graph_executor_);
  auto ret = graph_executor_->Resize(kernel_graph_, inputs, new_shapes);
  if (!ret) {
    return kCoreFailed;
  }
  auto new_outputs = graph_executor_->GetOutputInfos(kernel_graph_);
  if (new_outputs.empty()) {
    return kSuccess;
  }
  if (new_outputs.size() != outputs_.size()) {
    MS_LOG(ERROR) << "Output count " << new_outputs.size() << " get from executor != last output count "
                  << outputs_.size();
    return kCoreFailed;
  }
  for (size_t i = 0; i < new_shapes.size(); i++) {
    auto &input_shape = new_shapes[i];
    inputs_[i]->SetShape(input_shape);
    inputs_[i]->SetData(nullptr, false);  // reset data
  }
  for (size_t i = 0; i < outputs_.size(); i++) {
    auto &output = new_outputs[i];
    outputs_[i]->SetShape(output.shape_c());
    outputs_[i]->SetData(nullptr, false);  // reset data
  }
  return kSuccess;
}
std::vector<MutableTensorImplPtr> GraphSinkSession::GetOutputs() { return outputs_; }
std::vector<MutableTensorImplPtr> GraphSinkSession::GetInputs() { return inputs_; }
std::vector<std::string> GraphSinkSession::GetOutputNames() { return output_names_; }
std::vector<std::string> GraphSinkSession::GetInputNames() { return input_names_; }
MutableTensorImplPtr GraphSinkSession::GetOutputByTensorName(const std::string &tensorName) {
  for (size_t i = 0; i < output_names_.size(); i++) {
    if (output_names_[i] == tensorName) {
      return outputs_[i];
    }
  }
  return nullptr;
}
MutableTensorImplPtr GraphSinkSession::GetInputByTensorName(const std::string &name) {
  for (size_t i = 0; i < input_names_.size(); i++) {
    if (input_names_[i] == name) {
      return inputs_[i];
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
