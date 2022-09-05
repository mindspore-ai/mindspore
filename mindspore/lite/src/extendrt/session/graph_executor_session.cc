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

#include <vector>
#include <string>
#include <memory>

#include "extendrt/session/graph_executor_session.h"
#include "src/extendrt/utils/kernel_build_utils.h"
#include "extendrt/utils/tensor_default_impl.h"
#include "extendrt/utils/tensor_utils.h"
#include "extendrt/delegate/graph_executor/litert/graph_executor.h"

namespace mindspore {
Status GraphExecutorSession::Init(const std::shared_ptr<Context> context) {
  MS_LOG(INFO) << "GraphExecutorSession::Init";
  kernel_graph_utils_ = std::make_shared<mindspore::KernelGraphUtils>();
  return kSuccess;
}

Status GraphExecutorSession::CompileGraph(FuncGraphPtr graph, const void *data, size_t size) {
  MS_LOG(INFO) << "GraphExecutorSession::CompileGraph";
  func_graph_ = graph;
  std::vector<KernelGraphPtr> all_out_graph;
  kernel_graph_ = kernel_graph_utils_->ConstructKernelGraph(graph, &all_out_graph, mindspore::device::DeviceType::kCPU);
  MS_EXCEPTION_IF_NULL(kernel_graph_);
  auto &kernel_nodes = kernel_graph_->execution_order();
  for (const auto &kernel_node : kernel_nodes) {
    mindspore::infer::SetKernelInfo(kernel_node);
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
    MS_LOG(ERROR) << "GraphExecutorSession::CompileGraph compile graph failed";
    return kCoreFailed;
  }
  return InitGraphInputsOutputs();
}

Status GraphExecutorSession::InitGraphInputsOutputs() {
  std::vector<tensor::TensorPtr> graph_inputs, graph_outputs;
  kernel_graph_utils_->GetModelInputsInfo(kernel_graph_->graph_id(), &graph_inputs, &input_names_);
  kernel_graph_utils_->GetModelOutputsInfo(kernel_graph_->graph_id(), &graph_outputs, &output_names_);
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

Status GraphExecutorSession::RunGraph() { return kSuccess; }

Status GraphExecutorSession::RunGraph(const std::vector<tensor::Tensor> &inputs, std::vector<tensor::Tensor> *outputs) {
  MS_LOG(INFO) << "GraphExecutorSession::RunGraph";
  MS_EXCEPTION_IF_NULL(graph_executor_);
  MS_EXCEPTION_IF_NULL(outputs);
  bool ret = true;
  if (is_use_kernel_graph_) {
    ret = graph_executor_->RunGraph(kernel_graph_, inputs, outputs, options_);
  } else {
    ret = graph_executor_->RunGraph(func_graph_, inputs, outputs, options_);
  }
  if (!ret) {
    MS_LOG(ERROR) << "GraphExecutorSession::RunGraph run graph failed";
    return kCoreFailed;
  }
  return kSuccess;
}

Status GraphExecutorSession::Resize(const std::vector<tensor::Tensor> &inputs,
                                    const std::vector<std::vector<int64_t>> &new_shapes) {
  MS_LOG(INFO) << "GraphExecutorSession::Resize";
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
std::vector<MutableTensorImplPtr> GraphExecutorSession::GetOutputs() { return outputs_; }
std::vector<MutableTensorImplPtr> GraphExecutorSession::GetInputs() { return inputs_; }
std::vector<std::string> GraphExecutorSession::GetOutputNames() { return output_names_; }
std::vector<std::string> GraphExecutorSession::GetInputNames() { return input_names_; }
MutableTensorImplPtr GraphExecutorSession::GetOutputByTensorName(const std::string &tensorName) {
  for (size_t i = 0; i < output_names_.size(); i++) {
    if (output_names_[i] == tensorName) {
      return outputs_[i];
    }
  }
  return nullptr;
}
MutableTensorImplPtr GraphExecutorSession::GetInputByTensorName(const std::string &name) {
  for (size_t i = 0; i < input_names_.size(); i++) {
    if (input_names_[i] == name) {
      return inputs_[i];
    }
  }
  return nullptr;
}
}  // namespace mindspore
