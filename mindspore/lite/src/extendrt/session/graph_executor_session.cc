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
#include "extendrt/utils/tensor_utils.h"
#include "src/extendrt/utils/kernel_build_utils.h"

namespace mindspore {
Status GraphExecutorSession::Init(const std::shared_ptr<Context> context) {
  MS_LOG(INFO) << "GraphExecutorSession::Init";
  kernel_graph_utils_ = std::make_shared<mindspore::KernelGraphUtils>();
  return kSuccess;
}

Status GraphExecutorSession::CompileGraph(FuncGraphPtr graph, const void *data, size_t size) {
  MS_LOG(INFO) << "GraphExecutorSession::CompileGraph";
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

Status GraphExecutorSession::RunGraph() { return kSuccess; }
Status GraphExecutorSession::RunGraph(const std::vector<tensor::TensorPtr> &inputs,
                                      std::vector<tensor::TensorPtr> *outputs) {
  MS_LOG(INFO) << "GraphExecutorSession::RunGraph";
  MS_EXCEPTION_IF_NULL(outputs);
  std::vector<tensor::Tensor> executor_inputs, executor_outputs;
  executor_inputs = TensorUtils::TensorPtrToTensor(inputs);
  auto ret = graph_executor_->RunGraph(kernel_graph_, executor_inputs, &executor_outputs, options_);
  if (!ret) {
    return kCoreFailed;
  }
  *outputs = TensorUtils::TensorToTensorPtr(executor_outputs);
  inputs_ = inputs;
  outputs_ = *outputs;
  return kSuccess;
}
Status GraphExecutorSession::Resize(const std::vector<tensor::TensorPtr> &inputs,
                                    const std::vector<std::vector<int64_t>> &dims) {
  return kSuccess;
}
std::vector<tensor::TensorPtr> GraphExecutorSession::GetOutputs() { return outputs_; }
std::vector<tensor::TensorPtr> GraphExecutorSession::GetInputs() { return inputs_; }
std::vector<std::string> GraphExecutorSession::GetOutputNames() { return output_names_; }
std::vector<std::string> GraphExecutorSession::GetInputNames() { return input_names_; }
tensor::TensorPtr GraphExecutorSession::GetOutputByTensorName(const std::string &tensorName) { return nullptr; }
tensor::TensorPtr GraphExecutorSession::GetInputByTensorName(const std::string &name) { return nullptr; }
}  // namespace mindspore
