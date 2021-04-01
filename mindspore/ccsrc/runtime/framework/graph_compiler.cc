/**
 * Copyright 2021 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License"){}
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

#include "runtime/framework/graph_compiler.h"
#include "runtime/framework/graph_scheduler.h"

namespace mindspore {
namespace runtime {
void GraphCompiler::set_device_context(device::DeviceContext *device_context) {
  MS_EXCEPTION_IF_NULL(device_context);
  device_context_ = device_context;

  // The member variable 'session_' will be removed after removing session module.
  if (session_ == nullptr) {
    session_ = std::make_shared<session::SessionBasic>();
  }
}

GraphId GraphCompiler::CompileGraph(const AnfNodePtrList &nodes, const AnfNodePtrList &outputs) {
  MS_EXCEPTION_IF_NULL(session_);
  // Generate kernel graph.
  auto graph = session_->ConstructKernelGraph(nodes, outputs);
  MS_EXCEPTION_IF_NULL(graph);
  return CompileGraphImpl(graph);
}

GraphId GraphCompiler::CompileGraphImpl(const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(device_context_);
  // Optimization pass which is irrelevant to device type or format.
  device_context_->OptimizeGraphWithoutDeviceInfo(graph);

  device_context_->SetOperatorInfo(graph->execution_order());

  // Optimization pass which is relevant to device type or format.
  device_context_->OptimizeGraphWithDeviceInfo(graph);

  // Generate 'KernelMod' for all kernels and set 'KernelMod' into kernel,
  // 'KernelMod' is real executive object of kernel.
  device_context_->CreateKernel(graph->execution_order());

  // Transform graph to actor DAG, contains build and link.
  GraphScheduler::GetInstance().Transform(graph, device_context_);
  return graph->graph_id();
}

GraphId GraphCompiler::CompileGraph(session::OpRunInfo *op_run_info, const GraphInfo &graph_info,
                                    std::vector<tensor::TensorPtr> *input_tensors,
                                    const std::vector<int64_t> &tensors_mask) {
  // Check if the graph cache exists.
  auto iter = run_op_graphs_.find(graph_info);
  if (iter != run_op_graphs_.end()) {
    const auto &graph = iter->second;
    MS_EXCEPTION_IF_NULL(graph);
    return graph->graph_id();
  }
  // Generate kernel graph.
  MS_EXCEPTION_IF_NULL(session_);
  auto graph = session_->ConstructSingleOpGraph(*op_run_info, *input_tensors, tensors_mask);
  MS_EXCEPTION_IF_NULL(graph);

  MS_EXCEPTION_IF_NULL(device_context_);
  device_context_->SetOperatorInfo(graph->execution_order());

  device_context_->OptimizeSingleOpGraph(graph);
  MS_EXCEPTION_IF_NULL(session_);
  session_->RunOpHideNopNode(graph);
  session_->RunOpRemoveNopNode(graph);

  // Generate 'KernelMod' for kernel in graph.
  device_context_->CreateKernel(graph->execution_order());

  // Transform graph to actor DAG, contains build and link.
  GraphScheduler::GetInstance().Transform(graph, device_context_, input_tensors, GraphExecutionStrategy::kStep);
  run_op_graphs_[graph_info] = graph;
  return graph->graph_id();
}

KernelGraphPtr GraphCompiler::Fetch(GraphId graph_id) const {
  MS_EXCEPTION_IF_NULL(session_);
  return session_->GetGraph(graph_id);
}

KernelGraphPtr GraphCompiler::Fetch(const GraphInfo &graph_info) const {
  auto iter = run_op_graphs_.find(graph_info);
  if (iter == run_op_graphs_.end()) {
    MS_LOG(ERROR) << "Can't find graph for: " << graph_info;
    return nullptr;
  }
  return iter->second;
}
}  // namespace runtime
}  // namespace mindspore
