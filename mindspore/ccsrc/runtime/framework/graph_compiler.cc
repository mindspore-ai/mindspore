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

void GraphCompiler::RunGraph(const GraphId &graph_id, const std::vector<tensor::TensorPtr> &inputs,
                             VectorRef *outputs) {
  MS_EXCEPTION_IF_NULL(session_);
  auto graph = session_->GetGraph(graph_id);
  MS_EXCEPTION_IF_NULL(graph);
  auto actor_set = GraphScheduler::GetInstance().Fetch(graph);
  MS_EXCEPTION_IF_NULL(actor_set);
  GraphScheduler::GetInstance().Run(actor_set);
}

void GraphCompiler::CompileAndRunGraph(session::OpRunInfo *op_run_info, const GraphInfo &graph_info,
                                       std::vector<tensor::TensorPtr> *input_tensors,
                                       const std::vector<int64_t> &tensors_mask, VectorRef *outputs) {
  // Check if the graph cache exists.
  if (run_op_graphs_.find(graph_info) == run_op_graphs_.end()) {
    // Prepare the graph
    MS_EXCEPTION_IF_NULL(session_);
    auto graph = session_->ConstructSingleOpGraph(*op_run_info, *input_tensors, tensors_mask);
    MS_EXCEPTION_IF_NULL(graph);

    MS_EXCEPTION_IF_NULL(device_context_);
    device_context_->SetOperatorInfo(graph->execution_order());

    device_context_->OptimizeSingleOpGraph(graph);
    MS_EXCEPTION_IF_NULL(session_);
    session_->RunOpHideNopNode(graph);

    device_context_->CreateKernel(graph->execution_order());
    run_op_graphs_[graph_info] = graph;
  }

  session_->EraseValueNodeTensor(tensors_mask, input_tensors);

  // wait for allreduce
  for (auto &tensor : *input_tensors) {
    if (tensor->NeedWaitDevice()) {
      tensor->WaitDevice();
    }
  }

  // run op
  auto graph = run_op_graphs_[graph_info];
  MS_EXCEPTION_IF_NULL(graph);
  session_->RunOpRemoveNopNode(graph);

  GraphScheduler::GetInstance().Transform(graph, device_context_, input_tensors, GraphExecutionStrategy::kStep);
  auto actor_set = GraphScheduler::GetInstance().Fetch(graph);
  MS_EXCEPTION_IF_NULL(actor_set);
  GraphScheduler::GetInstance().Run(actor_set, GraphExecutionStrategy::kStep);
}
}  // namespace runtime
}  // namespace mindspore
