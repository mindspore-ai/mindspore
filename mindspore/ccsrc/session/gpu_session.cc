/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#include "session/gpu_session.h"
#include "device/gpu/kernel_info_setter.h"
#include "device/gpu/gpu_kernel_build.h"
#include "device/gpu/gpu_kernel_runtime.h"
#include "pre_activate/common/optimizer.h"
#include "pre_activate/common/pass_manager.h"
#include "pre_activate/ascend/ir_fusion/allreduce_fusion.h"
#include "device/kernel_runtime_manager.h"
#include "predict/predict.h"
#include "common/utils.h"
#include "utils/context/ms_context.h"

namespace mindspore {
namespace session {
namespace gpu {
using AnfAlgo = mindspore::session::AnfRuntimeAlgorithm;

void GPUSession::SelectKernel(const std::shared_ptr<KernelGraph> &kernel_graph) const {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  for (const auto &kernel_node : kernel_graph->execution_order()) {
    MS_EXCEPTION_IF_NULL(kernel_node);
    device::gpu::SetKernelInfo(kernel_node);
  }
}

void GPUSession::StartKernelRT() const {
  auto runtime_instance = device::KernelRuntimeManager::Instance().GetSingleKernelRuntime(kGPUDevice, device_id_);
  MS_EXCEPTION_IF_NULL(runtime_instance);
  if (!runtime_instance->Init()) {
    MS_LOG(EXCEPTION) << "GPU start kernel runtime failed";
  }
}

void GPUSession::Optimize(const std::shared_ptr<KernelGraph> &kernel_graph) {
  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  pm->AddPass(std::make_shared<opt::AllReduceFusion>());
  optimizer->AddPassManager(pm);
  (void)optimizer->Optimize(kernel_graph);
  kernel_graph->SetExecOrderByDefault();
}

void GPUSession::BuildKernel(const std::shared_ptr<KernelGraph> &kernel_graph) const {
  device::gpu::GpuBuild(kernel_graph);
}

void GPUSession::AllocateMemory(KernelGraph *kernel_graph) const {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto runtime_instance = device::KernelRuntimeManager::Instance().GetSingleKernelRuntime(kGPUDevice, device_id_);
  MS_EXCEPTION_IF_NULL(runtime_instance);
  runtime_instance->AssignMemory(kernel_graph);
}

void GPUSession::RunOpAllocateMemory(const std::vector<tensor::TensorPtr> &input_tensors,
                                     KernelGraph *kernel_graph) const {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto runtime_instance = device::KernelRuntimeManager::Instance().GetSingleKernelRuntime(kGPUDevice, device_id_);
  MS_EXCEPTION_IF_NULL(runtime_instance);
  runtime_instance->RunOpAssignMemory(input_tensors, kernel_graph);
}

void GPUSession::Execute(const std::shared_ptr<KernelGraph> &kernel_graph) const {
  auto runtime_instance = device::KernelRuntimeManager::Instance().GetSingleKernelRuntime(kGPUDevice, device_id_);
  MS_EXCEPTION_IF_NULL(runtime_instance);
  if (!runtime_instance->Run(kernel_graph.get())) {
    MS_LOG(EXCEPTION) << "GPU execute graph failed!";
  }
}

GraphId GPUSession::CompileGraph(const AnfNodePtrList &lst, const AnfNodePtrList &outputs) {
  // Construct graph, if construct successs, graph_sum_ + 1
  auto graph_id = graph_sum_;
  auto graph = ConstructKernelGraph(lst, outputs);
  // Select kernel build info
  SelectKernel(graph);
  // Convert kernel Graph to model
  predictmodel::StepConvertGraph(graph);
  // Start gpu kernel runtime
  StartKernelRT();
  // AllReduce Optimize
  Optimize(graph);
  // Build kernel if node is cnode
  BuildKernel(graph);
  // Set graph execution order before memory alloc, ensure that memory alloc is according to the reorder graph
  auto execution_order = graph->execution_order();
  Reorder(&execution_order);
  graph->set_execution_order(execution_order);
  // Alloc memeory, include static memory and dynamic memory
  AllocateMemory(graph.get());
  // Reset memory resource
  auto runtime_instance = device::KernelRuntimeManager::Instance().GetSingleKernelRuntime(kGPUDevice, device_id_);
  MS_EXCEPTION_IF_NULL(runtime_instance);
  runtime_instance->FreeHostMemory();
  return graph_id;
}

void GPUSession::RunGraph(const GraphId &graph_id, const std::vector<tensor::TensorPtr> &inputs, VectorRef *outputs) {
  auto &kernel_graph = graphs_[graph_id];
  // Load input data from user input
  LoadInputData(kernel_graph, inputs);
  MS_EXCEPTION_IF_NULL(kernel_graph);
  // Convert inputs to model
  predictmodel::StepConvertWeight(inputs);
  // Run graph on GPU
  Execute(kernel_graph);
  // Get result from GPU
  UpdateOutputs(kernel_graph, outputs, inputs);
  // Summary
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  if (context_ptr->enable_gpu_summary()) {
    Summary(kernel_graph.get());
  }
}

void GPUSession::BuildOp(const OpRunInfo &op_run_info, const GraphInfo &graph_info) {
  // Prepare the graph
  auto kernel_graph = ConstructSingleOpGraph(op_run_info);
  MS_EXCEPTION_IF_NULL(kernel_graph);
  SelectKernel(kernel_graph);
  StartKernelRT();
  BuildKernel(kernel_graph);
  run_op_graphs_[graph_info] = kernel_graph;
}

py::tuple GPUSession::RunOp(const OpRunInfo &op_run_info, const GraphInfo &graph_info) {
  auto kernel_graph = run_op_graphs_[graph_info];
  MS_EXCEPTION_IF_NULL(kernel_graph);
  std::vector<tensor::TensorPtr> input_tensors = {};
  std::vector<bool> tensors_mask = {};
  ToTensorPtr(op_run_info, &input_tensors, &tensors_mask);
  RunOpAllocateMemory(input_tensors, kernel_graph.get());
  // Execute the computation
  LoadInputData(kernel_graph, input_tensors);
  Execute(kernel_graph);
  // Fetch outputs
  VectorRef outputs;
  UpdateOutputs(kernel_graph, &outputs, input_tensors);
  // Trans output to tuple
  auto output_tensors = TransformBaseRefListToTuple(outputs);
  if (!utils::isa<PyObjectRef>(output_tensors) ||
      !py::isinstance<py::tuple>(utils::cast<PyObjectRef>(output_tensors).object_)) {
    MS_EXCEPTION(NotSupportError) << "The output tensors should be a tuple !";
  }
  py::object tuple_obj = utils::cast<PyObjectRef>(output_tensors).object_;
  py::tuple tuple_tensors = py::cast<py::tuple>(tuple_obj);
  run_op_graphs_.clear();
  return tuple_tensors;
}
}  // namespace gpu
}  // namespace session
}  // namespace mindspore
