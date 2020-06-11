/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#include "session/cpu_session.h"
#include <algorithm>
#include "ir/tensor.h"
#include "ir/anf.h"
#include "kernel/kernel.h"
#include "common/utils.h"
#include "session/anf_runtime_algorithm.h"
#include "device/kernel_runtime.h"
#include "predict/predict.h"
#include "kernel/cpu/cpu_kernel_factory.h"
#include "device/cpu/kernel_select_cpu.h"

namespace mindspore {
namespace session {
ParameterPtr CPUSession::CreateNewParameterFromParameter(const AnfNodePtr &anf, bool valid_input, KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(anf);
  if (!anf->isa<Parameter>()) {
    MS_LOG(EXCEPTION) << "anf[" << anf->DebugString() << "] is not a parameter";
  }
  auto valid_inputs = graph->MutableValidInputs();
  MS_EXCEPTION_IF_NULL(valid_inputs);
  auto graph_inputs = graph->MutableInputs();
  MS_EXCEPTION_IF_NULL(graph_inputs);
  TraceManager::DebugTrace(std::make_shared<TraceCopy>(anf->debug_info()));
  ParameterPtr new_parameter = graph->NewParameter(anf->cast<ParameterPtr>());
  TraceManager::EndTrace();
  graph_inputs->push_back(new_parameter);
  valid_inputs->push_back(valid_input);
  return new_parameter;
}

GraphId CPUSession::CompileGraph(const AnfNodePtrList &lst, const AnfNodePtrList &outputs) {
  auto graph_id = graph_sum_;
  auto graph = ConstructKernelGraph(lst, outputs);
  MS_EXCEPTION_IF_NULL(graph);
  MS_LOG(INFO) << "Set kernel info";
  SetKernelInfo(graph.get());
  predictmodel::StepConvertGraph(graph);
  MS_LOG(INFO) << "Build kernel";
  BuildKernel(graph.get());
  MS_LOG(INFO) << "Assign kernel address";
  runtime_.AssignKernelAddress(graph.get());
  return graph_id;
}

void CPUSession::RunGraph(const GraphId &graph_id, const std::vector<tensor::TensorPtr> &inputs, VectorRef *outputs) {
  auto &kernel_graph = graphs_[graph_id];
  MS_EXCEPTION_IF_NULL(kernel_graph);
  MS_LOG(INFO) << "Bind input output address";
  runtime_.BindInputOutput(kernel_graph.get(), inputs, outputs);
  MS_LOG(INFO) << "Run graph start";
  predictmodel::StepConvertWeight(inputs);
  auto execution_order = kernel_graph->execution_order();
  Reorder(&execution_order);

  bool enable_summary = summary_callback_ != nullptr;
  kernel_graph->set_execution_order(execution_order);
  NamedSummaryOutputs summary_outputs;
  if (enable_summary) {
    GetSummaryNodes(kernel_graph.get());
    summary_outputs = kernel_graph->summary_nodes();
    runtime_.IncreaseSummaryRefCount(summary_outputs);
  }

  bool ret = runtime_.Run(kernel_graph.get());
  if (!ret) {
    MS_LOG(EXCEPTION) << "Run graph failed";
  }

  if (enable_summary) {
    Summary(kernel_graph.get());
    runtime_.DecreaseSummaryRefCount(summary_outputs);
  }

  MS_LOG(INFO) << "Run graph end";
}

void CPUSession::SetKernelInfo(const KernelGraph *kernel_graph) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto &kernel_nodes = kernel_graph->execution_order();
  for (const auto &kernel_node : kernel_nodes) {
    MS_EXCEPTION_IF_NULL(kernel_node);
    device::cpu::SetKernelInfo(kernel_node);
  }
}

void CPUSession::BuildKernel(const KernelGraph *kernel_graph) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto &kernel_nodes = kernel_graph->execution_order();
  for (const auto &kernel_node : kernel_nodes) {
    MS_EXCEPTION_IF_NULL(kernel_node);
    std::string kernel_name = AnfAlgo::GetCNodeName(kernel_node);
    MS_LOG(INFO) << "Cpu building operator[" << kernel_name << "].";
    std::shared_ptr<kernel::CPUKernel> cpu_kernel =
      kernel::CPUKernelFactory::GetInstance().Create(kernel_name, kernel_node);
    if (cpu_kernel == nullptr) {
      MS_LOG(EXCEPTION) << "Operator[" << kernel_name << "] is not support.";
    }
    cpu_kernel->Init(kernel_node);
    AnfAlgo::SetKernelMod(cpu_kernel, kernel_node.get());
    MS_LOG(INFO) << "Cpu build success operator[" << kernel_name << "].";
  }
}
}  // namespace session
}  // namespace mindspore
