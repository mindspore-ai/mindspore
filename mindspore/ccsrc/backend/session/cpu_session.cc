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

#include "backend/session/cpu_session.h"
#include <algorithm>
#include <sstream>
#include "ir/tensor.h"
#include "ir/anf.h"
#include "backend/kernel_compiler/kernel.h"
#include "utils/ms_utils.h"
#include "backend/session/anf_runtime_algorithm.h"
#include "runtime/device/kernel_runtime.h"
#include "backend/kernel_compiler/cpu/cpu_kernel_factory.h"
#include "runtime/device/cpu/kernel_select_cpu.h"
#include "backend/optimizer/common/optimizer.h"
#include "backend/optimizer/common/pass_manager.h"
#include "backend/optimizer/pass/replace_node_by_proxy.h"
#ifdef ENABLE_DEBUGGER
#include "debug/debugger/debugger.h"
#endif
#if (ENABLE_CPU && (ENABLE_D || ENABLE_GPU))
#include "frontend/parallel/ps/util.h"
#endif

namespace mindspore {
namespace session {
ParameterPtr CPUSession::CreateNewParameterFromParameter(const AnfNodePtr &anf, bool valid_input, KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(anf);
  MS_EXCEPTION_IF_NULL(graph);
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

void CPUSession::Optimize(const std::shared_ptr<KernelGraph> &kernel_graph) {
  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  std::string pass_name = "replace_node_by_proxy";
  pass_name.append(std::to_string(graph_sum_));
  pm->AddPass(std::make_shared<opt::ReplaceNodeByProxy>(pass_name));
  optimizer->AddPassManager(pm);
  (void)optimizer->Optimize(kernel_graph);
  kernel_graph->SetExecOrderByDefault();
}

GraphId CPUSession::CompileGraph(const AnfNodePtrList &lst, const AnfNodePtrList &outputs) {
  auto graph_id = graph_sum_;
  auto graph = ConstructKernelGraph(lst, outputs);
  MS_EXCEPTION_IF_NULL(graph);
  MS_LOG(INFO) << "Set kernel info";
  SetKernelInfo(graph.get());
#if (ENABLE_CPU && (ENABLE_D || ENABLE_GPU))
  AssignParamKey(graph);
  if (parallel::ps::Util::IsRoleOfWorker()) {
    Optimize(graph);
  }
#endif
  MS_LOG(INFO) << "Build kernel";
  BuildKernel(graph.get());
  MS_LOG(INFO) << "Assign kernel address";
  runtime_.AssignKernelAddress(graph.get());
  return graph_id;
}

void CPUSession::RunGraph(const GraphId &graph_id, const std::vector<tensor::TensorPtr> &inputs, VectorRef *outputs) {
  auto &kernel_graph = graphs_[graph_id];
  MS_EXCEPTION_IF_NULL(kernel_graph);
#if (ENABLE_CPU && (ENABLE_D || ENABLE_GPU))
  InitPSParamAndOptim(kernel_graph, inputs);
#endif
  MS_LOG(INFO) << "Bind input output address";
  std::vector<tensor::TensorPtr> need_sync_outputs;
  runtime_.BindInputOutput(kernel_graph.get(), inputs, outputs, &need_sync_outputs);
  MS_LOG(INFO) << "Run graph start";
  auto execution_order = kernel_graph->execution_order();
  Reorder(&execution_order);

  bool enable_summary = summary_callback_ != nullptr;
  kernel_graph->set_execution_order(execution_order);
  NamedSummaryOutputs summary_outputs;
  if (enable_summary) {
    SetSummaryNodes(kernel_graph.get());
    summary_outputs = kernel_graph->summary_nodes();
    runtime_.IncreaseSummaryRefCount(summary_outputs);
  }
#ifdef ENABLE_DEBUGGER
  // debugger pre-execution processing
  if (debugger_) {
    debugger_->PreExecute(kernel_graph);
  }
#endif
  bool ret = runtime_.Run(kernel_graph.get());
  if (!ret) {
    MS_LOG(EXCEPTION) << "Run graph failed";
  }
  for (auto output : need_sync_outputs) {
    (void)output->data_sync();
  }

  if (enable_summary) {
    Summary(kernel_graph.get());
    runtime_.DecreaseSummaryRefCount(summary_outputs);
  }

#ifdef ENABLE_DEBUGGER
  // debugger post-execution processing
  if (debugger_) {
    debugger_->PostExecute();
  }
#endif
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

namespace {
void KernelNotSupportException(const AnfNodePtr &kernel_node) {
  std::string kernel_name = AnfAlgo::GetCNodeName(kernel_node);
  std::stringstream operator_info;
  operator_info << "Operator[" << kernel_name << "] ";
  auto kernel_info = dynamic_cast<device::KernelInfo *>(kernel_node->kernel_info());
  if (kernel_info == nullptr) {
    operator_info << "is not support.";
    MS_LOG(EXCEPTION) << operator_info.str();
  }
  auto kernel_build_Info = kernel_info->select_kernel_build_info();
  if (kernel_build_Info == nullptr) {
    operator_info << "is not support.";
    MS_LOG(EXCEPTION) << operator_info.str();
  }
  size_t input_num = kernel_build_Info->GetInputNum();
  if (input_num > 0) {
    operator_info << " input(";
    for (size_t i = 0; i < input_num; ++i) {
      operator_info << TypeIdLabel(kernel_build_Info->GetInputDeviceType(i));
      if (i != input_num - 1) {
        operator_info << ",";
      }
    }
    operator_info << ") ";
  }
  size_t output_num = kernel_build_Info->GetOutputNum();
  if (output_num > 0) {
    operator_info << "output(";
    for (size_t i = 0; i < output_num; ++i) {
      operator_info << TypeIdLabel(kernel_build_Info->GetOutputDeviceType(i));
      if (i != kernel_build_Info->GetOutputNum() - 1) {
        operator_info << ",";
      }
    }
    operator_info << ") ";
  }
  operator_info << "is not support.";
  MS_LOG(EXCEPTION) << operator_info.str();
}
}  // namespace

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
      KernelNotSupportException(kernel_node);
    }
    cpu_kernel->Init(kernel_node);
    AnfAlgo::SetKernelMod(cpu_kernel, kernel_node.get());
    MS_LOG(INFO) << "Cpu build success operator[" << kernel_name << "].";
  }
}
}  // namespace session
}  // namespace mindspore
