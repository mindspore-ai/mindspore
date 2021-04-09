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
#include <exception>
#include "ir/anf.h"
#include "utils/ms_utils.h"
#include "utils/trace_base.h"
#include "backend/session/anf_runtime_algorithm.h"
#include "runtime/device/kernel_runtime.h"
#include "backend/kernel_compiler/cpu/cpu_kernel_factory.h"
#include "runtime/device/cpu/kernel_select_cpu.h"
#include "backend/optimizer/common/optimizer.h"
#include "backend/optimizer/common/pass_manager.h"
#include "backend/optimizer/cpu/insert_cast_cpu.h"
#include "backend/optimizer/pass/replace_node_by_proxy.h"
#include "backend/optimizer/pass/erase_visit_attr.h"
#include "debug/anf_ir_dump.h"
#include "debug/dump_proto.h"
#include "debug/data_dump/dump_json_parser.h"
#if (ENABLE_CPU && (ENABLE_D || ENABLE_GPU))
#include "ps/util.h"
#include "ps/ps_context.h"
#endif

namespace mindspore {
namespace session {
void CPUSession::Init(uint32_t device_id) {
  // Dump json config file if dump is enabled
  DumpJsonParser::GetInstance().CopyJsonToDir();
  InitExecutor(kCPUDevice, device_id);
}

ParameterPtr CPUSession::CreateNewParameterFromParameter(const AnfNodePtr &anf, KernelGraph *graph) {
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
  valid_inputs->push_back(true);
  return new_parameter;
}

// Remove after PS feature finish adapting push/pull in auto_monad.
void CPUSession::Reorder(std::vector<CNodePtr> *node_list) { AnfAlgo::ReorderPosteriorExecList(NOT_NULL(node_list)); }

void CPUSession::Optimize(const std::shared_ptr<KernelGraph> &kernel_graph) {
  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
#if (ENABLE_CPU && (ENABLE_D || ENABLE_GPU))
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (ms_context->get_param<int>(MS_CTX_EXECUTION_MODE) != kPynativeMode && ps::PSContext::instance()->is_ps_mode()) {
    AssignParamKey(kernel_graph);
    if (ps::PSContext::instance()->is_worker()) {
      std::string pass_name = "replace_node_by_proxy";
      pass_name.append(std::to_string(graph_sum_));
      pm->AddPass(std::make_shared<opt::ReplaceNodeByProxy>(pass_name));
    }
  }
#endif
  pm->AddPass(std::make_shared<opt::InsertCastCPU>());
  pm->AddPass(std::make_shared<opt::EraseVisitAttr>());
  MS_LOG(INFO) << "insert cast pass";
  optimizer->AddPassManager(pm);
  (void)optimizer->Optimize(kernel_graph);
  kernel_graph->SetExecOrderByDefault();
}

GraphId CPUSession::CompileGraphImpl(const AnfNodePtrList &lst, const AnfNodePtrList &outputs) {
  auto graph_id = graph_sum_;
  auto graph = ConstructKernelGraph(lst, outputs);
  MS_EXCEPTION_IF_NULL(graph);
  UpdateGraphDynamicShapeAttr(NOT_NULL(graph));
  graph->UpdateGraphDynamicAttr();
  MS_LOG(INFO) << "Set kernel info";
  SetKernelInfo(graph.get());
  MS_LOG(INFO) << "Set kernel info end";
  Optimize(graph);
  MS_LOG(INFO) << "Build kernel";
  BuildKernel(graph.get());

  // Remove reorder after PS feature finish adapting push/pull in auto_monad.
  auto execution_order = graph->execution_order();
  Reorder(&execution_order);
  graph->set_execution_order(execution_order);

  // runtime init
  if (!runtime_.Init()) {
    MS_LOG(EXCEPTION) << "Kernel runtime init error.";
  }

  MS_LOG(INFO) << "Assign kernel address";
  runtime_.AssignKernelAddress(graph.get());

  DumpGraph(graph);
  return graph_id;
}

void CPUSession::CreateOutputTensors(const GraphId &graph_id, const std::vector<tensor::TensorPtr> &input_tensors,
                                     VectorRef *outputs,
                                     std::map<tensor::TensorPtr, session::KernelWithIndex> *tensor_to_node) {
  auto kernel_graph = GetGraph(graph_id);
  MS_EXCEPTION_IF_NULL(kernel_graph);
  runtime_.CreateOutputTensors(kernel_graph.get(), input_tensors, outputs, tensor_to_node);
}

void CPUSession::LoadInputData(const std::shared_ptr<KernelGraph> &kernel_graph,
                               const std::vector<tensor::TensorPtr> &inputs_const) const {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto &input_nodes = kernel_graph->inputs();
  if (input_nodes.size() != inputs_const.size()) {
    MS_LOG(EXCEPTION) << "Input size not equal to input node size!";
  }
  for (size_t input_idx = 0; input_idx < input_nodes.size(); ++input_idx) {
    auto &item = input_nodes[input_idx];
    MS_EXCEPTION_IF_NULL(item);
    if (item->isa<Parameter>() && !HasAbstractMonad(item)) {
      auto address = AnfAlgo::GetMutableOutputAddr(item, 0);
      auto tensor = inputs_const[input_idx];
      auto tensor_address = tensor->device_address();
      MS_EXCEPTION_IF_NULL(address);
      MS_EXCEPTION_IF_NULL(tensor);
      if (tensor_address != nullptr && tensor_address != address &&
          (std::dynamic_pointer_cast<device::DeviceAddress>(tensor_address)->DeviceType() !=
             device::DeviceAddressType::kCPU ||
           AnfAlgo::IsParameterWeight(item->cast<ParameterPtr>()))) {
        tensor->data_sync(false);
      }
    }
  }
}

void CPUSession::RunGraphImpl(const GraphId &graph_id, const std::vector<tensor::TensorPtr> &inputs,
                              VectorRef *outputs) {
  auto kernel_graph = GetGraph(graph_id);
  MS_EXCEPTION_IF_NULL(kernel_graph);
  MS_LOG(INFO) << "Bind input output address";
  runtime_.BindInputOutput(kernel_graph.get(), inputs, outputs);

#if (ENABLE_CPU && (ENABLE_D || ENABLE_GPU))
  InitPSParamAndOptim(kernel_graph, inputs);
#endif

  MS_LOG(INFO) << "Run graph start";

  bool enable_summary = summary_callback_ != nullptr;
  NamedSummaryOutputs summary_outputs;
  if (enable_summary) {
    SetSummaryNodes(kernel_graph.get());
    summary_outputs = kernel_graph->summary_nodes();
    runtime_.IncreaseSummaryRefCount(summary_outputs);
  }

  bool ret = runtime_.Run(kernel_graph.get(), false);
  if (!ret) {
    MS_LOG(EXCEPTION) << "Run graph failed";
  }

  if (enable_summary) {
    Summary(kernel_graph.get());
    runtime_.DecreaseSummaryRefCount(summary_outputs);
  }

  MS_LOG(INFO) << "Run graph end";
}

void CPUSession::BuildOpImpl(const OpRunInfo &op_run_info, const GraphInfo &graph_info,
                             const std::vector<tensor::TensorPtr> &input_tensors,
                             const std::vector<int64_t> &tensors_mask) {
  // Check if the graph cache exists.
  if (run_op_graphs_.find(graph_info) != run_op_graphs_.end()) {
    return;
  }
  // Prepare the graph
  auto kernel_graph = ConstructSingleOpGraph(op_run_info, input_tensors, tensors_mask);
  MS_EXCEPTION_IF_NULL(kernel_graph);
  SetKernelInfo(kernel_graph.get());
  Optimize(kernel_graph);
  BuildKernel(kernel_graph.get());
  run_op_graphs_[graph_info] = kernel_graph;
}

void CPUSession::SetOutputFlags(const VectorRef &base_ref, std::vector<tensor::TensorPtr> *outputs_tensors) {
  for (size_t i = 0; i < base_ref.size(); ++i) {
    if (utils::isa<VectorRef>(base_ref[i])) {
      auto ref_iter = utils::cast<VectorRef>(base_ref[i]);
      SetOutputFlags(ref_iter, outputs_tensors);
    } else if (utils::isa<tensor::TensorPtr>(base_ref[i])) {
      auto tensor_ptr = utils::cast<std::shared_ptr<tensor::Tensor>>(base_ref[i]);
      tensor_ptr->SetNeedWait(false);
      tensor_ptr->data_sync(false);
      outputs_tensors->push_back(tensor_ptr);
    }
  }
}

void CPUSession::RunOpImpl(const GraphInfo &graph_info, OpRunInfo *op_run_info,
                           std::vector<tensor::TensorPtr> *input_tensors, VectorRef *outputs,
                           const std::vector<int64_t> &tensors_mask) {
  MS_EXCEPTION_IF_NULL(input_tensors);
  MS_EXCEPTION_IF_NULL(op_run_info);
  BuildOpImpl(*op_run_info, graph_info, *input_tensors, tensors_mask);
  EraseValueNodeTensor(tensors_mask, input_tensors);

  auto kernel_graph = run_op_graphs_[graph_info];
  MS_EXCEPTION_IF_NULL(kernel_graph);

  // Remove reorder after PS feature finish adapting push/pull in auto_monad.
  auto execution_order = kernel_graph->execution_order();
  Reorder(&execution_order);
  kernel_graph->set_execution_order(execution_order);

  // runtime init
  if (!runtime_.Init()) {
    MS_LOG(EXCEPTION) << "Kernel runtime init error.";
  }
  runtime_.AssignKernelAddress(kernel_graph.get());
  std::map<tensor::TensorPtr, session::KernelWithIndex> tensor_to_node;
  runtime_.CreateOutputTensors(kernel_graph.get(), *input_tensors, outputs, &tensor_to_node);
  runtime_.BindInputOutput(kernel_graph.get(), *input_tensors, outputs);

  MS_LOG(INFO) << "Run Op start";

  bool ret = runtime_.Run(kernel_graph.get(), false);
  if (!ret) {
    MS_LOG(EXCEPTION) << "Run Op failed";
  }

  std::vector<tensor::TensorPtr> output_tensors;
  SetOutputFlags(*outputs, &output_tensors);
  runtime_.RunOpClearMemory(kernel_graph.get());
  MS_LOG(INFO) << "Run Op end";
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
  MS_LOG(EXCEPTION) << operator_info.str() << " Trace: " << trace::DumpSourceLines(kernel_node);
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
    try {
      cpu_kernel->Init(kernel_node);
    } catch (std::exception &e) {
      MS_LOG(EXCEPTION) << e.what() << "\nTrace: " << trace::DumpSourceLines(kernel_node);
    }
    AnfAlgo::SetKernelMod(cpu_kernel, kernel_node.get());
    MS_LOG(INFO) << "Cpu build success operator[" << kernel_name << "].";
  }
}
}  // namespace session
}  // namespace mindspore
