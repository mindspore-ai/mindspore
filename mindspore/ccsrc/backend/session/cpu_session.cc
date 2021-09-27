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
#include "backend/optimizer/cpu/insert_format_transform_op.h"
#include "backend/optimizer/pass/replace_node_by_proxy.h"
#include "backend/optimizer/pass/erase_visit_attr.h"
#include "debug/anf_ir_dump.h"
#include "debug/dump_proto.h"
#ifndef ENABLE_SECURITY
#include "debug/data_dump/dump_json_parser.h"
#endif
#if ((defined ENABLE_CPU) && (!defined _WIN32))
#include "ps/util.h"
#include "ps/ps_context.h"
#endif
#ifdef ENABLE_DUMP_IR
#include "debug/rdr/graph_recorder.h"
#include "debug/rdr/running_data_recorder.h"
#endif

namespace mindspore {
namespace session {
void CPUSession::Init(uint32_t device_id) {
#ifndef ENABLE_SECURITY
  // Dump json config file if dump is enabled
  auto &json_parser = DumpJsonParser::GetInstance();
  json_parser.Parse();
  json_parser.CopyMSCfgJsonToDir(rank_id_);
#endif
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
#if ((defined ENABLE_CPU) && (!defined _WIN32))
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
  pm->AddPass(std::make_shared<opt::InsertFormatTransformOpCPU>("insert_format_transform_op_cpu"));
  optimizer->AddPassManager(pm);
  (void)optimizer->Optimize(kernel_graph);
  kernel_graph->SetExecOrderByDefault();
}

void CPUSession::ProcessCast(const std::shared_ptr<KernelGraph> &kernel_graph) {
  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  MS_EXCEPTION_IF_NULL(pm);
  pm->AddPass(std::make_shared<opt::InsertCastCPU>("insert_cast_cpu"));
  MS_LOG(INFO) << "Insert cast pass";
  pm->AddPass(std::make_shared<opt::EraseVisitAttr>());
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
  FinalOptimize(graph);
  MS_LOG(INFO) << "Build kernel";
  BuildKernel(graph.get());
  ProcessCast(graph);
  // Remove reorder after PS feature finish adapting push/pull in auto_monad.
  auto execution_order = graph->execution_order();
  Reorder(&execution_order);
  graph->set_execution_order(execution_order);

#ifdef ENABLE_DUMP_IR
  std::string name = "graph_build." + std::to_string(graph->graph_id());
  DumpGraphParams dump_params = {true, static_cast<int>(kWholeStack)};
  (void)mindspore::RDR::RecordAnfGraph(SubModuleId::SM_SESSION, name, graph, dump_params, ".ir");

  const std::vector<CNodePtr> &exec_order = graph->execution_order();
  std::string exec_order_name = "graph_exec_order." + std::to_string(graph->graph_id());
  (void)mindspore::RDR::RecordGraphExecOrder(SubModuleId::SM_SESSION, exec_order_name, exec_order);
#endif

  // runtime init
  if (!runtime_.Init()) {
    MS_LOG(EXCEPTION) << "Kernel runtime init error.";
  }
  MS_LOG(INFO) << "Assign kernel address";
  runtime_.AssignKernelAddress(graph.get());
  // set summary node
#ifndef ENABLE_SECURITY
  SetSummaryNodes(graph.get());
#endif
  runtime_.IncreaseSummaryRefCount(graph->summary_nodes());
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
    MS_LOG(EXCEPTION) << "Input size " << inputs_const.size() << " is not equal to input node size "
                      << input_nodes.size();
  }
  for (size_t input_idx = 0; input_idx < input_nodes.size(); ++input_idx) {
    auto &input_node = input_nodes[input_idx];
    MS_EXCEPTION_IF_NULL(input_node);
    if (!input_node->isa<Parameter>() || HasAbstractMonad(input_node)) {
      continue;
    }
    auto address = AnfAlgo::GetMutableOutputAddr(input_node, 0);
    auto tensor = inputs_const[input_idx];
    auto tensor_address = tensor->device_address();
    MS_EXCEPTION_IF_NULL(address);
    MS_EXCEPTION_IF_NULL(tensor);
    if (tensor_address == nullptr || tensor_address == address) {
      continue;
    }
    auto input_param = input_node->cast<ParameterPtr>();
    if (AnfAlgo::IsParameterWeight(input_param) && !tensor->IsUpdatedByDevice()) {
      continue;
    }
    if (std::dynamic_pointer_cast<device::DeviceAddress>(tensor_address)->DeviceType() !=
        device::DeviceAddressType::kCPU) {
      tensor->data_sync(false);
    }
  }
}

void CPUSession::PreExecuteGraph(const std::shared_ptr<KernelGraph> &kernel_graph,
                                 const std::vector<tensor::TensorPtr> &inputs, VectorRef *const outputs) {
  MS_LOG(INFO) << "Bind input output address";
  runtime_.BindInputOutput(kernel_graph.get(), inputs, outputs);

#if ((defined ENABLE_CPU) && (!defined _WIN32))
  InitPSParamAndOptim(kernel_graph, inputs);
#endif
}

void CPUSession::PostExecuteGraph(const std::shared_ptr<KernelGraph> &kernel_graph,
                                  const std::vector<tensor::TensorPtr> &, VectorRef *const) {
#ifndef ENABLE_SECURITY
  Summary(kernel_graph.get());
#endif
}

void CPUSession::ExecuteGraph(const std::shared_ptr<KernelGraph> &kernel_graph) {
  bool ret = runtime_.Run(*kernel_graph, false);
  if (!ret) {
    MS_LOG(EXCEPTION) << "Run graph failed";
  }
}

KernelGraphPtr CPUSession::BuildOpImpl(const OpRunInfo &op_run_info, const GraphInfo &graph_info,
                                       const std::vector<tensor::TensorPtr> &input_tensors,
                                       const std::vector<int64_t> &tensors_mask) {
  // Check if the graph cache exists.
  auto it = run_op_graphs_.find(graph_info);
  if (it != run_op_graphs_.end()) {
    return it->second;
  }

  // Prepare the graph
  const auto &kernel_graph = ConstructSingleOpGraph(op_run_info, input_tensors, tensors_mask);
  MS_EXCEPTION_IF_NULL(kernel_graph);
  SetKernelInfo(kernel_graph.get());
  Optimize(kernel_graph);
  BuildKernel(kernel_graph.get());
  ProcessCast(kernel_graph);
  auto enable_op_graph_cache = MsContext::GetInstance()->get_param<bool>(MS_CTX_ENABLE_PYNATIVE_OP_GRAPH_CACHE);
  if (enable_op_graph_cache) {
    run_op_graphs_[graph_info] = kernel_graph;
  }
  return kernel_graph;
}

void CPUSession::SetOutputFlags(const VectorRef &base_ref) {
  for (size_t i = 0; i < base_ref.size(); ++i) {
    if (utils::isa<VectorRef>(base_ref[i])) {
      auto ref_iter = utils::cast<VectorRef>(base_ref[i]);
      SetOutputFlags(ref_iter);
    } else if (utils::isa<tensor::TensorPtr>(base_ref[i])) {
      auto tensor_ptr = utils::cast<std::shared_ptr<tensor::Tensor>>(base_ref[i]);
      tensor_ptr->SetNeedWait(false);
      tensor_ptr->data_sync(false);
    }
  }
}

void CPUSession::UpdateDynamicOutputShape(const std::map<tensor::TensorPtr, KernelWithIndex> &tensor_to_node) {
  for (const auto &tensor_node : tensor_to_node) {
    if (AnfAlgo::IsDynamicShape(tensor_node.second.first)) {
      const auto &kernel = tensor_node.second.first;
      const auto &output_index = tensor_node.second.second;
      const auto &shape = AnfAlgo::GetOutputInferShape(kernel, output_index);
      std::vector<int64_t> refresh_shape;
      (void)std::copy(shape.begin(), shape.end(), std::back_inserter(refresh_shape));
      MS_EXCEPTION_IF_NULL(tensor_node.first);
      tensor_node.first->set_shape(refresh_shape);
    }
  }
}

void CPUSession::RunOpImplOrigin(const GraphInfo &graph_info, OpRunInfo *op_run_info,
                                 std::vector<tensor::TensorPtr> *input_tensors, VectorRef *outputs,
                                 const std::vector<int64_t> &tensors_mask) {
  RunOpImpl(graph_info, op_run_info, input_tensors, outputs, tensors_mask);
}

void CPUSession::RunOpImpl(const GraphInfo &graph_info, OpRunInfo *op_run_info,
                           std::vector<tensor::TensorPtr> *input_tensors, VectorRef *outputs,
                           const std::vector<int64_t> &tensors_mask) {
  MS_EXCEPTION_IF_NULL(input_tensors);
  MS_EXCEPTION_IF_NULL(op_run_info);
  const auto &kernel_graph = BuildOpImpl(*op_run_info, graph_info, *input_tensors, tensors_mask);
  EraseValueNodeTensor(tensors_mask, input_tensors);
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

  bool ret = runtime_.Run(*kernel_graph, false);
  if (!ret) {
    MS_LOG(EXCEPTION) << "Run Op failed";
  }
  UpdateDynamicOutputShape(tensor_to_node);
  // update output abstract of dynamic op to op_run_info
  if (op_run_info->is_dynamic_shape) {
    UpdateOutputAbstract(kernel_graph, op_run_info);
  }
  SetOutputFlags(*outputs);
  runtime_.RunOpClearMemory(*kernel_graph);
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
