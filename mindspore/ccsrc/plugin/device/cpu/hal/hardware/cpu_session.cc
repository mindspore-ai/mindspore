/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/hal/hardware/cpu_session.h"
#include <algorithm>
#include <sstream>
#include <exception>
#include "ir/anf.h"
#include "utils/ms_utils.h"
#include "utils/trace_base.h"
#include "backend/common/graph_kernel/graph_kernel_flags.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "plugin/factory/ms_factory.h"
#include "runtime/device/kernel_runtime.h"
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/device/cpu/optimizer/print_value_type.h"
#ifdef ENABLE_AKG
#include "plugin/device/cpu/kernel/akg/akg_cpu_kernel_build.h"
#endif
#include "plugin/device/cpu/hal/device/kernel_select_cpu.h"
#include "include/backend/optimizer/optimizer.h"
#include "include/backend/optimizer/pass_manager.h"
#include "plugin/device/cpu/optimizer/insert_cast_cpu.h"
#include "plugin/device/cpu/optimizer/insert_format_transform_op.h"
#include "backend/common/graph_kernel/adapter/graph_kernel_optimization.h"
#include "backend/common/pass/replace_node_by_proxy.h"
#include "backend/common/pass/erase_visit_attr.h"
#include "include/common/debug/anf_ir_dump.h"
#include "backend/common/optimizer/common_backend_optimization.h"
#include "include/common/debug/dump_proto.h"
#include "kernel/graph_kernel_info.h"
#ifndef ENABLE_SECURITY
#include "include/backend/debug/data_dump/dump_json_parser.h"
#endif
#if defined(__linux__) && defined(WITH_BACKEND)
#include "include/backend/distributed/ps/util.h"
#include "include/backend/distributed/ps/ps_context.h"
#endif
#ifdef ENABLE_DUMP_IR
#include "debug/rdr/graph_recorder.h"
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
  auto graph_inputs = graph->MutableInputs();
  MS_EXCEPTION_IF_NULL(valid_inputs);
  MS_EXCEPTION_IF_NULL(graph_inputs);
  TraceManager::DebugTrace(std::make_shared<TraceCopy>(anf->debug_info()));
  ParameterPtr new_parameter = graph->NewParameter(anf->cast<ParameterPtr>());
  TraceManager::EndTrace();
  graph_inputs->push_back(new_parameter);
  valid_inputs->push_back(true);
  return new_parameter;
}

// Remove after PS feature finish adapting push/pull in auto_monad.
void CPUSession::Reorder(std::vector<CNodePtr> *node_list) const {
  common::AnfAlgo::ReorderPosteriorExecList(NOT_NULL(node_list));
}

void CPUSession::Optimize(const std::shared_ptr<KernelGraph> &kernel_graph) {
  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
#if defined(__linux__) && defined(WITH_BACKEND)
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (ms_context->get_param<int>(MS_CTX_EXECUTION_MODE) != kPynativeMode && ps::PSContext::instance()->is_ps_mode()) {
    if (ps::PSContext::instance()->is_worker()) {
      std::string pass_name = "replace_node_by_proxy";
      pass_name.append(std::to_string(graph_sum_));
      pm->AddPass(std::make_shared<opt::ReplaceNodeByProxy>(pass_name));
    }
  }
#endif
  pm->AddPass(std::make_shared<opt::InsertFormatTransformOpCPU>("insert_format_transform_op_cpu"));
  pm->AddPass(std::make_shared<opt::InsertCastCPU>("insert_cast"));
  pm->AddPass(std::make_shared<opt::EraseVisitAttr>());
  pm->AddPass(std::make_shared<opt::PrintValueType>("print_value_type"));
  optimizer->AddPassManager(pm);
  (void)optimizer->Optimize(kernel_graph);
  kernel_graph->SetExecOrderByDefault();
}

void CPUSession::GraphKernelOptimize(const std::shared_ptr<KernelGraph> &kernel_graph) const {
  if (!graphkernel::GraphKernelFlags::GetInstance().IsEnableGraphKernel()) {
    return;
  }
  MS_EXCEPTION_IF_NULL(kernel_graph);
  graphkernel::GraphKernelOptimize(kernel_graph);
  kernel_graph->SetExecOrderByDefault();
}

GraphId CPUSession::CompileGraphImpl(const AnfNodePtrList &lst, const AnfNodePtrList &outputs) {
  auto graph_id = graph_sum_;
  auto graph = ConstructKernelGraph(lst, outputs, DeviceType::kCPU);
  MS_EXCEPTION_IF_NULL(graph);
  opt::AddDynamicShapeAttrPass(graph);
  MS_LOG(INFO) << "Set kernel info";
  SetKernelInfo(graph.get());
  MS_LOG(INFO) << "Set kernel info end";
  Optimize(graph);
  FinalOptimize(graph);
  GraphKernelOptimize(graph);
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
  MS_LOG(INFO) << "Assign kernel graph address";
  runtime_.AssignKernelGraphAddress(graph.get());
  // set summary node
#ifndef ENABLE_SECURITY
  SetSummaryNodes(graph.get());
#endif
  runtime_.IncreaseSummaryRefCount(graph->summary_nodes());
  DumpGraphs({graph});
  return graph_id;
}

void CPUSession::CreateOutputTensors(const GraphId &graph_id, const std::vector<tensor::TensorPtr> &input_tensors,
                                     VectorRef *outputs,
                                     std::map<tensor::TensorPtr, session::KernelWithIndex> *tensor_to_node,
                                     KernelMapTensor *) {
  auto kernel_graph = GetGraph(graph_id);
  MS_EXCEPTION_IF_NULL(kernel_graph);
  runtime_.CreateOutputTensors(kernel_graph.get(), input_tensors, outputs, tensor_to_node);
}

void CPUSession::LoadInputData(const std::shared_ptr<KernelGraph> &kernel_graph,
                               const std::vector<tensor::TensorPtr> &inputs_const) const {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto &input_nodes = kernel_graph->input_nodes();
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
    MS_EXCEPTION_IF_NULL(address);
    MS_EXCEPTION_IF_NULL(tensor);
    auto tensor_address = tensor->device_address();
    if (tensor_address == nullptr || tensor_address == address) {
      continue;
    }
    auto input_param = input_node->cast<ParameterPtr>();
    if (common::AnfAlgo::IsParameterWeight(input_param) && !tensor->IsUpdatedByDevice()) {
      continue;
    }
    if (std::dynamic_pointer_cast<device::DeviceAddress>(tensor_address)->GetDeviceType() != device::DeviceType::kCPU) {
      tensor->data_sync(false);
    }
  }
}

void CPUSession::PreExecuteGraph(const std::shared_ptr<KernelGraph> &kernel_graph,
                                 const std::vector<tensor::TensorPtr> &inputs, VectorRef *const outputs) {
  MS_LOG(INFO) << "Bind input output address";
  runtime_.BindInputOutput(kernel_graph.get(), inputs, outputs);
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

KernelGraphPtr CPUSession::BuildOpImpl(const BackendOpRunInfoPtr &op_run_info, const GraphInfo &graph_info,
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
      MS_EXCEPTION_IF_NULL(tensor_ptr);
      tensor_ptr->SetNeedWait(false);
      tensor_ptr->data_sync(false);
    }
  }
}

void CPUSession::UpdateDynamicOutputShape(const std::map<tensor::TensorPtr, KernelWithIndex> &tensor_to_node) const {
  for (const auto &tensor_node : tensor_to_node) {
    if (common::AnfAlgo::IsDynamicShape(tensor_node.second.first)) {
      const auto &kernel = tensor_node.second.first;
      const auto &output_index = tensor_node.second.second;
      const auto &shape = common::AnfAlgo::GetOutputInferShape(kernel, output_index);
      MS_EXCEPTION_IF_NULL(tensor_node.first);
      (void)tensor_node.first->set_shape(shape);
    }
  }
}

void CPUSession::RunOpImplOrigin(const GraphInfo &graph_info, const BackendOpRunInfoPtr &op_run_info,
                                 std::vector<tensor::TensorPtr> *input_tensors, VectorRef *outputs,
                                 const std::vector<int64_t> &tensors_mask) {
  RunOpImpl(graph_info, op_run_info, input_tensors, outputs, tensors_mask);
}

void CPUSession::RunOpImpl(const GraphInfo &graph_info, const BackendOpRunInfoPtr &op_run_info,
                           std::vector<tensor::TensorPtr> *input_tensors, VectorRef *outputs,
                           const std::vector<int64_t> &tensors_mask) {
  MS_EXCEPTION_IF_NULL(input_tensors);
  MS_EXCEPTION_IF_NULL(op_run_info);
  ProcessInputTensorsForHeterogeneous("CPU", *input_tensors);
  const auto &kernel_graph = BuildOpImpl(op_run_info, graph_info, *input_tensors, tensors_mask);
  MS_EXCEPTION_IF_NULL(kernel_graph);
  EraseValueNodeTensor(tensors_mask, input_tensors);

  // Remove reorder after PS feature finish adapting push/pull in auto_monad.
  auto execution_order = kernel_graph->execution_order();
  Reorder(&execution_order);
  kernel_graph->set_execution_order(execution_order);

  // runtime init
  if (!runtime_.Init()) {
    MS_LOG(EXCEPTION) << "Kernel runtime init error.";
  }
  runtime_.AssignKernelGraphAddress(kernel_graph.get());
  std::map<tensor::TensorPtr, session::KernelWithIndex> tensor_to_node;
  runtime_.CreateOutputTensors(kernel_graph.get(), *input_tensors, outputs, &tensor_to_node);
  runtime_.BindInputOutput(kernel_graph.get(), *input_tensors, outputs);

  bool ret = runtime_.Run(*kernel_graph, false);
  if (!ret) {
    MS_LOG(EXCEPTION) << "Run Op failed";
  }
  UpdateDynamicOutputShape(tensor_to_node);
  SetOutputFlags(*outputs);
  runtime_.RunOpClearMemory(*kernel_graph);
}

void CPUSession::SetKernelInfo(const KernelGraph *kernel_graph) const {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto &kernel_nodes = kernel_graph->execution_order();
  auto kernel_info_setter = GraphKernelInfoManager::Instance().GetGraphKernelInfo(kCPUDevice);
  MS_EXCEPTION_IF_NULL(kernel_info_setter);
  for (const auto &kernel_node : kernel_nodes) {
    MS_EXCEPTION_IF_NULL(kernel_node);
    kernel_info_setter->SetKernelInfo(kernel_node, KernelType::UNKNOWN_KERNEL_TYPE);
  }
}

namespace {
void KernelNotSupportException(const AnfNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  std::string kernel_name = common::AnfAlgo::GetCNodeName(kernel_node);
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
  MS_LOG(EXCEPTION) << operator_info.str() << trace::DumpSourceLines(kernel_node);
}
}  // namespace

void CPUSession::BuildKernel(const KernelGraph *kernel_graph) const {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto &kernel_nodes = kernel_graph->execution_order();
  kernel::KernelMeta *bin_map = kernel::KernelMeta::GetInstance();
  MS_EXCEPTION_IF_NULL(bin_map);
  std::vector<AnfNodePtr> akg_nodes;
  for (const auto &kernel_node : kernel_nodes) {
    MS_EXCEPTION_IF_NULL(kernel_node);
    std::string kernel_name = common::AnfAlgo::GetCNodeName(kernel_node);
    MS_LOG(INFO) << "Cpu building operator[" << kernel_name << "].";
    if (session::AnfRuntimeAlgorithm::GetKernelType(kernel_node) == KernelType::AKG_KERNEL) {
      if (!bin_map->initialized()) {
        bin_map->Initialize();
      }
      akg_nodes.push_back(kernel_node);
      continue;
    }
    std::shared_ptr<kernel::NativeCpuKernelMod> cpu_kernel_mod =
      kernel::Factory<kernel::NativeCpuKernelMod>::Instance().Create(kernel_name);
    if (cpu_kernel_mod == nullptr) {
      KernelNotSupportException(kernel_node);
    }

    // This branch would be removed When KernelMode rectification is complete
    auto discard_cpu_kernel_mod = std::dynamic_pointer_cast<kernel::DeprecatedNativeCpuKernelMod>(cpu_kernel_mod);
    auto args = kernel::AbstractArgsFromCNode(kernel_node, discard_cpu_kernel_mod != nullptr);
    // inputs_tensor_map is ops's valueDepend input. if this input is const_value tensor,
    // we will put this tensor in args.inputs.data_.
    auto inputs_tensor_map = std::map<uint32_t, tensor::TensorPtr>();
    kernel::SetInputsByConstInputs(kernel_node, &inputs_tensor_map);
    kernel::SetInputsByDependMap(inputs_tensor_map, &args.inputs, true);
    if (discard_cpu_kernel_mod != nullptr) {
      try {
        kernel::SetArgsToCNode(kernel_node, args);
        discard_cpu_kernel_mod->SetCpuRefMapToKernelInfo(kernel_node);
        discard_cpu_kernel_mod->Init(kernel_node);
      } catch (std::exception &e) {
        MS_LOG(EXCEPTION) << e.what() << trace::DumpSourceLines(kernel_node);
      }
      AnfAlgo::SetKernelMod(discard_cpu_kernel_mod, kernel_node.get());
      MS_LOG(INFO) << "Cpu build success operator[" << kernel_name << "].";
    } else {
      auto kernel_attrs = cpu_kernel_mod->GetOpSupport();
      SetCpuRefMapToKernelInfo(kernel_node, kernel_attrs);
      auto ret = cpu_kernel_mod->Init(args.op, args.inputs, args.outputs);
      if (!ret) {
        MS_LOG(EXCEPTION) << trace::DumpSourceLines(kernel_node);
      }
      if (cpu_kernel_mod->Resize(args.op, args.inputs, args.outputs, inputs_tensor_map) ==
          static_cast<int>(kernel::KRET_RESIZE_FAILED)) {
        MS_LOG(EXCEPTION) << "CPU kernel op [" << kernel_node->fullname_with_scope() << "] Resize failed.";
      }
      AnfAlgo::SetKernelMod(cpu_kernel_mod, kernel_node.get());
      MS_LOG(INFO) << "Cpu build success operator[" << kernel_name << "].";
    }
  }
#ifdef ENABLE_AKG
  kernel::AkgCpuKernelBuilder akg_cpu_kernel_builder;
  (void)akg_cpu_kernel_builder.AkgKernelParallelBuild(akg_nodes);
#endif
}
}  // namespace session
}  // namespace mindspore
