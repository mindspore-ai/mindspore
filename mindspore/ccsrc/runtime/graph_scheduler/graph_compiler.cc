/**
 * Copyright 2021-2024 Huawei Technologies Co., Ltd
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

#include "runtime/graph_scheduler/graph_compiler.h"
#include <numeric>
#include <map>
#include <utility>
#include <algorithm>
#include <functional>
#include <list>
#include "runtime/graph_scheduler/graph_scheduler.h"
#include "runtime/device/device_address_utils.h"
#include "runtime/pynative/op_executor.h"
#include "include/backend/device_address.h"
#include "runtime/device/ms_device_shape_transfer.h"
#include "runtime/pynative/op_runtime_info.h"
#include "runtime/pynative/op_compiler.h"
#include "include/common/utils/convert_utils.h"
#include "backend/common/graph_kernel/graph_kernel_flags.h"
#include "backend/common/optimizer/common_backend_optimization.h"
#include "utils/ms_context.h"
#include "ir/tensor.h"
#include "kernel/framework_utils.h"
#include "include/backend/debug/profiler/profiling.h"
#include "include/backend/optimizer/helper.h"
#include "base/base_ref_utils.h"
#include "include/common/debug/dump_proto.h"
#include "include/common/utils/parallel_context.h"
#include "plugin/device/cpu/hal/hardware/cpu_device_context.h"
#ifdef ENABLE_DEBUGGER
#include "include/backend/debug/debugger/debugger.h"
#endif
#ifdef ENABLE_DUMP_IR
#include "include/common/debug/anf_ir_dump.h"
#endif
#ifndef ENABLE_SECURITY
#include "include/backend/debug/data_dump/dump_json_parser.h"
#include "include/backend/optimizer/graph_optimizer.h"
#endif
#if defined(__linux__) && defined(WITH_BACKEND)
#include "include/backend/distributed/ps/ps_context.h"
#include "runtime/graph_scheduler/embedding_cache_scheduler.h"
#endif
#include "include/common/profiler.h"
#include "include/common/utils/compile_cache_context.h"
#include "utils/phase.h"
#include "pipeline/jit/ps/base.h"
#include "ops/framework_ops.h"

namespace mindspore {
namespace runtime {
namespace {
void SetSummaryNodesRefCount(const KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(graph);
  if (!graph->summary_node_exist()) {
    return;
  }

  const std::map<std::string, std::pair<AnfNodePtr, int>> &summary_nodes = graph->summary_nodes();
  if (summary_nodes.empty()) {
    return;
  }

  for (const auto &item : summary_nodes) {
    const AnfNodePtr &node = item.second.first;
    size_t index = IntToSize(item.second.second);
    auto device_address = AnfAlgo::GetMutableOutputAddr(node, index, false);
    MS_EXCEPTION_IF_NULL(device_address);
    device_address->set_original_ref_count(SIZE_MAX);
    device_address->ResetRefCount();
  }
}

bool EnableBackendCompileCache(const FuncGraphPtr &func_graph, const device::DeviceType &device_type) {
  auto &context = CompileCacheContext::GetInstance();
  if (!context.enable_compile_cache()) {
    return false;
  }
  if (context.FrontGraph() != func_graph) {
    return false;
  }
  if (context.RestrictedScenarios()) {
    return false;
  }
  if (MsContext::GetInstance()->backend_policy() == "ge") {
    return false;
  }
  if (device_type != device::DeviceType::kAscend) {
    return false;
  }
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (ms_context->CellReuseLevel() != CellReuseLevel::kNoCellReuse) {
    return false;
  }
  return true;
}

bool UseCacheToCompileGraph(const FuncGraphPtr &func_graph, const device::DeviceType &device_type) {
  if (!EnableBackendCompileCache(func_graph, device_type)) {
    return false;
  }
  auto &context = CompileCacheContext::GetInstance();
  if (!context.UseCompileCache()) {
    return false;
  }
  return true;
}

bool ExportCompileCache(const FuncGraphPtr &func_graph, const device::DeviceType &device_type) {
  if (!EnableBackendCompileCache(func_graph, device_type)) {
    return false;
  }
  auto &context = CompileCacheContext::GetInstance();
  if (context.UseCompileCache()) {
    return false;
  }
  return true;
}

// Fetch the real input of the nop node recursively.
AnfNodePtr FetchRealNodeByNopNode(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if ((!node->isa<CNode>()) || (!common::AnfAlgo::IsNopNode(node))) {
    return node;
  }

  const auto &cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);

  const auto &inputs = cnode->inputs();
  if (inputs.size() <= 1) {
    MS_LOG_WITH_NODE(INTERNAL_EXCEPTION, cnode)
      << "#dmsg#Runtime error info:#dmsg#Invalid cnode:" << cnode->DebugString();
  }
  return FetchRealNodeByNopNode(inputs[1]);
}

void OptimizeNopNode(KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(graph);
  std::vector<CNodePtr> nop_nodes_need_set_ref;

  // Skip the graph mode.
  if (graph->is_graph_run_mode()) {
    return;
  }

  const auto &output_node = graph->output();
  const auto &ref_map = graph->GetRefMap();
  std::set<std::pair<AnfNodePtr, size_t>> ref_out_value;
  for (const auto &iter : ref_map) {
    ref_out_value.insert(iter.second);
  }
  MS_EXCEPTION_IF_NULL(output_node);
  const auto &graph_outputs = common::AnfAlgo::GetAllOutputWithIndex(output_node);
  // Collect all the nopnodes that can be eliminated.
  for (const auto &cnode : graph->execution_order()) {
    MS_EXCEPTION_IF_NULL(cnode);
    if ((!common::AnfAlgo::IsNopNode(cnode)) || ref_map.count({cnode, 0}) != 0 ||
        ref_out_value.count({cnode, 0}) != 0 ||
        std::find_if(graph_outputs.begin(), graph_outputs.end(),
                     [&cnode](const KernelWithIndex &output) {
                       const auto &real_output = common::AnfAlgo::FetchRealNodeSkipMonadControl(output);
                       return real_output == KernelWithIndex(cnode, 0);
                     }) != graph_outputs.end() ||
        std::find_if(cnode->inputs().begin(), cnode->inputs().end(), [](const auto &input) {
          return common::AnfAlgo::CheckPrimitiveType(input, prim::kPrimConditionGather) ||
                 common::AnfAlgo::CheckPrimitiveType(common::AnfAlgo::VisitKernelWithReturnType(input, 0).first,
                                                     prim::kPrimConditionGather);
        }) != cnode->inputs().end()) {
      continue;
    }
    // NopNode that does not meet the above conditions is set to Ref Node and is not deleted from the graph to avoid
    // incorrect shape information of KernelTensor obtained in KernelMod::Launch.
    (void)nop_nodes_need_set_ref.emplace_back(cnode);
  }

  // Add the ref node pairs, which must be after elimination to avoid using elimination nodes.
  for (auto &ref_node : nop_nodes_need_set_ref) {
    MS_EXCEPTION_IF_NULL(ref_node);
    auto input_node = common::AnfAlgo::GetInputNode(ref_node, 0);
    MS_EXCEPTION_IF_NULL(input_node);
    // Record the original information of ref node.
    auto origin_pair = common::AnfAlgo::VisitKernelWithReturnType(input_node, 0, false);
    MS_EXCEPTION_IF_NULL(origin_pair.first);
    // The device address of parameter as input may be not the running used in the heterogeneous or control flow
    // scenarios, and not set the ref node.
    if (origin_pair.first->isa<Parameter>() || origin_pair.first->isa<ValueNode>()) {
      continue;
    }
    // The ref node cannot be set for node pairs from different device target(appears in the kernel backoff scene).
    if (AnfAlgo::FetchDeviceTarget(origin_pair.first, graph) != AnfAlgo::FetchDeviceTarget(ref_node, graph)) {
      continue;
    }
    MS_LOG(INFO) << "The reference relation of nopnode " << ref_node->fullname_with_scope() << ", index: " << 0
                 << " to input " << origin_pair.first->fullname_with_scope() << ", index: " << origin_pair.second;
    graph->AddRefCorrespondPairs(std::make_pair(ref_node, 0), origin_pair);
  }
}

bool IsEnableZeroCopy(bool run_in_pynative) {
  if (run_in_pynative) {
    return false;
  }

  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  bool task_sink = ms_context->get_param<bool>(MS_CTX_ENABLE_TASK_SINK);
  bool is_multi_graph_sink = ms_context->get_param<bool>(MS_CTX_IS_MULTI_GRAPH_SINK);
  // If the run mode is not subgraph sink, the flag should not be set.
  if (!task_sink || is_multi_graph_sink) {
    return false;
  }

// In ps cache mode, the whole graph sink has set multi_graph_sink to false, the zero copy cannot be enabled.
#if defined(__linux__) && defined(WITH_BACKEND)
  if (ps::PSContext::instance()->cache_enable()) {
    return false;
  }
#endif

  auto parallel_context = parallel::ParallelContext::GetInstance();
  MS_EXCEPTION_IF_NULL(parallel_context);
  auto parallel_mode = parallel_context->parallel_mode();
  bool is_parallel_mode = parallel_mode == parallel::kSemiAutoParallel || parallel_mode == parallel::kAutoParallel ||
                          parallel_mode == parallel::kHybridParallel || parallel_mode == parallel::kDataParallel;
  // If there are auto parallel in graph, the flag should not be set. In parallel, the continue memory in communication
  // ops not support addr change.
  // force zero copy when use ge
  bool is_enable_ge = ms_context->backend_policy() == "ge";
  if (is_parallel_mode && !is_enable_ge) {
    return false;
  }
  return true;
}

void SetRunGraphBySingleOpFlag(const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  for (auto &node : graph->execution_order()) {
    MS_EXCEPTION_IF_NULL(node);
    MS_EXCEPTION_IF_NULL(node->input(0));
    bool enable = false;
    if (!AnfAlgo::NodeValueIsFuncGraph(node->input(0))) {
      if (!kernel::CheckResizeCondition(node) && graph->has_flag(kFlagPyNativeRunInGraph)) {
        MS_LOG(INFO) << "Enable Run Graph By Single Op";
        enable = true;
      }
    }
    // BpGraph contain bprop_cut node.
    auto contain_bprop_cut = common::AnfAlgo::IsBpropCutOpExecInBackend(node);
    if (enable || contain_bprop_cut) {
      MS_LOG(INFO) << "Set kFlagEnableRunGraphBySingleOp: NeedSkipResize:" << enable
                   << ", BpGraph contain bprop_cut node:" << contain_bprop_cut;
      graph->set_flag(kFlagEnableRunGraphBySingleOp, true);
      break;
    }
  }
}

void UseCacheToCompileGraphImpl(const KernelGraphPtr &graph, const DeviceContext *device_context) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(device_context);

  auto &compile_cache_context = CompileCacheContext::GetInstance();
  (void)profiler::CollectHostInfo(kModelNameRuntime, kEventCompileGraph, kStageCreateKernel, 1, 0, 0);
  compile_cache_context.SetFusionOpBuildInfoFlag(true);
  device_context->GetKernelExecutor(false)->CreateKernel(graph->execution_order());
  compile_cache_context.SetFusionOpBuildInfoFlag(false);
  (void)profiler::CollectHostInfo(kModelNameRuntime, kEventCompileGraph, kStageCreateKernel, 1, 0, 1);
  // Kernels that are not supported by other device can be backed off and rebuilt on the CPU.
#ifdef WITH_BACKEND
  if (!graph->is_from_single_op()) {
    auto cpu_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(
      {kCPUDevice, device_context->device_context_key().device_id_});
    MS_EXCEPTION_IF_NULL(cpu_context);
    auto cpu_executor = dynamic_cast<device::cpu::CPUKernelExecutor *>(cpu_context->GetKernelExecutor(false).get());
    MS_EXCEPTION_IF_NULL(cpu_executor);
    cpu_executor->RebuildKernelSelectBackoffOp(graph->execution_order());
  }
#endif
#ifndef ENABLE_SECURITY
  // Update needed dump kernels for mindRT.
  DumpJsonParser::GetInstance().UpdateNeedDumpKernels(*graph.get());
#endif
  if (graph->is_dynamic_shape()) {
    auto profiler_manage_inst = profiler::ProfilerManager::GetInstance();
    MS_EXCEPTION_IF_NULL(profiler_manage_inst);
    profiler_manage_inst->SetNetDynamicShapeStatus();
  }
}

bool IsValidSequence(const ValueSequencePtr &sequence_value) {
  MS_EXCEPTION_IF_NULL(sequence_value);
  const auto &values = sequence_value->value();
  if (values.empty()) {
    return true;
  }
  MS_EXCEPTION_IF_NULL(values[0]);
  if (values[0]->isa<ValueSequence>()) {
    return false;
  }
  if (values[0]->type() == nullptr) {
    MS_LOG(DEBUG) << "Failed to get type from value tuple:" << sequence_value->ToString();
    return false;
  }
  TypeId base_type = values[0]->type()->type_id();
  for (size_t i = 1; i < values.size(); ++i) {
    MS_EXCEPTION_IF_NULL(values[i]);
    MS_EXCEPTION_IF_NULL(values[i]->type());
    TypeId type = values[i]->type()->type_id();
    if (type != base_type) {
      MS_LOG(DEBUG) << "Invalid value type for value:" << sequence_value->ToString();
      return false;
    }
  }
  return true;
}

void CollectValueNodeForKernelGraph(const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  graph->ClearAllValueNode();
  const auto &nodes = TopoSort(graph->get_return());
  for (const auto &node : nodes) {
    MS_EXCEPTION_IF_NULL(node);
    if (!node->isa<ValueNode>() || node->kernel_info() == nullptr) {
      continue;
    }
    const auto &value_node = node->cast<ValueNodePtr>();
    MS_EXCEPTION_IF_NULL(value_node);
    const auto &value = value_node->value();
    MS_EXCEPTION_IF_NULL(value);
    if (value->isa<Primitive>() ||
        (value->isa<ValueSequence>() && (!IsValidSequence(value->cast<ValueSequencePtr>())))) {
      continue;
    }
    MS_LOG(DEBUG) << "Add value node:" << node->DebugString() << " for kernel graph:" << graph->ToString();
    graph->AddValueNodeToGraph(value_node);
  }
}

GraphId CompileAnyTypeInputGraph(const KernelGraphPtr &graph, const AnfNodePtrList &outputs,
                                 const DeviceContext *device_context) {
  MS_EXCEPTION_IF_NULL(graph);
  for (const auto &input : graph->inputs()) {
    MS_EXCEPTION_IF_NULL(input);
    MS_LOG(DEBUG) << "input node:" << input->DebugString()
                  << " abstract:" << (input->abstract() == nullptr ? "null" : input->abstract()->ToString());
  }
  MS_LOG(DEBUG) << "Pre construct any type input kernel graph:" << graph->ToString();
  graph->set_is_any_type_input(true);
  opt::OptimizationForAnyTypeKernelGraph(graph);
  graph->SetInputNodes();
  for (const auto &input : graph->input_nodes()) {
    MS_EXCEPTION_IF_NULL(input);
    MS_LOG(DEBUG) << "input node:" << input->DebugString()
                  << " abstract:" << (input->abstract() == nullptr ? "null" : input->abstract()->ToString());
    if (!input->isa<Parameter>()) {
      continue;
    }
    const auto &parameter = input->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(parameter);
    const auto &shape = parameter->Shape();
    if (shape != nullptr &&
        ((shape->isa<abstract::Shape>() && shape->IsDynamic()) || shape->isa<abstract::DynamicSequenceShape>())) {
      parameter->set_has_dynamic_shape(true);
    }
  }
  auto backend_output = graph->output();
  MS_EXCEPTION_IF_NULL(backend_output);
  graph->CacheGraphOutputToFrontNodeWithIndex({backend_output}, outputs);
  graph->UpdateInternalParameter();
  DeviceAddressUtils::CreateParameterDeviceAddress(device_context, graph);

  auto output_with_indexs = common::AnfAlgo::GetAllOutputWithIndex(graph->output());
  for (const auto &output_with_index : output_with_indexs) {
    const auto &output = output_with_index.first;
    MS_EXCEPTION_IF_NULL(output);
    if (common::AnfAlgo::IsBpropCutOpExecInBackend(output) || HasAbstractMonad(output)) {
      continue;
    }
    if (output->kernel_info() == nullptr) {
      output->set_kernel_info(std::make_shared<device::KernelInfo>());
    }
    auto kernel_info = dynamic_cast<device::KernelInfo *>(output->kernel_info());
    MS_EXCEPTION_IF_NULL(kernel_info);
    // select_kernel_build_info() has checked whether return pointer is null
    auto build_info = kernel_info->select_kernel_build_info();
    if (build_info != nullptr) {
      continue;
    }
    size_t output_num = 1;
    if (output->abstract() != nullptr) {
      output_num = common::AnfAlgo::GetOutputNumByAbstract(output->abstract());
    }
    kernel::KernelBuildInfo::KernelBuildInfoBuilder builder;
    builder.SetOutputsFormat(std::vector<std::string>(output_num, kOpFormat_DEFAULT));
    builder.SetOutputsDeviceType(std::vector<TypeId>(output_num, kTypeUnknown));
    builder.SetOutputsKernelObjectType(
      std::vector<kernel::KernelObjectType>(output_num, kernel::KernelObjectType::TENSOR));
    AnfAlgo::SetSelectKernelBuildInfo(builder.Build(), output.get());
    MS_LOG(DEBUG) << "Set kernel build info for node:" << output->DebugString() << " output num:" << output_num;
  }
  CollectValueNodeForKernelGraph(graph);
  DeviceAddressUtils::CreateValueNodeDeviceAddress(device_context, graph);
  DeviceAddressUtils::CreateGraphOutputDeviceAddress(device_context, graph);
  return graph->graph_id();
}
}  // namespace

GraphId GraphCompiler::CompileGraph(const GraphSegmentPtr &segment,
                                    const std::pair<AnfNodePtrList, AnfNodePtrList> &io_nodes,
                                    const DeviceContext *device_context, device::RunMode run_mode,
                                    bool run_in_pynative) {
  MS_EXCEPTION_IF_NULL(segment);
  MS_EXCEPTION_IF_NULL(device_context);
  MS_LOG(INFO) << "Status record: start compile graph.";
  auto nodes = segment->nodes_;
  auto device_target = device_context->GetDeviceType();
  // Generate kernel graph.
  (void)profiler::CollectHostInfo(kModelNameRuntime, kEventCompileGraph, kStageConstructKernelGraph, 1, 0, 0);
  auto kernel_graph =
    session_->ConstructKernelGraph(nodes, io_nodes.second, device_target, true, IsEnableZeroCopy(run_in_pynative));
  (void)profiler::CollectHostInfo(kModelNameRuntime, kEventCompileGraph, kStageConstructKernelGraph, 1, 0, 1);
  SetGraphDependency(kernel_graph, segment);
  return CompileGraph(kernel_graph, io_nodes, device_context, run_mode, run_in_pynative);
}

GraphId GraphCompiler::CompileGraph(const KernelGraphPtr &kernel_graph,
                                    const std::pair<AnfNodePtrList, AnfNodePtrList> &io_nodes,
                                    const DeviceContext *device_context, device::RunMode run_mode,
                                    bool run_in_pynative) {
  MS_EXCEPTION_IF_NULL(session_);
  MS_EXCEPTION_IF_NULL(device_context);
  MS_EXCEPTION_IF_NULL(kernel_graph);

  const auto &outputs = io_nodes.second;
  if (common::AnfAlgo::IsAnyTypeInput(io_nodes.first)) {
    return CompileAnyTypeInputGraph(kernel_graph, outputs, device_context);
  }
  kernel_graph->erase_flag(kFlagPyNativeRunInGraph);
  SetRunGraphBySingleOpFlag(kernel_graph);
  kernel_graph->UpdateGraphAquireGilAttr();
  if (run_mode == device::RunMode::kUnknown) {
    kernel_graph->set_run_mode(device_context->GetRunMode(kernel_graph));
  } else {
    kernel_graph->set_run_mode(run_mode);
  }
  auto manager = MakeManager({kernel_graph});
  if (manager) {
    manager->AddFuncGraph(kernel_graph);
    kernel_graph->set_manager(manager);
  }

  opt::OptimizationWithoutBackend(kernel_graph);
  // Unify the MindIR, must be before of the kernel_graph optimization.
  auto kernel_executor = device_context->GetKernelExecutor(false);
  if (kernel_executor != nullptr) {
    kernel_executor->AddMindIRPass(kernel_graph);
  }
  kernel_graph->SetInputNodes();
  auto context_ptr = MsContext::GetInstance();
  session_->SetInputNodeUsage(kernel_graph, manager);
  MS_EXCEPTION_IF_NULL(context_ptr);
  if (context_ptr->backend_policy() == "ge" && device_context->GetDeviceType() == device::DeviceType::kAscend &&
      !IsEnableRefMode()) {
    MS_EXCEPTION_IF_NULL(device_context->graph_executor_);
    if (!device_context->graph_executor_->CompileGraph(kernel_graph, {})) {
      MS_LOG(EXCEPTION) << "Compile kernel_graph failed: " << kernel_graph->graph_id();
    }
    kernel_graph->UpdateInternalParameter();
    kernel_graph->CacheGraphOutputToFrontNodeWithIndex({kernel_graph->output()}, outputs);
    kernel_graph->set_front_outputs(outputs);
    return kernel_graph->graph_id();
  }
  kernel_graph->SetOptimizerFlag();

  GraphId graph_id = 0;
  if (run_in_pynative) {
    MS_EXCEPTION_IF_NULL(session_);
    // kernel_graph kernel does not support pynative mode now, print a warning here.
    graphkernel::GraphKernelFlags::GetInstance().CheckSupport();
    graph_id = kernel_graph->graph_id();
  } else {
    graph_id = CompileGraphImpl(kernel_graph, device_context, run_in_pynative);
  }

  kernel_graph->set_front_outputs(outputs);

  kernel_graph->set_root_graph_id(graph_id);
  session_->DumpGraphs({kernel_graph});

  // The kernel_graph is not compiled yet in PyNative Mode.
  // Need to cache output latter when the kernel_graph is compiled.
  if (!run_in_pynative) {
    // Cache the backend kernel_graph output nodes to front nodes with output index.
    auto backend_node = kernel_graph->output();
    MS_EXCEPTION_IF_NULL(backend_node);
    kernel_graph->CacheGraphOutputToFrontNodeWithIndex({backend_node}, outputs);
  }
  AnfAlgo::UpdateGraphValidRefPair(kernel_graph);

  MS_LOG(INFO) << "Status record: end compile graph. graph id: " << graph_id;
  return graph_id;
}

GraphCompilerInfo::~GraphCompilerInfo() {
  GraphScheduler::GetInstance().Clear(name_, graphs_, origin_parameters_order_, control_node_parser_);
}

GraphId GraphCompiler::CompileDynamicGraph(const GraphSegmentPtr &segment, const AnfNodePtrList &outputs,
                                           const DeviceContext *device_context) {
  MS_EXCEPTION_IF_NULL(segment);
  MS_EXCEPTION_IF_NULL(device_context);
  MS_LOG(INFO) << "Status record: start compile graph.";

  auto nodes = segment->nodes_;
  auto device_target = device_context->GetDeviceType();
  // Generate kernel graph.
  (void)profiler::CollectHostInfo(kModelNameRuntime, kEventCompileGraph, kStageConstructKernelGraph, 1, 0, 0);
  const auto &kernel_graph = session_->ConstructKernelGraph(nodes, outputs, device_target, true, false);
  return CompileDynamicGraph(kernel_graph, device_context);
}

GraphId GraphCompiler::CompileDynamicGraph(const KernelGraphPtr &kernel_graph, const DeviceContext *device_context) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  MS_EXCEPTION_IF_NULL(device_context);
  // Dynamic shape or dynamic graph structure flag.
  kernel_graph->set_flag(kAttrMutableKernel, true);
  MS_LOG(INFO) << "Set kFlagEnableRunGraphBySingleOp: Dynamic shape or dynamic graph structure flag";
  kernel_graph->set_flag(kFlagEnableRunGraphBySingleOp, true);

  kernel_graph->UpdateGraphAquireGilAttr();
  kernel_graph->SetInputNodes();
  auto manager = Manage(kernel_graph);
  if (manager) {
    manager->AddFuncGraph(kernel_graph);
    kernel_graph->set_manager(manager);
  }
  session_->SetInputNodeUsage(kernel_graph, manager);
  kernel_graph->SetOptimizerFlag();
  kernel_graph->set_run_mode(device::RunMode::kKernelMode);

  // kernel_graph kernel does not support pynative mode now, print a warning here.
  graphkernel::GraphKernelFlags::GetInstance().CheckSupport();

  GraphId graph_id = kernel_graph->graph_id();
  kernel_graph->set_root_graph_id(graph_id);
  session_->DumpGraphs({kernel_graph});

  MS_LOG(INFO) << "Status record: end compile kernel_graph. kernel_graph id: " << graph_id;
  return graph_id;
}

KernelGraphPtr GraphCompiler::ConstructKernelGraphForGraphRunMode(const FuncGraphPtr &func_graph,
                                                                  const DeviceContext *device_context,
                                                                  std::vector<KernelGraphPtr> *const all_graphs,
                                                                  bool *const need_return_ahead) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(device_context);
  MS_EXCEPTION_IF_NULL(all_graphs);
  auto device_target = device_context->GetDeviceType();
  KernelGraphPtr root_graph = session_->ConstructKernelGraph(func_graph, all_graphs, device_target);
  MS_EXCEPTION_IF_NULL(root_graph);
  for (const auto &graph : *all_graphs) {
    MS_EXCEPTION_IF_NULL(graph);
    MS_LOG(INFO) << "Set root graph for graph: " << graph->graph_id() << " to: " << root_graph->graph_id() << ".";
    graph->set_root_graph_id(root_graph->graph_id());
    graph->set_run_mode(device::RunMode::kGraphMode);
    graph->set_is_loop_count_sink(true);
    graph->set_attrs(func_graph->attrs());
    opt::OptimizationWithoutBackend(graph);
  }

  // Unify the MindIR, must be before of the graph optimization.
  auto kernel_executor = device_context->GetKernelExecutor(false);
  if (kernel_executor != nullptr) {
    kernel_executor->AddMindIRPass(root_graph);
  }

  // todo: waiting for GraphExecutor
  MS_EXCEPTION_IF_NULL(MsContext::GetInstance());
  if (MsContext::GetInstance()->backend_policy() == "ge") {
    auto manager = MakeManager();
    MS_EXCEPTION_IF_NULL(manager);
    for (const auto &graph : *all_graphs) {
      MS_EXCEPTION_IF_NULL(graph);
      graph->set_flag(kFlagEnableZeroCopyInGraph, true);
      manager->AddFuncGraph(graph);
      graph->set_manager(manager);
      graph->SetInputNodes();
    }
    root_graph->SetInputNodes();
    MS_EXCEPTION_IF_NULL(device_context->graph_executor_);
    if (!device_context->graph_executor_->CompileGraph(root_graph, {})) {
      MS_LOG(EXCEPTION) << "Compile graph failed: " << root_graph->graph_id();
    }
    root_graph->CacheGraphOutputToFrontNodeWithIndex({root_graph->output()}, {func_graph->output()});
    *need_return_ahead = true;
  }
  if (*need_return_ahead) {
    return root_graph;
  }
  // set executing sink true in graph mode
  root_graph->set_run_mode(device::RunMode::kGraphMode);
  root_graph->set_is_loop_count_sink(true);
#if defined(__linux__) && defined(WITH_BACKEND)
  // Embedding cache need global step of compute graph, can not enable loop sink, move loop control to loop count actor.
  if (ps::PSContext::instance()->cache_enable()) {
    root_graph->set_is_loop_count_sink(false);
    for (const auto &graph : *all_graphs) {
      MS_EXCEPTION_IF_NULL(graph);
      graph->set_is_loop_count_sink(false);
    }
  }
#endif
  root_graph->SetInputNodes();
  return root_graph;
}

GraphId GraphCompiler::CompileWholeGraphForGraphRunMode(const FuncGraphPtr &func_graph,
                                                        const DeviceContext *device_context) {
  MS_EXCEPTION_IF_NULL(session_);
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(device_context);
  MS_LOG(INFO) << "Status record: start compile graph.";
  // Generate kernel graph.
  std::vector<KernelGraphPtr> all_graphs;
  auto device_target = device_context->GetDeviceType();
  KernelGraphPtr root_graph;
  bool need_return_ahead = false;
  if (UseCacheToCompileGraph(func_graph, device_target)) {
    root_graph = session_->ConstructKernelGraph(&all_graphs);
    use_cache_to_compile_graph_ = true;
  } else {
    root_graph = ConstructKernelGraphForGraphRunMode(func_graph, device_context, &all_graphs, &need_return_ahead);
  }
  GraphId graph_id = root_graph->graph_id();
  if (need_return_ahead) {
    return graph_id;
  }
  if (ExportCompileCache(func_graph, device_target)) {
    export_compile_cache_ = true;
  }
  if (!func_graph->has_flag(kFlagPyNativeRunInGraph)) {
    graph_id = CompileGraphImpl(root_graph, device_context);
  }
  auto &context = CompileCacheContext::GetInstance();
  if (context.enable_compile_cache()) {
    CompileCacheContext::GetInstance().Clear();
  }

  // dump all graphs.
  // for ascend mindRT.
  session_->DumpGraphs(all_graphs);

  if (!func_graph->has_flag(kFlagPyNativeRunInGraph)) {
    // Cache the backend graph output nodes to front nodes with output index.
    auto output = func_graph->output();
    MS_EXCEPTION_IF_NULL(output);
    auto backend_node = root_graph->output();
    MS_EXCEPTION_IF_NULL(backend_node);
    root_graph->CacheGraphOutputToFrontNodeWithIndex({backend_node}, {output});
    AnfAlgo::UpdateGraphValidRefPair(root_graph);
  } else {
    for (auto &node : root_graph->execution_order()) {
      if (common::AnfAlgo::IsBpropCutOpExecInBackend(node)) {
        MS_LOG(INFO) << "Set kFlagEnableRunGraphBySingleOp: IsBpropCutOpExecInBackend";
        root_graph->set_flag(kFlagEnableRunGraphBySingleOp, true);
      }
    }
    root_graph->set_front_outputs({func_graph->output()});
  }
  MS_LOG(INFO) << "Status record: end compile graph. graph id: " << graph_id;
  return graph_id;
}

GraphId GraphCompiler::CompileGraphImpl(const KernelGraphPtr &graph, const DeviceContext *device_context,
                                        bool run_in_pynative) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(device_context);
  MS_EXCEPTION_IF_NULL(session_);
  const auto &context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  if (use_cache_to_compile_graph_) {
    UseCacheToCompileGraphImpl(graph, device_context);
  } else {
#ifdef ENABLE_DUMP_IR
    if (context->CanDump(kIntroductory)) {
      // Dump .pb graph before graph optimization.
      DumpIRProto(graph, "before_opt_" + std::to_string(graph->graph_id()));
    }
#endif
    MS_EXCEPTION_IF_NULL(device_context->GetKernelExecutor(false));
    // Execute optimization pass.
    (void)profiler::CollectHostInfo(kModelNameRuntime, kEventCompileGraph, kStageOptimizeGraph, 1, 0, 0);
    PROF_START(OptimizeGraph);
    device_context->GetKernelExecutor(false)->OptimizeGraph(graph);
    PROF_END(OptimizeGraph);
    (void)profiler::CollectHostInfo(kModelNameRuntime, kEventCompileGraph, kStageOptimizeGraph, 1, 0, 1);
    // Generate 'KernelMod' for all kernels and set 'KernelMod' into kernel,
    // 'KernelMod' is real executive object of kernel.
    (void)profiler::CollectHostInfo(kModelNameRuntime, kEventCompileGraph, kStageCreateKernel, 1, 0, 0);
    PROF_START(CreateKernel);
    device_context->GetKernelExecutor(false)->CreateKernel(graph->execution_order());
    PROF_END(CreateKernel);
    (void)profiler::CollectHostInfo(kModelNameRuntime, kEventCompileGraph, kStageCreateKernel, 1, 0, 1);

    // Kernels that are not supported by other device can be backed off and rebuilt on the CPU.
#ifdef WITH_BACKEND
    if (!graph->is_from_single_op()) {
      auto cpu_device_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(
        {kCPUDevice, device_context->device_context_key().device_id_});
      MS_EXCEPTION_IF_NULL(cpu_device_context);
      auto cpu_executor =
        dynamic_cast<device::cpu::CPUKernelExecutor *>(cpu_device_context->GetKernelExecutor(false).get());
      MS_EXCEPTION_IF_NULL(cpu_executor);
      cpu_executor->RebuildKernelSelectBackoffOp(graph->execution_order());
    }
#endif

    // Read the output and input ref map and set to the kernel graph.
    AnfAlgo::AddOutInRefToGraph(graph);

    // Optimize the nop node.
    if (!run_in_pynative) {
      OptimizeNopNode(graph.get());
#ifdef ENABLE_DUMP_IR
      if (context->CanDump(kIntroductory)) {
        DumpIR("hwopt_comm_after_eliminate_nopnode_" + graph->ToString() + ".ir", graph, true);
      }
#endif
    }

#ifndef ENABLE_SECURITY
    session_->RecurseSetSummaryNodesForAllGraphs(graph.get());
    // Update needed dump kernels for mindRT.
    DumpJsonParser::GetInstance().UpdateNeedDumpKernels(*graph.get());
#endif

    // dynamic shape pass of graphmode
    if (graph->is_dynamic_shape()) {
      if (!graph->is_graph_run_mode()) {
        // Temporarily disable CustomActor for asynchronous InferShape and Resize for the dynamic shape scenario,
        // and implement the corresponding capability through direct InferShape and Resize in KernelActor.
        // opt::DynamicShapeConvertPass(graph);
      }
      auto profiler_manage_inst = profiler::ProfilerManager::GetInstance();
      MS_EXCEPTION_IF_NULL(profiler_manage_inst);
      profiler_manage_inst->SetNetDynamicShapeStatus();
    }
  }

  if (export_compile_cache_) {
    session_->CacheKernelGraph(graph);
  }
  // Adjust kernel graph before run graph.
  PROF_START(PreprocessBeforeRun);
  device_context->GetKernelExecutor(false)->PreprocessBeforeRun(graph);
  PROF_END(PreprocessBeforeRun);
  graph->UpdateInternalParameter();
  // Set device target for parameter affinity.
  AnfAlgo::SetParameterDeviceTarget(graph);

  // Create device address for all anf nodes of graph.
  CreateDeviceAddress(graph, device_context);

#if defined(__linux__) && defined(WITH_BACKEND)
  // Set device address for embedding cache parameter, only enable when enable embedding cache mode.
  // `CreateDeviceAddress` should execute before this step.
  EmbeddingCacheScheduler::GetInstance().SetEmbedCachedParamAddress(device_context, graph);
#endif

  SetSummaryNodesRefCount(graph.get());
#ifdef ENABLE_DUMP_IR
  // Dump .pb graph after graph optimization.
  if (context->CanDump(kIntroductory)) {
    DumpIRProto(graph, "after_opt_" + std::to_string(graph->graph_id()));
  }
#endif

#ifdef ENABLE_DEBUGGER
  auto debugger = Debugger::GetInstance();
  MS_EXCEPTION_IF_NULL(debugger);
  // Dump graph for GPU mindRT if dump is enabled.
  debugger->DumpInGraphCompiler(graph);
  if (debugger && debugger->DebuggerBackendEnabled()) {
    // Load graphs for GPU and Ascend mindRT.
    debugger->LoadGraphs(graph);
  }
#endif

  graph->EnableRuntimeCache();
  return graph->graph_id();
}

KernelGraphPtr GraphCompiler::Fetch(GraphId graph_id) const {
  MS_EXCEPTION_IF_NULL(session_);
  return session_->GetGraph(graph_id);
}

void GraphCompiler::CreateDeviceAddress(const KernelGraphPtr &graph, const DeviceContext *device_context) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_LOG(INFO) << "Status record: start create device address. graph id: " << graph->graph_id();
  DeviceAddressUtils::CreateParameterDeviceAddress(device_context, graph);
  DeviceAddressUtils::CreateValueNodeDeviceAddress(device_context, graph);
  DeviceAddressUtils::CreateKernelOutputDeviceAddress(device_context, graph, false);
  DeviceAddressUtils::CreateKernelWorkspaceDeviceAddress(device_context, graph);
  DeviceAddressUtils::UpdateDeviceAddressForInplaceNode(graph);
  DeviceAddressUtils::UpdateDeviceAddressForRefNode(graph);
  MS_LOG(INFO) << "Status record: end create device address. graph id: " << graph->graph_id();
}

void GraphCompiler::GetParamAndOutputIndex(
  const KernelGraphPtr &graph, const std::vector<TensorPtr> &inputs, VectorRef *const outputs,
  std::map<AnfNodePtr, size_t> *parameter_index,
  std::map<KernelWithIndex, std::vector<std::vector<size_t>>> *output_indexes) {
  MS_EXCEPTION_IF_NULL(session_);
  session_->GetParameterIndex(graph.get(), inputs, parameter_index);
  session_->CreateOutputPlaceholder(graph, inputs, outputs, output_indexes);
}

void GraphCompiler::GetSingleOpInputTensors(const CNodePtr &kernel,
                                            const std::map<KernelWithIndex, tensor::BaseTensorPtr> &op_output,
                                            const std::map<AnfNodePtr, size_t> &parameter_index,
                                            const std::vector<TensorPtr> &graph_inputs, bool is_run_pyboost,
                                            InputInfo *const input_info) {
  MS_EXCEPTION_IF_NULL(session_);
  if (is_run_pyboost) {
    session_->GetOpInputTensorsFromCNode(kernel, op_output, parameter_index, graph_inputs, input_info);
  } else {
    session_->GetOpInputTensors(kernel, op_output, parameter_index, graph_inputs, input_info);
  }
}

tensor::BaseTensorPtr GraphCompiler::GetSingleOpInputTensorByIndex(
  const CNodePtr &kernel, const std::map<KernelWithIndex, tensor::BaseTensorPtr> &op_output,
  const std::map<AnfNodePtr, size_t> &parameter_index, const std::vector<TensorPtr> &graph_inputs,
  InputInfo *const input_info, size_t input_index) {
  MS_EXCEPTION_IF_NULL(session_);
  return session_->GetOpInputTensorByIndex(kernel, op_output, parameter_index, graph_inputs, input_info, input_index);
}

void GraphCompiler::GetSingleOpRunInfoAndGraphInfo(const CNodePtr &kernel, const InputInfo &input_info,
                                                   bool use_dynamic_shape_process,
                                                   session::BackendOpRunInfoPtr *op_run_info,
                                                   const GraphOutputInfo *const graph_output_info) {
  MS_EXCEPTION_IF_NULL(session_);
  *op_run_info = session_->GetSingleOpRunInfo(kernel, input_info, graph_output_info);
  (*op_run_info)->base_op_run_info.use_dynamic_shape_process = use_dynamic_shape_process;
}

void GraphCompiler::CalculateRefCount(const KernelGraphPtr &graph, std::map<KernelWithIndex, size_t> *ref_count) const {
  MS_EXCEPTION_IF_NULL(session_);
  session_->GetRefCount(graph.get(), ref_count);
}

void GraphCompiler::CalculateForwardOpOutputCount(const KernelGraphPtr &graph,
                                                  const std::vector<tensor::TensorPtr> &inputs,
                                                  std::map<std::string, size_t> *forward_op_output_tensor_id,
                                                  const std::map<AnfNodePtr, size_t> &parameter_index) const {
  MS_EXCEPTION_IF_NULL(session_);
  MS_EXCEPTION_IF_NULL(forward_op_output_tensor_id);
  forward_op_output_tensor_id->clear();
  session_->GetForwardOpOutputRefCount(graph.get(), inputs, forward_op_output_tensor_id, parameter_index);
}

void GraphCompiler::UpdateRefCount(const std::set<KernelWithIndex> &input_kernels_with_index,
                                   std::map<KernelWithIndex, size_t> *ref_count,
                                   std::map<KernelWithIndex, tensor::BaseTensorPtr> *op_output_map) const {
  MS_EXCEPTION_IF_NULL(session_);
  session_->HandleOpInputs(input_kernels_with_index, ref_count, op_output_map);
}

void GraphCompiler::UpdateForwardOpOutputRefCount(const std::vector<ValuePtr> &input_values,
                                                  std::map<std::string, size_t> *forward_op_output_tensor_id) const {
  MS_EXCEPTION_IF_NULL(session_);
  MS_EXCEPTION_IF_NULL(forward_op_output_tensor_id);
  session_->ReleaseForwardOpOutput(input_values, forward_op_output_tensor_id);
}

void GraphCompiler::RecoverGraphOutput(const AnfNodePtr &kernel, const VectorRef &op_outputs,
                                       const std::map<KernelWithIndex, size_t> &ref_count,
                                       std::map<KernelWithIndex, tensor::BaseTensorPtr> *op_output_map,
                                       GraphOutputInfo *const graph_output_info) const {
  MS_EXCEPTION_IF_NULL(session_);
  session_->HandleOpOutputs(kernel, op_outputs, ref_count, op_output_map, graph_output_info);
}

void GraphCompiler::RegisterSummaryCallBackFunc(const CallBackFunc &callback) const {
  MS_EXCEPTION_IF_NULL(session_);
#ifndef ENABLE_SECURITY
  session_->RegisterSummaryCallBackFunc(callback);
#endif
}

void GraphCompiler::Summary(const std::vector<KernelGraphPtr> &graphs) const {
  MS_EXCEPTION_IF_NULL(session_);
  for (const auto &graph : graphs) {
#ifndef ENABLE_SECURITY
    session_->Summary(graph.get());
#endif
  }
}

void GraphCompiler::SetGraphDependency(const KernelGraphPtr &graph, const GraphSegmentPtr &segment) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(segment);
  segment->graph_id_ = graph->graph_id();
  for (auto &pre_segment : segment->pre_segments_) {
    MS_EXCEPTION_IF_NULL(pre_segment);
    auto pre_graph = Fetch(pre_segment->graph_id_);
    MS_EXCEPTION_IF_NULL(pre_graph);
    pre_graph->AddPostGraph(graph);
    graph->AddPreGraph(pre_graph);
    MS_LOG(INFO) << "Link graph " << pre_segment->graph_id_ << " to " << graph->graph_id();
  }
}
}  // namespace runtime
}  // namespace mindspore
