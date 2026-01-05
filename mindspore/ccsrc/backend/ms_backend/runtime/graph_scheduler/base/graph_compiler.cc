/**
 * Copyright 2021-2025 Huawei Technologies Co., Ltd
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

#include "backend/ms_backend/runtime/graph_scheduler/base/graph_compiler.h"
#include <algorithm>
#include <cctype>
#include <functional>
#include <list>
#include <map>
#include <numeric>
#include <regex>
#include <utility>
#include <set>
#include <vector>
#include <string>
#include "backend/ms_backend/runtime/graph_scheduler/base/graph_scheduler.h"
#include "include/backend/common/device_address_utils.h"
#include "device_address/device_address.h"
#include "utils/convert_utils.h"
#include "include/runtime/hardware_abstract/kernel_base/graph_fusion/graph_kernel_flags.h"
#include "include/backend/common/pass_manager/common_backend_optimization.h"
#include "include/backend/common/custom_pass/custom_pass_executor.h"
#include "utils/ms_context.h"
#include "ir/tensor.h"
#include "ir/graph_utils.h"
#include "include/runtime/hardware_abstract/kernel_base/graph_fusion/framework_utils.h"
#include "tools/profiler/profiling.h"
#include "include/backend/common/pass_manager/helper.h"
#include "mindspore/ccsrc/utils/ir_dump/dump_proto.h"
#include "include/utils/parallel_context.h"
#include "include/utils/callback.h"
#ifdef ENABLE_DUMP_IR
#include "mindspore/ccsrc/utils/ir_dump/anf_ir_dump.h"
#endif

#include "include/backend/common/pass_manager/graph_optimizer.h"
#if defined(__linux__) && defined(WITH_BACKEND)
#include "include/cluster/topology/ps_context.h"
#endif
#include "tools/profiler/profiler.h"
#include "include/utils/compile_cache_context.h"
#include "primitive/framework_ops.h"
#include "include/runtime/utils/runtime_conf/runtime_conf.h"
#include "primitive/auto_generate/gen_ops_primitive_c.h"
#include "primitive/auto_generate/gen_ops_primitive_g.h"
#include "include/backend/backend_manager/backend_jit_config.h"

namespace mindspore {
namespace runtime {
uint32_t GraphCompilerInfo::backend_graph_id_ = 0;
constexpr auto kAttrBpropValueNodeRefCount = "bprop_value_node_ref_count";
constexpr auto kAttrValueNodeForwardOuputFlags = "value_node_forward_output_flags";

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
    auto kernel_tensor = AnfAlgo::GetOutputKernelTensor(node, index, false);
    MS_EXCEPTION_IF_NULL(kernel_tensor);
    MS_LOG(DEBUG) << "Set new ref count to max for summary node:" << node->fullname_with_scope()
                  << " debug string:" << node->DebugString() << " output index:" << index
                  << " kernel tensor:" << kernel_tensor->ToString();
    kernel_tensor->set_new_ref_count(SIZE_MAX);
  }
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

bool IsSwitchInlineNopNode(const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  if (runtime::RuntimeConf::GetInstance()->mem_optimize_level() != kOptimizeO0) {
    return std::find_if(cnode->inputs().begin(), cnode->inputs().end(), [](const auto &input) {
             return common::AnfAlgo::CheckPrimitiveType(input, prim::kPrimConditionGather) ||
                    common::AnfAlgo::CheckPrimitiveType(common::AnfAlgo::VisitKernelWithReturnType(input, 0).first,
                                                        prim::kPrimConditionGather);
           }) != cnode->inputs().end();
  }
  return std::find_if(cnode->inputs().begin(), cnode->inputs().end(), [](const auto &input) {
           return common::AnfAlgo::CheckPrimitiveType(input, prim::kPrimConditionGather) ||
                  common::AnfAlgo::CheckPrimitiveType(common::AnfAlgo::VisitKernelWithReturnType(input, 0).first,
                                                      prim::kPrimConditionGather) ||
                  common::AnfAlgo::CheckPrimitiveType(input, prim::kPrimConditionSwitch) ||
                  common::AnfAlgo::CheckPrimitiveType(common::AnfAlgo::VisitKernelWithReturnType(input, 0).first,
                                                      prim::kPrimConditionSwitch);
         }) != cnode->inputs().end();
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
    ref_out_value.insert(iter.first);
    ref_out_value.insert(iter.second);
  }
  MS_EXCEPTION_IF_NULL(output_node);
  const auto &graph_outputs = common::AnfAlgo::GetAllOutputWithIndex(output_node);
  auto is_graph_output = [&graph_outputs](const AnfNodePtr &node) {
    return std::any_of(graph_outputs.begin(), graph_outputs.end(), [&node](const KernelWithIndex &output) {
      const auto &real_output = common::AnfAlgo::FetchRealNodeSkipMonadControl(output);
      return real_output == KernelWithIndex(node, 0);
    });
  };
  // Collect all the nopnodes that can be eliminated.
  for (const auto &cnode : graph->execution_order()) {
    MS_EXCEPTION_IF_NULL(cnode);
    if ((!common::AnfAlgo::IsNopNode(cnode)) || ref_out_value.count({cnode, 0}) != 0 || is_graph_output(cnode) ||
        IsSwitchInlineNopNode(cnode)) {
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
    if (origin_pair.first->isa<Parameter>() || origin_pair.first->isa<ValueNode>() ||
        ref_out_value.find(origin_pair) != ref_out_value.end() || common::AnfAlgo::IsViewNode(origin_pair.first)) {
      continue;
    }
    // The ref node cannot be set for node pairs from different device target(appears in the kernel backoff scene).
    if (AnfAlgo::FetchDeviceTarget(origin_pair.first, graph) != AnfAlgo::FetchDeviceTarget(ref_node, graph)) {
      continue;
    }
    MS_LOG(INFO) << "The reference relation of nopnode " << ref_node->fullname_with_scope() << ", index: " << 0
                 << " to input " << origin_pair.first->fullname_with_scope() << ", index: " << origin_pair.second;
    graph->AddRefCorrespondPairs(std::make_pair(ref_node, 0), origin_pair);
    if (ref_node->kernel_info() != nullptr) {
      auto kernel_info = dynamic_cast<KernelInfo *>(ref_node->kernel_info());
      MS_EXCEPTION_IF_NULL(kernel_info);
      kernel_info->AddRefMap(0, origin_pair.second);
      MS_LOG(DEBUG) << "Add ref pair: [0, " << origin_pair.second << "] for node:" << ref_node->fullname_with_scope();
    } else {
      MS_LOG(DEBUG) << "No kernel info for nopnode:" << ref_node->fullname_with_scope();
    }
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

HashMap<ValueNodePtr, size_t> GetGraphValueNodeRefCounts(const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  HashMap<ValueNodePtr, size_t> value_node_ref_counts;
  // For example:
  //   %1 MakeTuple(V1, V2)
  //   %2 TupleGetItem(0, %1)
  //   %3 Kernel(%2)
  // V2 is not used by kernel. Need to remove.
  auto execution_nodes = graph->execution_order();
  for (auto &node : execution_nodes) {
    std::vector<session::KernelWithIndex> real_inputs;
    common::AnfAlgo::GetRealInputs(node, &real_inputs);
    for (auto &real_input : real_inputs) {
      auto input = real_input.first;
      MS_EXCEPTION_IF_NULL(input);
      if (input->isa<ValueNode>()) {
        auto value_node = input->cast<ValueNodePtr>();
        value_node_ref_counts[value_node] += 1;
      }
    }
  }

  // ValueNodes as graph outputs
  auto outputs = common::AnfAlgo::GetAllOutput(graph->output());
  for (auto &output : outputs) {
    MS_EXCEPTION_IF_NULL(output);
    if (output->isa<ValueNode>()) {
      auto value_node = output->cast<ValueNodePtr>();
      MS_EXCEPTION_IF_NULL(value_node);
      value_node_ref_counts[value_node] += 1;
    }
  }

  return value_node_ref_counts;
}

void RemoveUnusedValueNodes(const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  auto value_node_ref_counts = GetGraphValueNodeRefCounts(graph);
  for (const auto &value_node : graph->graph_value_nodes()) {
    MS_EXCEPTION_IF_NULL(value_node);
    auto iter = value_node_ref_counts.find(value_node);
    if (iter == value_node_ref_counts.end()) {
      MS_LOG(DEBUG) << "Remove unused ValueNode " << value_node->DebugString();
      graph->RemoveNodeFromGraph(value_node);
    }
  }
}

// The device address of graph value node need to release
// if the value node is output of forward_graph in PyNative mode.
void GenerateRefCountForBpropValueNode(const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  std::vector<uint32_t> value_node_ref_count_list;
  std::vector<bool> value_node_forward_output_flags;
  for (auto &value_node : graph->graph_value_nodes()) {
    MS_EXCEPTION_IF_NULL(value_node);
    (void)value_node_ref_count_list.emplace_back(UINT32_MAX);
    (void)value_node_forward_output_flags.emplace_back(false);
  }
  graph->set_attr(kAttrBpropValueNodeRefCount, MakeValue(value_node_ref_count_list));
  graph->set_attr(kAttrValueNodeForwardOuputFlags, MakeValue(value_node_forward_output_flags));
}

void SetRefInfoForKernelGraph(const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  for (const auto &kernel : graph->execution_order()) {
    MS_EXCEPTION_IF_NULL(kernel);
    mindspore::ops::OpDefPtr op_def = mindspore::ops::GetOpDef(common::AnfAlgo::GetCNodeName(kernel));
    if (op_def == nullptr || kernel->kernel_info() == nullptr) {
      continue;
    }
    auto kernel_info = dynamic_cast<device::KernelInfo *>(kernel->kernel_info());
    MS_EXCEPTION_IF_NULL(kernel_info);
    if (op_def->is_graph_view_) {
      auto build_info = kernel_info->select_kernel_build_info();
      if (build_info != nullptr) {
        const auto output_size = build_info->GetOutputNum();
        for (size_t i = 0; i < output_size; ++i) {
          kernel_info->AddRefMap(i, 0);
        }
        MS_LOG(DEBUG) << "Add ref pair: " << output_size
                      << " output to the first input for kernel: " << kernel->fullname_with_scope();
      }
    }
    for (size_t i = 0; i < op_def->returns_.size(); ++i) {
      if (op_def->returns_[i].inplace_input_index_ != -1) {
        MS_LOG(DEBUG) << "Add ref pair:" << i << ", " << op_def->returns_[i].inplace_input_index_
                      << " for kernel:" << kernel->fullname_with_scope();
        kernel_info->AddRefMap(i, op_def->returns_[i].inplace_input_index_);
      }
    }
  }
}

void SetRefMapForAnyTypeGraph(const KernelGraphPtr &graph) {
  SetRefInfoForKernelGraph(graph);
  for (const auto &cnode : graph->execution_order()) {
    MS_EXCEPTION_IF_NULL(cnode);
    auto kernel_info = dynamic_cast<device::KernelInfo *>(cnode->kernel_info());
    if (kernel_info == nullptr) {
      continue;
    }
    for (const auto &ref : kernel_info->out_in_ref_map()) {
      size_t output_index = ref.first;
      size_t input_index = ref.second;
      auto final_pair = std::make_pair(cnode, output_index);
      auto origin_pair = common::AnfAlgo::VisitKernel(common::AnfAlgo::GetInputNode(cnode, input_index), 0);
      // Add to graph only if the input is not a monad.
      if (!HasAbstractUMonad(origin_pair.first) && !HasAbstractIOMonad(origin_pair.first)) {
        graph->AddRefCorrespondPairs(final_pair, origin_pair);
        MS_LOG(INFO) << "The reference relation output " << final_pair.first->fullname_with_scope()
                     << ", output index: " << final_pair.second << " to input "
                     << origin_pair.first->fullname_with_scope() << ", output index: " << origin_pair.second
                     << " for graph:" << graph->ToString();
      }
    }
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
    const auto &output =
      common::AnfAlgo::VisitKernelWithReturnType(output_with_index.first, output_with_index.second).first;
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

  if (JitPipelineCompiling()) {
    GenerateRefCountForBpropValueNode(graph);
  }
  SetRefMapForAnyTypeGraph(graph);
  return graph->graph_id();
}

void RecursiveSetRunMode(const KernelGraphPtr &graph, std::set<KernelGraphPtr> *memo) {
  if (memo->find(graph) != memo->end()) {
    return;
  }
  memo->insert(graph);
  MS_LOG(INFO) << "Kernel graph: " << graph->ToString()
               << ", set run mode:" << device::run_mode_to_name_map.at(graph->RunMode());
  for (auto &child_graph : graph->child_graph_order()) {
    auto child_graph_ptr = child_graph.lock();
    MS_EXCEPTION_IF_NULL(child_graph_ptr);
    auto run_mode = graph->RunMode();
    child_graph_ptr->set_run_mode(run_mode);
    RecursiveSetRunMode(child_graph_ptr, memo);
  }
}

void ResetNodeId(const std::vector<KernelGraphPtr> &graphs) {
  static mindspore::HashMap<std::string, int> node_ids;
  for (const auto &graph : graphs) {
    MS_EXCEPTION_IF_NULL(graph);
    if (graph->memory_managed_by_ge()) {
      continue;
    }
    if (!graph->backend_jit_config().IsGptoOptionsEmpty()) {
      continue;
    }

#ifdef ENABLE_DUMP_IR
    auto context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(context);
    bool save_graphs = context->CanDump(kIntroductory);
    if (save_graphs) {
      std::string file_name = "graph_before_reset_id_" + std::to_string(graph->graph_id()) + ".ir";
      DumpIR(file_name, graph, true, kWholeStack);
    }
#endif
    const auto &all_nodes = TopoSort(graph->get_return(), SuccDeeperSimple);
    for (const auto &node : all_nodes) {
      if (node != nullptr && node->isa<CNode>()) {
        const auto &cnode = node->cast<CNodePtr>();
        MS_EXCEPTION_IF_NULL(cnode);
        const auto &fullname = cnode->fullname_with_scope();
        auto op_index = fullname.rfind("-op");
        if (op_index != string::npos) {
          auto scope_prefix = fullname.substr(0, op_index);
          if (node_ids.find(scope_prefix) == node_ids.end()) {
            node_ids[scope_prefix] = 0;
          } else {
            node_ids[scope_prefix]++;
          }
          cnode->set_fullname_with_scope(scope_prefix + "-op" + std::to_string(node_ids[scope_prefix]));
        }
      }
    }
  }
}
void SetOffloadUserData(const KernelGraphPtr &kernel_graph, const backend::BackendJitConfig &backend_jit_config) {
  if (common::GetEnv("MS_DEV_HYPER_OFFLOAD") != "1") {
    return;
  }
  if (backend_jit_config.offload_activation) {
    kernel_graph->set_user_data<bool>("offload_activation", std::make_shared<bool>(true));
  }
  if (backend_jit_config.offload_parameter) {
    kernel_graph->set_user_data<bool>("offload_parameter", std::make_shared<bool>(true));
  }
}
}  // namespace

GraphId GraphCompiler::CompileGraph(const GraphSegmentPtr &segment,
                                    const std::pair<AnfNodePtrList, AnfNodePtrList> &io_nodes,
                                    const DeviceContext *device_context,
                                    const backend::BackendJitConfig &backend_jit_config) {
  MS_EXCEPTION_IF_NULL(segment);
  MS_EXCEPTION_IF_NULL(device_context);
  MS_LOG(INFO) << "Status record: start compile graph.";
  auto nodes = segment->nodes_;
  auto device_target = device_context->GetDeviceType();
  // Generate kernel graph.
  uint64_t start_time = profiler::GetClockSyscnt();
  PROF_START(ConstructKernelGraph);
  auto kernel_graph =
    session_->ConstructKernelGraph(nodes, io_nodes.second, device_target, backend_jit_config, true, true);
  SetOffloadUserData(kernel_graph, backend_jit_config);
  PROF_END(ConstructKernelGraph);

  (void)profiler::CollectHostInfo(kModelNameRuntime, kEventCompileGraph, kStageConstructKernelGraph, start_time,
                                  profiler::GetClockSyscnt(), 1);
  SetGraphDependency(kernel_graph, segment);
  return CompileGraph(kernel_graph, io_nodes, device_context);
}

GraphId GraphCompiler::CompileGraph(const KernelGraphPtr &kernel_graph,
                                    const std::pair<AnfNodePtrList, AnfNodePtrList> &io_nodes,
                                    const DeviceContext *device_context) {
  MS_EXCEPTION_IF_NULL(session_);
  MS_EXCEPTION_IF_NULL(device_context);
  MS_EXCEPTION_IF_NULL(kernel_graph);

  const auto &outputs = io_nodes.second;
  if (common::AnfAlgo::IsAnyTypeInput(io_nodes.first)) {
    return CompileAnyTypeInputGraph(kernel_graph, outputs, device_context);
  }
  kernel_graph->UpdateGraphAquireGilAttr();

  kernel_graph->set_run_mode(device::RunMode::kKernelMode);
  std::set<KernelGraphPtr> memo;
  RecursiveSetRunMode(kernel_graph, &memo);
  auto manager = MakeManager({kernel_graph});
  if (manager) {
    manager->AddFuncGraph(kernel_graph);
    kernel_graph->set_manager(manager);
  }

  opt::OptimizationWithoutBackend(kernel_graph);

  // Execute custom passes
  std::string device_target = GetDeviceNameByType(device_context->GetDeviceType());
  std::transform(device_target.begin(), device_target.end(), device_target.begin(), ::tolower);
  opt::CustomPassExecutor::ExecuteCustomPasses(kernel_graph, device_target);

  // Unify the MindIR, must be before of the kernel_graph optimization.
  auto kernel_executor = device_context->GetKernelExecutor();
  if (kernel_executor != nullptr) {
    kernel_executor->AddMindIRPass(kernel_graph);
  }
  kernel_graph->SetInputNodes();
  kernel_graph->SetExecOrderByDefault();
  auto context_ptr = MsContext::GetInstance();
  session_->SetInputNodeUsage(kernel_graph, manager);
  MS_EXCEPTION_IF_NULL(context_ptr);
  kernel_graph->SetOptimizerFlag();

  GraphId graph_id = 0;
  graph_id = CompileGraphImpl(kernel_graph, device_context);

  kernel_graph->set_front_outputs(outputs);
  kernel_graph->set_root_graph_id(graph_id);

  ResetNodeId({kernel_graph});
  session_->DumpGraphs({kernel_graph});

  // Cache the backend kernel_graph output nodes to front nodes with output index.
  auto backend_node = kernel_graph->output();
  MS_EXCEPTION_IF_NULL(backend_node);
  kernel_graph->CacheGraphOutputToFrontNodeWithIndex({backend_node}, outputs);
  AnfAlgo::UpdateGraphValidRefPair(kernel_graph);

  MS_LOG(INFO) << "Status record: end compile graph. graph id: " << graph_id;
  return graph_id;
}

GraphCompilerInfo::~GraphCompilerInfo() {
  GraphScheduler::GetInstance().Clear(name_, graphs_, origin_parameters_order_, control_node_parser_);
}

void BuildStreamForCompileCache(const KernelGraphPtr &kernel_graph, const DeviceContext *device_context) {
  MS_EXCEPTION_IF_NULL(device_context);
  MS_EXCEPTION_IF_NULL(kernel_graph);
  uint32_t max_stream_id = 0;
  for (const auto &node : kernel_graph->execution_order()) {
    MS_EXCEPTION_IF_NULL(node);
    const auto &device_kernel_info = dynamic_cast<device::KernelInfo *>(node->kernel_info());
    if (device_kernel_info == nullptr) {
      MS_LOG(INFO) << "The node " << node->DebugString() << " has no device_kernel_info.";
      continue;
    }
    uint32_t stream_id = device_kernel_info->stream_id();
    max_stream_id = std::max(stream_id, max_stream_id);
  }
  size_t stream_id = 0;
  while (max_stream_id >= device_context->device_res_manager_->QueryStreamSize()) {
    device_context->device_res_manager_->CreateStream(&stream_id);
    MS_LOG(INFO) << "Success to create stream id:" << stream_id << ".";
  }
}

bool GraphCompiler::CacheGraphKbk(const std::vector<KernelGraphPtr> &graphs) {
  MS_LOG(INFO) << "Start to cache kernel graph.";
  bool cache_kernel_graph = session_->CacheKernelGraph(graphs);
  MS_LOG(INFO) << "Cache Kernel Graph " << (cache_kernel_graph == true ? "success" : "failed.");
  return cache_kernel_graph;
}
namespace {
void UpdateAbstractForAkgParameter(const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  std::for_each(graph->execution_order().begin(), graph->execution_order().end(), [](const CNodePtr &kernel) {
    if (kernel == nullptr || kernel->kernel_info() == nullptr) {
      MS_LOG(DEBUG) << "Invalid kernel";
      return;
    }
    if (AnfAlgo::GetKernelType(kernel) == KernelType::AKG_KERNEL) {
      MS_LOG(DEBUG) << "Check kernel:" << kernel->DebugString() << " fulllname:" << kernel->fullname_with_scope();
      auto func_graph = common::AnfAlgo::GetNodeAttr<FuncGraphPtr>(kernel, kAttrFuncGraph);
      if (func_graph == nullptr || kernel->size() != func_graph->parameters().size() + 1) {
        MS_LOG(DEBUG) << "Invalid funcgraph";
        return;
      }
      for (size_t i = 0; i < func_graph->parameters().size(); ++i) {
        if (kernel->input(i + 1) == nullptr || !kernel->input(i + 1)->isa<ValueNode>() ||
            func_graph->parameters()[i] == nullptr || func_graph->parameters()[i]->abstract() == nullptr ||
            (func_graph->parameters()[i]->abstract()->GetValue() != nullptr &&
             !func_graph->parameters()[i]->abstract()->GetValue()->isa<ValueAny>())) {
          MS_LOG(DEBUG) << "Invalid funcgraph input index:" << i;
          continue;
        }
        const auto &valuenode = kernel->input(i + 1)->cast<ValueNodePtr>();
        if (valuenode == nullptr || valuenode->value() == nullptr || !valuenode->value()->isa<Scalar>()) {
          MS_LOG(DEBUG) << "Invalid value node index:" << i;
          continue;
        }
        func_graph->parameters()[i]->abstract()->set_value(valuenode->value());
        MS_LOG(INFO) << "Set value:" << valuenode->DebugString()
                     << " to abstract:" << func_graph->parameters()[i]->abstract()->ToString()
                     << " parameter:" << func_graph->parameters()[i]->DebugString()
                     << " graph:" << func_graph->ToString();
      }
    }
  });
}
}  // namespace

bool GraphCompiler::CompileGraphForKernelRunModeUseCache(const FuncGraphPtr &func_graph,
                                                         const DeviceContext *device_context) {
  MS_EXCEPTION_IF_NULL(session_);
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_LOG(INFO) << "Status record: start use cache to compile graph kbk.";
  std::vector<KernelGraphPtr> all_graphs;
  auto graphs = session_->ConstructKernelGraph(&all_graphs);
  if (graphs.empty()) {
    MS_LOG(ERROR) << "Invalid compile cache for:" << func_graph->ToString();
    return false;
  }
  const auto &context = MsContext::GetInstance();
  auto post_compile = [this, device_context, context](const KernelGraphPtr &graph) {
    BuildStreamForCompileCache(graph, device_context);
    // Create event before create kernelmod
    device_context->GetKernelExecutor()->CreateEventForCache(graph);
    PROF_START(CreateKernel);
    UpdateAbstractForAkgParameter(graph);
    device_context->GetKernelExecutor()->CreateKernel(graph->execution_order());
    PROF_END(CreateKernel);
#ifdef WITH_BACKEND
    if (!graph->is_from_single_op()) {
      auto cpu_device_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(
        {device::DeviceType::kCPU, device_context->device_context_key().device_id_});
      MS_EXCEPTION_IF_NULL(cpu_device_context);
      cpu_device_context->GetKernelExecutor()->RebuildKernelSelectBackoffOp(graph->execution_order());
    }
#endif
    // dynamic shape pass of graphmode
    if (graph->is_dynamic_shape()) {
      auto profiler_manage_inst = profiler::ProfilerManager::GetInstance();
      MS_EXCEPTION_IF_NULL(profiler_manage_inst);
      profiler_manage_inst->SetNetDynamicShapeStatus();
    }
    graph->UpdateInternalParameter();
    // Set device target for parameter affinity.
    AnfAlgo::SetParameterDeviceTarget(graph);
    // Create device address for all anf nodes of graph.
    CreateDeviceAddress(graph, device_context);
#ifdef ENABLE_DUMP_IR
    // Dump .pb graph after graph optimization.
    if (context->CanDump(kIntroductory)) {
      DumpIR("complile_cache_after_opt_" + std::to_string(graph->graph_id()), graph, true);
    }
#endif
    graph->EnableRuntimeCache();
  };
  if (func_graph->func_graphs_used_total().empty() || graphs.size() == 1) {
    MS_LOG(INFO) << "Compie Single backend graph for:" << func_graph->ToString();
    post_compile(graphs[0]);
  } else {
    MS_LOG(INFO) << "Compie multi backend graph for:" << func_graph->ToString();
    for (const auto &graph : graphs) {
      MS_EXCEPTION_IF_NULL(graph);
      post_compile(graph);
    }
  }
  MS_LOG(INFO) << "Status record: end use cache to compile graph kbk for: " << func_graph->ToString();
  return true;
}

GraphId GraphCompiler::CompileGraphImpl(const KernelGraphPtr &graph, const DeviceContext *device_context) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(device_context);
  MS_EXCEPTION_IF_NULL(session_);
  const auto &context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
#ifdef ENABLE_DUMP_IR
  if (context->CanDump(kIntroductory)) {
    // Dump .pb graph before graph optimization.
    DumpIRProto(graph, "before_opt_" + std::to_string(graph->graph_id()));
  }
#endif
  MS_EXCEPTION_IF_NULL(device_context->GetKernelExecutor());
  // Execute optimization pass.
  uint64_t start_time = profiler::GetClockSyscnt();
  PROF_START(OptimizeGraph);
  device_context->GetKernelExecutor()->OptimizeGraph(graph);
  PROF_END(OptimizeGraph);
  (void)profiler::CollectHostInfo(kModelNameRuntime, kEventCompileGraph, kStageOptimizeGraph, start_time,
                                  profiler::GetClockSyscnt(), 1);
  // Generate 'KernelMod' for all kernels and set 'KernelMod' into kernel,
  // 'KernelMod' is real executive object of kernel.
  start_time = profiler::GetClockSyscnt();
  PROF_START(CreateKernel);
  graph->SetExecOrderByDefault();
  device_context->GetKernelExecutor()->CreateKernel(graph->execution_order());
  PROF_END(CreateKernel);
  (void)profiler::CollectHostInfo(kModelNameRuntime, kEventCompileGraph, kStageCreateKernel, start_time,
                                  profiler::GetClockSyscnt(), 1);

  // Kernels that are not supported by other device can be backed off and rebuilt on the CPU.
#ifdef WITH_BACKEND
  if (!graph->is_from_single_op()) {
    auto cpu_device_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(
      {device::DeviceType::kCPU, device_context->device_context_key().device_id_});
    MS_EXCEPTION_IF_NULL(cpu_device_context);
    cpu_device_context->GetKernelExecutor()->RebuildKernelSelectBackoffOp(graph->execution_order());
  }
#endif
  SetRefInfoForKernelGraph(graph);
  // Read the output and input ref map and set to the kernel graph.
  AnfAlgo::AddOutInRefToGraph(graph);

  // Optimize the nop node.
  OptimizeNopNode(graph.get());
#ifdef ENABLE_DUMP_IR
  if (context->CanDump(kIntroductory)) {
    DumpIR("hwopt_comm_after_eliminate_nopnode_" + graph->ToString() + ".ir", graph, true);
  }
#endif

  session_->RecurseSetSummaryNodesForAllGraphs(graph.get());
  // Update needed dump kernels for mindRT.
  constexpr char kUpdateNeedDumpKernels[] = "UpdateNeedDumpKernels";
  static auto update_need_dump_kernels_callback =
    callback::CommonCallback::GetInstance().GetCallback<void, const session::KernelGraph &>(kUpdateNeedDumpKernels);
  if (update_need_dump_kernels_callback) {
    update_need_dump_kernels_callback(*graph.get());
  } else {
    MS_LOG(WARNING) << "Failed to get UpdateNeedDumpKernels, data dump function may not work.";
  }

  // dynamic shape pass of graphmode
  if (graph->is_dynamic_shape()) {
    auto profiler_manage_inst = profiler::ProfilerManager::GetInstance();
    MS_EXCEPTION_IF_NULL(profiler_manage_inst);
    profiler_manage_inst->SetNetDynamicShapeStatus();
  }

  if (export_compile_cache_) {
    MS_LOG(INFO) << "Start to cache kernel graph.";
    auto cache_kernel_graph = session_->CacheKernelGraph({graph});
    MS_LOG(INFO) << "Cache Kernel Graph " << (cache_kernel_graph == true ? "success" : "failed.");
  }
  // Adjust kernel graph before run graph.
  PROF_START(PreprocessBeforeRun);
  device_context->GetKernelExecutor()->PreprocessBeforeRun(graph);
  PROF_END(PreprocessBeforeRun);
  graph->UpdateInternalParameter();
  // Set device target for parameter affinity.
  AnfAlgo::SetParameterDeviceTarget(graph);

  PROF_START(CreateDeviceAddress);
  // Create device address for all anf nodes of graph.
  CreateDeviceAddress(graph, device_context);
  PROF_END(CreateDeviceAddress);

  SetSummaryNodesRefCount(graph.get());
  RemoveUnusedValueNodes(graph);
  if (JitPipelineCompiling()) {
    GenerateRefCountForBpropValueNode(graph);
  }

#ifdef ENABLE_DUMP_IR
  // Dump .pb graph after graph optimization.
  if (context->CanDump(kIntroductory)) {
    DumpIRProto(graph, "after_opt_" + std::to_string(graph->graph_id()));
  }
#endif

#ifdef ENABLE_DEBUGGER
  // Dump graph for GPU mindRT if dump is enabled.
  constexpr char kDumpInGraphCompiler[] = "DumpInGraphCompiler";
  static auto dump_in_graph_compiler_callback =
    callback::CommonCallback::GetInstance().GetCallback<void, const KernelGraphPtr &>(kDumpInGraphCompiler);
  if (!dump_in_graph_compiler_callback) {
    MS_LOG(WARNING) << "Failed to get DumpInGraphCompiler, data dump function may not work.";
  } else {
    dump_in_graph_compiler_callback(graph);
    bool enabled = false;
    constexpr char kDebuggerBackendEnabled[] = "DebuggerBackendEnabled";
    static auto debugger_backend_enabled_callback =
      callback::CommonCallback::GetInstance().GetCallback<bool>(kDebuggerBackendEnabled);
    if (debugger_backend_enabled_callback) {
      enabled = debugger_backend_enabled_callback();
    } else {
      MS_LOG(WARNING) << "Failed to get DebuggerBackendEnabled, data dump function may not work.";
    }
    if (enabled) {
      constexpr char kLoadGraphs[] = "DebuggerLoadGraphs";
      static auto load_graphs_callback =
        callback::CommonCallback::GetInstance().GetCallback<void, const KernelGraphPtr &>(kLoadGraphs);
      if (load_graphs_callback) {
        load_graphs_callback(graph);
      } else {
        MS_LOG(WARNING) << "Failed to get DebuggerLoadGraphs, data dump function may not work.";
      }
    }
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

void GraphCompiler::RegisterSummaryCallBackFunc() const {
  MS_EXCEPTION_IF_NULL(session_);
  session_->RegisterSummaryCallBackFunc();
}

void GraphCompiler::Summary(const std::vector<KernelGraphPtr> &graphs) const {
  MS_EXCEPTION_IF_NULL(session_);
  for (const auto &graph : graphs) {
    session_->Summary(graph.get());
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
