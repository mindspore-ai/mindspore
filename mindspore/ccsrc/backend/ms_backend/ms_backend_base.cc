/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include "backend/ms_backend/ms_backend_base.h"

#include <algorithm>
#include <map>
#include <vector>
#include <queue>
#include <regex>
#include <unordered_map>
#include <set>
#include <utility>
#include <memory>
#include <string>

#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#endif

#include "ir/tensor_new.h"
#include "primitive/array_ops.h"
#include "backend/ms_backend/segment_runner.h"
#include "backend/common/pass/erase_invalid_micro_depend.h"
#include "backend/common/pass/erase_not_cut_attr.h"
#include "backend/common/pass/switch_not_cut.h"
#include "include/utils/callbacks.h"
#include "include/utils/scoped_long_running.h"
#include "utils/convert_utils.h"
#include "mindspore/ccsrc/utils/ir_dump/anf_ir_dump.h"
#include "include/utils/pipeline/fallback.h"
#include "include/runtime/utils/runtime_conf/runtime_env.h"
#include "include/runtime/memory/mem_pool/mem_tracker.h"
#include "ir/anf.h"
#include "primitive/framework_ops.h"
#include "primitive/sequence_ops.h"
#include "primitive/sparse_tensor_ops.h"
#include "primitive/nn_ops.h"
#include "include/backend/common/device_address_utils.h"
#include "backend/ms_backend/runtime/graph_executor/pre_launch/pre_launch_comm.h"
#include "include/runtime/hardware_abstract/stream/multi_stream_controller.h"
#include "runtime/hardware_abstract/utils.h"
#include "backend/ms_backend/runtime/graph_scheduler/base/graph_compiler.h"
#include "backend/ms_backend/runtime/graph_scheduler/base/graph_scheduler.h"
#include "backend/ms_backend/runtime/graph_scheduler/base/parameter_store.h"
#include "backend/ms_backend/runtime/graph_scheduler/base/graph_adapter.h"
#include "include/runtime/pipeline/pipeline.h"
#include "pybind_api/gil_scoped_long_running.h"
#include "utils/file_utils.h"
#include "utils/log_adapter.h"
#include "utils/llm_manager.h"
#include "utils/ms_utils.h"
#include "utils/info.h"
#include "utils/trace_info.h"
#include "tools/profiler/profiling.h"
#include "tools/profiler/mstx/mstx_impl.h"
#if defined(__linux__) && defined(WITH_BACKEND)
#include "include/cluster/topology/ps_context.h"
#endif
#include "include/runtime/hardware_abstract/kernel_base/graph_fusion/graph_kernel_flags.h"
#include "include/backend/common/pass_manager/graph_optimizer.h"
#include "mindspore/ccsrc/include/utils/symbol_engine/symbol_engine_impl.h"

#include "include/utils/compile_cache_context.h"
#include "include/utils/common.h"
#include "include/utils/stub_tensor.h"
#include "include/runtime/utils/runtime_conf/runtime_conf.h"
#include "include/runtime/utils/runtime_conf/thread_bind_core.h"
#include "include/utils/callback.h"
#include "include/runtime/hardware_abstract/collective/collective_manager.h"
#include "ir/func_graph_flag.h"
#include "ir/graph_utils.h"

namespace mindspore {
namespace backend {
namespace ms_backend {

namespace {
constexpr auto kControlNodeJsonSuffix = "_backinfo.json";
using PrimTypePair = std::pair<PrimitivePtr, abstract::AbstractFunctionPtr>;
using TypedPrimitiveAbstractClosurePtr = std::shared_ptr<abstract::TypedPrimitiveAbstractClosure>;

int64_t GetTupleGetItemOutIndex(const CNodePtr &tuple_get_item) {
  MS_EXCEPTION_IF_NULL(tuple_get_item);
  if (tuple_get_item->size() != kTupleGetItemInputSize) {
    MS_LOG(INTERNAL_EXCEPTION) << "#dmsg#Runtime error info:#dmsg#The node tuple_get_item must have 2 inputs!";
  }
  auto output_index_value_node = tuple_get_item->input(kInputNodeOutputIndexInTupleGetItem);
  MS_EXCEPTION_IF_NULL(output_index_value_node);
  auto value_node = output_index_value_node->cast<ValueNodePtr>();
  MS_EXCEPTION_IF_NULL(value_node);
  auto value = value_node->value();
  MS_EXCEPTION_IF_NULL(value);
  auto idx = value->isa<Int64Imm>() ? GetValue<int64_t>(value) : GetValue<int>(value);
  return idx;
}

bool EnableKBKCompileCache(const FuncGraphPtr &func_graph, const device::DeviceType &device_type) {
  if (!CompileCacheEnable()) {
    MS_LOG(INFO) << "Disable backend compile cache by front config.";
    return false;
  }
  if (runtime::IsDisableRuntimeConfig(runtime::kRuntimeCache)) {
    MS_LOG(INFO) << "Disable backend compile cache by backend config.";
    return false;
  }
  auto &context = CompileCacheContext::GetInstance();
  if (context.FrontGraph() != func_graph) {
    MS_LOG(INFO) << "Disable backend compile cache by invalid funcgraph:"
                 << (func_graph == nullptr ? " null" : func_graph->ToString()) << ", and context graph:"
                 << (context.FrontGraph() == nullptr ? " null" : context.FrontGraph()->ToString()) << ".";
    return false;
  }
  if (device_type != device::DeviceType::kAscend) {
    MS_LOG(INFO) << "Disable backend compile cache by invalid backend type:" << device_type;
    return false;
  }
  if (!context.UseCompileCache()) {
    MS_LOG(INFO) << "Disable backend compile cache by context no cache";
    return false;
  }
  MS_LOG(INFO) << "Enable backend compile cache.";
  return true;
}

bool ExportCompileCacheKBK(const FuncGraphPtr &func_graph, const device::DeviceType &device_type) {
  MS_EXCEPTION_IF_NULL(func_graph);
  if (!CompileCacheEnable()) {
    MS_LOG(INFO) << "Compile cache: disable by front compile cache config.";
    return false;
  }
  if (runtime::IsDisableRuntimeConfig(runtime::kRuntimeCache)) {
    MS_LOG(INFO) << "Compile cache: disable by backend compile cache config.";
    return false;
  }
  auto &context = CompileCacheContext::GetInstance();
  if (context.FrontGraph() != func_graph) {
    MS_LOG(INFO) << "Compile cache: disable by funcgraph:" << func_graph->ToString() << " context front graph:"
                 << (context.FrontGraph() == nullptr ? "null" : context.FrontGraph()->ToString());
    return false;
  }
  if (device_type != device::DeviceType::kAscend) {
    MS_LOG(INFO) << "Compile cache: disable by device type:" << device_type;
    return false;
  }
  if (context.UseCompileCache()) {
    MS_LOG(INFO) << "Compile cache: disable by compile cache context.";
    return false;
  }
  return true;
}

KernelWithIndex VisitRealNodeWithNestLevel(const AnfNodePtr &anf_node, size_t index, size_t *nest_level) {
  MS_EXCEPTION_IF_NULL(anf_node);
  if (!anf_node->isa<CNode>()) {
    return {anf_node, index};
  }
  auto cnode = anf_node->cast<CNodePtr>();
  if (common::AnfAlgo::GetCNodeName(cnode) == mindspore::kTupleGetItemOpName) {
    (*nest_level)++;
    auto real_node_with_index = VisitRealNodeWithNestLevel(common::AnfAlgo::GetTupleGetItemRealInput(cnode),
                                                           common::AnfAlgo::GetTupleGetItemOutIndex(cnode), nest_level);
    auto real_node = real_node_with_index.first;
    auto real_index = real_node_with_index.second;
    MS_EXCEPTION_IF_NULL(real_node);
    if (real_node->isa<CNode>() && common::AnfAlgo::GetCNodeName(real_node) == mindspore::kMakeTupleOpName) {
      (*nest_level)--;
      auto make_tuple = real_node->cast<CNodePtr>();
      return VisitRealNodeWithNestLevel(make_tuple->input(real_index + 1), index, nest_level);
    }
    return real_node_with_index;
  }
  return common::AnfAlgo::VisitKernelWithReturnType(anf_node, index, false,
                                                    {prim::kPrimMakeTuple, prim::kPrimTupleGetItem});
}

bool NeedConvertToRealTupleGetItem(const CNodePtr &cnode) {
  if (cnode->size() != kTupleGetItemInputSize) {
    return false;
  }
  if (!cnode->input(kInputNodeOutputIndexInTupleGetItem)->isa<ValueNode>() || GetTupleGetItemOutIndex(cnode) < 0) {
    return true;
  }
  size_t nest_level = 0;
  const size_t nest_limit = 1;
  auto real_node = VisitRealNodeWithNestLevel(cnode, 0, &nest_level);
  if (!common::AnfAlgo::IsCallNode(real_node.first) && AnfUtils::IsRealCNodeKernel(real_node.first) &&
      nest_level > nest_limit) {
    return true;
  }
  return false;
}

void CheckNodeValid(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  // Check the joined any abstract.
  const auto &node_abs = node->abstract();
  if (node_abs != nullptr && node_abs->isa<abstract::AbstractJoinedAny>()) {
    auto abs_joined_any = node_abs->cast<abstract::AbstractJoinedAnyPtr>();
    if (abs_joined_any != nullptr) {
      TraceGuard guard(MakeTraceInfo<TraceTypeJoin>(node->debug_info()));
      abs_joined_any->ThrowException();
    }
  }
}

void UnifyIR(const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  static const std::map<std::string, std::string> kOpListToTupleNames = {
    {mindspore::kMakeListNewOpName, mindspore::kMakeTupleOpName},
    {mindspore::kListGetItemOpName, mindspore::kTupleGetItemOpName},
    {mindspore::kListSetItemOpName, mindspore::kTupleSetItemOpName}};
  // List name --> tuple name.
  auto &&op_name = common::AnfAlgo::GetCNodeName(cnode);
  auto iter = kOpListToTupleNames.find(op_name);
  if (iter != kOpListToTupleNames.end()) {
    common::AnfAlgo::SetNodeAttr(kAttrOpAdaptationProcessed, MakeValue(true), cnode);
    cnode->set_input(0, mindspore::NewValueNode(std::make_shared<Primitive>(iter->second)));
    // Reset full scope name.
    cnode->set_fullname_with_scope("");
    MS_LOG(INFO) << "Rename op from " << iter->first << " to " << iter->second << " for op "
                 << cnode->fullname_with_scope() << ", debug name:" << cnode->DebugString();
    op_name = iter->second;
  }

  // TupleGetItem --> RealTupleGetItem.
  if (op_name == mindspore::kTupleGetItemOpName && NeedConvertToRealTupleGetItem(cnode)) {
    common::AnfAlgo::SetNodeAttr(kAttrOpAdaptationProcessed, MakeValue(true), cnode);
    cnode->set_input(0, mindspore::NewValueNode(std::make_shared<Primitive>(mindspore::kRealTupleGetItemOpName)));
    // Reset full scope name.
    cnode->set_fullname_with_scope("");
    MS_LOG(INFO) << "Rename op from TupleGetItem to RealTupleGetItem for op " << cnode->fullname_with_scope()
                 << ", debug name:" << cnode->DebugString();
  }

  // MakeTuple --> RealMakeTuple
  if (op_name == mindspore::kMakeTupleOpName && common::AnfAlgo::IsDynamicSequence(cnode)) {
    common::AnfAlgo::SetNodeAttr(kAttrOpAdaptationProcessed, MakeValue(true), cnode);
    cnode->set_input(0, mindspore::NewValueNode(std::make_shared<Primitive>(mindspore::kRealMakeTupleOpName)));
    // Reset full scope name.
    cnode->set_fullname_with_scope("");
    MS_LOG(INFO) << "Rename op from MakeTuple to RealMakeTuple for op " << cnode->fullname_with_scope()
                 << ", debug name:" << cnode->DebugString();
  }
}

void DoUnifyMindIRPass(const FuncGraphPtr &graph, const std::shared_ptr<opt::GraphOptimizer> &optimizer) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(optimizer);
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  MS_LOG(INFO) << "Do unify mindir pass for graph " << graph->ToString();
#ifdef ENABLE_DUMP_IR
  if (context_ptr->CanDump(kIntroductory)) {
    std::string file_name = "hwopt_before_mindrt_unify_mindir_graph_" + graph->ToString() + ".ir";
    DumpIR(file_name, graph, true, kWholeStack);
  }
#endif
  (void)optimizer->Optimize(graph);
#ifdef ENABLE_DUMP_IR
  if (context_ptr->CanDump(kIntroductory)) {
    std::string file_name = "hwopt_end_mindrt_unify_mindir_graph_" + graph->ToString() + ".ir";
    DumpIR(file_name, graph, true, kWholeStack);
  }
#endif
}

bool HasSwitchNode(const FuncGraphPtr &func_graph) {
  if (func_graph == nullptr) {
    return false;
  }
  const auto &nodes = TopoSort(func_graph->get_return());
  return std::any_of(nodes.begin(), nodes.end(), [](const AnfNodePtr &node) {
    return node != nullptr && node->isa<CNode>() && common::AnfAlgo::CheckPrimitiveType(node, prim::kPrimSwitch);
  });
}

bool HasAbstractRefOutput(const abstract::AbstractBasePtr &abs) {
  if (abs == nullptr) {
    return false;
  }
  if (abs->isa<abstract::AbstractSequence>()) {
    const auto &seq_abs = abs->cast<abstract::AbstractSequencePtr>();
    MS_EXCEPTION_IF_NULL(seq_abs);
    if (seq_abs->dynamic_len()) {
      return false;
    }
    if (std::any_of(seq_abs->elements().begin(), seq_abs->elements().end(),
                    [](const abstract::AbstractBasePtr &sub_abs) { return HasAbstractRefOutput(sub_abs); })) {
      return true;
    }
  }
  if (abs->isa<abstract::AbstractRefTensor>()) {
    return true;
  }
  return false;
}

bool IsNodeValid(const AnfNodePtr &node) {
  if (node == nullptr) {
    return false;
  } else if (common::AnfAlgo::IsNodeOutputDynamicShape(node)) {
    MS_LOG(INFO) << "Disable switch inline for dynamic shape node:" << node->DebugString();
    return false;
  } else if (node->isa<CNode>() && common::AnfAlgo::IsTypeTransformOp(common::AnfAlgo::GetCNodeName(node))) {
    MS_LOG(INFO) << "Disable switch inline for backoff node:" << node->DebugString();
    return false;
  } else if (common::AnfAlgo::CheckPrimitiveType(node, prim::kPrimPyExecute)) {
    MS_LOG(INFO) << "Disable switch inline for fallback node:" << node->DebugString();
    return false;
  } else if (common::AnfAlgo::CheckPrimitiveType(node, prim::kPrimPartial)) {
    const auto &cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    if (cnode->size() <= 1 || cnode->input(1) == nullptr || !(IsValueNode<FuncGraph>(cnode->input(1)))) {
      return true;
    }
    const auto &func_graph = GetValueNode<FuncGraphPtr>(cnode->input(1));
    MS_EXCEPTION_IF_NULL(func_graph);
    if (std::any_of(func_graph->parameters().begin(), func_graph->parameters().end(), [](const AnfNodePtr &para) {
          return para != nullptr && para->abstract() != nullptr &&
                 para->abstract()->isa<abstract::AbstractSequence>() &&
                 (para->abstract()->cast<abstract::AbstractSequencePtr>()->dynamic_len() ||
                  para->abstract()->cast<abstract::AbstractSequencePtr>()->size() > 1);
        })) {
      MS_LOG(INFO) << "Disable switch inline for tuple input in graph:" << func_graph->ToString()
                   << " for partial node:" << node->DebugString();
      return false;
    }
  } else if (common::AnfAlgo::IsCallNode(node) && HasAbstractRefOutput(node->abstract())) {
    MS_LOG(INFO) << "Disable switch inline for call node with AbstractRefOutput:" << node->DebugString();
    return false;
  }
  return true;
}

// Check if src_node depends on dst_node.
bool IsTopoDependNode(const std::set<AnfNodePtr> &checked_calls, const AnfNodePtr &node,
                      std::set<AnfNodePtr> *checked_node) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(checked_node);
  if (checked_calls.find(node) != checked_calls.end()) {
    return true;
  }
  if (!node->isa<CNode>() || checked_node->find(node) != checked_node->end()) {
    return false;
  }

  (void)checked_node->emplace(node);
  const auto &cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  const auto &inputs = cnode->inputs();
  for (const auto &input : inputs) {
    MS_EXCEPTION_IF_NULL(input);
    if (IsTopoDependNode(checked_calls, input, checked_node)) {
      return true;
    }
  }
  return false;
}

bool HasParallelSwitchCall(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  std::vector<AnfNodePtr> switch_calls;
  const auto &nodes = TopoSort(func_graph->get_return());
  for (const auto &node : nodes) {
    if (!common::AnfAlgo::IsCallNode(node)) {
      continue;
    }
    const auto &cnode = node->cast<CNodePtr>();
    if (cnode == nullptr || cnode->size() == 0 || cnode->input(0) == nullptr ||
        (!common::AnfAlgo::CheckPrimitiveType(cnode->input(0), prim::kPrimSwitch))) {
      continue;
    }
    switch_calls.emplace_back(node);
  }
  if (switch_calls.size() <= 1) {
    return false;
  }
  constexpr size_t kMaxSwitchInlineSize = 10;
  if (switch_calls.size() >= kMaxSwitchInlineSize) {
    MS_LOG(INFO) << "Disable switch inline for switch node:" << switch_calls.size() << " more than 10.";
    return true;
  }
  std::set<AnfNodePtr> checked_calls{switch_calls.front()};
  for (size_t i = 1; i < switch_calls.size(); ++i) {
    std::set<AnfNodePtr> checked_nodes;
    if (!IsTopoDependNode(checked_calls, switch_calls[i], &checked_nodes)) {
      MS_LOG(INFO) << "Switch call node:" << switch_calls[i]->DebugString() << " has other parallel call node.";
      return true;
    }
    checked_calls.emplace(switch_calls[i]);
  }
  return false;
}

bool IsFuncGraphSupportSwitchInline(const FuncGraphPtr &graph) {
  return HasParallelSwitchCall(graph) ||
         std::any_of(graph->func_graphs_used_total().cbegin(), graph->func_graphs_used_total().cend(),
                     [](const auto &sub_graph) { return sub_graph != nullptr && HasParallelSwitchCall(sub_graph); });
}

bool IsEnableControlFlowInline(const FuncGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  if (std::any_of(
        graph->func_graphs_used_total().cbegin(), graph->func_graphs_used_total().cend(), [](const auto &sub_graph) {
          return sub_graph != nullptr && sub_graph->has_flag(FUNC_GRAPH_FLAG_CELL_REUSE) && HasSwitchNode(sub_graph);
        })) {
    MS_LOG(INFO) << "Set reuse level from:" << context->CellReuseLevel() << " to:" << CellReuseLevel::kNoInline;
    context->SetCellReuseLevel(CellReuseLevel::kNoInline);
  }

  static const auto is_disable_switch_inline = runtime::IsDisableRuntimeConfig(runtime::kRuntimeSwitchInline);
  if (is_disable_switch_inline) {
    MS_LOG(INFO) << "Disable switch inline by runtime config.";
    return false;
  }

  // Only support ascend, kernel by kernel mode and multi-funcgraph.
  static const bool is_enable_ge = (context->backend_policy() == "ge");
  if (!is_enable_ge || !context->IsKByKExecutorMode() || graph->func_graphs_used_total().empty()) {
    MS_LOG(INFO) << "Disable switch inline, executor mode:" << context->IsKByKExecutorMode();
    return false;
  }

  MS_EXCEPTION_IF_NULL(graph);
  // Not support recursive.
  if (std::any_of(graph->func_graphs_used_total().cbegin(), graph->func_graphs_used_total().cend(),
                  [](const auto &sub_graph) { return sub_graph->recursive(); })) {
    MS_LOG(INFO) << "Disable switch inline for recursive.";
    return false;
  }

  auto runtime_num_threads = static_cast<size_t>(context->get_param<uint32_t>(MS_CTX_RUNTIME_NUM_THREADS));
  if (runtime_num_threads <= 1) {
    MS_LOG(INFO) << "Disable switch inline for single thread.";
    return false;
  }

  if (context->CellReuseLevel() != CellReuseLevel::kLazyInline) {
    auto is_include_no_switch_call = [](const FuncGraphPtr &graph) {
      MS_EXCEPTION_IF_NULL(graph);
      const auto &nodes = TopoSort(graph->get_return());
      for (const auto &node : nodes) {
        MS_EXCEPTION_IF_NULL(node);
        if (common::AnfAlgo::IsCallNode(node)) {
          const auto &cnode = node->cast<CNodePtr>();
          MS_EXCEPTION_IF_NULL(cnode);
          if (!common::AnfAlgo::CheckPrimitiveType(cnode->input(0), prim::kPrimSwitch)) {
            return true;
          }
        }
      }
      return false;
    };
    if (is_include_no_switch_call(graph)) {
      MS_LOG(INFO) << "Disable switch inline for unsupported call node.";
      return false;
    }
    if (std::any_of(graph->func_graphs_used_total().begin(), graph->func_graphs_used_total().end(),
                    is_include_no_switch_call)) {
      MS_LOG(INFO) << "Disable switch inline for unsupported call node.";
      return false;
    }
  }
  const auto &all_nodes = TopoSort(graph->get_return(), SuccDeeperSimple);
  if (std::any_of(all_nodes.begin(), all_nodes.end(), [](const AnfNodePtr &node) { return !IsNodeValid(node); })) {
    return false;
  }
  MS_LOG(INFO) << "Start check parallel switch call.";
  if (IsFuncGraphSupportSwitchInline(graph)) {
    MS_LOG(INFO) << "Disable switch inline for parallel switch call node.";
    return false;
  }
  MS_LOG(INFO) << "Enable switch inline.";
  return true;
}

bool IsTupleOutputOfAnyType(const abstract::AbstractBasePtr &abstract, const tensor::TensorPtr &tensor) {
  if (abstract == nullptr || !abstract->isa<abstract::AbstractAny>() || tensor == nullptr) {
    return false;
  }
  auto device_tensor = std::dynamic_pointer_cast<device::DeviceAddress>(tensor->device_address());
  const auto &user_data_obj =
    tensor->GetUserData().get<kernel::PyExecuteOutputUserData>(kernel::PyExecuteOutputUserData::key);
  return device_tensor != nullptr && user_data_obj == nullptr && tensor->base_shape_ptr() != nullptr &&
         tensor->base_shape_ptr()->isa<abstract::SequenceShape>();
}

bool EnableSymbolEngine(const FuncGraphPtr &func_graph) {
  if (!graphkernel::GraphKernelFlags::GetInstance().IsEnableGraphKernel()) {
    return false;
  }
  return common::AnfAlgo::IsDynamicGraph(func_graph);
}

void BuildSymbolEngine(const FuncGraphPtr &func_graph) {
  if (func_graph == nullptr) {
    return;
  }
  if (!EnableSymbolEngine(func_graph)) {
    MS_LOG(INFO) << "Status record: skip build symbol engine for function graph: " << func_graph->ToString();
    return;
  }
  MS_LOG(INFO) << "Status record: start build symbol engine for function graph: " << func_graph->ToString();
  (void)symshape::SymbolEngineImpl::Build(func_graph);
  MS_LOG(INFO) << "Status record: end build symbol engine for function graph: " << func_graph->ToString();
#ifdef ENABLE_DUMP_IR
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  if (context_ptr->CanDump(kIntroductory)) {
    std::string file_name = "root_graph_with_symbol_engine.ir";
    DumpIR(file_name, func_graph, true, kWholeStack);
  }
#endif
}

std::string GetUniqueNodeId(const AnfNodePtr &node, bool must_have_unique_name = true) {
  MS_EXCEPTION_IF_NULL(node);
  const auto &name = node->user_data<std::string>(kUniqueCacheName);
  if (must_have_unique_name && name == nullptr) {
    MS_LOG(EXCEPTION) << "The node " << node->DebugString()
                      << " has not unique name, indicating that it is not exported to mindir.";
  }
  return name != nullptr ? *name : "node is nullptr";
}

// Insert the front_node related tensor in the input_tensor.
void PushTensor(const VectorRef &args, const std::vector<AnfNodePtr> &parameters, const AnfNodePtr &front_node,
                std::vector<tensor::TensorPtr> *input_tensors) {
  MS_EXCEPTION_IF_NULL(input_tensors);
  const auto &iter = std::find(parameters.begin(), parameters.end(), front_node);
  if (iter == parameters.end()) {
    (void)((*input_tensors).emplace_back(nullptr));
    return;
  }
  auto position = iter - parameters.begin();

  std::vector<tensor::TensorPtr> flatten_values;
  AnfAlgo::FlattenInputArg(args[position], front_node, &flatten_values);
  (void)std::copy(flatten_values.begin(), flatten_values.end(), std::back_inserter(*input_tensors));
}

void PushTupleTensor(const VectorRef &args, const std::vector<AnfNodePtr> &parameters, const AnfNodePtr &front_node,
                     size_t index, std::map<size_t, std::vector<tensor::TensorPtr>> *flatten_values,
                     std::vector<tensor::TensorPtr> *input_tensors) {
  MS_EXCEPTION_IF_NULL(input_tensors);
  MS_EXCEPTION_IF_NULL(flatten_values);

  const auto &iter = std::find(parameters.begin(), parameters.end(), front_node);
  const size_t position = static_cast<size_t>(iter - parameters.begin());
  // If the parameter is not found in the parameters of the root graph, it means that it is the input of the subgraph,
  // and there is no need to input a tensor.
  if (position >= args.size()) {
    MS_LOG(DEBUG) << "Position out of args range, position value is " << position << " and args size is " << args.size()
                  << ".";
    (void)input_tensors->emplace_back(nullptr);
    return;
  }

  // Avoid repeating flatten tuple for each args position.
  auto &flatten_value = (*flatten_values)[position];
  if (flatten_value.empty()) {
    AnfAlgo::FlattenInputArg(args[position], front_node, &flatten_value);
  }

  if (index >= flatten_value.size()) {
    MS_LOG(INTERNAL_EXCEPTION) << "#dmsg#Runtime error info:#dmsg#Index out of flatten_value range, index value is "
                               << index << " and flatten_value size is " << flatten_value.size() << ".";
  }
  auto tensor_input = flatten_value[index];
  MS_EXCEPTION_IF_NULL(tensor_input);
  input_tensors->push_back(tensor_input);
}

bool IsEmptySequence(const AnfNodePtr &output_node, const std::vector<tensor::TensorPtr> &output_tensors,
                     const size_t *const output_position) {
  MS_EXCEPTION_IF_NULL(output_node);
  MS_EXCEPTION_IF_NULL(output_position);
  // When the output node is a valuenode, the position may out of range.
  if (*output_position >= output_tensors.size()) {
    return false;
  }

  if (output_node->abstract() == nullptr || (!output_node->abstract()->isa<abstract::AbstractSequence>())) {
    return false;
  }
  const auto &tuple_abs = output_node->abstract()->cast<abstract::AbstractSequencePtr>();
  MS_EXCEPTION_IF_NULL(tuple_abs);
  if ((!tuple_abs->dynamic_len()) && tuple_abs->dynamic_len_element_abs() == nullptr) {
    return false;
  }
  const auto &tensor = output_tensors[*output_position];
  MS_EXCEPTION_IF_NULL(tensor);
  if (tensor->base_shape_ptr() == nullptr || (!tensor->base_shape_ptr()->isa<abstract::SequenceShape>())) {
    return false;
  }
  const auto &sequence_shape = tensor->base_shape_ptr()->cast<abstract::SequenceShapePtr>();
  MS_EXCEPTION_IF_NULL(sequence_shape);
  return sequence_shape->size() == 0;
}

void TraverseGraphMap(
  const FuncGraphManagerPtr &manager_ptr, FuncGraphTransaction *tr, const FuncGraphSet &fgs,
  const std::function<std::shared_ptr<FuncGraph>(const PrimitivePtr, const AbstractFunctionPtr)> &get_prim_graph) {
  MS_EXCEPTION_IF_NULL(manager_ptr);
  MS_EXCEPTION_IF_NULL(tr);
  auto is_prim_array_map_reduce_skip = [](const PrimitivePtr &value, int64_t key, bool key_is_const) {
    if (value == nullptr) {
      return false;
    }
    bool is_prim_array_map = !(prim::kPrimArrayMap->name().compare(value->name()));
    bool is_prim_array_reduce = !(prim::kPrimArrayReduce->name().compare(value->name()));
    return key == 1 && key_is_const && (is_prim_array_map || is_prim_array_reduce);
  };
  for (const auto &fg : fgs) {
    MS_EXCEPTION_IF_NULL(fg);
    for (const auto &ct_any : fg->value_nodes()) {
      AnfNodePtr const_primitive_node = ct_any.first;
      if (const_primitive_node == nullptr || !IsValueNode<Primitive>(const_primitive_node)) {
        continue;
      }
      auto users = manager_ptr->node_users()[const_primitive_node];
      for (auto &use : users) {
        CNodePtr node = use.first->cast<CNodePtr>();
        MS_EXCEPTION_IF_NULL(node);
        if (node->func_graph() != fg) {
          continue;
        }
        MS_EXCEPTION_IF_NULL(node->input(0));
        bool key_is_const = node->input(0)->isa<ValueNode>();
        int64_t key = use.second;
        if (key == 0) {
          continue;
        }
        PrimitivePtr value = GetValueNode<PrimitivePtr>(node->input(0));
        if (is_prim_array_map_reduce_skip(value, key, key_is_const)) {
          continue;
        }
        FuncGraphPtr g = get_prim_graph(GetValueNode<PrimitivePtr>(const_primitive_node),
                                        dyn_cast<AbstractFunction>(const_primitive_node->abstract()));
        tr->SetEdge(node, key, NewValueNode(g));
      }
    }
  }
}

runtime::KernelMapPosition FetchOriginOutputOrder(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  runtime::KernelMapPosition outputs_order;
  const auto &root_output = common::AnfAlgo::VisitKernelWithReturnType(node, 0, false, {prim::kPrimTupleGetItem}).first;
  size_t position = 0;
  auto outputs = common::AnfAlgo::GetAllOutputWithIndex(root_output);
  for (const auto &output : outputs) {
    if (outputs_order.count(output) == 0) {
      outputs_order[output] = {position++};
    } else {
      (void)outputs_order[output].emplace_back(position++);
    }
  }
  return outputs_order;
}
}  // namespace

void MSBackendBase::ClearGraphBuildMember() {
  MS_EXCEPTION_IF_NULL(graph_compiler_);
  graph_compiler_->ClearGraphBuildMember();

  root_graph_ = nullptr;
  graph_id_to_device_context_.clear();
  func_graph_to_kernel_graph_ids_.clear();
  func_graph_to_sub_segments_.clear();
  control_nodes_.clear();
}

void MSBackendBase::UnifyMindIR(const FuncGraphPtr &root_graph) const {
  MS_EXCEPTION_IF_NULL(root_graph);
  MS_EXCEPTION_IF_NULL(root_graph->manager());
  // When the input is an empty sequence, the number of inputs will be recorded as 0, and the tensor cannot be
  // expressed, so the empty sequence is set to dynamic len.
  for (const auto &parameter : root_graph->parameters()) {
    MS_EXCEPTION_IF_NULL(parameter);
    const auto &abs = parameter->abstract();
    if (abs != nullptr && abs->isa<abstract::AbstractSequence>()) {
      const auto &sequence_abs = abs->cast<abstract::AbstractSequencePtr>();
      MS_EXCEPTION_IF_NULL(sequence_abs);
      if ((!sequence_abs->dynamic_len()) && sequence_abs->empty()) {
        MS_LOG(INFO) << "Set dynamic len flag for empty sequence input:" << parameter->DebugString();
        sequence_abs->set_dynamic_len(true);
      }
    }
  }
  const auto &graphs = root_graph->manager()->func_graphs();
  for (const auto &graph : graphs) {
    MS_EXCEPTION_IF_NULL(graph);
    auto output = graph->get_return();
    if (!output->isa<CNode>()) {
      continue;
    }
    auto seen = NewSeenGeneration();
    std::queue<AnfNodePtr> to_visit;
    to_visit.emplace(output);
    while (!to_visit.empty()) {
      auto node = to_visit.front();
      to_visit.pop();
      MS_EXCEPTION_IF_NULL(node);
      CheckNodeValid(node);

      const auto &cnode = node->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(cnode);
      UnifyIR(cnode);
      for (auto &input : cnode->inputs()) {
        MS_EXCEPTION_IF_NULL(input);
        if (input->seen_ == seen || !input->isa<CNode>()) {
          continue;
        }
        to_visit.emplace(input);
        input->seen_ = seen;
      }
    }
  }

  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto unify_mindir_pm = std::make_shared<opt::PassManager>("unify_mindir_pm");
  unify_mindir_pm->AddPass(std::make_shared<opt::EraseInvalidMicroDepend>());
  if (common::AnfAlgo::IsDynamicGraph(root_graph)) {
    unify_mindir_pm->AddPass(std::make_shared<opt::EraseNotCutAttr>());
  }
  if (IsEnableControlFlowInline(root_graph)) {
    unify_mindir_pm->AddPass(std::make_shared<opt::SwitchNotCut>());
  }
  optimizer->AddPassManager(unify_mindir_pm);

  DoUnifyMindIRPass(root_graph, optimizer);
  const auto &sub_graphs = root_graph->manager()->func_graphs_used_total(root_graph);
  for (const auto &sub_graph : sub_graphs) {
    MS_EXCEPTION_IF_NULL(sub_graph);
    DoUnifyMindIRPass(sub_graph, optimizer);
  }
}

void MSBackendBase::ProcessNotSupportCnode(const FuncGraphPtr &func_graph,
                                           const mindspore::device::DeviceType &old_target,
                                           const mindspore::device::DeviceType &new_target) const {
  const auto &all_nodes = TopoSort(func_graph->return_node(), SuccDeeperSimple, AlwaysInclude);
  for (const auto &node : all_nodes) {
    MS_EXCEPTION_IF_NULL(node);
    if (!node->isa<CNode>()) {
      continue;
    }

    auto cnode = node->cast<CNodePtr>();
    if (!common::AnfAlgo::HasNodeAttr(mindspore::kAttrNotSupportOpForDevice, cnode)) {
      continue;
    }

    auto not_support_device = common::AnfAlgo::GetNodeAttr<std::string>(node, mindspore::kAttrNotSupportOpForDevice);
    if (device::GetDeviceTypeByName(not_support_device) != old_target) {
      continue;
    }

    common::AnfAlgo::SetNodeAttr(kAttrPrimitiveTarget, MakeValue(device::GetDeviceNameByType(new_target)), node);
  }
}

void MSBackendBase::CacheFuncGraphWithKernelGraphId(const FuncGraphPtr &func_graph, const GraphId &graph_id,
                                                    DeviceContext *device_context) {
  graph_id_to_device_context_[graph_id] = device_context;
  if (func_graph_to_kernel_graph_ids_.find(func_graph) == func_graph_to_kernel_graph_ids_.end()) {
    (void)func_graph_to_kernel_graph_ids_[func_graph].emplace_back(std::vector<GraphId>{graph_id});
  } else {
    (void)func_graph_to_kernel_graph_ids_[func_graph].back().emplace_back(graph_id);
  }
}

void MSBackendBase::CompileGraphFromSegment(const FuncGraphPtr &func_graph, const GraphSegmentPtr &segment,
                                            const BackendJitConfig &backend_jit_config) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(segment);
  // Compile the normal nodes, which doesn't contain the cut node.
  if (segment->nodes_.empty()) {
    MS_LOG(INTERNAL_EXCEPTION) << "#dmsg#Runtime error info:#dmsg#The segments size is 0.";
  }
  if (!segment->is_cut_) {
    MS_EXCEPTION_IF_NULL(segment->nodes_[0]);
    MS_LOG(INFO) << "Compile normal segment, the first node: " << segment->nodes_[0]->DebugString();

    // Get the device context.
    const auto &cur_device_name = device::GetDeviceTypeByName(GetCNodeTarget(segment->nodes_[0]));
    auto device_context =
      device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext({cur_device_name, device_id_});
    MS_EXCEPTION_IF_NULL(device_context);
    device_context->Initialize();

    // Transform nodes to inputs and outputs.
    FuncGraphPtr fg;
    AnfNodePtrList inputs;
    AnfNodePtrList outputs;
    std::tie(fg, inputs, outputs) = compile::TransformSegmentToAnfGraph(segment->nodes_);

    GraphId graph_id =
      graph_compiler_->CompileGraph(segment, std::make_pair(inputs, outputs), device_context, backend_jit_config);
    auto new_fg = graph_compiler_->Fetch(graph_id);
    func_graph_to_sub_segments_[func_graph].emplace_back(graph_id);
    MS_EXCEPTION_IF_NULL(new_fg);
    CacheFuncGraphWithKernelGraphId(segment->nodes_[0]->func_graph(), graph_id, device_context);
  } else {
    // Compile the cut node.
    auto cut_node = segment->nodes_[0];
    MS_EXCEPTION_IF_NULL(cut_node);
    MS_LOG(INFO) << "Compile cut segment, the cut node: " << cut_node->DebugString();
    control_nodes_.push_back(cut_node);
    func_graph_to_sub_segments_[func_graph].emplace_back(cut_node);
    if (common::AnfAlgo::IsCallNode(cut_node) || common::AnfAlgo::CheckPrimitiveType(cut_node, prim::kPrimSwitch) ||
        common::AnfAlgo::CheckPrimitiveType(cut_node, prim::kPrimSwitchLayer)) {
      const auto &func_graph = cut_node->func_graph();
      MS_EXCEPTION_IF_NULL(func_graph);
      (void)func_graph_to_kernel_graph_ids_[func_graph].emplace_back(std::vector<GraphId>());
    }
  }
}

void MSBackendBase::TransformGraphToActorDAG(const GraphCompilerInfo &graph_compiler_info) {
  const auto &actor_set = runtime::GraphScheduler::GetInstance().Transform(graph_compiler_info);
  runtime::GraphScheduler::GetInstance().Schedule(actor_set);
}

void MSBackendBase::CompileGraph(const FuncGraphPtr &func_graph, const BackendJitConfig &backend_jit_config) {
  MS_EXCEPTION_IF_NULL(func_graph);
  uint64_t start_time = profiler::GetClockSyscnt();
  // Split graph to segments.
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto backend_name = ms_context->backend_policy();
  auto &cut_list = compile::GetMSNonlinearOps();
  auto graph_partition = std::make_shared<GraphPartition>(cut_list, backend_name);
  MS_EXCEPTION_IF_NULL(graph_partition);
  const auto &segments = graph_partition->Partition(func_graph);
  (void)profiler::CollectHostInfo(kModelNameRuntime, kEventCompileGraph, kStageGraphPartition, start_time,
                                  profiler::GetClockSyscnt(), 1);
  MS_LOG(INFO) << "Compile graph: " << func_graph->ToString() << ", Split segments size: " << segments.size();

  // Foreach the segments to compile graph.
  for (const auto &segment : segments) {
    CompileGraphFromSegment(func_graph, segment, backend_jit_config);
  }
}

void MSBackendBase::CompileSubGraph(const FuncGraphPtr &func_graph, const BackendJitConfig &backend_jit_config) {
  auto root_graph = WrapPrimitives(func_graph);
  MS_EXCEPTION_IF_NULL(root_graph);
  auto manager = root_graph->manager();
  CompileGraph(root_graph, backend_jit_config);
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  MS_EXCEPTION_IF_NULL(manager);
  const auto &sub_graphs = manager->func_graphs_used_total(root_graph);
  std::vector<FuncGraphPtr> cand_graph(sub_graphs.begin(), sub_graphs.end());
  std::sort(cand_graph.begin(), cand_graph.end(),
            [](const FuncGraphPtr &a, const FuncGraphPtr &b) { return a->ToString() < b->ToString(); });
  for (const auto &sub_graph : cand_graph) {
    MS_EXCEPTION_IF_NULL(sub_graph);
    bool skip_inline_graph =
      (sub_graph->has_flag(FUNC_GRAPH_FLAG_CELL_REUSE) && context->CellReuseLevel() == CellReuseLevel::kLazyInline) ||
      sub_graph->has_flag(kFlagSwitchInline);
    if (sub_graph != func_graph && sub_graph != nullptr && !sub_graph->has_flag(kFlagJitCallGraph) &&
        !skip_inline_graph) {
      MS_LOG(INFO) << "Compile sub graph " << sub_graph->ToString();
      CompileGraph(sub_graph, backend_jit_config);
    }
  }
}

bool MSBackendBase::CompileGraphsByKbkCache(const FuncGraphPtr &func_graph, DeviceContext *device_context) {
  MS_EXCEPTION_IF_NULL(graph_compiler_);
  try {
    MS_LOG(INFO) << "Status record: Start load backend kernel graph.";
    nlohmann::json data_json;
    if (!CheckBackendInfoValid(&data_json)) {
      return false;
    }
    if (!graph_compiler_->CompileGraphForKernelRunModeUseCache(func_graph, device_context)) {
      MS_LOG(INFO) << "Failed to load kernel graph cache.";
      return false;
    }
    if (!LoadBackendInfo(data_json)) {
      MS_LOG(INFO) << "Failed to load backend info for compile cache.";
      return false;
    }
    MS_LOG(INFO) << "Status record: End load backend kernel graph.";
    return true;
  } catch (std::exception &e) {
    MS_LOG(WARNING) << "Fail to load backend compile cache, error info:" << e.what()
                    << ".\nCurrent compile cache is invalid, please delete the current compile cache and recompile.";
    return false;
  }
}

bool MSBackendBase::DumpBackendInfo() {
  MS_LOG(DEBUG) << "Start dump control node";
  auto &context = CompileCacheContext::GetInstance();
  auto func_graph = context.FrontGraph();
  if (func_graph == nullptr) {
    MS_LOG(WARNING) << "The front graph to be cached is null, backend graph cache Missed.";
    return false;
  }

  auto cache_path = context.GetBackendGraphCachePath(func_graph);
  auto backinfo_json_path = cache_path + kControlNodeJsonSuffix;
  auto backinfo_json_real_path = Common::CreatePrefixPath(backinfo_json_path, true);
  if (!backinfo_json_real_path.has_value()) {
    MS_LOG(ERROR) << "Invalid backinfo json path:" << backinfo_json_real_path.value();
  }
  MS_LOG(DEBUG) << "Backinfo Json path:" << backinfo_json_real_path.value();
  std::ifstream backinfo_json_stream(backinfo_json_real_path.value());
  nlohmann::json backinfo_json;
  if (!backinfo_json_stream.good()) {
    MS_LOG(INFO) << "Backinfo json: " << backinfo_json_real_path.value()
                 << " does not exist. So make new Backinfo json file.";
  } else {
    if (!backinfo_json_stream.is_open()) {
      MS_LOG(ERROR) << "Load backinfo json file: " << backinfo_json_real_path.value()
                    << " error, backend graph cache missed.";
      return false;
    }
    backinfo_json_stream >> backinfo_json;
    MS_LOG(INFO) << "Load backinfo json file: " << backinfo_json_real_path.value() << " succeed.";
    backinfo_json_stream.close();
  }
  nlohmann::json new_data_json;
  std::vector<nlohmann::json> kernel_graph_to_device_context_json;

  // Save graph_id_to_device_context_;
  for (const auto &graph_id_to_device_context : graph_id_to_device_context_) {
    nlohmann::json kernel_graph_json;
    MS_EXCEPTION_IF_NULL(graph_id_to_device_context.second);
    const auto &graph_id = graph_id_to_device_context.first;
    MS_EXCEPTION_IF_NULL(graph_id_to_device_context.second);
    const auto &device_id = graph_id_to_device_context.second->device_context_key().device_id_;
    const auto &device_name =
      device::GetDeviceNameByType(graph_id_to_device_context.second->device_context_key().device_type_);
    kernel_graph_json[kGraphId] = graph_id;
    kernel_graph_json[kKernelGraphToDeviceId] = device_id;
    kernel_graph_json[kKernelGraphToDeviceName] = device_name;
    kernel_graph_to_device_context_json.push_back(kernel_graph_json);
  }
  backinfo_json[kKernelGraphNum] = kernel_graph_to_device_context_json.size();
  MS_LOG(DEBUG) << "Dump root graph number for compile cache, number:" << kernel_graph_to_device_context_json.size();

  // Collect all funcgraph valuenode.
  std::map<FuncGraphPtr, AnfNodePtr> func_graph_to_value_node;
  MS_EXCEPTION_IF_NULL(root_graph_);
  const auto &all_value_nodes = TopoSort(root_graph_->get_return(), SuccDeeperSimple);
  for (const auto &node : all_value_nodes) {
    MS_EXCEPTION_IF_NULL(node);
    if (node->isa<ValueNode>() && AnfAlgo::NodeValueIsFuncGraph(node)) {
      const auto &sub_graph = GetValueNode<FuncGraphPtr>(node);
      MS_EXCEPTION_IF_NULL(sub_graph);
      func_graph_to_value_node[sub_graph] = node;
      MS_LOG(DEBUG) << "Add funcgraph:" << sub_graph->ToString() << " to node:" << node->DebugString();
    }
  }

  // Save func_graph_to_kernel_graph_ids_.
  std::vector<nlohmann::json> func_graph_to_kernel_graph_ids_json;
  for (const auto &pair : func_graph_to_kernel_graph_ids_) {
    nlohmann::json sub_func_graph_id;
    const auto &sub_graph = pair.first;
    MS_EXCEPTION_IF_NULL(sub_graph);
    if (sub_graph == root_graph_) {
      sub_func_graph_id[kFuncGraphPtrId] = kIsRootGraph;
    } else {
      const auto &iter = func_graph_to_value_node.find(sub_graph);
      if (iter != func_graph_to_value_node.end()) {
        sub_func_graph_id[kFuncGraphPtrId] = GetUniqueNodeId(iter->second, true);
      } else {
        MS_LOG(WARNING) << "Failed to get valuenode for funcgraph:" << sub_graph->ToString();
      }
    }
    sub_func_graph_id[kSubFuncGraphId] = pair.second;
    func_graph_to_kernel_graph_ids_json.push_back(sub_func_graph_id);
  }

  // Save control node.
  std::vector<nlohmann::json> control_node_json;
  for (const auto &control_node : control_nodes_) {
    MS_EXCEPTION_IF_NULL(control_node);
    const auto &control_node_id = GetUniqueNodeId(control_node, true);
    MS_LOG(DEBUG) << " control_node: " << control_node->ToString() << " control_node_id: " << control_node_id;
    control_node_json.push_back(control_node_id);
  }

  new_data_json[kKernelGraphToDeviceContext] = kernel_graph_to_device_context_json;
  new_data_json[kFuncGraphToKernelGraphIds] = func_graph_to_kernel_graph_ids_json;
  new_data_json[kControlNodeId] = control_node_json;
  new_data_json[kDeviceName] = device_name_;
  new_data_json[kDeviceId] = device_id_;
  backinfo_json[kControlNodeCache] = new_data_json;
  MS_LOG(DEBUG) << "Dump backinfo json to " << backinfo_json_real_path.value() << ".";
  return Common::SaveStringToFile(backinfo_json_real_path.value(), backinfo_json.dump());
}

bool MSBackendBase::CheckBackendInfoValid(nlohmann::json *data_json) {
  MS_EXCEPTION_IF_NULL(data_json);
  auto &context = CompileCacheContext::GetInstance();
  auto func_graph = context.FrontGraph();
  if (func_graph == nullptr) {
    MS_LOG(ERROR) << "The frontend graph to be cached is null";
    return false;
  }
  MS_LOG(INFO) << "Start check backend compile cache valid for funcgraph:" << func_graph->ToString();
  auto cache_path = context.GetBackendGraphCachePath(func_graph);
  auto json_path = cache_path + kControlNodeJsonSuffix;
  if (json_path.empty() || cache_path.empty()) {
    MS_LOG(ERROR) << "Invalid json path, please check compile cache path.";
    return false;
  }
  auto real_json_path = FileUtils::GetRealPath(json_path.c_str());
  if (!real_json_path.has_value()) {
    MS_LOG(ERROR) << "Failed to get real path: " << json_path;
    return false;
  }
  MS_LOG(DEBUG) << "Json path: " << real_json_path.value();
  std::ifstream json_stream(real_json_path.value());
  if (!json_stream.is_open()) {
    MS_LOG(ERROR) << "Load json file: " << real_json_path.value() << " error, backend graph cache missed.";
    return false;
  }
  json_stream >> (*data_json);
  json_stream.close();
  auto control_node_json = (*data_json)[kControlNodeCache];
  // Load device context.
  if (control_node_json.contains(kKernelGraphToDeviceContext)) {
    const auto &kernel_graph_json = control_node_json[kKernelGraphToDeviceContext];
    for (const auto &kernelgraph : kernel_graph_json) {
      const auto &graph_id = kernelgraph[kGraphId].get<GraphId>();
      if (graph_compiler_->Fetch(graph_id) != nullptr) {
        MS_LOG(INFO) << "The kernel graph id:" << graph_id
                     << " in compile cache has been used, the cache cannot be load for funcgraph:"
                     << func_graph->ToString();
        return false;
      }
    }
  }
  return true;
}

bool MSBackendBase::LoadBackendInfo(const nlohmann::json &data_json) {
  MS_LOG(INFO) << "Use compile cache to load control node cache, be ware of correctness risks.";
  auto &context = CompileCacheContext::GetInstance();
  auto func_graph = context.FrontGraph();
  if (!data_json.contains(kControlNodeCache)) {
    MS_LOG(WARNING) << "No control node info in control cache json file.";
    return true;
  }

  auto control_node_json = data_json[kControlNodeCache];
  try {
    // Load device context.
    if (control_node_json.contains(kKernelGraphToDeviceContext)) {
      const auto &kernel_graph_json = control_node_json[kKernelGraphToDeviceContext];
      for (const auto &kernelgraph : kernel_graph_json) {
        const auto &graph_id = kernelgraph[kGraphId].get<GraphId>();
        const auto &graph_device_id = kernelgraph[kKernelGraphToDeviceId].get<GraphId>();
        const auto &graph_device_name =
          device::GetDeviceTypeByName(kernelgraph[kKernelGraphToDeviceName].get<std::string>());
        const auto &device_context =
          device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext({graph_device_name, graph_device_id});
        MS_EXCEPTION_IF_NULL(device_context);
        graph_id_to_device_context_[graph_id] = device_context;
      }
    }

    // Load funcgraph to kernel graph id.
    if (control_node_json.contains(kFuncGraphToKernelGraphIds)) {
      const auto &func_graph_to_kernel_graph_ids_json = control_node_json[kFuncGraphToKernelGraphIds];
      for (const auto &sub_func_graph_ids_json : func_graph_to_kernel_graph_ids_json) {
        std::vector<std::vector<GraphId>> sub_graph_ids;
        const auto &sub_func_graph_id_json = sub_func_graph_ids_json[kSubFuncGraphId];
        for (const auto &graph_ids_json : sub_func_graph_id_json) {
          std::vector<GraphId> graph_ids;
          (void)(std::transform(graph_ids_json.begin(), graph_ids_json.end(), std::back_inserter(graph_ids),
                                [](const nlohmann::json &graph_id) { return graph_id.get<GraphId>(); }));
          sub_graph_ids.push_back(graph_ids);
        }

        const auto &sub_graph_name = sub_func_graph_ids_json[kFuncGraphPtrId].get<std::string>();
        FuncGraphPtr target_graph = nullptr;
        if (sub_graph_name == kIsRootGraph) {
          target_graph = root_graph_;
        } else {
          const auto &value_node = context.FindFrontNodeByFrontName(sub_graph_name);
          if (value_node != nullptr) {
            target_graph = GetValueNode<FuncGraphPtr>(value_node);
          }
        }
        if (target_graph == nullptr) {
          MS_LOG(WARNING) << "Failed to get funcgraph by name:" << sub_graph_name;
          continue;
        }
        MS_LOG(DEBUG) << "Target graph: " << target_graph->ToString() << " sub_graph_ids: " << sub_graph_ids;
        func_graph_to_kernel_graph_ids_.insert(std::make_pair(target_graph, sub_graph_ids));
      }
    }

    // Load control node.
    const auto &control_nodes_ids_json = control_node_json[kControlNodeId];
    for (const auto &control_node_id : control_nodes_ids_json) {
      const auto &control_node = context.FindFrontNodeByFrontName(control_node_id.get<std::string>());
      if (control_node == nullptr) {
        MS_LOG(ERROR) << "Fail to find front control node by control_node_id: " << control_node_id << ".";
      } else {
        MS_LOG(DEBUG) << "control_node_id: " << control_node_id << " control_node: " << control_node->DebugString();
      }
      control_nodes_.emplace_back(control_node);
    }
    device_name_ = control_node_json[kDeviceName];
    device_id_ = control_node_json[kDeviceId].get<uint32_t>();
    MS_LOG(INFO) << "Load control node cache success for funcgraph:" << func_graph->ToString();
  } catch (std::exception &e) {
    MS_LOG(EXCEPTION) << "Fail to load control node cache for funcgraph:" << func_graph->ToString()
                      << ", error info:" << e.what();
    return false;
  }
  return true;
}

bool MSBackendBase::CacheCompileGraphs() {
  MS_EXCEPTION_IF_NULL(graph_compiler_);
  try {
    std::vector<KernelGraphPtr> graphs;
    for (const auto &pair : graph_id_to_device_context_) {
      (void)graphs.emplace_back(graph_compiler_->Fetch(pair.first));
    }
    MS_LOG(INFO) << "Status record: Start cache backend kernel graph.";
    bool is_dump_kernel_graph = graph_compiler_->CacheGraphKbk(graphs);
    bool is_dump_control_node_cache = DumpBackendInfo();
    if (!is_dump_kernel_graph || !is_dump_control_node_cache) {
      MS_LOG(INFO) << "Status record: Cache backend kernel graph "
                   << (is_dump_kernel_graph == true ? "success" : "failed") << ", dump control node cache"
                   << (is_dump_control_node_cache == true ? "success." : "failed.");
      return false;
    }
    MS_LOG(INFO) << "Status record: End cache backend kernel graph and control node info.";
    return true;
  } catch (std::exception &e) {
    MS_LOG(WARNING) << "Fail to dump backend compile cache, error info:" << e.what();
    return false;
  }
}

void MSBackendBase::ClearGraph(BackendGraphId backend_graph_id) {
  auto iter = actor_to_graph_compiler_info_.find(backend_graph_id);
  if (iter == actor_to_graph_compiler_info_.end()) {
    MS_LOG(INFO) << "Can not find graph compiler info for backend graph id: " << backend_graph_id;
    return;
  }
  MS_LOG(INFO) << "Clear compiler info for graph: " << backend_graph_id;
  const auto &compiled_graphs = iter->second->graphs_;
  auto &graphs = graph_compiler_->session_ptr()->graphs();
  for (const auto &graph : compiled_graphs) {
    (void)graphs.erase(graph->graph_id());
  }
  (void)actor_to_graph_compiler_info_.erase(iter);
}

std::shared_ptr<GraphCompilerInfo> MSBackendBase::ConstructGraphCompilerInfo(
  const FuncGraphPtr &root_graph, const BackendJitConfig &backend_jit_config) {
  MS_EXCEPTION_IF_NULL(root_graph);
  MS_EXCEPTION_IF_NULL(graph_compiler_);

  std::vector<KernelGraphPtr> graphs;
  std::vector<DeviceContext *> device_contexts;
  std::string name = "kernel_graph";
  size_t graph_index = 0;
  for (const auto &graph_id_to_context : graph_id_to_device_context_) {
    (void)graphs.emplace_back(graph_compiler_->Fetch(graph_id_to_context.first));
    (void)device_contexts.emplace_back(graph_id_to_context.second);
    if (graph_index == 0) {
      (void)name.append("_").append(std::to_string(graph_id_to_context.first));
    } else if (graph_index == graph_id_to_device_context_.size() - 1) {
      (void)name.append("-").append(std::to_string(graph_id_to_context.first));
    }
    ++graph_index;
  }
  auto parser = std::make_shared<ControlNodeParser>();
  const auto &root_output =
    common::AnfAlgo::VisitKernelWithReturnType(root_graph->output(), 0, false, {prim::kPrimTupleGetItem}).first;
  auto outputs_num = common::AnfAlgo::GetAllOutputWithIndex(root_output).size();
  runtime::KernelMapPosition outputs_order = FetchOriginOutputOrder(root_graph->output());

  std::vector<std::vector<int64_t> *> tensors_mask;
  std::vector<std::vector<tensor::TensorPtr> *> input_tensors;
  auto strategy = runtime::GraphExecutionStrategy::kPipeline;
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  if (runtime::RuntimeConf::GetInstance()->mem_optimize_level() != kOptimizeO0) {
    strategy = runtime::GraphExecutionStrategy::kPipelineWithExecutionOrder;
  }
  auto compile_func = [graph_compiler = this->graph_compiler_, backend_jit_config](
                        const GraphSegmentPtr &segment, const std::pair<AnfNodePtrList, AnfNodePtrList> &io_nodes,
                        const DeviceContext *device_context) -> KernelGraphPtr {
    auto graph_id = graph_compiler->CompileGraph(segment, io_nodes, device_context, backend_jit_config);
    return graph_compiler->Fetch(graph_id);
  };

  return std::make_shared<GraphCompilerInfo>(graphs, device_contexts, tensors_mask, input_tensors, control_nodes_,
                                             root_graph->parameters(), parser, outputs_order, outputs_num,
                                             root_graph->GetPositionalArgsCount(), name, false, strategy, compile_func,
                                             root_graph->phase());
}

void MSBackendBase::ParseControlNodes(const GraphCompilerInfo &graph_compile_info) {
  MS_EXCEPTION_IF_NULL(graph_compiler_);
  MS_EXCEPTION_IF_NULL(graph_compile_info.control_node_parser_);

  FuncGraphToKernelGraphGroup func_graph_to_kernel_graphs;
  for (const auto &func_graph_to_kernel_graph_ids : graph_compile_info.func_graph_to_kernel_graph_ids_) {
    const auto &func_graph = func_graph_to_kernel_graph_ids.first;
    for (const auto &sub_kernel_graphs_ids : func_graph_to_kernel_graph_ids.second) {
      std::vector<KernelGraphPtr> kernel_graphs;
      for (const auto &graph_id : sub_kernel_graphs_ids) {
        const auto &kernel_graph = graph_compiler_->Fetch(graph_id);
        MS_EXCEPTION_IF_NULL(kernel_graph);
        (void)kernel_graphs.emplace_back(kernel_graph);
      }
      (void)func_graph_to_kernel_graphs[func_graph].emplace_back(kernel_graphs);
    }
  }

  std::map<FuncGraphPtr, std::vector<std::variant<AnfNodePtr, KernelGraphPtr>>> func_graph_to_parts;
  for (const auto &func_graph_to_sub_segment : func_graph_to_sub_segments_) {
    const auto &func_graph = func_graph_to_sub_segment.first;
    for (const auto &sub_segment : func_graph_to_sub_segment.second) {
      if (std::holds_alternative<GraphId>(sub_segment)) {
        const auto &kernel_graph = graph_compiler_->Fetch(std::get<GraphId>(sub_segment));
        MS_EXCEPTION_IF_NULL(kernel_graph);
        (void)func_graph_to_parts[func_graph].emplace_back(kernel_graph);
      } else {
        (void)func_graph_to_parts[func_graph].emplace_back(std::get<AnfNodePtr>(sub_segment));
      }
    }
  }
  graph_compile_info.control_node_parser_->Parse(
    graph_compile_info.control_nodes_, graph_compile_info.graphs_, graph_compile_info.device_contexts_,
    graph_compile_info.root_func_graph_, func_graph_to_kernel_graphs, func_graph_to_parts);
}

bool MSBackendBase::CheckEnableGraphPipeline(const std::shared_ptr<GraphCompilerInfo> &graph_compiler_info) {
  MS_EXCEPTION_IF_NULL(graph_compiler_info);

#ifndef WITH_BACKEND
  // Disabled graph pipeline in UT scenario, because the scenario does not use the MS backend, but instead uses the VM
  // backend.
  return false;
#endif

  if (runtime::IsDisableRuntimeConfig(runtime::kRuntimeGraphPipeline)) {
    return false;
  }

  bool is_pynative_in_kbk_mode = JitPipelineCompiling();
  if (!is_pynative_in_kbk_mode) {
    return false;
  }

  bool is_pynative_bprop_graph = root_graph_->has_flag(kFlagIsPynativeBpropGraph);
  if (is_pynative_bprop_graph) {
    return false;
  }

  for (const auto &graph : graph_compiler_info->graphs_) {
    MS_EXCEPTION_IF_NULL(graph);
    if (std::any_of(graph->execution_order().begin(), graph->execution_order().end(), [&](const CNodePtr &kernel) {
          MS_EXCEPTION_IF_NULL(kernel);
          auto name = common::AnfAlgo::GetCNodeName(kernel);
          return name == "PyExecute" || name == "JoinedStr" || common::AnfAlgo::HasNodeAttr(kAttrPrimPyFunc, kernel);
        })) {
      MS_LOG(INFO) << "Disable pynative and graph pipeline for graph: " << graph_compiler_info->name_
                   << ", because the graph contains PyExecute op.";
      return false;
    }
  }

  MS_LOG(INFO) << "Enable pynative and graph pipeline for graph: " << graph_compiler_info->name_;
  return true;
}

void MSBackendBase::BindCoreForMainThread() {
  static bool is_bind_core_ = false;
  if (is_bind_core_) {
    return;
  }
  auto &bind_core_manager = runtime::ThreadBindCore::GetInstance();
  if (!bind_core_manager.is_enable_thread_bind_core_) {
    return;
  }

  const auto &core_list = bind_core_manager.get_thread_bind_core_list(runtime::kBindCoreModule::kMAIN);
  if (core_list.empty()) {
    MS_LOG(WARNING) << "Skip to bind thread core for 'main' as no available core assigned.";
  } else {
    bind_core_manager.bind_thread_core(core_list);
  }
  is_bind_core_ = true;
}

void MSBackendBase::WaitMultiStream(const GraphCompilerInfo &graph_compiler_info) {
  for (auto device_context : graph_compiler_info.device_contexts_) {
    MS_EXCEPTION_IF_NULL(device_context);
    if (device_context->device_res_manager_->single_op_multi_stream_enable()) {
      device::DeviceContextManager::GetInstance()
        .GetMultiStreamController(device_context->device_context_key().device_type_)
        ->WaitMultiStream(kDefaultStreamIndex);
    }
  }
}

std::vector<std::vector<tensor::TensorPtr>> MSBackendBase::GetRunGraphInputs(
  const GraphCompilerInfo &graph_compiler_info, const VectorRef &args) {
  runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kRuntime, runtime::ProfilerEvent::kInputProcess,
                                     graph_compiler_info.name_);
  const auto &origin_parameters = graph_compiler_info.origin_parameters_order_;
  std::vector<std::vector<tensor::TensorPtr>> input_tensor_lists;
  std::map<size_t, std::vector<tensor::TensorPtr>> flatten_values;

  for (const auto &kernel_graph : graph_compiler_info.graphs_) {
    std::vector<tensor::TensorPtr> input_tensors;
    MS_EXCEPTION_IF_NULL(kernel_graph);
    for (const auto &input_node : kernel_graph->input_nodes()) {
      auto element_pair = kernel_graph->GetElementInTupleBackendFrontIndexMap(input_node);
      if (element_pair.first) {
        PushTupleTensor(args, origin_parameters, element_pair.first, element_pair.second, &flatten_values,
                        &input_tensors);
      } else {
        const auto &front_node = kernel_graph->GetFrontAnfByBackendAnf(input_node);
        PushTensor(args, origin_parameters, front_node, &input_tensors);
      }
    }
    (void)input_tensor_lists.emplace_back(input_tensors);
  }

  // Input tensors of the control node.
  std::vector<tensor::TensorPtr> input_tensors;
  MS_EXCEPTION_IF_NULL(graph_compiler_info.control_node_parser_);
  // Get inputs of control node which come from the host actor.
  const auto &control_node_parameters = graph_compiler_info.control_node_parser_->control_node_parameters();
  for (const auto &parameter_with_index : control_node_parameters) {
    const auto &parameter = parameter_with_index.first;
    MS_EXCEPTION_IF_NULL(parameter);
    const auto &abs = parameter->abstract();
    MS_EXCEPTION_IF_NULL(abs);
    if (abs->isa<abstract::AbstractSequence>() && (!common::AnfAlgo::IsDynamicSequence(parameter))) {
      MS_LOG(DEBUG) << "Fetch input tensor for tuple parameter:" << parameter->DebugString() << " in control flow.";
      PushTupleTensor(args, origin_parameters, parameter, parameter_with_index.second, &flatten_values, &input_tensors);
    } else {
      PushTensor(args, origin_parameters, parameter, &input_tensors);
    }
  }
  (void)input_tensor_lists.emplace_back(input_tensors);

  return input_tensor_lists;
}

BaseRef MSBackendBase::ConstructOutputByAbstract(const abstract::AbstractBasePtr &abstract,
                                                 const std::vector<tensor::TensorPtr> &output_tensors,
                                                 size_t *output_position, std::vector<tensor::TensorPtr> *tuple_tensors,
                                                 const std::vector<TypePtr> &output_types) {
  MS_EXCEPTION_IF_NULL(abstract);
  MS_EXCEPTION_IF_NULL(output_position);
  MS_EXCEPTION_IF_NULL(tuple_tensors);

  size_t outputs_num = common::AnfAlgo::GetOutputNumByAbstract(abstract);
  if (*output_position + outputs_num > output_tensors.size()) {
    MS_LOG(INTERNAL_EXCEPTION) << "#dmsg#Runtime error info:#dmsg#The output position is out of range: "
                               << *output_position << " need:" << outputs_num << " total:" << output_tensors.size();
  }

  if (!abstract->isa<abstract::AbstractSequence>()) {
    if (IsTupleOutputOfAnyType(abstract, output_tensors[*output_position])) {
      MS_LOG(DEBUG) << "Any output for position:" << *output_position;
      VectorRef outputs;
      ConstructOutputByTupleTensor(
        output_tensors[*output_position],
        output_tensors[*output_position]->base_shape_ptr()->cast<abstract::SequenceShapePtr>(), &outputs, tuple_tensors,
        output_types[*output_position]);
      (*output_position)++;
      std::vector<ValuePtr> values;

      (void)std::transform(outputs.begin(), outputs.end(), std::back_inserter(values),
                           [](const auto &output) { return utils::cast<ValuePtr>(output); });
      return std::make_shared<ValueList>(values);
    }

    (*output_position)++;
    return output_tensors[(*output_position) - 1];
  }

  VectorRef outputs;
  const auto &tuple_abstract = abstract->cast<abstract::AbstractSequencePtr>();
  MS_EXCEPTION_IF_NULL(tuple_abstract);
  // Dynamic len tuple.
  if (tuple_abstract->dynamic_len()) {
    auto &output_tensor = output_tensors[*output_position];
    MS_EXCEPTION_IF_NULL(output_tensor);
    auto &tensor_shape = output_tensor->base_shape_ptr();
    // Restore the tuple output by the tensor of tuple.
    if ((tensor_shape != nullptr) && tensor_shape->isa<abstract::SequenceShape>()) {
      ConstructOutputByTupleTensor(output_tensor, tensor_shape->cast<abstract::SequenceShapePtr>(), &outputs,
                                   tuple_tensors, output_types[*output_position]);
      (*output_position)++;
      return outputs;
    }
  }

  const auto &sub_abstracts = tuple_abstract->elements();
  for (const auto &sub_abstract : sub_abstracts) {
    MS_EXCEPTION_IF_NULL(sub_abstract);
    outputs.emplace_back(
      ConstructOutputByAbstract(sub_abstract, output_tensors, output_position, tuple_tensors, output_types));
  }
  return outputs;
}

void MSBackendBase::ConstructOutputByTupleTensor(tensor::TensorPtr output_tensor,
                                                 const abstract::SequenceShapePtr &tensor_shape, VectorRef *outputs,
                                                 std::vector<tensor::TensorPtr> *tuple_tensors,
                                                 const TypePtr &output_type) const {
  MS_EXCEPTION_IF_NULL(output_tensor);
  MS_EXCEPTION_IF_NULL(tensor_shape);
  MS_EXCEPTION_IF_NULL(outputs);
  MS_EXCEPTION_IF_NULL(tuple_tensors);
  MS_LOG(DEBUG) << "Tensor shape:" << tensor_shape->ToString();
  // If outputs an empty sequence return an empty sequence value.
  if (tensor_shape->size() == 0) {
    if (tensor_shape->isa<abstract::TupleShape>()) {
      outputs->emplace_back(std::make_shared<ValueTuple>(std::vector<ValuePtr>()));
    } else {
      outputs->emplace_back(std::make_shared<ValueList>(std::vector<ValuePtr>()));
    }
    return;
  }
  // No need split multi tensors when the tuple size is not greater than 1.
  if (tensor_shape->size() <= 1) {
    outputs->emplace_back(output_tensor);
    return;
  }

  auto tensor_type_id = output_tensor->data_type();
  auto device_tensor = output_tensor->device_address();
  MS_EXCEPTION_IF_NULL(device_tensor);
  auto tensor_device_ptr = device_tensor->GetMutablePtr();
  auto tensor_device_size = device_tensor->GetSize();
  MS_EXCEPTION_IF_NULL(tensor_device_ptr);
  auto device_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(
    {device_tensor->GetDeviceType(), device_tensor->device_id()});
  MS_EXCEPTION_IF_NULL(device_context);
  MS_EXCEPTION_IF_NULL(device_context->device_res_manager_);

  MS_EXCEPTION_IF_NULL(output_type);
  TuplePtr output_tuple_type = output_type->cast<TuplePtr>();
  MS_EXCEPTION_IF_NULL(output_tuple_type);
  const auto &element_types = output_tuple_type->elements();
  if (tensor_shape->size() != element_types.size()) {
    MS_LOG(EXCEPTION) << "The tensor shape size[" << tensor_shape->size() << "] is not equal to output element size["
                      << element_types.size() << "].";
  }

  // Split the tensor of tuple to tensors.
  (void)tuple_tensors->emplace_back(output_tensor);
  size_t copy_offset_size = 0;
  for (size_t i = 0; i < tensor_shape->size(); ++i) {
    // Create split tensor.
    auto split_tensor_shape = BaseShapeToShape((*tensor_shape)[i]);
    auto split_tensor_size = SizeOf(split_tensor_shape) * GetTypeByte(TypeIdToType(tensor_type_id));
    auto split_tensor = tensor::from_spec(tensor_type_id, split_tensor_shape, device::DeviceType::kNone);

    auto kernel_tensor = AnfAlgo::CreateKernelTensor(
      nullptr, split_tensor_size, output_tensor->format(), output_tensor->data_type(), split_tensor_shape,
      device::GetDeviceNameByType(device_context->device_context_key().device_type_),
      device_context->device_context_key().device_id_);
    kernel_tensor->SetType(element_types[i]);
    kernel_tensor->SetShape((*tensor_shape)[i]);
    kernel_tensor->set_stream_id(device_tensor->stream_id());
    auto split_device_tensor = kernel_tensor->device_address();
    MS_EXCEPTION_IF_NULL(split_device_tensor);
    MS_LOG(DEBUG) << "Create kernel tensor:" << kernel_tensor->ToString();
    // Copy data from origin tensor to the split tensor.
    device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddTask, "ConstructOutputByTupleTensor",
                                                   "ConstructOutputByTupleTensor", "", false);
    device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddMemInfo, "ConstructOutputByTupleTensor",
                                                   memory::mem_pool::MemType::kOther, split_device_tensor->GetSize(),
                                                   split_device_tensor.get());
    if (!device_context->device_res_manager_->AllocateMemory(split_device_tensor.get())) {
      MS_LOG(EXCEPTION) << "#umsg#Memory not enough:#umsg#Device(id:" << device_context->device_context_key().device_id_
                        << ") memory isn't enough and alloc failed, kernel name: Split tuple outputs, alloc size: "
                        << split_device_tensor->GetSize() << "B.";
    } else {
      static std::string name = "Alloc memory";
      kernel_tensor->IncreaseNewRefCount(name);
    }
    if (copy_offset_size + split_tensor_size > tensor_device_size) {
      MS_LOG(INTERNAL_EXCEPTION) << "#dmsg#Runtime error info:#dmsg#The copy size is out of range, copy size:"
                                 << split_tensor_size << ", copy offset size:" << copy_offset_size
                                 << ", total size:" << tensor_device_size;
    }
    if (!device_context->device_res_manager_->SyncAllStreams() ||
        !device_context->device_res_manager_->Copy(
          split_device_tensor->GetMutablePtr(), AddressOffset(tensor_device_ptr, copy_offset_size), split_tensor_size,
          device::CopyType::kD2D, split_device_tensor->stream_id())) {
      MS_LOG(INTERNAL_EXCEPTION) << "#dmsg#Runtime error info:#dmsg#Sync device to device failed, device type:"
                                 << split_device_tensor->GetDeviceType() << ", copy size:" << split_tensor_size
                                 << ", output node: Split tuple outputs.";
    }
    copy_offset_size += split_tensor_size;

    // Fill the outputs.
    split_tensor->set_device_address(split_device_tensor);
    split_tensor->set_storage_info(kernel_tensor->tensor_storage_info());
    outputs->emplace_back(split_tensor);
  }
}

void MSBackendBase::ConstructOutputs(const AnfNodePtr &output_node,
                                     const std::vector<tensor::TensorPtr> &output_tensors, size_t *output_position,
                                     VectorRef *outputs, std::vector<tensor::TensorPtr> *tuple_tensors,
                                     const std::vector<TypePtr> &output_types) {
  MS_EXCEPTION_IF_NULL(output_node);
  MS_EXCEPTION_IF_NULL(outputs);
  MS_EXCEPTION_IF_NULL(output_position);
  MS_EXCEPTION_IF_NULL(tuple_tensors);
  static const PrimitiveSet expand_prims{
    prim::kPrimMakeTuple,
    prim::kPrimMakeCSRTensor,
    prim::kPrimMakeCOOTensor,
    prim::kPrimMakeRowTensor,
  };
  MS_LOG(DEBUG) << "output node:" << output_node->DebugString();
  // If outputs an empty sequence return an empty sequence value.
  if (IsEmptySequence(output_node, output_tensors, output_position)) {
    if (output_node->abstract()->isa<abstract::AbstractTuple>()) {
      outputs->emplace_back(std::make_shared<ValueTuple>(std::vector<ValuePtr>()));
    } else {
      outputs->emplace_back(std::make_shared<ValueList>(std::vector<ValuePtr>()));
    }
    ++(*output_position);
    return;
  }

  // The MakeTuple/MakeSaprse node need expand and recurse.
  if (IsOneOfPrimitiveCNode(output_node, expand_prims)) {
    auto make_tuple = output_node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(make_tuple);
    VectorRef make_tuple_output;
    for (size_t i = 1; i < make_tuple->size(); i++) {
      ConstructOutputs(make_tuple->input(i), output_tensors, output_position, &make_tuple_output, tuple_tensors,
                       output_types);
    }
    outputs->emplace_back(std::move(make_tuple_output));
    return;
  }

  // The depend node need get the real node.
  if (common::AnfAlgo::CheckPrimitiveType(output_node, prim::kPrimDepend)) {
    auto depend_node = output_node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(depend_node);
    ConstructOutputs(depend_node->input(kRealInputIndexInDepend), output_tensors, output_position, outputs,
                     tuple_tensors, output_types);
    return;
  }

  auto outputs_num = AnfAlgo::GetOutputElementNum(output_node);
  // The value node uses the value to be output, to avoid the host memory of value free due to value node destruction.
  if (output_node->isa<ValueNode>()) {
    auto value_node = output_node->cast<ValueNodePtr>();
    MS_EXCEPTION_IF_NULL(value_node);
    auto value = value_node->value();
    MS_EXCEPTION_IF_NULL(value);
    if (value->isa<ValueSequence>()) {
      outputs->emplace_back(value);
      (*output_position) += CountValueNum(value->cast<ValueSequencePtr>());
    } else if (outputs_num != 0) {
      outputs->emplace_back(value);
      (*output_position) += outputs_num;
    }
    // The empty value node return the empty VectorRef.
    return;
  }

  if (common::AnfAlgo::IsCallNode(output_node) ||
      (output_node->abstract() != nullptr && output_node->abstract()->isa<abstract::AbstractSequence>())) {
    auto abstract = output_node->abstract();
    MS_EXCEPTION_IF_NULL(abstract);
    outputs->emplace_back(
      ConstructOutputByAbstract(abstract, output_tensors, output_position, tuple_tensors, output_types));
    return;
  }

  auto &output_abstract = output_node->abstract();
  MS_EXCEPTION_IF_NULL(output_abstract);
  // Wrap output to VectorRef if the output is tuple.
  MS_LOG(DEBUG) << "output abstract:" << output_abstract->ToString();
  if (output_abstract->isa<abstract::AbstractSequence>()) {
    VectorRef output_tuple;
    for (size_t i = 0; i < outputs_num; ++i) {
      MS_LOG(DEBUG) << "output index:" << i;
      if (*output_position >= output_tensors.size()) {
        MS_LOG(INTERNAL_EXCEPTION) << "#dmsg#Runtime error info:#dmsg#The output position is out of range: "
                                   << *output_position;
      }
      auto &output_tensor = output_tensors[*output_position];
      MS_EXCEPTION_IF_NULL(output_tensor);
      auto &tensor_shape = output_tensor->base_shape_ptr();
      // Restore the tuple output by the tensor of tuple.
      if ((tensor_shape != nullptr) && tensor_shape->isa<abstract::SequenceShape>()) {
        ConstructOutputByTupleTensor(output_tensor, tensor_shape->cast<abstract::SequenceShapePtr>(), &output_tuple,
                                     tuple_tensors, output_types[*output_position]);
      } else {
        output_tuple.emplace_back(output_tensor);
      }
      ++(*output_position);
    }
    outputs->emplace_back(std::move(output_tuple));
  } else {
    for (size_t i = 0; i < outputs_num; ++i) {
      if (*output_position >= output_tensors.size()) {
        MS_LOG(INTERNAL_EXCEPTION) << "#dmsg#Runtime error info:#dmsg#The output position is out of range: "
                                   << *output_position;
      }
      outputs->emplace_back(output_tensors[*output_position]);
      ++(*output_position);
    }
  }
}

void MSBackendBase::ConstructOutputs(runtime::ActorSet *actor_set, VectorRef *outputs, const FuncGraphPtr &root_graph,
                                     bool enable_graph_pipeline) {
  MS_EXCEPTION_IF_NULL(actor_set);
  MS_EXCEPTION_IF_NULL(outputs);
  MS_EXCEPTION_IF_NULL(root_graph);

  MS_EXCEPTION_IF_NULL(actor_set->output_actor_);
  // Update device address for output node of graph.
  // Summary processing will use the output device address, so must be after the summary processing.
  actor_set->output_actor_->UpdateOutputDeviceAddress();

  if (enable_graph_pipeline) {
    MS_LOG(DEBUG) << "Enable pynative graph pipeline for actor set: " << actor_set->name_
                  << ", early stop ConstructOutputs.";
    return;
  }

  // Fetch outputs.
  auto &output_tensors = actor_set->output_actor_->outputs();
  auto &output_types = actor_set->output_actor_->output_types();
  if (!output_tensors.empty()) {
    size_t output_position = 0;
    std::vector<tensor::TensorPtr> tuple_tensors;
    ConstructOutputs(root_graph->output(), output_tensors, &output_position, outputs, &tuple_tensors, output_types);

    // The tensor may be repeated, so it needs to be set null last.
    for (auto &tuple_tensor : tuple_tensors) {
      MS_EXCEPTION_IF_NULL(tuple_tensor);
      tuple_tensor->set_device_address(nullptr);
      tuple_tensor->set_storage_info(nullptr);
    }
  }
}

void MSBackendBase::CreateTensorArgs(const VectorRef &args, const GraphCompilerInfo &) {
  for (const auto &arg : args) {
    if (utils::isa<tensor::TensorPtr>(arg)) {
      auto value = utils::cast<tensor::TensorPtr>(arg);
    } else if (utils::isa<stub::TensorNode>(arg)) {
      auto tensor_stub = utils::cast<std::shared_ptr<stub::TensorNode>>(arg);
      MS_EXCEPTION_IF_NULL(tensor_stub);
      auto value = tensor_stub->WaitValue();
      MS_EXCEPTION_IF_NULL(value);
      auto tensor = value->cast<tensor::TensorPtr>();
      MS_EXCEPTION_IF_NULL(tensor);
    } else if (utils::isa<ValuePtr>(arg)) {
      auto value = utils::cast<ValuePtr>(arg);
      MS_EXCEPTION_IF_NULL(value);
      if (!value->isa<ValueSequence>()) {
        return;
      }
      auto value_tuple = value->cast<ValueSequencePtr>();
      MS_EXCEPTION_IF_NULL(value_tuple);
      auto tuple_value = value_tuple->value();
      for (const auto &v : tuple_value) {
        if (!v->isa<tensor::Tensor>()) {
          continue;
        }
        auto t = v->cast<tensor::TensorPtr>();
      }
    }
  }
}

#ifdef ENABLE_DEBUGGER
void MSBackendBase::SetDebuggerInit() const {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  constexpr char kInit[] = "DebuggerInit";
  static auto init_callback =
    callback::CommonCallback::GetInstance().GetCallback<void, const uint32_t, const std::string &>(kInit);
  if (init_callback) {
    init_callback(device_id_, ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET));
  } else {
    MS_LOG(WARNING) << "Failed to get DebuggerInit, data dump function may not work.";
  }
}
#endif

MSBackendBase::MSBackendBase() {
  graph_compiler_ = std::make_shared<GraphCompiler>();
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  device_name_ = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  const auto &device_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(
    {device::GetDeviceTypeByName(device_name_), device_id});
  MS_EXCEPTION_IF_NULL(device_context);
  uint64_t start_time = profiler::GetClockSyscnt();
  device_context->Initialize();
  (void)profiler::CollectHostInfo(kModelNameRuntime, kEventDeviceInit, kStageDeviceInit, start_time,
                                  profiler::GetClockSyscnt(), 1);
  device_id_ = device_context->device_context_key().device_id_;
#ifdef ENABLE_DEBUGGER
  SetDebuggerInit();
#endif
  runtime::GraphScheduler::GetInstance().Initialize();
}

FuncGraphPtr MSBackendBase::WrapPrimitives(const FuncGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  FuncGraphManagerPtr manager_ptr = graph->manager();
  MS_EXCEPTION_IF_NULL(manager_ptr);
  std::map<PrimTypePair, FuncGraphPtr> prim_graphs;
  const auto &get_prim_graph = [&prim_graphs](const PrimitivePtr &prim, const AbstractFunctionPtr &type) {
    PrimTypePair prim_type = std::make_pair(prim, type);
    if (prim_graphs.end() == prim_graphs.find(prim_type)) {
      FuncGraphPtr g = std::make_shared<FuncGraph>();
      std::vector<AnfNodePtr> args;
      ValueNodePtr prim_ct = NewValueNode(prim);
      MS_EXCEPTION_IF_NULL(prim_ct);
      prim_ct->set_abstract(type);
      args.push_back(prim_ct);
      MS_EXCEPTION_IF_NULL(type);
      TypedPrimitiveAbstractClosurePtr tp = dyn_cast<abstract::TypedPrimitiveAbstractClosure>(type->GetUnique());
      if (tp == nullptr) {
        MS_LOG(INTERNAL_EXCEPTION) << "Not TypedPrimitiveAbstractClosure, but got " << type->GetUnique()->ToString();
      }
      MS_EXCEPTION_IF_NULL(g);
      for (const auto &t : tp->args_abs_list()) {
        ParameterPtr p = g->add_parameter();
        p->set_abstract(t);
        args.push_back(p);
      }
      AnfNodePtr out = g->NewCNode(args);
      out->set_abstract(tp->output());
      g->set_output(out);
      prim_graphs[prim_type] = g;
    }

    return prim_graphs[prim_type];
  };

  FuncGraphTransaction tr = manager_ptr->Transact();
  auto &fgs = manager_ptr->func_graphs();
  TraverseGraphMap(manager_ptr, &tr, fgs, get_prim_graph);
  tr.Commit();

  return graph;
}
BackendGraphId CompileSegment(const GraphSegmentPtr &segment, const AnfNodePtrList &inputs,
                              const AnfNodePtrList &outputs, MSBackendBase *backend) {
  MS_EXCEPTION_IF_NULL(segment);
  MS_EXCEPTION_IF_NULL(backend);
  MS_LOG(DEBUG) << "Compile segment";
  auto func_graph = std::make_shared<FuncGraph>();
  std::unordered_map<AnfNodePtr, AnfNodePtr> node_map;
  for (const auto &input : inputs) {
    MS_EXCEPTION_IF_NULL(input);
    MS_EXCEPTION_IF_NULL(input->abstract());
    if (HasAbstractMonad(input)) {
      auto u = NewValueNode(kUMonad);
      u->set_abstract(kUMonad->ToAbstract());
      node_map[input] = u;
      continue;
    }
    auto parameter = func_graph->add_parameter();
    parameter->set_abstract(input->abstract()->Clone());
    node_map[input] = parameter;
  }
  for (const auto &node : segment->nodes_) {
    MS_EXCEPTION_IF_NULL(node);
    if (!node->isa<CNode>()) {
      MS_LOG(EXCEPTION) << "Invalid node:" << node->DebugString() << " in segment.";
    }
    const auto &cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    MS_EXCEPTION_IF_NULL(cnode->abstract());
    const auto &prim = common::AnfAlgo::GetCNodePrimitive(cnode);
    const auto &new_prim = std::make_shared<Primitive>(prim->name());
    new_prim->SetAttrs(prim->attrs());
    new_prim->EraseAttr("split_op");
    AnfNodePtrList new_inputs{NewValueNode(new_prim)};
    for (size_t i = 1; i < cnode->size(); ++i) {
      MS_EXCEPTION_IF_NULL(cnode->input(i));
      if (cnode->input(i)->isa<ValueNode>()) {
        new_inputs.emplace_back(cnode->input(i));
        continue;
      }
      const auto &iter = node_map.find(cnode->input(i));
      if (iter == node_map.end()) {
        MS_LOG(EXCEPTION) << "Failed to get input node:" << cnode->input(i)->DebugString();
      }
      new_inputs.emplace_back(iter->second);
    }
    const auto &new_cnode = func_graph->NewCNode(new_inputs);
    new_cnode->set_abstract(cnode->abstract()->Clone());
    new_cnode->set_attrs(cnode->attrs());
    new_cnode->set_primal_attrs(cnode->primal_attrs());
    new_cnode->CloneUserData(cnode);
    new_cnode->set_scope(cnode->scope());
    new_cnode->set_fullname_with_scope(cnode->fullname_with_scope());
    node_map[cnode] = new_cnode;
  }
  AnfNodePtrList make_tuple_inputs{NewValueNode(std::make_shared<Primitive>(prim::kPrimMakeTuple->name()))};
  std::vector<abstract::AbstractBasePtr> make_tuple_abs_inputs;
  std::vector<AnfNodePtr> output_monads;
  for (const auto &output : outputs) {
    MS_EXCEPTION_IF_NULL(output);
    const auto &iter = node_map.find(output);
    if (iter == node_map.end()) {
      MS_LOG(EXCEPTION) << "Failed to get new node by:" << output->DebugString();
    }
    if (HasAbstractMonad(output)) {
      output_monads.emplace_back(iter->second);
      continue;
    }
    make_tuple_inputs.emplace_back(iter->second);
    make_tuple_abs_inputs.emplace_back(iter->second->abstract());
  }
  const auto &make_tuple = func_graph->NewCNode(make_tuple_inputs);
  make_tuple->set_abstract(std::make_shared<abstract::AbstractTuple>(make_tuple_abs_inputs));
  auto last_node = make_tuple;
  for (const auto &output_monad : output_monads) {
    std::vector<AnfNodePtr> depend_inputs{NewValueNode(std::make_shared<Primitive>(prim::kPrimDepend->name())),
                                          last_node, output_monad};
    auto depend_node = func_graph->NewCNode(depend_inputs);
    depend_node->set_abstract(last_node->abstract());
    last_node = depend_node;
  }
  std::vector<AnfNodePtr> return_inputs{NewValueNode(prim::kPrimReturn), make_tuple};
  auto return_node = func_graph->NewCNode(return_inputs);
  func_graph->set_return(return_node);
  auto manager = MakeManager({func_graph});
  MS_EXCEPTION_IF_NULL(manager);
  manager->AddFuncGraph(func_graph);
  func_graph->set_manager(manager);
  DumpIR("my_func_graph", func_graph, true);
  const auto &backend_jit_config = backend::BackendJitConfig::ParseBackendJitConfig();
  return backend->Build(func_graph, backend_jit_config);
}

std::vector<GraphFragmentPtr> MSBackendBase::Split(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  auto root_graph = WrapPrimitives(func_graph);
  MS_EXCEPTION_IF_NULL(root_graph);
  const auto &sub_graphs = root_graph->func_graphs_used_total();
  if (!sub_graphs.empty()) {
    MS_LOG(EXCEPTION) << "Not support graph split for multi-graph.";
  }
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto backend_name = ms_context->backend_policy();
  auto cut_list = compile::GetMSNonlinearOps();
  auto graph_partition = std::make_shared<GraphPartition>(cut_list, backend_name);
  MS_EXCEPTION_IF_NULL(graph_partition);
  const auto &segments = graph_partition->Partition(root_graph);
  MS_LOG(INFO) << "Segment size:" << segments.size();
  if (segments.size() <= 1) {
    MS_LOG(WARNING) << "No kernel in graph:" << func_graph->ToString() << ".";
    return {};
  }
  std::vector<GraphFragmentPtr> fragments;
  std::unordered_map<AnfNodePtr, std::pair<int, std::string>> output_nodes;
  for (const auto &node : root_graph->parameters()) {
    MS_EXCEPTION_IF_NULL(node);
    const auto &parameter = node->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(parameter);
    output_nodes[parameter] = std::make_pair(-1, parameter->name());
  }
  for (size_t i = 0; i < segments.size() - 1; ++i) {
    FuncGraphPtr fg;
    AnfNodePtrList inputs;
    AnfNodePtrList outputs;
    const auto &segment = segments[i];
    MS_EXCEPTION_IF_NULL(segment);
    std::tie(fg, inputs, outputs) = compile::TransformSegmentToAnfGraph(segment->nodes_);
    if (segment->nodes_.empty()) {
      MS_LOG(EXCEPTION) << "Invalid segment index:" << i;
    }
    size_t output_index = 0;
    for (const auto &output : outputs) {
      if (HasAbstractMonad(output)) {
        continue;
      }
      output_nodes[output] = std::make_pair(SizeToInt(i), std::to_string(output_index++));
    }
    MS_EXCEPTION_IF_NULL(segment->nodes_[0]);
    bool is_graph = true;
    std::string func_key = "";
    const auto &cnode = segment->nodes_[0]->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    auto primitive = common::AnfAlgo::GetCNodePrimitive(cnode);
    MS_EXCEPTION_IF_NULL(primitive);
    if (primitive->HasAttr("split_op")) {
      auto split_op_value = primitive->GetAttr("split_op");
      MS_EXCEPTION_IF_NULL(split_op_value);
      if (GetValue<bool>(split_op_value)) {
        is_graph = false;
        if (!primitive->HasAttr("func_id")) {
          MS_LOG(EXCEPTION) << "Failed to get func_id by split op:" << cnode->DebugString();
        }
        auto func_key_value = primitive->GetAttr("func_id");
        MS_EXCEPTION_IF_NULL(func_key_value);
        func_key = GetValue<std::string>(func_key_value);
      }
    }

    auto runner = [this, segment, inputs, outputs](GraphFragment *frag, const VectorRef &input_list,
                                                   VectorRef *output_list) {
      MS_EXCEPTION_IF_NULL(frag);
      size_t input_size = IntToSize(
        std::count_if(inputs.begin(), inputs.end(), [](const auto &node) { return !HasAbstractMonad(node); }));
      if (input_list.size() != input_size) {
        MS_LOG(EXCEPTION) << "Invalid input size:" << input_list.size() << " need:" << input_size
                          << " for fragment id:" << frag->id_;
      }
      if (frag->graph_id_ == UINT32_MAX) {
        frag->graph_id_ = CompileSegment(segment, inputs, outputs, this);
      }
      MS_LOG(DEBUG) << "inputs :" << input_list.ToString();
      Run(frag->graph_id_, input_list, output_list);
      return kRunningSuccess;
    };
    std::vector<std::pair<int, std::string>> args_list;
    for (const auto &input : inputs) {
      MS_EXCEPTION_IF_NULL(input);
      if (HasAbstractMonad(input)) {
        continue;
      }
      const auto &iter = output_nodes.find(input);
      if (iter == output_nodes.end()) {
        MS_LOG(EXCEPTION) << "Failed to get input node:" << input->DebugString() << " for segment:" << i;
      }
      args_list.emplace_back(iter->second);
    }
    fragments.emplace_back(std::make_shared<GraphFragment>(i, is_graph, func_key, args_list, runner));
  }
  return fragments;
}
BackendGraphId MSBackendBase::Build(const FuncGraphPtr &func_graph, const BackendJitConfig &backend_jit_config) {
  WaitTaskFinish();
  MS_EXCEPTION_IF_NULL(graph_compiler_);
  MS_EXCEPTION_IF_NULL(func_graph);
  // Clear the resource of last graph.
  ClearGraphBuildMember();

  MS_LOG(INFO) << "Status record: start compile function graph: " << func_graph->ToString();
  uint64_t start_time = profiler::GetClockSyscnt();
  PROF_START(compile_backend_graph);

  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  device_name_ = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  const auto &device_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(
    {device::GetDeviceTypeByName(device_name_), device_id_});
  MS_EXCEPTION_IF_NULL(device_context);
  device_context->Initialize();
  device_context->device_res_manager_->BindDeviceToCurrentThread(false);

  auto root_graph = WrapPrimitives(func_graph);
  MS_EXCEPTION_IF_NULL(root_graph);

  PROF_START(WaitAllCommInit);
  (void)distributed::collective::CollectiveManager::instance()->WaitAllCommInitDone();
  PROF_END(WaitAllCommInit);
  UnifyMindIR(root_graph);
  root_graph_ = root_graph;
  auto origin_output_node = root_graph->output();

  // Register a summary callback function, which is called in the final stages of summary.
  graph_compiler_->RegisterSummaryCallBackFunc();

  // Compile root graph.
  bool load_compile_cache = false;
  if (EnableKBKCompileCache(func_graph, device_context->GetDeviceType())) {
    PROF_START(Load_backend_compile_cache);
    load_compile_cache = CompileGraphsByKbkCache(func_graph, device_context);
    MS_LOG(INFO) << "Load backend cache and compile graphs " << (load_compile_cache == true ? "success." : "failed.");
    PROF_END(Load_backend_compile_cache);
  }
  if (!load_compile_cache) {
    PROF_START(CompileSubGraph);
    bool is_dynamic_graph = common::AnfAlgo::IsDynamicShapeFuncGraph(func_graph);
    MS_LOG(INFO) << "Start to compile graph: " << func_graph->ToString() << ", is_dynamic_graph: " << is_dynamic_graph;
    ProcessNotSupportCnode(func_graph, device_context->GetDeviceType(), mindspore::device::DeviceType::kCPU);
    BuildSymbolEngine(func_graph);
    CompileSubGraph(func_graph, backend_jit_config);
    PROF_END(CompileSubGraph);
  }

  if (ExportCompileCacheKBK(func_graph, device_context->GetDeviceType()) && !load_compile_cache) {
    PROF_START(save_backend_compile_cache);
    bool is_success = CacheCompileGraphs();
    PROF_END(save_backend_compile_cache);
    if (!is_success) {
      MS_LOG(WARNING) << "Failed to cache backend graph.";
    }
  }

  // Construct the graph compiler info.
  auto graph_compiler_info = ConstructGraphCompilerInfo(root_graph, backend_jit_config);
  MS_LOG(INFO) << "Status record: construct the graph compiler info.";
  MS_EXCEPTION_IF_NULL(graph_compiler_info);
  graph_compiler_info->root_func_graph_ = root_graph_;
  graph_compiler_info->enable_graph_pipeline_ = CheckEnableGraphPipeline(graph_compiler_info);
  graph_compiler_info->func_graph_to_kernel_graph_ids_ = func_graph_to_kernel_graph_ids_;
  // Use kernel graph, which output maybe change by backed pass, so backup output
  graph_compiler_info->origin_output_node_ = origin_output_node;
  graph_compiler_info->is_pynative_mode_ = JitPipelineCompiling();

  if ((!graph_compiler_info->graphs_.empty() || graph_compiler_info->control_nodes_.size() > 1)) {
    MS_LOG(DEBUG) << "Start transform";
    PROF_START(GraphScheduler);
    // Transform graph to actor DAG, and schedule the actor DAG.
    ParseControlNodes(*graph_compiler_info);
    TransformGraphToActorDAG(*graph_compiler_info);
    PROF_END(GraphScheduler);
  }

  (void)actor_to_graph_compiler_info_.emplace(graph_compiler_info->id_, graph_compiler_info);

  for (const auto &graph_id_to_context : graph_id_to_device_context_) {
    auto context = graph_id_to_context.second;
    device::DeviceContextManager::GetInstance()
      .GetMultiStreamController(context->device_context_key().device_type_)
      ->Refresh();
  }

  PROF_END(compile_backend_graph);
  (void)profiler::CollectHostInfo(kModelNameRuntime, kEventCompileGraph, kStageCompileGraphs, start_time,
                                  profiler::GetClockSyscnt(), 1);
  MS_LOG(INFO) << "Status record: end compile function graph: " << func_graph->ToString()
               << ", produce actor: " << graph_compiler_info->name_
               << ", backend_graph_id: " << graph_compiler_info->id_;

  // Clear the temp members.
  ClearGraphBuildMember();

  return graph_compiler_info->id_;
}

namespace {
bool IsMemoryLeak(const KernelTensorPtr &kernel_tensor) {
  if (kernel_tensor == nullptr || kernel_tensor->new_ref_count() == SIZE_MAX) {
    return false;
  }
  if (kernel_tensor->device_ptr() != nullptr) {
    return true;
  }
  if (kernel_tensor->new_ref_count() != 0) {
    MS_LOG(INFO) << "Ref count is not 0 after run graph for device address:" << kernel_tensor->ToString();
  }
  return false;
}

void CheckMemoryLeak(const runtime::AbstractActorPtr &actor, const KernelTensorPtr &kernel_tensor) {
  if (kernel_tensor == nullptr) {
    return;
  }
  if (IsMemoryLeak(kernel_tensor)) {
    MS_LOG(EXCEPTION) << "Memory leak detected in actor:" << actor->GetAID()
                      << " output kernel tensor:" << kernel_tensor->ToString();
  }
}

void CheckMemoryLeakV2(const runtime::KernelRunnerPtr &actor, const KernelTensorPtr &kernel_tensor) {
  if (kernel_tensor == nullptr) {
    return;
  }
  if (IsMemoryLeak(kernel_tensor)) {
    MS_LOG(EXCEPTION) << "Memory leak detected in actor:" << actor->GetAID()
                      << " output kernel tensor:" << kernel_tensor->ToString();
  }
}

void StrictCheckForDeviceAddress(const runtime::ActorSet *actor_set) {
  if (!runtime::IsEnableRuntimeConfig(runtime::kRuntimeNewRefCount)) {
    return;
  }
  MS_EXCEPTION_IF_NULL(actor_set);
  for (const auto &kernel_actor : actor_set->kernel_actors_) {
    MS_EXCEPTION_IF_NULL(kernel_actor);
    MS_LOG(DEBUG) << "Check for kernel actor:" << kernel_actor->GetAID();
    for (size_t i = 0; i < kernel_actor->output_kernel_tensors().size(); ++i) {
      const auto &kernel_tensor = kernel_actor->output_kernel_tensors()[i];
      CheckMemoryLeak(kernel_actor, kernel_tensor);
    }
    for (size_t i = 0; i < kernel_actor->workspace_kernel_tensors().size(); ++i) {
      const auto &kernel_tensor = kernel_actor->workspace_kernel_tensors()[i];
      CheckMemoryLeak(kernel_actor, kernel_tensor);
    }
  }

  for (const auto &copy_actor : actor_set->copy_actors_) {
    MS_EXCEPTION_IF_NULL(copy_actor);
    const auto &kernel_tensor = copy_actor->output();
    CheckMemoryLeak(copy_actor, kernel_tensor);
  }

  for (const auto &super_kernel_actor : actor_set->super_kernel_actors_) {
    MS_EXCEPTION_IF_NULL(super_kernel_actor);
    MS_LOG(DEBUG) << "Check for super kernel actor:" << super_kernel_actor->GetAID();
    for (const auto &kernel_actor : super_kernel_actor->kernel_actors()) {
      MS_EXCEPTION_IF_NULL(kernel_actor);
      MS_LOG(DEBUG) << "Check output for actor:" << kernel_actor->GetAID();
      for (size_t i = 0; i < kernel_actor->output_kernel_tensors().size(); ++i) {
        const auto &kernel_tensor = kernel_actor->output_kernel_tensors()[i];
        CheckMemoryLeakV2(kernel_actor, kernel_tensor);
      }
      MS_LOG(DEBUG) << "Check workspace for actor:" << kernel_actor->GetAID();
      for (size_t i = 0; i < kernel_actor->workspace_kernel_tensors().size(); ++i) {
        const auto &kernel_tensor = kernel_actor->workspace_kernel_tensors()[i];
        CheckMemoryLeakV2(kernel_actor, kernel_tensor);
      }
    }
  }

  MS_LOG(DEBUG) << "Check parameter store";
  auto graph_parameter_store = runtime::ParameterStore::GetInstance().GetGraphParameterStore();
  if (graph_parameter_store != nullptr) {
    for (const auto &pair : graph_parameter_store->GetAll()) {
      for (const auto &sub_pair : pair) {
        const auto &kernel_tensor = sub_pair.first;
        if (kernel_tensor == nullptr) {
          continue;
        }
        if (IsMemoryLeak(kernel_tensor)) {
          MS_LOG(EXCEPTION) << "Memory leak detected in parameter store for kernel tensor:"
                            << kernel_tensor->ToString();
        }
      }
    }
  }
  MS_LOG(DEBUG) << "Check end";
}
}  // namespace

RunningStatus MSBackendBase::Run(BackendGraphId graph_id, const VectorRef &inputs, VectorRef *outputs) {
  MSTX_RANGE_GUARD("MSBackendRun", nullptr, profiler::MSTX_DOMAIN_MODEL_PREPARATION);
  runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kRuntime, runtime::ProfilerEvent::kBackendGraphRunInner,
                                     std::to_string(graph_id), true);
  // Main thread bind to core.
  BindCoreForMainThread();

  // Fetch the graph compiler info.
  const auto &graph_iter = actor_to_graph_compiler_info_.find(graph_id);
  if (graph_iter == actor_to_graph_compiler_info_.end()) {
    MS_LOG(INTERNAL_EXCEPTION) << "#dmsg#Runtime error info:#dmsg#Can't find the graph compiler info.";
  }
  MS_EXCEPTION_IF_NULL(graph_iter->second);
  const auto &graph_compiler_info = *(graph_iter->second);
  auto root_graph = graph_compiler_info.root_func_graph_;
  MS_EXCEPTION_IF_NULL(root_graph);

  if (common::AnfAlgo::IsGraphOutputValueNodeOrParameter(root_graph->output(), inputs, outputs)) {
    return kRunningSuccess;
  }

  const auto &context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  if (context_ptr->get_param<bool>(MS_CTX_PRECOMPILE_ONLY) || common::GetEnv("MS_DEV_PRECOMPILE_ONLY") == "1") {
    MS_LOG(INFO) << "PrecompileOnly, stop run graph";
    return kRunningSuccess;
  }

  // Open abstract_lock for dynamic_shape
  AnfUtils::OpenAbstractLock();

  // Run in the pynative mode.
  MS_EXCEPTION_IF_NULL(outputs);
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  // There will be more than one kernel graph in heterogeneous scenario in a jit of PyNative Mode.
  if (graph_compiler_info.is_pynative_mode_) {
    RunGraphByCondition(graph_id, graph_compiler_info, inputs, outputs);
    return kRunningSuccess;
  }

  WaitTaskFinish();
  WaitMultiStream(graph_compiler_info);
  MS_LOG(INFO) << "Status record: start run actor: " << graph_compiler_info.name_;
  uint64_t start_time_ = profiler::GetClockSyscnt();
  std::vector<std::vector<tensor::TensorPtr>> input_tensors;
  // Release python gil.
  mindspore::ScopedLongRunning long_running;
  // Run actor DAG.
  const auto &actor_set = runtime::GraphScheduler::GetInstance().Fetch(graph_id);
  MS_EXCEPTION_IF_NULL(actor_set);
  static auto disable_pre_build_comm = runtime::IsDisableRuntimeConfig(runtime::kRuntimePreBuildCommKernel);
  if (!disable_pre_build_comm) {
    PROF_START(PreLaunchCommKernel);
    runtime::PreLaunchComm::GetInstance().CachePreLaunchOrder(graph_id);
    runtime::PreLaunchComm::GetInstance().PreLaunchCommKernel(actor_set);
    PROF_END(PreLaunchCommKernel);
  }
  runtime::GraphScheduler::GetInstance().Run(actor_set, input_tensors, inputs);

  {
    uint64_t start_time = 0;
    PROFILER_START(start_time);
    MS_EXCEPTION_IF_NULL(graph_compiler_);
    graph_compiler_->Summary(graph_compiler_info.graphs_);
    ConstructOutputs(actor_set, outputs, root_graph, graph_compiler_info.enable_graph_pipeline_);
    actor_set->output_actor_->FreeSummaryNodeMem();
    runtime::GraphScheduler::GetInstance().ClearActorData(actor_set);
    if (ms_context->IsEnableInferBoost()) {
      auto &llm_manager = LLMManager::GetInstance();
      llm_manager.reset_graph_inputs();
    }
    PROFILER_END(start_time, runtime::ProfilerModule::kKernel, runtime::ProfilerEvent::kOutputProcess, actor_set->name_,
                 false);
  }
  StrictCheckForDeviceAddress(actor_set);
  // Close abstract_lock for dynamic_shape
  AnfUtils::CloseAbstractLock();
  (void)profiler::CollectHostInfo(kModelNameRuntime, kEventRunGraph, kStageRunGraph, start_time_,
                                  profiler::GetClockSyscnt(), 1);
  MS_LOG(INFO) << "Status record: end run actor: " << graph_id;
  return kRunningSuccess;
}
}  // namespace ms_backend
}  // namespace backend
}  // namespace mindspore
