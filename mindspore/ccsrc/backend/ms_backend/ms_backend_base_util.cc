/**
 * Copyright 2025-2026 Huawei Technologies Co., Ltd
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

#include "backend/ms_backend/ms_backend_base_util.h"

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
#include "include/utils/ir_dump/anf_ir_dump_interface.h"
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
  auto &&op_name = common::AnfAlgo::GetCNodeName(cnode);
  auto iter = kOpListToTupleNames.find(op_name);
  if (iter != kOpListToTupleNames.end()) {
    common::AnfAlgo::SetNodeAttr(kAttrOpAdaptationProcessed, MakeValue(true), cnode);
    cnode->set_input(0, mindspore::NewValueNode(std::make_shared<Primitive>(iter->second)));
    cnode->set_fullname_with_scope("");
    MS_LOG(INFO) << "Rename op from " << iter->first << " to " << iter->second << " for op "
                 << cnode->fullname_with_scope() << ", debug name:" << cnode->DebugString();
    op_name = iter->second;
  }

  if (op_name == mindspore::kTupleGetItemOpName && NeedConvertToRealTupleGetItem(cnode)) {
    common::AnfAlgo::SetNodeAttr(kAttrOpAdaptationProcessed, MakeValue(true), cnode);
    cnode->set_input(0, mindspore::NewValueNode(std::make_shared<Primitive>(mindspore::kRealTupleGetItemOpName)));
    cnode->set_fullname_with_scope("");
    MS_LOG(INFO) << "Rename op from TupleGetItem to RealTupleGetItem for op " << cnode->fullname_with_scope()
                 << ", debug name:" << cnode->DebugString();
  }

  if (op_name == mindspore::kMakeTupleOpName && common::AnfAlgo::IsDynamicSequence(cnode)) {
    common::AnfAlgo::SetNodeAttr(kAttrOpAdaptationProcessed, MakeValue(true), cnode);
    cnode->set_input(0, mindspore::NewValueNode(std::make_shared<Primitive>(mindspore::kRealMakeTupleOpName)));
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

  static const bool is_enable_ge = (context->backend_policy() == "ge");
  if (!is_enable_ge || !context->IsKByKExecutorMode() || graph->func_graphs_used_total().empty()) {
    MS_LOG(INFO) << "Disable switch inline, executor mode:" << context->IsKByKExecutorMode();
    return false;
  }

  MS_EXCEPTION_IF_NULL(graph);
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

std::string GetUniqueNodeId(const AnfNodePtr &node, bool must_have_unique_name) {
  MS_EXCEPTION_IF_NULL(node);
  const auto &name = node->user_data<std::string>(kUniqueCacheName);
  if (must_have_unique_name && name == nullptr) {
    MS_LOG(EXCEPTION) << "The node " << node->DebugString()
                      << " has not unique name, indicating that it is not exported to mindir.";
  }
  return name != nullptr ? *name : "node is nullptr";
}

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
  if (position >= args.size()) {
    MS_LOG(DEBUG) << "Position out of args range, position value is " << position << " and args size is " << args.size()
                  << ".";
    (void)input_tensors->emplace_back(nullptr);
    return;
  }

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
        auto node = use.first->cast<CNodePtr>();
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
}  // namespace ms_backend
}  // namespace backend
}  // namespace mindspore
