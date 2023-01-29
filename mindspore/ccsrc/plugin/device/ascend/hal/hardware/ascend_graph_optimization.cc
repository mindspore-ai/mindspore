/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "plugin/device/ascend/hal/hardware/ascend_graph_optimization.h"

#include <set>
#include <unordered_set>
#include <string>
#include <memory>
#include <utility>
#include "backend/common/optimizer/common_backend_optimization.h"
#include "plugin/device/ascend/optimizer/ascend_backend_optimization.h"
#include "plugin/device/ascend/optimizer/ascend_comm_op_reuse.h"
#include "plugin/device/ascend/hal/common/ascend_utils.h"
#include "common/graph_kernel/adapter/graph_kernel_optimization.h"
#include "common/graph_kernel/adapter/expander.h"
#include "common/graph_kernel/value_graph_binder.h"
#include "plugin/device/ascend/hal/hardware/ascend_auto_monad.h"
#include "common/graph_kernel/graph_kernel_flags.h"
#include "plugin/device/ascend/hal/device/kernel_select_ascend.h"
#include "plugin/device/ascend/hal/device/kernel_adjust.h"

#ifndef ENABLE_SECURITY
#include "include/common/debug/anf_ir_dump.h"
#include "include/common/debug/dump_proto.h"
#include "ir/func_graph_cloner.h"
#endif

namespace mindspore {
namespace device {
namespace ascend {
using AscendAutoMonad = mindspore::session::AscendAutoMonad;

namespace {
const std::unordered_set<std::string> kDefaultFormatAclOps = {kAddNOpName};
const size_t DEFAULT_MAX_COMM_OP_REUSE_NUM = 1000;

void RemoveUnusedValueNode(const KernelGraphPtr &graph) {
  auto m = graph->manager();
  auto node_users = m->node_users();
  mindspore::HashSet<ValueNodePtr> unused_value_nodes;
  for (auto &value_node : graph->graph_value_nodes()) {
    if (node_users.find(value_node) == node_users.end()) {
      unused_value_nodes.insert(value_node);
    }
  }
  for (auto &value_node : unused_value_nodes) {
    graph->RemoveNodeFromGraph(value_node);
  }
}

void ReplaceTbeKernelWithAclKernel(const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  auto exec_nodes = graph->execution_order();
  for (const auto &node : exec_nodes) {
    if (AnfAlgo::GetKernelType(node) == TBE_KERNEL) {
      auto new_builder =
        std::make_shared<kernel::KernelBuildInfo::KernelBuildInfoBuilder>(AnfAlgo::GetSelectKernelBuildInfo(node));
      MS_EXCEPTION_IF_NULL(new_builder);
      new_builder->SetKernelType(ACL_KERNEL);
      MS_LOG(INFO) << "SUCCESS SET ACL KERNEL FOR" << node->DebugString();
      AnfAlgo::SetSelectKernelBuildInfo(new_builder->Build(), node.get());
    }
  }
}

bool SetDefaultFormatForSpecialAclOp(const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  auto exec_nodes = graph->execution_order();
  bool need_change_format = false;
  for (auto &node : exec_nodes) {
    if (kDefaultFormatAclOps.count(common::AnfAlgo::GetCNodeName(node))) {
      need_change_format = true;
      auto new_builder =
        std::make_shared<kernel::KernelBuildInfo::KernelBuildInfoBuilder>(AnfAlgo::GetSelectKernelBuildInfo(node));
      MS_EXCEPTION_IF_NULL(new_builder);
      auto inputs_format = AnfAlgo::GetAllInputFormats(node);
      auto outputs_format = AnfAlgo::GetAllOutputFormats(node);
      new_builder->SetInputsFormat(std::vector<std::string>(inputs_format.size(), kOpFormat_DEFAULT));
      new_builder->SetOutputsFormat(std::vector<std::string>(outputs_format.size(), kOpFormat_DEFAULT));
      AnfAlgo::SetSelectKernelBuildInfo(new_builder->Build(), node.get());
    }
  }
  return need_change_format;
}

AnfNodePtr DoInline(const FuncGraphPtr &func_graph, const FuncGraphPtr &target_func_graph,
                    const AnfNodePtrList &func_graph_args, const ScopePtr &scope) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(target_func_graph);
  Cloner cloner({}, false);
  if (scope != nullptr) {
    cloner.set_scope(scope);
  }
  cloner.AddClone(func_graph, target_func_graph, func_graph_args, kInline);
  auto node_list = TopoSort(func_graph->output());
  for (auto &ori_node : node_list) {
    if (ori_node->isa<Parameter>()) {
      continue;
    }
    auto new_node = cloner[ori_node];
    auto kernel_info = dynamic_cast<device::KernelInfo *>(new_node->kernel_info());
    // deep copy kernel info
    if (kernel_info != nullptr && new_node->kernel_info()->has_build_info()) {
      // some check
      MS_EXCEPTION_IF_CHECK_FAIL(kernel_info->MutableKernelMod() == nullptr,
                                 "Inline ERROR: " + ori_node->DebugString() + ", kernel mod is not nullptr");
      MS_EXCEPTION_IF_CHECK_FAIL(kernel_info->output_address_list().empty(),
                                 "Inline ERROR: " + ori_node->DebugString() + ", output_address_list is not empty");
      MS_EXCEPTION_IF_CHECK_FAIL(kernel_info->workspace_address_list().empty(),
                                 "Inline ERROR: " + ori_node->DebugString() + ", workspace_address_list is not empty");

      auto new_kernel_info = std::make_shared<device::KernelInfo>();
      auto builder =
        std::make_shared<kernel::KernelBuildInfo::KernelBuildInfoBuilder>(AnfAlgo::GetSelectKernelBuildInfo(new_node));
      new_kernel_info->set_select_kernel_build_info(builder->Build());
      new_kernel_info->set_graph_id(kernel_info->graph_id());
      new_kernel_info->set_feature_map_flag(kernel_info->is_feature_map());
      new_kernel_info->set_ref_map(false, kernel_info->out_in_ref_map());
      new_node->set_kernel_info(new_kernel_info);
    }
    if (ori_node->isa<CNode>()) {
      auto ori_cnode = ori_node->cast<CNodePtr>();
      if (common::AnfAlgo::HasNodeAttr(kAttrIsUBFusionOp, ori_cnode) &&
          common::AnfAlgo::GetNodeAttr<bool>(ori_node, kAttrIsUBFusionOp)) {
        // already done fusion compile
        auto ori_full_name = ori_cnode->fullname_with_scope();
        common::AnfAlgo::SetNodeAttr(kAttrOriFusionName, MakeValue(ori_full_name), new_node);
      }
      common::AnfAlgo::SetNodeAttr(kAttrNeedInline, MakeValue(ori_node->fullname_with_scope()), new_node);
      common::AnfAlgo::SetNodeAttr(kAttrPreKernelGraph, MakeValue(func_graph), new_node);
    }
  }
  return cloner[func_graph->output()];
}
}  // namespace

void AscendGraphOptimization::Reset() {
  MS_LOG(INFO) << "Clear Ascend Graph Optimization Resource.";
  memo_.clear();
  graph_manager_->Clear();
}

void AscendGraphOptimization::InlineSubGraph(const KernelGraphPtr &graph) {
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
#ifdef ENABLE_DUMP_IR
  bool save_graphs = context_ptr->CanDump(kAdvanced);
  if (save_graphs) {
    std::string file_name = "hwopt_d_before_inline_graph_" + std::to_string(graph->graph_id()) + ".ir";
    DumpIR(file_name, graph, true, kWholeStack);
  }
#endif
  auto kernel_cnodes = graph->execution_order();
  for (auto &kernel_cnode : kernel_cnodes) {
    if (common::AnfAlgo::CheckPrimitiveType(kernel_cnode, prim::kPrimCallInline)) {
      auto sub_graph = common::AnfAlgo::GetNodeAttr<KernelGraphPtr>(kernel_cnode, kAttrKernelGraph);
      MS_LOG(INFO) << "InlineSubGraph: " << kernel_cnode->DebugString() << ", sub graph: " << sub_graph->graph_id()
                   << ", need inline: " << sub_graph->need_inline();
      auto main_graph = kernel_cnode->func_graph();
      auto mng = main_graph->manager();
      AnfNodePtrList inp(kernel_cnode->inputs().begin() + 1, kernel_cnode->inputs().end());
      auto out = DoInline(sub_graph, main_graph, inp, kernel_cnode->input(0)->scope());
      (void)mng->Replace(kernel_cnode, out);
    }
  }
  memo_.clear();
  opt::AscendAfterInlineOptimization(graph);
  memo_.clear();
#ifdef ENABLE_DUMP_IR
  if (save_graphs) {
    std::string file_name = "hwopt_d_after_inline_graph_" + std::to_string(graph->graph_id()) + ".ir";
    DumpIR(file_name, graph, true, kWholeStack);
  }
#endif
}

void AscendGraphOptimization::OptimizeGraph(const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_LOG(INFO) << "Status record: start optimize graph. graph id: " << graph->graph_id();

  // empty graph dont entry to backend
  if (graph->execution_order().empty()) {
    MS_LOG(INFO) << graph->ToString() << " is empty graph.";
    AnfAlgo::InsertMakeTupleForOutput(NOT_NULL(graph));
    graph->set_executable(false);
    MS_LOG(INFO) << "Status record: end optimize graph. graph id: " << graph->graph_id();
  }

  OptimizeGraphWithoutDeviceInfo(graph);
  SelectKernel(graph);
  OptimizeGraphWithDeviceInfo(graph);

  // inline func before gen execution order
  InlineSubGraph(graph);
  OptimizeExecutionOrder(graph);
  PostOptimization(graph);

  memo_.clear();
  // clear and reset graph_manager_ after optimization
  graph_manager_ = MakeManager();
  MS_LOG(INFO) << "Status record: end optimize graph. graph id: " << graph->graph_id();
}

void AscendGraphOptimization::OptimizeSingleOpGraph(const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);

  if (graph->has_flag(kAttrMutableKernel)) {
    AclOpOptimize(graph);
  } else {
    opt::RunOpAscendBackendIRFusionOptimization(graph);
    SelectKernel(graph);
    opt::RunOpAscendBackendOptimization(graph);
  }

  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  // Cannot Hide nop node in PyNative mode.
  // If there is more than one node in the graph,
  // and one of the nodes is a nop node, the node will be hidden.
  // The DAG of Actors will be invalid(lack an input edge).

  // must clear memo_ which holds kernel graph after using AscendGraphOptimization class.
  memo_.clear();
}

void AscendGraphOptimization::AclOpOptimize(const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  opt::RunOpAscendBackendIRFusionOptimization(graph);

  auto nodes = graph->execution_order();
  for (auto &node : nodes) {
    AnfAlgo::SetDynamicAttrToPrim(common::AnfAlgo::GetCNodePrimitive(node));
  }

  SelectKernel(graph);

  bool need_change_format = SetDefaultFormatForSpecialAclOp(graph);
  bool has_aicpu = std::any_of(nodes.begin(), nodes.end(),
                               [](const CNodePtr &node) { return AnfAlgo::GetKernelType(node) == AICPU_KERNEL; });
  if (has_aicpu || need_change_format) {
    // Insert Cast and TransData.
    opt::RunOpAscendBackendOptimization(graph);
  } else {
    // Only insert Cast.
    opt::AscendMixPrecision(graph);
  }

  ReplaceTbeKernelWithAclKernel(graph);
}

void AscendGraphOptimization::OptimizeGraphWithoutDeviceInfo(const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  CheckControlFlowDynamicShape(graph);
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  CommOpReuse(graph);
  if (context_ptr->get_param<bool>(MS_CTX_IS_MULTI_GRAPH_SINK)) {
    HandleControlFlow(NOT_NULL(graph));
  }

  // add all graphs to manager first, so that don't have to make new manager in following passes.
  memo_.clear();
  AddGraphToManager(NOT_NULL(graph), NOT_NULL(graph_manager_));
  memo_.clear();
  IRFusionOptimization(graph);
}

void AscendGraphOptimization::OptimizeGraphWithDeviceInfo(const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  memo_.clear();
  HardWareOptimization(graph);
  memo_.clear();
  // copy child graph ref output map to father graph ref output map
  UpdateRefOutputMap(graph);
  AnfAlgo::InsertMakeTupleForOutput(NOT_NULL(graph));
  RemoveUnusedValueNode(graph);
}

void AscendGraphOptimization::OptimizeExecutionOrder(const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_LOG(INFO) << "Status record: start optimize execution order. graph id: " << graph->graph_id();
  PROF_START(optimize_execution_order);
  // root root_graph validate,include generate execute order and so on
  RootGraphExecutorValidate(NOT_NULL(graph));

#ifdef ENABLE_DUMP_IR
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  if (context->CanDump(kAdvanced)) {
    DumpIRProto(graph, "before_removeNop_" + std::to_string(graph->graph_id()));
  }
#endif

  if (graph->is_graph_run_mode()) {
    opt::HideNopNode(graph.get());
  }

  auto execution_order = graph->execution_order();
  graph->EnableRuntimeCache();
  common::AnfAlgo::ReorderExecList(NOT_NULL(&execution_order));
  graph->DisableRuntimeCache();
  graph->set_execution_order(execution_order);

  device::KernelAdjust::GetInstance().InsertOverflowCheckOperations(NOT_NULL(graph));

#ifdef ENABLE_DUMP_IR
  if (context->CanDump(kAdvanced)) {
    DumpIR("after_adjust_kernel.ir", graph);
  }
#endif
  PROF_END(optimize_execution_order);
  MS_LOG(INFO) << "Status record: end optimize execution order. graph id: " << graph->graph_id();
}

void AscendGraphOptimization::PostOptimization(const KernelGraphPtr &graph) const {
  MS_LOG(INFO) << "Status record: start post optimization. graph id: " << graph->graph_id();
  graph->SetOptimizerFlag();
  MS_LOG(INFO) << "Status record: end post optimization. graph id: " << graph->graph_id();
}

void AscendGraphOptimization::CommOpReuse(const KernelGraphPtr &graph) const {
  MS_EXCEPTION_IF_NULL(graph);
  auto max_comm_op_reuse_num_env = common::GetEnv("MS_COMM_COMPILER_OPT");
  if (!graph->is_graph_run_mode() || max_comm_op_reuse_num_env.empty()) {
    return;
  }
  int64_t max_comm_op_reuse_num_l = -1;
  TRY_AND_CATCH_WITH_EXCEPTION((max_comm_op_reuse_num_l = std::stol(max_comm_op_reuse_num_env)),
                               "Invalid MS_COMM_COMPILER_OPT value! It should be -1 or a positive integer.");
  size_t max_comm_op_reuse_num;
  if (max_comm_op_reuse_num_l == -1) {
    max_comm_op_reuse_num = DEFAULT_MAX_COMM_OP_REUSE_NUM;
  } else if (max_comm_op_reuse_num_l > 0) {
    max_comm_op_reuse_num = LongToSize(max_comm_op_reuse_num_l);
  } else {
    MS_LOG(WARNING) << "MS_COMM_COMPILER_OPT should be -1 or a positive integer but set to " << max_comm_op_reuse_num_l
                    << ". Comm subgraph reuse is disabled.";
    return;
  }
  MS_LOG(INFO) << "MAX_COMM_OP_REUSE_NUM: " << max_comm_op_reuse_num;
  MS_LOG(INFO) << "Status record: start comm op reuse. graph id: " << graph->graph_id();
  opt::AscendCommOpReuse comm_io_reuse(graph, max_comm_op_reuse_num);
  comm_io_reuse.Run();
  MS_LOG(INFO) << "Status record: end comm op reuse. graph id: " << graph->graph_id();

#ifdef ENABLE_DUMP_IR
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  if (context->CanDump(kAdvanced)) {
    std::string file_name = "hwopt_comm_reuse_after_graph_" + std::to_string(graph->graph_id()) + ".ir";
    DumpIR(file_name, graph);
  }
#endif
}

void AscendGraphOptimization::HardWareOptimization(const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_LOG(INFO) << "Status record: start hardware optimize. graph id: " << graph->graph_id();
  if (memo_.find(graph) != memo_.end()) {
    return;
  }
  (void)memo_.insert(graph);
  for (auto &child_graph : graph->child_graph_order()) {
    HardWareOptimization(child_graph.lock());
  }
  opt::AscendBackendOptimization(graph);
  opt::CommonFinalOptimization(graph);
  if (graphkernel::GraphKernelFlags::GetInstance().IsEnableGraphKernel()) {
    graphkernel::GraphKernelOptimize(graph);
    graph->SetExecOrderByDefault();
  }
  MS_LOG(INFO) << "Status record: end hardware optimize. graph id: " << graph->graph_id();
}

void AscendGraphOptimization::AddGraphToManager(const NotNull<KernelGraphPtr> graph,
                                                const NotNull<FuncGraphManagerPtr> manager, bool is_root) {
  if (memo_.find(graph) != memo_.end()) {
    return;
  }
  (void)memo_.insert(graph.get());
  manager->AddFuncGraph(graph.get(), is_root);

  for (auto &child_graph : graph->child_graph_order()) {
    AddGraphToManager(NOT_NULL(child_graph.lock()), manager, false);
  }
}

void AscendGraphOptimization::IRFusionOptimization(const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  if (memo_.find(graph) != memo_.end()) {
    return;
  }
  (void)memo_.insert(graph);

  opt::AscendBackendIRFusionOptimization(graph);
  for (auto &child_graph : graph->child_graph_order()) {
    IRFusionOptimization(NOT_NULL(child_graph.lock()));
  }
}

void AscendGraphOptimization::HandleControlFlow(const NotNull<KernelGraphPtr> graph) const {
  MS_LOG(INFO) << "Status record: start handle control flow. graph id: " << graph->graph_id();
  PROF_START(handle_control_flow);
  AscendAutoMonad auto_monad(graph);
  auto_monad.Run();
  PROF_END(handle_control_flow);
  MS_LOG(INFO) << "Status record: end handle control flow. graph id: " << graph->graph_id();
}

void AscendGraphOptimization::RootGraphExecutorValidate(const NotNull<KernelGraphPtr> graph) {
  MS_LOG(INFO) << "Status record: start graph executor validate. graph id: " << graph->graph_id();
  AscendAutoMonad auto_monad(graph);
  auto_monad.GenerateExecuteOrder();
  MS_LOG(INFO) << "Status record: end graph executor validate. graph id: " << graph->graph_id();
}

void AscendGraphOptimization::RecurseSelectKernelInfo(const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  if (memo_.find(graph) != memo_.end()) {
    return;
  }
  (void)memo_.insert(graph);
  for (auto &child_graph : graph->child_graph_order()) {
    if (child_graph.lock()->need_inline()) {
      RecurseSelectKernelInfo(child_graph.lock());
    }
  }
#ifdef ENABLE_DUMP_IR
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  if (context->CanDump(kIntroductory)) {
    std::string file_name = "select_kernel_before_graph_" + std::to_string(graph->graph_id()) + ".ir";
    DumpIR(file_name, graph, true, kTopStack);
  }
#endif
  MS_LOG(INFO) << "Status record: start select kernel info. graph id: " << graph->graph_id();
  SetOperatorInfo(graph);
  MS_LOG(INFO) << "Status record: end select kernel info. graph id: " << graph->graph_id();
#ifdef ENABLE_DUMP_IR
  if (context->CanDump(kIntroductory)) {
    std::string file_name = "select_kernel_after_graph_" + std::to_string(graph->graph_id()) + ".ir";
    DumpIR(file_name, graph);
  }
#endif
  for (auto &child_graph : graph->child_graph_order()) {
    if (!child_graph.lock()->need_inline()) {
      RecurseSelectKernelInfo(child_graph.lock());
    }
  }
}

void AscendGraphOptimization::SelectKernel(const KernelGraphPtr &graph) {
  MS_LOG(INFO) << "Status record: start select kernel info. graph id: " << graph->graph_id();
  PROF_START(select_kernel);
  raise_precision_count_ = 0;
  reduce_precision_count_ = 0;
  memo_.clear();
  RecurseSelectKernelInfo(graph);
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (ms_context->get_param<int>(MS_CTX_EXECUTION_MODE) == kGraphMode) {
    if (raise_precision_count_ > 0) {
      MS_LOG(INFO) << "There are " << raise_precision_count_
                   << " node/nodes used raise precision to selected the kernel!";
    }
    if (reduce_precision_count_ > 0) {
      MS_LOG(INFO) << "There are " << reduce_precision_count_
                   << " node/nodes used reduce precision to selected the kernel!";
    }
  }
  PROF_END(select_kernel);
  MS_LOG(INFO) << "Status record: end select kernel info. graph id: " << graph->graph_id();
}

void AscendGraphOptimization::UpdateRefOutputMap(const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  if (memo_.find(graph) != memo_.end()) {
    return;
  }
  (void)memo_.insert(graph);

  for (auto &child_graph : graph->child_graph_order()) {
    auto child_graph_ptr = child_graph.lock();
    MS_EXCEPTION_IF_NULL(child_graph_ptr);
    UpdateRefOutputMap(NOT_NULL(child_graph_ptr));
    // copy ref map to final graph
    auto child_ref_map = child_graph_ptr->GetRefMap();
    for (auto item = child_ref_map.cbegin(); item != child_ref_map.cend(); ++item) {
      if (graph->IsInRefOutputMap(item->first)) {
        MS_LOG(DEBUG) << "The ref pair <" << item->first.first->DebugString() << ", " << item->first.second
                      << "> is already in " << graph->ToString();
        continue;
      }
      graph->AddRefCorrespondPairs(item->first, item->second);
    }
  }
}

void AscendGraphOptimization::UnifyMindIR(const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_LOG(INFO) << "Status record: start unify mindir. graph id: " << graph->graph_id();
  PROF_START(unify_mindir);
  opt::CommonUnifyMindIR(graph);
  opt::AscendUnifyMindIR(graph);
  PROF_END(unify_mindir);
  // must clear memo_ which holds kernelgraph after using AscendGraphOptimization class.
  memo_.clear();
  MS_LOG(INFO) << "Status record: end unify mindir. graph id: " << graph->graph_id();
}

void AscendGraphOptimization::OpAdaptation(const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_LOG(INFO) << "Status record: start op adaptation. graph id: " << graph->graph_id();
  PROF_START(op_adaptation);
  opt::AscendOpAdaptation(graph);
  PROF_END(op_adaptation);
  // must clear memo_ which holds kernel graph after using AscendGraphOptimization class.
  memo_.clear();
  MS_LOG(INFO) << "Status record: end op adaptation. graph id: " << graph->graph_id();
}

void AscendGraphOptimization::SetOperatorInfo(const KernelGraphPtr &graph) {
  auto mng = graph->manager();
  if (mng == nullptr) {
    mng = Manage(graph, true);
    graph->set_manager(mng);
  }
  bool do_expand = false;
  auto &node_list = graph->execution_order();
  for (auto &node : node_list) {
    auto [status, msg, etype] = device::ascend::SelectKernelInfoWithMsg(node);
    common::AnfAlgo::EraseNodeAttr(kAttrPynativeNextOpName, node);
    common::AnfAlgo::EraseNodeAttr(kAttrPynativeNextIndex, node);
    if (status != device::ascend::kNoMatched) {
      if (status == device::ascend::kStatusRaisePrecision) {
        raise_precision_count_++;
      } else if (status == device::ascend::kStatusReducePrecision) {
        reduce_precision_count_++;
      }
      MS_LOG(DEBUG) << "Select ApplyKernel: " << node->DebugString();
    } else {
      auto f = [](const CNodePtr &n) {
        // change the kernel to static-shape kernel name. it's a temporary solution.
        if (IsOneOfPrimitiveCNode(n, {prim::kPrimReduceSum, prim::kPrimReduceMin, prim::kPrimReduceMax})) {
          auto primitive = GetCNodePrimitive(n);
          auto new_prim = std::make_shared<Primitive>(primitive->name() + "D", primitive->attrs());
          n->set_input(0, NewValueNode(new_prim));
        }
        auto res = device::ascend::SelectKernelInfoWithMsg(n);
        constexpr int one = 1;
        return std::get<one>(res).empty();
      };
      auto cnode = graphkernel::TryExpandCNode(node, f);
      if (cnode == nullptr) {
        if (graph == nullptr || graph->is_from_single_op() || graph->is_graph_run_mode()) {
          MS_EXCEPTION(etype) << msg;
        }
        MS_LOG(INFO) << "Try to use backoff CPU kernel, node:" << node->fullname_with_scope();
        std::pair<std::string, ExceptionType> failure_info = std::make_pair(msg, etype);
        AnfAlgo::SetKernelSelectBackoffInfo(node, failure_info);
        continue;
      }
      (void)mng->Replace(node, cnode);
      MS_LOG(INFO) << msg << " but expand success.";
      auto expand_fg = GetCNodeFuncGraph(cnode);
      graphkernel::InlineExpandFuncGraph(cnode, expand_fg);
      do_expand = true;
    }
  }
  if (do_expand) {
    graphkernel::BindValueToGraph().Run(graph);
    graph->SetExecOrderByDefault();
  }
}
void AscendGraphOptimization::GetAllGraphs(const KernelGraphPtr &root_graph) {
  if (memo_.find(root_graph) != memo_.end()) {
    return;
  }
  (void)memo_.insert(root_graph);
  auto node_list = TopoSort(root_graph->get_return());
  for (auto node : node_list) {
    if (!IsValueNode<FuncGraph>(node)) {
      continue;
    }
    auto child_graph = GetValueNode<FuncGraphPtr>(node);
    MS_EXCEPTION_IF_NULL(child_graph);
    auto child_kernel_graph = child_graph->cast<KernelGraphPtr>();
    MS_EXCEPT_CHECK_NULL(child_kernel_graph);
    GetAllGraphs(child_kernel_graph);
  }
}

void AscendGraphOptimization::CheckControlFlowDynamicShape(const KernelGraphPtr &root_graph) {
  MS_EXCEPT_CHECK_NULL(root_graph);
  memo_.clear();
  GetAllGraphs(root_graph);
  if (memo_.size() <= 1) {
    memo_.clear();
    return;
  }

  for (auto &graph : memo_) {
    if (graph->is_dynamic_shape()) {
      MS_LOG(EXCEPTION) << "Dynamic shape is not supported with control flow(loop control statements and conditions "
                           "control statements).";
    }
  }
  memo_.clear();
}
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
