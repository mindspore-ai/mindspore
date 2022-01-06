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

#include "runtime/hardware/ascend/ascend_graph_optimization.h"
#include <set>
#include "backend/optimizer/common/common_backend_optimization.h"
#include "backend/optimizer/ascend/ascend_backend_optimization.h"
#include "backend/optimizer/graph_kernel/graph_kernel_optimization.h"
#include "backend/session/ascend_auto_monad.h"
#include "utils/context/graph_kernel_flags.h"
#include "runtime/device/ascend/kernel_select_ascend.h"
#include "runtime/device/kernel_adjust.h"

#ifndef ENABLE_SECURITY
#include "debug/anf_ir_dump.h"
#include "debug/dump_proto.h"
#endif

namespace mindspore {
namespace device {
namespace ascend {
using AscendAutoMonad = mindspore::session::AscendAutoMonad;

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
  OptimizeExecutionOrder(graph);
  PostOptimization(graph);
  // must clear memo_ which holds kernelgraph after using AscendGraphOptimization class.
  memo_.clear();
  MS_LOG(INFO) << "Status record: end optimize graph. graph id: " << graph->graph_id();
}

void AscendGraphOptimization::OptimizeSingleOpGraph(const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  opt::RunOpAscendBackendIRFusionOptimization(graph);
  SelectKernel(graph);
  opt::RunOpAscendBackendOptimization(graph);

  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  // Cannot Hide nop node in PyNative mode.
  // If there is more than one node in the graph,
  // and one of the nodes is a nop node, the node will be hidden.
  // The DAG of Actors will be invalid(lack an input edge).
}

void AscendGraphOptimization::OptimizeGraphWithoutDeviceInfo(const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  CheckControlFlowDynamicShape(graph);
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  if (context_ptr->get_param<bool>(MS_CTX_IS_MULTI_GRAPH_SINK)) {
    HandleControlFlow(NOT_NULL(graph));
  }

  // add all graphs to manager first, so that don't have to make new manager in following passes.
  auto manager = Manage(graph, true);
  memo_.clear();
  AddGraphToManager(NOT_NULL(graph), NOT_NULL(manager));

  memo_.clear();
  IRFusionOptimization(graph);
}

void AscendGraphOptimization::OptimizeGraphWithDeviceInfo(const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  memo_.clear();
  HardWareOptimization(graph);
  // copy child graph ref output map to father graph ref output map
  memo_.clear();
  UpdateRefOutputMap(graph);
  AnfAlgo::InsertMakeTupleForOutput(NOT_NULL(graph));
}

void AscendGraphOptimization::OptimizeExecutionOrder(const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_LOG(INFO) << "Status record: start optimize execution order. graph id: " << graph->graph_id();
  // root root_graph validate,include generate execute order and so on
  RootGraphExecutorValidate(NOT_NULL(graph));

#ifdef ENABLE_DUMP_IR
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  bool save_graphs = context_ptr->get_param<bool>(MS_CTX_SAVE_GRAPHS_FLAG);
  if (save_graphs) {
    DumpIRProto(graph, "before_removeNop_" + std::to_string(graph->graph_id()));
  }
#endif

  // TODO(sida): do not hide nop op in kernel_by_kernel mode
  if (graph->is_executing_sink()) {
    opt::HideNopNode(graph.get());
  }

  auto execution_order = graph->execution_order();
  AnfAlgo::ReorderExecList(NOT_NULL(&execution_order));
  graph->set_execution_order(execution_order);

  device::KernelAdjust::GetInstance().InsertOverflowCheckOperations(NOT_NULL(graph));

#ifdef ENABLE_DUMP_IR
  if (save_graphs) {
    DumpIR("after_adjust_kernel.ir", graph);
  }
#endif
  MS_LOG(INFO) << "Status record: end optimize execution order. graph id: " << graph->graph_id();
}

void AscendGraphOptimization::PostOptimization(const KernelGraphPtr &graph) {
  MS_LOG(INFO) << "Status record: start post optimization. graph id: " << graph->graph_id();
  graph->SetOptimizerFlag();
  MS_LOG(INFO) << "Status record: end post optimization. graph id: " << graph->graph_id();
}

void AscendGraphOptimization::HardWareOptimization(const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_LOG(INFO) << "Status record: start hardware optimize. graph id: " << graph->graph_id();
  if (memo_.find(graph) != memo_.end()) {
    return;
  }
  memo_.insert(graph);
  opt::AscendBackendOptimization(graph);
  opt::CommonFinalOptimization(graph);
  if (graphkernel::GraphKernelFlags::GetInstance().IsEnableGraphKernel()) {
    graphkernel::GraphKernelOptimize(graph);
    graph->SetExecOrderByDefault();
  }
  MS_LOG(INFO) << "Status record: end hardware optimize. graph id: " << graph->graph_id();

  for (auto &child_graph : graph->child_graph_order()) {
    HardWareOptimization(child_graph.lock());
  }
}

void AscendGraphOptimization::AddGraphToManager(const NotNull<KernelGraphPtr> graph,
                                                NotNull<FuncGraphManagerPtr> manager) {
  if (memo_.find(graph) != memo_.end()) {
    return;
  }
  memo_.insert(graph.get());
  manager->AddFuncGraph(graph.get(), false);

  for (auto &child_graph : graph->child_graph_order()) {
    AddGraphToManager(NOT_NULL(child_graph.lock()), manager);
  }
}

void AscendGraphOptimization::IRFusionOptimization(const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  if (memo_.find(graph) != memo_.end()) {
    return;
  }
  memo_.insert(graph);

  opt::AscendBackendIRFusionOptimization(graph);

#ifdef ENABLE_DUMP_IR
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  bool save_graphs = context_ptr->get_param<bool>(MS_CTX_SAVE_GRAPHS_FLAG);
  if (save_graphs) {
    std::string file_name = "select_kernel_before_graph_" + std::to_string(graph->graph_id()) + ".ir";
    DumpIR(file_name, graph);
  }
#endif

  for (auto &child_graph : graph->child_graph_order()) {
    IRFusionOptimization(NOT_NULL(child_graph.lock()));
  }
}

void AscendGraphOptimization::HandleControlFlow(const NotNull<KernelGraphPtr> graph) {
  MS_LOG(INFO) << "Status record: start handle control flow. graph id: " << graph->graph_id();
  AscendAutoMonad auto_monad(graph);
  auto_monad.Run();
  MS_LOG(INFO) << "Status record: end handle control flow. graph id: " << graph->graph_id();
}

void AscendGraphOptimization::RootGraphExecutorValidate(NotNull<KernelGraphPtr> graph) {
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
  memo_.insert(graph);
  MS_LOG(INFO) << "Status record: start select kernel info. graph id: " << graph->graph_id();
  SetOperatorInfo(graph->execution_order());
  MS_LOG(INFO) << "Status record: end select kernel info. graph id: " << graph->graph_id();

#ifdef ENABLE_DUMP_IR
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  bool save_graphs = context_ptr->get_param<bool>(MS_CTX_SAVE_GRAPHS_FLAG);
  if (save_graphs) {
    std::string file_name = "select_kernel_after_graph_" + std::to_string(graph->graph_id()) + ".ir";
    DumpIR(file_name, graph);
  }
#endif

  for (auto &child_graph : graph->child_graph_order()) {
    RecurseSelectKernelInfo(child_graph.lock());
  }
}

void AscendGraphOptimization::SelectKernel(const KernelGraphPtr &graph) {
  MS_LOG(INFO) << "Status record: start select kernel info. graph id: " << graph->graph_id();
  raise_precision_count_ = 0;
  reduce_precision_count_ = 0;
  memo_.clear();
  RecurseSelectKernelInfo(graph);
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (ms_context->get_param<int>(MS_CTX_EXECUTION_MODE) == kGraphMode) {
    if (raise_precision_count_ > 0) {
      MS_LOG(WARNING) << "There are " << raise_precision_count_
                      << " node/nodes used raise precision to selected the kernel!";
    }
    if (reduce_precision_count_ > 0) {
      MS_LOG(WARNING) << "There are " << reduce_precision_count_
                      << " node/nodes used reduce precision to selected the kernel!";
    }
  }
  MS_LOG(INFO) << "Status record: end select kernel info. graph id: " << graph->graph_id();
}

void AscendGraphOptimization::UpdateRefOutputMap(const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  if (memo_.find(graph) != memo_.end()) {
    return;
  }
  memo_.insert(graph);

  for (auto &child_graph : graph->child_graph_order()) {
    auto child_graph_ptr = child_graph.lock();
    MS_EXCEPTION_IF_NULL(child_graph_ptr);
    UpdateRefOutputMap(NOT_NULL(child_graph_ptr));
    // copy ref map to final graph
    auto child_ref_map = child_graph_ptr->GetRefMap();
    for (auto &item : child_ref_map) {
      if (graph->IsInRefOutputMap(item.first)) {
        MS_LOG(WARNING) << "The ref pair <" << item.first.first->DebugString() << ", " << item.first.second
                        << "> is already in " << graph->ToString();
        continue;
      }
      graph->AddRefCorrespondPairs(item.first, item.second);
    }
  }
}

void AscendGraphOptimization::UnifyMindIR(const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_LOG(INFO) << "Status record: start unify mindir. graph id: " << graph->graph_id();
  opt::CommonUnifyMindIR(graph);
  opt::AscendUnifyMindIR(graph);
  MS_LOG(INFO) << "Status record: end unify mindir. graph id: " << graph->graph_id();
}

void AscendGraphOptimization::SetOperatorInfo(const std::vector<CNodePtr> &nodes) {
  for (const auto &node : nodes) {
    auto status = device::ascend::SelectKernelInfo(node);
    AnfAlgo::EraseNodeAttr(kAttrPynativeNextOpName, node);
    AnfAlgo::EraseNodeAttr(kAttrPynativeNextIndex, node);
    if (status == device::ascend::kStatusRaisePrecision) {
      raise_precision_count_++;
    } else if (status == device::ascend::kStatusReducePrecision) {
      reduce_precision_count_++;
    }
    MS_LOG(DEBUG) << "Select ApplyKernel: " << node->DebugString();
  }
}

void AscendGraphOptimization::GetAllGraphs(const KernelGraphPtr &root_graph) {
  if (memo_.find(root_graph) != memo_.end()) {
    return;
  }
  memo_.insert(root_graph);
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
