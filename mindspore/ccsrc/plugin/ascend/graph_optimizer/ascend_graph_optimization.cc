/**
 * Copyright 2023-2025 Huawei Technologies Co., Ltd
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

#include "plugin/ascend/graph_optimizer/ascend_graph_optimization.h"
#include <string>
#include <memory>
#include "primitive/framework_ops.h"
#include "include/utils/callback.h"
#include "include/utils/anfalgo.h"
#include "include/backend/common/pass_manager/common_backend_optimization.h"
#include "include/runtime/hardware_abstract/kernel_base/graph_fusion/graph_kernel_flags.h"
#include "plugin/ascend/graph_optimizer/pass/ascend_pass_optimization.h"
#include "plugin/ascend/graph_optimizer/pass/backend_common_unify_mindir.h"
#include "utils/ms_context.h"
#include "include/backend/common/kernel_graph/anf_runtime_algorithm.h"
#include "ir/graph_utils.h"
#include "tools/profiler/profiling.h"
#include "mindspore/ccsrc/utils/ir_dump/dump_proto.h"
#include "primitive/auto_generate/gen_ops_primitive_u.h"

namespace mindspore {
namespace device {
namespace ascend {
namespace {
void MarkRefGraph(const KernelGraphPtr &kernel_graph) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  MS_LOG(INFO) << "Mark graph is ref graph: " << kernel_graph->graph_id();
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto is_kbk = ms_context->IsKByKExecutorMode();
  auto manager = kernel_graph->manager();
  if (manager == nullptr || kernel_graph->has_attr(kIsRefGraph)) {
    return;
  }
  for (const auto &node : TopoSort(kernel_graph->get_return(), SuccDeeperSimple, AlwaysInclude)) {
    if (!node->isa<CNode>()) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    if (cnode == nullptr) {
      continue;
    }
    auto is_side_effect = common::AnfAlgo::HasNodeAttr(GRAPH_FLAG_SIDE_EFFECT_MEM, cnode) &&
                          common::AnfAlgo::GetNodeAttr<bool>(cnode, GRAPH_FLAG_SIDE_EFFECT_MEM);
    if (!(is_side_effect && cnode->fullname_with_scope().find("optimizer") != std::string::npos)) {
      continue;
    }
    for (const auto &node_pair : manager->node_users()[cnode]) {
      if (IsPrimitiveCNode(node_pair.first, prim::kPrimUpdateState)) {
        kernel_graph->set_attr(kIsRefGraph, MakeValue(true));
        MS_LOG(INFO) << "graph is ref graph: " << kernel_graph->graph_id();
        if (!is_kbk) {
          return;
        }
        common::AnfAlgo::SetNodeAttr(kFromRefGraph, MakeValue(true), cnode);
        break;
      }
    }
  }
}
}  // namespace

void AscendGraphOptimization::OptimizeACLGraph(const KernelGraphPtr &graph, std::set<KernelGraphPtr> *const memo) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(memo);
  PROF_START(OptimizeACLGraph);
  if (memo->find(graph) != memo->end()) {
    return;
  }
  memo->insert(graph);
  MS_LOG(DEBUG) << "Status record: start optimize acl graph. graph id: " << graph->graph_id();
  // empty graph dont entry to backend
  if (graph->execution_order().empty()) {
    MS_LOG(DEBUG) << graph->ToString() << " is empty graph.";
    AnfAlgo::InsertMakeTupleForOutput(NOT_NULL(graph));
    graph->set_executable(false);
    MS_LOG(DEBUG) << "Status record: end optimize acl graph. graph id: " << graph->graph_id();
  }
  MarkRefGraph(graph);
  opt::AscendUnfoldInputsForSpecialNodes(graph);
  opt::AscendGraphOptimizeACL(graph);
  for (auto &child_graph : graph->child_graph_order()) {
    OptimizeACLGraph(child_graph.lock(), memo);
  }
  PROF_END(OptimizeACLGraph);
  MS_LOG(DEBUG) << "Status record: end optimize acl graph. graph id: " << graph->graph_id();
}

void AscendGraphOptimization::OptimizeACLGraphAfterKernelSelect(const KernelGraphPtr &graph,
                                                                std::set<KernelGraphPtr> *const memo) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(memo);
  PROF_START(OptimizeACLGraphAfterKernelSelect);
  if (memo->find(graph) != memo->end()) {
    return;
  }
  memo->insert(graph);
  MS_LOG(DEBUG) << "Status record: start optimize acl graph after kernel select. graph id: " << graph->graph_id();
  // empty graph dont entry to backend
  if (graph->execution_order().empty()) {
    MS_LOG(DEBUG) << graph->ToString() << " is empty graph.";
    AnfAlgo::InsertMakeTupleForOutput(NOT_NULL(graph));
    graph->set_executable(false);
    MS_LOG(DEBUG) << "Status record: end optimize acl graph after kernel select. graph id: " << graph->graph_id();
  }
  if (!graph->is_from_single_op() && graphkernel::GraphKernelFlags::GetInstance().IsEnableGraphKernel()) {
    constexpr char kGraphKernelOptimizeCallBackFunc[] = "GraphKernelOptimize";
    static auto graphkernel_optimize_callback =
      callback::CommonCallback::GetInstance().GetCallback<void, const KernelGraphPtr &>(
        kGraphKernelOptimizeCallBackFunc);
    if (graphkernel_optimize_callback) {
      graphkernel_optimize_callback(graph);
    }
  }
  opt::AscendGraphOptimizeACLAfterKernelSelect(graph);
  if (!graph->is_from_single_op() && graphkernel::GraphKernelFlags::GetInstance().IsEnableKernelPacket() &&
      common::AnfAlgo::IsDynamicGraph(graph)) {
    constexpr char kKernelPacketOptimizeCallBackFunc[] = "KernelPacketOptimize";
    static auto kernelpacket_optimize_callback =
      callback::CommonCallback::GetInstance().GetCallback<void, const KernelGraphPtr &>(
        kKernelPacketOptimizeCallBackFunc);
    if (kernelpacket_optimize_callback) {
      kernelpacket_optimize_callback(graph);
    }
  }
  // after kernel packet
  opt::AscendGraphOptimizeACLAfterKernelPacket(graph);
  for (auto &child_graph : graph->child_graph_order()) {
    OptimizeACLGraphAfterKernelSelect(child_graph.lock(), memo);
  }
  PROF_END(OptimizeACLGraphAfterKernelSelect);
  MS_LOG(DEBUG) << "Status record: end optimize acl graph after kernel select. graph id: " << graph->graph_id();
}

void AscendGraphOptimization::OptimizeACLGraphAfterCreateKernel(const KernelGraphPtr &graph) {
  PROF_START(OptimizeACLGraphAfterCreateKernel);
  // pynaitve process the pass in AscendGraphOptimizeACLAfterKernelSelect
  if (!IsJit()) {
    return;
  }
  MS_EXCEPTION_IF_NULL(graph);
  MS_LOG(DEBUG) << "Status record: start optimize acl graph after create kernel. graph id: " << graph->graph_id();
  // empty graph dont entry to backend
  if (graph->execution_order().empty()) {
    MS_LOG(DEBUG) << graph->ToString() << " is empty graph.";
    AnfAlgo::InsertMakeTupleForOutput(NOT_NULL(graph));
    graph->set_executable(false);
    MS_LOG(DEBUG) << "Status record: end optimize acl graph after create kernel. graph id: " << graph->graph_id();
  }
  opt::AclAfterCreateKernel(graph);
  PROF_END(OptimizeACLGraphAfterCreateKernel);
  MS_LOG(DEBUG) << "Status record: end optimize acl graph after create kernel. graph id: " << graph->graph_id();
}

void AscendGraphOptimization::OptimizeACLGraphAfterInline(const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_LOG(DEBUG) << "Status record: start optimize acl graph after inline. graph id: " << graph->graph_id();
  // empty graph dont entry to backend
  if (graph->execution_order().empty()) {
    MS_LOG(DEBUG) << graph->ToString() << " is empty graph.";
    AnfAlgo::InsertMakeTupleForOutput(NOT_NULL(graph));
    graph->set_executable(false);
    MS_LOG(DEBUG) << "Status record: end optimize acl graph after inline. graph id: " << graph->graph_id();
  }
  opt::AscendAfterInlineOptimize(graph);
  MS_LOG(DEBUG) << "Status record: end optimize acl graph after inline. graph id: " << graph->graph_id();
}

void AscendGraphOptimization::UnifyMindIR(const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_LOG(INFO) << "Status record: start unify mindir. graph id: " << graph->graph_id();
  uint64_t start_time = profiler::GetClockSyscnt();
  PROF_START(UnifyMindIR);
  opt::CommonUnifyMindIR(graph);
  opt::AscendUnifyMindIR(graph);
  PROF_END(UnifyMindIR);
  (void)profiler::CollectHostInfo("Ascend", "Graph Optimization", "UnifyMindIR", start_time, profiler::GetClockSyscnt(),
                                  0);
  MS_LOG(INFO) << "Status record: end unify mindir. graph id: " << graph->graph_id();
}

void AscendGraphOptimization::AscendUnifyMindIR(const KernelGraphPtr &graph) const { opt::AscendUnifyMindIR(graph); }
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
