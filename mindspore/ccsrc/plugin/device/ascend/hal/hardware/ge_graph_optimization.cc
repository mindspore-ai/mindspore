/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "plugin/device/ascend/hal/hardware/ge_graph_optimization.h"
#include <string>
#include <memory>
#include "backend/common/optimizer/common_backend_optimization.h"
#include "backend/common/graph_kernel/graph_kernel_flags.h"
#include "backend/common/graph_kernel/adapter/graph_kernel_optimization.h"
#include "plugin/device/ascend/optimizer/ge_backend_optimization.h"
#include "plugin/device/ascend/optimizer/backend_common_unify_mindir.h"
#include "plugin/device/ascend/hal/device/kernel_adjust.h"
#ifndef ENABLE_SECURITY
#include "include/common/debug/dump_proto.h"
#endif

namespace mindspore {
namespace device {
namespace ascend {
void GEGraphOptimization::OptimizeGEGraph(const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_LOG(DEBUG) << "Status record: start optimize ge graph. graph id: " << graph->graph_id();
  // empty graph dont entry to backend
  if (graph->execution_order().empty()) {
    MS_LOG(DEBUG) << graph->ToString() << " is empty graph.";
    AnfAlgo::InsertMakeTupleForOutput(NOT_NULL(graph));
    graph->set_executable(false);
    MS_LOG(DEBUG) << "Status record: end optimize ge graph. graph id: " << graph->graph_id();
  }
  opt::GEBackendOptimizeACL(graph);
  opt::GEBackendOptimization(graph);
  if (graphkernel::GraphKernelFlags::GetInstance().IsEnableGraphKernel()) {
    graphkernel::GraphKernelOptimize(graph);
    graph->SetExecOrderByDefault();
  }
  MS_LOG(DEBUG) << "Status record: end optimize ge graph. graph id: " << graph->graph_id();
}

void GEGraphOptimization::OptimizeACLGraph(const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_LOG(DEBUG) << "Status record: start optimize acl graph. graph id: " << graph->graph_id();
  // empty graph dont entry to backend
  if (graph->execution_order().empty()) {
    MS_LOG(DEBUG) << graph->ToString() << " is empty graph.";
    AnfAlgo::InsertMakeTupleForOutput(NOT_NULL(graph));
    graph->set_executable(false);
    MS_LOG(DEBUG) << "Status record: end optimize acl graph. graph id: " << graph->graph_id();
  }
  opt::AscendUnfoldInputsForSpecialNodes(graph);
  opt::GEBackendOptimizeACL(graph);
  MS_LOG(DEBUG) << "Status record: end optimize acl graph. graph id: " << graph->graph_id();
}

void GEGraphOptimization::OptimizeACLGraphAfterKernelSelect(const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_LOG(DEBUG) << "Status record: start optimize acl graph after kernel select. graph id: " << graph->graph_id();
  // empty graph dont entry to backend
  if (graph->execution_order().empty()) {
    MS_LOG(DEBUG) << graph->ToString() << " is empty graph.";
    AnfAlgo::InsertMakeTupleForOutput(NOT_NULL(graph));
    graph->set_executable(false);
    MS_LOG(DEBUG) << "Status record: end optimize acl graph after kernel select. graph id: " << graph->graph_id();
  }
  opt::GEBackendOptimizeACLAfterKernelSelect(graph);
  MS_LOG(DEBUG) << "Status record: end optimize acl graph after kernel select. graph id: " << graph->graph_id();
}

void GEGraphOptimization::UnifyMindIR(const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_LOG(INFO) << "Status record: start unify mindir. graph id: " << graph->graph_id();
  profiler::CollectHostInfo("Ascend", "Graph Optimization", "UnifyMindIR", 0, 0, 0);
  PROF_START(unify_mindir);
  opt::CommonUnifyMindIR(graph);
  opt::GEUnifyMindIR(graph);
  PROF_END(unify_mindir);
  profiler::CollectHostInfo("Ascend", "Graph Optimization", "UnifyMindIR", 0, 0, 1);
  MS_LOG(INFO) << "Status record: end unify mindir. graph id: " << graph->graph_id();
}

void GEGraphOptimization::GEMindIRPass(const KernelGraphPtr &graph) const { opt::GEUnifyMindIR(graph); }
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
