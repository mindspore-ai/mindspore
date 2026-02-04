/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include "backend/ge_backend/runtime/graph_compiler.h"
#include <numeric>
#include <map>
#include <utility>
#include <algorithm>
#include <functional>
#include <list>
#include <regex>
#include "backend/ge_backend/runtime/graph_scheduler.h"
#include "backend/ge_backend/utils/device_address_utils.h"
#include "device_address/device_address.h"
#include "utils/convert_utils.h"
#include "include/runtime/hardware_abstract/kernel_base/graph_fusion/graph_kernel_flags.h"
#include "backend/ge_backend/pass/ge_backend_optimization.h"
#include "utils/ms_context.h"
#include "ir/tensor.h"
#include "ir/graph_utils.h"
#include "include/runtime/hardware_abstract/kernel_base/graph_fusion/framework_utils.h"
#include "tools/profiler/profiling.h"
#include "include/backend/common/pass_manager/helper.h"
#include "mindspore/ccsrc/utils/ir_dump/dump_proto.h"
#include "include/utils/parallel_context.h"
#ifdef ENABLE_DUMP_IR
#include "mindspore/ccsrc/utils/ir_dump/anf_ir_dump.h"
#endif
#include "include/backend/common/pass_manager/graph_optimizer.h"
#include "tools/profiler/profiler.h"
#include "include/utils/compile_cache_context.h"
#include "primitive/framework_ops.h"
#include "include/runtime/utils/runtime_conf/runtime_conf.h"
#include "backend/ge_backend/executor/ge_graph_executor.h"
#include "include/backend/backend_manager/backend_jit_config.h"

namespace mindspore {
namespace ge_backend {
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
    auto kernel_tensor = AnfAlgo::GetOutputKernelTensor(node, index, false);
    MS_EXCEPTION_IF_NULL(kernel_tensor);
    kernel_tensor->set_original_ref_count(SIZE_MAX);
    kernel_tensor->ResetRefCount();
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
}  // namespace

GraphId GraphCompiler::CompileGraph(const GraphSegmentPtr &segment,
                                    const std::pair<AnfNodePtrList, AnfNodePtrList> &io_nodes,
                                    const backend::BackendJitConfig &backend_jit_config) {
  MS_EXCEPTION_IF_NULL(segment);
  MS_LOG(INFO) << "Status record: start compile graph.";
  auto nodes = segment->nodes_;
  auto device_target = device::GetDeviceTypeByName(GetCNodeTarget(nodes[0]));
  // Generate kernel graph.
  uint64_t start_time = profiler::GetClockSyscnt();
  PROF_START(ConstructKernelGraph);
  auto kernel_graph =
    session_->ConstructKernelGraph(nodes, io_nodes.second, device_target, backend_jit_config, true, true);
  PROF_END(ConstructKernelGraph);

  (void)profiler::CollectHostInfo(kModelNameRuntime, kEventCompileGraph, kStageConstructKernelGraph, start_time,
                                  profiler::GetClockSyscnt(), 1);
  SetGraphDependency(kernel_graph, segment);
  return CompileGraph(kernel_graph, io_nodes);
}

GraphId GraphCompiler::CompileGraph(const KernelGraphPtr &kernel_graph,
                                    const std::pair<AnfNodePtrList, AnfNodePtrList> &io_nodes) {
  MS_EXCEPTION_IF_NULL(session_);
  MS_EXCEPTION_IF_NULL(kernel_graph);

  const auto &outputs = io_nodes.second;
  kernel_graph->UpdateGraphAquireGilAttr();

  auto manager = MakeManager({kernel_graph});
  if (manager) {
    manager->AddFuncGraph(kernel_graph);
    kernel_graph->set_manager(manager);
  }

  backend::ge_backend::opt::OptimizationWithoutBackend(kernel_graph);
  backend::ge_backend::opt::GEUnifyMindIR(kernel_graph);

  kernel_graph->SetInputNodes();
  kernel_graph->SetExecOrderByDefault();
  session_->SetInputNodeUsage(kernel_graph, manager);
  kernel_graph->SetOptimizerFlag();

  GraphId graph_id = CompileGraphImpl(kernel_graph);

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

GraphId GraphCompiler::CompileGraphImpl(const KernelGraphPtr &graph) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(session_);
  const auto &context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);

#ifdef ENABLE_DUMP_IR
  if (context->CanDump(kIntroductory)) {
    // Dump .pb graph before graph optimization.
    DumpIRProto(graph, "before_opt_" + std::to_string(graph->graph_id()));
  }
#endif
  // Execute optimization pass.
  uint64_t start_time = profiler::GetClockSyscnt();
  PROF_START(OptimizeGraph);
  std::set<KernelGraphPtr> memo;
  backend::ge_backend::opt::OptimizeGEGraph(graph, &memo);
  PROF_END(OptimizeGraph);
  (void)profiler::CollectHostInfo(kModelNameRuntime, kEventCompileGraph, kStageOptimizeGraph, start_time,
                                  profiler::GetClockSyscnt(), 1);

  // Read the output and input ref map and set to the kernel graph.
  AnfAlgo::AddOutInRefToGraph(graph);

  session_->RecurseSetSummaryNodesForAllGraphs(graph.get());

  // dynamic shape pass of graphmode
  if (graph->is_dynamic_shape()) {
    auto profiler_manage_inst = profiler::ProfilerManager::GetInstance();
    MS_EXCEPTION_IF_NULL(profiler_manage_inst);
    profiler_manage_inst->SetNetDynamicShapeStatus();
  }

  // Adjust kernel graph before run graph.
  start_time = profiler::GetClockSyscnt();
  PROF_START(PreprocessBeforeRun);
  if (!AnfAlgo::IsNoRealKernelGraph(graph)) {
    MS_EXCEPTION_IF_NULL(graph_executor_);
    graph_executor_->CompileGraph(std::dynamic_pointer_cast<FuncGraph>(graph), {});
  }
  PROF_END(PreprocessBeforeRun);
  (void)profiler::CollectHostInfo("Ascend", "PreprocessBeforeRun", "GePreprocess", start_time,
                                  profiler::GetClockSyscnt(), 1);
  graph->UpdateInternalParameter();
  // Set device target for parameter affinity.
  AnfAlgo::SetParameterDeviceTarget(graph);

  PROF_START(CreateDeviceAddress);
  // Create device address for all anf nodes of graph.
  CreateDeviceAddress(graph);
  PROF_END(CreateDeviceAddress);

  SetSummaryNodesRefCount(graph.get());
#ifdef ENABLE_DUMP_IR
  // Dump .pb graph after graph optimization.
  if (context->CanDump(kIntroductory)) {
    DumpIRProto(graph, "after_opt_" + std::to_string(graph->graph_id()));
  }
#endif

  graph->EnableRuntimeCache();
  return graph->graph_id();
}

KernelGraphPtr GraphCompiler::Fetch(GraphId graph_id) const {
  MS_EXCEPTION_IF_NULL(session_);
  return session_->GetGraph(graph_id);
}

void GraphCompiler::CreateDeviceAddress(const KernelGraphPtr &graph) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_LOG(INFO) << "Status record: start create device address. graph id: " << graph->graph_id();
  backend::ge_backend::DeviceAddressUtils::CreateParameterDeviceAddress(graph);
  backend::ge_backend::DeviceAddressUtils::CreateValueNodeDeviceAddress(graph);
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
}  // namespace ge_backend
}  // namespace mindspore
