/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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
#include "runtime/graph_scheduler/graph_scheduler.h"
#include "runtime/device/device_address_utils.h"
#include "runtime/pynative/op_executor.h"
#include "runtime/device/device_address.h"
#include "runtime/device/ms_device_shape_transfer.h"
#include "runtime/pynative/op_runtime_info.h"
#include "include/common/utils/convert_utils.h"
#include "common/graph_kernel/graph_kernel_flags.h"
#include "utils/ms_context.h"
#include "ir/tensor.h"
#include "kernel/common_utils.h"
#include "profiler/device/profiling.h"
#include "backend/common/optimizer/helper.h"
#include "base/base_ref_utils.h"
#include "include/common/debug/dump_proto.h"
#include "include/common/utils/parallel_context.h"
#include "plugin/device/cpu/hal/hardware/cpu_device_context.h"
#ifdef ENABLE_DEBUGGER
#include "debug/debugger/debugger.h"
#endif
#ifdef ENABLE_DUMP_IR
#include "include/common/debug/anf_ir_dump.h"
#endif
#ifndef ENABLE_SECURITY
#include "debug/data_dump/dump_json_parser.h"
#include "backend/common/optimizer/graph_optimizer.h"
#endif
#if defined(__linux__) && defined(WITH_BACKEND)
#include "ps/ps_context.h"
#include "runtime/graph_scheduler/embedding_cache_scheduler.h"
#endif

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
}  // namespace

GraphCompilerInfo::~GraphCompilerInfo() {
  GraphScheduler::GetInstance().Clear(name_, graphs_, origin_parameters_order_, control_node_parser_);
}

namespace {
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
    MS_LOG(EXCEPTION) << "Invalid cnode:" << cnode->DebugString();
  }
  return FetchRealNodeByNopNode(inputs[1]);
}

// Recursively delete the nodes in the eliminate nodes list in the graph, check node records
// the nodes that have been checked during the recursive process.
void EliminateNodesFromGraph(CNode *node, const std::set<AnfNodePtr> &eliminate_nodes,
                             std::set<CNode *> *checked_nodes) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(checked_nodes);

  if (checked_nodes->find(node) != checked_nodes->end()) {
    return;
  }

  (void)checked_nodes->emplace(node);
  const auto &inputs = node->inputs();
  std::vector<AnfNodePtr> new_inputs;
  for (auto &input : inputs) {
    MS_EXCEPTION_IF_NULL(input);
    if (!input->isa<CNode>()) {
      (void)new_inputs.emplace_back(input);
      continue;
    }

    if (eliminate_nodes.find(input) == eliminate_nodes.end()) {
      (void)new_inputs.emplace_back(input);
    } else {
      // If input is an eliminate node, replace it by its real input.
      const auto &real_input = FetchRealNodeByNopNode(input);
      MS_EXCEPTION_IF_NULL(real_input);

      // Since the output of previous node will be cached, the cache needs to be updated after eliminating the nopnode.
      auto kernel_info = node->kernel_info();
      if (kernel_info) {
        auto runtime_cache = kernel_info->runtime_cache();
        if (runtime_cache.runtime_cache().is_valid()) {
          runtime_cache.runtime_cache().update_prev_node_output(
            new_inputs.size() - 1, common::AnfAlgo::VisitKernelWithReturnType(real_input, 0));
        }
      }
      (void)new_inputs.emplace_back(real_input);
    }
    const auto &cnode = input->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    EliminateNodesFromGraph(cnode.get(), eliminate_nodes, checked_nodes);
  }
  node->set_inputs(new_inputs);
}

// Check whether a cnode has a monad input.
bool HasMonadInput(const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  const auto &inputs = cnode->inputs();
  for (const auto &input : inputs) {
    MS_EXCEPTION_IF_NULL(input);
    if (HasAbstractMonad(input)) {
      return true;
    }
  }
  return false;
}

// Collect all nopnodes which are input of kernel that not support multi-thread execute.
std::set<CNodePtr> FetchNopNodeNotSupportEliminate(const KernelGraph *const graph) {
  MS_EXCEPTION_IF_NULL(graph);
  std::set<CNodePtr> invalid_nopnodes;

  // In the implementation of some cpu operators, shape information is obtained through the input of the kernel
  // in launchkernel stage. The nopnode input of these operators cannot be eliminated.
  const std::set<std::string> kCPUOpNoEliminateList = {kConcatOpName};

  for (const auto &cnode : graph->execution_order()) {
    MS_EXCEPTION_IF_NULL(cnode);
    // If target is not cpu, the total cnode in graph can skip.
    auto target = GetCNodeTarget(cnode);
    if (target != kCPUDevice) {
      break;
    }

    // kernel not support multi-thread execute will be inited in launch kernel, so its input cannot be eliminated.
    if (kCPUOpNoEliminateList.find(common::AnfAlgo::GetCNodeName(cnode)) != kCPUOpNoEliminateList.end()) {
      const auto &inputs = cnode->inputs();
      for (const auto &input : inputs) {
        MS_EXCEPTION_IF_NULL(input);
        const auto &input_with_index = common::AnfAlgo::VisitKernelWithReturnType(input, 0);
        if ((input_with_index.first != nullptr) && (input_with_index.first->isa<CNode>()) &&
            common::AnfAlgo::IsNopNode(input_with_index.first)) {
          // Collect all of the nopnode inputs.
          (void)invalid_nopnodes.emplace(input->cast<CNodePtr>());
          MS_LOG(INFO) << "Add invalid nopnode:" << input->DebugString()
                       << " for node not support mulit-thread execute list.";
        }
      }
    }
  }
  return invalid_nopnodes;
}

void OptimizeNopNode(KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(graph);
  std::vector<CNodePtr> new_execution_order;
  std::vector<CNodePtr> nop_nodes_need_set_ref;
  std::set<AnfNodePtr> nop_nodes_need_eliminated;

  // Skip the graph mode.
  if (graph->is_graph_run_mode()) {
    return;
  }

  // Invalid nopnode is those cannot be eliminated in some scene.
  const auto &invalid_nopnodes = FetchNopNodeNotSupportEliminate(graph);
  const auto &output_node = graph->output();
  MS_EXCEPTION_IF_NULL(output_node);
  const auto &graph_outputs = common::AnfAlgo::GetAllOutputWithIndex(output_node);
  // Collect all the nopnodes that can be eliminated.
  for (const auto &cnode : graph->execution_order()) {
    MS_EXCEPTION_IF_NULL(cnode);
    if ((!common::AnfAlgo::IsNopNode(cnode)) || graph->IsInRefOutputMap({cnode, 0}) ||
        graph->IsRefOutputMapValue({cnode, 0}) ||
        (std::find_if(graph_outputs.begin(), graph_outputs.end(), [&cnode](const KernelWithIndex &output) {
           const auto &real_output = common::AnfAlgo::FetchRealNodeSkipMonadControl(output);
           return real_output == KernelWithIndex(cnode, 0);
         }) != graph_outputs.end())) {
      (void)new_execution_order.emplace_back(cnode);
      continue;
    }
    // The nopnode which satisfies the following conditions cannot be eliminated and set to ref node:
    // 1.dynamic shape 2.side effect 3. must not be eliminated.
    if (graph->is_dynamic_shape() || HasMonadInput(cnode) || (invalid_nopnodes.find(cnode) != invalid_nopnodes.end())) {
      (void)new_execution_order.emplace_back(cnode);
      (void)nop_nodes_need_set_ref.emplace_back(cnode);
    } else {
      MS_LOG(DEBUG) << "Eliminate node:" << cnode->DebugString();
      (void)nop_nodes_need_eliminated.emplace(cnode);
    }
  }

  // Add the ref node pairs.
  for (auto &ref_node : nop_nodes_need_set_ref) {
    MS_EXCEPTION_IF_NULL(ref_node);
    auto input_node = common::AnfAlgo::GetInputNode(ref_node, 0);
    MS_EXCEPTION_IF_NULL(input_node);
    auto origin_pair = common::AnfAlgo::VisitKernelWithReturnType(input_node, 0, true);
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

  std::set<CNode *> checked_nodes;
  MS_EXCEPTION_IF_NULL(graph->return_node());
  EliminateNodesFromGraph(graph->return_node().get(), nop_nodes_need_eliminated, &checked_nodes);
  graph->set_execution_order(new_execution_order);
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
  if (is_parallel_mode) {
    return false;
  }

  if (common::GetEnv("DISABLE_ZERO_COPY") == "1") {
    return false;
  }
  return true;
}
}  // namespace
void SetRunGraphBySingleOpFlag(const KernelGraphPtr &graph) {
  for (auto &node : graph->execution_order()) {
    MS_EXCEPTION_IF_NULL(node->input(0));
    bool enable = false;
    if (!AnfAlgo::NodeValueIsFuncGraph(node->input(0))) {
      if (kernel::IfNeedSkipResize(node) && graph->has_flag(kFlagPyNativeRunInGraph)) {
        MS_LOG(DEBUG) << "Enable Run Graph By Single Op";
        enable = true;
      }
    }
    // BpGraph contain bprop_cut node.
    if (common::AnfAlgo::IsControlOpExecInBackend(node) || enable) {
      graph->set_flag(kFlagEnableRunGraphBySingleOp, true);
      break;
    }
  }
}
GraphId GraphCompiler::CompileGraph(const GraphSegmentPtr &segment, const AnfNodePtrList &outputs,
                                    const DeviceContext *device_context, device::RunMode run_mode,
                                    bool run_in_pynative) {
  MS_EXCEPTION_IF_NULL(session_);
  MS_EXCEPTION_IF_NULL(segment);
  MS_EXCEPTION_IF_NULL(device_context);
  MS_LOG(INFO) << "Status record: start compile graph.";
  auto nodes = segment->nodes_;
  auto device_terget = device_context->GetDeviceType();
  // Generate kernel graph.
  KernelGraphPtr graph =
    session_->ConstructKernelGraph(nodes, outputs, device_terget, true, IsEnableZeroCopy(run_in_pynative));
  MS_EXCEPTION_IF_NULL(graph);
  SetRunGraphBySingleOpFlag(graph);
  opt::EliminateIllegalDataTypePass(graph);
  SetGraphDependency(graph, segment);

  // Unify the MindIR, must be before of the graph optimization.
  auto deprecated_kernel_executor =
    dynamic_cast<device::DeprecatedKernelExecutor *>(device_context->kernel_executor_.get());
  if (deprecated_kernel_executor != nullptr) {
    deprecated_kernel_executor->UnifyMindIR(graph);
  } else {
    opt::CommonUnifyMindIR(graph);
  }

  // The graph common optimization.
  graph->UpdateGraphAquireGilAttr();
  opt::BackendCommonOptimization(graph);
  graph->SetInputNodes();
  auto manager = MakeManager({graph});
  if (manager) {
    manager->AddFuncGraph(graph);
    graph->set_manager(manager);
  }
  session_->SetInputNodeUsage(graph, manager);
  graph->SetOptimizerFlag();

  if (run_mode == device::RunMode::kUnknown) {
    graph->set_run_mode(device_context->GetRunMode(graph));
  } else {
    graph->set_run_mode(run_mode);
  }

  GraphId graph_id = 0;
  if (run_in_pynative) {
    MS_EXCEPTION_IF_NULL(session_);
    // Graph kernel does not support pynative mode now, print a warning here.
    graphkernel::GraphKernelFlags::GetInstance().CheckSupport();
    graph_id = graph->graph_id();
  } else {
    graph_id = CompileGraphImpl(graph, device_context, run_in_pynative);
  }

  graph->set_front_outputs(outputs);

  graph->set_root_graph_id(graph_id);
  session_->DumpGraphs({graph});

  // The graph is not compiled yet in PyNative Mode.
  // Need to cache output latter when the graph is compiled.
  if (!run_in_pynative) {
    // Cache the backend graph output nodes to front nodes with output index.
    auto backend_node = graph->output();
    MS_EXCEPTION_IF_NULL(backend_node);
    graph->CacheGraphOutputToFrontNodeWithIndex({backend_node}, outputs);
  }
  AnfAlgo::UpdateGraphValidRefPair(graph);

  MS_LOG(INFO) << "Status record: end compile graph. graph id: " << graph_id;
  return graph_id;
}

namespace {
void OptimizeDynamicGraph(const KernelGraphPtr &graph, const DeviceContext *device_context) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(device_context);
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
#ifdef ENABLE_DUMP_IR
  if (context_ptr->CanDump(kIntroductory)) {
    std::string file_name = "hwopt_d_before_opt_dynamic_graph_" + std::to_string(graph->graph_id()) + ".ir";
    DumpIR(file_name, graph);
    DumpIRProto(graph, "before_opt_dynamic_graph_hwopt_" + std::to_string(graph->graph_id()));
  }
#endif
  auto opt = std::make_shared<opt::GraphOptimizer>();
  opt->AddPassManager(opt::GetEliminateIllegalDataTypePassManager());
  // Unify the MindIR, must be before of the graph optimization.
  auto deprecated_kernel_executor =
    dynamic_cast<device::DeprecatedKernelExecutor *>(device_context->kernel_executor_.get());
  if (deprecated_kernel_executor != nullptr) {
    deprecated_kernel_executor->AddUnifyMindIRPass(opt);
  } else {
    opt->AddPassManager(opt::GetCommonUnifyMindIRPassManager());
  }
  opt->AddPassManager(opt::GetBackendCommonOptimizationPassManagerPtr(graph));
  (void)opt->Optimize(graph);
  graph->SetExecOrderByDefault();
#ifdef ENABLE_DUMP_IR
  if (context_ptr->CanDump(kIntroductory)) {
    std::string file_name = "hwopt_d_after_opt_dynamic_graph_" + std::to_string(graph->graph_id()) + ".ir";
    DumpIR(file_name, graph);
  }
#endif
}
}  // namespace

GraphId GraphCompiler::CompileDynamicGraph(const GraphSegmentPtr &segment, const AnfNodePtrList &outputs,
                                           const DeviceContext *device_context) {
  MS_EXCEPTION_IF_NULL(session_);
  MS_EXCEPTION_IF_NULL(segment);
  MS_EXCEPTION_IF_NULL(device_context);
  MS_LOG(INFO) << "Status record: start compile graph.";
  auto nodes = segment->nodes_;
  auto device_terget = device_context->GetDeviceType();
  // Generate kernel graph.
  KernelGraphPtr graph = session_->ConstructKernelGraph(nodes, outputs, device_terget, true, false);
  MS_EXCEPTION_IF_NULL(graph);

  // Dynamic shape or dynamic graph structure flag.
  graph->set_flag(kAttrMutableKernel, true);
  graph->set_flag(kFlagEnableRunGraphBySingleOp, true);

  OptimizeDynamicGraph(graph, device_context);

  graph->UpdateGraphAquireGilAttr();
  graph->SetInputNodes();
  auto manager = MakeManager({graph});
  if (manager) {
    manager->AddFuncGraph(graph);
    graph->set_manager(manager);
  }
  session_->SetInputNodeUsage(graph, manager);
  graph->SetOptimizerFlag();
  graph->set_run_mode(device::RunMode::kKernelMode);

  // Graph kernel does not support pynative mode now, print a warning here.
  graphkernel::GraphKernelFlags::GetInstance().CheckSupport();

  GraphId graph_id = graph->graph_id();
  graph->set_root_graph_id(graph_id);
  session_->DumpGraphs({graph});

  auto exec_nodes = graph->execution_order();
  std::for_each(exec_nodes.begin(), exec_nodes.end(),
                [](const CNodePtr &node) { common::AnfAlgo::SetNodeAttr(kAttrMutableKernel, MakeValue(true), node); });

  MS_LOG(INFO) << "Status record: end compile graph. graph id: " << graph_id;
  return graph_id;
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
  KernelGraphPtr root_graph = session_->ConstructKernelGraph(func_graph, &all_graphs, device_target);
  MS_EXCEPTION_IF_NULL(root_graph);
  for (const auto &graph : all_graphs) {
    MS_EXCEPTION_IF_NULL(graph);
    MS_LOG(INFO) << "Set root graph for graph: " << graph->graph_id() << " to: " << root_graph->graph_id() << ".";
    graph->set_root_graph_id(root_graph->graph_id());
    graph->set_run_mode(device::RunMode::kGraphMode);
    graph->set_is_loop_count_sink(true);
    // Convert the illegal data type.
    opt::EliminateIllegalDataTypePass(graph);
  }

  // todo: waiting for GraphExecutor
  auto jit_level = common::AnfAlgo::GetJitLevel(func_graph);
  MS_EXCEPTION_IF_NULL(MsContext::GetInstance());
  if (MsContext::GetInstance()->backend_policy() == "ge" && (jit_level == "O3" || jit_level == "")) {
    auto manager = MakeManager();
    MS_EXCEPTION_IF_NULL(manager);
    for (const auto &graph : all_graphs) {
      MS_EXCEPTION_IF_NULL(graph);
      manager->AddFuncGraph(graph);
      graph->set_manager(manager);
    }
    MS_EXCEPTION_IF_NULL(device_context->graph_executor_);
    if (!device_context->graph_executor_->CompileGraph(root_graph, {})) {
      MS_LOG(EXCEPTION) << "Compile graph failed: " << root_graph->graph_id();
    }
    root_graph->CacheGraphOutputToFrontNodeWithIndex({root_graph->output()}, {func_graph->output()});
    return root_graph->graph_id();
  }

  // set executing sink true in graph mode
  root_graph->set_run_mode(device::RunMode::kGraphMode);
  root_graph->set_is_loop_count_sink(true);
#if defined(__linux__) && defined(WITH_BACKEND)
  // Embedding cache need global step of compute graph, can not enable loop sink, move loop control to loop count actor.
  if (ps::PSContext::instance()->cache_enable()) {
    root_graph->set_is_loop_count_sink(false);
    for (const auto &graph : all_graphs) {
      MS_EXCEPTION_IF_NULL(graph);
      graph->set_is_loop_count_sink(false);
    }
  }
#endif

  // Unify the MindIR, must be before of the graph optimization.
  auto deprecated_kernel_executor =
    dynamic_cast<device::DeprecatedKernelExecutor *>(device_context->kernel_executor_.get());
  if (deprecated_kernel_executor != nullptr) {
    deprecated_kernel_executor->UnifyMindIR(root_graph);
  }

  // The graph common optimization.
  opt::BackendCommonOptimization(root_graph);
  root_graph->SetInputNodes();

  GraphId graph_id = 0;
  if (!func_graph->has_flag(kFlagPyNativeRunInGraph)) {
    graph_id = CompileGraphImpl(root_graph, device_context);
  } else {
    graph_id = root_graph->graph_id();
  }
  // Set summary nodes for all graphs.
  session_->SetSummaryNodesForAllGraphs(root_graph.get(), all_graphs);

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
      if (common::AnfAlgo::IsControlOpExecInBackend(node)) {
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
  const auto &ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  MS_EXCEPTION_IF_NULL(session_);

#ifdef ENABLE_DUMP_IR
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  if (context->CanDump(kIntroductory)) {
    // Dump .pb graph before graph optimization.
    DumpIRProto(graph, "before_opt_" + std::to_string(graph->graph_id()));
  }
#endif
  MS_EXCEPTION_IF_NULL(device_context->kernel_executor_);
  // Execute optimization pass.
  device_context->kernel_executor_->OptimizeGraph(graph);
  // Generate 'KernelMod' for all kernels and set 'KernelMod' into kernel,
  // 'KernelMod' is real executive object of kernel.
  device_context->kernel_executor_->CreateKernel(graph->execution_order());

  // Kernels that are not supported by other device can be backed off and rebuilt on the CPU.
#ifdef WITH_BACKEND
  if (!graph->is_from_single_op()) {
    auto cpu_device_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(
      {kCPUDevice, device_context->device_context_key().device_id_});
    MS_EXCEPTION_IF_NULL(cpu_device_context);
    auto cpu_executor = dynamic_cast<device::cpu::CPUKernelExecutor *>(cpu_device_context->kernel_executor_.get());
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
  session_->SetSummaryNodes(graph.get());
  // Update needed dump kernels for mindRT.
  DumpJsonParser::GetInstance().UpdateNeedDumpKernels(*graph.get());
#endif

  // dynamic shape pass of graphmode
  auto kernel_graph = graph->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  if (kernel_graph->is_dynamic_shape()) {
    opt::DynamicShapeConvertPass(graph);
  }
  auto profiler_manage_inst = profiler::ProfilerManager::GetInstance();
  MS_EXCEPTION_IF_NULL(profiler_manage_inst);
  if (kernel_graph->is_dynamic_shape()) {
    profiler_manage_inst->SetNetDynamicShapeStatus();
  }

  // Adjust kernel graph before run graph.
  device_context->kernel_executor_->PreprocessBeforeRun(graph);

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
                                            const std::map<KernelWithIndex, TensorPtr> &op_output,
                                            const std::map<AnfNodePtr, size_t> &parameter_index,
                                            const std::vector<TensorPtr> &graph_inputs,
                                            InputTensorInfo *const input_tensor_info) {
  MS_EXCEPTION_IF_NULL(session_);
  session_->GetOpInputTensors(kernel, op_output, parameter_index, graph_inputs, input_tensor_info);
}

TensorPtr GraphCompiler::GetSingleOpInputTensorByIndex(const CNodePtr &kernel,
                                                       const std::map<KernelWithIndex, TensorPtr> &op_output,
                                                       const std::map<AnfNodePtr, size_t> &parameter_index,
                                                       const std::vector<TensorPtr> &graph_inputs,
                                                       InputTensorInfo *const input_tensor_info, size_t input_index) {
  MS_EXCEPTION_IF_NULL(session_);
  return session_->GetOpInputTensorByIndex(kernel, op_output, parameter_index, graph_inputs, input_tensor_info,
                                           input_index);
}

void GraphCompiler::GetSingleOpRunInfoAndGraphInfo(const CNodePtr &kernel, const InputTensorInfo &tensor_info,
                                                   bool use_dynamic_shape_process,
                                                   session::BackendOpRunInfoPtr *op_run_info, GraphInfo *graph_info,
                                                   const GraphOutputInfo *const graph_output_info) {
  MS_EXCEPTION_IF_NULL(session_);
  MS_EXCEPTION_IF_NULL(graph_info);
  *op_run_info = session_->GetSingleOpRunInfo(kernel, *graph_info, tensor_info, graph_output_info);
  (*op_run_info)->base_op_run_info.use_dynamic_shape_process = use_dynamic_shape_process;
  session_->GetSingleOpGraphInfo(kernel, tensor_info, graph_info, *op_run_info);
  MS_EXCEPTION_IF_NULL(*op_run_info);
  (*op_run_info)->base_op_run_info.graph_info = *graph_info;
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
                                   std::map<KernelWithIndex, tensor::TensorPtr> *op_output_map) const {
  MS_EXCEPTION_IF_NULL(session_);
  session_->HandleOpInputs(input_kernels_with_index, ref_count, op_output_map);
}

void GraphCompiler::UpdateForwardOpOutputRefCount(const std::vector<tensor::TensorPtr> &input_tensor,
                                                  std::map<std::string, size_t> *forward_op_output_tensor_id) const {
  MS_EXCEPTION_IF_NULL(session_);
  MS_EXCEPTION_IF_NULL(forward_op_output_tensor_id);
  session_->ReleaseForwardOpOutput(input_tensor, forward_op_output_tensor_id);
}

void GraphCompiler::RecoverGraphOutput(const AnfNodePtr &kernel, const VectorRef &op_outputs,
                                       const std::map<KernelWithIndex, size_t> &ref_count,
                                       std::map<KernelWithIndex, TensorPtr> *op_output_map,
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
