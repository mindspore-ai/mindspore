/**
 * Copyright 2021-2025 Huawei Technologies Co., Ltd
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

#include "backend/ms_backend/runtime/graph_scheduler/base/graph_scheduler.h"
#include <algorithm>
#include <functional>
#include <queue>
#include <vector>
#include <set>
#include <map>
#include <utility>
#include <memory>
#include <string>
#include "primitive/sequence_ops.h"
#include "primitive/framework_ops.h"
#include "primitive/math_op_name.h"
#include "mindspore/core/mindrt/include/thread/threadpool.h"
#include "backend/ms_backend/runtime/graph_scheduler/base/scheduler_helper.h"
#include "backend/ms_backend/runtime/actors/base/memory_manager_actor.h"
#include "backend/ms_backend/runtime/actors/base/kernel_async_launch_actor.h"
#include "backend/ms_backend/runtime/actors/dynamic_shape/kernel_async_infer_actor.h"
#include "backend/ms_backend/runtime/actors/dynamic_shape/kernel_async_resize_actor.h"
#include "backend/ms_backend/runtime/actors/base/debug_actor.h"
#include "backend/ms_backend/runtime/actors/base/profiler_actor.h"
#include "backend/ms_backend/runtime/actors/base/recorder_actor.h"
#include "backend/ms_backend/runtime/graph_scheduler/optimizer/optimizer.h"
#include "backend/ms_backend/runtime/graph_scheduler/optimizer/memory_actor_insert.h"
#include "backend/ms_backend/runtime/graph_scheduler/optimizer/invalid_data_arrow_elimination.h"
#include "backend/ms_backend/runtime/graph_scheduler/optimizer/batch_data_arrow_fusion.h"
#include "backend/ms_backend/runtime/graph_scheduler/optimizer/multi_actor_fusion.h"
#include "backend/ms_backend/runtime/graph_scheduler/base/parameter_store.h"
#include "backend/ms_backend/runtime/graph_scheduler/base/graph_parameter_store.h"
#include "backend/ms_backend/runtime/graph_executor/kernel_capture/graph_capture_manager.h"
#include "include/runtime/hardware_abstract/device_context/device_context_manager.h"
#include "include/runtime/utils/runtime_conf/runtime_conf.h"
#include "include/runtime/utils/runtime_conf/thread_bind_core.h"
#include "include/runtime/pipeline/pipeline.h"
#include "backend/ms_backend/runtime/graph_executor/pipeline/runtime_pipeline.h"
#include "tools/profiler/profiler.h"
#include "actor/actormgr.h"
#include "async/async.h"
#include "device_address/device_address.h"
#include "include/backend/common/kernel_graph/anf_runtime_algorithm.h"
#include "include/utils/callback.h"
#include "include/utils/anfalgo.h"
#include "include/utils/parallel_context.h"
#include "include/backend/common/pass_manager/helper.h"
#include "utils/config_manager.h"
#include "utils/log_adapter.h"
#include "utils/convert_utils.h"
#include "utils/ms_context.h"
#include "utils/profile.h"
#include "include/runtime/hardware_abstract/kernel_base/common_utils.h"
#if !defined(_WIN32) && !defined(_WIN64) && !defined(__APPLE__)
#include "include/cluster/topology/compute_graph_node.h"
#endif
#include "tools/profiler/profiling.h"
#include "include/utils/common.h"
#include "include/runtime/hardware_abstract/collective/collective_manager.h"
#if defined(__linux__) && defined(WITH_BACKEND)
#include "include/cluster/topology/cluster_context.h"
#else
#include "include/cluster/topology/dummy_cluster_context.h"
#endif
#include "abstract/ops/primitive_infer_map.h"
#include "utils/file_utils.h"

#if !defined(_WIN32) && !defined(_WIN64) && !defined(__APPLE__)
#include "utils/numa_interface.h"
#endif

#if !defined(_WIN32) && !defined(_WIN64) && !defined(__APPLE__)
#include "include/utils/signal_util.h"
#endif

#include "primitive/auto_generate/gen_ops_primitive_d.h"
#include "primitive/auto_generate/gen_ops_primitive_l.h"
#include "primitive/auto_generate/gen_ops_primitive_m.h"
#include "primitive/auto_generate/gen_ops_primitive_s.h"
#include "primitive/auto_generate/gen_ops_primitive_u.h"

namespace mindspore {
namespace runtime {
using distributed::cluster::ClusterContext;
using distributed::collective::CollectiveManager;
namespace {
constexpr char kNumaEnableEnv[] = "MS_ENABLE_NUMA";
constexpr char kNumaEnableEnv2[] = "DATASET_ENABLE_NUMA";

// For the transform state synchronization.
constexpr char kTransformFinishPrefix[] = "TRANSFORM_FINISH_";
constexpr char kTransformFinishReady[] = "1";
static const size_t kRetry = 200;
static const size_t kInterval = 3;

static constexpr size_t kAsyncLaunchThreadNum = 1;
static constexpr size_t kMultiPipelineThreadNum = 3;

static constexpr size_t kMaxBindCoreThreadNum = 5;
static const size_t kBindCoreThreadStep = 5;

bool GetNeedSyncStream(const GraphCompilerInfo &graph_compiler_info) {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  static const bool enable_internal_kernel = ms_context->IsEnableInferBoost();
  static auto enable_syn = common::GetEnv("MS_SYNC_RUN");
  if ((enable_internal_kernel || IsInferPhase(graph_compiler_info.graph_phase_)) && enable_syn != "on") {
    return false;
  }
  const auto &graphs = graph_compiler_info.graphs_;
  if (graphs.empty() && graph_compiler_info.control_nodes_.size() > 1) {
    return true;
  }
  if (graphs.empty()) {
    MS_LOG(INTERNAL_EXCEPTION) << "#dmsg#Runtime error info:#dmsg#No graphs found in GraphCompilerInfo";
  }
  MS_EXCEPTION_IF_NULL(graphs[0]);
  return GraphPipelineCompiling();
}

int64_t GetLoopCount(const GraphCompilerInfo &graph_compiler_info) {
  const auto &graphs = graph_compiler_info.graphs_;
  if (graphs.empty() && graph_compiler_info.control_nodes_.size() > 1) {
    return 1;
  }
  if (graphs.empty()) {
    MS_LOG(INTERNAL_EXCEPTION) << "#dmsg#Runtime error info:#dmsg#No graphs found in GraphCompilerInfo";
  }
  MS_EXCEPTION_IF_NULL(graphs[0]);
  auto loop_count = ConfigManager::GetInstance().iter_num();
  if ((graph_compiler_info.strategy_ == GraphExecutionStrategy::kStep) ||
      (graphs.size() == 1 && graphs[0]->is_loop_count_sink()) || JitPipelineCompiling()) {
    loop_count = 1;
  }

  return loop_count;
}

bool IsDeviceTypeNotSame(const DeviceContext *from_device_context, const DeviceContext *to_device_context) {
  MS_EXCEPTION_IF_NULL(from_device_context);
  MS_EXCEPTION_IF_NULL(to_device_context);

  if (from_device_context->GetDeviceType() == to_device_context->GetDeviceType()) {
    return false;
  } else {
    return true;
  }
}

inline bool IsSingleOpActorSet(const ActorSet *actor_set) {
  MS_EXCEPTION_IF_NULL(actor_set);
  return actor_set->kernel_actors_.size() == 1;
}

bool IsTakenOverByControlFlow(const AnfNodePtr &front_node, const KernelGraphPtr &graph,
                              const ControlNodeParserPtr &parser) {
  MS_EXCEPTION_IF_NULL(front_node);
  MS_EXCEPTION_IF_NULL(graph);
  if (common::AnfAlgo::IsCallNode(front_node)) {
    return true;
  }

  if (parser != nullptr && parser->IsInited() && (!parser->IsSameKernelGraphGroup(front_node, graph))) {
    return true;
  }

  return false;
}

void ClearNodeInfo(const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);

  // Clear input parameter device tensor and device tensor store.
  for (const auto &input_node : graph->input_nodes()) {
    MS_EXCEPTION_IF_NULL(input_node);
    if (!input_node->isa<Parameter>()) {
      continue;
    }
    auto parameter = input_node->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(parameter);
    parameter->DecreaseUsedGraphCount();
    // Only the parameter has no graph used, then clear the device tensor.
    if (parameter->used_graph_count() != 0) {
      continue;
    }
    auto front_input_node = AnfAlgo::FetchFrontNodeByBackendNode(input_node, *graph);
    DeviceTensorStore::GetInstance().Remove(front_input_node.get());
    size_t output_num = AnfAlgo::GetOutputTensorNum(input_node);
    for (size_t index = 0; index < output_num; ++index) {
      if (AnfAlgo::OutputAddrExist(input_node, index)) {
        AnfAlgo::SetOutputAddr(nullptr, nullptr, index, input_node);
      }
    }
  }

  // Clear input value node device tensor and device tensor store.
  for (const auto &value_node : graph->graph_value_nodes()) {
    auto front_value_node = AnfAlgo::FetchFrontNodeByBackendNode(value_node, *graph);
    DeviceTensorStore::GetInstance().Remove(front_value_node.get());
    if (AnfAlgo::OutputAddrExist(value_node, 0)) {
      AnfAlgo::SetOutputAddr(nullptr, nullptr, 0, value_node);
    }
  }

  // Clear cnode device tensor.
  for (const auto &cnode : graph->execution_order()) {
    size_t output_num = AnfAlgo::GetOutputTensorNum(cnode);
    for (size_t index = 0; index < output_num; ++index) {
      if (AnfAlgo::OutputAddrExist(cnode, index)) {
        AnfAlgo::SetOutputAddr(nullptr, nullptr, index, cnode);
      }
    }
  }
}

// Check whether this graph could optimize input data prepare.
bool CheckInputOptimizeCondition(const GraphCompilerInfo &graph_compiler_info) {
  auto graphs = graph_compiler_info.graphs_;
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (!ms_context->IsKByKExecutorMode()) {
    MS_LOG(DEBUG) << "Context is not kbk executor mode, enable input optimize failed.";
    return false;
  }

  for (const auto &graph : graphs) {
    MS_EXCEPTION_IF_NULL(graph);
    if (!graph->enable_input_optimize()) {
      MS_LOG(DEBUG) << "Disable input optimize.";
      return false;
    }
  }

  return true;
}

void UpdateInputOptimizeForCurActorSet(const ActorSet *actor_set) {
  MS_EXCEPTION_IF_NULL(actor_set);
  ActorDispatcher::set_enable_input_optimize_for_cur_actor_set(actor_set->enable_input_optimize_);
}

// Try to enable input optimize.
void TryEnableInputOptimize(const GraphCompilerInfo &graph_compiler_info, ActorSet *const actor_set) {
  MS_EXCEPTION_IF_NULL(actor_set);

  actor_set->enable_input_optimize_ = CheckInputOptimizeCondition(graph_compiler_info);
  UpdateInputOptimizeForCurActorSet(actor_set);
  if (EnableInputOptimize()) {
    MS_LOG(INFO) << "Enable input optimize for actor set: " << actor_set->name_
                 << ", phase: " << graph_compiler_info.graph_phase_;
  }
}

void TryEnableCaptureGraph(const GraphCompilerInfo &graph_compiler_info) {
  if (!EnableInputOptimize() && GraphCaptureManager::GetInstance().GetEnableGraphCapture()) {
    MS_LOG(EXCEPTION) << "You have to enable input optimize before enable capture graph. It seems that the "
                         "input optimize feature verification has failed. Please enable the debug "
                         "log by export MS_SUBMODULE_LOG_v=\"{RUNTIME_FRAMEWORK:0}\" and search for the field 'enable "
                         "input optimize failed.' to determine whether "
                         "input optimize is having issues in some of your scenarios. ";
  }
}

// Check whether this graph could be executed as kbk graph mode which disable kernel actor message mechanism.
bool CheckKbkSubGraphExecConditon(const std::vector<KernelGraphPtr> &graphs) {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (!ms_context->IsKByKExecutorMode()) {
    MS_LOG(INFO) << "Disable kbk sub graph mode because the executor mode is not kbk.";
    return false;
  }
  return true;
}

void UpdateEnableSubGraphExecForCurActorSet(const ActorSet *actor_set) {
  MS_EXCEPTION_IF_NULL(actor_set);
  ActorDispatcher::set_enable_sub_graph_execute_for_cur_actor_set(actor_set->enable_kbk_sub_graph_execute_);
}

void ChangeGraphMode(const GraphCompilerInfo &graph_compiler_info) {
  if (EnableKbkSubGraphExecute()) {
    for (size_t i = 0; i < graph_compiler_info.graphs_.size(); ++i) {
      const auto &graph = graph_compiler_info.graphs_[i];
      MS_LOG(DEBUG) << "Enable kbk subgraph execute and set run mode for graph: " << graph->graph_id()
                    << " to GraphMode.";
      if (graph->RunMode() == device::RunMode::kGraphMode) {
        MS_LOG(WARNING) << "Can not set graph sink when execute sub graph kernel by kernel mode.";
      } else {
        graph->set_run_mode(device::RunMode::kGraphMode);
      }
    }
  }
}

// Try to enable the actor set execute as kbk sub graph mode which disable kernel actor message mechanism.
void TryEnableKbkSubGraphExecMode(const GraphCompilerInfo &graph_compiler_info, ActorSet *const actor_set) {
  MS_EXCEPTION_IF_NULL(actor_set);

  actor_set->enable_kbk_sub_graph_execute_ = CheckKbkSubGraphExecConditon(graph_compiler_info.graphs_);
  UpdateEnableSubGraphExecForCurActorSet(actor_set);
  // Adaptive operator direct Launch mode (no message mechanism).
  ChangeGraphMode(graph_compiler_info);
  if (EnableKbkSubGraphExecute()) {
    MS_LOG(INFO) << "Enable kbk subgraph execute mode for actor set: " << actor_set->name_
                 << ", phase: " << graph_compiler_info.graph_phase_;
  }
}

void InitGraphParameterStore(const GraphCompilerInfo &graph_compiler_info) {
  auto enable_kbk_sub_graph_execute = CheckKbkSubGraphExecConditon(graph_compiler_info.graphs_);
  ActorDispatcher::set_enable_sub_graph_execute_for_cur_actor_set(enable_kbk_sub_graph_execute);
  auto enable_input_optimize = enable_kbk_sub_graph_execute && CheckInputOptimizeCondition(graph_compiler_info);
  ActorDispatcher::set_enable_input_optimize_for_cur_actor_set(enable_input_optimize);
  if (!enable_input_optimize) {
    return;
  }

  ParameterStore::GetInstance().Insert(graph_compiler_info.name_);
  auto graph_parameter_store = ParameterStore::GetInstance().GetGraphParameterStore();
  MS_EXCEPTION_IF_NULL(graph_parameter_store);
  auto outer_size = graph_compiler_info.origin_parameters_order_.size();
  graph_parameter_store->Resize(outer_size);
  MS_LOG(INFO) << "Init graph parameter store: " << graph_compiler_info.name_ << ", outer size: " << outer_size;
  for (size_t i = 0; i < graph_compiler_info.origin_parameters_order_.size(); ++i) {
    auto abstract = graph_compiler_info.origin_parameters_order_[i]->abstract();
    MS_EXCEPTION_IF_NULL(abstract);
    if (abstract->isa<abstract::AbstractTensor>()) {
      graph_parameter_store->SetPositionTensor(i, true);
    }
    auto inner_size = common::AnfAlgo::GetOutputNumByAbstract(abstract);
    graph_parameter_store->ResizePosition(i, inner_size);
    graph_parameter_store->SetFrontNodeToIndex(graph_compiler_info.origin_parameters_order_[i].get(), i);
    if (graph_compiler_info.origin_parameters_order_[i]->isa<Parameter>() &&
        (common::AnfAlgo::IsParameterWeight(graph_compiler_info.origin_parameters_order_[i]->cast<ParameterPtr>()))) {
      graph_parameter_store->SetPositionWeight(i, true);
    }
    MS_LOG(DEBUG) << "Init store inner: outer index: " << i << ", inner size: " << inner_size
                  << ", parameter: " << graph_compiler_info.origin_parameters_order_[i]->DebugString();
  }
}

void FetchContinuousMemoryInfo(const CNodePtr &node, bool is_input) {
  MS_EXCEPTION_IF_NULL(node);

  auto continuous_kernel_tensors = std::make_shared<std::vector<std::weak_ptr<KernelTensor>>>();
  if (is_input) {
    const auto &intput_sizes = AnfAlgo::GetNodeInputSizeList(node);
    for (size_t i = 0; i < intput_sizes.size(); ++i) {
      const auto &kernel_tensor = AnfAlgo::GetPrevNodeOutputKernelTensor(node, i, false);
      MS_EXCEPTION_IF_NULL(kernel_tensor);
      auto device_tensor = kernel_tensor->device_address();
      MS_EXCEPTION_IF_NULL(device_tensor);
      kernel_tensor->set_continuous_kernel_tensors(continuous_kernel_tensors);
      continuous_kernel_tensors->emplace_back(std::weak_ptr<KernelTensor>(kernel_tensor));
    }
  } else {
    const auto &kernel_mod = AnfAlgo::GetKernelMod(node);
    MS_EXCEPTION_IF_NULL(kernel_mod);
    const auto &output_sizes = kernel_mod->GetOutputSizeList();
    for (size_t i = 0; i < output_sizes.size(); ++i) {
      const auto &kernel_tensor = AnfAlgo::GetOutputKernelTensor(node, i, false);
      MS_EXCEPTION_IF_NULL(kernel_tensor);
      auto device_tensor = kernel_tensor->device_address();
      MS_EXCEPTION_IF_NULL(device_tensor);
      kernel_tensor->set_continuous_kernel_tensors(continuous_kernel_tensors);
      continuous_kernel_tensors->emplace_back(std::weak_ptr<KernelTensor>(kernel_tensor));
    }
  }
}

bool IsInternalParameterInParameterStore(const AnfNodePtr &front_node) {
  MS_EXCEPTION_IF_NULL(front_node);
  auto real_front_node = common::AnfAlgo::FetchRealNodeSkipMonadControl({front_node, 0}).first;
  MS_EXCEPTION_IF_NULL(real_front_node);
  if (EnableInputOptimize() && real_front_node->isa<Parameter>()) {
    return true;
  }
  return false;
}

KernelWithIndex FetchRealFrontNode(const KernelWithIndex &node_with_index, const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(node_with_index.first);
  auto real_node_with_index = common::AnfAlgo::FetchRealNodeSkipMonadControl(node_with_index);
  MS_EXCEPTION_IF_NULL(real_node_with_index.first);
  KernelWithIndex front_node_with_idx{nullptr, 0};
  if (IsInternalParameter(real_node_with_index.first, graph)) {
    front_node_with_idx = graph->GetFrontNodeByInternalParameter(real_node_with_index.first);
  } else {
    front_node_with_idx = graph->GetElementInTupleBackendFrontIndexMap(real_node_with_index.first);
    if (front_node_with_idx.first == nullptr) {
      front_node_with_idx = {AnfAlgo::FetchFrontNodeByBackendNode(real_node_with_index.first, *graph), 0};
    }
  }
  return front_node_with_idx;
}

void ContinuePipelineByCondition() {
  if (EnableRuntimeNewPipeline()) {
    if (ActorDispatcher::enable_runtime_multi_pipeline()) {
      RuntimePipeline::GetInstance().ContinueAll();
    } else if (ActorDispatcher::enable_async_launch_kernel()) {
      RuntimePipeline::GetInstance().launch_queue()->Continue();
    }

    // For performance, remove bind device operation in ExecuteLaunchKernelTask function, and need bind device for all
    // async launch kernel task.
    if (ActorDispatcher::enable_async_launch_kernel()) {
      RuntimePipeline::GetInstance().launch_queue()->BindDevice(RuntimePipeline::GetInstance().GetAllDeviceContexts());
    }
  }
}

void CheckInferPerformanceFeature(const GraphCompilerInfo &graph_compiler_info, const ActorSet *actor_set) {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  const bool enable_infer_boost = ms_context->IsEnableInferBoost();
  if (!enable_infer_boost) {
    return;
  }

  bool is_increment_graph = (graph_compiler_info.graph_phase_.find("increment") != std::string::npos);
  if (!EnableKbkSubGraphExecute()) {
    MS_LOG(WARNING) << "Executing the inference network without enabling KBK subgraph mode may not achieve optimal "
                       "performance, please check whether exist PyExecute kernel in graphs: "
                    << graph_compiler_info.name_;
  } else if (is_increment_graph) {
    MS_EXCEPTION_IF_NULL(actor_set);
    for (auto &super_kernel_actor : actor_set->super_kernel_actors_) {
      MS_EXCEPTION_IF_NULL(super_kernel_actor);
      const auto &kernel_runners = super_kernel_actor->kernel_actors();
      for (auto &kernel_runner : kernel_runners) {
        MS_EXCEPTION_IF_NULL(kernel_runner);
        auto *kernel_mod = kernel_runner->kernel_mod();
        MS_EXCEPTION_IF_NULL(kernel_mod);
        if (kernel_mod->IsNeedUpdateOutputShapeAndSize()) {
          MS_EXCEPTION_IF_NULL(kernel_runner->kernel());
          MS_EXCEPTION_IF_NULL(super_kernel_actor->graph());
          MS_LOG(EXCEPTION) << "The kernel(" << kernel_runner->kernel()->fullname_with_scope()
                            << ") need update output shape after launch, this kernel does not support converting "
                               "dynamic shape to static "
                               "shape feature of llm inference. The graph info: "
                            << super_kernel_actor->graph()->ToString();
        }
      }
    }
  }

  if (!is_increment_graph) {
    return;
  }

  bool enable_trace_memory = EnableTraceMemory();
  auto multi_sub_graph = graph_compiler_info.graphs_.size() > 1;
  // The trace memory can not use in multi-stream and multi-subgraph case.
  if (enable_trace_memory) {
    for (auto &graph : graph_compiler_info.graphs_) {
      MS_EXCEPTION_IF_NULL(graph);
      if (graph->enable_multi_stream()) {
        MS_LOG(EXCEPTION)
          << "The trace memory feature doesn't support multi-stream case, please eliminate "
             "multi-stream(communication and computing pperators use different stream), or disable "
             "trace memory feature by export MS_ENABLE_TRACE_MEMORY=off. Note: Disabling the trace "
             "memory feature will degrade memory management performance. Additionally, it will automatically disable "
             "the kernel group launch(parallel launch), which may reduce network execution "
             "performance.";
      }
    }

    if (multi_sub_graph) {
      MS_LOG(EXCEPTION)
        << "The increment/decode graph(" << graph_compiler_info.name_
        << ") has multi sub graph, it may be due to the cutting of graphs caused by "
           "heterogeneity(cpu kernel) or control flow."
        << " Sub graphs num: " << graph_compiler_info.graphs_.size()
        << ". The trace memory feature can not work in this case. Please eliminate heterogeneity(cpu "
           "kernel) and control flow, or disable trace memory feature by export MS_ENABLE_TRACE_MEMORY=off. Note: "
           "Disabling the trace "
           "memory feature will degrade memory management performance. Additionally, it will automatically disable the "
           "kernel group launch(parallel launch), which may reduce network execution "
           "performance.";
    }
  }

  // The parallel dispatch feature can not use in multi-subgraph case.
  bool enable_parallel_dispatch = EnableParallelDispatchKernel();
  if (enable_parallel_dispatch && multi_sub_graph) {
    MS_LOG(EXCEPTION)
      << "The increment/decode graph(" << graph_compiler_info.name_
      << ") has multi sub graph, it may be due to the cutting of graphs caused by "
         "heterogeneity(cpu kernel) or control flow."
      << " Sub graphs num: " << graph_compiler_info.graphs_.size()
      << ". The kernel group launch(parallel launch) feature can not work in this case. Please "
         "eliminate heterogeneity(cpu kernel) and control flow, or disable kernel group launch feature.";
  }

  bool enable_capture_graph = EnableCaptureGraph();
  // The capture graph can not use in multi-stream and multi-subgraph case.
  if (enable_capture_graph) {
    for (auto &graph : graph_compiler_info.graphs_) {
      MS_EXCEPTION_IF_NULL(graph);
      if (graph->enable_multi_stream()) {
        MS_LOG(EXCEPTION)
          << "The capture graph feature doesn't support multi-stream case, please eliminate "
             "multi-stream(communication and computing pperators use different stream), or disable "
             "capture graph feature by set capture graph false in your python script. Note: Disabling the capture "
             "graph feature will reduce network execution performance. ";
      }
    }

    if (multi_sub_graph) {
      MS_LOG(EXCEPTION) << "The increment/decode graph(" << graph_compiler_info.name_
                        << ") has multi sub graph, it may be due to the cutting of graphs caused by "
                           "heterogeneity(cpu kernel) or control flow."
                        << " Sub graphs num: " << graph_compiler_info.graphs_.size()
                        << ". The capture graph feature can not work in this case. Please eliminate heterogeneity(cpu "
                           "kernel) and control flow, or disable capture graph feature by set capture graph false in "
                           "your python script. Note: "
                           "Disabling the capture will  reduce network execution performance.";
    }
  }
}
}  // namespace

GraphScheduler &GraphScheduler::GetInstance() noexcept {
  static GraphScheduler instance{};
  return instance;
}

void GraphScheduler::Clear(const ActorInfo &actor_info, const std::vector<KernelGraphPtr> &graphs,
                           const std::vector<AnfNodePtr> &root_graph_parameters,
                           const ControlNodeParserPtr &parser) noexcept {
  // Terminate the actors of actor info.
  if (actors_.count(actor_info) > 0) {
    auto actor_manager = ActorMgr::GetActorMgrRef();
    if (actor_manager == nullptr) {
      MS_LOG(ERROR) << "Actor manager is not exist.";
      return;
    }
    auto actor_set = actors_[actor_info];
    const auto &base_actors = actor_set->all_actors_;
    for (auto &base_actor : base_actors) {
      MS_EXCEPTION_IF_NULL(base_actor);
      EraseActor(base_actor->GetAID().Name());
      if (base_actor->parent_fusion_actor_ == nullptr) {
        actor_manager->Terminate(base_actor->GetAID());
      }
    }
  }

  // Clear device tensor and device tensor store.
  for (auto &graph : graphs) {
    ClearNodeInfo(graph);
  }

  if (parser != nullptr && parser->IsInited()) {
    const auto &front_value_nodes = parser->front_value_nodes();
    for (const auto &front_value_node : front_value_nodes) {
      const auto &node = front_value_node.first.first;
      size_t index = front_value_node.first.second;
      if (AnfAlgo::OutputAddrExist(node, index)) {
        AnfAlgo::SetOutputAddr(nullptr, nullptr, index, node);
      }
    }
  }

  // Clear the member of DeviceTensorStore.
  for (auto &root_graph_parameter : root_graph_parameters) {
    DeviceTensorStore::GetInstance().Remove(root_graph_parameter.get());
  }

  if (EnableInputOptimize()) {
    ParameterStore::GetInstance().Clear(actor_info);
  }

  // Clear global maps of actor info.
  (void)actors_.erase(actor_info);
}

void GraphScheduler::Clear() {
  GraphCaptureManager::GetInstance().Finalize();
  // Terminate all actors.
  auto actor_manager = ActorMgr::GetActorMgrRef();
  MS_EXCEPTION_IF_NULL(actor_manager);
  actor_manager->Finalize();

  // Clear the member of DeviceTensorStore.
  DeviceTensorStore::GetInstance().Clear();

  // Clear the member of ParameterStore.
  if (EnableInputOptimize()) {
    ParameterStore::GetInstance().Clear();
  }

  // Clear global maps.
  actors_.clear();
  ClearAllActors();

  // Clear all cache memory info before process exits.
  MemoryTraceManager::GetInstance().ClearAllCache();
}

void GraphScheduler::ClearActorData(const ActorSet *actor_set) {
  MS_EXCEPTION_IF_NULL(actor_set);

  // Clear the member of KernelTensorCopyStore.
  KernelTensorCopyStore::GetInstance().Clear();

  // Clear the output tensors of output actor.
  if (actor_set->output_actor_ != nullptr) {
    actor_set->output_actor_->outputs_.clear();
    actor_set->output_actor_->outputs_.resize(actor_set->output_actor_->outputs_num_);
  }

  if (EnableInputOptimize()) {
    ParameterStore::GetInstance().GetGraphParameterStore()->ReleaseData();
  } else {
    for (auto &data_source_actor : actor_set->data_source_actors_) {
      MS_EXCEPTION_IF_NULL(data_source_actor);
      data_source_actor->ReleaseData();
    }
  }

  for (auto &super_kernel_actor : actor_set->super_kernel_actors_) {
    MS_EXCEPTION_IF_NULL(super_kernel_actor);
    super_kernel_actor->memory_free_lists_ = std::queue<std::vector<KernelTensorPtr>>();
  }

  control_node_scheduler_.ClearActorData(actor_set->control_actors_.get());

  // At the end of the step, the op data sent to the stack actor in each actor should be clear.
  const auto &total_actors = actor_set->all_actors_;
  for (auto &actor : total_actors) {
    MS_EXCEPTION_IF_NULL(actor);
    actor->to_stack_data_.clear();
  }
}

using DataArrowLinkFunc = void (GraphScheduler::*)(AbstractActor *const, AbstractActor *const, const KernelWithIndex &,
                                                   const KernelWithIndex &, const KernelGraphPtr &);
static std::map<KernelTransformType, DataArrowLinkFunc> kKernelTypeToLinkFunc;

void GraphScheduler::Initialize() {
  if (init_) {
    return;
  }
  init_ = true;

  BindNumaNode();
  (void)kKernelTypeToLinkFunc.emplace(KernelTransformType::kGraphParameterStore,
                                      &GraphScheduler::LinkDataArrowForGraphParameterStore);
  (void)kKernelTypeToLinkFunc.emplace(KernelTransformType::kHostDataSourceActor,
                                      &GraphScheduler::LinkDataArrowForHostDSActor);
  (void)kKernelTypeToLinkFunc.emplace(KernelTransformType::kKernelActor, &GraphScheduler::LinkDataArrowForKernelActor);
  (void)kKernelTypeToLinkFunc.emplace(KernelTransformType::kSuperKernelActor,
                                      &GraphScheduler::LinkDataArrowForBaseActor);
  (void)kKernelTypeToLinkFunc.emplace(KernelTransformType::kAnyTypeKernelActor,
                                      &GraphScheduler::LinkDataArrowForBaseActor);
  (void)kKernelTypeToLinkFunc.emplace(KernelTransformType::kDeviceTensorStore,
                                      &GraphScheduler::LinkDataArrowForDeviceTensorStore);
  (void)kKernelTypeToLinkFunc.emplace(KernelTransformType::kInternalParameter,
                                      &GraphScheduler::LinkDataArrowForInternalParameter);

  // Create the thread pool of actor runtime and Set the OMP_NUM_THREADS env.
  size_t actor_thread_num = 0;
  size_t actor_and_kernel_thread_num = 0;
  ComputeThreadNums(&actor_thread_num, &actor_and_kernel_thread_num);
  default_actor_thread_num_ = actor_thread_num;

  if (EnableRuntimePipeline() && EnableRuntimeNewPipeline()) {
    if (actor_thread_num > kMultiPipelineThreadNum) {
      actor_thread_num -= kMultiPipelineThreadNum;
      actor_and_kernel_thread_num -= kMultiPipelineThreadNum;
    } else if (actor_thread_num > kAsyncLaunchThreadNum) {
      actor_thread_num -= kAsyncLaunchThreadNum;
      actor_and_kernel_thread_num -= kAsyncLaunchThreadNum;
    }
  }
  auto actor_manager = ActorMgr::GetActorMgrRef();
  MS_EXCEPTION_IF_NULL(actor_manager);
  size_t actor_queue_size = 81920;

  auto ret =
    actor_manager->Initialize(true, actor_thread_num, actor_and_kernel_thread_num, actor_queue_size, numa_cpus_);
  if (ret != MINDRT_OK) {
    MS_LOG(INTERNAL_EXCEPTION) << "#dmsg#Runtime error info:#dmsg#Actor manager init failed.";
  }
  common::SetOMPThreadNum();
  MS_LOG(INFO) << "The actor thread number: " << actor_thread_num
               << ", the kernel thread number: " << (actor_and_kernel_thread_num - actor_thread_num);

  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);

  auto runtime_threads_num = context_ptr->get_param<uint32_t>(MS_CTX_RUNTIME_NUM_THREADS);
  if (runtime::RuntimeConf::GetInstance()->IsDispatchThreadsNumConfigured()) {
    runtime_threads_num = runtime::RuntimeConf::GetInstance()->dispatch_threads_num();
  }
  if (default_actor_thread_num_ <= kAsyncLaunchThreadNum && EnableRuntimePipeline() &&
      runtime_threads_num == static_cast<uint32_t>(1)) {
    MS_LOG(WARNING) << "The number of actor threads is only: " << default_actor_thread_num_
                    << ", and pipelined runtime optimization include the switch inline is not enabled, the performance "
                       "may not reach the optimal level. Please "
                       "increase the value of `runtime_num_threads` in context or not set `runtime_num_threads`.";
  }

  BuildAndScheduleGlobalActor();

  // Runtime pipeline optimization must enable when executing graph kernel by kernel mode.
  bool disable_kbk_sub_graph_execute = (default_actor_thread_num_ <= kAsyncLaunchThreadNum) || !EnableRuntimePipeline();
  if (disable_kbk_sub_graph_execute) {
    ActorDispatcher::set_disable_kbk_sub_graph_execute(true);
  }
}

void GraphScheduler::BuildAndScheduleGlobalActor() {
  auto actor_manager = ActorMgr::GetActorMgrRef();
  MS_EXCEPTION_IF_NULL(actor_manager);

  // Create and schedule memory manager actor.
  auto &memory_manager_actor = MemoryManagerActor::GetInstance();
  MS_EXCEPTION_IF_NULL(memory_manager_actor);
  memory_manager_aid_ = memory_manager_actor->GetAID();
  auto base_actor = static_cast<ActorReference>(memory_manager_actor);
  // Bind single thread to response to memory alloc and free quickly.
  (void)actor_manager->Spawn(base_actor, true);

  // Create and schedule recorder actor.
  bool recorder_actor_need = false;
  if (profiler::ProfilerManager::GetInstance()->GetProfilingEnableFlag()) {
    recorder_actor_need = true;
  }
  if (recorder_actor_need) {
    auto recorder_actor = std::make_shared<RecorderActor>();
    MS_EXCEPTION_IF_NULL(recorder_actor);
    recorder_aid_ = &(recorder_actor->GetAID());
    auto base_recorder_actor = static_cast<ActorReference>(recorder_actor);
    (void)actor_manager->Spawn(base_recorder_actor, true);
  }

  bool profiler_actor_need = false;
  auto profiler = profiler::Profiler::GetInstance(kAscendDevice);
  if ((profiler != nullptr && profiler->IsInitialized())) {
    profiler_actor_need = true;
  }
  if (profiler_actor_need) {
    auto profiler_actor = std::make_shared<ProfilerActor>();
    MS_EXCEPTION_IF_NULL(profiler_actor);
    profiler_aid_ = &(profiler_actor->GetAID());
    auto base_profiler_actor = static_cast<ActorReference>(profiler_actor);
    (void)actor_manager->Spawn(base_profiler_actor, true);
  }

  // Create and schedule debug actor.
  // debugger_actor_need is true for CPU when e2e dump is enabled and for Ascend and GPU is true when debugger or dump
  // is enabled.
  constexpr char kParse[] = "DumpJsonParserParse";
  static auto parse_callback = callback::CommonCallback::GetInstance().GetCallback<void>(kParse);
  if (parse_callback) {
    parse_callback();
  } else {
    MS_LOG(WARNING) << "Failed to get DumpJsonParserParse, data dump function may not work.";
  }
  constexpr char kE2eDumpEnabled[] = "E2eDumpEnabled";
  static auto e2e_dump_enabled_callback = callback::CommonCallback::GetInstance().GetCallback<bool>(kE2eDumpEnabled);
  bool debugger_actor_need = false;
  if (e2e_dump_enabled_callback) {
    debugger_actor_need = e2e_dump_enabled_callback();
  } else {
    MS_LOG(WARNING) << "Failed to get e2e_dump_enabled, data dump function may not work.";
  }
#ifdef ENABLE_DEBUGGER
  constexpr char kDebuggerBackendEnabled[] = "DebuggerBackendEnabled";
  static auto debugger_backend_enabled_callback =
    callback::CommonCallback::GetInstance().GetCallback<bool>(kDebuggerBackendEnabled);
  if (debugger_backend_enabled_callback) {
    debugger_actor_need = debugger_backend_enabled_callback() ? true : debugger_actor_need;
  } else {
    MS_LOG(WARNING) << "Failed to get DebuggerBackendEnabled callback, data dump function may not work.";
  }
#endif
  // if silent check is enabled, create debugger actor for CheckSum
  constexpr char kNeedEnableCheckSum[] = "NeedEnableCheckSum";
  static const auto need_enable_checksum =
    callback::CommonCallback::GetInstance().GetCallback<bool>(kNeedEnableCheckSum);
  if (need_enable_checksum) {
    debugger_actor_need = need_enable_checksum() ? true : debugger_actor_need;
  } else {
    MS_LOG(WARNING) << "Failed to get NeedEnableCheckSum callback, silent detect function may not work.";
  }
  if (debugger_actor_need) {
    auto debug_actor = std::make_shared<DebugActor>();
    MS_EXCEPTION_IF_NULL(debug_actor);
    debug_aid_ = &(debug_actor->GetAID());
    auto base_debug_actor = static_cast<ActorReference>(debug_actor);
    (void)actor_manager->Spawn(base_debug_actor, true);
  }
}

ActorSet *GraphScheduler::Transform(const GraphCompilerInfo &graph_compiler_info) {
  struct ScopeCleaner {
    GraphScheduler *const scheduler_;
    explicit ScopeCleaner(GraphScheduler *scheduler) : scheduler_(scheduler) {}
    ~ScopeCleaner() {
      // Local maps and vectors clear.
      if (scheduler_ == nullptr) {
        return;
      }
      scheduler_->graph_output_to_actor_.clear();
      scheduler_->copy_actors_.clear();
      scheduler_->execution_order_running_ = false;
    }
  };
  // cppcheck-suppress unreadVariable
  ScopeCleaner cleaner(this);
  uint64_t start_time_0 = profiler::GetClockSyscnt();
  MS_LOG(INFO) << "Graph(" << graph_compiler_info.name_
               << ") transforms actor begin, strategy:" << kGraphExecutionStrategyStr.at(graph_compiler_info.strategy_);
  if (graph_compiler_info.graphs_.size() == 0 && graph_compiler_info.control_nodes_.size() <= 1) {
    MS_LOG(INTERNAL_EXCEPTION) << "#dmsg#Runtime error info:#dmsg#The number of graphs is zero.";
  }
  if (graph_compiler_info.graphs_.size() != graph_compiler_info.device_contexts_.size()) {
    MS_LOG(INTERNAL_EXCEPTION)
      << "#dmsg#Runtime error info:#dmsg#The number of graphs is not equal to the number of device contexts.";
  }

  if (graph_compiler_info.strategy_ == GraphExecutionStrategy::kPipelineWithExecutionOrder) {
    execution_order_running_ = true;
    graph_compiler_info.strategy_ = GraphExecutionStrategy::kPipeline;
  }
  InitGraphParameterStore(graph_compiler_info);
  PersistDeviceTensor(graph_compiler_info);
  uint64_t start_time_1 = profiler::GetClockSyscnt();
  const auto &actor_set = Build(graph_compiler_info);
  (void)profiler::CollectHostInfo(kModelNameRuntime, kEventCompileGraph, kStageBuild, start_time_1,
                                  profiler::GetClockSyscnt(), 1);
  MS_EXCEPTION_IF_NULL(actor_set);
  CacheGraphOutputToActor(graph_compiler_info);
  start_time_1 = profiler::GetClockSyscnt();
  PROF_START(GraphSchedulerLink);
  Link(actor_set.get(), graph_compiler_info);
  PROF_END(GraphSchedulerLink);

  // Process continuous memory and set flag for data prepare actor.
  ProcessContinuousMemoryInfo(actor_set, graph_compiler_info);

  (void)profiler::CollectHostInfo(kModelNameRuntime, kEventCompileGraph, kStageLink, start_time_1,
                                  profiler::GetClockSyscnt(), 1);
  DumpActor(actor_set.get(), graph_compiler_info);
  if (graph_compiler_info.strategy_ == GraphExecutionStrategy::kPipeline) {
    SchedulerHelper::CheckActorValid(actor_set.get());
  }

  start_time_1 = profiler::GetClockSyscnt();
  Optimize(actor_set, graph_compiler_info);
  (void)profiler::CollectHostInfo(kModelNameRuntime, kEventCompileGraph, kStageOptimize, start_time_1,
                                  profiler::GetClockSyscnt(), 1);
  DumpFinalActor(actor_set.get(), graph_compiler_info);
  MS_LOG(INFO) << "Graph(" << graph_compiler_info.name_ << ") transforms actor end.";

  for (const auto &graph : graph_compiler_info.graphs_) {
    MS_EXCEPTION_IF_NULL(graph);
    if (graph->is_dynamic_shape()) {
      actor_set->has_dynamic_shape_ = true;
      break;
    }
  }
  for (const auto &graph : graph_compiler_info.graphs_) {
    MS_EXCEPTION_IF_NULL(graph);
    if (graph->enable_multi_stream()) {
      actor_set->enable_multi_stream_ = true;
      break;
    }
  }
  for (const auto &graph : graph_compiler_info.graphs_) {
    MS_EXCEPTION_IF_NULL(graph);
    if (graph->has_kernel_need_user_data()) {
      actor_set->has_kernel_need_user_data_ = true;
      break;
    }
  }

  actor_set->all_actors_ = SchedulerHelper::CollectActors(actor_set.get());
  (void)profiler::CollectHostInfo(kModelNameRuntime, kEventCompileGraph, kStageGraphTransform, start_time_0,
                                  profiler::GetClockSyscnt(), 1);

  CheckInferPerformanceFeature(graph_compiler_info, actor_set.get());
  // save constants for UCE error to resume training
  static auto tft_save_constants = GET_COMMON_CALLBACK(TftSaveConstants, void, const std::vector<KernelGraphPtr> &);
  if (tft_save_constants != nullptr) {
    tft_save_constants(graph_compiler_info.graphs_);
  }
  return actor_set.get();
}

void GraphScheduler::SpawnMultiPipelineActor(ActorSet *const actor_set, ActorThreadPool *const thread_pool) {
  auto actor_manager = ActorMgr::GetActorMgrRef();
  MS_EXCEPTION_IF_NULL(actor_manager);

  bool enable_runtime_pipeline = EnableRuntimePipeline();
  ActorDispatcher::set_enable_async_launch_kernel(enable_runtime_pipeline &&
                                                  (EnableKbkSubGraphExecute() || !actor_set->kernel_actors_.empty()) &&
                                                  (default_actor_thread_num_ > kAsyncLaunchThreadNum));
  if (ActorDispatcher::enable_async_launch_kernel()) {
    thread_pool->DisableOccupiedActorThread();
  }
  if (ActorDispatcher::enable_async_launch_kernel() && !already_spawn_kernel_async_launch_actor_) {
    if (!EnableRuntimeNewPipeline()) {
      auto &kernel_async_launch_actor = KernelAsyncLaunchActor::GetInstance();
      MS_EXCEPTION_IF_NULL(kernel_async_launch_actor);
      (void)actor_manager->Spawn(kernel_async_launch_actor, false);

      kernel_async_launch_actor->Initialize();
    } else {
      // Early make worker_ to make sure get_thread_id safety. eg. InferShape GetValue for parameter.
      RuntimePipeline::GetInstance().launch_queue()->Init();
    }
    already_spawn_kernel_async_launch_actor_ = true;
  }

  // If enable runtime multi pipeline, async launch kernel will be enabled.
  ActorDispatcher::set_enable_runtime_multi_pipeline(
    enable_runtime_pipeline && actor_set->has_dynamic_shape_ &&
    (EnableKbkSubGraphExecute() || !actor_set->kernel_actors_.empty()) &&
    (default_actor_thread_num_ > kMultiPipelineThreadNum));
  if (ActorDispatcher::enable_runtime_multi_pipeline() && !already_spawn_kernel_async_infer_resize_actor_) {
    if (!EnableRuntimeNewPipeline()) {
      auto &kernel_async_infer_actor = KernelAsyncInferActor::GetInstance();
      MS_EXCEPTION_IF_NULL(kernel_async_infer_actor);
      (void)actor_manager->Spawn(kernel_async_infer_actor, false);

      auto &kernel_async_resize_actor = KernelAsyncResizeActor::GetInstance();
      MS_EXCEPTION_IF_NULL(kernel_async_resize_actor);
      (void)actor_manager->Spawn(kernel_async_resize_actor, false);

      kernel_async_infer_actor->Initialize();
      kernel_async_resize_actor->Initialize();
    } else {
      // Early make worker_ to make sure get_thread_id safety. eg. InferShape GetValue for parameter.
      RuntimePipeline::GetInstance().infer_queue()->Init();
      RuntimePipeline::GetInstance().resize_queue()->Init();
    }
    already_spawn_kernel_async_infer_resize_actor_ = true;
  }
}

void GraphScheduler::Schedule(const ActorSet *actor_set) {
  MS_EXCEPTION_IF_NULL(actor_set);
  const auto &actors = actor_set->all_actors_;
  // Schedule actors.
  auto actor_manager = ActorMgr::GetActorMgrRef();
  MS_EXCEPTION_IF_NULL(actor_manager);

  for (auto actor : actors) {
    MS_EXCEPTION_IF_NULL(actor);
    // The sub actors in the fusion actor do not participate in message interaction.
    if (actor->parent_fusion_actor_ == nullptr) {
      (void)actor_manager->Spawn(actor);
    } else {
      actor->Init();
    }
  }
}

void CheckBeforeGraphRun(ActorSet *const actor_set) {
  static auto tft_check_cb = GET_COMMON_CALLBACK(TftCheckBeforeGraphRun, void);
  if (tft_check_cb != nullptr) {
    tft_check_cb();
  }
  // Some exception could happen after one step is completed, need to check exception at the beginning to avoid thread
  // hanging.
  MsException::Instance().CheckException();
  // Check the actor set state.
  if (actor_set->is_execution_failed_) {
    MS_LOG(EXCEPTION) << "#umsg#Model execution error:#umsg#An error occurred in the previous step of this model "
                         "execution, and the current step cannot be executed.";
  }
}

template <typename T>
void ResetActorState(const std::vector<T> &actors, OpContext<KernelTensor> *const context) {
  for (auto &actor : actors) {
    if (actor != nullptr) {
      actor->ResetState(context);
    }
  }
}

void ClearControlActorData(ActorSet *const actor_set, OpContext<KernelTensor> *const context) {
  MS_LOG(INFO) << "Start to clean control actors data.";
  if (actor_set->control_actors_ != nullptr) {
    ResetActorState(actor_set->control_actors_->entrance_actors_, context);
    ResetActorState(actor_set->control_actors_->gather_actors_, context);
    ResetActorState(actor_set->control_actors_->switch_actors_, context);
    ResetActorState(actor_set->control_actors_->stack_actors_, context);
    ResetActorState(actor_set->control_actors_->exit_actors_, context);
  }
  MS_LOG(INFO) << "End to clean control actors data.";
}

void ClearKernelActorData(ActorSet *const actor_set, OpContext<KernelTensor> *const context) {
  MS_LOG(INFO) << "Start to clean kernel actors data.";
  for (auto &kernel_actor : actor_set->kernel_actors_) {
    if (kernel_actor == nullptr) {
      continue;
    }
    for (auto output_kernel_tensor : kernel_actor->GetOutputDeviceTensors()) {
      MS_EXCEPTION_IF_NULL(output_kernel_tensor);
      const auto &output_device_tensor = output_kernel_tensor->device_address();
      MS_EXCEPTION_IF_NULL(output_device_tensor);
      if (output_kernel_tensor->new_ref_count() != SIZE_MAX) {
        output_kernel_tensor->set_new_ref_count(0);
      }
    }
    kernel_actor->ResetState(context);
  }
  for (auto &super_kernel_actor : actor_set->super_kernel_actors_) {
    if (super_kernel_actor == nullptr) {
      continue;
    }
    for (auto &kernel_actor : super_kernel_actor->kernel_actors()) {
      if (kernel_actor == nullptr) {
        continue;
      }
      for (auto output_kernel_tensor : kernel_actor->GetOutputDeviceTensors()) {
        MS_EXCEPTION_IF_NULL(output_kernel_tensor);
        const auto &output_device_tensor = output_kernel_tensor->device_address();
        MS_EXCEPTION_IF_NULL(output_device_tensor);
        if (output_kernel_tensor->new_ref_count() != SIZE_MAX) {
          output_kernel_tensor->set_new_ref_count(0);
        }
      }
      kernel_actor->ResetState();
    }
    super_kernel_actor->ResetState(context);
  }
  MS_LOG(INFO) << "End to clean kernel actors data.";
}

void GraphScheduler::ResetActorState(ActorSet *const actor_set, OpContext<KernelTensor> *const context) {
  MS_LOG(INFO) << "Clear state start.";
  ClearKernelActorData(actor_set, context);
  ClearControlActorData(actor_set, context);

  actor_set->loop_count_actor_->ResetState(context);
  actor_set->output_actor_->ResetState(context);
  actor_set->is_execution_failed_ = false;
  ClearActorData(actor_set);
  if (actor_set != nullptr && actor_set->control_actors_ != nullptr) {
    control_node_scheduler_.ClearExitActorDeviceTensors(actor_set->control_actors_->exit_actors_);
  }
  MS_LOG(INFO) << "Clear state end.";
}

void RefreshGraphParameterStore(ActorSet *const actor_set, const VectorRef &args) {
  if (!EnableInputOptimize()) {
    return;
  }
  ParameterStore::GetInstance().SetChosenGraphName(actor_set->name_);
  auto graph_parameter_store = ParameterStore::GetInstance().GetGraphParameterStore();
  MS_EXCEPTION_IF_NULL(graph_parameter_store);
  MS_LOG(INFO) << "Input args size: " << args.size();
  graph_parameter_store->SetInputArgs(args);
  UpdateInputOptimizeForCurActorSet(actor_set);
}

void GraphScheduler::Run(ActorSet *const actor_set, const std::vector<std::vector<TensorPtr>> &input_tensors,
                         const VectorRef &args, GraphExecutionStrategy strategy) {
#if !defined(_WIN32) && !defined(_WIN64) && !defined(__APPLE__)
  if (!RegisterGlobalSignalHandler(DefaultIntHandler)) {
    MS_EXCEPTION(RuntimeError) << "Failed to register the callback signal handling.";
  }
#endif
  // If spin is turned on in the configuration, it will be turned off when entering RunGraph.
  if (runtime::Pipeline::Get().frontend_stage()->Spin() && !is_shut_spin_ && is_bind_core_) {
    runtime::Pipeline::Get().SetSpin(false);
    is_shut_spin_ = true;
  }
  MS_EXCEPTION_IF_NULL(actor_set);
  MS_EXCEPTION_IF_NULL(actor_set->data_prepare_actor_);

  CheckBeforeGraphRun(actor_set);

  // Create recorder actor in the running to support the profiler in callback scene.
  if (profiler::ProfilerManager::GetInstance()->GetProfilingEnableFlag() && (recorder_aid_ == nullptr)) {
    auto recorder_actor = std::make_shared<RecorderActor>();
    MS_EXCEPTION_IF_NULL(recorder_actor);
    recorder_aid_ = &(recorder_actor->GetAID());
    auto base_recorder_actor = static_cast<ActorReference>(recorder_actor);
    auto actor_manager = ActorMgr::GetActorMgrRef();
    MS_EXCEPTION_IF_NULL(actor_manager);
    (void)actor_manager->Spawn(base_recorder_actor, true);
    if (actor_set->loop_count_actor_ != nullptr) {
      actor_set->loop_count_actor_->recorder_aid_ = recorder_aid_;
    }
  }

  // Construct OpContext.
  actor_set->op_context_and_promises_.first = std::make_unique<OpContext<KernelTensor>>();
  actor_set->op_context_and_promises_.second = std::make_unique<std::vector<Promise<int>>>(1);
  actor_set->op_context_and_promises_.first->sequential_num_ = RandInt::Instance().Get();
  actor_set->op_context_and_promises_.first->results_ = actor_set->op_context_and_promises_.second.get();
  auto op_context_ptr = actor_set->op_context_and_promises_.first.get();

  ResetTraceMemoryStatus();
  // Trigger data prepare actor running.
  MS_EXCEPTION_IF_NULL(ActorMgr::GetActorMgrRef());
  auto thread_pool = ActorMgr::GetActorMgrRef()->GetActorThreadPool();
  MS_EXCEPTION_IF_NULL(thread_pool);
  UpdateEnableSubGraphExecForCurActorSet(actor_set);
  // Store step parameter input.
  UpdateInputOptimizeForCurActorSet(actor_set);
  RefreshGraphParameterStore(actor_set, args);
  SpawnMultiPipelineActor(actor_set, thread_pool);
  if (actor_set->is_multi_thread_execution_) {
    thread_pool->SetSpinCountMaxValue();
  }
  ActorDispatcher::set_is_multi_thread_execution(actor_set->is_multi_thread_execution_);
  ActorDispatcher::set_enable_multi_stream(actor_set->enable_multi_stream_);
  ActorDispatcher::set_has_kernel_need_user_data(actor_set->has_kernel_need_user_data_);
  ContinuePipelineByCondition();
  double start_time = GetTime();
  ActorDispatcher::SendSync(actor_set->data_prepare_actor_->GetAID(), &DataPrepareActor::PrepareData, input_tensors,
                            args, op_context_ptr, GraphExecutionStrategy::kPipeline);

  // Get the run result.
  auto result_future = actor_set->op_context_and_promises_.second->at(0).GetFuture();
  result_future.Wait();
  thread_pool->SetSpinCountMinValue();
  if (!result_future.IsOK()) {
    actor_set->is_execution_failed_ = true;
    // When temporary variable 'op_context' has beed set failed status, the main thread need wait other threads until
    // they finish respective task, otherwise segmentation fault will happen when these task access 'op_context',
    // because it has been destroyed.
    std::mutex mutex;
    std::unique_lock<std::mutex> locker(mutex);
    std::condition_variable thread_blocker;
    const int64_t kTimeToWait = 3;
    (void)thread_blocker.wait_for(locker, std::chrono::seconds(kTimeToWait));
    WaitRuntimePipelineFinish(op_context_ptr, "GraphSchedulerRun");
    if (EnableRuntimeNewPipeline()) {
      RuntimePipeline::GetInstance().PauseAll();
    }
    ResetPipelineAndTraceMemoryStatus();

    // trigger Train Fault Tolerance callbacks for resuming training
    static auto tft_process_error_cb =
      GET_COMMON_CALLBACK(TftProcessGraphRunError, void, const std::function<void()> &, const std::function<void()> &);
    if (tft_process_error_cb != nullptr) {
      tft_process_error_cb([this, actor_set, op_context_ptr]() { this->ResetActorState(actor_set, op_context_ptr); },
                           [actor_set]() { actor_set->is_execution_failed_ = false; });
    }

    // May set exception in the wait time, need throw the exception to avoid affecting the next execution.
    MsException::Instance().CheckException();
    MS_LOG(EXCEPTION) << actor_set->op_context_and_promises_.first->error_info_;
  }

  if (EnableRuntimeNewPipeline()) {
    RuntimePipeline::GetInstance().PauseAll();
  }
  ResetPipelineStatus();
  MsException::Instance().CheckException();
  double end_time = GetTime();
  const size_t kSecondsToMilliseconds = 1000;
  SetActorExecutionStrategy(actor_set, strategy, (end_time - start_time) * kSecondsToMilliseconds);
  (void)SkipOrResetCopyAction(true);
  (void)SkipOrResetSyncAction(true);

  // If spin is turned on in the configuration, it will be turned on when exiting RunGraph.
  if (is_shut_spin_ && is_bind_core_) {
    runtime::Pipeline::Get().SetSpin(true);
    is_shut_spin_ = false;
  }
}

void GraphScheduler::ChildAfterFork() {
  MS_LOG(DEBUG) << "GraphScheduler reinitialize after fork.";
  auto actor_manager = ActorMgr::GetActorMgrRef();
  MS_EXCEPTION_IF_NULL(actor_manager);

  if (already_spawn_kernel_async_infer_resize_actor_) {
    already_spawn_kernel_async_infer_resize_actor_ = false;
    if (!EnableRuntimeNewPipeline()) {
      actor_manager->ResetActorAfterFork(KernelAsyncInferActor::GetInstance());
      actor_manager->ResetActorAfterFork(KernelAsyncResizeActor::GetInstance());
    }
  }

  if (already_spawn_kernel_async_launch_actor_) {
    already_spawn_kernel_async_launch_actor_ = false;
    if (!EnableRuntimeNewPipeline()) {
      actor_manager->ResetActorAfterFork(KernelAsyncLaunchActor::GetInstance());
    }
  }

  MS_LOG(DEBUG) << "GraphScheduler reinitialize after fork done.";
}

bool GraphScheduler::CheckSingleThreadRunningCondition(ActorSet *const actor_set,
                                                       GraphExecutionStrategy strategy) const {
#if defined(_WIN32) || defined(_WIN64) || defined(__APPLE__)
  return false;
#endif

  // The step mode uses the default multi thread.
  if (strategy == GraphExecutionStrategy::kStep) {
    return false;
  }

  // The constraint condition of not supporting the single thread execution.
  if ((actor_set->control_actors_ != nullptr) || (actor_set->copy_actors_.size() > 0) ||
      (actor_set->super_kernel_actors_.size() > 0) || (actor_set->loop_count_actor_->loop_count() > 1) ||
      (actor_set->kernel_actors_.size() > ActorDispatcher::kSingleThreadExecutionActorMaxNum)) {
    return false;
  }

  return true;
}

#if defined(__linux__) && defined(BIND_CORE)
void GraphScheduler::GetRuntimeThreadIds(ActorThreadPool *thread_pool, std::vector<pthread_t> *threads) const {
  MS_EXCEPTION_IF_NULL(thread_pool);
  MS_EXCEPTION_IF_NULL(threads);
  auto ret = thread_pool->GetActorWorkerThreads(threads);
  if (ret != THREAD_OK) {
    MS_LOG(EXCEPTION) << "Get actor worker thread ids failed.";
  }

  if (threads->size() == default_actor_thread_num_) {
    return;
  }

  if (EnableRuntimePipeline() && EnableRuntimeNewPipeline()) {
    if (default_actor_thread_num_ > kMultiPipelineThreadNum) {
      const auto &infer_worker = RuntimePipeline::GetInstance().infer_queue()->worker();
      MS_EXCEPTION_IF_NULL(infer_worker);
      threads->push_back(infer_worker->native_handle());

      const auto &resize_worker = RuntimePipeline::GetInstance().resize_queue()->worker();
      MS_EXCEPTION_IF_NULL(resize_worker);
      threads->push_back(resize_worker->native_handle());

      const auto &launch_worker = RuntimePipeline::GetInstance().launch_queue()->worker();
      MS_EXCEPTION_IF_NULL(launch_worker);
      threads->push_back(launch_worker->native_handle());
    } else if (default_actor_thread_num_ > kAsyncLaunchThreadNum) {
      const auto &launch_worker = RuntimePipeline::GetInstance().launch_queue()->worker();
      MS_EXCEPTION_IF_NULL(launch_worker);
      threads->push_back(launch_worker->native_handle());
    }
  }
  if (threads->size() != default_actor_thread_num_) {
    MS_LOG(EXCEPTION) << "Get invalid total thread number, expected: " << default_actor_thread_num_
                      << ", but got: " << threads->size();
  }
}

void GraphScheduler::BindCoreForRuntimeThread(ActorThreadPool *thread_pool) const {
  static bool is_bind_core_ = false;
  if (is_bind_core_) {
    return;
  }
  auto &bind_core_manager = runtime::ThreadBindCore::GetInstance();
  if (!bind_core_manager.is_enable_thread_bind_core_) {
    return;
  }

  const auto &cpu_list = bind_core_manager.get_thread_bind_core_list(runtime::kBindCoreModule::kRUNTIME);
  if (cpu_list.empty()) {
    MS_LOG(WARNING) << "Skip to bind thread core for 'runtime actor' as no available core assigned.";
  } else {
    std::vector<pthread_t> threads;
    GetRuntimeThreadIds(thread_pool, &threads);
    const auto &actor_thread_fix_bind = runtime::GetRuntimeConfigValue(runtime::kRuntimeActorThreadFixBind);
    thread_pool->APIThreadPoolSetAffinity(threads, cpu_list, actor_thread_fix_bind);
  }
  is_bind_core_ = true;
}
#endif

void GraphScheduler::SetActorExecutionStrategy(ActorSet *const actor_set, GraphExecutionStrategy strategy,
                                               double execution_time) const {
  MS_EXCEPTION_IF_NULL(actor_set);
  ++actor_set->execution_count_;
  MS_LOG(INFO) << actor_set->name_ << " execution count: " << actor_set->execution_count_
               << ", execution time: " << execution_time
               << " ms in multi thread or not: " << actor_set->is_multi_thread_execution_ << ".";
  MS_VLOG(VL_RUNTIME_FRAMEWORK_PRINT_PROF)
    << actor_set->name_ << " execution count: " << actor_set->execution_count_ << ", execution time: " << execution_time
    << " ms in multi thread or not: " << actor_set->is_multi_thread_execution_ << ".";

  MS_EXCEPTION_IF_NULL(ActorMgr::GetActorMgrRef());
  auto thread_pool = ActorMgr::GetActorMgrRef()->GetActorThreadPool();
  MS_EXCEPTION_IF_NULL(thread_pool);
  if (actor_set->execution_count_ == kBindCoreThreadStep) {
#if defined(__linux__) && defined(BIND_CORE)
    static bool bind_core_flag = false;
    static bool env_runtime_reserved_empty = common::GetEnv("CONFIG_BIND_RUNTIME_LIST").empty();
    if (!env_runtime_reserved_empty && bind_core_flag == false) {
      std::vector<pthread_t> threads;
      GetRuntimeThreadIds(thread_pool, &threads);
      thread_pool->ThreadPoolSetAffinity(threads);
      bind_core_flag = true;
    }

    BindCoreForRuntimeThread(thread_pool);
#endif
  }

  if (!CheckSingleThreadRunningCondition(actor_set, strategy)) {
    return;
  }

  // When the constraint condition of single thread execution is met,
  // if the actor threads num are less than or equal to 1, it will be run in sync mode.
  if (thread_pool->GetActorThreadNum() <= 1) {
    actor_set->is_multi_thread_execution_ = false;
    return;
  }

  if ((actor_set->is_multi_thread_execution_) &&
      (actor_set->execution_count_ >= ActorDispatcher::kMultiThreadExecutionCountBegin) &&
      (actor_set->execution_count_ <= ActorDispatcher::kMultiThreadExecutionCountEnd)) {
    actor_set->multi_thread_execution_time_ += execution_time;
    if (actor_set->execution_count_ == ActorDispatcher::kMultiThreadExecutionCountEnd) {
      actor_set->multi_thread_execution_time_ /=
        ((ActorDispatcher::kMultiThreadExecutionCountEnd - ActorDispatcher::kMultiThreadExecutionCountBegin) + 1);
      actor_set->is_multi_thread_execution_ = false;
    }
    return;
  }

  if ((!actor_set->is_multi_thread_execution_) &&
      (actor_set->execution_count_ >= ActorDispatcher::kSingleThreadExecutionCountBegin) &&
      (actor_set->execution_count_ <= ActorDispatcher::kSingleThreadExecutionCountEnd)) {
    actor_set->single_thread_execution_time_ += execution_time;
    if (actor_set->execution_count_ == ActorDispatcher::kSingleThreadExecutionCountEnd) {
      actor_set->single_thread_execution_time_ /=
        (ActorDispatcher::kSingleThreadExecutionCountEnd - ActorDispatcher::kSingleThreadExecutionCountBegin + 1);
      actor_set->is_multi_thread_execution_ =
        (actor_set->multi_thread_execution_time_ <= actor_set->single_thread_execution_time_) ? true : false;
      MS_LOG(INFO) << "Multi thread execution time cost: " << actor_set->multi_thread_execution_time_
                   << " ms, single thread execution time cost: " << actor_set->single_thread_execution_time_
                   << " ms, decide to use multi thread execution or not: " << actor_set->is_multi_thread_execution_
                   << ".";
    }
    return;
  }
}

ActorSet *GraphScheduler::Fetch(const ActorInfo &actor_info) const {
  auto iter = actors_.find(actor_info);
  if (iter != actors_.end()) {
    return iter->second.get();
  } else {
    MS_LOG(DEBUG) << "Can't find the actors map of " << actor_info;
    return nullptr;
  }
}

ActorSet *GraphScheduler::Fetch(uint32_t actor_id) const {
  auto iter = std::find_if(actors_.begin(), actors_.end(),
                           [actor_id](const auto &i) { return (i.second->actor_id_ == actor_id); });
  if (iter != actors_.end()) {
    return iter->second.get();
  }
  MS_LOG(DEBUG) << "Can't find the actors map of " << actor_id;
  return nullptr;
}

ActorSetPtr GraphScheduler::Build(const GraphCompilerInfo &graph_compiler_info) {
  auto actor_set = std::make_shared<ActorSet>(graph_compiler_info.name_);
  MS_EXCEPTION_IF_NULL(actor_set);
  actor_set->graph_phase_ = graph_compiler_info.graph_phase_;
  actor_set->actor_id_ = graph_compiler_info.id_;
  (void)actors_.emplace(actor_set->name_, actor_set);

  TryEnableKbkSubGraphExecMode(graph_compiler_info, actor_set.get());
  TryEnableInputOptimize(graph_compiler_info, actor_set.get());
  TryEnableCaptureGraph(graph_compiler_info);

  // Create the graph_parameter_store of the current graph
  if (EnableInputOptimize()) {
    BuildGraphParameterStore(graph_compiler_info);
    actor_set->data_prepare_actor_ = BuildDataPrepareActorForGraphParameterStore(graph_compiler_info);
  } else {
    auto host_queue = std::make_shared<HostTensorQueue>();
    actor_set->data_source_actors_ = BuildDataSourceActor(graph_compiler_info, host_queue);
    actor_set->data_prepare_actor_ =
      BuildDataPrepareActor(graph_compiler_info, actor_set->data_source_actors_, host_queue);
  }

  actor_set->kernel_actors_ = BuildKernelActor(graph_compiler_info);
  actor_set->super_kernel_actors_ = BuildSuperKernelActor(graph_compiler_info);
  actor_set->any_type_kernel_actors_ =
    any_type_graph_scheduler_.Build(graph_compiler_info, memory_manager_aid_, debug_aid_);
  actor_set->loop_count_actor_ = BuildLoopCountActor(graph_compiler_info);
  actor_set->output_actor_ = BuildOutputActor(graph_compiler_info);
  actor_set->control_actors_ = control_node_scheduler_.Build(graph_compiler_info, memory_manager_aid_);

  return actor_set;
}

void GraphScheduler::CacheGraphOutputToActor(const GraphCompilerInfo &graph_compiler_info) {
  if (graph_compiler_info.strategy_ == GraphExecutionStrategy::kStep) {
    return;
  }

  for (const auto &graph : graph_compiler_info.graphs_) {
    MS_EXCEPTION_IF_NULL(graph);
    auto graph_id = graph->graph_id();
    // As the cse optimization of kernel graph, front cnodes with the same inputs will be optimized to the same
    // backend cnode, it means that multiple front nodes will correspond to the same backend node. In order to
    // ensure that all front nodes are obtained, only the front node to graph output map can be used, not the
    // output to front node.
    for (const auto &front_backend_pair : graph->front_node_to_graph_output_map()) {
      const auto &output_with_index = front_backend_pair.second;
      auto output_kernel = output_with_index.first;
      auto output_index = output_with_index.second;
      MS_EXCEPTION_IF_NULL(output_kernel);
      auto origin_output_with_index = front_backend_pair.first;
      if (origin_output_with_index.first == nullptr) {
        MS_LOG(WARNING) << "The graph " << graph_id << " output node:" << output_kernel->fullname_with_scope()
                        << " with index: " << output_index << " has no front node.";
        continue;
      }

      auto kernel_type = FetchKernelTransformType(output_kernel, graph, graph_compiler_info.origin_parameters_order_);
      auto output_actor = FetchActor(kernel_type, graph_compiler_info.name_, output_kernel, graph);
      if (EnableInputOptimize()) {
        auto graph_parameter_store = ParameterStore::GetInstance().GetGraphParameterStore();
        MS_EXCEPTION_IF_NULL(graph_parameter_store);
        const auto &front_node_with_idx = FetchRealFrontNode(output_with_index, graph);
        if (kernel_type == KernelTransformType::kGraphParameterStore ||
            (front_node_with_idx.first != nullptr && front_node_with_idx.first->isa<Parameter>())) {
          auto data_prepare_actor_name =
            ParameterStore::GetInstance().GetChosenGraphName() + kDataPrepareActorNameSuffix;
          output_actor = FetchActor(data_prepare_actor_name);
          (void)graph_parameter_store->CorrectFrontNodeMap(origin_output_with_index, front_node_with_idx);
        }
      }
      // Internal parameter need update the output actor and output kernel through the front node of last graph.
      if (kernel_type == KernelTransformType::kInternalParameter) {
        auto front_output_with_index = graph->GetOriginFrontNodeByInternalParameter(
          common::AnfAlgo::FetchRealNodeSkipMonadControl(output_with_index).first);
        if (graph_output_to_actor_.count(front_output_with_index) > 0) {
          output_actor = graph_output_to_actor_[front_output_with_index].first;
          output_kernel = graph_output_to_actor_[front_output_with_index].second.first;
          output_index = graph_output_to_actor_[front_output_with_index].second.second;
          MS_LOG(INFO) << "Graph " << graph_id << " output node:" << output_with_index.first->fullname_with_scope()
                       << " with index:" << output_with_index.second
                       << " is internal parameter, and fetch last graph real output node:"
                       << output_kernel->fullname_with_scope() << " with index:" << output_index
                       << ", from front node:" << front_output_with_index.first->fullname_with_scope()
                       << " with index:" << front_output_with_index.second;
        }
      }
      // Only the device tensor store not need cache output actor.
      if ((output_actor == nullptr) && (kernel_type != KernelTransformType::kDeviceTensorStore)) {
        MS_LOG(INFO) << "Graph " << graph_id << " output node:" << output_with_index.first->fullname_with_scope()
                     << " with index:" << output_with_index.second
                     << ", from front node:" << origin_output_with_index.first->fullname_with_scope()
                     << " with index:" << origin_output_with_index.second
                     << " is not actor, and the kernel type is:" << kernel_type;
      }

      auto output_actor_name = (output_actor != nullptr) ? output_actor->GetAID().Name() : "";
      (void)graph_output_to_actor_.emplace(origin_output_with_index,
                                           GraphOutputPair(output_actor, {output_kernel, output_index}));
      MS_LOG(INFO) << "Cache graph " << graph_id << " output node:" << output_with_index.first->fullname_with_scope()
                   << " debug string:" << output_with_index.first->DebugString()
                   << " node addr:" << output_with_index.first.get() << " with index:" << output_with_index.second
                   << " to actor:" << output_actor_name
                   << ", from front node:" << origin_output_with_index.first->fullname_with_scope()
                   << " debug string:" << origin_output_with_index.first->DebugString()
                   << " front node add:" << origin_output_with_index.first.get()
                   << " with index:" << origin_output_with_index.second;

      SchedulerHelper::AddSomasInfoForGraphOutput(output_actor, output_index, graph_id);
    }
  }
}

void AddControlArrowForNoInputActor(const AbstractActorPtr &from_actor, const AbstractActorPtr &to_actor) {
  MS_EXCEPTION_IF_NULL(from_actor);
  MS_EXCEPTION_IF_NULL(to_actor);
  if (to_actor->input_data_arrow_aids().size() == 0 && to_actor->input_control_arrow_aids().size() == 0) {
    MS_LOG(DEBUG) << "Add control arrow for no input arrow actor: " << to_actor->GetAID().Name();
    SchedulerHelper::AddControlArrow(from_actor.get(), to_actor.get());
  }
}

void GraphScheduler::LinkControlArrowForNoInputArrowActor(const ActorSet *actor_set) {
  if (!EnableInputOptimize()) {
    return;
  }

  MS_EXCEPTION_IF_NULL(actor_set);
  std::vector<AbstractActorPtr> actors;
  for (auto &kernel_actor : actor_set->kernel_actors_) {
    MS_EXCEPTION_IF_NULL(kernel_actor);
    (void)actors.emplace_back(static_cast<AbstractActorPtr>(kernel_actor));
  }
  for (auto &super_kernel_actor : actor_set->super_kernel_actors_) {
    MS_EXCEPTION_IF_NULL(super_kernel_actor);
    (void)actors.emplace_back(static_cast<AbstractActorPtr>(super_kernel_actor));
  }
  for (auto &memory_actor : actor_set->memory_actors_) {
    MS_EXCEPTION_IF_NULL(memory_actor);
    (void)actors.emplace_back(static_cast<AbstractActorPtr>(memory_actor));
  }
  for (auto &copy_actor : actor_set->copy_actors_) {
    MS_EXCEPTION_IF_NULL(copy_actor);
    (void)actors.emplace_back(static_cast<AbstractActorPtr>(copy_actor));
  }
  for (auto &fusion_actor : actor_set->fusion_actors_) {
    MS_EXCEPTION_IF_NULL(fusion_actor);
    (void)actors.emplace_back(static_cast<AbstractActorPtr>(fusion_actor));
  }
  if (actor_set->loop_count_actor_ != nullptr) {
    (void)actors.emplace_back(static_cast<AbstractActorPtr>(actor_set->loop_count_actor_));
  }
  if (actor_set->output_actor_ != nullptr) {
    (void)actors.emplace_back(static_cast<AbstractActorPtr>(actor_set->output_actor_));
  }

  for (auto &actor : actors) {
    AddControlArrowForNoInputActor(actor_set->data_prepare_actor_, actor);
  }
}

void GraphScheduler::Link(ActorSet *actor_set, const GraphCompilerInfo &graph_compiler_info) {
  MS_EXCEPTION_IF_NULL(actor_set);
  std::vector<AbstractActor *> auto_monad_actors;
  GroupNameToCommuNodes group_name_to_communication_nodes;
  std::string default_group_name = "";
  const auto &parser = graph_compiler_info.control_node_parser_;
  MS_EXCEPTION_IF_NULL(parser);

  // Maintain shared pointer for graphs manager which will be used in Link and LinkKernelActorsForSubGraphExecute phase.
  std::vector<FuncGraphManagerPtr> graph_managers;
  graph_managers.reserve(graph_compiler_info.graphs_.size());
  for (const auto &graph : graph_compiler_info.graphs_) {
    MS_EXCEPTION_IF_NULL(graph);
    if (graph->execution_order().empty()) {
      continue;
    }
    auto manager = graph->manager();
    if (manager == nullptr) {
      manager = Manage(graph, true);
      graph->set_manager(manager);
    }
    graph_managers.push_back(manager);
  }

  for (const auto &graph : graph_compiler_info.graphs_) {
    MS_EXCEPTION_IF_NULL(graph);
    if (graph->execution_order().empty()) {
      MS_LOG(INFO) << "The graph " << graph->graph_id() << " is an empty graph and skips linking.";
      continue;
    }

    if (graph->is_graph_run_mode() || graph->is_any_type_input()) {
      PROF_START(GraphSchedulerLinkSinkMode);
      LinkDataArrowInSinkMode(graph, graph_compiler_info, &auto_monad_actors);
      PROF_END(GraphSchedulerLinkSinkMode);
    } else {
      // In the control flow, the communication nodes need to be guaranteed to be executed in order. The order
      // within the kernel graph group needs to add control arrows between the communication nodes, and the order
      // between groups is guaranteed by the control flow framework. Therefore, communication nodes need to be
      // grouped by group name. And this is not required in non-control flow, the default unified group name is used.
      std::vector<CNodePtr> communication_nodes;
      const auto &group_name = (parser->IsInited() ? parser->FetchGroupNameByKernelGraph(graph) : default_group_name);
      PROF_START(GraphSchedulerLinkNoSinkMode);
      LinkDataArrowInNonSinkMode(graph, graph_compiler_info, &auto_monad_actors, &communication_nodes);
      PROF_END(GraphSchedulerLinkNoSinkMode);
      (void)group_name_to_communication_nodes[group_name].first.insert(
        group_name_to_communication_nodes[group_name].first.end(), communication_nodes.begin(),
        communication_nodes.end());
      (void)group_name_to_communication_nodes[group_name].second.emplace_back(graph);
    }
  }

  LinkGlobalControlArrow(actor_set, group_name_to_communication_nodes, auto_monad_actors, graph_compiler_info);
  LinkOutputResultArrowForOutputActor(actor_set->output_actor_.get(), graph_compiler_info);

  // The copy actors are built in the link, so need push into the actor set after link.
  actor_set->copy_actors_ = copy_actors_;
  // Link the arrow in the control flow scene.
  if (graph_compiler_info.strategy_ == GraphExecutionStrategy::kPipeline &&
      graph_compiler_info.control_node_parser_ != nullptr && graph_compiler_info.control_node_parser_->IsInited()) {
    control_node_scheduler_.Link(actor_set, graph_compiler_info);
  }

  // Need to call after all link task finish, because all kernel actor of super kernel actor will be initialized and
  // need to known the graph output(ref count: SIZE_MAX)
  LinkKernelActorsForSubGraphExecute(actor_set);
  LinkControlArrowForNoInputArrowActor(actor_set);
}

void GraphScheduler::ProcessContinuousMemoryInfo(const ActorSetPtr &actor_set,
                                                 const GraphCompilerInfo &graph_compiler_info) {
  // Cache the nodes which need continuous memory for actor set.
  if (graph_compiler_info.strategy_ == GraphExecutionStrategy::kPipeline) {
    for (size_t index = 0; index < graph_compiler_info.graphs_.size(); ++index) {
      const auto &graph = graph_compiler_info.graphs_[index];
      MS_EXCEPTION_IF_NULL(graph);
      auto ms_context = MsContext::GetInstance();
      MS_EXCEPTION_IF_NULL(ms_context);
      // Somas will take over the continuous memory.
      const bool using_somas =
        (graph->is_graph_run_mode() && !EnableKbkSubGraphExecute()) || (graph->somas_whole_block_size() != 0);
      if (using_somas) {
        continue;
      }

      auto &execution_order = graph->execution_order();
      for (auto &kernel : execution_order) {
        if (!AnfAlgo::IsNeedContinuesMemoryOp(kernel)) {
          continue;
        }
        auto key =
          std::make_pair(kernel, device::FetchRealDeviceContext(kernel, graph_compiler_info.device_contexts_[index]));
        auto value = std::make_pair(false, false);
        if (common::AnfAlgo::GetInputTensorNum(kernel) > 1) {
          value.first = true;
        }
        if (AnfAlgo::GetOutputTensorNum(kernel) > 1) {
          value.second = true;
        }
        if (value.first || value.second) {
          actor_set->continuous_memory_nodes_[key] = value;
        }
      }
    }
  }

  // Process continuous memory info.
  for (auto &iter : actor_set->continuous_memory_nodes_) {
    // Inputs need continuous memory.
    if (iter.second.first) {
      const auto &cnode = iter.first.first;
      MS_LOG(INFO) << "Init continuous_memory_nodes_ cnode : " << cnode->fullname_with_scope() << ".";
      FetchContinuousMemoryInfo(cnode, true);
    }
    // Outputs need continuous memory.
    if (iter.second.second) {
      const auto &cnode = iter.first.first;
      MS_LOG(INFO) << "Init  continuous_memory_nodes_ cnode : " << cnode->fullname_with_scope() << ".";
      FetchContinuousMemoryInfo(cnode, false);
    }
  }

  // Update continuous memory flag for data prepare actor.
  actor_set->data_prepare_actor_->set_has_continuous_memory(actor_set->continuous_memory_nodes_.size() > 0);
}

void GraphScheduler::Optimize(const ActorSetPtr &actor_set, const GraphCompilerInfo &graph_compiler_info) const {
  MS_EXCEPTION_IF_NULL(actor_set);

  auto optimizer = std::make_shared<ActorSetOptimizer>();
  MS_EXCEPTION_IF_NULL(optimizer);

  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (runtime::RuntimeConf::GetInstance()->mem_optimize_level() != kOptimizeO0) {
    optimizer->AddPass(std::make_shared<MemoryActorInsert>());
  }
  optimizer->AddPass(std::make_shared<InvalidDataArrowElimination>());
  optimizer->AddPass(std::make_shared<MultiActorFusion>());
  optimizer->AddPass(std::make_shared<BatchDataArrowFusion>());
  optimizer->Optimize(actor_set);
  control_node_scheduler_.Optimize(actor_set, graph_compiler_info);
  any_type_graph_scheduler_.Optimize(actor_set, graph_output_to_actor_);
}

DeviceTensorPosition GetDeviceTensorPosition(const KernelWithIndex &front_node_with_index,
                                             const KernelWithIndex &origin_node_with_index,
                                             const KernelGraphPtr &graph) {
  auto graph_parameter_store = ParameterStore::GetInstance().GetGraphParameterStore();
  MS_EXCEPTION_IF_NULL(graph_parameter_store);
  return std::make_pair(graph_parameter_store->GetFrontNodeToIndex(front_node_with_index.first.get()),
                        front_node_with_index.second);
}

bool IsRootGraphParameter(const AnfNodePtr &input_node, const KernelGraphPtr &graph,
                          const std::vector<AnfNodePtr> &root_graph_parameters) {
  if (input_node == nullptr || graph == nullptr) {
    return false;
  }
  const auto &internal_parameter = graph->GetFrontNodeByInternalParameter(input_node);
  if (internal_parameter.first == nullptr) {
    return false;
  }
  return std::find(root_graph_parameters.begin(), root_graph_parameters.end(), internal_parameter.first) !=
         root_graph_parameters.end();
}

void SetSummaryInputUserCount(const std::map<std::string, std::pair<AnfNodePtr, int>> &summary_nodes,
                              const AnfNodePtr input_node, size_t real_outer_idx, size_t real_inner_idx) {
  MS_EXCEPTION_IF_NULL(input_node);
  auto cur_graph_parameter_store = ParameterStore::GetInstance().GetGraphParameterStore();
  auto iter =
    std::find_if(summary_nodes.begin(), summary_nodes.end(), [&input_node, &real_inner_idx](const auto &pair) {
      return pair.second.first == input_node && pair.second.second == SizeToInt(real_inner_idx);
    });
  if (iter != summary_nodes.end()) {
    (void)cur_graph_parameter_store->SetUserCnt(real_outer_idx, real_inner_idx, SIZE_MAX);
  }
}

void GraphScheduler::BuildGraphParameterStore(const GraphCompilerInfo &graph_compiler_info) {
  ParameterStore &parameterStore = ParameterStore::GetInstance();
  auto cur_graph_parameter_store = parameterStore.GetGraphParameterStore();

  for (size_t i = 0; i < graph_compiler_info.graphs_.size(); ++i) {
    const auto &graph = graph_compiler_info.graphs_[i];
    MS_EXCEPTION_IF_NULL(graph);
    const std::vector<AnfNodePtr> &input_nodes = graph->input_nodes();
    const auto &root_parameters = graph_compiler_info.origin_parameters_order_;
    bool exist_summary = graph->summary_node_exist();
    auto summary_nodes = graph->summary_nodes();

    for (size_t j = 0; j < input_nodes.size(); j++) {
      const auto &input_node = input_nodes[j];
      MS_EXCEPTION_IF_NULL(input_node);
      if (!IsHostQueueDSActor(input_node, graph, root_parameters, graph_compiler_info.strategy_) &&
          !IsRootGraphParameter(input_node, graph, root_parameters)) {
        continue;
      }
      MS_EXCEPTION_IF_NULL(graph_compiler_info.control_node_parser_);
      KernelWithIndex front_node_with_index = graph->GetElementInTupleBackendFrontIndexMap(input_node);
      if (graph_compiler_info.control_node_parser_->IsInited()) {
        if ((front_node_with_index.first != nullptr) && front_node_with_index.first->isa<Parameter>() &&
            (find(root_parameters.begin(), root_parameters.end(), front_node_with_index.first) ==
             root_parameters.end())) {
          continue;
        }
      }
      if (front_node_with_index.first == nullptr) {
        front_node_with_index = {AnfAlgo::FetchFrontNodeByBackendNode(input_node, *graph), 0};
        MS_LOG(DEBUG) << "Init backend input node:" << input_node->DebugString() << " for host data source actor.";
      }
      MS_EXCEPTION_IF_NULL(front_node_with_index.first);
      graph_compiler_info.origin_parameters_to_backend_parameters_[front_node_with_index.first].emplace_back(
        std::make_pair(front_node_with_index, KernelWithIndex(input_node, 0)));

      auto cur_kernel_tensor = AnfAlgo::GetOutputKernelTensor(input_node, 0, false);
      MS_EXCEPTION_IF_NULL(cur_kernel_tensor);
      size_t real_outer_idx = cur_graph_parameter_store->GetFrontNodeToIndex(front_node_with_index.first.get());
      size_t real_inner_idx = front_node_with_index.second;
      // Record dynamic shape info.
      auto input_param = input_node->cast<ParameterPtr>();
      if (input_param == nullptr) {
        continue;
      }
      if (input_param->has_dynamic_shape()) {
        cur_graph_parameter_store->SetIsPositionDynamic(real_outer_idx, real_inner_idx, true);
      }

      auto store_kernel_tensor = cur_graph_parameter_store->Fetch(real_outer_idx, real_inner_idx);
      // Same device type as device address in store, do not store.
      // if device type is different, there is heterogeneous, store non cpu device address.
      if (store_kernel_tensor != nullptr && store_kernel_tensor->device_address() != nullptr &&
          cur_kernel_tensor->device_address() != nullptr) {
        auto store_device_tensor = store_kernel_tensor->device_address();
        auto cur_device_tensor = cur_kernel_tensor->device_address();
        if (store_device_tensor->GetDeviceType() == cur_device_tensor->GetDeviceType() ||
            store_device_tensor->GetDeviceType() != device::DeviceType::kCPU) {
          continue;
        }
      }

      // The device tensor is not hold by runtime.
      if (cur_kernel_tensor->device_ptr() != nullptr) {
        auto cur_device_tensor = cur_kernel_tensor->device_address();
        MS_EXCEPTION_IF_NULL(cur_device_tensor);
        auto cloned_device_tensor = cur_device_tensor->CloneDeviceAddress();
        cloned_device_tensor->set_ptr(nullptr);
        cur_kernel_tensor->set_device_address(cloned_device_tensor);
        cur_kernel_tensor->set_tensor_storage_info(cur_kernel_tensor->tensor_storage_info());
      }

      (void)cur_graph_parameter_store->Push(real_outer_idx, real_inner_idx, cur_kernel_tensor, 0);
      MS_LOG(DEBUG) << "Build graph parameter :" << input_node->DebugString()
                    << " for front node:" << front_node_with_index.first->DebugString()
                    << " index:" << front_node_with_index.second << " position:" << real_outer_idx
                    << " kernel tensor:" << cur_kernel_tensor->ToString();
      if (exist_summary) {
        SetSummaryInputUserCount(summary_nodes, input_node, real_outer_idx, real_inner_idx);
      }
    }
  }
  control_node_scheduler_.BuildGraphParameterStoreForControlNode(graph_compiler_info, memory_manager_aid_);
}

std::map<KernelWithIndex, std::vector<KernelTensorPtr>> CollectRefKernelTensor(const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  std::map<KernelWithIndex, std::vector<KernelTensorPtr>> ref_kernel_tensors;
  for (const auto &pair : graph->GetRefMap()) {
    MS_EXCEPTION_IF_NULL(pair.first.first);
    const auto &origin_node_with_index = graph->GetRefNodeRecursive(pair.first);
    if (origin_node_with_index.first->isa<Parameter>() &&
        AnfAlgo::OutputAddrExist(pair.first.first, pair.first.second) &&
        AnfAlgo::OutputAddrExist(origin_node_with_index.first, origin_node_with_index.second)) {
      ref_kernel_tensors[origin_node_with_index].emplace_back(
        AnfAlgo::GetOutputKernelTensor(pair.first.first, pair.first.second, false));
    }
  }
  return ref_kernel_tensors;
}

std::vector<DataSourceActorPtr> GraphScheduler::BuildDataSourceActor(const GraphCompilerInfo &graph_compiler_info,
                                                                     const HostTensorQueuePtr &host_queue) {
  std::vector<DataSourceActorPtr> data_source_actors;
  HostQueueDSActorPtr host_queue_ds_actor = nullptr;
  size_t data_node_position = 0;
  std::map<KernelWithIndex, size_t> front_node_position_temp_map;

  for (size_t i = 0; i < graph_compiler_info.graphs_.size(); ++i) {
    const auto &graph = graph_compiler_info.graphs_[i];
    const auto &graph_device_context = graph_compiler_info.device_contexts_[i];
    MS_EXCEPTION_IF_NULL(graph);
    // Build host queue data source actor.
    const std::vector<AnfNodePtr> &input_nodes = graph->input_nodes();
    const auto &root_parameters = graph_compiler_info.origin_parameters_order_;
    auto ref_kernel_tensors = CollectRefKernelTensor(graph);
    for (size_t j = 0; j < input_nodes.size(); j++) {
      const auto &input_node = input_nodes[j];
      MS_EXCEPTION_IF_NULL(input_node);
      const auto &device_context = device::FetchRealDeviceContext(input_node, graph_device_context);
      MS_EXCEPTION_IF_NULL(device_context);

      if (IsHostQueueDSActor(input_node, graph, root_parameters, graph_compiler_info.strategy_)) {
        // In control flow, parameters from subgraph need not init in data source actor.
        MS_EXCEPTION_IF_NULL(graph_compiler_info.control_node_parser_);
        if (graph_compiler_info.control_node_parser_->IsInited()) {
          auto node_with_index = graph->GetElementInTupleBackendFrontIndexMap(input_node);
          if ((node_with_index.first != nullptr) && node_with_index.first->isa<Parameter>() &&
              (find(root_parameters.begin(), root_parameters.end(), node_with_index.first) == root_parameters.end())) {
            continue;
          }
        }

        if (host_queue_ds_actor == nullptr) {
          auto actor_name = graph_compiler_info.name_ + kHostDSActorNameSuffix;
          MS_LOG(INFO) << "Create host queue data source actor: " << actor_name;
          host_queue_ds_actor = std::make_shared<HostQueueDataSourceActor>(
            actor_name, 1, memory_manager_aid_, nullptr, nullptr, host_queue, graph_compiler_info.graph_phase_);
          InsertActor(host_queue_ds_actor.get());
          (void)data_source_actors.emplace_back(host_queue_ds_actor);
        }

        KernelWithIndex front_node_with_index = graph->GetElementInTupleBackendFrontIndexMap(input_node);
        if (front_node_with_index.first == nullptr) {
          front_node_with_index = {AnfAlgo::FetchFrontNodeByBackendNode(input_node, *graph), 0};
          MS_LOG(DEBUG) << "Init backend input node:" << input_node->DebugString() << " for host data source actor.";
        }
        MS_EXCEPTION_IF_NULL(front_node_with_index.first);

        // After graph partition and graph compile, multiple kernel graphs will share the same parameter. If the
        // parameter is already in the data node map, there is no need to process it again.
        if (host_queue_ds_actor->data_node_position_map_.find(std::make_pair(input_node, 0)) !=
            host_queue_ds_actor->data_node_position_map_.end()) {
          continue;
        }
        graph_compiler_info.origin_parameters_to_backend_parameters_[front_node_with_index.first].emplace_back(
          std::make_pair(front_node_with_index, KernelWithIndex(input_node, 0)));
        // In the scenario where multiple backend nodes correspond to the same front node, only the first backend node
        // is saved in the host queue data source actor. Particularly, the same front parameter corresponds to multiple
        // backend parameters in heterogeneous scenarios, and these heterogeneous parameters need to be placed in the
        // data source actor.
        if (front_node_position_temp_map.count(front_node_with_index) > 0) {
          auto front_node_index = front_node_position_temp_map[front_node_with_index];
          if (!IsDeviceTypeNotSame(device_context, host_queue_ds_actor->device_contexts_[front_node_index])) {
            (void)host_queue_ds_actor->data_node_position_map_.emplace(KernelWithIndex(input_node, 0),
                                                                       front_node_index);
            continue;
          } else {
            host_queue_ds_actor->heter_index_pair_.emplace_back(front_node_index, data_node_position);
            MS_LOG(DEBUG) << "Add heter ref node:" << input_node->DebugString() << " index:" << data_node_position
                          << " to node:"
                          << host_queue_ds_actor->data_node_with_indexs_[front_node_index].first->DebugString()
                          << " index:" << front_node_index
                          << " front node:" << front_node_with_index.first->DebugString() << " to data source actor.";
          }
        }
        (void)host_queue_ds_actor->data_node_with_indexs_.emplace_back(input_node, 0);
        auto iter = ref_kernel_tensors.find({input_node, 0});
        if (iter != ref_kernel_tensors.end()) {
          host_queue_ds_actor->ref_kernel_tensors_[{input_node, 0}] = iter->second;
          for (const auto &kernel_tensor : iter->second) {
            MS_LOG(DEBUG) << "Add ref device tensor:" << kernel_tensor
                          << " for input node:" << input_node->DebugString();
          }
        }
        (void)host_queue_ds_actor->device_contexts_.emplace_back(device_context);
        (void)host_queue_ds_actor->data_node_position_map_.emplace(KernelWithIndex(input_node, 0), data_node_position);
        // In control flow, need to rely on the front node to find the location of the corresponding real parameter.
        (void)host_queue_ds_actor->data_node_position_map_.emplace(front_node_with_index, data_node_position);
        MS_LOG(DEBUG) << "Insert data source parameter:" << input_node->DebugString()
                      << " for front node:" << front_node_with_index.first->DebugString()
                      << " index:" << front_node_with_index.second << " position:" << data_node_position;
        (void)front_node_position_temp_map.emplace(front_node_with_index, data_node_position);
        data_node_position++;
      }
    }
  }
  control_node_scheduler_.BuildDataSourceActorForControlNode(graph_compiler_info, host_queue, host_queue_ds_actor,
                                                             memory_manager_aid_, &data_source_actors);
  return data_source_actors;
}

std::vector<KernelActorPtr> GraphScheduler::BuildKernelActor(const GraphCompilerInfo &graph_compiler_info) {
  std::vector<KernelActorPtr> kernel_actors;
  auto root_weights = GatherAllParams(graph_compiler_info);

  for (size_t i = 0; i < graph_compiler_info.graphs_.size(); ++i) {
    const auto &graph = graph_compiler_info.graphs_[i];
    const auto &device_context = graph_compiler_info.device_contexts_[i];
    MS_EXCEPTION_IF_NULL(graph);
    graph->CacheRootWeight(root_weights);
    if (graph->is_graph_run_mode() || graph->is_any_type_input() || EnableKbkSubGraphExecute()) {
      continue;
    }

    auto execution_order = graph->execution_order();
    // Single op graph in step mode, kernel actor executes synchronously.
    bool is_single_op_graph = execution_order.size() == 1;
    GraphExecutionStrategy strategy = graph_compiler_info.strategy_;
    if (strategy == GraphExecutionStrategy::kStep) {
      strategy = (is_single_op_graph ? strategy : GraphExecutionStrategy::kPipeline);
    }

    // Stream recv node need task id on stream from send node. Here pass stream send actor to stream recv actor.
    mindspore::HashMap<uint32_t, std::pair<KernelActorPtr, KernelActorPtr>> send_recv_nodes;
    for (auto &kernel : execution_order) {
      MS_EXCEPTION_IF_NULL(kernel);
      if (IsKernelActor(kernel, graph_compiler_info.strategy_) && (!IsSkippedKernelActor(kernel))) {
        auto ref_input_indexes = FetchModifiableRefInputIndex(kernel);
        auto ref_output_indexes = FetchModifiableRefOutputIndex(kernel, graph);
        const auto &real_device_context = device::FetchRealDeviceContext(kernel, device_context);
        MS_EXCEPTION_IF_NULL(real_device_context);
        KernelActorPtr kernel_actor = nullptr;
        if (IsInnerControlFlowActor(kernel)) {
          MS_LOG(EXCEPTION) << "Can not build a sub graph which contains ConditionSwitch or ConditionSwitch by kbk.";
        } else {
          kernel_actor = std::make_shared<KernelActor>(GenerateActorIdByKernel(kernel), kernel, real_device_context,
                                                       memory_manager_aid_, debug_aid_, recorder_aid_, strategy,
                                                       ref_input_indexes, ref_output_indexes);
        }
        MS_EXCEPTION_IF_NULL(kernel_actor);
        // Set the member of kernel actor.
        kernel_actor->is_launch_skipped_ =
          common::AnfAlgo::IsNopNode(kernel) && graph->IsInRefOutputMap(std::make_pair(kernel, 0));
        kernel_actor->inputs_continuous_memory_ =
          AnfAlgo::IsNeedContinuesMemoryOp(kernel) && (common::AnfAlgo::GetInputTensorNum(kernel) > 1);

        if (IsPrimitiveCNode(kernel, prim::kPrimStreamSend)) {
          SchedulerHelper::ProcessStreamSendRecvEventPair(&send_recv_nodes, kernel, kernel_actor, true);
        } else if (IsPrimitiveCNode(kernel, prim::kPrimStreamRecv)) {
          SchedulerHelper::ProcessStreamSendRecvEventPair(&send_recv_nodes, kernel, kernel_actor, false);
        }

        InsertActor(kernel_actor.get());
        (void)kernel_actors.emplace_back(kernel_actor);
      }
    }
    for (auto &[event_pair_id, send_recv_actor] : send_recv_nodes) {
      auto [send_actor, recv_actor] = send_recv_actor;
      MS_LOG(DEBUG) << "Stream send/recv pair : " << event_pair_id << ", send_actor : " << send_actor
                    << ", recv_actor : " << recv_actor << ".";
      recv_actor->set_stream_send_actor(send_actor.get());
    }
  }
  return kernel_actors;
}

std::vector<AnfNodePtr> GraphScheduler::GatherAllParams(const GraphCompilerInfo &graph_compiler_info) {
  std::vector<AnfNodePtr> root_weights;
  for (size_t i = 0; i < graph_compiler_info.graphs_.size(); ++i) {
    const auto &graph = graph_compiler_info.graphs_[i];
    MS_EXCEPTION_IF_NULL(graph);
    const auto &params = graph->parameters();
    (void)root_weights.insert(root_weights.end(), params.begin(), params.end());
  }
  return root_weights;
}

std::vector<SuperKernelActorPtr> GraphScheduler::BuildSuperKernelActor(const GraphCompilerInfo &graph_compiler_info) {
  std::vector<SuperKernelActorPtr> super_kernel_actors;
  auto root_weights = GatherAllParams(graph_compiler_info);

  for (size_t i = 0; i < graph_compiler_info.graphs_.size(); ++i) {
    const auto &graph = graph_compiler_info.graphs_[i];
    const auto &device_context = graph_compiler_info.device_contexts_[i];
    MS_EXCEPTION_IF_NULL(graph);
    if ((!graph->is_graph_run_mode() && !EnableKbkSubGraphExecute()) || graph->is_any_type_input()) {
      continue;
    }

    if (graph->execution_order().empty()) {
      MS_LOG(INFO) << "The graph " << graph->graph_id() << " is an empty graph and skips building.";
      continue;
    }
    graph->CacheRootWeight(root_weights);
    auto actor_name = graph->ToString() + kSuperKernelActorNameSuffix;
    auto super_kernel_actor = std::make_shared<SuperKernelActor>(
      actor_name, graph, graph_compiler_info.graph_phase_, device_context, memory_manager_aid_, debug_aid_, nullptr);
    MS_EXCEPTION_IF_NULL(super_kernel_actor);
    InsertActor(super_kernel_actor.get());
    (void)super_kernel_actors.emplace_back(super_kernel_actor);
  }
  return super_kernel_actors;
}

LoopCountActorPtr GraphScheduler::BuildLoopCountActor(const GraphCompilerInfo &graph_compiler_info) {
  auto actor_set = Fetch(graph_compiler_info.name_);
  if ((graph_compiler_info.strategy_ == GraphExecutionStrategy::kStep) && IsSingleOpActorSet(actor_set)) {
    return nullptr;
  }

  auto loop_count = GetLoopCount(graph_compiler_info);
  auto sink_size = ConfigManager::GetInstance().iter_num();
  auto actor_name = graph_compiler_info.name_ + kLoopCountActorNameSuffix;
  auto is_need_sync_stream = GetNeedSyncStream(graph_compiler_info);
  auto loop_count_actor = std::make_shared<LoopCountActor>(
    actor_name, graph_compiler_info.name_, loop_count, sink_size, memory_manager_aid_, debug_aid_, recorder_aid_,
    profiler_aid_, graph_compiler_info.strategy_, graph_compiler_info.device_contexts_, is_need_sync_stream);
  MS_LOG(INFO) << "Create loop count actor: " << actor_name;
  MS_EXCEPTION_IF_NULL(loop_count_actor);

  InsertActor(loop_count_actor.get());
  return loop_count_actor;
}

OutputActorPtr GraphScheduler::BuildOutputActor(const GraphCompilerInfo &graph_compiler_info) const {
  auto actor_set = Fetch(graph_compiler_info.name_);
  if ((graph_compiler_info.strategy_ == GraphExecutionStrategy::kStep) && IsSingleOpActorSet(actor_set)) {
    return nullptr;
  }

  auto loop_count = GetLoopCount(graph_compiler_info);
  auto actor_name = graph_compiler_info.name_ + kOutputActorNameSuffix;
  // get summary node form graph_compiler_info and build a output actor
  std::vector<KernelWithIndex> summary_nodes;
  auto graphs = graph_compiler_info.graphs_;
  for (const auto &graph : graphs) {
    MS_EXCEPTION_IF_NULL(graph);
    if (!graph->summary_node_exist()) {
      continue;
    }
    const std::map<std::string, std::pair<AnfNodePtr, int>> &nodes = graph->summary_nodes();
    if (nodes.empty()) {
      continue;
    }
    (void)std::transform(nodes.cbegin(), nodes.cend(), std::back_inserter(summary_nodes), [](const auto &out) {
      return std::make_pair(out.second.first, IntToSize(out.second.second));
    });
  }
  auto output_actor =
    std::make_shared<OutputActor>(actor_name, loop_count, graph_compiler_info.outputs_num_, summary_nodes);
  MS_LOG(INFO) << "Create output actor: " << actor_name;
  MS_EXCEPTION_IF_NULL(output_actor);
  InsertActor(output_actor.get());
  return output_actor;
}

DataPrepareActorPtr GraphScheduler::BuildDataPrepareActorForGraphParameterStore(
  const GraphCompilerInfo &graph_compiler_info) {
  auto actor_name = graph_compiler_info.name_ + kDataPrepareActorNameSuffix;
  auto data_prepare_actor = std::make_shared<DataPrepareActor>(actor_name, memory_manager_aid_, debug_aid_,
                                                               profiler_aid_, &graph_compiler_info, nullptr, nullptr);
  MS_LOG(INFO) << "Create data prepare actor: " << actor_name;
  MS_EXCEPTION_IF_NULL(data_prepare_actor);
  InsertActor(data_prepare_actor.get());
  return data_prepare_actor;
}

DataPrepareActorPtr GraphScheduler::BuildDataPrepareActor(const GraphCompilerInfo &graph_compiler_info,
                                                          const std::vector<DataSourceActorPtr> &data_source_actors,
                                                          const HostTensorQueuePtr &host_queue) {
  HostQueueDSActorPtr host_queue_ds_actor = nullptr;
  auto iter = std::find_if(data_source_actors.begin(), data_source_actors.end(), [&](const auto &data_source_actor) {
    MS_EXCEPTION_IF_NULL(data_source_actor);
    return data_source_actor->type_ == KernelTransformType::kHostDataSourceActor;
  });
  if (iter != data_source_actors.end()) {
    host_queue_ds_actor = std::dynamic_pointer_cast<HostQueueDataSourceActor>(*iter);
  }
  auto actor_name = graph_compiler_info.name_ + kDataPrepareActorNameSuffix;
  auto data_prepare_actor = std::make_shared<DataPrepareActor>(
    actor_name, memory_manager_aid_, debug_aid_, profiler_aid_, &graph_compiler_info, host_queue_ds_actor, host_queue);
  MS_EXCEPTION_IF_NULL(data_prepare_actor);
  if (host_queue_ds_actor != nullptr) {
    data_prepare_actor->ref_kernel_tensors_ = host_queue_ds_actor->ref_kernel_tensors_;
  }
  MS_LOG(INFO) << "Create data prepare actor: " << actor_name;
  InsertActor(data_prepare_actor.get());
  return data_prepare_actor;
}

std::vector<AbstractActorPtr> GraphScheduler::BuildNoInputKernelActor(const ActorSet *actor_set,
                                                                      GraphExecutionStrategy strategy) const {
  MS_EXCEPTION_IF_NULL(actor_set);
  std::vector<AbstractActorPtr> no_input_kernel_actors;

  for (auto &super_kernel_actor : actor_set->super_kernel_actors_) {
    MS_EXCEPTION_IF_NULL(super_kernel_actor);

    if ((super_kernel_actor->input_datas_num_ == 0) && (super_kernel_actor->input_controls_num_ == 0)) {
      (void)no_input_kernel_actors.emplace_back(super_kernel_actor);
    }
  }

  for (auto &kernel_actor : actor_set->kernel_actors_) {
    MS_EXCEPTION_IF_NULL(kernel_actor);
    // Framework will trigger kernel actor running in the step execution strategy.
    if (strategy == GraphExecutionStrategy::kStep && IsSingleOpActorSet(actor_set)) {
      kernel_actor->input_controls_num_++;
      continue;
    }

    if ((kernel_actor->input_datas_num_ == 0) && (kernel_actor->input_controls_num_ == 0)) {
      (void)no_input_kernel_actors.emplace_back(kernel_actor);
    }
  }
  return no_input_kernel_actors;
}

namespace {
void GetAllUInputByCNode(const CNodePtr &cnode,
                         mindspore::HashMap<AnfNodePtr, std::set<AnfNodePtr>> *cnode_to_monad_inputs) {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(cnode_to_monad_inputs);
  if (cnode_to_monad_inputs->find(cnode) != cnode_to_monad_inputs->end()) {
    return;
  }
  (*cnode_to_monad_inputs)[cnode] = {};
  for (auto &weak_input : cnode->weak_inputs()) {
    auto input = weak_input.lock();
    MS_EXCEPTION_IF_NULL(input);
    if (!input->isa<CNode>()) {
      continue;
    }
    const auto &cinput = input->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cinput);
    if (common::AnfAlgo::GetCNodeName(cinput) == kUpdateStateOpName) {
      (void)(*cnode_to_monad_inputs)[cnode].emplace(cinput);
    }
    GetAllUInputByCNode(cinput, cnode_to_monad_inputs);
    (*cnode_to_monad_inputs)[cnode].insert((*cnode_to_monad_inputs)[cinput].begin(),
                                           (*cnode_to_monad_inputs)[cinput].end());
  }
}

void GetAllCNodeUInputByGraph(const KernelGraphPtr &graph,
                              mindspore::HashMap<AnfNodePtr, std::set<AnfNodePtr>> *cnode_to_monad_inputs) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(cnode_to_monad_inputs);
  for (const auto &kernel : graph->execution_order()) {
    MS_EXCEPTION_IF_NULL(kernel);
    GetAllUInputByCNode(kernel, cnode_to_monad_inputs);
  }
}

// Check if the first input of update state should be linked, if the other inputs of update state has depend the first
// input, it would not be linked.
bool IsNeedLinkForFirstInput(const CNodePtr &cnode,
                             const mindspore::HashMap<AnfNodePtr, std::set<AnfNodePtr>> &cnode_to_monad_inputs) {
  MS_EXCEPTION_IF_NULL(cnode);
  if (cnode->size() <= kUpdateStateStateInput) {
    MS_LOG_WITH_NODE(EXCEPTION, cnode) << "#dmsg#Runtime error info:#dmsg#Invalid update state node:"
                                       << cnode->DebugString();
  }
  const auto &u_input = cnode->input(kUpdateStateStateInput);
  MS_EXCEPTION_IF_NULL(u_input);
  for (size_t i = kUpdateStateRealInput; i < cnode->size(); ++i) {
    MS_EXCEPTION_IF_NULL(cnode->input(i));
    const auto &iter = cnode_to_monad_inputs.find(cnode->input(i));
    if (iter != cnode_to_monad_inputs.end() && iter->second.find(u_input) != iter->second.end()) {
      return false;
    }
  }
  return true;
}
}  // namespace

void GraphScheduler::LinkControlArrowForAnyTypeKernelActor(AbstractActor *to_actor, const AnfNodePtr &input_node,
                                                           const KernelGraphPtr &graph,
                                                           const ControlNodeParserPtr &parser) const {
  MS_EXCEPTION_IF_NULL(to_actor);
  MS_EXCEPTION_IF_NULL(input_node);
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(parser);
  if ((to_actor->type() != KernelTransformType::kAnyTypeKernelActor &&
       to_actor->type() != KernelTransformType::kSuperKernelActor) ||
      !IsInternalParameter(input_node, graph)) {
    return;
  }
  auto front_output_with_index = graph->GetOriginFrontNodeByInternalParameter(input_node);
  MS_EXCEPTION_IF_NULL(front_output_with_index.first);
  if (IsTakenOverByControlFlow(front_output_with_index.first, graph, parser)) {
    MS_LOG(DEBUG) << "Skip in control flow from node:" << front_output_with_index.first->DebugString()
                  << " is not in the graph:" << graph->ToString();
    return;
  }
  const auto &iter = graph_output_to_actor_.find(front_output_with_index);
  if (iter == graph_output_to_actor_.end()) {
    MS_LOG_WITH_NODE(INTERNAL_EXCEPTION, front_output_with_index.first)
      << "#dmsg#Runtime error info:#dmsg#Can't find graph output by front node:"
      << front_output_with_index.first->DebugString();
  }
  const auto &backend_from_kernel = iter->second.second.first;
  if (backend_from_kernel == nullptr) {
    MS_LOG(DEBUG) << "Cannot get backend node by:" << front_output_with_index.first->DebugString()
                  << " to actor:" << to_actor->GetAID();
    return;
  }
  const auto &from_graph = backend_from_kernel->func_graph();
  if (from_graph == nullptr) {
    MS_LOG(DEBUG) << "Cannot get funcgraph by:" << backend_from_kernel->DebugString()
                  << " to actor:" << to_actor->GetAID();
    return;
  }
  const auto &from_actor = FetchActor(from_graph->ToString() + kSuperKernelActorNameSuffix);
  if (from_actor != nullptr) {
    MS_LOG(DEBUG) << "Add control arrow from:" << from_actor->GetAID() << " to actor:" << to_actor->GetAID();
    SchedulerHelper::AddControlArrow(from_actor, to_actor);
  }
}

void GraphScheduler::LinkDataArrowInSinkMode(const KernelGraphPtr &graph, const GraphCompilerInfo &graph_compiler_info,
                                             std::vector<AbstractActor *> *const auto_monad_actors) {
  MS_EXCEPTION_IF_NULL(graph);
  auto to_actor_name = graph->ToString() + kSuperKernelActorNameSuffix;
  if (graph->is_any_type_input()) {
    to_actor_name = graph->ToString() + kAnyTypeKernelActorNameSuffix;
  }
  auto to_actor = FetchActor(to_actor_name);
  MS_EXCEPTION_IF_NULL(to_actor);
  const auto &parser = graph_compiler_info.control_node_parser_;
  MS_EXCEPTION_IF_NULL(parser);

  auto &input_nodes = graph->input_nodes();
  for (size_t node_index = 0; node_index < input_nodes.size(); ++node_index) {
    auto &input_node = input_nodes[node_index];
    MS_EXCEPTION_IF_NULL(input_node);
    if (parser->IsControlFlowDataArrow(graph, input_node)) {
      continue;
    }
    if (SchedulerHelper::HasMonadControl(input_node, graph)) {
      MS_LOG(INFO) << "The graph:" << graph->graph_id()
                   << " has abstract monad input node:" << input_node->DebugString() << ", input index:" << node_index;
      LinkControlArrowByAutoMonad(to_actor, input_node, graph);
      LinkControlArrowForAnyTypeKernelActor(to_actor, input_node, graph, parser);
    }
    // No data arrow for monad input.
    if (HasAbstractMonad(input_node)) {
      continue;
    }
    KernelWithIndex from_kernel_with_output_idx = std::make_pair(input_node, 0);
    KernelWithIndex to_kernel_with_input_idx = std::make_pair(input_node, node_index);
    // The gather of linking data arrows of kernel by the different from kernel type.
    LinkDataArrow(to_actor, graph_compiler_info, graph, from_kernel_with_output_idx, to_kernel_with_input_idx);
  }

  if (graph->is_any_type_input()) {
    return;
  }

  // Foreach the execution order to add the auto monad device tensor stores.
  auto &execution_order = graph->execution_order();
  (void)std::for_each(execution_order.begin(), execution_order.end(), [&](const CNodePtr &kernel) {
    SchedulerHelper::AddMonadDeviceTensorStore(to_actor, kernel, graph);
  });
  if (to_actor->auto_monad_device_tensor_stores_.size() > 0) {
    (void)auto_monad_actors->emplace_back(to_actor);
  }
}

namespace {
bool IsNeedLinkControlArrowByMonad(const KernelGraphPtr &graph, const GraphCompilerInfo &graph_compiler_info) {
  MS_EXCEPTION_IF_NULL(graph);
  if (EnableKbkSubGraphExecute() || graph->is_graph_run_mode() || graph->is_any_type_input() ||
      graph->execution_order().empty()) {
    MS_LOG(INFO) << "No need to link control arrow for graph:" << graph->ToString();
    return false;
  }

  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  if (runtime::RuntimeConf::GetInstance()->mem_optimize_level() != kOptimizeO0) {
    MS_LOG(INFO) << "No need to link control arrow for graph:" << graph->ToString();
    return false;
  }

  for (const auto &kernel : graph->execution_order()) {
    MS_EXCEPTION_IF_NULL(kernel);
    if (common::AnfAlgo::IsNaiveCommunicationOp(kernel)) {
      MS_LOG(INFO) << "No need to link control arrow for graph:" << graph->ToString()
                   << " by kernel:" << kernel->fullname_with_scope();
      return false;
    }
  }
  MS_LOG(INFO) << "Need to link control arrow for graph:" << graph->ToString();
  return true;
}
}  // namespace

void GraphScheduler::LinkDataArrowInNonSinkMode(const KernelGraphPtr &graph,
                                                const GraphCompilerInfo &graph_compiler_info,
                                                std::vector<AbstractActor *> *const auto_monad_actors,
                                                std::vector<CNodePtr> *const communication_nodes) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(auto_monad_actors);
  MS_EXCEPTION_IF_NULL(communication_nodes);

  if (EnableKbkSubGraphExecute()) {
    return;
  }

  // Collect all the depend updatestate nodes of the kernels for linking control arrow.
  mindspore::HashMap<AnfNodePtr, std::set<AnfNodePtr>> cnode_to_monad_inputs;
  MS_LOG(INFO) << "Get all monad input of cnode in graph:" << graph->ToString() << " start.";
  bool is_need_auto_monad_link = IsNeedLinkControlArrowByMonad(graph, graph_compiler_info);
  if (is_need_auto_monad_link) {
    GetAllCNodeUInputByGraph(graph, &cnode_to_monad_inputs);
  }
  MS_LOG(INFO) << "Get all monad input of cnode in graph:" << graph->ToString() << " end.";

  auto &execution_order = graph->execution_order();
  // Foreach the execution order to link the actors.
  for (const auto &kernel : execution_order) {
    MS_EXCEPTION_IF_NULL(kernel);
    MS_LOG(DEBUG) << "Graph " << graph->graph_id() << " execution order node: " << kernel->fullname_with_scope();
    if (common::AnfAlgo::IsNaiveCommunicationOp(kernel)) {
      MS_LOG(DEBUG) << "Graph " << graph->graph_id()
                    << " execution order communication node: " << kernel->fullname_with_scope();
      (void)communication_nodes->emplace_back(kernel);
    }
    if (IsSkippedKernelActor(kernel) || (!IsKernelActor(kernel, graph_compiler_info.strategy_))) {
      continue;
    }
    const auto &kernel_actor = FetchActor(GetActorIdByKernel(kernel));
    MS_EXCEPTION_IF_NULL(kernel_actor);

    for (size_t i = 0; i < common::AnfAlgo::GetInputNum(kernel); ++i) {
      auto input_node = common::AnfAlgo::GetInputNode(kernel, i);
      // Link the control arrows of kernel actor by the auto monad, the inputs include monad node.
      if (is_need_auto_monad_link && SchedulerHelper::HasMonadControl(input_node, graph)) {
        std::set<AnfNodePtr> checked_nodes;
        LinkControlArrowByAutoMonad(kernel_actor, input_node, graph, graph_compiler_info.control_node_parser_,
                                    cnode_to_monad_inputs, &checked_nodes);
      }
      // No data arrow for monad input.
      if (HasAbstractMonad(input_node)) {
        continue;
      }

      KernelWithIndex from_kernel_with_output_idx = common::AnfAlgo::VisitKernelWithReturnType(input_node, 0, false);
      KernelWithIndex to_kernel_with_input_idx = std::make_pair(kernel, i);
      // The data arrow linking is taken over by the control flow.
      if (graph_compiler_info.control_node_parser_ != nullptr &&
          graph_compiler_info.control_node_parser_->IsControlFlowDataArrow(graph, from_kernel_with_output_idx.first)) {
        continue;
      }
      // The gather of linking data arrows of kernel by the different from kernel type.
      LinkDataArrow(kernel_actor, graph_compiler_info, graph, from_kernel_with_output_idx, to_kernel_with_input_idx);
    }

    // Add the auto monad device tensor stores.
    SchedulerHelper::AddMonadDeviceTensorStore(kernel_actor, kernel, graph);
    if (kernel_actor->auto_monad_device_tensor_stores_.size() > 0) {
      (void)auto_monad_actors->emplace_back(kernel_actor);
    }
  }

  // Link the control arrows for allreduce kernel by the send/recv nodes in the kernel graph.
  LinkControlArrowBySendRecvNodes(graph);
}

void GraphScheduler::LinkDataArrow(AbstractActor *const to_actor, const GraphCompilerInfo &graph_compiler_info,
                                   const KernelGraphPtr &graph, const KernelWithIndex &from_kernel_with_output_idx,
                                   const KernelWithIndex &to_kernel_with_input_idx) {
  MS_EXCEPTION_IF_NULL(to_actor);
  MS_EXCEPTION_IF_NULL(graph);

  auto from_kernel = from_kernel_with_output_idx.first;
  MS_EXCEPTION_IF_NULL(from_kernel);
  auto kernel_type = FetchKernelTransformType(from_kernel, graph, graph_compiler_info.origin_parameters_order_,
                                              graph_compiler_info.strategy_);
  auto from_actor = FetchActor(kernel_type, graph_compiler_info.name_, from_kernel, graph);

  if (kKernelTypeToLinkFunc.count(kernel_type) == 0) {
    if (graph_compiler_info.strategy_ == GraphExecutionStrategy::kPipeline) {
      MS_LOG_WITH_NODE(INTERNAL_EXCEPTION, from_kernel)
        << "#dmsg#Runtime error info:#dmsg#Invalid from node:" << from_kernel->fullname_with_scope()
        << " to actor:" << to_actor->GetAID().Name() << ", type:" << kernel_type;
    }
    return;
  }
  (this->*kKernelTypeToLinkFunc[kernel_type])(from_actor, to_actor, from_kernel_with_output_idx,
                                              to_kernel_with_input_idx, graph);
}

void GraphScheduler::LinkDataArrowForDeviceTensorStore(AbstractActor *const, AbstractActor *const to_actor,
                                                       const KernelWithIndex &from_kernel_with_output_idx,
                                                       const KernelWithIndex &to_kernel_with_input_idx,
                                                       const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(to_actor);
  MS_EXCEPTION_IF_NULL(graph);

  auto from_kernel = from_kernel_with_output_idx.first;
  MS_EXCEPTION_IF_NULL(from_kernel);
  auto device_tensor_store_key = AnfAlgo::FetchFrontNodeByBackendNode(from_kernel, *graph);
  MS_EXCEPTION_IF_NULL(device_tensor_store_key);
  // front graph:
  // ------- %0 = TupleGetItem((1), 0)
  // ------- %1 = MakeTuple(%0)
  // ------- %2 = Convolution(para1, para2, %1)
  // return %2
  // after split the graph:
  // backend graph 1:
  // ------- %0 = MakeTuple(1)
  // ------- return %0
  // backend graph 2:
  // ------- %0 = MakeTuple(para3)
  // ------- %1 = Convolution(para1, para2, %0)
  // ------- return %1
  // the front node of para3 would be the scalar 1 in value tuple (1), and it only exist in graph 1.
  if (from_kernel->isa<Parameter>() && device_tensor_store_key->isa<ValueNode>() &&
      device_tensor_store_key->abstract() != nullptr &&
      device_tensor_store_key->abstract()->isa<abstract::AbstractSequence>()) {
    auto front_node_with_index = graph->GetFrontNodeByInternalParameter(from_kernel);
    if (graph_output_to_actor_.count(front_node_with_index) > 0 &&
        graph_output_to_actor_[front_node_with_index].second.first != nullptr) {
      MS_LOG(DEBUG) << "Update device tensor store key from:" << device_tensor_store_key->DebugString()
                    << " device_tensor_store_key:" << device_tensor_store_key.get()
                    << " to:" << graph_output_to_actor_[front_node_with_index].second.first->DebugString()
                    << " to node:" << graph_output_to_actor_[front_node_with_index].second.first.get()
                    << " index:" << to_kernel_with_input_idx.second << " for actor:" << to_actor->GetAID();
      device_tensor_store_key = graph_output_to_actor_[front_node_with_index].second.first;
      if (AnfAlgo::ExistOutputKernelTensor(from_kernel, 0)) {
        auto kernel_tensor = AnfAlgo::GetOutputKernelTensor(from_kernel, 0, false);
        if (kernel_tensor != nullptr && device_tensor_store_key != nullptr &&
            device_tensor_store_key->isa<ValueNode>()) {
          const auto &value_node = device_tensor_store_key->cast<ValueNodePtr>();
          MS_EXCEPTION_IF_NULL(value_node);
          kernel_tensor->set_value(value_node->value());
          SchedulerHelper::AddDeviceTensorStore(graph_output_to_actor_[front_node_with_index].second.first,
                                                kernel_tensor);
        }
      }
    }
  }
  (void)to_actor->device_tensor_store_keys_.emplace_back(to_kernel_with_input_idx.second, device_tensor_store_key);
}

void GraphScheduler::LinkDataArrowForInternalParameter(AbstractActor *const, AbstractActor *to_actor,
                                                       const KernelWithIndex &from_kernel_with_output_idx,
                                                       const KernelWithIndex &to_kernel_with_input_idx,
                                                       const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(to_actor);
  MS_EXCEPTION_IF_NULL(graph);
  auto internal_parameter = from_kernel_with_output_idx.first;
  MS_EXCEPTION_IF_NULL(internal_parameter);

  // Parameter ---> front node.
  auto front_output_with_index = graph->GetOriginFrontNodeByInternalParameter(internal_parameter);
  auto front_output_node = front_output_with_index.first;
  MS_EXCEPTION_IF_NULL(front_output_node);
  if (IsSwitchActor(front_output_node)) {
    return;
  }

  auto real_from_kernel_with_output_idx = from_kernel_with_output_idx;
  AbstractActor *real_from_actor = nullptr;
  KernelTransformType kernel_type;
  if (IsInternalParameterInParameterStore(front_output_node)) {
    kernel_type = KernelTransformType::kGraphParameterStore;
  } else if (IsPersistentDeviceTensor(front_output_node)) {
    kernel_type = KernelTransformType::kDeviceTensorStore;
  } else {
    // front node ---> actor.
    if (graph_output_to_actor_.count(front_output_with_index) == 0) {
      MS_LOG_WITH_NODE(INTERNAL_EXCEPTION, front_output_node)
        << "#dmsg#Runtime error info:#dmsg#Can't find actor by front node:" << front_output_node->DebugString()
        << ", internal parameter:" << internal_parameter->DebugString();
    }
    auto actor_pair = graph_output_to_actor_[front_output_with_index];
    real_from_actor = actor_pair.first;
    MS_EXCEPTION_IF_NULL(actor_pair.second.first);
    real_from_kernel_with_output_idx = common::AnfAlgo::FetchRealNodeSkipMonadControl(actor_pair.second);
    if (real_from_actor == nullptr) {
      if (!actor_pair.second.first->isa<ValueNode>()) {
        MS_LOG(EXCEPTION) << "Cannot get from actor by front node:" << front_output_node->DebugString()
                          << " index:" << front_output_with_index.second
                          << " for internal parameter:" << internal_parameter->DebugString()
                          << " to actor:" << to_actor->GetAID();
      }
      MS_LOG(INFO) << "Cannot get from actor by front node:" << front_output_node->DebugString()
                   << " index:" << front_output_with_index.second
                   << " for internal parameter:" << internal_parameter->DebugString()
                   << " to actor:" << to_actor->GetAID() << " and check the device tensor store.";
      LinkDataArrowForDeviceTensorStore(real_from_actor, to_actor, real_from_kernel_with_output_idx,
                                        to_kernel_with_input_idx, graph);
      return;
    }
    MS_LOG(INFO) << "Graph " << graph->graph_id() << " internal parameter:" << internal_parameter->DebugString()
                 << ", corresponding front node:" << front_output_node->fullname_with_scope()
                 << " with index:" << front_output_with_index.second
                 << ", from actor:" << actor_pair.first->GetAID().Name()
                 << " node:" << actor_pair.second.first->fullname_with_scope()
                 << " with index:" << actor_pair.second.second << ", to actor:" << to_actor->GetAID().Name()
                 << " with index:" << to_kernel_with_input_idx.second;
    // The data arrow need skip the monad node.
    kernel_type = actor_pair.first->type_;

    // The format of internal parameter need update in the heterogeneous scene.
    auto parameter_kernel_tensor = AnfAlgo::GetOutputKernelTensor(internal_parameter, 0);
    if ((parameter_kernel_tensor != nullptr) && !(graph->is_graph_run_mode() && !EnableKbkSubGraphExecute())) {
      auto format =
        AnfAlgo::GetOutputFormat(real_from_kernel_with_output_idx.first, real_from_kernel_with_output_idx.second);
      parameter_kernel_tensor->set_format(kernel::GetFormatFromStrToEnum(format));
    }
  }

  if (kKernelTypeToLinkFunc.count(kernel_type) == 0) {
    MS_EXCEPTION_IF_NULL(internal_parameter);
    MS_LOG_WITH_NODE(INTERNAL_EXCEPTION, internal_parameter)
      << "#dmsg#Runtime error info:#dmsg#Invalid internal parameter:" << internal_parameter->DebugString()
      << ", type:" << kernel_type;
  }
  (this->*kKernelTypeToLinkFunc[kernel_type])(real_from_actor, to_actor, real_from_kernel_with_output_idx,
                                              to_kernel_with_input_idx, graph);
}

DeviceContext *GetFromActorDeviceContext(AbstractActor *const from_actor, AbstractActor *const to_actor,
                                         const KernelWithIndex &from_kernel_with_output_idx) {
  DeviceContext *from_device_context = nullptr;
  if (EnableInputOptimize() && from_actor->type() == KernelTransformType::kDataPrepareActor) {
    auto device_tensor =
      AnfAlgo::GetMutableOutputAddr(from_kernel_with_output_idx.first, from_kernel_with_output_idx.second, false);
    MS_EXCEPTION_IF_NULL(device_tensor);
    from_device_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(
      {device_tensor->GetDeviceType(), device_tensor->device_id()});
    return from_device_context;
  }
  auto position = from_actor->FetchNodePosition({from_kernel_with_output_idx.first, 0});
  if ((from_actor->device_contexts().size() <= position) || to_actor->device_contexts().empty()) {
    MS_LOG(INTERNAL_EXCEPTION) << "#dmsg#Runtime error info:#dmsg#The device contexts size is wrong.";
  }
  from_device_context = const_cast<DeviceContext *>(from_actor->device_contexts()[position]);
  return from_device_context;
}

bool IsNeedInsertCopyActor(AbstractActor *const from_actor, AbstractActor *const to_actor,
                           const KernelWithIndex &from_kernel_with_output_idx,
                           const KernelWithIndex &to_kernel_with_input_idx) {
  MS_EXCEPTION_IF_NULL(from_actor);
  MS_EXCEPTION_IF_NULL(to_actor);

  auto from_kernel = from_kernel_with_output_idx.first;
  MS_EXCEPTION_IF_NULL(from_kernel);
  auto to_input_index = to_kernel_with_input_idx.second;

  auto from_device_context = GetFromActorDeviceContext(from_actor, to_actor, from_kernel_with_output_idx);
  MS_EXCEPTION_IF_NULL(from_device_context);

  bool need_copy = true;
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  static const bool enable_infer_boost = ms_context->IsEnableInferBoost();
  auto to_kernel_actor = dynamic_cast<KernelActor *>(to_actor);
  if (to_kernel_actor != nullptr && !enable_infer_boost) {
    auto to_kernel = to_kernel_actor->kernel();
    auto cnode = to_kernel->cast<CNodePtr>();
    if (cnode != nullptr) {
      MS_LOG(DEBUG) << "Process shape depend attribute for cnode : " << cnode->fullname_with_scope();
      const auto &only_depend_shape_attr = common::AnfAlgo::GetCNodePrimitiveAttr(cnode, kAttrOnlyDependShape);
      if (only_depend_shape_attr != nullptr) {
        auto only_depend_shape = GetValue<std::vector<bool>>(only_depend_shape_attr);
        if (only_depend_shape.size() <= to_input_index) {
          MS_LOG(DEBUG) << "Index out of range: to_input_index=" << to_input_index
                        << ", only_depend_shape size=" << only_depend_shape.size();
        } else {
          need_copy = !only_depend_shape[to_input_index];
          MS_LOG(DEBUG) << "only_depend_shape [" << to_input_index << "] " << need_copy;
        }
      }
    }
  }
  auto need_copy_actor = (!SchedulerHelper::IsIgnoredInputAddress(to_actor, to_input_index)) && need_copy &&
                         IsDeviceTypeNotSame(from_device_context, to_actor->device_contexts()[0]);
  return need_copy_actor;
}

CopyActor *GraphScheduler::CreateCopyActor(AbstractActor *const from_actor, AbstractActor *const to_actor,
                                           const KernelWithIndex &from_kernel_with_output_idx,
                                           const KernelWithIndex &to_kernel_with_input_idx) {
  MS_EXCEPTION_IF_NULL(from_actor);
  MS_EXCEPTION_IF_NULL(to_actor);
  auto from_kernel = from_kernel_with_output_idx.first;
  MS_EXCEPTION_IF_NULL(from_kernel);
  std::string name =
    "copy_from:" + from_actor->GetAID().Name() + "_node:" +
    (from_kernel->fullname_with_scope() == "" ? std::to_string(reinterpret_cast<int64_t>(from_kernel.get()))
                                              : from_kernel->fullname_with_scope()) +
    "_output_index:" + std::to_string(from_kernel_with_output_idx.second);
  CopyActor *copy_actor = dynamic_cast<CopyActor *>(FetchActor(name));
  // Link between from actor and copy actor.
  if (copy_actor == nullptr) {
    auto to_kernel = to_kernel_with_input_idx.first;
    if ((from_kernel->isa<CNode>() && from_kernel->cast<CNodePtr>()->HasAttr(kAttrStreamId)) ||
        (to_kernel->isa<CNode>() && to_kernel->cast<CNodePtr>()->HasAttr(kAttrStreamId))) {
      MS_LOG(INTERNAL_EXCEPTION)
        << "#dmsg#Runtime error info:#dmsg#Currently not supported usr set stream for heterogeneous scenarios.";
    }
    KernelGraphPtr from_graph = nullptr;
    if (from_actor->type() == KernelTransformType::kSuperKernelActor ||
        from_actor->type() == KernelTransformType::kAnyTypeKernelActor) {
      auto super_kernel_actor = dynamic_cast<SuperKernelActor *>(from_actor);
      MS_EXCEPTION_IF_NULL(super_kernel_actor);
      from_graph = super_kernel_actor->graph();
    }
    // Create the copy actor.
    auto copy_actor_shared_ptr = std::make_shared<CopyActor>(name, from_kernel.get(), from_graph, memory_manager_aid_);
    (void)copy_actors_.emplace_back(copy_actor_shared_ptr);
    copy_actor = copy_actor_shared_ptr.get();
    MS_EXCEPTION_IF_NULL(copy_actor);
    InsertActor(copy_actor);

    // Set the member device_contexts_ of the copy actor.
    auto from_device_context = GetFromActorDeviceContext(from_actor, to_actor, from_kernel_with_output_idx);
    auto to_device_context = to_actor->device_contexts_[0];
    MS_EXCEPTION_IF_NULL(from_device_context);
    MS_EXCEPTION_IF_NULL(to_device_context);
    (void)copy_actor->device_contexts_.emplace_back(from_device_context);
    (void)copy_actor->device_contexts_.emplace_back(to_device_context);

    // Set the member output_ of the copy actor.
    if (!EnableKbkSubGraphExecute() && (to_actor->type_ == KernelTransformType::kSuperKernelActor ||
                                        to_actor->type_ == KernelTransformType::kAnyTypeKernelActor)) {
      // Use address of to_kernel directly to avoid data copy in the subgraph sink.
      copy_actor->output_ = AnfAlgo::GetOutputKernelTensor(to_kernel_with_input_idx.first, 0, false);
    } else {
      const auto &pre_kernel_tensor = !to_kernel_with_input_idx.first->isa<CNode>()
                                        ? AnfAlgo::GetOutputKernelTensor(to_kernel_with_input_idx.first, 0, false)
                                        : AnfAlgo::GetPrevNodeOutputKernelTensor(
                                            to_kernel_with_input_idx.first, to_kernel_with_input_idx.second, false);
      MS_EXCEPTION_IF_NULL(pre_kernel_tensor);

      const auto new_kernel_tensor =
        SchedulerHelper::CloneKernelTensorWithDeviceInfo(pre_kernel_tensor, to_device_context);
      MS_EXCEPTION_IF_NULL(new_kernel_tensor);
      new_kernel_tensor->set_device_ptr(nullptr);

      copy_actor->output_ = new_kernel_tensor;
      MS_LOG(DEBUG) << "Create kernel tensor:" << copy_actor->output_;
    }
    MS_EXCEPTION_IF_NULL(copy_actor->output_);
    if (copy_actor->output_->GetDeviceType() != to_device_context->GetDeviceType()) {
      MS_LOG(INTERNAL_EXCEPTION) << "#dmsg#Runtime error info:#dmsg#The device type is not equal, output device type:"
                                 << copy_actor->output_->GetDeviceType()
                                 << ", to device context type:" << to_device_context->GetDeviceType();
    }
    copy_actor->is_need_update_output_size_ = common::AnfAlgo::IsDynamicShape(to_kernel_with_input_idx.first) ||
                                              common::AnfAlgo::IsNodeOutputDynamicShape(to_kernel_with_input_idx.first);

    // Link between from actor and copy actor.
    SchedulerHelper::AddDataArrow(from_actor, copy_actor, from_kernel_with_output_idx.second, 0, from_kernel);
  }
  return copy_actor;
}

bool IsRefNode(const KernelWithIndex &from_node_with_index, const KernelWithIndex &to_node_with_index) {
  auto from_node = from_node_with_index.first;
  MS_EXCEPTION_IF_NULL(from_node);
  auto to_node = to_node_with_index.first;
  if (to_node != nullptr && to_node->isa<CNode>()) {
    const auto &cnode = to_node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    if (to_node_with_index.second < cnode->size() && cnode->input(to_node_with_index.second) != nullptr &&
        cnode->input(to_node_with_index.second)->isa<CNode>()) {
      return false;
    }
  }
  return from_node->abstract() != nullptr &&
         common::AnfAlgo::FetchAbstractByIndex(from_node->abstract(), from_node_with_index.second)
           ->isa<abstract::AbstractRefTensor>();
}

void GraphScheduler::LinkDataArrowForGraphParameterStore(AbstractActor *const, AbstractActor *const to_actor,
                                                         const KernelWithIndex &from_kernel_with_output_idx,
                                                         const KernelWithIndex &to_kernel_with_input_idx,
                                                         const KernelGraphPtr &graph) {
  // Obtain the corresponding front node from back node.
  MS_EXCEPTION_IF_NULL(from_kernel_with_output_idx.first);
  auto real_from_kernel_with_idx = common::AnfAlgo::FetchRealNodeSkipMonadControl(from_kernel_with_output_idx);
  MS_EXCEPTION_IF_NULL(real_from_kernel_with_idx.first);
  KernelWithIndex front_node_with_idx{nullptr, 0};
  if (IsInternalParameter(real_from_kernel_with_idx.first, graph)) {
    front_node_with_idx = graph->GetFrontNodeByInternalParameter(real_from_kernel_with_idx.first);
  } else {
    front_node_with_idx = graph->GetElementInTupleBackendFrontIndexMap(real_from_kernel_with_idx.first);
    if (front_node_with_idx.first == nullptr) {
      front_node_with_idx = {AnfAlgo::FetchFrontNodeByBackendNode(real_from_kernel_with_idx.first, *graph), 0};
    }
  }

  // from actor is null, set data prepare actor as from actor.
  auto data_prepare_actor_name = ParameterStore::GetInstance().GetChosenGraphName() + kDataPrepareActorNameSuffix;
  auto data_prepare_actor = FetchActor(data_prepare_actor_name);
  MS_EXCEPTION_IF_NULL(data_prepare_actor);

  SchedulerHelper::InsertParameterIndexsForActor(to_actor, front_node_with_idx, real_from_kernel_with_idx,
                                                 to_kernel_with_input_idx, graph);
}

void GraphScheduler::LinkDataArrowForBaseActor(AbstractActor *const from_actor, AbstractActor *const to_actor,
                                               const KernelWithIndex &from_kernel_with_output_idx,
                                               const KernelWithIndex &to_kernel_with_input_idx,
                                               const KernelGraphPtr &) {
  MS_EXCEPTION_IF_NULL(from_actor);
  MS_EXCEPTION_IF_NULL(to_actor);

  auto from_kernel = from_kernel_with_output_idx.first;
  MS_EXCEPTION_IF_NULL(from_kernel);
  auto from_output_index = from_kernel_with_output_idx.second;
  auto to_input_index = to_kernel_with_input_idx.second;

  auto is_need_copy =
    IsNeedInsertCopyActor(from_actor, to_actor, from_kernel_with_output_idx, to_kernel_with_input_idx);
  if (is_need_copy) {
    LinkDataArrowForCopyActor(from_actor, to_actor, from_kernel_with_output_idx, to_kernel_with_input_idx);
  } else {
    SchedulerHelper::AddDataArrow(from_actor, to_actor, from_output_index, to_input_index, from_kernel);
  }
}

void GraphScheduler::LinkDataArrowForHostDSActor(AbstractActor *const from_actor, AbstractActor *const to_actor,
                                                 const KernelWithIndex &from_kernel_with_output_idx,
                                                 const KernelWithIndex &to_kernel_with_input_idx,
                                                 const KernelGraphPtr &graph) {
  auto host_ds_actor = dynamic_cast<HostQueueDataSourceActor *>(from_actor);
  MS_EXCEPTION_IF_NULL(host_ds_actor);
  MS_EXCEPTION_IF_NULL(from_kernel_with_output_idx.first);

  KernelWithIndex real_from_kernel_with_output_idx = from_kernel_with_output_idx;
  // Get the position and real kernel by from kernel in the data source actor.
  auto position = host_ds_actor->FetchNodePosition({from_kernel_with_output_idx.first, 0});
  real_from_kernel_with_output_idx.first = host_ds_actor->FetchNode(position).first;

  LinkDataArrowForBaseActor(from_actor, to_actor, real_from_kernel_with_output_idx, to_kernel_with_input_idx, graph);
}

void GraphScheduler::LinkDataArrowForKernelActor(AbstractActor *const from_actor, AbstractActor *const to_actor,
                                                 const KernelWithIndex &from_kernel_with_output_idx,
                                                 const KernelWithIndex &to_kernel_with_input_idx,
                                                 const KernelGraphPtr &graph) {
  auto real_from_actor = from_actor;
  auto real_from_kernel_with_output_idx = from_kernel_with_output_idx;
  auto from_kernel = from_kernel_with_output_idx.first;

  // Update the from kernel info by the real node info.
  MS_EXCEPTION_IF_NULL(from_kernel);
  if (IsSkippedKernelActor(from_kernel)) {
    real_from_kernel_with_output_idx = common::AnfAlgo::GetPrevNodeOutput(from_kernel, 0, false);
    MS_EXCEPTION_IF_NULL(real_from_kernel_with_output_idx.first);
    MS_EXCEPTION_IF_NULL(to_kernel_with_input_idx.first);
    LinkControlArrowBySkippedNode(to_actor, from_kernel, graph);
    MS_LOG(INFO) << "Link data arrow for inplace node, aggregate node: "
                 << to_kernel_with_input_idx.first->fullname_with_scope()
                 << ", aggregate input index: " << to_kernel_with_input_idx.second
                 << ", skip node: " << from_kernel->fullname_with_scope()
                 << ", real node: " << real_from_kernel_with_output_idx.first->fullname_with_scope();
    real_from_actor = FetchActor(GetActorIdByKernel(real_from_kernel_with_output_idx.first));
    MS_EXCEPTION_IF_NULL(real_from_actor);
  }

  LinkDataArrowForBaseActor(real_from_actor, to_actor, real_from_kernel_with_output_idx, to_kernel_with_input_idx,
                            graph);
}

void GraphScheduler::LinkDataArrowForCopyActor(AbstractActor *const from_actor, AbstractActor *const to_actor,
                                               const KernelWithIndex &from_kernel_with_output_idx,
                                               const KernelWithIndex &to_kernel_with_input_idx) {
  MS_EXCEPTION_IF_NULL(from_actor);
  MS_EXCEPTION_IF_NULL(to_actor);
  auto from_kernel = from_kernel_with_output_idx.first;
  MS_EXCEPTION_IF_NULL(from_kernel);
  MS_LOG(DEBUG) << "Link data arrow for copy actor from actor:" << from_actor->GetAID()
                << " to actor:" << to_actor->GetAID() << " from kernel:" << from_kernel->DebugString()
                << " from index:" << from_kernel_with_output_idx.second << " to kernel:"
                << (to_kernel_with_input_idx.first == nullptr ? "null" : to_kernel_with_input_idx.first->DebugString())
                << " to index:" << to_kernel_with_input_idx.second;
  auto copy_actor = CreateCopyActor(from_actor, to_actor, from_kernel_with_output_idx, to_kernel_with_input_idx);

  // If the copy actor already exists, only need link between copy actor and to actor.
  SchedulerHelper::AddDataArrow(copy_actor, to_actor, 0, to_kernel_with_input_idx.second, nullptr);
  MS_EXCEPTION_IF_NULL(copy_actor->output_);
  auto copy_output_kernel_tensor = copy_actor->output_;
  MS_EXCEPTION_IF_NULL(copy_output_kernel_tensor);
  copy_output_kernel_tensor->ClearFlag(device::kDeviceAddressFlagNotUsed);
  if (!EnableKbkSubGraphExecute() && (to_actor->type_ == KernelTransformType::kSuperKernelActor ||
                                      to_actor->type_ == KernelTransformType::kAnyTypeKernelActor)) {
    return;
  }
  if (IsRefNode(from_kernel_with_output_idx, to_kernel_with_input_idx)) {
    MS_LOG(DEBUG) << "Set ref count to max for ref output of kernel:" << from_kernel->DebugString()
                  << " index:" << from_kernel_with_output_idx.second;
    const auto &pre_device_tensor =
      !to_kernel_with_input_idx.first->isa<CNode>()
        ? AnfAlgo::GetMutableOutputAddr(to_kernel_with_input_idx.first, 0, false)
        : AnfAlgo::GetPrevNodeMutableOutputAddr(to_kernel_with_input_idx.first, to_kernel_with_input_idx.second, false);
    MS_EXCEPTION_IF_NULL(pre_device_tensor);
    copy_actor->ref_parameter_device_tensors_.emplace(pre_device_tensor);
    MS_LOG(DEBUG) << "Add ref parameter device address:" << pre_device_tensor << " for actor:" << copy_actor->GetAID();
  }
}

namespace {
std::vector<AnfNodePtr> FetchRealDependInput(
  const AnfNodePtr &node, const mindspore::HashMap<AnfNodePtr, std::set<AnfNodePtr>> &cnode_to_monad_inputs) {
  std::vector<AnfNodePtr> real_depend_inputs;
  const auto &cnode = node->cast<CNodePtr>();
  if (common::AnfAlgo::CheckPrimitiveType(node, prim::kPrimDepend) ||
      common::AnfAlgo::CheckPrimitiveType(node, prim::kPrimLoad)) {
    MS_EXCEPTION_IF_NULL(cnode);
    // In particular, in the depend->updatestate scene, in order to prevent the loss of the topo relationship,
    // the first input of depend must be linked. In the othe side, real input may be this scene:  depend/load -->
    // load/depend, so need add the control arrow for real input node in this scene.
    real_depend_inputs.push_back(cnode->input(kRealInputIndexInDepend));
    real_depend_inputs.push_back(cnode->input(kDependAttachNodeIndex));
  } else if (common::AnfAlgo::CheckPrimitiveType(node, prim::kPrimUpdateState)) {
    MS_EXCEPTION_IF_NULL(cnode);
    if (IsNeedLinkForFirstInput(cnode, cnode_to_monad_inputs) && cnode->size() > kUpdateStateStateInput) {
      // If all other inputs of the update state do not depend on the first input, we need to link control arrow
      // for the first input.
      real_depend_inputs.push_back(cnode->input(kUpdateStateStateInput));
    }
    for (size_t i = kUpdateStateRealInput; i < cnode->size(); ++i) {
      MS_EXCEPTION_IF_NULL(cnode);
      real_depend_inputs.push_back(cnode->input(i));
    }
  } else {
    real_depend_inputs.push_back(node);
  }
  return real_depend_inputs;
}
}  // namespace

void GraphScheduler::LinkControlArrowByAutoMonad(
  AbstractActor *to_actor, const AnfNodePtr &from_node, const KernelGraphPtr &graph, const ControlNodeParserPtr &parser,
  const mindspore::HashMap<AnfNodePtr, std::set<AnfNodePtr>> &cnode_to_monad_inputs,
  std::set<AnfNodePtr> *checked_nodes) {
  MS_EXCEPTION_IF_NULL(to_actor);
  MS_EXCEPTION_IF_NULL(from_node);
  MS_EXCEPTION_IF_NULL(graph);
  // Find the real input node, include the monad node and make tuple node.
  const std::vector<PrimitivePtr> return_types = {prim::kPrimDepend, prim::kPrimUpdateState, prim::kPrimLoad,
                                                  prim::kPrimMakeTuple};
  const auto &input_kernel_with_output_idx =
    common::AnfAlgo::VisitKernelWithReturnType(from_node, 0, false, return_types);
  auto input_anfnode = input_kernel_with_output_idx.first;
  MS_EXCEPTION_IF_NULL(input_anfnode);
  CNodePtr input_cnode = nullptr;
  if (input_anfnode->isa<CNode>()) {
    input_cnode = input_anfnode->cast<CNodePtr>();
  }

  if (checked_nodes != nullptr) {
    auto context_ptr = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(context_ptr);
    auto parallel_context = parallel::ParallelContext::GetInstance();
    MS_EXCEPTION_IF_NULL(parallel_context);
    auto stages = parallel_context->pipeline_stage_split_num();
    if (stages > 1 && context_ptr->CellReuseLevel() == CellReuseLevel::kLazyInline &&
        context_ptr->IsKByKExecutorMode()) {
      return;
    }
    if (checked_nodes->find(input_cnode) != checked_nodes->end()) {
      return;
    } else {
      (void)checked_nodes->emplace(input_cnode);
    }
  }

  // Make tuple node needs to be expanded.
  if (common::AnfAlgo::CheckPrimitiveType(input_anfnode, prim::kPrimMakeTuple)) {
    MS_EXCEPTION_IF_NULL(input_cnode);
    for (size_t i = 1; i < input_cnode->size(); ++i) {
      LinkControlArrowByAutoMonad(to_actor, input_cnode->input(i), graph, parser, cnode_to_monad_inputs, checked_nodes);
    }
    return;
  }

  // When processing the control arrow of the monad node, updatestate start from its second input. By default,
  // the first input will not be processed.
  const mindspore::HashSet<PrimitivePtr, PrimitiveHasher, PrimitiveEqual> recursion_prims = {
    prim::kPrimDepend, prim::kPrimUpdateState, prim::kPrimLoad, prim::kPrimMakeTuple};
  // Get the real depend input by monad node which needs to link the control arrow.
  std::vector<AnfNodePtr> real_depend_inputs = FetchRealDependInput(input_anfnode, cnode_to_monad_inputs);
  for (const auto &real_depend_input : real_depend_inputs) {
    MS_EXCEPTION_IF_NULL(real_depend_input);
    MS_LOG(DEBUG) << "Add real depend node:" << real_depend_input->DebugString()
                  << " by input node:" << input_anfnode->DebugString();
    auto real_depend_input_with_idx =
      common::AnfAlgo::VisitKernelWithReturnType(real_depend_input, 0, false, return_types);
    MS_EXCEPTION_IF_NULL(real_depend_input_with_idx.first);
    auto real_depend_kernel = real_depend_input_with_idx.first;
    auto real_graph = graph;
    // Update the real depend kernel in the subgraphs connecting scene.
    if (IsInternalParameter(real_depend_kernel, graph)) {
      auto front_output_with_index = graph->GetOriginFrontNodeByInternalParameter(real_depend_kernel);
      MS_EXCEPTION_IF_NULL(front_output_with_index.first);
      if (IsTakenOverByControlFlow(front_output_with_index.first, graph, parser)) {
        MS_LOG(DEBUG) << "Skip in control flow from node:" << front_output_with_index.first->DebugString()
                      << " is not in the graph:" << graph->ToString();
        continue;
      }

      if (graph_output_to_actor_.count(front_output_with_index) == 0) {
        MS_LOG_WITH_NODE(INTERNAL_EXCEPTION, front_output_with_index.first)
          << "#dmsg#Runtime error info:#dmsg#Can't find graph output by front node:"
          << front_output_with_index.first->DebugString();
      }
      real_depend_kernel = graph_output_to_actor_[front_output_with_index].second.first;
      MS_EXCEPTION_IF_NULL(real_depend_kernel);
      const auto &func_graph = real_depend_kernel->func_graph();
      if (func_graph == nullptr || std::dynamic_pointer_cast<KernelGraph>(func_graph) == nullptr) {
        MS_LOG(WARNING) << "Cannot get kernel graph for node:" << real_depend_kernel->DebugString();
      } else {
        real_graph = std::dynamic_pointer_cast<KernelGraph>(func_graph);
      }
      MS_EXCEPTION_IF_NULL(real_depend_kernel);
      MS_LOG(INFO) << "The graph " << graph->graph_id() << " link control arrow by auto monad from internal parameter: "
                   << real_depend_input_with_idx.first->DebugString()
                   << ", front output node: " << front_output_with_index.first->fullname_with_scope()
                   << ", backend output node: " << real_depend_kernel->fullname_with_scope();
      auto from_actor = graph_output_to_actor_[front_output_with_index].first;
      if (from_actor != nullptr) {
        MS_LOG(INFO) << "Link control arrow by auto monad from actor:  " << from_actor->GetAID().Name()
                     << ", to actor: " << to_actor->GetAID().Name() << " for the graph: " << graph->graph_id();
        SchedulerHelper::AddControlArrow(from_actor, to_actor);
        continue;
      }
    }

    // The monad node and make tuple node need recursion.
    if (IsOneOfPrimitiveCNode(real_depend_kernel, recursion_prims)) {
      LinkControlArrowByAutoMonad(to_actor, real_depend_kernel, real_graph, parser, cnode_to_monad_inputs,
                                  checked_nodes);
      continue;
    }

    auto type = FetchKernelTransformType(real_depend_kernel, nullptr);
    auto from_actor = FetchActor(type, "", real_depend_kernel);
    if (from_actor == nullptr) {
      MS_LOG(DEBUG) << "Link control arrow by auto monad from depend node:" << real_depend_kernel->fullname_with_scope()
                    << " is not actor for the graph: " << graph->graph_id();
      continue;
    }
    MS_LOG(INFO) << "Link control arrow by auto monad from actor:  " << from_actor->GetAID().Name()
                 << ", to actor: " << to_actor->GetAID().Name() << " for the graph: " << graph->graph_id();
    SchedulerHelper::AddControlArrow(from_actor, to_actor);
  }
}

void GraphScheduler::LinkControlArrowBySkippedNode(AbstractActor *to_actor, const AnfNodePtr &skipped_node,
                                                   const KernelGraphPtr &graph) const {
  MS_EXCEPTION_IF_NULL(to_actor);
  MS_EXCEPTION_IF_NULL(skipped_node);
  MS_EXCEPTION_IF_NULL(graph);

  // Link the control arrow from all the inputs of skipped node to the user of skipped node.
  auto input_num = common::AnfAlgo::GetInputTensorNum(skipped_node);
  for (size_t i = 0; i < input_num; ++i) {
    auto kernel_with_index = common::AnfAlgo::GetPrevNodeOutput(skipped_node, i, false);
    MS_EXCEPTION_IF_NULL(kernel_with_index.first);
    auto from_actor = FetchActor(GetActorIdByKernel(kernel_with_index.first));
    // Get the from actor by the internal parameter.
    if (IsInternalParameter(kernel_with_index.first, graph)) {
      auto front_output_with_index = graph->GetOriginFrontNodeByInternalParameter(kernel_with_index.first);
      if (graph_output_to_actor_.count(front_output_with_index) > 0) {
        from_actor = graph_output_to_actor_.at(front_output_with_index).first;
      }
    }
    if (from_actor == nullptr) {
      MS_LOG(INFO) << "Skip control arrow by skipped node: " << skipped_node->fullname_with_scope()
                   << ", with input index: " << i << ", to actor: " << to_actor->GetAID().Name();
      continue;
    }

    MS_LOG(INFO) << "Link control arrow by skipped node: " << skipped_node->fullname_with_scope()
                 << ", from actor: " << from_actor->GetAID().Name() << ", to actor: " << to_actor->GetAID().Name();
    SchedulerHelper::AddControlArrow(from_actor, to_actor);
  }
}

void GraphScheduler::LinkControlArrowBySendRecvNodes(const KernelGraphPtr &graph) const {
  MS_EXCEPTION_IF_NULL(graph);
  for (auto &to_iter : graph->send_recv_pairs_for_parallel_op_outputs()) {
    auto parallel_node = to_iter.first;
    for (auto pair : to_iter.second) {
      auto send_node = pair.first;
      auto recv_node = pair.second;
      MS_EXCEPTION_IF_NULL(parallel_node);
      MS_EXCEPTION_IF_NULL(send_node);
      MS_EXCEPTION_IF_NULL(recv_node);
      MS_LOG(INFO) << "Link control arrow for parallel node output: " << parallel_node->fullname_with_scope();
      auto parallel_actor = FetchActor(GetActorIdByKernel(parallel_node));
      auto send_actor = FetchActor(GetActorIdByKernel(send_node));
      auto recv_actor = dynamic_cast<KernelActor *>(FetchActor(GetActorIdByKernel(recv_node)));
      MS_EXCEPTION_IF_NULL(parallel_actor);
      MS_EXCEPTION_IF_NULL(send_actor);
      MS_EXCEPTION_IF_NULL(recv_actor);

      // In the scene of allreduce op and computing op parallel multi stream, the input memory of allreduce can be
      // reused only when the recv node runs finished, which is expressed by the reference count increased.
      for (size_t i = 0; i < common::AnfAlgo::GetInputTensorNum(parallel_node); ++i) {
        auto kernel_tensor = AnfAlgo::GetPrevNodeOutputKernelTensor(parallel_node, i, false);
        MS_EXCEPTION_IF_NULL(kernel_tensor);
        (void)recv_actor->external_reference_tensors_.emplace_back(kernel_tensor);
      }

      auto kernel_mod = AnfAlgo::GetKernelMod(parallel_node);
      MS_EXCEPTION_IF_NULL(kernel_mod);
      auto workspace_num = kernel_mod->GetWorkspaceSizeList().size();
      for (size_t i = 0; i < workspace_num; ++i) {
        auto kernel_tensor = AnfAlgo::GetWorkspaceKernelTensor(parallel_node, i);
        MS_EXCEPTION_IF_NULL(kernel_tensor);
        (void)recv_actor->external_reference_tensors_.emplace_back(kernel_tensor);
      }
    }
  }
}

void GraphScheduler::LinkGlobalControlArrow(ActorSet *const actor_set,
                                            const GroupNameToCommuNodes &communication_node_groups,
                                            const std::vector<AbstractActor *> &auto_monad_actors,
                                            const GraphCompilerInfo &graph_compiler_info) {
  MS_EXCEPTION_IF_NULL(actor_set);
  // Link the control arrow by the execution order.
  if (execution_order_running_) {
    for (const auto &graph : graph_compiler_info.graphs_) {
      LinkControlArrowByExecutionOrder(graph, graph_compiler_info);
    }
  }

  for (const auto &communication_nodes : communication_node_groups) {
    // Link the control arrows by the communication nodes to ensure communication nodes running order.
    LinkControlArrowByCommunicationNode(communication_nodes.second.first, communication_nodes.second.second,
                                        graph_compiler_info);
  }

  // Auto monad actor may modify the device tensor store.
  LinkDeviceTensorStoreForAutoMonadActor(auto_monad_actors, graph_compiler_info);

  // BuildNoInputKernelActor depends on whether kernel actors have input, so must be behind the link of kernel actors.
  actor_set->no_input_kernel_actors_ = BuildNoInputKernelActor(actor_set, graph_compiler_info.strategy_);

  // Link the control arrows of data prepare actor, which depends on the no input kernel actors.
  if ((graph_compiler_info.strategy_ == GraphExecutionStrategy::kPipeline) || (!IsSingleOpActorSet(actor_set))) {
    LinkControlArrowForDataPrepareActor(actor_set->data_prepare_actor_.get(), actor_set,
                                        graph_compiler_info.control_node_parser_);
  }

  LinkControlArrowForLoopCountActor(actor_set->loop_count_actor_.get(), actor_set,
                                    graph_compiler_info.control_node_parser_);

  LinkControlArrowForOutputActor(actor_set->output_actor_.get(), actor_set);

  LinkControlArrowForCopyActor(actor_set);
}

void GraphScheduler::LinkControlArrowByExecutionOrder(const KernelGraphPtr &graph,
                                                      const GraphCompilerInfo &graph_compiler_info) const {
  MS_EXCEPTION_IF_NULL(graph);
  if (graph->is_graph_run_mode() || graph->is_any_type_input() || !graph->inline_sub_graph_kernels().empty()) {
    return;
  }

  auto &execution_order = graph->execution_order();
  if (execution_order.size() <= 1) {
    return;
  }
  auto first_kernel = execution_order[0];
  const auto first_kernel_type = FetchKernelTransformType(first_kernel, graph, {}, GraphExecutionStrategy::kPipeline);
  auto last_actor = FetchActor(first_kernel_type, graph_compiler_info.name_, first_kernel, graph);
  for (size_t i = 1; i < execution_order.size(); ++i) {
    const auto &to_kernel = execution_order[i];
    const auto to_kernel_type = FetchKernelTransformType(to_kernel, graph, {}, GraphExecutionStrategy::kPipeline);
    auto to_actor = FetchActor(to_kernel_type, graph_compiler_info.name_, to_kernel, graph);
    if ((last_actor != nullptr) && (to_actor != nullptr)) {
      SchedulerHelper::AddControlArrow(last_actor, to_actor);
    } else {
      MS_LOG(WARNING) << "Skip add control arrow, from kernel: " << execution_order[i - 1]->fullname_with_scope()
                      << ", to kernel: " << to_kernel->fullname_with_scope();
    }
    if (to_actor != nullptr) {
      last_actor = to_actor;
    }
  }
}

void GraphScheduler::LinkControlArrowByCommunicationNode(const std::vector<CNodePtr> &communication_nodes,
                                                         const std::vector<KernelGraphPtr> &graphs,
                                                         const GraphCompilerInfo &graph_compiler_info) const {
  if (communication_nodes.empty()) {
    return;
  }

  if (std::none_of(graphs.begin(), graphs.end(), [](const KernelGraphPtr &graph) {
        return graph != nullptr && (!graph->inline_sub_graph_kernels().empty());
      })) {
    // Ensure communication node to execute orderly.
    for (size_t i = 1; i < communication_nodes.size(); ++i) {
      auto from_actor = FetchActor(GetActorIdByKernel(communication_nodes[i - 1]));
      auto to_actor = FetchActor(GetActorIdByKernel(communication_nodes[i]));
      MS_EXCEPTION_IF_NULL(from_actor);
      MS_EXCEPTION_IF_NULL(to_actor);
      SchedulerHelper::AddControlArrow(from_actor, to_actor);
    }
  }

  // Ensure all actors execute orderly to optimize the execution performance in the multi device scenario currently.
  // Using the multi stream to optimize the performance in the future.
  if (!execution_order_running_) {
    for (const auto &graph : graphs) {
      LinkControlArrowByExecutionOrder(graph, graph_compiler_info);
    }
  }
}

void GraphScheduler::LinkControlArrowForDataPrepareActor(DataPrepareActor *data_prepare_actor,
                                                         const ActorSet *actor_set,
                                                         const ControlNodeParserPtr &parser) const {
  MS_EXCEPTION_IF_NULL(data_prepare_actor);
  MS_EXCEPTION_IF_NULL(actor_set);
  MS_EXCEPTION_IF_NULL(parser);

  if (!EnableInputOptimize()) {
    // Data prepare actor --> data source actor.
    for (auto &data_source_actor : actor_set->data_source_actors_) {
      MS_EXCEPTION_IF_NULL(data_source_actor);
      SchedulerHelper::AddControlArrow(data_prepare_actor, data_source_actor.get());
    }
  }

  // In control flow, control arrow of no input kernel actor needs to be connected to the corresponding entrance actor.
  if (!parser->IsInited()) {
    // Data prepare actor --> no input kernel actor.
    for (auto &no_input_kernel_actor : actor_set->no_input_kernel_actors_) {
      MS_EXCEPTION_IF_NULL(no_input_kernel_actor);
      SchedulerHelper::AddControlArrow(data_prepare_actor, no_input_kernel_actor.get());
    }
  }

  // Data prepare actor --> loop count actor.
  if ((actor_set->data_source_actors_.size() + actor_set->no_input_kernel_actors_.size() == 0) &&
      (actor_set->loop_count_actor_ != nullptr)) {
    SchedulerHelper::AddControlArrow(data_prepare_actor, actor_set->loop_count_actor_.get());
  }
}

void GraphScheduler::LinkControlArrowForLoopCountActor(LoopCountActor *loop_count_actor, const ActorSet *actor_set,
                                                       const ControlNodeParserPtr &parser) {
  MS_EXCEPTION_IF_NULL(actor_set);
  MS_EXCEPTION_IF_NULL(parser);
  // There is no loop count actor in step mode.
  if (loop_count_actor == nullptr) {
    return;
  }

  auto is_no_output_actor = [](const AbstractActorPtr &actor) {
    return (actor->output_data_arrows_.size() == 0) && (actor->output_control_arrows_.size() == 0);
  };

  // Collect the actors which have no output.
  std::vector<AbstractActor *> no_output_actors;
  for (auto &super_actor : actor_set->super_kernel_actors_) {
    MS_EXCEPTION_IF_NULL(super_actor);
    if (is_no_output_actor(super_actor)) {
      (void)no_output_actors.emplace_back(super_actor.get());
    }
  }

  for (auto &actor : actor_set->any_type_kernel_actors_) {
    MS_EXCEPTION_IF_NULL(actor);
    if (is_no_output_actor(actor)) {
      (void)no_output_actors.emplace_back(actor.get());
    }
  }

  for (auto &kernel_actor : actor_set->kernel_actors_) {
    MS_EXCEPTION_IF_NULL(kernel_actor);
    // The no output kernel control side in subgraph needs to be connected to the corresponding output switch actor.
    if (is_no_output_actor(kernel_actor)) {
      (void)no_output_actors.emplace_back(kernel_actor.get());
    }
  }
  for (auto &data_actor : actor_set->data_source_actors_) {
    MS_EXCEPTION_IF_NULL(data_actor);
    if (is_no_output_actor(data_actor)) {
      (void)no_output_actors.emplace_back(data_actor.get());
    }
  }
  for (auto &copy_actor : copy_actors_) {
    MS_EXCEPTION_IF_NULL(copy_actor);
    if (is_no_output_actor(copy_actor)) {
      (void)no_output_actors.emplace_back(copy_actor.get());
    }
  }

  // No output actor --> loop count actor.
  // In control flow scenario, no output actor needs to be connected to the corresponding exit actor, not loop count.
  if (!parser->IsInited()) {
    for (auto &no_output_actor : no_output_actors) {
      SchedulerHelper::AddControlArrow(no_output_actor, loop_count_actor);
    }
  }

  // Loop count actor --> output actor.
  SchedulerHelper::AddControlArrow(loop_count_actor, actor_set->output_actor_.get());

  // Loop count actor --> data prepare actor.
  MS_EXCEPTION_IF_NULL(actor_set->data_prepare_actor_);
  loop_count_actor->data_prepare_aid_ = actor_set->data_prepare_actor_->GetAID();
  actor_set->data_prepare_actor_->input_controls_num_++;
  (void)actor_set->data_prepare_actor_->input_control_arrow_aids_.emplace_back(
    std::pair(loop_count_actor->GetAID(), nullptr));
}

void GraphScheduler::LinkControlArrowForOutputActor(OutputActor *output_actor, const ActorSet *actor_set) const {
  MS_EXCEPTION_IF_NULL(actor_set);
  // There is no output actor in step mode.
  if (output_actor == nullptr) {
    return;
  }

  // Output actor --> data prepare actor.
  // The output actor needs to free the output memory in the running and needs this control arrow.
  SchedulerHelper::AddControlArrow(output_actor, actor_set->data_prepare_actor_.get());
}

void GraphScheduler::LinkControlArrowForCopyActor(const ActorSet *actor_set) const {
  MS_EXCEPTION_IF_NULL(actor_set);
  MS_LOG(DEBUG) << "Link control arrow for copy actor start, copy actor size:" << copy_actors_.size();
  for (const auto &copy_actor : copy_actors_) {
    MS_EXCEPTION_IF_NULL(copy_actor);
    if (copy_actor->from_kernel_ == nullptr || !copy_actor->from_kernel_->isa<CNode>() ||
        copy_actor->from_kernel_->abstract() == nullptr ||
        !copy_actor->from_kernel_->abstract()->isa<abstract::AbstractRefTensor>() ||
        copy_actor->output_data_arrows_.empty()) {
      MS_LOG(DEBUG) << "Skip check add control arrow for copy actor:" << copy_actor->GetAID() << " from kernel:"
                    << (copy_actor->from_kernel_ == nullptr ? "nullptr"
                                                            : copy_actor->from_kernel_->fullname_with_scope())
                    << " from graph:"
                    << (copy_actor->from_graph_ == nullptr ? "nullptr" : copy_actor->from_graph_->ToString());
      continue;
    }
    const auto &output_arrow = copy_actor->output_data_arrows_[0];
    MS_EXCEPTION_IF_NULL(output_arrow);
    const auto &to_actor = FetchActor(output_arrow->to_op_id_.Name());
    if (to_actor == nullptr) {
      MS_LOG(WARNING) << "Failed to fetch actor by:" << output_arrow->to_op_id_.Name()
                      << " for copy actor:" << copy_actor->GetAID();
      continue;
    }
    MS_LOG(DEBUG) << "Need add control arrow for copy actor:" << copy_actor->GetAID()
                  << " by to actor:" << to_actor->GetAID();
    for (const auto &input_pair : to_actor->input_control_arrow_aids_) {
      const auto &from_actor = FetchActor(input_pair.first.Name());
      if (from_actor == nullptr || IsControlFlowActor(from_actor->type())) {
        continue;
      }
      SchedulerHelper::AddControlArrow(from_actor, copy_actor.get());
    }
  }
}

void GraphScheduler::LinkOutputResultArrowForOutputActor(OutputActor *to_actor,
                                                         const GraphCompilerInfo &graph_compiler_info) const {
  if (graph_compiler_info.strategy_ == GraphExecutionStrategy::kStep ||
      (graph_compiler_info.control_node_parser_ != nullptr && graph_compiler_info.control_node_parser_->IsInited())) {
    // In control flow, the exit actor of the root graph sends output data to the output actor.
    return;
  }
  MS_EXCEPTION_IF_NULL(to_actor);

  for (const auto &origin_output_order : graph_compiler_info.origin_outputs_order_) {
    const auto &front_output_with_index = origin_output_order.first;
    if (graph_output_to_actor_.count(front_output_with_index) == 0) {
      MS_EXCEPTION_IF_NULL(front_output_with_index.first);
      MS_LOG_WITH_NODE(INTERNAL_EXCEPTION, front_output_with_index.first)
        << "#dmsg#Runtime error info:#dmsg#Can't find graph output by front node:"
        << front_output_with_index.first->DebugString();
    }
    const auto &graph_output_pair = graph_output_to_actor_.at(front_output_with_index);
    const auto &from_actor = graph_output_pair.first;
    const auto &output_with_index = common::AnfAlgo::FetchRealNodeSkipMonadControl(graph_output_pair.second);
    auto real_from_kernel = output_with_index.first;
    auto real_from_index = output_with_index.second;
    MS_EXCEPTION_IF_NULL(real_from_kernel);
    if (IsPersistentDeviceTensor(real_from_kernel)) {
      // In the scenario where the ValueTuple is expanded, the output_with_index.second may be incorrect, so use 0 as
      // output_idx directly.
      real_from_index = 0;
    } else {
      if (from_actor == nullptr) {
        MS_EXCEPTION_IF_NULL(front_output_with_index.first);
        MS_LOG_WITH_NODE(INTERNAL_EXCEPTION, front_output_with_index.first)
          << "#dmsg#Runtime error info:#dmsg#Can't find output actor by front node:"
          << front_output_with_index.first->DebugString() << ", output node:" << real_from_kernel->DebugString();
      }
      // Update the real node in the host data source actor.
      if (from_actor->type() == KernelTransformType::kHostDataSourceActor) {
        auto host_queue_ds_actor = dynamic_cast<HostQueueDataSourceActor *>(from_actor);
        MS_EXCEPTION_IF_NULL(host_queue_ds_actor);
        auto position = host_queue_ds_actor->FetchNodePosition({real_from_kernel, 0});
        real_from_kernel = host_queue_ds_actor->FetchNode(position).first;
      }
    }

    if (EnableInputOptimize()) {
      for (auto &output_position : origin_output_order.second) {
        if (from_actor != nullptr && from_actor->type() == KernelTransformType::kDataPrepareActor) {
          MS_LOG(DEBUG) << "Output position: " << output_position
                        << " has input parameter link from node: " << front_output_with_index.first->DebugString();
          auto device_tensor = AnfAlgo::GetMutableOutputAddr(real_from_kernel, real_from_index, false);
          MS_EXCEPTION_IF_NULL(device_tensor);
          auto device_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(
            {device_tensor->GetDeviceType(), device_tensor->device_id()});
          SchedulerHelper::AddResultParameter(from_actor, to_actor, front_output_with_index, device_context,
                                              output_position);
        } else {
          SchedulerHelper::AddResultArrow(from_actor, to_actor, real_from_kernel, real_from_index, output_position);
        }
      }
      continue;
    }

    for (auto &output_position : origin_output_order.second) {
      SchedulerHelper::AddResultArrow(from_actor, to_actor, real_from_kernel, real_from_index, output_position);
    }
  }
}

void GraphScheduler::LinkKernelActorsForSubGraphExecute(const ActorSet *actor_set) const {
  MS_EXCEPTION_IF_NULL(actor_set);
  if (EnableKbkSubGraphExecute()) {
    for (auto &super_kernel_actor : actor_set->super_kernel_actors_) {
      MS_EXCEPTION_IF_NULL(super_kernel_actor);
      super_kernel_actor->BuildAndLinkKernelActors();

      std::map<size_t, std::pair<AID, DataArrow *>> input_index_to_input_arrow;
      for (const auto &pair : super_kernel_actor->input_data_arrow_aids_) {
        if (pair.second == nullptr) {
          continue;
        }
        input_index_to_input_arrow[pair.second->to_input_index_] = pair;
      }
      for (size_t i = 0; i < super_kernel_actor->is_input_used_.size(); ++i) {
        if (super_kernel_actor->is_input_used_[i]) {
          continue;
        }
        const auto &iter = input_index_to_input_arrow.find(i);
        if (iter == input_index_to_input_arrow.end()) {
          continue;
        }
        const auto &input_arrow = iter->second.second;
        const auto &from_actor = FetchActor(iter->second.first.Name());
        if (from_actor == nullptr) {
          continue;
        }
        if (from_actor->type() == KernelTransformType::kCopyActor) {
          const auto &copy_actor = dynamic_cast<CopyActor *>(from_actor);
          MS_EXCEPTION_IF_NULL(from_actor);
          copy_actor->output_free_size_++;
          MS_LOG(INFO) << "Add free size for copy actor:" << copy_actor->GetAID()
                       << " to super kernel actor:" << super_kernel_actor->GetAID();
        } else if (from_actor->type() == KernelTransformType::kSuperKernelActor) {
          const auto &from_super_kernel_actor = dynamic_cast<SuperKernelActor *>(from_actor);
          MS_EXCEPTION_IF_NULL(from_super_kernel_actor);
          if (from_super_kernel_actor->output_data_nodes_.size() !=
              from_super_kernel_actor->output_data_arrows_.size()) {
            MS_LOG(DEBUG) << "Invalid output node size:" << from_super_kernel_actor->output_data_nodes_.size()
                          << " and arrow size:" << from_super_kernel_actor->output_data_arrows_.size()
                          << " for actor:" << from_super_kernel_actor->GetAID();
            continue;
          }
          const auto &arrow_iter = std::find_if(
            from_super_kernel_actor->output_data_arrows_.begin(), from_super_kernel_actor->output_data_arrows_.end(),
            [input_arrow](const auto &arrow) { return arrow.get() == input_arrow; });
          if (arrow_iter == from_super_kernel_actor->output_data_arrows_.end()) {
            MS_LOG(DEBUG) << "Invalid input_data arrow, to actor:" << input_arrow->to_op_id_
                          << " for actor:" << super_kernel_actor->GetAID();
            continue;
          }
          size_t output_index = LongToSize(arrow_iter - from_super_kernel_actor->output_data_arrows_.begin());
          const auto &from_kernel = from_super_kernel_actor->output_data_nodes_[output_index];
          if (from_kernel == nullptr || !from_kernel->isa<CNode>() ||
              from_super_kernel_actor->cnode_to_kernel_actor_.find(from_kernel) ==
                from_super_kernel_actor->cnode_to_kernel_actor_.end()) {
            MS_LOG(DEBUG) << "Invalid from kernel:" << (from_kernel == nullptr ? "nullptr" : from_kernel->DebugString())
                          << " from actor:" << from_super_kernel_actor->GetAID()
                          << " to actor:" << super_kernel_actor->GetAID();
            continue;
          }
          const auto &kernel_actor = from_super_kernel_actor->cnode_to_kernel_actor_[from_kernel];
          MS_EXCEPTION_IF_NULL(kernel_actor);
          if (LongToSize(input_arrow->from_output_index_) >= kernel_actor->output_kernel_tensors_.size()) {
            MS_LOG(DEBUG) << "Invalid kernel actor:" << kernel_actor->GetAID()
                          << " output index:" << input_arrow->from_output_index_
                          << " for kernel:" << from_kernel->fullname_with_scope();
            continue;
          }
          auto &free_list = kernel_actor->new_memory_free_list_;
          if (free_list.size() < kernel_actor->input_free_index_.size() + kernel_actor->output_free_index_.size()) {
            MS_LOG(DEBUG) << "Invalid kernel actor:" << kernel_actor
                          << " input free list:" << kernel_actor->input_free_index_
                          << " output free list:" << kernel_actor->output_free_index_
                          << " in actor:" << from_super_kernel_actor->GetAID();
            continue;
          }
          free_list.insert(
            free_list.begin() + kernel_actor->input_free_index_.size() + kernel_actor->output_free_index_.size(),
            kernel_actor->output_kernel_tensors_[input_arrow->from_output_index_]);
          MS_LOG(INFO) << "Add free index:" << input_arrow->from_output_index_ << " device address:"
                       << kernel_actor->output_kernel_tensors_[input_arrow->from_output_index_]->device_address()
                       << " for kernel actor:" << kernel_actor->GetAID()
                       << " in super kernel actor:" << from_super_kernel_actor->GetAID();
          kernel_actor->output_free_index_.emplace_back(input_arrow->from_output_index_);
        } else {
          MS_LOG(INFO) << "Skip fix ref count for actor:" << from_actor->GetAID()
                       << " to actor:" << super_kernel_actor->GetAID() << " to index:" << i;
        }
      }
    }
  }
}

void GraphScheduler::CorrectControlArrowForAutoMonadActor(AbstractActor *const auto_monad_actor,
                                                          const AbstractActorPtr &copy_actor) {
  std::vector<AbstractActor *> output_contorl_actors;
  for (auto &output_contorl : auto_monad_actor->output_control_arrows_) {
    MS_EXCEPTION_IF_NULL(output_contorl);
    (void)output_contorl_actors.emplace_back(FetchActor(output_contorl->to_op_id_.Name()));
  }
  // Move the control arrows from auto monad actor to auto monad actor users.
  auto_monad_actor->output_control_arrows_.clear();
  for (auto &output_contorl_actor : output_contorl_actors) {
    MS_EXCEPTION_IF_NULL(output_contorl_actor);
    for (auto iter = output_contorl_actor->input_control_arrow_aids_.begin();
         iter != output_contorl_actor->input_control_arrow_aids_.end();) {
      if ((*iter).first.Name() == auto_monad_actor->GetAID().Name()) {
        iter = output_contorl_actor->input_control_arrow_aids_.erase(iter);
        output_contorl_actor->input_controls_num_--;
      } else {
        ++iter;
      }
    }
  }

  // Link from auto monad actor to copy actor.
  SchedulerHelper::AddControlArrow(auto_monad_actor, copy_actor.get());
  // Link from copy actor to auto monad actor users.
  for (auto &output_contorl_actor : output_contorl_actors) {
    SchedulerHelper::AddControlArrow(copy_actor.get(), output_contorl_actor);
  }
}

void GraphScheduler::LinkDeviceTensorStoreForAutoMonadActor(const std::vector<AbstractActor *> &auto_monad_actors,
                                                            const GraphCompilerInfo &graph_compiler_info) {
  const size_t kNeedUpdateDeviceTensorStoreNum = 2;
  for (auto &auto_monad_actor : auto_monad_actors) {
    MS_EXCEPTION_IF_NULL(auto_monad_actor);
    for (auto &auto_monad_device_tensor_store : auto_monad_actor->auto_monad_device_tensor_stores_) {
      if (EnableInputOptimize() && auto_monad_device_tensor_store->isa<Parameter>()) {
        MS_LOG(DEBUG) << "Input optimize skip add copy actor for auto monad: " << auto_monad_actor->GetAID().Name()
                      << ", node: " << auto_monad_device_tensor_store->DebugString();
        continue;
      }

      auto kernel_tensors = DeviceTensorStore::GetInstance().Fetch(auto_monad_device_tensor_store.get());
      if (kernel_tensors.size() < kNeedUpdateDeviceTensorStoreNum) {
        continue;
      }

      // Create the copy actor.
      std::string name = "copy_from:" + auto_monad_actor->GetAID().Name() + kCopyActorNameSignFromStore +
                         auto_monad_device_tensor_store->fullname_with_scope();
      if (FetchActor(name) != nullptr) {
        continue;
      }
      AnfNode *from_kernel = nullptr;
      KernelGraphPtr from_graph = nullptr;
      if (auto_monad_actor->type() == KernelTransformType::kKernelActor) {
        auto kernel_actor = dynamic_cast<KernelActor *>(auto_monad_actor);
        MS_EXCEPTION_IF_NULL(kernel_actor);
        from_kernel = kernel_actor->kernel().get();
      } else if (auto_monad_actor->type() == KernelTransformType::kSuperKernelActor ||
                 auto_monad_actor->type() == KernelTransformType::kAnyTypeKernelActor) {
        auto super_kernel_actor = dynamic_cast<SuperKernelActor *>(auto_monad_actor);
        MS_EXCEPTION_IF_NULL(super_kernel_actor);
        from_graph = super_kernel_actor->graph();
      }
      auto copy_actor = std::make_shared<CopyActor>(name, from_kernel, from_graph, memory_manager_aid_);
      MS_EXCEPTION_IF_NULL(copy_actor);
      MS_LOG(DEBUG) << "Create copy actor:" << copy_actor->GetAID();
      (void)copy_actors_.emplace_back(copy_actor);
      InsertActor(copy_actor.get());

      // Set the member of the copy actor.
      (void)copy_actor->device_tensor_store_keys_.emplace_back(0, auto_monad_device_tensor_store);

      auto input_device_context = auto_monad_actor->device_contexts_[0];
      (void)copy_actor->device_contexts_.emplace_back(input_device_context);
      auto another_device_tensor = (kernel_tensors[0]->GetDeviceType() == input_device_context->GetDeviceType())
                                     ? kernel_tensors[1]->device_address()
                                     : kernel_tensors[0]->device_address();
      MS_EXCEPTION_IF_NULL(another_device_tensor);
      const auto &another_device_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(
        {another_device_tensor->GetDeviceType(), input_device_context->device_context_key().device_id_});
      MS_EXCEPTION_IF_NULL(another_device_context);
      (void)copy_actor->device_contexts_.emplace_back(another_device_context);

      MS_LOG(INFO) << "The auto monad actor:" << auto_monad_actor->GetAID().Name()
                   << " has control arrows number:" << auto_monad_actor->output_control_arrows_.size()
                   << ", add the copy actor for store:" << auto_monad_device_tensor_store->fullname_with_scope();
      CorrectControlArrowForAutoMonadActor(auto_monad_actor, copy_actor);
    }
  }
}

void GraphScheduler::PersistDeviceTensorForGraphOutput(const KernelGraphPtr &graph,
                                                       device::DeviceType device_type) const {
  MS_EXCEPTION_IF_NULL(graph);
  for (const auto &front_backend_pair : graph->front_node_to_graph_output_map()) {
    const auto &front_node_with_index = front_backend_pair.first;
    const auto &front_node = front_node_with_index.first;
    size_t front_index = front_node_with_index.second;
    const auto &backend_node_with_index = front_backend_pair.second;
    const auto &backend_node = backend_node_with_index.first;
    const auto &real_front_node = graph->GetFrontAnfByBackendAnf(backend_node_with_index.first);
    if (front_node == nullptr || backend_node == nullptr || real_front_node == nullptr || front_index != 0 ||
        front_node == real_front_node || !front_node->isa<ValueNode>() || !backend_node->isa<ValueNode>() ||
        !real_front_node->isa<ValueNode>()) {
      MS_LOG(DEBUG) << "Front value node addr:" << front_node.get()
                    << " and real_front_node addr:" << real_front_node.get()
                    << " has same backend addr:" << backend_node.get();
      continue;
    }
    MS_LOG(DEBUG) << "Front value node:" << front_node->DebugString()
                  << " full name:" << front_node->fullname_with_scope() << " node addr:" << front_node.get()
                  << " and  node:" << real_front_node->DebugString()
                  << " full name:" << real_front_node->fullname_with_scope() << " node addr:" << real_front_node.get()
                  << " has same backend node:" << backend_node->DebugString()
                  << " full name:" << backend_node->fullname_with_scope() << " node addr:" << backend_node.get();
    if (DeviceTensorStore::GetInstance().Fetch(front_node.get(), device_type) == nullptr) {
      MS_LOG(DEBUG) << "Fetch no device tensor store by:" << front_node->fullname_with_scope()
                    << ", type:" << device_type;
      if (!AnfAlgo::OutputAddrExist(backend_node, 0)) {
        MS_LOG(DEBUG) << "The kernel tensor is not exist: " << backend_node->DebugString();
        continue;
      }
      auto kernel_tensor = AnfAlgo::GetOutputKernelTensor(backend_node, 0, false);
      if (kernel_tensor == nullptr) {
        MS_LOG(DEBUG) << "The kernel tensor is null: " << backend_node->DebugString();
        continue;
      }
      SchedulerHelper::AddDeviceTensorStore(front_node, kernel_tensor);
      MS_LOG(DEBUG) << "Add kernel tensor:" << kernel_tensor->ToString()
                    << " for front node:" << front_node->DebugString() << " node addr:" << front_node.get();
    }
  }
}

void GraphScheduler::PersistDeviceTensor(const GraphCompilerInfo &graph_compiler_info) const {
  for (size_t i = 0; i < graph_compiler_info.graphs_.size(); ++i) {
    const auto &graph = graph_compiler_info.graphs_[i];
    const auto &device_context = graph_compiler_info.device_contexts_[i];
    MS_EXCEPTION_IF_NULL(graph);
    MS_LOG(DEBUG) << "PersistDeviceTensor graph:" << graph->ToString();
    for (auto &value_node : graph->graph_value_nodes()) {
      PersistDeviceTensorForValueNode(value_node, graph, device_context);
    }

    for (auto &input_node : graph->input_nodes()) {
      const auto &real_device_context = device::FetchRealDeviceContext(input_node, device_context);
      MS_EXCEPTION_IF_NULL(real_device_context);
      PersistDeviceTensorForParameter(input_node, graph, graph_compiler_info, real_device_context);
    }

    // The device tensor store used by backoff kernel need update with the real device context.
    for (auto &kernel : graph->execution_order()) {
      if (!AnfAlgo::IsKernelSelectBackoffOp(kernel)) {
        continue;
      }
      const auto &real_device_context = device::FetchRealDeviceContext(kernel, device_context);
      MS_EXCEPTION_IF_NULL(real_device_context);
      for (size_t j = 0; j < common::AnfAlgo::GetInputTensorNum(kernel); ++j) {
        const auto &input_node = common::AnfAlgo::GetInputNode(kernel, j);
        const auto &real_input_node = common::AnfAlgo::VisitKernelWithReturnType(input_node, 0, false).first;
        MS_EXCEPTION_IF_NULL(real_input_node);
        if (real_input_node->isa<ValueNode>()) {
          PersistDeviceTensorForValueNode(real_input_node, graph, real_device_context);
        }
        if (real_input_node->isa<Parameter>()) {
          PersistDeviceTensorForParameter(real_input_node, graph, graph_compiler_info, real_device_context);
        }
      }
    }
    // Two front value nodes with the same value will be optimized to the same value node in the backend CSE, and one of
    // Them will be lost in the backend-to-front map. If this node is the output of the graph and is used by the kernel
    // in The next kernel graph, the device address corresponding to this node will not be initialized, resulting in an
    // empty Address obtained by the kernel. Therefore, the address of such internal parameters in the kernel graph must
    // be Initialized to the address of the value node.
    PersistDeviceTensorForGraphOutput(graph, device_context->device_context_key().device_type_);
  }

  PersistDeviceTensorForRootGraphControlNode(graph_compiler_info);
}

void GraphScheduler::PersistDeviceTensorForValueNode(const AnfNodePtr &value_node, const KernelGraphPtr &graph,
                                                     const DeviceContext *device_context) const {
  MS_EXCEPTION_IF_NULL(value_node);
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(device_context);
  if (!AnfAlgo::OutputAddrExist(value_node, 0)) {
    MS_LOG(INFO) << "The device address is not exist: " << value_node->DebugString()
                 << " node addr:" << value_node.get();
    return;
  }
  auto old_kernel_tensor = AnfAlgo::GetOutputKernelTensor(value_node, 0, false);
  MS_EXCEPTION_IF_NULL(old_kernel_tensor);
  auto device_tensor = old_kernel_tensor->device_address();
  MS_EXCEPTION_IF_NULL(device_tensor);
  const auto &front_node = AnfAlgo::FetchFrontNodeByBackendNode(value_node, *graph);
  device_tensor->SetNodeIndex(value_node, 0);
  SchedulerHelper::AddDeviceTensorStore(front_node, old_kernel_tensor);

  // If the device tensor store of this device type is not exist, then create the new device tensor of this type.
  MS_LOG(DEBUG) << "Front node:" << front_node->DebugString() << ", front node addr:" << front_node.get()
                << ", old_kernel_tensor:" << old_kernel_tensor->ToString() << " graph:" << graph->ToString()
                << " value node:" << value_node->DebugString() << " value node addr:" << value_node.get();
  if (DeviceTensorStore::GetInstance().Fetch(front_node.get(), device_context->GetDeviceType()) == nullptr) {
    MS_LOG(INFO) << "Fetch no device tensor store by:" << front_node->fullname_with_scope()
                 << ", type:" << device_context->GetDeviceType() << " dtype:" << old_kernel_tensor->dtype_id()
                 << " current device address:" << device_tensor << " in value node:" << value_node->DebugString()
                 << " node addr:" << value_node.get();

    const auto &kernel_tensor = AnfAlgo::CreateOutputKernelTensorWithDeviceInfo(
      {value_node, 0}, nullptr, device_tensor->GetSize(), kernel::GetFormatFromEnumToStr(old_kernel_tensor->format()),
      old_kernel_tensor->dtype_id(), old_kernel_tensor->GetShapeVector(),
      device::GetDeviceNameByType(device_context->device_context_key().device_type_),
      device_context->device_context_key().device_id_);
    kernel_tensor->set_stream_id(device_tensor->stream_id());
    const auto &other_type_device_tensor = kernel_tensor->device_address();
    MS_EXCEPTION_IF_NULL(other_type_device_tensor);
    other_type_device_tensor->SetNodeIndex(value_node, 0);
    other_type_device_tensor->set_from_persistent_mem(true);
    MS_LOG(DEBUG) << "Create kernel tensor:" << kernel_tensor->ToString() << " node addr:" << value_node.get();
    SchedulerHelper::AddDeviceTensorStore(front_node, kernel_tensor);
  }
}

void GraphScheduler::PersistDeviceTensorForParameter(const AnfNodePtr &parameter, const KernelGraphPtr &graph,
                                                     const GraphCompilerInfo &graph_compiler_info,
                                                     const DeviceContext *device_context) const {
  MS_EXCEPTION_IF_NULL(parameter);
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(device_context);

  const auto &parser = graph_compiler_info.control_node_parser_;
  MS_EXCEPTION_IF_NULL(parser);
  AnfNodePtr front_node = nullptr;
  if (IsInternalParameter(parameter, graph)) {
    auto front_output_with_index = graph->GetFrontNodeByInternalParameter(parameter);
    front_node = front_output_with_index.first;
  } else if (IsPersistentDeviceTensor(parameter)) {
    front_node = AnfAlgo::FetchFrontNodeByBackendNode(parameter, *graph);
  }
  // The front node may be value node in the heterogeneous scene, needs to handle.
  if ((front_node == nullptr) ||
      (!front_node->isa<ValueNode>() && !parser->IsRootGraphPersistentDeviceTensor(front_node))) {
    return;
  }

  auto old_kernel_tensor = AnfAlgo::GetOutputKernelTensor(parameter, 0, false);
  MS_EXCEPTION_IF_NULL(old_kernel_tensor);
  auto device_tensor = old_kernel_tensor->device_address();
  MS_EXCEPTION_IF_NULL(device_tensor);
  if (IsPersistentDeviceTensor(parameter) || old_kernel_tensor->is_ptr_persisted()) {
    device_tensor->SetNodeIndex(parameter, 0);
    SchedulerHelper::AddDeviceTensorStore(front_node, old_kernel_tensor);
  }

  const auto &device_name = device::GetDeviceNameByType(device_context->device_context_key().device_type_);
  auto device_id = device_context->device_context_key().device_id_;
  if (EnableInputOptimize()) {
    auto graph_parameter_store = ParameterStore::GetInstance().GetGraphParameterStore();
    MS_EXCEPTION_IF_NULL(graph_parameter_store);
    if (graph_parameter_store->IsFrontNodeInStore(front_node.get())) {
      auto outer_index = graph_parameter_store->GetFrontNodeToIndex(front_node.get());
      if (graph_parameter_store->Fetch(outer_index, 0) == nullptr) {
        MS_LOG(INFO) << "Fetch no device tensor store by:" << front_node->fullname_with_scope()
                     << ", type:" << device_context->GetDeviceType();
        const auto &kernel_tensor = AnfAlgo::CreateOutputKernelTensorWithDeviceInfo(
          {parameter, 0}, nullptr, device_tensor->GetSize(),
          kernel::GetFormatFromEnumToStr(old_kernel_tensor->format()), old_kernel_tensor->dtype_id(),
          old_kernel_tensor->GetShapeVector(), device_name, device_id);
        kernel_tensor->set_stream_id(device_tensor->stream_id());
        const auto &other_type_device_tensor = kernel_tensor->device_address();
        other_type_device_tensor->SetNodeIndex(parameter, 0);
        other_type_device_tensor->set_from_persistent_mem(true);
        MS_LOG(DEBUG) << "Create kernel tensor:" << kernel_tensor->ToString();
        SchedulerHelper::AddDeviceTensorStore(front_node, kernel_tensor);
      }
      // If enable input optimize, the weight should push into parameter store, so return to avoid to push into
      // device tensor store.
      // The value node should push into device tensor store.
      return;
    }
  }

  // If the device tensor store of this device type is not exist, then create the new device tensor of this type.
  if (DeviceTensorStore::GetInstance().Fetch(front_node.get(), device_context->GetDeviceType()) == nullptr) {
    MS_LOG(INFO) << "Fetch no device tensor store by:" << front_node->fullname_with_scope()
                 << " debug string:" << front_node->DebugString() << ", type:" << device_context->GetDeviceType();

    const auto &kernel_tensor = AnfAlgo::CreateOutputKernelTensorWithDeviceInfo(
      {parameter, 0}, nullptr, device_tensor->GetSize(), kernel::GetFormatFromEnumToStr(old_kernel_tensor->format()),
      old_kernel_tensor->dtype_id(), old_kernel_tensor->GetShapeVector(), device_name, device_id);
    kernel_tensor->set_stream_id(device_tensor->stream_id());
    const auto &other_type_device_tensor = kernel_tensor->device_address();
    if (front_node->isa<ValueNode>()) {
      const auto &value_node = front_node->cast<ValueNodePtr>();
      MS_EXCEPTION_IF_NULL(value_node);
      if (value_node->value() != nullptr) {
        kernel_tensor->set_value(value_node->value());
      }
    }
    other_type_device_tensor->SetNodeIndex(parameter, 0);
    other_type_device_tensor->set_from_persistent_mem(true);
    MS_LOG(DEBUG) << "Create kernel tensor:" << kernel_tensor->ToString();
    SchedulerHelper::AddDeviceTensorStore(front_node, kernel_tensor);
  }
}

void GraphScheduler::PersistDeviceTensorForRootGraphControlNode(const GraphCompilerInfo &graph_compiler_info) const {
  const auto &parser = graph_compiler_info.control_node_parser_;
  if (parser == nullptr || (!parser->IsInited())) {
    return;
  }

  for (auto &root_graph_parameter : graph_compiler_info.origin_parameters_order_) {
    MS_EXCEPTION_IF_NULL(root_graph_parameter);
    if (!IsPersistentDeviceTensor(root_graph_parameter)) {
      continue;
    }

    // The graph parameter store has been done in the backend kernel graph corresponding to the root graph.
    if (EnableInputOptimize()) {
      auto graph_parameter_store = ParameterStore::GetInstance().GetGraphParameterStore();
      MS_EXCEPTION_IF_NULL(graph_parameter_store);
      auto outer_idx = graph_parameter_store->GetFrontNodeToIndex(root_graph_parameter.get());
      if (graph_parameter_store->Fetch(outer_idx, 0) != nullptr) {
        continue;
      }
    }

    // The device tensor store has been done in the backend kernel graph corresponding to the root graph.
    if (!DeviceTensorStore::GetInstance().Fetch(root_graph_parameter.get()).empty()) {
      continue;
    }

    // The different root graph parameters may correspond to parameter of same sub kernel graph when call the same sub
    // graph using the different root graph parameters. So can not use the device tensor of sub kernel graph parameter
    // directly and choose the first backend parameter in sub kernel graphs to create new device tensor to make sure
    // that the device tensor of root graph parameters are different.
    const auto &node_with_index_with_context =
      parser->FetchBackendParameterWithContextByFrontParameter({root_graph_parameter, 0});
    if (node_with_index_with_context.first.first == nullptr) {
      MS_LOG_WITH_NODE(INTERNAL_EXCEPTION, root_graph_parameter)
        << "#dmsg#Runtime error info:#dmsg#Can't find backend node for weight parameter:"
        << root_graph_parameter->DebugString();
    }
    const auto &backend_node = node_with_index_with_context.first.first;
    const auto &index = node_with_index_with_context.first.second;
    auto device_context = node_with_index_with_context.second;
    MS_EXCEPTION_IF_NULL(backend_node);
    MS_EXCEPTION_IF_NULL(device_context);
    const auto &parameter_device = AnfAlgo::GetParameterDeviceStr(root_graph_parameter);
    std::shared_ptr<AddressAllocator> allocator = nullptr;
    if (!parameter_device.empty() && parameter_device == kToCpu) {
      if (device_context->device_res_manager_->pin_mem_allocator() != nullptr) {
        allocator = device_context->device_res_manager_->pin_mem_allocator();
        MS_LOG(DEBUG) << "Use PinMemoryAllocator for offloaded parameter. Parameter: "
                      << root_graph_parameter->fullname_with_scope();
      }
      device_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(
        {device::GetDeviceTypeByName(parameter_device), device_context->device_context_key().device_id_});
      MS_LOG(INFO) << "Offloaded parameter:" << root_graph_parameter->fullname_with_scope();
    }
    if (index != 0) {
      MS_LOG_WITH_NODE(INTERNAL_EXCEPTION, backend_node)
        << "#dmsg#Runtime error info:#dmsg#Device tensor store does not support tuple type, node:"
        << backend_node->DebugString() << " index:" << index;
    }
    auto sub_kernel_tensor = AnfAlgo::GetOutputKernelTensor(backend_node, index, false);
    MS_EXCEPTION_IF_NULL(sub_kernel_tensor);
    auto sub_device_tensor = sub_kernel_tensor->device_address();
    MS_EXCEPTION_IF_NULL(sub_device_tensor);

    const auto &kernel_tensor = AnfAlgo::CreateOutputKernelTensorWithDeviceInfo(
      {backend_node, index}, nullptr, sub_device_tensor->GetSize(),
      kernel::GetFormatFromEnumToStr(sub_kernel_tensor->format()), sub_kernel_tensor->dtype_id(),
      sub_kernel_tensor->GetShapeVector(),
      device::GetDeviceNameByType(device_context->device_context_key().device_type_),
      device_context->device_context_key().device_id_);
    MS_EXCEPTION_IF_NULL(device_context->device_res_manager_);
    kernel_tensor->set_stream_id(AnfAlgo::GetStreamId(backend_node));
    const auto &new_device_tensor = kernel_tensor->device_address();
    MS_EXCEPTION_IF_NULL(new_device_tensor);
    new_device_tensor->SetNodeIndex(backend_node, index);
    kernel_tensor->set_is_ptr_persisted(sub_kernel_tensor->is_ptr_persisted());
    new_device_tensor->set_from_persistent_mem(true);
    new_device_tensor->set_allocator(allocator);
    kernel_tensor->set_user_data(sub_kernel_tensor->user_data());

    SchedulerHelper::AddDeviceTensorStore(root_graph_parameter, kernel_tensor);
    MS_LOG(INFO) << "Add device tensor store by root graph parameter:" << root_graph_parameter->fullname_with_scope()
                 << ", backend node:" << backend_node->DebugString() << ", type:" << device_context->GetDeviceType()
                 << " device_tensor:" << new_device_tensor << ", kernel tensor: " << kernel_tensor;
  }
}

void GraphScheduler::DumpActor(const ActorSet *actor_set, const GraphCompilerInfo &graph_compiler_info) const {
  MS_EXCEPTION_IF_NULL(actor_set);
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  if (!context->CanDump(kIntroductory)) {
    return;
  }

  // Get the saved actor set name.
  std::string strategy = kGraphExecutionStrategyStr.at(graph_compiler_info.strategy_);
  if (execution_order_running_) {
    strategy = "pipeline_with_excution_order";
  }
  std::string save_name = "actor_set/0_actor_set_" + strategy + "_" + actor_set->name_;
  std::string path_name = GetSaveGraphsPathName(save_name + ".ir");
  auto realpath = Common::CreatePrefixPath(path_name);
  if (!realpath.has_value()) {
    MS_LOG(ERROR) << "Get real path failed, path: " << path_name;
    return;
  }

  ChangeFileMode(realpath.value(), S_IWUSR);
  std::ofstream ofs(realpath.value());
  if (!ofs.is_open()) {
    MS_LOG(ERROR) << "Open file [" << realpath.value() << "] failed!";
    return;
  }
  DumpDeviceTensorStore(graph_compiler_info, ofs);
  SchedulerHelper::DumpActorSet(actor_set, ofs);
  ChangeFileMode(realpath.value(), S_IRUSR);
}

void GraphScheduler::DumpFinalActor(const ActorSet *actor_set, const GraphCompilerInfo &graph_compiler_info) {
  MS_EXCEPTION_IF_NULL(actor_set);
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  if (!context->CanDump(kIntroductory) || common::GetEnv("MS_DEV_DISABLE_FINAL_ACTOR_IR") == "1") {
    return;
  }

  std::string save_name = "actor_set/final_actor_set_" + actor_set->name_;
  std::string path_name = GetSaveGraphsPathName(save_name + ".ir");
  auto realpath = Common::CreatePrefixPath(path_name);
  if (!realpath.has_value()) {
    MS_LOG(ERROR) << "Get real path failed, path: " << path_name;
    return;
  }
  if (actor_set->output_actor_ == nullptr) {
    return;
  }
  ChangeFileMode(realpath.value(), S_IWUSR);
  std::ofstream ofs(realpath.value());
  if (!ofs.is_open()) {
    MS_LOG(ERROR) << "Open file [" << realpath.value() << "] failed!";
    return;
  }
  SchedulerHelper::DumpFormatActorSet(actor_set, ofs);
  control_node_scheduler_.DumpFormatControlActorSet(actor_set, graph_compiler_info, graph_output_to_actor_, ofs);
  ChangeFileMode(realpath.value(), S_IRUSR);
}

void DumpParameterTensor(const GraphCompilerInfo &graph_compiler_info, const KernelGraphPtr &graph,
                         std::ofstream &ofs) {
  if (EnableInputOptimize()) {
    const auto &root_parameters = graph_compiler_info.origin_parameters_order_;
    std::vector<bool> dump_already(root_parameters.size(), false);
    auto graph_parameter_store = ParameterStore::GetInstance().GetGraphParameterStore();
    MS_EXCEPTION_IF_NULL(graph_parameter_store);
    for (auto &input_node : graph->input_nodes()) {
      MS_EXCEPTION_IF_NULL(input_node);
      const auto &front_node = AnfAlgo::FetchFrontNodeByBackendNode(input_node, *graph);
      if (front_node == nullptr ||
          find(root_parameters.begin(), root_parameters.end(), front_node) == root_parameters.end()) {
        continue;
      }

      auto index = graph_parameter_store->GetFrontNodeToIndex(front_node.get());
      if (index >= dump_already.size()) {
        MS_LOG(INFO) << "Dump index is larger than the size of parameter device tensor.";
        continue;
      }
      if (dump_already[index]) {
        continue;
      }
      dump_already[index] = true;

      auto unfold_output_num = common::AnfAlgo::GetOutputNumByAbstract(front_node->abstract());
      for (size_t i = 0; i < unfold_output_num; ++i) {
        const auto &kernel_tensor = graph_parameter_store->Fetch(index, i);
        MS_EXCEPTION_IF_NULL(front_node);
        ofs << "\t\tgraph parameter front node:" << front_node->DebugString() << "\tunfold index:" << i << "\n"
            << "\n";

        MS_EXCEPTION_IF_NULL(kernel_tensor);
        auto device_tensor = kernel_tensor->device_address().get();
        MS_EXCEPTION_IF_NULL(device_tensor);
        ofs << "\t\t\tdevice tensor value:" << device_tensor << "\tptr:" << device_tensor->GetPtr()
            << "\tsize:" << device_tensor->GetSize() << "\tnew_ref_count:" << kernel_tensor->new_ref_count()
            << "\tflag:" << kernel_tensor->flag() << "\tdevice_type:" << device_tensor->GetDeviceType()
            << "\tis_ptr_persisted:" << kernel_tensor->is_ptr_persisted() << "\n ";
      }
    }
    return;
  }

  for (auto &input_node : graph->input_nodes()) {
    MS_EXCEPTION_IF_NULL(input_node);
    if (!IsPersistentDeviceTensor(input_node)) {
      continue;
    }
    const auto &front_node = AnfAlgo::FetchFrontNodeByBackendNode(input_node, *graph);
    const auto &root_parameters = graph_compiler_info.origin_parameters_order_;
    if (front_node == nullptr ||
        find(root_parameters.begin(), root_parameters.end(), front_node) == root_parameters.end()) {
      continue;
    }
    const auto &kernel_tensors = DeviceTensorStore::GetInstance().Fetch(front_node.get());
    MS_EXCEPTION_IF_NULL(front_node);
    ofs << "\t\tdevice tensor key:" << front_node->fullname_with_scope() << "\tvalue size:" << kernel_tensors.size()
        << "\n";
    for (const auto &kernel_tensor : kernel_tensors) {
      MS_EXCEPTION_IF_NULL(kernel_tensor);
      auto device_tensor = kernel_tensor->device_address().get();
      MS_EXCEPTION_IF_NULL(device_tensor);
      ofs << "\t\t\tdevice tensor value:" << device_tensor << "\tptr:" << device_tensor->GetPtr()
          << "\tsize:" << device_tensor->GetSize() << "\tflag:" << kernel_tensor->flag()
          << "\tdevice_type:" << device_tensor->GetDeviceType()
          << "\tis_ptr_persisted:" << kernel_tensor->is_ptr_persisted() << "\n ";
    }
  }
}

void GraphScheduler::DumpDeviceTensorStore(const GraphCompilerInfo &graph_compiler_info, std::ofstream &ofs) const {
  ofs << "[Device tensor stores]\n";

  for (const auto &graph : graph_compiler_info.graphs_) {
    MS_EXCEPTION_IF_NULL(graph);
    ofs << "\tgraph_id:" << graph->graph_id() << "\tis_graph_run_mode:" << graph->is_graph_run_mode()
        << "\tis_loop_count_sink:" << graph->is_loop_count_sink()
        << "\texecution_strategy:" << graph_compiler_info.strategy_ << "\n";

    for (auto &value_node : graph->graph_value_nodes()) {
      MS_EXCEPTION_IF_NULL(value_node);
      if (!AnfAlgo::OutputAddrExist(value_node, 0)) {
        continue;
      }
      const auto &front_node = AnfAlgo::FetchFrontNodeByBackendNode(value_node, *graph);
      MS_EXCEPTION_IF_NULL(front_node);
      const auto &kernel_tensors = DeviceTensorStore::GetInstance().Fetch(front_node.get());
      ofs << "\t\tdevice tensor key:" << front_node->fullname_with_scope()
          << "\tbackend node name:" << value_node->fullname_with_scope() << "\tvalue size:" << kernel_tensors.size()
          << "\n";
      for (const auto &kernel_tensor : kernel_tensors) {
        MS_EXCEPTION_IF_NULL(kernel_tensor);
        auto device_tensor = kernel_tensor->device_address().get();
        MS_EXCEPTION_IF_NULL(device_tensor);
        ofs << "\t\t\tdevice tensor value:" << device_tensor << "\tptr:" << device_tensor->GetPtr()
            << "\tsize:" << device_tensor->GetSize() << "\tstream id:" << device_tensor->stream_id()
            << "\tflag:" << kernel_tensor->flag() << "\tdevice_type:" << device_tensor->GetDeviceType()
            << "\tis_ptr_persisted:" << kernel_tensor->is_ptr_persisted() << "\n ";
      }
    }

    ofs << "\n";

    for (auto &backend_front_map : graph->backend_front_anf_map()) {
      MS_EXCEPTION_IF_NULL(backend_front_map.first);
      MS_EXCEPTION_IF_NULL(backend_front_map.second);
      MS_LOG(DEBUG) << "Graph: " << graph->graph_id()
                    << ", backend node: " << backend_front_map.first->fullname_with_scope()
                    << ", front node: " << backend_front_map.second->DebugString();
    }
  }

  if (EnableInputOptimize()) {
    ofs << "[Graph parameter stores]\n";
  }
  for (const auto &graph : graph_compiler_info.graphs_) {
    MS_EXCEPTION_IF_NULL(graph);
    ofs << "\tgraph_id:" << graph->graph_id() << "\tis_graph_run_mode:" << graph->is_graph_run_mode()
        << "\tis_loop_count_sink:" << graph->is_loop_count_sink()
        << "\texecution_strategy:" << graph_compiler_info.strategy_ << "\n";
    DumpParameterTensor(graph_compiler_info, graph, ofs);
  }
}

void GraphScheduler::BindNumaNode() {
  auto numa_enable = common::GetEnv(kNumaEnableEnv);
  auto numa_enable2 = common::GetEnv(kNumaEnableEnv2);
  if ((numa_enable.empty() || numa_enable != "1") && (numa_enable2.empty() || numa_enable2 != "1")) {
    return;
  }

#if !defined(_WIN32) && !defined(_WIN64) && !defined(__APPLE__) && !defined(ENABLE_ANDROID)
  if (!numa_cpus_.empty()) {
    MS_LOG(WARNING) << "Have already been bound numa node to " << numa_cpus_ << ", will not bind again.";
    return;
  }
  uint32_t rank_id = CollectiveManager::instance()->local_rank_id();
  MS_LOG(INFO) << "Bind numa node for rank " << rank_id;
  if (numa_handle_ == nullptr) {
    numa_handle_ = GetNumaAdapterHandle();
    if (numa_handle_ == nullptr) {
      MS_LOG(WARNING) << "Load numa library failed.";
      return;
    }
  }
  (void)LoadNumaCpuInfo(numa_handle_.get(), rank_id, &numa_cpus_);
  auto ret = NumaBind(numa_handle_.get(), rank_id);
  if (ret != StatusCode::kSuccess) {
    MS_LOG(WARNING) << "Bind numa node failed, ret = " << ret.GetErrDescription();
    return;
  }
  MS_LOG(INFO) << "Numa bind memory and cpu successful.";
#endif
}
}  // namespace runtime
}  // namespace mindspore
