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

#include <algorithm>
#include <queue>
#include <map>
#include <memory>
#include <utility>
#include <set>
#include <vector>
#include <string>
#include "primitive/sequence_ops.h"
#include "primitive/framework_ops.h"
#include "backend/ge_backend/runtime/scheduler_helper.h"
#include "backend/ge_backend/runtime/actor/memory_manager_actor.h"
#include "backend/ge_backend/runtime/actor/debug_actor.h"
#include "backend/ge_backend/runtime/actor/profiler_actor.h"
#include "backend/ge_backend/runtime/actor/recorder_actor.h"
#include "backend/ge_backend/runtime/graph_scheduler.h"
#include "include/runtime/utils/runtime_conf/runtime_conf.h"
#include "include/runtime/utils/runtime_conf/thread_bind_core.h"
#include "tools/profiler/profiler.h"
#include "actor/actormgr.h"
#include "async/async.h"
#include "device_address/device_address.h"
#include "include/backend/common/kernel_graph/anf_runtime_algorithm.h"
#include "include/utils/anfalgo.h"
#include "include/utils/parallel_context.h"
#include "utils/config_manager.h"
#include "utils/log_adapter.h"
#include "utils/convert_utils.h"
#include "utils/ms_context.h"
#include "utils/profile.h"
#include "include/runtime/hardware_abstract/kernel_base/common_utils.h"
#ifndef ENABLE_SECURITY
#include "backend/ge_backend/dump/hook_debugger.h"
#endif
#include "tools/profiler/profiling.h"
#include "include/utils/common.h"
#include "abstract/ops/primitive_infer_map.h"
#include "utils/file_utils.h"

#if !defined(_WIN32) && !defined(_WIN64) && !defined(__APPLE__)
#include "utils/numa_interface.h"
#endif
#include "utils/ms_exception.h"

namespace mindspore {
namespace ge_backend {
namespace runtime {
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
  return !graphs[0]->has_flag(kFlagPyNativeRunInGraph);
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
      (graphs.size() == 1 && graphs[0]->is_loop_count_sink()) || graphs[0]->has_flag(kFlagPyNativeRunInGraph)) {
    loop_count = 1;
  }

  return loop_count;
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
      actor_manager->Terminate(base_actor->GetAID());
    }
  }

  // Clear device tensor and device tensor store.
  for (auto &graph : graphs) {
    ClearNodeInfo(graph);
  }

  if (parser != nullptr && parser->IsInited()) {
    const auto &front_value_nodes = parser->front_value_nodes();
    for (const auto &front_value_node : front_value_nodes) {
      const auto &node = front_value_node.first;
      size_t index = front_value_node.second;
      if (AnfAlgo::OutputAddrExist(node, index)) {
        AnfAlgo::SetOutputAddr(nullptr, nullptr, index, node);
      }
    }
  }

  // Clear the member of DeviceTensorStore.
  for (auto &root_graph_parameter : root_graph_parameters) {
    DeviceTensorStore::GetInstance().Remove(root_graph_parameter.get());
  }

  // Clear global maps of actor info.
  (void)actors_.erase(actor_info);
}

void GraphScheduler::Clear() {
  // Terminate all actors.
  auto actor_manager = ActorMgr::GetActorMgrRef();
  MS_EXCEPTION_IF_NULL(actor_manager);
  actor_manager->Finalize();

  // Clear the member of DeviceTensorStore.
  DeviceTensorStore::GetInstance().Clear();

  // Clear global maps.
  actors_.clear();
  ClearAllActors();
}

void GraphScheduler::ClearActorData(const ActorSet *actor_set) {
  MS_EXCEPTION_IF_NULL(actor_set);

  // Clear the output tensors of output actor.
  if (actor_set->output_actor_ != nullptr) {
    actor_set->output_actor_->outputs_.clear();
    actor_set->output_actor_->outputs_.resize(actor_set->output_actor_->outputs_num_);
  }

  for (auto &data_source_actor : actor_set->data_source_actors_) {
    MS_EXCEPTION_IF_NULL(data_source_actor);
    data_source_actor->ReleaseData();
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

  (void)kKernelTypeToLinkFunc.emplace(KernelTransformType::kHostDataSourceActor,
                                      &GraphScheduler::LinkDataArrowForHostDSActor);
  (void)kKernelTypeToLinkFunc.emplace(KernelTransformType::kSuperKernelActor,
                                      &GraphScheduler::LinkDataArrowForBaseActor);
  (void)kKernelTypeToLinkFunc.emplace(KernelTransformType::kDeviceTensorStore,
                                      &GraphScheduler::LinkDataArrowForDeviceTensorStore);
  (void)kKernelTypeToLinkFunc.emplace(KernelTransformType::kInternalParameter,
                                      &GraphScheduler::LinkDataArrowForInternalParameter);

  // Create the thread pool of actor runtime and Set the OMP_NUM_THREADS env.
  size_t actor_thread_num = 0;
  size_t actor_and_kernel_thread_num = 0;
  mindspore::runtime::ComputeThreadNums(&actor_thread_num, &actor_and_kernel_thread_num);
  auto actor_manager = ActorMgr::GetActorMgrRef();
  MS_EXCEPTION_IF_NULL(actor_manager);
  size_t actor_queue_size = 81920;

  auto ret = actor_manager->Initialize(true, actor_thread_num, actor_and_kernel_thread_num, actor_queue_size);
  if (ret != MINDRT_OK) {
    MS_LOG(INTERNAL_EXCEPTION) << "#dmsg#Runtime error info:#dmsg#Actor manager init failed.";
  }
  default_actor_thread_num_ = actor_thread_num;
  common::SetOMPThreadNum();
  MS_LOG(INFO) << "The actor thread number: " << actor_thread_num
               << ", the kernel thread number: " << (actor_and_kernel_thread_num - actor_thread_num);

  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);

  BuildAndScheduleGlobalActor();
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
  bool debugger_actor_need = false;
#ifndef ENABLE_SECURITY
  debugger_actor_need = dump::HookDebugger::GetInstance().IsHookerEnabled();
#endif
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

  (void)profiler::CollectHostInfo(kModelNameRuntime, kEventCompileGraph, kStageLink, start_time_1,
                                  profiler::GetClockSyscnt(), 1);
  DumpActor(actor_set.get(), graph_compiler_info);
  DumpFinalActor(actor_set.get(), graph_compiler_info);
  MS_LOG(INFO) << "Graph(" << graph_compiler_info.name_ << ") transforms actor end.";

  for (const auto &graph : graph_compiler_info.graphs_) {
    MS_EXCEPTION_IF_NULL(graph);
    if (graph->is_dynamic_shape()) {
      actor_set->has_dynamic_shape_ = true;
      break;
    }
  }

  actor_set->all_actors_ = SchedulerHelper::CollectActors(actor_set.get());
  (void)profiler::CollectHostInfo(kModelNameRuntime, kEventCompileGraph, kStageGraphTransform, start_time_0,
                                  profiler::GetClockSyscnt(), 1);
  return actor_set.get();
}

void GraphScheduler::Schedule(const ActorSet *actor_set) {
  MS_EXCEPTION_IF_NULL(actor_set);
  const auto &actors = actor_set->all_actors_;
  // Schedule actors.
  auto actor_manager = ActorMgr::GetActorMgrRef();
  MS_EXCEPTION_IF_NULL(actor_manager);

  for (auto actor : actors) {
    MS_EXCEPTION_IF_NULL(actor);
    (void)actor_manager->Spawn(actor);
  }
}

void GraphScheduler::Run(ActorSet *const actor_set, const std::vector<std::vector<TensorPtr>> &input_tensors,
                         const VectorRef &args, GraphExecutionStrategy strategy) {
  MS_EXCEPTION_IF_NULL(actor_set);
  MS_EXCEPTION_IF_NULL(actor_set->data_prepare_actor_);

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
  OpContext<KernelTensor> op_context;
  std::vector<Promise<int>> result(1);
  op_context.sequential_num_ = RandInt::Instance().Get();
  op_context.results_ = &result;

  // Trigger data prepare actor running.
  MS_EXCEPTION_IF_NULL(ActorMgr::GetActorMgrRef());
  auto thread_pool = ActorMgr::GetActorMgrRef()->GetActorThreadPool();
  MS_EXCEPTION_IF_NULL(thread_pool);
  if (actor_set->is_multi_thread_execution_) {
    thread_pool->SetSpinCountMaxValue();
  }
  ActorDispatcher::set_is_multi_thread_execution(actor_set->is_multi_thread_execution_);
  ActorDispatcher::SendSync(actor_set->data_prepare_actor_->GetAID(), &DataPrepareActor::PrepareData, input_tensors,
                            args, &op_context, GraphExecutionStrategy::kPipeline);

  // Get the run result.
  auto result_future = result[0].GetFuture();
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

    // May set exception in the wait time, need throw the exception to avoid affecting the next execution.
    MsException::Instance().CheckException();
    MS_LOG(EXCEPTION) << op_context.error_info_;
  }

  MsException::Instance().CheckException();
  (void)SkipOrResetCopyAction(true);
  (void)SkipOrResetSyncAction(true);
}

void GraphScheduler::ChildAfterFork() {
  MS_LOG(DEBUG) << "GraphScheduler reinitialize after fork.";
  auto actor_manager = ActorMgr::GetActorMgrRef();
  MS_EXCEPTION_IF_NULL(actor_manager);
  MS_LOG(DEBUG) << "GraphScheduler reinitialize after fork done.";
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
  (void)actors_.emplace(actor_set->name_, actor_set);

  auto host_queue = std::make_shared<HostTensorQueue>();
  actor_set->data_source_actors_ = BuildDataSourceActor(graph_compiler_info, host_queue);
  actor_set->data_prepare_actor_ =
    BuildDataPrepareActor(graph_compiler_info, actor_set->data_source_actors_, host_queue);

  actor_set->super_kernel_actors_ = BuildSuperKernelActor(graph_compiler_info);
  actor_set->loop_count_actor_ = BuildLoopCountActor(graph_compiler_info);
  actor_set->output_actor_ = BuildOutputActor(graph_compiler_info);
  actor_set->control_actors_ = control_node_scheduler_.Build(graph_compiler_info, memory_manager_aid_);

  return actor_set;
}

void GraphScheduler::CacheGraphOutputToActor(const GraphCompilerInfo &graph_compiler_info) {
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
                   << " with index:" << output_with_index.second << " to actor:" << output_actor_name
                   << ", from front node:" << origin_output_with_index.first->fullname_with_scope()
                   << " debug string:" << origin_output_with_index.first->DebugString()
                   << " with index:" << origin_output_with_index.second;
    }
  }
}

void GraphScheduler::Link(ActorSet *actor_set, const GraphCompilerInfo &graph_compiler_info) {
  MS_EXCEPTION_IF_NULL(actor_set);
  std::vector<AbstractActor *> auto_monad_actors;
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

    if (graph->is_graph_run_mode()) {
      PROF_START(GraphSchedulerLinkSinkMode);
      LinkDataArrowInSinkMode(graph, graph_compiler_info, &auto_monad_actors);
      PROF_END(GraphSchedulerLinkSinkMode);
    }
  }

  LinkGlobalControlArrow(actor_set, auto_monad_actors, graph_compiler_info);
  LinkOutputResultArrowForOutputActor(actor_set->output_actor_.get(), graph_compiler_info);

  // Link the arrow in the control flow scene.
  if (graph_compiler_info.strategy_ == GraphExecutionStrategy::kPipeline &&
      graph_compiler_info.control_node_parser_ != nullptr && graph_compiler_info.control_node_parser_->IsInited()) {
    control_node_scheduler_.Link(actor_set, graph_compiler_info);
  }
}

std::vector<DataSourceActorPtr> GraphScheduler::BuildDataSourceActor(const GraphCompilerInfo &graph_compiler_info,
                                                                     const HostTensorQueuePtr &host_queue) {
  std::vector<DataSourceActorPtr> data_source_actors;
  HostQueueDSActorPtr host_queue_ds_actor = nullptr;
  size_t data_node_position = 0;
  std::map<KernelWithIndex, size_t> front_node_position_temp_map;

  for (size_t i = 0; i < graph_compiler_info.graphs_.size(); ++i) {
    const auto &graph = graph_compiler_info.graphs_[i];
    MS_EXCEPTION_IF_NULL(graph);
    // Build host queue data source actor.
    const std::vector<AnfNodePtr> &input_nodes = graph->input_nodes();
    const auto &root_parameters = graph_compiler_info.origin_parameters_order_;
    for (size_t j = 0; j < input_nodes.size(); j++) {
      const auto &input_node = input_nodes[j];
      MS_EXCEPTION_IF_NULL(input_node);

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
          (void)host_queue_ds_actor->data_node_position_map_.emplace(KernelWithIndex(input_node, 0), front_node_index);
          continue;
        }
        (void)host_queue_ds_actor->data_node_with_indexs_.emplace_back(input_node, 0);
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
    MS_EXCEPTION_IF_NULL(graph);
    if (!graph->is_graph_run_mode()) {
      continue;
    }

    if (graph->execution_order().empty()) {
      MS_LOG(INFO) << "The graph " << graph->graph_id() << " is an empty graph and skips building.";
      continue;
    }
    graph->CacheRootWeight(root_weights);
    auto actor_name = graph->ToString() + kSuperKernelActorNameSuffix;
    auto super_kernel_actor =
      std::make_shared<SuperKernelActor>(actor_name, graph, graph_compiler_info.graph_phase_, memory_manager_aid_,
                                         debug_aid_, nullptr, graph_compiler_info.graph_executor_);
    MS_EXCEPTION_IF_NULL(super_kernel_actor);
    InsertActor(super_kernel_actor.get());
    (void)super_kernel_actors.emplace_back(super_kernel_actor);
  }
  return super_kernel_actors;
}

LoopCountActorPtr GraphScheduler::BuildLoopCountActor(const GraphCompilerInfo &graph_compiler_info) {
  auto loop_count = GetLoopCount(graph_compiler_info);
  auto sink_size = ConfigManager::GetInstance().iter_num();
  auto actor_name = graph_compiler_info.name_ + kLoopCountActorNameSuffix;
  auto is_need_sync_stream = GetNeedSyncStream(graph_compiler_info);
  auto loop_count_actor = std::make_shared<LoopCountActor>(
    actor_name, graph_compiler_info.name_, loop_count, sink_size, memory_manager_aid_, debug_aid_, recorder_aid_,
    profiler_aid_, graph_compiler_info.strategy_, is_need_sync_stream);
  MS_LOG(INFO) << "Create loop count actor: " << actor_name;
  MS_EXCEPTION_IF_NULL(loop_count_actor);

  InsertActor(loop_count_actor.get());
  return loop_count_actor;
}

OutputActorPtr GraphScheduler::BuildOutputActor(const GraphCompilerInfo &graph_compiler_info) const {
  if (graph_compiler_info.strategy_ == GraphExecutionStrategy::kStep) {
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

void GraphScheduler::LinkDataArrowInSinkMode(const KernelGraphPtr &graph, const GraphCompilerInfo &graph_compiler_info,
                                             std::vector<AbstractActor *> *const auto_monad_actors) {
  MS_EXCEPTION_IF_NULL(graph);
  auto to_actor_name = graph->ToString() + kSuperKernelActorNameSuffix;
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
    }
    // No data arrow for monad input.
    if (HasAbstractMonad(input_node)) {
      continue;
    }

    UpdateRefCount(input_node, 0, true);

    KernelWithIndex from_kernel_with_output_idx = std::make_pair(input_node, 0);
    KernelWithIndex to_kernel_with_input_idx = std::make_pair(input_node, node_index);
    // The gather of linking data arrows of kernel by the different from kernel type.
    LinkDataArrow(to_actor, graph_compiler_info, graph, from_kernel_with_output_idx, to_kernel_with_input_idx);
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
  if (IsPersistentDeviceTensor(front_output_node)) {
    kernel_type = KernelTransformType::kDeviceTensorStore;
  } else {
    // front node ---> actor.
    if (graph_output_to_actor_.count(front_output_with_index) == 0) {
      MS_LOG_WITH_NODE(INTERNAL_EXCEPTION, front_output_node)
        << "#dmsg#Runtime error info:#dmsg#Can't find actor by front node:"
        << common::AnfAlgo::GetNodeDebugString(front_output_node)
        << ", internal parameter:" << common::AnfAlgo::GetNodeDebugString(internal_parameter);
    }
    auto actor_pair = graph_output_to_actor_[front_output_with_index];
    MS_EXCEPTION_IF_NULL(actor_pair.first);
    MS_EXCEPTION_IF_NULL(actor_pair.second.first);
    MS_LOG(INFO) << "Graph " << graph->graph_id() << " internal parameter:" << internal_parameter->DebugString()
                 << ", corresponding front node:" << front_output_node->fullname_with_scope()
                 << " with index:" << front_output_with_index.second
                 << ", from actor:" << actor_pair.first->GetAID().Name()
                 << " node:" << actor_pair.second.first->fullname_with_scope()
                 << " with index:" << actor_pair.second.second << ", to actor:" << to_actor->GetAID().Name()
                 << " with index:" << to_kernel_with_input_idx.second;
    real_from_actor = actor_pair.first;
    // The data arrow need skip the monad node.
    real_from_kernel_with_output_idx = common::AnfAlgo::FetchRealNodeSkipMonadControl(actor_pair.second);
    kernel_type = actor_pair.first->type_;
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

  SchedulerHelper::AddDataArrow(from_actor, to_actor, from_output_index, to_input_index, from_kernel);
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

void GraphScheduler::LinkGlobalControlArrow(ActorSet *const actor_set,
                                            const std::vector<AbstractActor *> &auto_monad_actors,
                                            const GraphCompilerInfo &graph_compiler_info) {
  MS_EXCEPTION_IF_NULL(actor_set);

  // BuildNoInputKernelActor depends on whether kernel actors have input, so must be behind the link of kernel actors.
  actor_set->no_input_kernel_actors_ = BuildNoInputKernelActor(actor_set, graph_compiler_info.strategy_);

  // Link the control arrows of data prepare actor, which depends on the no input kernel actors.
  LinkControlArrowForDataPrepareActor(actor_set->data_prepare_actor_.get(), actor_set,
                                      graph_compiler_info.control_node_parser_);

  LinkControlArrowForLoopCountActor(actor_set->loop_count_actor_.get(), actor_set,
                                    graph_compiler_info.control_node_parser_);

  LinkControlArrowForOutputActor(actor_set->output_actor_.get(), actor_set);
}

void GraphScheduler::LinkControlArrowForDataPrepareActor(DataPrepareActor *data_prepare_actor,
                                                         const ActorSet *actor_set,
                                                         const ControlNodeParserPtr &parser) const {
  MS_EXCEPTION_IF_NULL(data_prepare_actor);
  MS_EXCEPTION_IF_NULL(actor_set);
  MS_EXCEPTION_IF_NULL(parser);

  // Data prepare actor --> data source actor.
  for (auto &data_source_actor : actor_set->data_source_actors_) {
    MS_EXCEPTION_IF_NULL(data_source_actor);
    SchedulerHelper::AddControlArrow(data_prepare_actor, data_source_actor.get());
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

  for (auto &data_actor : actor_set->data_source_actors_) {
    MS_EXCEPTION_IF_NULL(data_actor);
    if (is_no_output_actor(data_actor)) {
      (void)no_output_actors.emplace_back(data_actor.get());
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
        UpdateRefCount(real_from_kernel, real_from_index, true);
        real_from_kernel = host_queue_ds_actor->FetchNode(position).first;
      }
    }

    for (auto &output_position : origin_output_order.second) {
      SchedulerHelper::AddResultArrow(from_actor, to_actor, real_from_kernel, real_from_index, output_position);
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

void GraphScheduler::PersistDeviceTensor(const GraphCompilerInfo &graph_compiler_info) const {
  for (size_t i = 0; i < graph_compiler_info.graphs_.size(); ++i) {
    const auto &graph = graph_compiler_info.graphs_[i];
    for (auto &value_node : graph->graph_value_nodes()) {
      PersistDeviceTensorForValueNode(value_node, graph);
    }

    for (auto &input_node : graph->input_nodes()) {
      PersistDeviceTensorForParameter(input_node, graph, graph_compiler_info);
    }

    // The device tensor store used by backoff kernel need update with the real device context.
    for (auto &kernel : graph->execution_order()) {
      if (!AnfAlgo::IsKernelSelectBackoffOp(kernel)) {
        continue;
      }
      for (size_t j = 0; j < common::AnfAlgo::GetInputTensorNum(kernel); ++j) {
        const auto &input_node = common::AnfAlgo::GetInputNode(kernel, j);
        const auto &real_input_node = common::AnfAlgo::VisitKernelWithReturnType(input_node, 0, false).first;
        MS_EXCEPTION_IF_NULL(real_input_node);
        if (real_input_node->isa<ValueNode>()) {
          PersistDeviceTensorForValueNode(real_input_node, graph);
        }
        if (real_input_node->isa<Parameter>()) {
          PersistDeviceTensorForParameter(real_input_node, graph, graph_compiler_info);
        }
      }
    }
  }

  PersistDeviceTensorForRootGraphControlNode(graph_compiler_info);
}

void GraphScheduler::PersistDeviceTensorForValueNode(const AnfNodePtr &value_node, const KernelGraphPtr &graph) const {
  MS_EXCEPTION_IF_NULL(value_node);
  MS_EXCEPTION_IF_NULL(graph);
  if (!AnfAlgo::OutputAddrExist(value_node, 0)) {
    MS_LOG(INFO) << "The device address is not exist: " << value_node->ToString();
    return;
  }
  auto old_kernel_tensor = AnfAlgo::GetOutputKernelTensor(value_node, 0, false);
  MS_EXCEPTION_IF_NULL(old_kernel_tensor);
  auto device_tensor = old_kernel_tensor->device_address();
  MS_EXCEPTION_IF_NULL(device_tensor);

  const auto &front_node = AnfAlgo::FetchFrontNodeByBackendNode(value_node, *graph);
  device_tensor->SetNodeIndex(value_node, 0);
  SchedulerHelper::AddDeviceTensorStore(front_node, old_kernel_tensor);

  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  const auto &device_name = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);

  // If the device tensor store of this device type is not exist, then create the new device tensor of this type.
  if (DeviceTensorStore::GetInstance().Fetch(front_node.get(), device::GetDeviceTypeByName(device_name)) == nullptr) {
    MS_LOG(INFO) << "Fetch no device tensor store by:" << front_node->fullname_with_scope() << ", type:" << device_name
                 << " dtype:" << old_kernel_tensor->dtype_id()
                 << " current kernel tensor:" << old_kernel_tensor->ToString()
                 << " in value node:" << value_node->DebugString();

    const auto &kernel_tensor = AnfAlgo::CreateOutputKernelTensorWithDeviceInfo(
      {value_node, 0}, nullptr, device_tensor->GetSize(), kernel::GetFormatFromEnumToStr(old_kernel_tensor->format()),
      old_kernel_tensor->dtype_id(), old_kernel_tensor->GetShapeVector(), device_name, device_id);
    kernel_tensor->set_stream_id(device_tensor->stream_id());
    auto other_type_device_tensor = kernel_tensor->device_address().get();
    MS_EXCEPTION_IF_NULL(other_type_device_tensor);
    other_type_device_tensor->SetNodeIndex(value_node, 0);
    other_type_device_tensor->set_from_persistent_mem(true);
    MS_LOG(DEBUG) << "Create device tensor:" << kernel_tensor->ToString()
                  << " by kernel tensor: " << kernel_tensor->ToString();
    SchedulerHelper::AddDeviceTensorStore(front_node, kernel_tensor);
  }
}

void GraphScheduler::PersistDeviceTensorForParameter(const AnfNodePtr &parameter, const KernelGraphPtr &graph,
                                                     const GraphCompilerInfo &graph_compiler_info) const {
  MS_EXCEPTION_IF_NULL(parameter);
  MS_EXCEPTION_IF_NULL(graph);

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

  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  const auto &device_name = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);

  // If the device tensor store of this device type is not exist, then create the new device tensor of this type.
  if (DeviceTensorStore::GetInstance().Fetch(front_node.get(), device::GetDeviceTypeByName(device_name)) == nullptr) {
    MS_LOG(INFO) << "Fetch no device tensor store by:" << front_node->fullname_with_scope() << ", type:" << device_name;

    const auto &kernel_tensor = AnfAlgo::CreateOutputKernelTensorWithDeviceInfo(
      {parameter, 0}, nullptr, device_tensor->GetSize(), kernel::GetFormatFromEnumToStr(old_kernel_tensor->format()),
      old_kernel_tensor->dtype_id(), old_kernel_tensor->GetShapeVector(), device_name, device_id);
    kernel_tensor->set_stream_id(device_tensor->stream_id());
    auto other_type_device_tensor = kernel_tensor->device_address().get();
    if (front_node->isa<ValueNode>()) {
      const auto &value_node = front_node->cast<ValueNodePtr>();
      MS_EXCEPTION_IF_NULL(value_node);
      if (value_node->value() != nullptr) {
        kernel_tensor->set_value(value_node->value());
      }
    }
    other_type_device_tensor->SetNodeIndex(parameter, 0);
    other_type_device_tensor->set_from_persistent_mem(true);
    MS_LOG(DEBUG) << "Create kernel tensor:" << kernel_tensor->ToString()
                  << " by kernel tensor: " << old_kernel_tensor->ToString();
    SchedulerHelper::AddDeviceTensorStore(front_node, kernel_tensor);
  }
}

void GraphScheduler::PersistDeviceTensorForRootGraphControlNode(const GraphCompilerInfo &graph_compiler_info) const {
  const auto &parser = graph_compiler_info.control_node_parser_;
  if (parser == nullptr || (!parser->IsInited())) {
    return;
  }

  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  const auto &device_name = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);

  for (auto &root_graph_parameter : graph_compiler_info.origin_parameters_order_) {
    MS_EXCEPTION_IF_NULL(root_graph_parameter);
    if (!IsPersistentDeviceTensor(root_graph_parameter)) {
      continue;
    }

    // The device tensor store has been done in the backend kernel graph corresponding to the root graph.
    if (!DeviceTensorStore::GetInstance().Fetch(root_graph_parameter.get()).empty()) {
      continue;
    }

    // The different root graph parameters may correspond to parameter of same sub kernel graph when call the same sub
    // graph using the different root graph parameters. So can not use the device tensor of sub kernel graph parameter
    // directly and choose the first backend parameter in sub kernel graphs to create new device tensor to make sure
    // that the device tensor of root graph parameters are different.
    const auto &node_with_index = parser->FetchBackendParameterWithContextByFrontParameter({root_graph_parameter, 0});
    if (node_with_index.first == nullptr) {
      MS_LOG_WITH_NODE(INTERNAL_EXCEPTION, root_graph_parameter)
        << "#dmsg#Runtime error info:#dmsg#Can't find backend node for weight parameter:"
        << root_graph_parameter->DebugString();
    }
    const auto &backend_node = node_with_index.first;
    const auto &index = node_with_index.second;
    MS_EXCEPTION_IF_NULL(backend_node);
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
      sub_kernel_tensor->GetShapeVector(), device_name, device_id);
    kernel_tensor->set_stream_id(AnfAlgo::GetStreamId(backend_node));
    auto new_device_tensor = kernel_tensor->device_address().get();
    MS_EXCEPTION_IF_NULL(new_device_tensor);
    new_device_tensor->SetNodeIndex(backend_node, index);
    kernel_tensor->set_is_ptr_persisted(sub_kernel_tensor->is_ptr_persisted());
    new_device_tensor->set_from_persistent_mem(true);
    kernel_tensor->set_user_data(sub_kernel_tensor->user_data());

    SchedulerHelper::AddDeviceTensorStore(root_graph_parameter, kernel_tensor);
    MS_LOG(INFO) << "Add device tensor store by root graph parameter:" << root_graph_parameter->fullname_with_scope()
                 << ", backend node:" << backend_node->DebugString()
                 << ", type:" << device::GetDeviceTypeByName(device_name) << " device_tensor:" << new_device_tensor
                 << ", kernel tensor: " << kernel_tensor;
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
      const auto kernel_tensors = DeviceTensorStore::GetInstance().Fetch(front_node.get());
      ofs << "\t\tdevice tensor key:" << front_node->fullname_with_scope()
          << "\tbackend node name:" << value_node->fullname_with_scope() << "\tvalue size:" << kernel_tensors.size()
          << "\n";
      for (const auto &kernel_tensor : kernel_tensors) {
        MS_EXCEPTION_IF_NULL(kernel_tensor);
        auto device_tensor = kernel_tensor->device_address().get();
        MS_EXCEPTION_IF_NULL(device_tensor);
        ofs << "\t\t\tdevice tensor value:" << device_tensor << "\tptr:" << device_tensor->GetPtr()
            << "\tsize:" << device_tensor->GetSize() << "\tstream id:" << device_tensor->stream_id()
            << "\toriginal_ref_count:" << kernel_tensor->original_ref_count()
            << "\tdynamic_ref_count:" << kernel_tensor->dynamic_ref_count() << "\tflag:" << kernel_tensor->flag()
            << "\tdevice_type:" << device_tensor->GetDeviceType()
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
}
}  // namespace runtime
}  // namespace ge_backend
}  // namespace mindspore
