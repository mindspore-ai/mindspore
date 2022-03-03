/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include "runtime/graph_scheduler/graph_scheduler.h"
#include <queue>
#include "runtime/graph_scheduler/actor/memory_manager_actor.h"
#include "runtime/graph_scheduler/actor/debug_actor.h"
#include "runtime/graph_scheduler/actor/recorder_actor.h"
#include "runtime/hardware/device_context_manager.h"
#include "mindrt/src/actor/actormgr.h"
#include "mindrt/include/async/async.h"
#include "backend/common/session/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "backend/common/optimizer/helper.h"
#include "utils/anf_utils.h"
#include "include/common/utils/config_manager.h"
#include "utils/log_adapter.h"
#include "include/common/utils/convert_utils.h"
#include "utils/ms_context.h"
#include "utils/profile.h"
#if !defined(_WIN32) && !defined(_WIN64)
#include "include/common/utils/signal_util.h"
#endif
#ifndef ENABLE_SECURITY
#include "debug/data_dump/dump_json_parser.h"
#endif
#ifdef ENABLE_DUMP_IR
#include "debug/rdr/recorder_manager.h"
#include "debug/rdr/running_data_recorder.h"
#endif
#ifdef ENABLE_DEBUGGER
#include "debug/debugger/debugger.h"
#endif
#include "profiler/device/profiling.h"
#include "debug/common.h"

namespace mindspore {
namespace runtime {
namespace {
bool IsNeedInsertCopyActor(const DeviceContext *from_device_context, const DeviceContext *to_device_context) {
  MS_EXCEPTION_IF_NULL(from_device_context);
  MS_EXCEPTION_IF_NULL(to_device_context);

  if (from_device_context->GetDeviceAddressType() == to_device_context->GetDeviceAddressType()) {
    return false;
  } else {
    return true;
  }
}

inline bool IsSingleOpActorSet(const ActorSet *actor_set) {
  MS_EXCEPTION_IF_NULL(actor_set);
  return actor_set->kernel_actors_.size() == 1;
}

// Convert the actors vector by the actor set.
std::vector<AbstractActorPtr> CollectActors(const ActorSet *actor_set) {
  MS_EXCEPTION_IF_NULL(actor_set);
  std::vector<AbstractActorPtr> actors;

  if (actor_set->data_prepare_actor_ != nullptr) {
    (void)actors.emplace_back(static_cast<AbstractActorPtr>(actor_set->data_prepare_actor_));
  }
  for (auto &data_source_actor : actor_set->data_source_actors_) {
    MS_EXCEPTION_IF_NULL(data_source_actor);
    (void)actors.emplace_back(static_cast<AbstractActorPtr>(data_source_actor));
  }
  for (auto &custom_actor : actor_set->custom_actors_) {
    MS_EXCEPTION_IF_NULL(custom_actor);
    (void)actors.emplace_back(static_cast<AbstractActorPtr>(custom_actor));
  }
  for (auto &kernel_actor : actor_set->kernel_actors_) {
    MS_EXCEPTION_IF_NULL(kernel_actor);
    (void)actors.emplace_back(static_cast<AbstractActorPtr>(kernel_actor));
  }
  for (auto &super_kernel_actor : actor_set->super_kernel_actors_) {
    MS_EXCEPTION_IF_NULL(super_kernel_actor);
    (void)actors.emplace_back(static_cast<AbstractActorPtr>(super_kernel_actor));
  }
  for (auto &copy_actor : actor_set->copy_actors_) {
    MS_EXCEPTION_IF_NULL(copy_actor);
    (void)actors.emplace_back(static_cast<AbstractActorPtr>(copy_actor));
  }
  if (actor_set->loop_count_actor_ != nullptr) {
    (void)actors.emplace_back(static_cast<AbstractActorPtr>(actor_set->loop_count_actor_));
  }
  if (actor_set->output_actor_ != nullptr) {
    (void)actors.emplace_back(static_cast<AbstractActorPtr>(actor_set->output_actor_));
  }
  if (actor_set->control_actors_ != nullptr) {
    const auto &control_actor_set = actor_set->control_actors_;
    for (auto &switch_actor : control_actor_set->switch_actors_) {
      MS_EXCEPTION_IF_NULL(switch_actor);
      (void)actors.emplace_back(static_cast<AbstractActorPtr>(switch_actor));
    }
    for (auto &gather_actor : control_actor_set->gather_actors_) {
      MS_EXCEPTION_IF_NULL(gather_actor);
      (void)actors.emplace_back(static_cast<AbstractActorPtr>(gather_actor));
    }
    for (auto &entrance_actor : control_actor_set->entrance_actors_) {
      MS_EXCEPTION_IF_NULL(entrance_actor);
      (void)actors.emplace_back(static_cast<AbstractActorPtr>(entrance_actor));
    }
    for (auto &exit_actor : control_actor_set->exit_actors_) {
      MS_EXCEPTION_IF_NULL(exit_actor);
      (void)actors.emplace_back(static_cast<AbstractActorPtr>(exit_actor));
    }
    for (auto &stack_actor : control_actor_set->stack_actors_) {
      MS_EXCEPTION_IF_NULL(stack_actor);
      (void)actors.emplace_back(static_cast<AbstractActorPtr>(stack_actor));
    }
  }

  return actors;
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
    auto front_input_node = FetchFrontNodeByBackendNode(input_node, graph);
    DeviceTensorStore::GetInstance().Remove(front_input_node.get());
    size_t output_num = common::AnfAlgo::GetOutputTensorNum(input_node);
    for (size_t index = 0; index < output_num; ++index) {
      if (AnfAlgo::OutputAddrExist(input_node, index)) {
        AnfAlgo::SetOutputAddr(nullptr, index, input_node.get());
      }
    }
  }

  // Clear input value node device tensor and device tensor store.
  for (const auto &value_node : graph->graph_value_nodes()) {
    auto front_value_node = FetchFrontNodeByBackendNode(value_node, graph);
    DeviceTensorStore::GetInstance().Remove(front_value_node.get());
    if (AnfAlgo::OutputAddrExist(value_node, 0)) {
      AnfAlgo::SetOutputAddr(nullptr, 0, value_node.get());
    }
  }

  // Clear cnode device tensor.
  for (const auto &cnode : graph->execution_order()) {
    size_t output_num = common::AnfAlgo::GetOutputTensorNum(cnode);
    for (size_t index = 0; index < output_num; ++index) {
      if (AnfAlgo::OutputAddrExist(cnode, index)) {
        AnfAlgo::SetOutputAddr(nullptr, index, cnode.get());
      }
    }
  }
}

#if !defined(_WIN32) && !defined(_WIN64)
void IntHandler(int, siginfo_t *, void *) {
  int this_pid = getpid();
  MS_LOG(WARNING) << "Process " << this_pid << " receive KeyboardInterrupt signal.";
  (void)kill(this_pid, SIGTERM);
}
#endif
}  // namespace

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
    auto base_actors = CollectActors(actor_set.get());
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
      const auto &node = front_value_node.first.first;
      size_t index = front_value_node.first.second;
      if (AnfAlgo::OutputAddrExist(node, index)) {
        AnfAlgo::SetOutputAddr(nullptr, index, node.get());
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

  // Clear the member of DeviceTensorCopyStore.
  DeviceTensorCopyStore::GetInstance().Clear();

  for (auto &super_kernel_actor : actor_set->super_kernel_actors_) {
    MS_EXCEPTION_IF_NULL(super_kernel_actor);
    super_kernel_actor->memory_free_lists_ = std::queue<std::vector<DeviceTensor *>>();
  }

  control_node_scheduler_.ClearActorData(actor_set->control_actors_.get());

  // At the end of the step, the op data sent to the stack actor in each actor should be clear.
  auto total_actors = CollectActors(actor_set);
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

  (void)kKernelTypeToLinkFunc.emplace(KernelTransformType::kDeviceDataSourceActor,
                                      &GraphScheduler::LinkDataArrowForBaseActor);
  (void)kKernelTypeToLinkFunc.emplace(KernelTransformType::kHostDataSourceActor,
                                      &GraphScheduler::LinkDataArrowForHostDSActor);
  (void)kKernelTypeToLinkFunc.emplace(KernelTransformType::kKernelActor, &GraphScheduler::LinkDataArrowForKernelActor);
  (void)kKernelTypeToLinkFunc.emplace(KernelTransformType::kSuperKernelActor,
                                      &GraphScheduler::LinkDataArrowForBaseActor);
  (void)kKernelTypeToLinkFunc.emplace(KernelTransformType::kDeviceTensorStore,
                                      &GraphScheduler::LinkDataArrowForDeviceTensorStore);
  (void)kKernelTypeToLinkFunc.emplace(KernelTransformType::kInternalParameter,
                                      &GraphScheduler::LinkDataArrowForInternalParameter);
  (void)kKernelTypeToLinkFunc.emplace(KernelTransformType::kSendActor, &GraphScheduler::LinkDataArrowForBaseActor);
  (void)kKernelTypeToLinkFunc.emplace(KernelTransformType::kRecvActor, &GraphScheduler::LinkDataArrowForBaseActor);

  // Create the thread pool of actor runtime and Set the OMP_NUM_THREADS env.
  size_t actor_thread_num = 0;
  size_t actor_and_kernel_thread_num = 0;
  ComputeThreadNums(&actor_thread_num, &actor_and_kernel_thread_num);
  auto actor_manager = ActorMgr::GetActorMgrRef();
  MS_EXCEPTION_IF_NULL(actor_manager);
  auto ret = actor_manager->Initialize(true, actor_thread_num, actor_and_kernel_thread_num);
  if (ret != MINDRT_OK) {
    MS_LOG(EXCEPTION) << "Actor manager init failed.";
  }
  common::SetOMPThreadNum();
  MS_LOG(INFO) << "The actor thread number: " << actor_thread_num
               << ", the kernel thread number: " << (actor_and_kernel_thread_num - actor_thread_num);

#ifdef ENABLE_RPC_ACTOR
  // Create and initialize RpcNodeScheduler.
  rpc_node_scheduler_ = std::make_unique<RpcNodeScheduler>();
  MS_EXCEPTION_IF_NULL(rpc_node_scheduler_);
  rpc_node_scheduler_->Initialize();
#endif

  BuildAndScheduleGlobalActor();
}

void GraphScheduler::BuildAndScheduleGlobalActor() {
  auto actor_manager = ActorMgr::GetActorMgrRef();
  MS_EXCEPTION_IF_NULL(actor_manager);

  // Create and schedule memory manager actor.
  auto memory_manager_actor = std::make_shared<MemoryManagerActor>();
  MS_EXCEPTION_IF_NULL(memory_manager_actor);
  memory_manager_aid_ = memory_manager_actor->GetAID();
  auto base_actor = static_cast<ActorReference>(memory_manager_actor);
  // Bind single thread to response to memory alloc and free quickly.
  (void)actor_manager->Spawn(base_actor, false);

  // Create and schedule recorder actor.
  auto recorder_actor = std::make_shared<RecorderActor>();
  MS_EXCEPTION_IF_NULL(recorder_actor);
  recorder_aid_ = &(recorder_actor->GetAID());
  auto base_recorder_actor = static_cast<ActorReference>(recorder_actor);
  (void)actor_manager->Spawn(base_recorder_actor, true);

  // Create and schedule debug actor.
  // debugger_actor_need is true for CPU when e2e dump is enabled and for Ascend and GPU is true when debugger or dump
  // is enabled.
#ifndef ENABLE_SECURITY
  bool debugger_actor_need = DumpJsonParser::GetInstance().e2e_dump_enabled();
#endif
#ifdef ENABLE_DEBUGGER
  if (Debugger::GetInstance()->DebuggerBackendEnabled()) {
    debugger_actor_need = true;
  }
#endif
#ifndef ENABLE_SECURITY
  if (debugger_actor_need) {
    auto debug_actor = std::make_shared<DebugActor>();
    MS_EXCEPTION_IF_NULL(debug_actor);
    debug_aid_ = &(debug_actor->GetAID());
    auto base_debug_actor = static_cast<ActorReference>(debug_actor);
    (void)actor_manager->Spawn(base_debug_actor, true);
  }
#endif
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
    }
  };
  // cppcheck-suppress unreadVariable
  ScopeCleaner cleaner(this);
  MS_LOG(INFO) << "Graph(" << graph_compiler_info.name_ << ") transforms actor begin.";
  if (graph_compiler_info.graphs_.size() == 0) {
    MS_LOG(EXCEPTION) << "The number of graphs is zero.";
  }
  if (graph_compiler_info.graphs_.size() != graph_compiler_info.device_contexts_.size()) {
    MS_LOG(EXCEPTION) << "The number of graphs is not equal to the number of device contexts.";
  }

  PersistDeviceTensor(graph_compiler_info);
  const auto &actor_set = Build(graph_compiler_info);
  MS_EXCEPTION_IF_NULL(actor_set);
  CacheGraphOutputToActor(graph_compiler_info);
  Link(actor_set.get(), graph_compiler_info);
  Optimize(actor_set.get());

  DumpActor(actor_set.get(), graph_compiler_info);
  if (graph_compiler_info.strategy_ == GraphExecutionStrategy::kPipeline) {
    CheckActorValid(actor_set.get());
  }
  MS_LOG(INFO) << "Graph(" << graph_compiler_info.name_ << ") transforms actor end.";

  return actor_set.get();
}

void GraphScheduler::Schedule(const ActorSet *actor_set) {
  MS_EXCEPTION_IF_NULL(actor_set);
  auto actors = CollectActors(actor_set);
  // Schedule actors.
  auto actor_manager = ActorMgr::GetActorMgrRef();
  MS_EXCEPTION_IF_NULL(actor_manager);
  for (auto actor : actors) {
    (void)actor_manager->Spawn(actor);
  }

#ifdef ENABLE_RPC_ACTOR
  // Build physical connections in 'RpcNodeScheduler::Schedule()' method. This costs some time.
  MS_EXCEPTION_IF_NULL(rpc_node_scheduler_);
  rpc_node_scheduler_->Schedule();
#endif
}

void GraphScheduler::Run(ActorSet *const actor_set, const std::vector<DeviceContext *> &device_contexts,
                         const std::vector<std::vector<TensorPtr>> &input_tensors,
                         const std::vector<TensorPtr> &input_tensors_with_value_node, GraphExecutionStrategy strategy) {
  MS_EXCEPTION_IF_NULL(actor_set);
  MS_EXCEPTION_IF_NULL(actor_set->data_prepare_actor_);
#if !defined(_WIN32) && !defined(_WIN64)
  SignalGuard sg(IntHandler);
#endif

  // Construct OpContext.
  OpContext<DeviceTensor> op_context;
  std::vector<Promise<int>> result(1);
  op_context.sequential_num_ = RandInt::Instance().Get();
  op_context.results_ = &result;

#ifdef ENABLE_RPC_ACTOR
  // Set OpContext to rpc node scheduler.
  MS_EXCEPTION_IF_NULL(rpc_node_scheduler_);
  rpc_node_scheduler_->SetOpcontext(&op_context);
#endif

  if ((strategy == GraphExecutionStrategy::kStep) && IsSingleOpActorSet(actor_set)) {
    actor_set->data_prepare_actor_->PrepareData(input_tensors, &op_context, GraphExecutionStrategy::kStep);
    MS_EXCEPTION_IF_NULL(actor_set->kernel_actors_[0]);
    actor_set->kernel_actors_[0]->RunOpControlWithInputTensor(nullptr, &op_context, &input_tensors_with_value_node);
    return;
  }

  // Trigger data prepare actor running.
  MS_EXCEPTION_IF_NULL(ActorMgr::GetActorMgrRef());
  auto thread_pool = ActorMgr::GetActorMgrRef()->GetActorThreadPool();
  MS_EXCEPTION_IF_NULL(thread_pool);
  ActorDispatcher::is_multi_thread_execution(actor_set->is_multi_thread_execution_);
  double start_time = GetTime();
  ActorDispatcher::Send(actor_set->data_prepare_actor_->GetAID(), &DataPrepareActor::PrepareData, input_tensors,
                        &op_context, GraphExecutionStrategy::kPipeline);

  // Get the run result.
  auto result_future = result[0].GetFuture();
  result_future.Wait();
  MsException::Instance().CheckException();
  if (!result_future.IsOK()) {
#ifdef ENABLE_DUMP_IR
    mindspore::RDR::TriggerAll();
#endif
    // When temporary variable 'op_context' has beed set failed status, the main thread need wait other threads until
    // they finish respective task, otherwise segmentation fault will happen when these task access 'op_context',
    // because it has been destroyed.
    std::mutex mutex;
    std::unique_lock<std::mutex> locker(mutex);
    std::condition_variable thread_blocker;
    const int64_t kTimeToWait = 2;
    (void)thread_blocker.wait_for(locker, std::chrono::seconds(kTimeToWait));
    // May set exception in the wait time, need throw the exception to avoid affecting the next execution.
    MsException::Instance().CheckException();
    MS_LOG(EXCEPTION) << op_context.error_info_;
  }

  // Sync device stream.
  if (strategy == GraphExecutionStrategy::kPipeline) {
    std::set<DeviceContext *> sync_stream_device_contexts;
    for (auto &device_context : device_contexts) {
      MS_EXCEPTION_IF_NULL(device_context);
      if ((sync_stream_device_contexts.count(device_context) == 0) && (!device_context->SyncStream())) {
        MS_LOG(EXCEPTION) << "Sync stream failed:" << device_context->device_context_key().ToString();
      }
      (void)sync_stream_device_contexts.insert(device_context);
    }
  }

  double end_time = GetTime();
  const size_t kSecondsToMilliseconds = 1000;
  SetActorExecutionStrategy(actor_set, strategy, (end_time - start_time) * kSecondsToMilliseconds);
}

void GraphScheduler::SetActorExecutionStrategy(ActorSet *const actor_set, GraphExecutionStrategy strategy,
                                               double execution_time) const {
  MS_EXCEPTION_IF_NULL(actor_set);
  MS_EXCEPTION_IF_NULL(actor_set->loop_count_actor_);
  ++actor_set->execution_count_;
  MS_LOG(DEBUG) << "Execution count: " << actor_set->execution_count_ << ", execution time cost: " << execution_time
                << " ms in multi thread or not: " << actor_set->is_multi_thread_execution_ << ".";
#if defined(_WIN32) || defined(_WIN64)
  return;
#endif

  // The step mode uses the default multi thread.
  if (strategy == GraphExecutionStrategy::kStep) {
    return;
  }

  // The constraint condition of not supporting the single thread execution.
  if ((actor_set->control_actors_ != nullptr) || (actor_set->copy_actors_.size() > 0) ||
      (actor_set->super_kernel_actors_.size() > 0) || (actor_set->loop_count_actor_->loop_count() > 1) ||
      (actor_set->kernel_actors_.size() > ActorDispatcher::kSingleThreadExecutionActorMaxNum)) {
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
    MS_LOG(ERROR) << "Can't find the actors map of " << actor_info;
    return nullptr;
  }
}

ActorSetPtr GraphScheduler::Build(const GraphCompilerInfo &graph_compiler_info) {
  auto actor_set = std::make_shared<ActorSet>(graph_compiler_info.name_);
  MS_EXCEPTION_IF_NULL(actor_set);
  (void)actors_.emplace(actor_set->name_, actor_set);

  auto host_queue = std::make_shared<HostTensorQueue>();
  actor_set->data_source_actors_ = BuildDataSourceActor(graph_compiler_info, host_queue);
  actor_set->custom_actors_ = BuildCustomActor(graph_compiler_info);
  actor_set->kernel_actors_ = BuildKernelActor(graph_compiler_info);
  actor_set->super_kernel_actors_ = BuildSuperKernelActor(graph_compiler_info);
  actor_set->loop_count_actor_ = BuildLoopCountActor(graph_compiler_info);
  actor_set->output_actor_ = BuildOutputActor(graph_compiler_info);
  actor_set->data_prepare_actor_ =
    BuildDataPrepareActor(graph_compiler_info, actor_set->data_source_actors_, host_queue);
  actor_set->control_actors_ = control_node_scheduler_.Build(graph_compiler_info, memory_manager_aid_);

#ifdef ENABLE_RPC_ACTOR
  MS_EXCEPTION_IF_NULL(rpc_node_scheduler_);
  actor_set->rpc_actors_ = rpc_node_scheduler_->Build(graph_compiler_info);
#endif
  return actor_set;
}

void GraphScheduler::CacheGraphOutputToActor(const GraphCompilerInfo &graph_compiler_info) {
  if (graph_compiler_info.strategy_ == GraphExecutionStrategy::kStep) {
    return;
  }

  for (const auto &graph : graph_compiler_info.graphs_) {
    MS_EXCEPTION_IF_NULL(graph);
    auto outputs = common::AnfAlgo::GetAllOutputWithIndex(graph->output());
    for (const auto &output_with_index : outputs) {
      auto output_kernel = output_with_index.first;
      MS_EXCEPTION_IF_NULL(output_kernel);
      auto origin_output_with_index = graph->GetFrontNodeWithIndexByGraphOutput(output_with_index);
      if (origin_output_with_index.first == nullptr) {
        MS_LOG(WARNING) << "The graph " << graph->graph_id() << " output node:" << output_kernel->fullname_with_scope()
                        << " with index: " << output_with_index.second << " has no front node.";
        continue;
      }

      auto kernel_type = FetchKernelTransformType(output_kernel, graph, graph_compiler_info.origin_parameters_order_);
      auto output_actor = FetchActor(kernel_type, graph_compiler_info.name_, output_kernel, graph);
      if (output_actor == nullptr) {
        MS_LOG(INFO) << "The graph " << graph->graph_id() << " output node:" << output_kernel->fullname_with_scope()
                     << " with index:" << output_with_index.second
                     << " is not actor, and the kernel type is:" << kernel_type;
      }
      auto output_actor_name = (output_actor != nullptr) ? output_actor->GetAID().Name() : "";
      (void)graph_output_to_actor_.emplace(origin_output_with_index, GraphOutputPair(output_actor, output_with_index));
      MS_LOG(INFO) << "Cache the graph " << graph->graph_id() << " output node:" << output_kernel->fullname_with_scope()
                   << " with index:" << output_with_index.second << " to actor:" << output_actor_name
                   << ", from front node:" << origin_output_with_index.first->fullname_with_scope()
                   << " with index:" << origin_output_with_index.second;
    }
  }
}

void GraphScheduler::Link(ActorSet *actor_set, const GraphCompilerInfo &graph_compiler_info) {
  MS_EXCEPTION_IF_NULL(actor_set);
  std::vector<AbstractActor *> auto_monad_actors;
  GroupNameToCommuNodes group_name_to_communication_nodes;
  std::string default_group_name = "";
  const auto &parser = graph_compiler_info.control_node_parser_;
  MS_EXCEPTION_IF_NULL(parser);

  for (const auto &graph : graph_compiler_info.graphs_) {
    MS_EXCEPTION_IF_NULL(graph);
    if (graph->execution_order().empty()) {
      MS_LOG(INFO) << "The graph " << graph->graph_id() << " is an empty graph and skips linking.";
      continue;
    }
    if (graph->is_executing_sink()) {
      LinkDataArrowInSinkMode(graph, graph_compiler_info, &auto_monad_actors);
    } else {
      // In the control flow, the communication nodes need to be guaranteed to be executed in order. The order
      // within the kernel graph group needs to add control arrows between the communication nodes, and the order
      // between groups is guaranteed by the control flow framework. Therefore, communication nodes need to be
      // grouped by group name. And this is not required in non-control flow, the default unified group name is used.
      std::vector<CNodePtr> communication_nodes;
      const auto &group_name = (parser->IsInited() ? parser->FetchGroupNameByKernelGraph(graph) : default_group_name);
      LinkDataArrowInNonSinkMode(graph, graph_compiler_info, &auto_monad_actors, &communication_nodes);
      group_name_to_communication_nodes[group_name].insert(group_name_to_communication_nodes[group_name].end(),
                                                           communication_nodes.begin(), communication_nodes.end());
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

#ifdef ENABLE_RPC_ACTOR
  // Link inter-process arrows for rpc actors.
  MS_EXCEPTION_IF_NULL(rpc_node_scheduler_);
  rpc_node_scheduler_->Link(actor_set);
#endif
}

void GraphScheduler::Optimize(ActorSet *const actor_set) {
  MS_EXCEPTION_IF_NULL(actor_set);
  control_node_scheduler_.Optimize(actor_set->control_actors_.get());
}

std::vector<DataSourceActorPtr> GraphScheduler::BuildDataSourceActor(const GraphCompilerInfo &graph_compiler_info,
                                                                     const HostTensorQueuePtr &host_queue) {
  std::vector<DataSourceActorPtr> data_source_actors;
  HostQueueDSActorPtr host_queue_ds_actor = nullptr;
  size_t data_node_position = 0;
  mindspore::HashMap<AnfNodePtr, size_t> front_node_position_temp_map;

  for (size_t i = 0; i < graph_compiler_info.graphs_.size(); ++i) {
    const auto &graph = graph_compiler_info.graphs_[i];
    const auto &device_context = graph_compiler_info.device_contexts_[i];
    MS_EXCEPTION_IF_NULL(graph);
    // Build host queue data source actor.
    const std::vector<AnfNodePtr> &input_nodes = graph->input_nodes();
    const auto &root_parameters = graph_compiler_info.origin_parameters_order_;

    for (size_t j = 0; j < input_nodes.size(); j++) {
      const auto &input_node = input_nodes[j];
      MS_EXCEPTION_IF_NULL(input_node);

      if (IsHostQueueDSActor(input_node, graph, root_parameters, graph_compiler_info.strategy_)) {
        // In control flow, parameters from subgraph need not init in data source actor.
        if (graph_compiler_info.control_node_parser_->IsInited()) {
          auto node_with_index = graph->GetElementInTupleBackendFrontIndexMap(input_node);
          if (node_with_index.first != nullptr && node_with_index.first->isa<Parameter>() &&
              find(root_parameters.begin(), root_parameters.end(), node_with_index.first) == root_parameters.end())
            continue;
        }

        if (host_queue_ds_actor == nullptr) {
          auto actor_name = graph_compiler_info.name_ + kHostDSActorNameSuffix;
          MS_LOG(INFO) << "Create host queue data source actor: " << actor_name;
          host_queue_ds_actor = std::make_shared<HostQueueDataSourceActor>(actor_name, 1, memory_manager_aid_, nullptr,
                                                                           nullptr, host_queue);
          InsertActor(host_queue_ds_actor.get());
          (void)data_source_actors.emplace_back(host_queue_ds_actor);
        }

        const auto &front_node = FetchFrontNodeByBackendNode(input_node, graph);
        // In the scenario where multiple backend nodes correspond to the same front node, only the first backend node
        // is saved in the host queue data source actor.
        if (front_node_position_temp_map.count(front_node) > 0) {
          (void)host_queue_ds_actor->data_node_position_map_.emplace(input_node,
                                                                     front_node_position_temp_map[front_node]);
          continue;
        }
        (void)host_queue_ds_actor->data_nodes_.emplace_back(input_node);
        (void)host_queue_ds_actor->device_contexts_.emplace_back(device_context);
        (void)host_queue_ds_actor->data_node_position_map_.emplace(input_node, data_node_position);
        // In control flow, need to rely on the front node to find the location of the corresponding real parameter.
        (void)host_queue_ds_actor->data_node_position_map_.emplace(front_node, data_node_position);
        (void)front_node_position_temp_map.emplace(front_node, data_node_position);
        data_node_position++;
      }
    }

    // The graph sink mode has no device queue data source actor.
    if (!graph->is_executing_sink()) {
      // Build device queue data source actor.
      const auto &execution_order = graph->execution_order();
      const auto &iter =
        std::find_if(execution_order.begin(), execution_order.end(), [&graph_compiler_info](const CNodePtr &node) {
          return IsDeviceQueueDSActor(node, graph_compiler_info.strategy_);
        });
      if (iter != execution_order.end()) {
        auto actor_name =
          graph_compiler_info.name_ + kDeviceDSActorNameSuffix + "_" + std::to_string(graph->graph_id());
        MS_LOG(INFO) << "Create queue data source actor: " << actor_name;
        auto device_queue_ds_actor = std::make_shared<DeviceQueueDataSourceActor>(
          actor_name, 1, device_context, memory_manager_aid_, debug_aid_, recorder_aid_);
        MS_EXCEPTION_IF_NULL(device_queue_ds_actor);
        InsertActor(device_queue_ds_actor.get());
        (void)data_source_actors.emplace_back(device_queue_ds_actor);
        device_queue_ds_actor->data_kernel_ = *iter;
        device_queue_ds_actor->kernel_info_ = dynamic_cast<device::KernelInfo *>((*iter)->kernel_info());
      }
    }
  }

  const auto parser = graph_compiler_info.control_node_parser_;
  MS_EXCEPTION_IF_NULL(parser);

  // Initialize the parameter in the control node, first get all the front parameters in the control node, then find
  // the corresponding backend parameter from the map, and insert it into the host data source actor.
  const auto &control_node_parameters = parser->control_node_parameters();
  for (const auto &parameter : control_node_parameters) {
    if (IsPersistentDeviceTensor(parameter)) {
      continue;
    }
    if (host_queue_ds_actor == nullptr) {
      auto actor_name = graph_compiler_info.name_ + kHostDSActorNameSuffix;
      MS_LOG(INFO) << "Create host queue data source actor: " << actor_name;
      host_queue_ds_actor =
        std::make_shared<HostQueueDataSourceActor>(actor_name, 1, memory_manager_aid_, nullptr, nullptr, host_queue);
      InsertActor(host_queue_ds_actor.get());
      (void)data_source_actors.emplace_back(host_queue_ds_actor);
    }

    auto &node_map = host_queue_ds_actor->data_node_position_map_;
    if (node_map.find(parameter) != node_map.end()) {
      continue;
    }
    const auto &backend_parameter_with_context =
      parser->FetchBackendParameterWithContextByFrontParameter({parameter, 0});
    const auto &backend_node = backend_parameter_with_context.first;
    MS_EXCEPTION_IF_NULL(backend_node);
    auto iter = find(host_queue_ds_actor->data_nodes_.begin(), host_queue_ds_actor->data_nodes_.end(), backend_node);
    if (iter != host_queue_ds_actor->data_nodes_.end()) {
      (void)node_map.emplace(parameter, iter - host_queue_ds_actor->data_nodes_.begin());
    } else {
      (void)node_map.emplace(parameter, host_queue_ds_actor->data_nodes_.size());
      (void)node_map.emplace(backend_node, host_queue_ds_actor->data_nodes_.size());
      (void)host_queue_ds_actor->data_nodes_.emplace_back(backend_node);
      (void)host_queue_ds_actor->device_contexts_.emplace_back(backend_parameter_with_context.second);
    }
  }

  return data_source_actors;
}

std::vector<CustomActorPtr> GraphScheduler::BuildCustomActor(const GraphCompilerInfo &graph_compiler_info) {
  std::vector<CustomActorPtr> custom_actors;
  for (size_t i = 0; i < graph_compiler_info.graphs_.size(); ++i) {
    const auto &device_context = graph_compiler_info.device_contexts_[i];
    const auto &graph = graph_compiler_info.graphs_[i];
    MS_EXCEPTION_IF_NULL(graph);
    if (graph->is_executing_sink()) {
      continue;
    }

    auto all_nodes = TopoSort(graph->get_return());
    for (const auto &node : all_nodes) {
      if (!AnfUtils::IsCustomActorNode(node)) {
        continue;
      }

      auto actor_name = AnfUtils::GetCustomActorName(node);
      auto custom_actor = std::make_shared<CustomActor>(actor_name, node, device_context, recorder_aid_);
      MS_EXCEPTION_IF_NULL(custom_actor);
      InsertActor(custom_actor.get());
      custom_actors.emplace_back(custom_actor);
    }
  }
  return custom_actors;
}

std::vector<KernelActorPtr> GraphScheduler::BuildKernelActor(const GraphCompilerInfo &graph_compiler_info) {
  std::vector<KernelActorPtr> kernel_actors;

  for (size_t i = 0; i < graph_compiler_info.graphs_.size(); ++i) {
    const auto &graph = graph_compiler_info.graphs_[i];
    const auto &device_context = graph_compiler_info.device_contexts_[i];
    MS_EXCEPTION_IF_NULL(graph);
    if (graph->is_executing_sink()) {
      continue;
    }

    auto execution_order = graph->execution_order();
    // Single op graph in step mode, kernel actor executes synchronously.
    bool is_single_op_graph = execution_order.size() == 1;
    GraphExecutionStrategy strategy = graph_compiler_info.strategy_;
    if (strategy == GraphExecutionStrategy::kStep) {
      strategy = (is_single_op_graph ? strategy : GraphExecutionStrategy::kPipeline);
    }

    for (auto &kernel : execution_order) {
      MS_EXCEPTION_IF_NULL(kernel);
      if (IsKernelActor(kernel, graph_compiler_info.strategy_) && (!IsSkippedKernelActor(kernel))) {
        auto ref_input_indexes = FetchModifiableRefInputIndex(kernel);
        auto ref_output_indexes = FetchModifiableRefOutputIndex(kernel, graph);
        KernelActorPtr kernel_actor = nullptr;
        if (IsRpcActor(kernel)) {
          kernel_actor = GenerateRpcActor(kernel, device_context, strategy, ref_input_indexes, ref_output_indexes);
        } else {
          kernel_actor =
            std::make_shared<KernelActor>(kernel->fullname_with_scope(), kernel, device_context, memory_manager_aid_,
                                          debug_aid_, recorder_aid_, strategy, ref_input_indexes, ref_output_indexes);
        }
        MS_EXCEPTION_IF_NULL(kernel_actor);
        InsertActor(kernel_actor.get());
        (void)kernel_actors.emplace_back(kernel_actor);
      }
    }
  }
  return kernel_actors;
}

std::vector<SuperKernelActorPtr> GraphScheduler::BuildSuperKernelActor(const GraphCompilerInfo &graph_compiler_info) {
  std::vector<SuperKernelActorPtr> super_kernel_actors;

  for (size_t i = 0; i < graph_compiler_info.graphs_.size(); ++i) {
    const auto &graph = graph_compiler_info.graphs_[i];
    const auto &device_context = graph_compiler_info.device_contexts_[i];
    MS_EXCEPTION_IF_NULL(graph);
    if (!graph->is_executing_sink()) {
      continue;
    }

    if (graph->execution_order().empty()) {
      MS_LOG(INFO) << "The graph " << graph->graph_id() << " is an empty graph and skips building.";
      continue;
    }

    auto actor_name = graph->ToString() + kSuperKernelActorNameSuffix;
    auto super_kernel_actor =
      std::make_shared<SuperKernelActor>(actor_name, graph, device_context, memory_manager_aid_, debug_aid_, nullptr);
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

  auto loop_count = ConfigManager::GetInstance().iter_num();
  if ((graph_compiler_info.strategy_ == GraphExecutionStrategy::kStep) ||
      (graph_compiler_info.graphs_.size() == 1 && graph_compiler_info.graphs_[0]->is_loop_count_sink())) {
    loop_count = 1;
  }

  auto actor_name = graph_compiler_info.name_ + kLoopCountActorNameSuffix;
  auto loop_count_actor =
    std::make_shared<LoopCountActor>(actor_name, loop_count, memory_manager_aid_, debug_aid_, recorder_aid_);
  MS_LOG(INFO) << "Create loop count actor: " << actor_name;
  MS_EXCEPTION_IF_NULL(loop_count_actor);

  InsertActor(loop_count_actor.get());
  return loop_count_actor;
}

OutputActorPtr GraphScheduler::BuildOutputActor(const GraphCompilerInfo &graph_compiler_info) {
  auto actor_set = Fetch(graph_compiler_info.name_);
  if ((graph_compiler_info.strategy_ == GraphExecutionStrategy::kStep) && IsSingleOpActorSet(actor_set)) {
    return nullptr;
  }

  auto loop_count = ConfigManager::GetInstance().iter_num();
  if ((graph_compiler_info.strategy_ == GraphExecutionStrategy::kStep) ||
      (graph_compiler_info.graphs_.size() == 1 && graph_compiler_info.graphs_[0]->is_loop_count_sink())) {
    loop_count = 1;
  }

  auto actor_name = graph_compiler_info.name_ + kOutputActorNameSuffix;
  auto output_actor = std::make_shared<OutputActor>(actor_name, loop_count, graph_compiler_info.outputs_num_);
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
    return data_source_actor->type_ == KernelTransformType::kHostDataSourceActor;
  });
  if (iter != data_source_actors.end()) {
    host_queue_ds_actor = std::dynamic_pointer_cast<HostQueueDataSourceActor>(*iter);
  }
  auto actor_name = graph_compiler_info.name_ + kDataPrepareActorNameSuffix;
  auto data_prepare_actor = std::make_shared<DataPrepareActor>(actor_name, memory_manager_aid_, debug_aid_,
                                                               &graph_compiler_info, host_queue_ds_actor, host_queue);
  MS_LOG(INFO) << "Create data prepare actor: " << actor_name;
  MS_EXCEPTION_IF_NULL(data_prepare_actor);

  // Cache the nodes which need continuous memory.
  if (graph_compiler_info.strategy_ == GraphExecutionStrategy::kPipeline) {
    for (size_t index = 0; index < graph_compiler_info.graphs_.size(); ++index) {
      const auto &graph = graph_compiler_info.graphs_[index];
      MS_EXCEPTION_IF_NULL(graph);
      if (graph->is_executing_sink()) {
        continue;
      }

      auto &execution_order = graph->execution_order();
      for (auto &kernel : execution_order) {
        if (!common::AnfAlgo::IsCommunicationOp(kernel)) {
          continue;
        }
        auto key = std::make_pair(kernel, graph_compiler_info.device_contexts_[index]);
        auto value = std::make_pair(false, false);
        if (common::AnfAlgo::GetInputTensorNum(kernel) > 1) {
          value.first = true;
        }
        if (common::AnfAlgo::GetOutputTensorNum(kernel) > 1) {
          value.second = true;
        }
        if ((value.first == true) || (value.second == true)) {
          data_prepare_actor->continuous_memory_nodes_[key] = value;
        }
      }
    }
  }

  InsertActor(data_prepare_actor.get());
  return data_prepare_actor;
}

std::vector<AbstractActorPtr> GraphScheduler::BuildNoInputKernelActor(const ActorSet *actor_set,
                                                                      GraphExecutionStrategy strategy) {
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

KernelActorPtr GraphScheduler::GenerateRpcActor(const CNodePtr &kernel, const DeviceContext *device_context,
                                                GraphExecutionStrategy strategy,
                                                const std::set<size_t> &ref_input_indexes,
                                                const std::set<size_t> &ref_output_indexes) {
  MS_EXCEPTION_IF_NULL(kernel);
  MS_EXCEPTION_IF_NULL(device_context);
#ifdef ENABLE_RPC_ACTOR
  MS_EXCEPTION_IF_NULL(rpc_node_scheduler_);
  if (common::AnfAlgo::GetCNodeName(kernel) == kRpcSendOpName) {
    auto send_actor =
      std::make_shared<SendActor>(kernel->fullname_with_scope(), kernel, device_context, memory_manager_aid_,
                                  debug_aid_, recorder_aid_, strategy, ref_input_indexes, ref_output_indexes);
    MS_EXCEPTION_IF_NULL(send_actor);
    rpc_node_scheduler_->InsertSendActor(send_actor);
    return send_actor;
  } else if (common::AnfAlgo::GetCNodeName(kernel) == kRpcRecvOpName) {
    auto recv_actor =
      std::make_shared<RecvActor>(kernel->fullname_with_scope(), kernel, device_context, memory_manager_aid_,
                                  debug_aid_, recorder_aid_, strategy, ref_input_indexes, ref_output_indexes);
    MS_EXCEPTION_IF_NULL(recv_actor);
    rpc_node_scheduler_->InsertRecvActor(recv_actor);
    return recv_actor;
  } else {
    MS_LOG(EXCEPTION) << "Kernel " << kernel->fullname_with_scope() << " is not an rpc kernel.";
  }
#endif
  return nullptr;
}

void GraphScheduler::LinkDataArrowInSinkMode(const KernelGraphPtr &graph, const GraphCompilerInfo &graph_compiler_info,
                                             std::vector<AbstractActor *> *const auto_monad_actors) {
  MS_EXCEPTION_IF_NULL(graph);
  // The data arrow linking is taken over by the control flow.
  if (graph_compiler_info.control_node_parser_ != nullptr &&
      graph_compiler_info.control_node_parser_->IsControlFlowDataArrow(graph, nullptr)) {
    return;
  }

  auto to_actor_name = graph->ToString() + kSuperKernelActorNameSuffix;
  auto to_actor = FetchActor(to_actor_name);
  MS_EXCEPTION_IF_NULL(to_actor);

  auto &input_nodes = graph->input_nodes();
  for (size_t node_index = 0; node_index < input_nodes.size(); ++node_index) {
    auto &input_node = input_nodes[node_index];
    MS_EXCEPTION_IF_NULL(input_node);
    if (HasAbstractMonad(input_node)) {
      MS_LOG(INFO) << "The graph:" << graph->graph_id()
                   << " has abstract monad input node:" << input_node->DebugString() << ", input index:" << node_index;
      LinkControlArrowByAutoMonad(to_actor, input_node, graph);
      continue;  // No data arrow for monad input.
    }

    UpdateRefCount(input_node, 0, true);
    KernelWithIndex from_kernel_with_output_idx = std::make_pair(input_node, 0);
    KernelWithIndex to_kernel_with_input_idx = std::make_pair(input_node, node_index);
    // The gather of linking data arrows of kernel by the different from kernel type.
    LinkDataArrow(to_actor, graph_compiler_info, graph, from_kernel_with_output_idx, to_kernel_with_input_idx);
  }

  std::vector<CNodePtr> auto_monad_kernels;
  // Foreach the execution order to get the auto monad kernels.
  auto &execution_order = graph->execution_order();
  (void)std::for_each(execution_order.begin(), execution_order.end(), [&](const CNodePtr &kernel) {
    for (size_t i = 0; i < common::AnfAlgo::GetInputNum(kernel); ++i) {
      auto input_node = common::AnfAlgo::GetInputNode(kernel, i);
      if (HasAbstractMonad(input_node)) {
        (void)auto_monad_kernels.emplace_back(kernel);
        continue;
      }
    }
  });
  // Foreach auto monad kernels to get the auto monad device tensor stores.
  (void)std::for_each(auto_monad_kernels.begin(), auto_monad_kernels.end(), [&](const CNodePtr &kernel) {
    for (size_t i = 0; i < common::AnfAlgo::GetInputTensorNum(kernel); ++i) {
      KernelWithIndex from_kernel_with_output_idx = common::AnfAlgo::GetPrevNodeOutput(kernel, i, false);
      auto front_node = FetchFrontNodeByBackendNode(from_kernel_with_output_idx.first, graph);
      if (IsPersistentDeviceTensor(front_node)) {
        (void)to_actor->auto_monad_device_tensor_stores_.insert(front_node);
      }
    }
  });
  if (to_actor->auto_monad_device_tensor_stores_.size() > 0) {
    (void)auto_monad_actors->emplace_back(to_actor);
  }
}

void GraphScheduler::LinkDataArrowInNonSinkMode(const KernelGraphPtr &graph,
                                                const GraphCompilerInfo &graph_compiler_info,
                                                std::vector<AbstractActor *> *const auto_monad_actors,
                                                std::vector<CNodePtr> *const communication_nodes) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(auto_monad_actors);
  MS_EXCEPTION_IF_NULL(communication_nodes);

  const mindspore::HashSet<PrimitivePtr, PrimitiveHasher, PrimitiveEqual> auto_monad_prims = {
    prim::kPrimDepend, prim::kPrimUpdateState, prim::kPrimLoad};
  auto &execution_order = graph->execution_order();
  // Foreach the execution order to link the actors.
  for (const auto &kernel : execution_order) {
    MS_EXCEPTION_IF_NULL(kernel);
    if (common::AnfAlgo::IsCommunicationOp(kernel)) {
      (void)communication_nodes->emplace_back(kernel);
    }
    if (IsSkippedKernelActor(kernel) || (!IsKernelActor(kernel, graph_compiler_info.strategy_))) {
      continue;
    }
    const auto &kernel_actor = FetchActor(kernel->fullname_with_scope());
    MS_EXCEPTION_IF_NULL(kernel_actor);

    for (size_t i = 0; i < common::AnfAlgo::GetInputNum(kernel); ++i) {
      auto input_node = common::AnfAlgo::GetInputNode(kernel, i);
      // Link the control arrows of kernel actor by the auto monad, the inputs include monad node.
      if (IsOneOfPrimitiveCNode(input_node, auto_monad_prims) || HasAbstractMonad(input_node)) {
        LinkControlArrowByAutoMonad(kernel_actor, input_node, graph, graph_compiler_info.control_node_parser_);
      }
      if (HasAbstractMonad(input_node)) {
        (void)auto_monad_actors->emplace_back(kernel_actor);
        continue;  // No data arrow for monad input.
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
      MS_LOG(WARNING) << "Invalid from node:" << from_kernel->fullname_with_scope() << ", type:" << kernel_type;
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
  auto device_tensor_store_key = FetchFrontNodeByBackendNode(from_kernel, graph);
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
  auto front_output_with_index = graph->GetFrontNodeByInternalParameter(internal_parameter);
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
      MS_LOG(EXCEPTION) << "Can't find actor by front node:" << common::AnfAlgo::GetNodeDebugString(front_output_node)
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
    real_from_kernel_with_output_idx = actor_pair.second;
    kernel_type = actor_pair.first->type_;
  }

  if (kKernelTypeToLinkFunc.count(kernel_type) == 0) {
    MS_LOG(EXCEPTION) << "Invalid internal parameter:" << internal_parameter->DebugString() << ", type:" << kernel_type;
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

  // Get the position of from kernel in the data source actor.
  auto position = from_actor->FetchNodePosition(from_kernel);
  if ((from_actor->device_contexts_.size() <= position) || (to_actor->device_contexts_.size() <= 0)) {
    MS_LOG(EXCEPTION) << "The device contexts size is wrong.";
  }

  if (IsNeedInsertCopyActor(from_actor->device_contexts_[position], to_actor->device_contexts_[0])) {
    LinkDataArrowForCopyActor(from_actor, to_actor, from_kernel_with_output_idx, to_kernel_with_input_idx);
  } else {
    AddDataArrow(from_actor, to_actor, from_kernel, from_output_index, to_input_index);
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
  auto position = host_ds_actor->FetchNodePosition(from_kernel_with_output_idx.first);
  real_from_kernel_with_output_idx.first = host_ds_actor->FetchNode(position);

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
    LinkControlArrowBySkippedNode(to_actor, from_kernel);

    MS_EXCEPTION_IF_NULL(to_kernel_with_input_idx.first);
    MS_LOG(INFO) << "Link data arrow for inplace node, aggregate node: "
                 << to_kernel_with_input_idx.first->fullname_with_scope()
                 << ", aggregate input index: " << to_kernel_with_input_idx.second
                 << ", skip node: " << from_kernel->fullname_with_scope()
                 << ", real node: " << real_from_kernel_with_output_idx.first->fullname_with_scope();
    real_from_actor = FetchActor(real_from_kernel_with_output_idx.first->fullname_with_scope());
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

  std::string name = "copy_from:" + from_actor->GetAID().Name() + "_node:" + from_kernel->fullname_with_scope() +
                     "_output_index:" + std::to_string(from_kernel_with_output_idx.second);
  CopyActor *copy_actor = dynamic_cast<CopyActor *>(FetchActor(name));
  // Link between from actor and copy actor.
  if (copy_actor == nullptr) {
    // Create the copy actor.
    auto copy_actor_shared_ptr = std::make_shared<CopyActor>(name, memory_manager_aid_);
    (void)copy_actors_.emplace_back(copy_actor_shared_ptr);
    copy_actor = copy_actor_shared_ptr.get();
    MS_EXCEPTION_IF_NULL(copy_actor);
    InsertActor(copy_actor);

    // Set the member device_contexts_ of the copy actor.
    auto position = from_actor->FetchNodePosition(from_kernel);
    if ((from_actor->device_contexts_.size() <= position) || (to_actor->device_contexts_.size() <= 0)) {
      MS_LOG(EXCEPTION) << "The device contexts size is wrong.";
    }
    auto from_device_context = from_actor->device_contexts_[position];
    auto to_device_context = to_actor->device_contexts_[0];
    MS_EXCEPTION_IF_NULL(from_device_context);
    MS_EXCEPTION_IF_NULL(to_device_context);
    (void)copy_actor->device_contexts_.emplace_back(from_device_context);
    (void)copy_actor->device_contexts_.emplace_back(to_device_context);

    // Set the member output_ of the copy actor.
    if (to_actor->type_ == KernelTransformType::kSuperKernelActor) {
      copy_actor->output_ = AnfAlgo::GetMutableOutputAddr(to_kernel_with_input_idx.first, 0, false);
    } else {
      copy_actor->output_ =
        AnfAlgo::GetPrevNodeMutableOutputAddr(to_kernel_with_input_idx.first, to_kernel_with_input_idx.second, false);
    }
    MS_EXCEPTION_IF_NULL(copy_actor->output_);
    if (copy_actor->output_->DeviceType() != to_device_context->GetDeviceAddressType()) {
      MS_LOG(EXCEPTION) << "The device type is not equal, output device type:" << copy_actor->output_->DeviceType()
                        << ", to device context type:" << to_device_context->GetDeviceAddressType();
    }

    // Link between from actor and copy actor.
    AddDataArrow(from_actor, copy_actor, from_kernel, from_kernel_with_output_idx.second, 0);
  }

  // If the copy actor already exists, only need link between copy actor and to actor.
  AddDataArrow(copy_actor, to_actor, nullptr, 0, to_kernel_with_input_idx.second);
  if (to_actor->type_ == KernelTransformType::kSuperKernelActor) {
    UpdateRefCount(copy_actor->output_.get(), true);
  } else {
    UpdateRefCount(copy_actor->output_.get(), false);
  }
}

void GraphScheduler::LinkControlArrowByAutoMonad(AbstractActor *to_actor, const AnfNodePtr &from_node,
                                                 const KernelGraphPtr &graph, const ControlNodeParserPtr &parser) {
  MS_EXCEPTION_IF_NULL(to_actor);
  MS_EXCEPTION_IF_NULL(from_node);
  MS_EXCEPTION_IF_NULL(graph);
  // Find the real input node, include the monad node and make tuple node.
  const std::vector<PrimitivePtr> return_types = {prim::kPrimDepend, prim::kPrimUpdateState, prim::kPrimLoad,
                                                  prim::kPrimMakeTuple};
  const auto &input_kernel_with_output_idx =
    common::AnfAlgo::VisitKernelWithReturnType(from_node, 0, false, return_types);
  MS_EXCEPTION_IF_NULL(input_kernel_with_output_idx.first);
  auto input_anfnode = input_kernel_with_output_idx.first;
  CNodePtr input_cnode = nullptr;
  if (input_anfnode->isa<CNode>()) {
    input_cnode = input_anfnode->cast<CNodePtr>();
  }
  // Make tuple node needs to be expanded.
  if (common::AnfAlgo::CheckPrimitiveType(input_anfnode, prim::kPrimMakeTuple)) {
    MS_EXCEPTION_IF_NULL(input_cnode);
    for (size_t i = 1; i < input_cnode->inputs().size(); ++i) {
      LinkControlArrowByAutoMonad(to_actor, input_cnode->input(i), graph, parser);
    }
    return;
  }

  const mindspore::HashSet<PrimitivePtr, PrimitiveHasher, PrimitiveEqual> recursion_prims = {
    prim::kPrimDepend, prim::kPrimUpdateState, prim::kPrimLoad, prim::kPrimMakeTuple};
  // Get the real depend input by monad node which needs to link the control arrow.
  std::vector<AnfNodePtr> real_depend_inputs;
  if (common::AnfAlgo::CheckPrimitiveType(input_anfnode, prim::kPrimDepend) ||
      common::AnfAlgo::CheckPrimitiveType(input_anfnode, prim::kPrimLoad)) {
    MS_EXCEPTION_IF_NULL(input_cnode);
    real_depend_inputs.push_back(input_cnode->input(kDependAttachNodeIndex));
    // The real input may be this scene:  depend/load --> load/depend, so need add the control arrow for real input
    // node in this scene.
    if (IsOneOfPrimitiveCNode(input_cnode->input(kRealInputIndexInDepend), recursion_prims)) {
      real_depend_inputs.push_back(input_cnode->input(kRealInputIndexInDepend));
    }
  } else if (common::AnfAlgo::CheckPrimitiveType(input_anfnode, prim::kPrimUpdateState)) {
    MS_EXCEPTION_IF_NULL(input_cnode);
    for (size_t i = kUpdateStateRealInput; i < input_cnode->inputs().size(); ++i) {
      real_depend_inputs.push_back(input_cnode->input(i));
    }
  } else {
    real_depend_inputs.push_back(input_anfnode);
  }

  for (const auto &real_depend_input : real_depend_inputs) {
    auto real_depend_input_with_idx =
      common::AnfAlgo::VisitKernelWithReturnType(real_depend_input, 0, false, return_types);
    MS_EXCEPTION_IF_NULL(real_depend_input_with_idx.first);
    auto real_depend_kernel = real_depend_input_with_idx.first;
    // Update the real depend kernel in the subgraphs connecting scene.
    if (IsInternalParameter(real_depend_kernel, graph)) {
      auto front_output_with_index = graph->GetFrontNodeByInternalParameter(real_depend_kernel);
      MS_EXCEPTION_IF_NULL(front_output_with_index.first);
      if (graph_output_to_actor_.count(front_output_with_index) == 0) {
        if (common::AnfAlgo::IsCallNode(front_output_with_index.first)) {
          continue;
        }
        MS_LOG(EXCEPTION) << "Can't find graph output by front node:" << front_output_with_index.first->DebugString();
      }

      if (parser != nullptr && parser->IsInited() &&
          (!parser->IsSameKernelGraphGroup(front_output_with_index.first, graph))) {
        MS_LOG(DEBUG) << "Skip in control flow from node:" << front_output_with_index.first->DebugString()
                      << " is not in the graph:" << graph->ToString();
        continue;
      }
      real_depend_kernel = graph_output_to_actor_[front_output_with_index].second.first;
      MS_EXCEPTION_IF_NULL(real_depend_kernel);
      MS_LOG(INFO) << "The graph " << graph->graph_id() << " link control arrow by auto monad from internal parameter: "
                   << real_depend_input_with_idx.first->DebugString()
                   << ", front output node: " << front_output_with_index.first->fullname_with_scope()
                   << ", backend output node: " << real_depend_kernel->fullname_with_scope();
      auto from_actor = graph_output_to_actor_[front_output_with_index].first;
      if (from_actor != nullptr) {
        MS_LOG(INFO) << "Link control arrow by auto monad from actor:  " << from_actor->GetAID().Name()
                     << ", to actor: " << to_actor->GetAID().Name() << " for the graph: " << graph->graph_id();
        AddControlArrow(from_actor, to_actor);
        continue;
      }
    }

    // The monad node and make tuple node need recursion.
    if (IsOneOfPrimitiveCNode(real_depend_kernel, recursion_prims)) {
      LinkControlArrowByAutoMonad(to_actor, real_depend_kernel, graph, parser);
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
    AddControlArrow(from_actor, to_actor);
  }
}

void GraphScheduler::LinkControlArrowBySkippedNode(AbstractActor *to_actor, const AnfNodePtr &skipped_node) {
  MS_EXCEPTION_IF_NULL(to_actor);
  MS_EXCEPTION_IF_NULL(skipped_node);

  // Link the control arrow from all the inputs of skipped node to the user of skipped node.
  auto input_num = common::AnfAlgo::GetInputTensorNum(skipped_node);
  for (size_t i = 0; i < input_num; ++i) {
    auto kernel_with_index = common::AnfAlgo::GetPrevNodeOutput(skipped_node, i, false);
    MS_EXCEPTION_IF_NULL(kernel_with_index.first);
    auto from_actor = FetchActor(kernel_with_index.first->fullname_with_scope());
    MS_EXCEPTION_IF_NULL(from_actor);
    MS_LOG(INFO) << "Link control arrow by skipped node: " << skipped_node->fullname_with_scope()
                 << ", from actor: " << from_actor->GetAID().Name() << ", to actor: " << to_actor->GetAID().Name();
    AddControlArrow(from_actor, to_actor);
  }
}

void GraphScheduler::LinkControlArrowBySendRecvNodes(const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  for (auto &from_iter : graph->allreduce_from_send_recv_pairs()) {
    auto to_allreduce_node = from_iter.first;
    auto from_send_node = from_iter.second.first;
    auto from_recv_node = from_iter.second.second;
    MS_EXCEPTION_IF_NULL(to_allreduce_node);
    MS_EXCEPTION_IF_NULL(from_send_node);
    MS_EXCEPTION_IF_NULL(from_recv_node);
    MS_LOG(INFO) << "Link control arrow for to_allreduce_node: " << to_allreduce_node->fullname_with_scope();
    auto to_allreduce_actor = FetchActor(to_allreduce_node->fullname_with_scope());
    auto from_send_actor = FetchActor(from_send_node->fullname_with_scope());
    auto from_recv_actor = FetchActor(from_recv_node->fullname_with_scope());
    MS_EXCEPTION_IF_NULL(to_allreduce_actor);
    MS_EXCEPTION_IF_NULL(from_send_actor);
    MS_EXCEPTION_IF_NULL(from_recv_actor);

    // inputs of to_allreduce_actor  --> from_send_actor
    for (auto &input_aid : to_allreduce_actor->input_data_arrow_aids_) {
      auto input_actor = dynamic_cast<KernelActor *>(FetchActor(input_aid.Name()));
      if (input_actor != nullptr) {
        AddControlArrow(input_actor, from_send_actor);
      }
    }
    // from_send_actor --> from_recv_actor
    AddControlArrow(from_send_actor, from_recv_actor);
    // from_recv_actor --> to_allreduce_actor
    AddControlArrow(from_recv_actor, to_allreduce_actor);
  }

  for (auto &to_iter : graph->allreduce_to_send_recv_pairs()) {
    auto from_allreduce_node = to_iter.first;
    auto to_send_node = to_iter.second.first;
    auto to_recv_node = to_iter.second.second;
    MS_EXCEPTION_IF_NULL(from_allreduce_node);
    MS_EXCEPTION_IF_NULL(to_send_node);
    MS_EXCEPTION_IF_NULL(to_recv_node);
    MS_LOG(INFO) << "Link control arrow for from_allreduce_node: " << from_allreduce_node->fullname_with_scope();
    auto from_allreduce_actor = FetchActor(from_allreduce_node->fullname_with_scope());
    auto to_send_actor = FetchActor(to_send_node->fullname_with_scope());
    auto to_recv_actor = dynamic_cast<KernelActor *>(FetchActor(to_recv_node->fullname_with_scope()));
    MS_EXCEPTION_IF_NULL(from_allreduce_actor);
    MS_EXCEPTION_IF_NULL(to_send_actor);
    MS_EXCEPTION_IF_NULL(to_recv_actor);

    // from_allreduce_actor  --> to_send_actor
    AddControlArrow(from_allreduce_actor, to_send_actor);
    // to_send_actor --> to_recv_actor
    AddControlArrow(to_send_actor, to_recv_actor);
    // to_recv_actor --> outputs of from_allreduce_actor
    for (auto &output_data_arrow : from_allreduce_actor->output_data_arrows_) {
      auto output_actor = FetchActor(output_data_arrow->to_op_id_.Name());
      if (output_actor != nullptr) {
        AddControlArrow(to_recv_actor, output_actor);
      }
    }

    // In the scene of allreduce op and computing op parallel multi stream, the input memory of allreduce can be
    // reused only when the recv node runs finished, which is expressed by the reference count increased.
    for (size_t i = 0; i < common::AnfAlgo::GetInputTensorNum(from_allreduce_node); ++i) {
      auto device_tensor = AnfAlgo::GetPrevNodeMutableOutputAddr(from_allreduce_node, i, false);
      MS_EXCEPTION_IF_NULL(device_tensor);
      UpdateRefCount(device_tensor.get());
      (void)to_recv_actor->external_reference_tensors_.emplace_back(device_tensor.get());
    }
  }
}

void GraphScheduler::LinkGlobalControlArrow(ActorSet *const actor_set,
                                            const GroupNameToCommuNodes &communication_node_groups,
                                            const std::vector<AbstractActor *> &auto_monad_actors,
                                            const GraphCompilerInfo &graph_compiler_info) {
  MS_EXCEPTION_IF_NULL(actor_set);
  for (const auto &communication_nodes : communication_node_groups) {
    // Link the control arrows by the communication nodes to ensure communication nodes running order.
    LinkControlArrowByCommunicationNode(communication_nodes.second, graph_compiler_info);
  }

  // Auto monad actor may modify the device tensor store.
  LinkDeviceTensorStoreForAutoMonadActor(auto_monad_actors);

  // BuildNoInputKernelActor depends on whether kernel actors have input, so must be behind the link of kernel actors.
  actor_set->no_input_kernel_actors_ = BuildNoInputKernelActor(actor_set, graph_compiler_info.strategy_);

  // Link the control arrows of data prepare actor, which depends on the no input kernel actors.
  if ((graph_compiler_info.strategy_ == GraphExecutionStrategy::kPipeline) || (!IsSingleOpActorSet(actor_set))) {
    LinkControlArrowForDataPrepareActor(actor_set->data_prepare_actor_.get(), actor_set,
                                        graph_compiler_info.control_node_parser_);
  }
  // Link control arrows for custom actor
  LinkControlArrowForCustomActor(actor_set, graph_compiler_info);

  LinkControlArrowForLoopCountActor(actor_set->loop_count_actor_.get(), actor_set,
                                    graph_compiler_info.control_node_parser_);
}

void GraphScheduler::LinkControlArrowForCustomActor(ActorSet *const actor_set,
                                                    const GraphCompilerInfo &graph_compiler_info) {
  constexpr size_t kDependFromIdx = 2;
  constexpr size_t kDependToIdx = 1;
  MS_EXCEPTION_IF_NULL(actor_set);
  MS_EXCEPTION_IF_NULL(actor_set->data_prepare_actor_);
  // prepare for kernel => actor map
  HashMap<AnfNodePtr, AbstractActorPtr> kernel_to_actors = {};
  HashSet<AbstractActorPtr> no_depend_custom_actors = {};
  for (const auto &actor : actor_set->custom_actors_) {
    MS_EXCEPTION_IF_NULL(actor);
    auto kernel = actor->kernel().lock();
    MS_EXCEPTION_IF_NULL(kernel);
    kernel_to_actors.emplace(kernel, actor);
    no_depend_custom_actors.insert(actor);
  }
  for (const auto &actor : actor_set->kernel_actors_) {
    MS_EXCEPTION_IF_NULL(actor);
    auto kernel = actor->kernel();
    MS_EXCEPTION_IF_NULL(kernel);
    kernel_to_actors.emplace(kernel, actor);
  }
  for (const auto &actor : actor_set->data_source_actors_) {
    MS_EXCEPTION_IF_NULL(actor);
    auto device_data_source_actor = dynamic_cast<DeviceQueueDataSourceActor *>(actor.get());
    if (device_data_source_actor != nullptr) {
      auto kernel = device_data_source_actor->data_kernel();
      MS_EXCEPTION_IF_NULL(kernel);
      if (common::AnfAlgo::GetCNodeName(kernel) == kGetNextOpName) {
        kernel_to_actors.emplace(kernel, actor);
      }
    }
  }
  // find depend(custom, custom)
  for (size_t i = 0; i < graph_compiler_info.graphs_.size(); ++i) {
    const auto &graph = graph_compiler_info.graphs_[i];
    MS_EXCEPTION_IF_NULL(graph);
    if (graph->is_executing_sink()) {
      continue;
    }

    auto all_nodes = TopoSort(graph->get_return());
    for (const auto &node : all_nodes) {
      if (!IsPrimitiveCNode(node, prim::kPrimDepend)) {
        continue;
      }
      auto depend_cnode = node->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(depend_cnode);
      MS_EXCEPTION_IF_CHECK_FAIL(depend_cnode->size() > kDependFromIdx,
                                 "depend node " + depend_cnode->DebugString() + " input size " +
                                   std::to_string(depend_cnode->size()) + " is invalid.");
      MS_EXCEPTION_IF_NULL(depend_cnode->input(kDependFromIdx));
      MS_EXCEPTION_IF_NULL(depend_cnode->input(kDependToIdx));
      auto from_node = depend_cnode->input(kDependFromIdx);
      auto to_node = depend_cnode->input(kDependToIdx);
      if (!AnfUtils::IsCustomActorNode(from_node) && !AnfUtils::IsCustomActorNode(to_node)) {
        continue;
      }
      auto from_iter = kernel_to_actors.find(from_node);
      if (from_iter == kernel_to_actors.end()) {
        MS_LOG(INFO) << from_node->fullname_with_scope() << " is a CNode but cannot find Actor.";
        continue;
      }
      auto to_iter = kernel_to_actors.find(to_node);
      if (to_iter == kernel_to_actors.end()) {
        MS_LOG(INFO) << to_node->fullname_with_scope() << " is a CNode but cannot find Actor.";
        continue;
      }
      AddControlArrow(from_iter->second.get(), to_iter->second.get());
      no_depend_custom_actors.erase(std::dynamic_pointer_cast<CustomActor>(to_iter->second));
    }
  }

  for (const auto &custom_actor : no_depend_custom_actors) {
    AddControlArrow(actor_set->data_prepare_actor_.get(), custom_actor.get());
  }
}

void GraphScheduler::LinkControlArrowByCommunicationNode(const std::vector<CNodePtr> &communication_nodes,
                                                         const GraphCompilerInfo &graph_compiler_info) {
  const size_t kCommunicationNodesMinNum = 2;
  if (communication_nodes.size() < kCommunicationNodesMinNum) {
    return;
  }

  // Ensure communication node to execute orderly.
  for (size_t i = 1; i < communication_nodes.size(); ++i) {
    auto from_actor = FetchActor(communication_nodes[i - 1]->fullname_with_scope());
    auto to_actor = FetchActor(communication_nodes[i]->fullname_with_scope());
    MS_EXCEPTION_IF_NULL(from_actor);
    MS_EXCEPTION_IF_NULL(to_actor);
    AddControlArrow(from_actor, to_actor);
  }

  // Ensure all actors execute orderly to optimize the execution performance in the multi device scenario currently.
  // Using the multi stream to optimize the performance in the future.
  for (auto &graph : graph_compiler_info.graphs_) {
    MS_EXCEPTION_IF_NULL(graph);
    auto &execution_order = graph->execution_order();
    for (size_t i = 1; i < execution_order.size(); ++i) {
      auto from_actor = FetchActor(execution_order[i - 1]->fullname_with_scope());
      auto to_actor = FetchActor(execution_order[i]->fullname_with_scope());
      if ((from_actor != nullptr) && (to_actor != nullptr)) {
        AddControlArrow(from_actor, to_actor);
      }
    }
  }
}

void GraphScheduler::LinkControlArrowForDataPrepareActor(DataPrepareActor *data_prepare_actor,
                                                         const ActorSet *actor_set,
                                                         const ControlNodeParserPtr &parser) {
  MS_EXCEPTION_IF_NULL(data_prepare_actor);
  MS_EXCEPTION_IF_NULL(actor_set);
  MS_EXCEPTION_IF_NULL(parser);

  // Data prepare actor --> data source actor.
  for (auto &data_source_actor : actor_set->data_source_actors_) {
    MS_EXCEPTION_IF_NULL(data_source_actor);
    AddControlArrow(data_prepare_actor, data_source_actor.get());
  }

  // In control flow, control arrow of no input kernel actor needs to be connected to the corresponding entrance actor.
  if (!parser->IsInited()) {
    // Data prepare actor --> no input kernel actor.
    for (auto &no_input_kernel_actor : actor_set->no_input_kernel_actors_) {
      MS_EXCEPTION_IF_NULL(no_input_kernel_actor);
      AddControlArrow(data_prepare_actor, no_input_kernel_actor.get());
    }
  }

  // Data prepare actor --> loop count actor.
  if ((actor_set->data_source_actors_.size() + actor_set->no_input_kernel_actors_.size() == 0) &&
      (actor_set->loop_count_actor_ != nullptr)) {
    AddControlArrow(data_prepare_actor, actor_set->loop_count_actor_.get());
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

  // Collect the actors which have no output.
  std::vector<AbstractActor *> no_output_actors;
  for (auto &super_actor : actor_set->super_kernel_actors_) {
    if ((super_actor->output_data_arrows_.size() == 0) && (super_actor->output_control_arrows_.size() == 0)) {
      (void)no_output_actors.emplace_back(super_actor.get());
    }
  }
  for (auto &kernel_actor : actor_set->kernel_actors_) {
    // The no output kernel control side in subgraph needs to be connected to the corresponding output switch actor.
    if ((kernel_actor->output_data_arrows_.size() == 0) && (kernel_actor->output_control_arrows_.size() == 0)) {
      (void)no_output_actors.emplace_back(kernel_actor.get());
    }
  }
  for (auto &data_actor : actor_set->data_source_actors_) {
    if ((data_actor->output_data_arrows_.size() == 0) && (data_actor->output_control_arrows_.size() == 0)) {
      (void)no_output_actors.emplace_back(data_actor.get());
    }
  }
  for (auto &copy_actor : copy_actors_) {
    if ((copy_actor->output_data_arrows_.size() == 0) && (copy_actor->output_control_arrows_.size() == 0)) {
      (void)no_output_actors.emplace_back(copy_actor.get());
    }
  }
  for (auto &custom_actor : actor_set->custom_actors_) {
    if ((custom_actor->output_data_arrows_.size() == 0) && (custom_actor->output_control_arrows_.size() == 0)) {
      (void)no_output_actors.emplace_back(custom_actor.get());
    }
  }

  // No output actor --> loop count actor.
  // In control flow scenario, no output actor needs to be connected to the corresponding exit actor, not loop count.
  if (!parser->IsInited()) {
    for (auto &no_output_actor : no_output_actors) {
      AddControlArrow(no_output_actor, loop_count_actor);
    }
  }

  // Loop count actor --> output actor.
  AddControlArrow(loop_count_actor, actor_set->output_actor_.get());

  // Loop count actor --> data prepare actor.
  MS_EXCEPTION_IF_NULL(actor_set->data_prepare_actor_);
  loop_count_actor->data_prepare_aid_ = actor_set->data_prepare_actor_->GetAID();
}

void GraphScheduler::LinkOutputResultArrowForOutputActor(OutputActor *to_actor,
                                                         const GraphCompilerInfo &graph_compiler_info) {
  if (graph_compiler_info.strategy_ == GraphExecutionStrategy::kStep ||
      (graph_compiler_info.control_node_parser_ != nullptr && graph_compiler_info.control_node_parser_->IsInited())) {
    // In control flow, the exit actor of the root graph sends output data to the output actor.
    return;
  }
  MS_EXCEPTION_IF_NULL(to_actor);

  for (size_t i = 0; i < graph_compiler_info.graphs_.size(); ++i) {
    const auto &graph = graph_compiler_info.graphs_[i];
    MS_EXCEPTION_IF_NULL(graph);
    auto outputs = common::AnfAlgo::GetAllOutputWithIndex(graph->output());
    std::set<std::vector<size_t>> unique_output_positions;
    std::set<KernelWithIndex> unique_outputs;
    for (const auto &output : outputs) {
      if (IsInternalParameter(output.first, graph)) {
        MS_LOG(INFO) << "Ignore the internal parameter node:" << output.first->DebugString();
        continue;
      }
      (void)unique_outputs.insert(output);
    }
    for (const auto &output_with_index : unique_outputs) {
      MS_EXCEPTION_IF_NULL(output_with_index.first);
      auto origin_output_with_index = FetchFrontNodeWithIndexByGraphOutput(output_with_index, graph);
      const auto &iter = graph_compiler_info.origin_outputs_order_.find(origin_output_with_index);
      if (iter == graph_compiler_info.origin_outputs_order_.end()) {
        continue;
      }

      // Skip duplicate position.
      if (unique_output_positions.count(iter->second) > 0) {
        continue;
      }
      (void)unique_output_positions.insert(iter->second);
      for (auto &output_position : iter->second) {
        if (output_position >= to_actor->device_contexts_.size()) {
          MS_LOG(EXCEPTION) << "The output position is out of range.";
        }
        to_actor->device_contexts_[output_position] = graph_compiler_info.device_contexts_[i];

        // The graph output is from device tensor store.
        if (IsPersistentDeviceTensor(output_with_index.first)) {
          (void)to_actor->device_tensor_store_keys_.emplace_back(output_position, output_with_index.first);
          if (!AnfAlgo::OutputAddrExist(output_with_index.first, 0, false)) {
            MS_EXCEPTION_IF_NULL(output_with_index.first);
            MS_LOG(WARNING) << output_with_index.first->DebugString() << " device address not exit";
            continue;
          }
          // In the scenario where the ValueTuple is expanded, the output_with_index.second may be incorrect, so use 0
          // as output_idx directly.
          auto device_tensor = AnfAlgo::GetMutableOutputAddr(output_with_index.first, 0, false);
          MS_EXCEPTION_IF_NULL(device_tensor);
          // The output actor need use the relevant information of node to create output tensor.
          device_tensor->SetNodeIndex(output_with_index.first, 0);
          continue;
        }

        // The graph output is from kernel actor or data source actor.
        auto kernel_type = FetchKernelTransformType(
          output_with_index.first, graph, graph_compiler_info.origin_parameters_order_, graph_compiler_info.strategy_);
        auto from_actor = FetchActor(kernel_type, graph_compiler_info.name_, output_with_index.first, graph);
        if (from_actor == nullptr) {
          continue;
        }

        auto real_from_kernel = output_with_index.first;
        // Update the real node in the host data source actor.
        if (kernel_type == KernelTransformType::kHostDataSourceActor) {
          auto host_queue_ds_actor = dynamic_cast<HostQueueDataSourceActor *>(from_actor);
          MS_EXCEPTION_IF_NULL(host_queue_ds_actor);
          auto position = host_queue_ds_actor->FetchNodePosition(output_with_index.first);
          real_from_kernel = host_queue_ds_actor->FetchNode(position);
          UpdateRefCount(output_with_index.first, output_with_index.second, true);
        }
        AddResultArrow(from_actor, to_actor, real_from_kernel, output_with_index.second, output_position);
      }
    }
  }
}

void GraphScheduler::LinkDeviceTensorStoreForAutoMonadActor(const std::vector<AbstractActor *> &auto_monad_actors) {
  const size_t kNeedUpdateDeviceTensorStoreNum = 2;
  for (auto &auto_monad_actor : auto_monad_actors) {
    MS_EXCEPTION_IF_NULL(auto_monad_actor);
    for (auto &device_tensor_store_key : auto_monad_actor->device_tensor_store_keys_) {
      auto device_tensors = DeviceTensorStore::GetInstance().Fetch(device_tensor_store_key.second.get());
      if (device_tensors.size() < kNeedUpdateDeviceTensorStoreNum) {
        continue;
      }
      // Find the device tensor store that needs to be processed accurately.
      if ((auto_monad_actor->type_ == KernelTransformType::kSuperKernelActor) &&
          (auto_monad_actor->auto_monad_device_tensor_stores_.find(device_tensor_store_key.second) ==
           auto_monad_actor->auto_monad_device_tensor_stores_.end())) {
        continue;
      }

      // Create the copy actor.
      std::string name = "copy_from:" + auto_monad_actor->GetAID().Name() +
                         "_device_tensor_store:" + device_tensor_store_key.second->fullname_with_scope();
      if (FetchActor(name) != nullptr) {
        continue;
      }
      auto copy_actor = std::make_shared<CopyActor>(name, memory_manager_aid_);
      MS_EXCEPTION_IF_NULL(copy_actor);
      (void)copy_actors_.emplace_back(copy_actor);
      InsertActor(copy_actor.get());

      // Set the member of the copy actor.
      (void)copy_actor->device_tensor_store_keys_.emplace_back(0, device_tensor_store_key.second);
      auto input_device_context = auto_monad_actor->device_contexts_[0];
      (void)copy_actor->device_contexts_.emplace_back(input_device_context);
      auto another_device_tensor = (device_tensors[0]->DeviceType() == input_device_context->GetDeviceAddressType())
                                     ? device_tensors[1]
                                     : device_tensors[0];
      MS_EXCEPTION_IF_NULL(another_device_tensor);
      auto another_device_type = another_device_tensor->DeviceType();
      const auto &another_device_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(
        {device::kDeviceTypeToName.at(another_device_type), input_device_context->device_context_key().device_id_});
      MS_EXCEPTION_IF_NULL(another_device_context);
      (void)copy_actor->device_contexts_.emplace_back(another_device_context);

      MS_LOG(INFO) << "The auto monad actor: " << auto_monad_actor->GetAID().Name()
                   << "has control arrows number:" << auto_monad_actor->output_control_arrows_.size();
      // Link from copy actor to auto monad actor users.
      for (auto &output_contorl : auto_monad_actor->output_control_arrows_) {
        (void)copy_actor->output_control_arrows_.emplace_back(output_contorl);
      }
      // Move the control arrows from auto monad actor to auto monad actor users.
      auto_monad_actor->output_control_arrows_.clear();

      // Link from auto monad actor to copy actor.
      AddControlArrow(auto_monad_actor, copy_actor.get());
    }
  }
}

void GraphScheduler::AddDeviceTensorStore(const AnfNode *anf_node, const DeviceTensorPtr &device_tensor) {
  MS_EXCEPTION_IF_NULL(device_tensor);
  DeviceTensorStore::GetInstance().Insert(const_cast<AnfNode *>(anf_node), device_tensor);
  UpdateRefCount(device_tensor.get(), true);
}

void GraphScheduler::AddDataArrow(AbstractActor *const from_actor, AbstractActor *const to_actor,
                                  const AnfNodePtr &from_kernel, size_t from_output_index, size_t to_input_index) {
  MS_EXCEPTION_IF_NULL(from_actor);
  MS_EXCEPTION_IF_NULL(to_actor);

  auto data_arrow = std::make_shared<DataArrow>(from_output_index, to_actor->GetAID(), to_input_index);
  (void)from_actor->output_data_arrows_.emplace_back(data_arrow);
  (void)from_actor->output_data_nodes_.emplace_back(from_kernel);
  to_actor->input_datas_num_++;
  (void)to_actor->input_data_arrow_aids_.emplace_back(from_actor->GetAID());

  if (from_kernel == nullptr) {
    return;
  }
  // Update the reference count of from_kernel.
  // The device address of super kernel actor can't be changed, so set the max reference count.
  if ((from_actor->type_ == KernelTransformType::kSuperKernelActor) ||
      (to_actor->type_ == KernelTransformType::kSuperKernelActor)) {
    UpdateRefCount(from_kernel, from_output_index, true);
  } else {
    UpdateRefCount(from_kernel, from_output_index, false);
  }
}

void GraphScheduler::AddResultArrow(AbstractActor *const from_actor, OutputActor *const to_actor,
                                    const AnfNodePtr &from_kernel, size_t from_output_index, size_t output_position) {
  MS_EXCEPTION_IF_NULL(from_actor);
  MS_EXCEPTION_IF_NULL(to_actor);
  MS_EXCEPTION_IF_NULL(from_kernel);

  auto result_arrow = std::make_shared<DataArrow>(from_output_index, to_actor->GetAID(), output_position);
  (void)from_actor->output_data_arrows_.insert(from_actor->output_data_arrows_.begin(), result_arrow);
  (void)from_actor->output_data_nodes_.insert(from_actor->output_data_nodes_.begin(), from_kernel);
  to_actor->input_datas_num_++;
  (void)to_actor->input_data_arrow_aids_.emplace_back(from_actor->GetAID());

  auto device_tensor = AnfAlgo::GetMutableOutputAddr(from_kernel, from_output_index, false);
  MS_EXCEPTION_IF_NULL(device_tensor);
  // The output actor need use the relevant information of node to create output tensor.
  device_tensor->SetNodeIndex(from_kernel, from_output_index);

  // The device tensor of graph out need be taken over by host tensor, so set the max reference count.
  UpdateRefCount(device_tensor.get(), true);
}

void GraphScheduler::AddControlArrow(AbstractActor *const from_actor, AbstractActor *const to_actor) {
  MS_EXCEPTION_IF_NULL(from_actor);
  MS_EXCEPTION_IF_NULL(to_actor);
  (void)from_actor->output_control_arrows_.emplace_back(to_actor->GetAID());
  to_actor->input_controls_num_++;
  (void)to_actor->input_control_arrow_aids_.emplace_back(from_actor->GetAID());
}

void GraphScheduler::CheckActorValid(const ActorSet *actor_set) const {
  MS_EXCEPTION_IF_NULL(actor_set);
  auto actors = CollectActors(actor_set);
  for (auto &actor : actors) {
    MS_EXCEPTION_IF_NULL(actor);
    if (actor->type_ >= KernelTransformType::kSwitchActor) {
      continue;
    }

    if ((actor->input_datas_num_ != actor->input_data_arrow_aids_.size()) ||
        (actor->input_controls_num_ != actor->input_control_arrow_aids_.size())) {
      MS_LOG(EXCEPTION) << "The input num of " << actor->GetAID().Name()
                        << " is wrong, expect data num: " << actor->input_datas_num_
                        << ", actual data num: " << actor->input_data_arrow_aids_.size()
                        << ", expect control num: " << actor->input_controls_num_
                        << ", actual control num: " << actor->input_control_arrow_aids_.size();
    }

    if ((actor->type_ != KernelTransformType::kOutputActor) && (actor->type_ != KernelTransformType::kCustomActor) &&
        (actor->output_data_arrows_.size() == 0) && (actor->output_control_arrows_.size() == 0)) {
      MS_LOG(EXCEPTION) << actor->GetAID().Name() << " has no user.";
    }
    if ((actor->type_ != KernelTransformType::kDataPrepareActor) &&
        (actor->type_ != KernelTransformType::kCustomActor) && (actor->input_datas_num_ == 0) &&
        (actor->input_controls_num_ == 0)) {
      MS_LOG(EXCEPTION) << actor->GetAID().Name() << " has no source.";
    }

    // Check the input of kernel actors and copy actors.
    if ((actor->type_ == KernelTransformType::kKernelActor) || (actor->type_ == KernelTransformType::kCopyActor)) {
      size_t expect_toal_input_num = 1;
      if (actor->type_ == KernelTransformType::kKernelActor) {
        auto kernel_actor = dynamic_cast<KernelActor *>(actor.get());
        MS_EXCEPTION_IF_NULL(kernel_actor);
        expect_toal_input_num = common::AnfAlgo::GetInputTensorNum(kernel_actor->kernel_);
      }
      auto input_data_num = actor->input_datas_num_;
      auto device_tensor_store_num = actor->device_tensor_store_keys_.size();
      if (input_data_num + device_tensor_store_num != expect_toal_input_num) {
        MS_LOG(EXCEPTION) << "The input building of " << actor->GetAID().Name()
                          << " is wrong, input data num: " << input_data_num
                          << ", device tensor store num: " << device_tensor_store_num
                          << ", total input num: " << expect_toal_input_num;
      }
    }
  }

  // Check the output actor.
  auto output_actor = actor_set->output_actor_;
  MS_EXCEPTION_IF_NULL(output_actor);
  if (output_actor->input_datas_num_ + output_actor->device_tensor_store_keys_.size() != output_actor->outputs_num_) {
    MS_LOG(EXCEPTION) << "The outputs num of output actor is wrong, the total outputs num: "
                      << output_actor->outputs_num_ << ", the input data arrows num: " << output_actor->input_datas_num_
                      << ", the device tensor store num: " << output_actor->device_tensor_store_keys_.size();
  }
  control_node_scheduler_.CheckActorValid(actor_set);
}

void GraphScheduler::PersistDeviceTensor(const GraphCompilerInfo &graph_compiler_info) {
  const auto &parser = graph_compiler_info.control_node_parser_;
  MS_EXCEPTION_IF_NULL(parser);
  for (size_t i = 0; i < graph_compiler_info.graphs_.size(); ++i) {
    const auto &graph = graph_compiler_info.graphs_[i];
    const auto &device_context = graph_compiler_info.device_contexts_[i];
    MS_EXCEPTION_IF_NULL(graph);
    MS_EXCEPTION_IF_NULL(device_context);

    for (auto &value_node : graph->graph_value_nodes()) {
      MS_EXCEPTION_IF_NULL(value_node);
      if (!AnfAlgo::OutputAddrExist(value_node, 0)) {
        MS_LOG(INFO) << "The device address is not exist: " << value_node->ToString();
        continue;
      }
      auto device_tensor = AnfAlgo::GetMutableOutputAddr(value_node, 0, false);
      const auto &front_node = FetchFrontNodeByBackendNode(value_node, graph);
      device_tensor->SetNodeIndex(value_node, 0);
      AddDeviceTensorStore(front_node.get(), device_tensor);
    }

    for (auto &input_node : graph->input_nodes()) {
      MS_EXCEPTION_IF_NULL(input_node);
      AnfNodePtr front_node = nullptr;
      if (IsInternalParameter(input_node, graph)) {
        auto front_output_with_index = graph->GetFrontNodeByInternalParameter(input_node);
        front_node = front_output_with_index.first;
      } else if (IsPersistentDeviceTensor(input_node)) {
        front_node = FetchFrontNodeByBackendNode(input_node, graph);
      }
      if (front_node == nullptr || (!parser->IsRootGraphPersistentDeviceTensor(front_node))) {
        continue;
      }

      auto device_tensor = AnfAlgo::GetMutableOutputAddr(input_node, 0, false);
      MS_EXCEPTION_IF_NULL(device_tensor);
      if (IsPersistentDeviceTensor(input_node)) {
        device_tensor->SetNodeIndex(input_node, 0);
        AddDeviceTensorStore(front_node.get(), device_tensor);
      }

      if (device_tensor->is_ptr_persisted()) {
        device_tensor->SetNodeIndex(input_node, 0);
        AddDeviceTensorStore(front_node.get(), device_tensor);
      }

      // If the device tensor store of this device type is not exist, then create the new device tensor of this type.
      if (DeviceTensorStore::GetInstance().Fetch(front_node.get(), device_context->GetDeviceAddressType()) == nullptr) {
        MS_LOG(WARNING) << "Fetch no device tensor store by:" << front_node->fullname_with_scope()
                        << ", type:" << device_context->GetDeviceAddressType();
        auto other_type_device_tensor = device_context->CreateDeviceAddress(
          nullptr, device_tensor->GetSize(), device_tensor->format(), device_tensor->type_id());
        other_type_device_tensor->SetNodeIndex(input_node, 0);
        other_type_device_tensor->set_from_persistent_mem(input_node->isa<Parameter>());
        AddDeviceTensorStore(front_node.get(), other_type_device_tensor);
      }
    }
  }
  PersistDeviceTensorForRootGraphControlNode(graph_compiler_info);
}

void GraphScheduler::PersistDeviceTensorForRootGraphControlNode(const GraphCompilerInfo &graph_compiler_info) {
  const auto &parser = graph_compiler_info.control_node_parser_;
  if (parser == nullptr || (!parser->IsInited())) {
    return;
  }

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
    const auto &backend_parameter_with_context =
      parser->FetchBackendParameterWithContextByFrontParameter({root_graph_parameter, 0});
    if (backend_parameter_with_context.first == nullptr) {
      MS_LOG(EXCEPTION) << "Cannot find backend node for weight parameter:" << root_graph_parameter->DebugString();
    }
    const auto &backend_node = backend_parameter_with_context.first;
    const auto &device_context = backend_parameter_with_context.second;
    MS_EXCEPTION_IF_NULL(backend_node);
    MS_EXCEPTION_IF_NULL(device_context);
    auto sub_device_tensor = AnfAlgo::GetMutableOutputAddr(backend_node, 0, false);
    MS_EXCEPTION_IF_NULL(sub_device_tensor);

    auto new_device_tensor = device_context->CreateDeviceAddress(
      nullptr, sub_device_tensor->GetSize(), sub_device_tensor->format(), sub_device_tensor->type_id());
    MS_EXCEPTION_IF_NULL(new_device_tensor);
    new_device_tensor->SetNodeIndex(backend_node, 0);
    new_device_tensor->set_is_ptr_persisted(sub_device_tensor->is_ptr_persisted());
    new_device_tensor->set_from_persistent_mem(true);
    AddDeviceTensorStore(root_graph_parameter.get(), new_device_tensor);
    MS_LOG(INFO) << "Add device tensor store by root graph parameter:" << root_graph_parameter->fullname_with_scope()
                 << ", backend node:" << backend_node->DebugString()
                 << ", type:" << device_context->GetDeviceAddressType();
  }
}

void GraphScheduler::DumpActor(const ActorSet *actor_set, const GraphCompilerInfo &graph_compiler_info) const {
  MS_EXCEPTION_IF_NULL(actor_set);
  const auto &context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  auto save_graphs = context_ptr->get_param<bool>(MS_CTX_SAVE_GRAPHS_FLAG);
  if (!save_graphs) {
    return;
  }

  // Get the saved actor set name.
  auto &kernel_graphs = graph_compiler_info.graphs_;
  MS_EXCEPTION_IF_NULL(kernel_graphs.front());
  auto first_graph_id = kernel_graphs.front()->graph_id();
  MS_EXCEPTION_IF_NULL(kernel_graphs.back());
  auto last_graph_id = kernel_graphs.back()->graph_id();
  std::string strategy = (graph_compiler_info.strategy_ == GraphExecutionStrategy::kPipeline) ? "pipeline" : "step";
  std::string save_name = "actor_set_" + strategy + "_kernel_graph_" + std::to_string(first_graph_id);
  if (last_graph_id != first_graph_id) {
    save_name = save_name + "-" + std::to_string(last_graph_id);
  }

  std::string filename = GetSaveGraphsPathName(save_name + ".ir");
  std::ofstream ofs(filename);
  if (!ofs.is_open()) {
    MS_LOG(ERROR) << "Open file [" << filename << "] failed!";
    return;
  }

  DumpDeviceTensorStore(graph_compiler_info, ofs);
  DumpDataPrepareActor(actor_set->data_prepare_actor_, ofs);
  DumpDSActors(actor_set->data_source_actors_, ofs);
  DumpKernelActors(actor_set->kernel_actors_, ofs);
  DumpSuperKernelActors(actor_set->super_kernel_actors_, ofs);
  // The on input kernel actors are taken over by control actor in the control flow scene.
  if ((graph_compiler_info.control_node_parser_ == nullptr) ||
      (!graph_compiler_info.control_node_parser_->IsInited())) {
    DumpNoInputKernelActors(actor_set->no_input_kernel_actors_, ofs);
  }
  DumpCopyActors(actor_set->copy_actors_, ofs);
  DumpLoopCountActor(actor_set->loop_count_actor_, ofs);
  DumpOutputActor(actor_set->output_actor_, ofs);
  DumpControlActors(actor_set->control_actors_, ofs);
  DumpCustomActors(actor_set->custom_actors_, ofs);
}

void GraphScheduler::DumpDeviceTensorStore(const GraphCompilerInfo &graph_compiler_info, std::ofstream &ofs) const {
  ofs << "[Device tensor stores]\n";

  for (const auto &graph : graph_compiler_info.graphs_) {
    MS_EXCEPTION_IF_NULL(graph);
    ofs << "\tgraph_id:" << graph->graph_id() << "\tis_executing_sink:" << graph->is_executing_sink()
        << "\tis_loop_count_sink:" << graph->is_loop_count_sink()
        << "\texecution_strategy:" << graph_compiler_info.strategy_ << "\n";

    for (auto &value_node : graph->graph_value_nodes()) {
      MS_EXCEPTION_IF_NULL(value_node);
      if (!AnfAlgo::OutputAddrExist(value_node, 0)) {
        continue;
      }
      const auto &front_node = FetchFrontNodeByBackendNode(value_node, graph);
      MS_EXCEPTION_IF_NULL(front_node);
      const auto device_tensors = DeviceTensorStore::GetInstance().Fetch(front_node.get());
      ofs << "\t\tdevice tensor key:" << front_node->fullname_with_scope() << "\tvalue size:" << device_tensors.size()
          << "\n";
      for (const auto &device_tensor : device_tensors) {
        MS_EXCEPTION_IF_NULL(device_tensor);
        ofs << "\t\t\tdevice tensor value:" << device_tensor << "\tptr:" << device_tensor->GetPtr()
            << "\tsize:" << device_tensor->GetSize() << "\toriginal_ref_count:" << device_tensor->original_ref_count()
            << "\tdynamic_ref_count:" << device_tensor->dynamic_ref_count()
            << "\tdevice_type:" << device_tensor->DeviceType()
            << "\tis_ptr_persisted:" << device_tensor->is_ptr_persisted() << "\n ";
      }
    }

    for (auto &input_node : graph->input_nodes()) {
      MS_EXCEPTION_IF_NULL(input_node);
      if (!IsPersistentDeviceTensor(input_node)) {
        continue;
      }
      const auto &sub_front_node = FetchFrontNodeByBackendNode(input_node, graph);
      // The sub front nodes share the device tensor store with the root front node.
      auto front_node = sub_front_node;
      if (graph_compiler_info.control_node_parser_ != nullptr) {
        front_node = graph_compiler_info.control_node_parser_->FetchRootGraphFrontNodeBySubFrontNode(sub_front_node);
      }
      const auto device_tensors = DeviceTensorStore::GetInstance().Fetch(front_node.get());
      MS_EXCEPTION_IF_NULL(front_node);
      ofs << "\t\tdevice tensor key:" << front_node->fullname_with_scope() << "\tvalue size:" << device_tensors.size()
          << "\n";
      for (const auto &device_tensor : device_tensors) {
        MS_EXCEPTION_IF_NULL(device_tensor);
        ofs << "\t\t\tdevice tensor value:" << device_tensor << "\tptr:" << device_tensor->GetPtr()
            << "\tsize:" << device_tensor->GetSize() << "\toriginal_ref_count:" << device_tensor->original_ref_count()
            << "\tdynamic_ref_count:" << device_tensor->dynamic_ref_count()
            << "\tdevice_type:" << device_tensor->DeviceType()
            << "\tis_ptr_persisted:" << device_tensor->is_ptr_persisted() << "\n ";
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
}  // namespace mindspore
