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

#ifndef MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_SET_H_
#define MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_SET_H_

#if defined(__linux__) && defined(WITH_BACKEND)
#define ENABLE_RPC_ACTOR
#endif

#include <vector>
#include <string>
#include <memory>
#include <utility>
#include <unordered_map>
#include <map>
#include <set>
#include "runtime/graph_scheduler/actor/abstract_actor.h"
#include "runtime/graph_scheduler/actor/data_prepare_actor.h"
#include "runtime/graph_scheduler/actor/data_source_actor.h"
#include "runtime/graph_scheduler/actor/loop_count_actor.h"
#include "runtime/graph_scheduler/actor/kernel_actor.h"
#include "runtime/graph_scheduler/actor/kernel_infer_actor.h"
#include "runtime/graph_scheduler/actor/kernel_resize_actor.h"
#include "runtime/graph_scheduler/actor/custom_actor.h"
#include "runtime/graph_scheduler/actor/super_kernel_actor.h"
#include "runtime/graph_scheduler/actor/any_type_kernel_actor.h"
#include "runtime/graph_scheduler/actor/output_actor.h"
#include "runtime/graph_scheduler/actor/copy_actor.h"
#include "runtime/graph_scheduler/actor/fusion/fusion_actor.h"
#include "runtime/graph_scheduler/actor/control_flow/switch_actor.h"
#include "runtime/graph_scheduler/actor/control_flow/gather_actor.h"
#include "runtime/graph_scheduler/actor/control_flow/entrance_actor.h"
#include "runtime/graph_scheduler/actor/control_flow/exit_actor.h"
#include "runtime/graph_scheduler/actor/control_flow/stack_actor.h"
#include "runtime/graph_scheduler/actor/memory/memory_swap_actor.h"

#ifdef ENABLE_RPC_ACTOR
#include "runtime/graph_scheduler/actor/rpc/send_actor.h"
#include "runtime/graph_scheduler/actor/rpc/recv_actor.h"
#endif

namespace mindspore {
namespace runtime {
using ActorInfo = std::string;

// Control actor set is a series of actors used to implement control flow:
// switch actor judges which branch to output according to the input index;
// gather actor is used to collect the actual parameters required by the subgraph call Entrance actor is used
// as the entrance of the subgraph to receive the actual parameters and branch id sent by gather, and send them
// to the kernel actor;
// exit actor is used as the output of the subgraph to collect the calculation results of the subgraph and return
// to the caller according to the branch id;
// The stack actor collects the output of the kernel actor and exit actor in the untail recursion, and the output
// of the kernel actor needs to be saved by the stack.
struct ControlActorSet {
  std::vector<SwitchActorPtr> switch_actors_;
  std::vector<GatherActorPtr> gather_actors_;
  std::vector<EntranceActorPtr> entrance_actors_;
  std::vector<ExitActorPtr> exit_actors_;
  std::vector<StackActorPtr> stack_actors_;
};
using ControlActorSetPtr = std::shared_ptr<ControlActorSet>;

#ifdef ENABLE_RPC_ACTOR
// Rpc actor set is a series of actors implemented to communicate with other processes. In distributed execution mode,
// the graph could be considered as partitioned to different processes, which is connected by these rpc actors. Send
// actors are in charge of sending data to other processes. Recv actors are in charge of receiving data from other
// processes.
struct RpcActorSet {
  std::vector<SendActorPtr> send_actors_;
  std::vector<RecvActorPtr> recv_actors_;
};
using RpcActorSetPtr = std::shared_ptr<RpcActorSet>;
using RpcActorSetWeakPtr = std::weak_ptr<RpcActorSet>;
#endif

// The actor set generated by graph transformer is the execution unit of actor runtime.
// It includes data source actor, kernel actor, switch actor, copy actor, loop count actor and output actor.
// The data prepare actor is used to prepare data for device tensor store and host tensor queue to represent the begin
// of one step.
// The data source actor is used to obtain data and process them into device tensors, and send them to kernel actor.
// The kernel actor is used to receive the device tensors to luanch kernel.
// The Super kernel actor is used to represent the sink executing of graph which is the combination of kernels.
// The no input kernel actor means that this actor has no input arrow and needs to be triggered externally.
// The copy actor is used to convert the device tensor between the different device kernel.
// The loop count actor is used to receive the control of tail kernel actor to represent the end of one step
// and decide whether to loop execution by loop count.
// The output actor is used to receive the output result of actor which represents the graph output.
struct ActorSet {
  explicit ActorSet(const ActorInfo &name) : name_(name) {}
  DataPrepareActorPtr data_prepare_actor_{nullptr};
  std::vector<DataSourceActorPtr> data_source_actors_;
  std::vector<KernelActorPtr> kernel_actors_;
  std::vector<KernelInferActorPtr> kernel_infer_actors_;
  std::vector<KernelResizeActorPtr> kernel_resize_actors_;
  std::vector<CustomActorPtr> custom_actors_;
  std::vector<SuperKernelActorPtr> super_kernel_actors_;
  std::vector<AnyTypeKernelActorPtr> any_type_kernel_actors_;
  // No input kernel actors need be triggered specifically.
  std::vector<AbstractActorPtr> no_input_kernel_actors_;
  std::vector<MemoryAwareActorPtr> memory_actors_;
  std::vector<CopyActorPtr> copy_actors_;
  std::vector<FusionActorPtr> fusion_actors_;
  std::vector<std::vector<MemSwapActorPtr>> swap_actors_;
  LoopCountActorPtr loop_count_actor_{nullptr};
  OutputActorPtr output_actor_{nullptr};
  ControlActorSetPtr control_actors_{nullptr};
#ifdef ENABLE_RPC_ACTOR
  RpcActorSetPtr rpc_actors_{nullptr};
#endif
  ActorInfo name_;
  // The related statistics information of multi thread and single thread to decide whether use the multi thread.
  bool is_multi_thread_execution_{true};
  size_t execution_count_{0};
  double multi_thread_execution_time_{0};
  double single_thread_execution_time_{0};
  // Record the execution state.
  bool is_execution_failed_{false};
};
using ActorSetPtr = std::shared_ptr<ActorSet>;

// The operation of the map of kActorNameToActor.
void InsertActor(AbstractActor *actor);
AbstractActor *FetchActor(const std::string &actor_name);
AbstractActor *FetchActor(KernelTransformType kernel_type, const std::string &actor_set_name,
                          const AnfNodePtr &node = nullptr, const KernelGraphPtr &graph = nullptr);
void EraseActor(const std::string &actor_name);
void ClearAllActors();
}  // namespace runtime
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_SET_H_
