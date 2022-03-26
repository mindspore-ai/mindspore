/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_RUNTIME_GRAPH_SCHEDULER_RPC_NODE_SCHEDULER_H_
#define MINDSPORE_CCSRC_RUNTIME_GRAPH_SCHEDULER_RPC_NODE_SCHEDULER_H_

#include <vector>
#include <string>
#include <memory>
#include "runtime/graph_scheduler/actor/actor_set.h"
#include "runtime/graph_scheduler/graph_compiler.h"

namespace mindspore {
namespace runtime {
using mindspore::device::DeviceContext;
using mindspore::session::KernelGraph;
using mindspore::session::KernelWithIndex;

// Scheduler for rpc actors, e.g., it adds inter-process arrows, generate router for actors, etc.
class RpcNodeScheduler {
 public:
  RpcNodeScheduler() = default;
  ~RpcNodeScheduler() = default;

  // Build rpc actors and return rpc actor set.
  RpcActorSetPtr Build(const ActorSet *actor_set);

  // Link rpc actors with inter-process arrows.
  void Link(const ActorSet *actor_set);

  // This should be called by 'GraphScheduler::Scheduler()' method.
  // Used to start servers for recv actors and create connections for send actors.
  void Schedule(const ActorSet *actor_set);

  // Set op context to rpc actors.
  void SetOpcontext(const RpcActorSetPtr &rpc_actors, OpContext<DeviceTensor> *const op_context);

  // Reset op context for rpc actors.
  void ResetOpcontext(const RpcActorSetPtr &rpc_actors);

 private:
  // Create new route table proxy.
  ActorRouteTableProxyPtr CreateRouteTableProxy();
};

// The setter of op context for rpc actors.
class RpcActorOpContextSetter {
 public:
  explicit RpcActorOpContextSetter(RpcNodeScheduler *rpc_node_scheduler, const RpcActorSetPtr &rpc_actors,
                                   OpContext<DeviceTensor> *const op_context)
      : rpc_node_scheduler_(rpc_node_scheduler), rpc_actors_(rpc_actors), op_context_(op_context) {
    rpc_node_scheduler_->SetOpcontext(rpc_actors_, op_context_);
  }
  ~RpcActorOpContextSetter() { rpc_node_scheduler_->ResetOpcontext(rpc_actors_); }

 private:
  RpcNodeScheduler *rpc_node_scheduler_;
  RpcActorSetPtr rpc_actors_;
  OpContext<DeviceTensor> *op_context_;
};
}  // namespace runtime
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_GRAPH_SCHEDULER_RPC_NODE_SCHEDULER_H_
