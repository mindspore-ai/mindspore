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
  RpcNodeScheduler() : op_context_(nullptr), rpc_actors_(nullptr) {}
  ~RpcNodeScheduler() = default;

  // Build rpc actors and return rpc actor set.
  RpcActorSetPtr Build(const ActorSet *actor_set);

  // Link rpc actors with inter-process arrows.
  void Link(const ActorSet *actor_set) const;

  // This should be called by 'GraphScheduler::Scheduler()' method.
  // Used to start servers for recv actors and create connections for send actors.
  void Schedule(const ActorSet *actor_set) const;

  // Set op context to rpc actors.
  void SetOpcontext(const RpcActorSetPtr &rpc_actors, OpContext<DeviceTensor> *const op_context);

  // Reset op context for rpc actors.
  void ResetOpcontext(const RpcActorSetPtr &rpc_actors) const;

  // Abort rpc communication. This is usually called when the cluster exits with exception.
  void Abort();

 private:
  /**
   * @description: Update reference counts of rpc actors's inputs and workspaces.
   *               Because the memory of inputs and workspaces should not be released by the framework until rpc module
   *               done sending or receiving.
   * @param {RpcActorSetPtr} rpc_actor_set: The rpc actors set.
   * @return {void}
   */
  void UpdateRpcActorRefCounts(RpcActorSetPtr rpc_actor_set) const;

  // Create new route table proxy.
  ActorRouteTableProxyPtr CreateRouteTableProxy() const;

  OpContext<DeviceTensor> *op_context_;

  RpcActorSetPtr rpc_actors_;
};

// The setter of op context for rpc actors.
class RpcActorOpContextSetter {
 public:
  explicit RpcActorOpContextSetter(RpcNodeScheduler *rpc_node_scheduler, const RpcActorSetPtr &rpc_actors,
                                   OpContext<DeviceTensor> *const op_context)
      : rpc_node_scheduler_(rpc_node_scheduler), rpc_actors_(rpc_actors), op_context_(op_context) {
    rpc_node_scheduler_->SetOpcontext(rpc_actors_, op_context_);
  }
  ~RpcActorOpContextSetter() {
    try {
      rpc_node_scheduler_->ResetOpcontext(rpc_actors_);
    } catch (const std::exception &) {
      MS_LOG(ERROR) << "Failed to reset op context.";
    }
  }

 private:
  RpcNodeScheduler *rpc_node_scheduler_;
  RpcActorSetPtr rpc_actors_;
  OpContext<DeviceTensor> *op_context_;
};

// This class is used to refresh the state of the rpc actor. For example, the mux recv actor receives requests for
// the service process. Currently, the requests are processed serially. After each request (that is, the execution of an
// actor dag) ends, the state of the Recv actor needs to be refreshed. Make it in the ready state to continue with the
// next request.
class RpcActorStatusUpdater {
 public:
  static RpcActorStatusUpdater &GetInstance();

  // Set rpc actors which need to be update status.
  void set_rpc_actors(const RpcActorSetPtr &rpc_actors);

  // Update rpc actors' status.
  void UpdateRpcActorStatus() const;

 private:
  RpcActorStatusUpdater() = default;
  ~RpcActorStatusUpdater() = default;
  DISABLE_COPY_AND_ASSIGN(RpcActorStatusUpdater);

  // Record rpc actors which need to update status.
  RpcActorSetWeakPtr rpc_actors_;
};
}  // namespace runtime
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_GRAPH_SCHEDULER_RPC_NODE_SCHEDULER_H_
