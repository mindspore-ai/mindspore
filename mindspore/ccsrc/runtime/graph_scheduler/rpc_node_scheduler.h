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
#include "distributed/rpc/tcp/tcp_client.h"
#include "distributed/rpc/tcp/tcp_server.h"

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

  // Cast some actors to rpc actors according to its kernel name.
  RpcActorSetPtr Build(const ActorSetPtr &actor_set);

  // Link rpc actors with inter-process arrows.
  void Link(const ActorSetPtr &actor_set);
};
}  // namespace runtime
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_GRAPH_SCHEDULER_RPC_NODE_SCHEDULER_H_
