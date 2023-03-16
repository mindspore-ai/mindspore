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

#ifndef MINDSPORE_CCSRC_DISTRIBUTED_CLUSTER_ACTOR_ROUTE_TABLE_PROXY_H_
#define MINDSPORE_CCSRC_DISTRIBUTED_CLUSTER_ACTOR_ROUTE_TABLE_PROXY_H_

#include <string>
#include <memory>
#include <chrono>
#include "proto/topology.pb.h"
#include "include/backend/distributed/constants.h"
#include "include/backend/distributed/cluster/topology/compute_graph_node.h"

namespace mindspore {
namespace distributed {
namespace cluster {
// The timeout in milliseconds for one lookup.
constexpr uint32_t kDefaultLookupTimeout = 300000;

// The time in milliseconds between two lookup operations.
constexpr uint32_t kLookupInterval = 3000;

// Actor route table proxy for nodes like workers and server. This class helps update actor route table in scheduler
// across the network.
class ActorRouteTableProxy {
 public:
  explicit ActorRouteTableProxy(const std::shared_ptr<topology::ComputeGraphNode> &cgn,
                                uint32_t lookup_timeout = kDefaultLookupTimeout)
      : cgn_(cgn), lookup_timeout_(std::chrono::milliseconds(lookup_timeout)) {}
  ~ActorRouteTableProxy() = default;

  // Register actor address to the route table stored in scheduler.
  bool RegisterRoute(const std::string &actor_id, const topology::ActorAddress &actor_addr);

  // Get the actor address for the specified actor_id from the route table stored in scheduler.
  topology::ActorAddress LookupRoute(const std::string &actor_id) const;

 private:
  // The cgn variable helps proxy to communicate with meta server.
  std::shared_ptr<topology::ComputeGraphNode> cgn_;

  // The timeout window for lookup route operation because time of route lookup_timeout of each process is different.
  std::chrono::milliseconds lookup_timeout_;
};

using ActorRouteTableProxyPtr = std::shared_ptr<ActorRouteTableProxy>;
}  // namespace cluster
}  // namespace distributed
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_DISTRIBUTED_CLUSTER_ACTOR_ROUTE_TABLE_PROXY_H_
