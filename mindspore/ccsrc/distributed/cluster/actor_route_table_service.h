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

#ifndef MINDSPORE_CCSRC_DISTRIBUTED_CLUSTER_ACTOR_ROUTE_TABLE_SERVICE_H_
#define MINDSPORE_CCSRC_DISTRIBUTED_CLUSTER_ACTOR_ROUTE_TABLE_SERVICE_H_

#include <map>
#include <mutex>
#include <string>
#include <memory>
#include <shared_mutex>
#include "proto/comm.pb.h"
#include "utils/log_adapter.h"
#include "include/backend/distributed/constants.h"

namespace mindspore {
namespace distributed {
namespace cluster {
using ps::core::ActorAddress;
// Metadata of actor's route table is physically stored in scheduler node. It receives requests from other nodes like
// workers and servers to update the actor route table.
class ActorRouteTableService {
 public:
  ActorRouteTableService() = default;
  ~ActorRouteTableService() = default;

  bool Initialize() const;

  // Register actor address to the route table. Parameter 'error' represents the failure information if this operation
  // failed.
  bool RegisterRoute(const std::string &actor_id, const ActorAddress &actor_addr, std::string *error);

  // Delete the actor address of the specified actor_id. Parameter 'error' represents the failure information if this
  // operation failed.
  bool DeleteRoute(const std::string &actor_id, std::string *error);

  // Get the actor address for the specified actor_id. Parameter 'error' represents the failure information if this
  // operation failed.
  ActorAddress LookupRoute(const std::string &actor_id, std::string *error);

 private:
  // Metadata of actor address which will used in rpc actors' inter-process communication as 'actor route table'.
  std::map<std::string, ActorAddress> actor_addresses_;

  // Read/write lock for the actor route table.
  std::shared_mutex mtx_;
};
}  // namespace cluster
}  // namespace distributed
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_DISTRIBUTED_CLUSTER_ACTOR_ROUTE_TABLE_SERVICE_H_
