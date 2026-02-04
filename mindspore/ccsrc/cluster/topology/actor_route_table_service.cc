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

#include "cluster/topology/actor_route_table_service.h"
#include <mutex>
#include <shared_mutex>
#include "utils/log_adapter.h"

namespace mindspore {
namespace distributed {
namespace cluster {
bool ActorRouteTableService::Initialize() const { return true; }

bool ActorRouteTableService::RegisterRoute(const std::string &actor_id, const ActorAddress &actor_addr,
                                           std::string *error) {
  MS_ERROR_IF_NULL_W_RET_VAL(error, false);
  std::unique_lock lock(mtx_);
  if (actor_addresses_.count(actor_id) != 0) {
    *error = "The address of actor id " + actor_id + " already exists.";
    return false;
  }
  actor_addresses_[actor_id] = actor_addr;
  return true;
}

bool ActorRouteTableService::DeleteRoute(const std::string &actor_id, std::string *error) {
  MS_ERROR_IF_NULL_W_RET_VAL(error, false);
  std::unique_lock lock(mtx_);
  if (actor_addresses_.count(actor_id) == 0) {
    *error = "The address of actor id " + actor_id + " does not exist.";
    return false;
  }
  (void)actor_addresses_.erase(actor_id);
  return true;
}

ActorAddress ActorRouteTableService::LookupRoute(const std::string &actor_id, std::string *error) {
  MS_ERROR_IF_NULL_W_RET_VAL(error, {});
  std::shared_lock lock(mtx_);
  if (actor_addresses_.count(actor_id) == 0) {
    *error = "The address of actor id " + actor_id + " does not exist.";
    return {};
  }
  return actor_addresses_[actor_id];
}
}  // namespace cluster
}  // namespace distributed
}  // namespace mindspore
