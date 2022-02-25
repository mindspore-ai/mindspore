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

#include <string>
#include <vector>
#include "distributed/cluster/actor_route_table_proxy.h"

namespace mindspore {
namespace distributed {
namespace cluster {
bool ActorRouteTableProxy::RegisterRoute(const std::string &actor_id, const ActorAddress &actor_addr) {
  MS_EXCEPTION_IF_NULL(node_);
  std::shared_ptr<std::vector<unsigned char>> output = nullptr;
  if (!node_->SendToScheduler(actor_addr.SerializeAsString().data(), actor_addr.SerializeAsString().size(),
                              NodeCommand::REGISTER_ACTOR_ROUTE, &output)) {
    MS_LOG(EXCEPTION) << "Failed to send register route request to scheduler.";
  }

  GeneralResponseMsg register_route_rsp_msg;
  MS_EXCEPTION_IF_NULL(output);
  (void)register_route_rsp_msg.ParseFromArray(output->data(), SizeToInt(output->size()));
  if (!register_route_rsp_msg.is_success()) {
    MS_LOG(ERROR) << "Register route for actor " << actor_id << " failed. " << register_route_rsp_msg.error();
    return false;
  }
  return true;
}

bool ActorRouteTableProxy::DeleteRoute(const std::string &actor_id) {
  MS_EXCEPTION_IF_NULL(node_);
  std::shared_ptr<std::vector<unsigned char>> output = nullptr;
  if (!node_->SendToScheduler(actor_id.data(), actor_id.size(), NodeCommand::DELETE_ACTOR_ROUTE, &output)) {
    MS_LOG(EXCEPTION) << "Failed to send delete route request to scheduler.";
  }

  GeneralResponseMsg delete_route_rsp_msg;
  MS_EXCEPTION_IF_NULL(output);
  (void)delete_route_rsp_msg.ParseFromArray(output->data(), SizeToInt(output->size()));
  if (!delete_route_rsp_msg.is_success()) {
    MS_LOG(ERROR) << "Delete route for actor " << actor_id << " failed. " << delete_route_rsp_msg.error();
    return false;
  }
  return true;
}

ActorAddress ActorRouteTableProxy::LookupRoute(const std::string &actor_id) const {
  MS_EXCEPTION_IF_NULL(node_);
  // Whether this lookup operation is successful.
  bool lookup_success = false;
  // Lookup last timestamp before timeout.
  auto timeout_ts = CURRENT_TIMESTAMP_MILLI + lookup_timeout_;
  std::shared_ptr<std::vector<unsigned char>> output = nullptr;
  ActorAddress lookup_route_rsp_msg;
  do {
    if (!node_->SendToScheduler(actor_id.data(), actor_id.size(), NodeCommand::LOOKUP_ACTOR_ROUTE, &output)) {
      MS_LOG(EXCEPTION) << "Failed to send lookup route request to scheduler.";
    }

    MS_EXCEPTION_IF_NULL(output);
    (void)lookup_route_rsp_msg.ParseFromArray(output->data(), SizeToInt(output->size()));
    // An actor route could not be registered yet because another process could be launched slow.
    // If the response actor id is empty, this means the adderess is not registered yet.
    if (lookup_route_rsp_msg.actor_id().empty()) {
      MS_LOG(DEBUG) << "Actor route for actor " << actor_id << " is not registered yet, please try later.";
      std::this_thread::sleep_for(std::chrono::milliseconds(kLookupInterval));
    } else {
      lookup_success = true;
    }
  } while (!lookup_success && CURRENT_TIMESTAMP_MILLI <= timeout_ts);

  return lookup_route_rsp_msg;
}
}  // namespace cluster
}  // namespace distributed
}  // namespace mindspore
