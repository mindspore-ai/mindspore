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
#include "utils/convert_utils_base.h"

namespace mindspore {
namespace distributed {
namespace cluster {
static const size_t kInterval = 3;
bool ActorRouteTableProxy::RegisterRoute(const std::string &actor_id, const topology::ActorAddress &actor_addr) {
  MS_EXCEPTION_IF_NULL(cgn_);

  bool success = false;
  while (!success) {
    success = cgn_->PutMetadata(actor_id, actor_addr.SerializeAsString(), false);
    if (!success) {
      MS_LOG(WARNING) << "Retry to register the address for actor: " << actor_id;
      (void)sleep(kInterval);
    }
  }
  MS_LOG(INFO) << "The address of actor " << actor_id << " has been registered successfully.";
  return true;
}

topology::ActorAddress ActorRouteTableProxy::LookupRoute(const std::string &actor_id) const {
  MS_EXCEPTION_IF_NULL(cgn_);
  // Whether this lookup operation is successful.
  bool lookup_success = false;
  // Lookup last timestamp before timeout.
  auto timeout_ts = CURRENT_TIMESTAMP_MILLI + lookup_timeout_;
  topology::ActorAddress lookup_route_rsp_msg;

  do {
    auto route = cgn_->GetMetadata(actor_id);
    if (route.length() == 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(kLookupInterval));
    } else {
      (void)lookup_route_rsp_msg.ParseFromArray(route.c_str(), SizeToInt(route.size()));
      lookup_success = true;
    }

    // An actor route could not be registered yet because another process could be launched slow.
    // If the response actor id is empty, this means the adderess is not registered yet.
    if (lookup_route_rsp_msg.actor_id().empty()) {
      MS_LOG(WARNING) << "Retry to get the address of actor " << actor_id;
    }
  } while (!lookup_success && CURRENT_TIMESTAMP_MILLI <= timeout_ts);

  if (!lookup_success) {
    MS_LOG(EXCEPTION) << "Failed to lookup actor address for " << actor_id
                      << ".\nMaybe the distributed graph is not properly partitioned or training process is not "
                         "launched with correct number. Please check python code or launching script.";
  }
  return lookup_route_rsp_msg;
}
}  // namespace cluster
}  // namespace distributed
}  // namespace mindspore
