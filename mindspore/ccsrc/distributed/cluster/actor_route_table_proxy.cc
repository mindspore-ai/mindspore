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
#include "distributed/cluster/actor_route_table_proxy.h"

namespace mindspore {
namespace distributed {
namespace cluster {
bool ActorRouteTableProxy::RegisterRoute(const std::string &actor_id, const ActorAddress &actor_addr) { return true; }

bool ActorRouteTableProxy::DeleteRoute(const std::string &actor_id) { return true; }

ActorAddress ActorRouteTableProxy::LookupRoute(const std::string &actor_id) const { return {}; }
}  // namespace cluster
}  // namespace distributed
}  // namespace mindspore
