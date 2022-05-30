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

#ifndef MINDSPORE_CCSRC_DISTRIBUTED_CLUSTER_DUMMY_ACTOR_ROUTE_TABLE_PROXY_H_
#define MINDSPORE_CCSRC_DISTRIBUTED_CLUSTER_DUMMY_ACTOR_ROUTE_TABLE_PROXY_H_

#include <string>
#include "proto/topology.pb.h"

namespace mindspore {
namespace distributed {
namespace cluster {
using distributed::cluster::topology::ActorAddress;
// The dummy ActorRouteTableProxy interface. This class is for ut test and windows compiling so the implementation is
// empty.
class ActorRouteTableProxy {
 public:
  ActorRouteTableProxy() = default;
  ~ActorRouteTableProxy() = default;

  bool RegisterRoute(const std::string &, const ActorAddress &) { return true; }
  bool DeleteRoute(const std::string &) { return true; }
  ActorAddress LookupRoute(const std::string &) const { return {}; }
};
}  // namespace cluster
}  // namespace distributed
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_DISTRIBUTED_CLUSTER_DUMMY_ACTOR_ROUTE_TABLE_PROXY_H_
