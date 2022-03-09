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

#ifndef MINDSPORE_CCSRC_DISTRIBUTED_CLUSTER_TOPOLOGY_COMMON_H_
#define MINDSPORE_CCSRC_DISTRIBUTED_CLUSTER_TOPOLOGY_COMMON_H_

#include <string>

namespace mindspore {
namespace distributed {
namespace cluster {
namespace topology {
// The address of meta server node used by compute graph nodes to register and get addresses of other compute graph
// nodes dynamically.
struct MetaServerAddress {
  std::string GetUrl() { return ip + ":" + std::to_string(port); }
  std::string ip;
  int port;
};

// The address of meta server node.
// This address is set or obtained through environment variables.
constexpr char kEnvMetaServerHost[] = "MS_SCHED_HOST";
constexpr char kEnvMetaServerPort[] = "MS_SCHED_PORT";

constexpr char kEnvNodeId[] = "MS_NODE_ID";

// For port number conversion.
static const int kDecimal = 10;

// All kinds of messages sent between compute graph nodes and meta server node.
enum class MessageName { kRegistration, kHeartbeat };
}  // namespace topology
}  // namespace cluster
}  // namespace distributed
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_DISTRIBUTED_CLUSTER_TOPOLOGY_COMMON_H_
