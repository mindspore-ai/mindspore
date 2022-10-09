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
#include <chrono>

namespace mindspore {
namespace distributed {
namespace cluster {
namespace topology {
// Indicates the state of the cluster physical topology.
enum class TopoState {
  // All the nodes of this cluster are in the process of starting up.
  kInitializing = 0,

  // All the nodes of this cluster has been started and registered to the meta server node successfully.
  kInitialized,

  // The topo of this cluster failed to construct at specified time.
  kFailed,

  // All the nodes of this cluster have finished their tasks and unregistered successfully.
  kFinished
};

// The address of meta server node used by compute graph nodes to register and get addresses of other compute graph
// nodes dynamically.
struct MetaServerAddress {
  std::string GetUrl() { return ip + ":" + std::to_string(port); }
  std::string ip;
  int port{-1};
};

// The address of meta server node.
// This address is set or obtained through environment variables.
constexpr char kEnvMetaServerHost[] = "MS_SCHED_HOST";
constexpr char kEnvMetaServerPort[] = "MS_SCHED_PORT";

constexpr char kEnvNodeId[] = "MS_NODE_ID";

// The key of compute graph node's hostname metadata stored in meta server.
constexpr char kHostNames[] = "hostnames";

// For port number conversion.
static const int kDecimal = 10;

// The timeout(second) for heartbeat from compute graph node to meta server.
static const uint64_t kDefaultNodeTimeout = 30;

// The timeout for initializing the cluster topology.
static const std::chrono::milliseconds kTopoInitTimeout = std::chrono::milliseconds(1000 * 60 * 10);

// All kinds of messages sent between compute graph nodes and meta server node.
enum class MessageName {
  kRegistration,
  kUnregistration,
  kHeartbeat,
  kSuccess,
  kInvalidNode,
  kUninitTopo,
  kWriteMetadata,
  kReadMetadata,
  kDeleteMetadata,
  kGetHostNames,
  kValidMetadata,
  kInvalidMetadata
};

// The retry and interval configuration used for the macro `EXECUTE_WITH_RETRY`.
static const size_t kExecuteRetryNum = 210;
// The retry number of cgn and msn for reconnecting.
static const size_t kCgnExecuteRetryNum = 60;
static const size_t kMsnExecuteRetryNum = 210;
static const size_t kNoRetry = 1;
static const uint32_t kExecuteInterval = 3;

#define EXECUTE_WITH_RETRY(func, retry, interval, err_msg)                     \
  do {                                                                         \
    bool success = false;                                                      \
    for (size_t i = 1; i <= retry; ++i) {                                      \
      success = func();                                                        \
      if (!success) {                                                          \
        MS_LOG(WARNING) << err_msg << ", retry(" << i << "/" << retry << ")."; \
        (void)sleep(interval);                                                 \
      } else {                                                                 \
        break;                                                                 \
      }                                                                        \
    }                                                                          \
    if (!success) {                                                            \
      return false;                                                            \
    }                                                                          \
  } while (false)

#define EXECUTE_WITH_TIMEOUT(func, interval, err_msg, success, time) \
  do {                                                               \
    success = false;                                                 \
    while (!success) {                                               \
      success = func;                                                \
      if (!success) {                                                \
        MS_LOG(WARNING) << err_msg << ", retry...";                  \
        (void)sleep(interval);                                       \
        if ((time - interval) < 0) break;                            \
        time -= interval;                                            \
      } else {                                                       \
        break;                                                       \
      }                                                              \
    }                                                                \
    if (!success && time <= 0) {                                     \
      MS_LOG(ERROR) << err_msg;                                      \
      return false;                                                  \
    }                                                                \
  } while (false)

#define EXECUTE_WITH_EXPECTED(func, expected, interval, err_msg, time) \
  do {                                                                 \
    bool success = false;                                              \
    while (!success) {                                                 \
      success = (func == expected);                                    \
      if (!success) {                                                  \
        MS_LOG(WARNING) << err_msg << ", retry...";                    \
        (void)sleep(interval);                                         \
        if ((time - interval) < 0) break;                              \
        time -= interval;                                              \
      } else {                                                         \
        break;                                                         \
      }                                                                \
    }                                                                  \
    if (!success && time <= 0) {                                       \
      MS_LOG(ERROR) << err_msg;                                        \
      return false;                                                    \
    }                                                                  \
  } while (false)
}  // namespace topology
}  // namespace cluster
}  // namespace distributed
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_DISTRIBUTED_CLUSTER_TOPOLOGY_COMMON_H_
