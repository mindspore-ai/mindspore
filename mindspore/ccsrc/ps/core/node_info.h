/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PS_CORE_NODE_INFO_H_
#define MINDSPORE_CCSRC_PS_CORE_NODE_INFO_H_

#include <string>

#include "proto/comm.pb.h"
#include "proto/ps.pb.h"

namespace mindspore {
namespace ps {
namespace core {
// Events reported to the business layer, include cluster event and node event.
enum class ClusterEvent {
  NODE_TIMEOUT = 1,
  SCHEDULER_TIMEOUT = 2,
  READY_FOR_SCALE_OUT = 3,
  READY_FOR_SCALE_IN = 4,
  CLUSTER_SCALE_OUT_DONE = 5,
  CLUSTER_SCALE_IN_DONE = 6,
  ON_PREPARE_PERSIST = 7,
  ON_BEGIN_PERSIST = 8,
  ON_SEND_META_DATA = 9,
  CLUSTER_SCALE_OUT_ROLLBACK_DONE = 10
};

struct NodeInfo {
  NodeInfo() : ip_(""), port_(0), node_role_(NodeRole::SCHEDULER), rank_id_(UINT32_MAX), is_alive(false) {}
  // ip
  std::string ip_;
  // the port of this node
  uint16_t port_;
  // the current Node unique id:0,1,2...
  std::string node_id_;
  // the role of the node: worker,server,scheduler
  NodeRole node_role_;
  // the current Node rank id,the worker node range is:[0,numOfWorker-1], the server node range is:[0, numOfServer-1]
  uint32_t rank_id_;
  // After the node registration is successful, it is alive.If the node's heartbeat times out, then it is not alive
  bool is_alive;
};
}  // namespace core
}  // namespace ps
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PS_CORE_NODE_INFO_H_
