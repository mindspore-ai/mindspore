/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FL_SERVER_COLLECTIVE_OPS_IMPL_H_
#define MINDSPORE_CCSRC_FL_SERVER_COLLECTIVE_OPS_IMPL_H_

#include <memory>
#include <string>
#include <vector>
#include <functional>
#include "proto/ps.pb.h"
#include "ps/ps_context.h"
#include "ps/core/server_node.h"
#include "fl/server/common.h"

namespace mindspore {
namespace fl {
namespace server {
// The timeout for server collective communication in case of network jitter.
constexpr uint32_t kCollectiveCommTimeout = 30;

// CollectiveOpsImpl is the collective communication API of the server.
// For now, it implements two AllReduce algorithms: RingAllReduce and BroadcastAllReduce. Elastic AllReduce is also
// supported for the elastic scaling feature of the server.
class CollectiveOpsImpl {
 public:
  static CollectiveOpsImpl &GetInstance() {
    static CollectiveOpsImpl instance;
    return instance;
  }

  void Initialize(const std::shared_ptr<ps::core::ServerNode> &server_node);

  template <typename T>
  bool AllReduce(const void *sendbuff, void *recvbuff, size_t count);

  // Reinitialize the ring for collective communication after scaling operations are done.
  bool ReInitForScaling();

 private:
  CollectiveOpsImpl() : server_node_(nullptr), local_rank_(0), server_num_(0) {}
  ~CollectiveOpsImpl() = default;
  CollectiveOpsImpl(const CollectiveOpsImpl &) = delete;
  CollectiveOpsImpl &operator=(const CollectiveOpsImpl &) = delete;

  // Implementation of RingAllReduce.
  template <typename T>
  bool RingAllReduce(const void *sendbuff, void *recvbuff, size_t count);

  // Implementation of BroadcastAllReduce.
  template <typename T>
  bool ReduceBroadcastAllReduce(const void *sendbuff, void *recvbuff, size_t count);

  std::shared_ptr<ps::core::ServerNode> server_node_;
  uint32_t local_rank_;
  uint32_t server_num_;

  // The mutex to ensure that collective communication is threadsafe.
  std::mutex mtx_;
};
}  // namespace server
}  // namespace fl
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FL_SERVER_COLLECTIVE_OPS_IMPL_H_
