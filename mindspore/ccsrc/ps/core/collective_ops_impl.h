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

#include <map>
#include <memory>
#include <string>
#include <vector>
#include <functional>
#include "proto/ps.pb.h"
#include "include/backend/distributed/ps/ps_context.h"
#include "ps/core/server_node.h"

namespace mindspore {
namespace fl {
namespace server {
// The timeout for server collective communication in case of network jitter.
constexpr uint32_t kCollectiveCommTimeout = 30;
// The max timeout for server collective communication, used in disaster recovery to prevent networking flapping.
constexpr uint32_t kCollectiveCommMaxTimeout = 300;

// The collective communication groups which are composed of multiple processes. Refer to MPI_Group.
struct CommunicationGroupInfo {
  // This group's rank size.
  uint32_t size;

  // This process's global rank id.
  uint32_t global_rank;

  // The group ranks consists of global ranks of the processes.
  std::vector<uint32_t> group_ranks;

  // The mapping of global ranks and group ranks.
  std::map<uint32_t, uint32_t> global_to_group_ranks;
  std::map<uint32_t, uint32_t> group_to_global_ranks;
};

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
  bool AllReduce(const std::string &data_name, void *sendbuff, void *recvbuff, size_t count);

  template <typename T>
  bool AllGather(const void *sendbuff, void *recvbuff, size_t send_count, const ps::core::AbstractNodePtr &node);

  // Collective broadcast within the specified group. The parameter "root" is the group rank of the root process.
  // Normally 0.
  template <typename T>
  bool Broadcast(const void *sendbuff, void *recvbuff, size_t count, uint32_t root,
                 const ps::core::AbstractNodePtr &node, const CommunicationGroupInfo &group_info);

 private:
  CollectiveOpsImpl()
      : server_node_(nullptr),
        rank_id_(0),
        server_num_(0),
        node_(nullptr),
        node_role_(ps::core::NodeRole::WORKER),
        rank_size_(0) {}
  ~CollectiveOpsImpl() = default;
  CollectiveOpsImpl(const CollectiveOpsImpl &) = delete;
  CollectiveOpsImpl &operator=(const CollectiveOpsImpl &) = delete;

  // Implementation of RingAllReduce.
  template <typename T>
  bool RunRingAllReduce(const std::string &data_name, uint32_t send_to_rank, uint32_t recv_from_rank,
                        const std::vector<size_t> &chunk_sizes, const std::vector<size_t> &chunk_offset,
                        T *output_buff);

  // Implementation of RingAllReduce.
  template <typename T>
  bool RingAllReduce(const std::string &data_name, const void *sendbuff, void *recvbuff, size_t count);

  // Implementation of BroadcastAllReduce.
  template <typename T>
  bool ReduceBroadcastAllReduce(const std::string &data_name, const void *sendbuff, void *recvbuff, size_t count);

  // Implementation of RingAllGather.
  template <typename T>
  bool RingAllGather(const void *sendbuff, void *recvbuff, size_t send_count);

  // Implementation of Broadcast. The parameter "root" is the group rank of the root process. Normally 0.
  template <typename T>
  bool Broadcast(const void *sendbuff, void *recvbuff, size_t count, uint32_t root,
                 const CommunicationGroupInfo &group_info);

  std::shared_ptr<ps::core::ServerNode> server_node_;
  uint32_t rank_id_;
  uint32_t server_num_;

  // The mutex to ensure that collective communication is threadsafe.
  std::mutex mtx_;

  // The abstract node could be worker or server. Only nodes which have the same role could use collective
  // communication.
  ps::core::AbstractNodePtr node_;
  ps::core::NodeRole node_role_;
  uint32_t rank_size_;
};
}  // namespace server
}  // namespace fl
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FL_SERVER_COLLECTIVE_OPS_IMPL_H_
