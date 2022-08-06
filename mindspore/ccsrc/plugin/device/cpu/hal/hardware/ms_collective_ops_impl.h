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

#ifndef MINDSPORE_CCSRC_RUNTIME_HARDWARE_CPU_MS_COLLECTIVE_OPS_IMPL_H_
#define MINDSPORE_CCSRC_RUNTIME_HARDWARE_CPU_MS_COLLECTIVE_OPS_IMPL_H_

#include <map>
#include <memory>
#include <string>
#include <vector>
#include <functional>
#include "plugin/device/cpu/hal/hardware/ms_collective_topo.h"

namespace mindspore {
namespace device {
namespace cpu {
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

// MSCollectiveOpsImpl is the collective communication API of the server.
// For now, it implements two AllReduce algorithms: RingAllReduce and BroadcastAllReduce. Elastic AllReduce is also
// supported for the elastic scaling feature of the server.
class MSCollectiveOpsImpl {
 public:
  explicit MSCollectiveOpsImpl(const std::shared_ptr<TopologyNode> &topo_node)
      : rank_id_(0), rank_size_(0), topo_node_(topo_node) {}
  ~MSCollectiveOpsImpl() = default;

  bool Initialize();

  template <typename T>
  bool AllReduce(const std::string &data_name, void *sendbuff, void *recvbuff, size_t count);

  template <typename T>
  bool AllGather(const void *sendbuff, void *recvbuff, size_t send_count);

  // Collective broadcast within the specified group. The parameter "root" is the group rank of the root process.
  // Normally 0.
  template <typename T>
  bool Broadcast(const void *sendbuff, void *recvbuff, size_t count, uint32_t root,
                 const CommunicationGroupInfo &group_info);

 private:
  MSCollectiveOpsImpl(const MSCollectiveOpsImpl &) = delete;
  MSCollectiveOpsImpl &operator=(const MSCollectiveOpsImpl &) = delete;

  // Implementation of RingAllGather.
  template <typename T>
  bool RingAllGather(const void *sendbuff, void *recvbuff, size_t send_count);

  template <typename T>
  bool RingAllGatherImpl(uint32_t send_to_rank, uint32_t recv_from_rank, T *output_buff,
                         const std::vector<size_t> &chunk_offset, const std::vector<size_t> &chunk_sizes);

  uint32_t rank_id_;
  uint32_t rank_size_;

  std::shared_ptr<TopologyNode> topo_node_{nullptr};

  // The mutex to ensure that collective communication is threadsafe.
  std::mutex mtx_;
};

template bool MSCollectiveOpsImpl::AllGather<float>(const void *sendbuff, void *recvbuff, size_t send_count);
template bool MSCollectiveOpsImpl::AllGather<uint64_t>(const void *sendbuff, void *recvbuff, size_t send_count);
template bool MSCollectiveOpsImpl::AllGather<int>(const void *sendbuff, void *recvbuff, size_t send_count);
template bool MSCollectiveOpsImpl::AllGather<char>(const void *sendbuff, void *recvbuff, size_t send_count);

template bool MSCollectiveOpsImpl::RingAllGather<float>(const void *sendbuff, void *recvbuff, size_t send_count);
template bool MSCollectiveOpsImpl::RingAllGather<uint64_t>(const void *sendbuff, void *recvbuff, size_t send_count);
template bool MSCollectiveOpsImpl::RingAllGather<int>(const void *sendbuff, void *recvbuff, size_t send_count);
template bool MSCollectiveOpsImpl::RingAllGather<char>(const void *sendbuff, void *recvbuff, size_t send_count);

template bool MSCollectiveOpsImpl::Broadcast<float>(const void *sendbuff, void *recvbuff, size_t count, uint32_t root,
                                                    const CommunicationGroupInfo &group_info);
template bool MSCollectiveOpsImpl::Broadcast<uint64_t>(const void *sendbuff, void *recvbuff, size_t count,
                                                       uint32_t root, const CommunicationGroupInfo &group_info);
template bool MSCollectiveOpsImpl::Broadcast<int>(const void *sendbuff, void *recvbuff, size_t count, uint32_t root,
                                                  const CommunicationGroupInfo &group_info);
template bool MSCollectiveOpsImpl::Broadcast<char>(const void *sendbuff, void *recvbuff, size_t count, uint32_t root,
                                                   const CommunicationGroupInfo &group_info);
}  // namespace cpu
}  // namespace device
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_RUNTIME_HARDWARE_CPU_MS_COLLECTIVE_OPS_IMPL_H_
