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

#ifndef MINDSPORE_CCSRC_INCLUDE_RUNTIME_HARDWARE_ABSTRACT_COLLECTIVE_COLLECTIVE_COMMUNICATION_LIB_H_
#define MINDSPORE_CCSRC_INCLUDE_RUNTIME_HARDWARE_ABSTRACT_COLLECTIVE_COLLECTIVE_COMMUNICATION_LIB_H_

#include <atomic>
#include <map>
#include <memory>
#include <vector>
#include <string>
#include <variant>
#include <unordered_map>
#include "ir/dtype/type_id.h"
#include "include/runtime/hardware_abstract/collective/communication_group.h"
#include "runtime/hardware_abstract/visible.h"

namespace mindspore {
namespace device {
// This the config passed to 'CreateCommunicationGroup' method. It controls initialization mode for communication group.
struct RUNTIME_HARDWARE_EXPORT GroupOptions {
  // Whether creating communication group asynchonizely.
  bool async = false;
  // Used to specify some hccl settings.
  std::unordered_map<std::string, std::variant<int64_t, uint32_t, std::string>> hccl_config = {};
};

// The reduce type of collective operations.
enum CollectiveOpReduceType : int64_t {
  Reduce_Mean = 0,
  Reduce_Max = 1,
  Reduce_Min = 2,
  Reduce_Prod = 3,
  Reduce_Sum = 4,
  Reduce_Sum_Square = 5,
  Reduce_ASum = 6,
  Reduce_All = 7
};

// The base class of collective communication library.
// For collective communication on the device side like GPU, the entry is NvidiaCollectiveCommLib which calls NCCL.
// For collective communication on the host side, the entry is MPICollectiveCommLib which call OpenMPI, or
// MsCollectiveCommLib which uses the host-side communication library developed by MindSpore.
class RUNTIME_HARDWARE_EXPORT CollectiveCommunicationLib {
 public:
  CollectiveCommunicationLib()
      : initialized_(false), finalized_(false), global_rank_id_(0), local_rank_id_(0), global_rank_size_(0) {}
  virtual ~CollectiveCommunicationLib() { groups_.clear(); }

  // Initialize collecitve communication library.
  // Inputs 'global_rank' and 'global_rank_size' represents this process's global rank and the group size of the world
  // group.
  // Normally, collective communication libraries on host side will generate these two parameters inside 'Initialize'
  // method. But collective communication libraries on device side needs these inputs passed by the caller.
  virtual bool Initialize(uint32_t global_rank = UINT32_MAX, uint32_t global_rank_size = UINT32_MAX,
                          uint32_t local_rank_id = UINT32_MAX) = 0;

  // Finalize collecitve communication library.
  virtual bool Finalize();

  // Create communication group. This is the precondition for all collective operations on both host and device side.
  virtual bool CreateCommunicationGroup(const std::string &group_name, const std::vector<uint32_t> &group_ranks,
                                        uint32_t local_group_rank, uint32_t local_group_size,
                                        const GroupOptions &config = {}) {
    return true;
  }

  // Create device communication group. This is only needed on device side with rank_table.
  virtual bool CreateDeviceCommunicationGroup(const std::string &group_name, const std::vector<uint32_t> &group_ranks) {
    return true;
  }

  // Destroy device group. This is only needed on device side with rank_table.
  virtual bool DestroyDeviceCommunicationGroup(const std::string &group_name) { return true; }

  // Destroy hccl comm createed on host side. This is needed when host distribute needed.
  virtual bool DestroyCollectiveComms() { return true; }

  // Destroy the communication group.
  virtual bool DestroyCommunicationGroup(const std::string &group_name);

  // Switch network interface card due to borrowing or switchback
  virtual bool CommSwitchNic(const std::vector<uint32_t> &global_ranks, const std::vector<bool> &use_backup) {
    return true;
  }

  // Get the inner communicator name of the specified group.
  virtual std::string CommName(const std::string &group_name) { return ""; }

  // Get the rank id of this process in the specified group.
  virtual uint32_t GetRankId(const std::string &group_name);

  virtual uint32_t GetLocalRankId(const std::string &group_name);

  virtual uint32_t GetLocalGroupSize(const std::string &group_name);

  // Get the size of the specified group.
  virtual uint32_t GetGroupSize(const std::string &group_name);

  virtual uint32_t GetWorldRankFromGroupRank(const std::string &group_name, uint32_t local_rank);

  virtual uint32_t GetGroupRankFromWorldRank(uint32_t group_rank, const std::string &group_name);

  // Assign the local rank id for this process. Normally used by collective communication library on the host side.
  virtual bool AssignLocalRank() { return true; }

  // Return communication group pointer.
  virtual CommunicationGroupPtr GetGroup(const std::string &group_name);

  // AllGather host names of all nodes, used to initialize collective communication.
  virtual bool AllGatherHostHashName(size_t host_hash_name, std::vector<size_t> *host_hash_names) { return true; }

  // Broadcast the device root information to all nodes on host side, used to initialize collective communication.
  virtual bool BroadcastUniqueID(const std::string &group_name, size_t root_info_size, void *root_info) { return true; }

  virtual void ClearUniqueID(const std::string &group_name) const {}

  // Primitive of collective operations.
  virtual bool AllGather(const void *send_buff, void *recv_buff, size_t send_count, TypeId data_type,
                         const std::string &group_name, void *stream = nullptr) {
    return true;
  }
  virtual bool AllReduce(const void *send_buff, void *recv_buff, size_t send_count, TypeId data_type,
                         CollectiveOpReduceType reduce_op, const std::string &group_name, void *stream = nullptr) {
    return true;
  }
  virtual bool Broadcast(const void *send_buff, void *recv_buff, size_t send_count, TypeId data_type,
                         uint32_t root_rank, const std::string &group_name, void *stream = nullptr) {
    return true;
  }
  virtual bool ReduceScatter(const void *send_buff, void *recv_buff, size_t recv_count, TypeId data_type,
                             CollectiveOpReduceType reduce_op, const std::string &group_name, void *stream = nullptr) {
    return true;
  }
  virtual bool Gather(const void *send_buff, void *recv_buff, size_t send_count, TypeId data_type, uint32_t root_rank,
                      const std::string &group_name, void *stream = nullptr) {
    return true;
  }
  virtual bool Scatter(const void *send_buff, void *recv_buff, size_t send_count, TypeId data_type, uint32_t root_rank,
                       const std::string &group_name, void *stream = nullptr) {
    return true;
  }

  virtual bool Send(const void *send_buff, size_t count, TypeId data_type, uint32_t peer, const std::string &group_name,
                    void *stream = nullptr) {
    return true;
  }

  virtual bool Recv(void *recv_buff, size_t count, TypeId data_type, uint32_t peer, const std::string &group_name,
                    void *stream = nullptr) {
    return true;
  }

  // Returns the global group name of this collective communication library. For NCCL, it's 'nccl_world_group'. For
  // HCCL, it's 'hccl_world_group'.
  const std::string &global_group_name() const;

  // Returns global rank id of this process.
  uint32_t global_rank_id() const;

  // Returns local rank id of this process.
  uint32_t local_rank_id() const;

  // Returns global rank size. This is used to create global communication group.
  uint32_t global_rank_size() const;

  virtual void SetLocalGroupRank(const std::string &group_name, uint32_t local_rank_id);

  virtual void SetLocalGroupSize(const std::string &group_name, uint32_t local_group_size);

  // A helper func to pass the host communication lib into CCOOL to assist in creating a communication group
  virtual void SetHelperCommLib(CollectiveCommunicationLib *comm_lib);

  virtual bool ResumeHcclComm() { return false; }

 protected:
  // Whether this collective communication library is initialized.
  bool initialized_;

  // Whether this collective communication library is finalized.
  std::atomic_bool finalized_;

  // The global group name.
  std::string global_group_name_;

  // The global rank id of this process. Normally this range is 0 to `total process number - 1`.
  uint32_t global_rank_id_;

  // The local rank id of this process within the same node. This is usually used as device id.
  uint32_t local_rank_id_;

  // The global rank size. Normally this is equal to `total process number`.
  uint32_t global_rank_size_;

  // This map stores the groups which will be accessed and used by the caller.
  std::map<std::string, std::shared_ptr<CommunicationGroup>> groups_;
};
using CollectiveCommunicationLibPtr = CollectiveCommunicationLib *;
}  // namespace device
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_INCLUDE_RUNTIME_HARDWARE_ABSTRACT_COLLECTIVE_COLLECTIVE_COMMUNICATION_LIB_H_
