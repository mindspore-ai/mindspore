/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_RUNTIME_HARDWARE_ASCEND_COOL_COLLECTIVE_COMM_LIB_H_
#define MINDSPORE_CCSRC_RUNTIME_HARDWARE_ASCEND_COOL_COLLECTIVE_COMM_LIB_H_

#include <map>
#include <memory>
#include <vector>
#include <string>
#include <fstream>
#include "utils/ms_utils.h"
#include "include/runtime/hardware_abstract/kernel_base/kernel.h"
#include "include/runtime/hardware_abstract/collective/collective_communication_lib.h"
#include "plugin/ascend/res_manager/event/ascend_event.h"
#include "plugin/ascend/res_manager/collective/ccool_communication_group.h"
#include "plugin/ascend/res_manager/collective/ascend_collective_comm_lib.h"
#include "plugin/ascend/res_manager/collective/ascend_communication_group.h"
#include "kernel/ascend/aclnn/kernel_mod_impl/customize/add_aclnn_kernel.h"
#ifndef EXPORT_WRAPPER
#define EXPORT_WRAPPER __attribute__((visibility("default")))
#endif

namespace mindspore {
namespace device {
namespace ascend {
using GroupOptions = mindspore::device::GroupOptions;

// CCOOL: Collection Communication operator orchestration
class EXPORT_WRAPPER CcoolCollectiveCommLib : public CollectiveCommunicationLib {
 public:
  static CcoolCollectiveCommLib &GetInstance();

  bool Initialize(uint32_t global_rank, uint32_t global_rank_size, uint32_t local_rank_id) override;

  bool CreateCommunicationGroup(const std::string &group_name, const std::vector<uint32_t> &group_ranks,
                                uint32_t local_group_rank, uint32_t local_group_size,
                                const GroupOptions &config = {}) override;

  bool CreateDeviceCommunicationGroup(const std::string &group_name, const std::vector<uint32_t> &group_ranks) override;

  bool DestroyCommunicationGroup(const std::string &group_name) override;

  bool DestroyDeviceCommunicationGroup(const std::string &group_name) override;

  bool AllGather(const void *send_buff, void *recv_buff, size_t send_count, TypeId data_type,
                 const std::string &group_name, void *stream = nullptr) override;

  bool AllReduce(const void *send_buff, void *recv_buff, size_t send_count, TypeId data_type,
                 CollectiveOpReduceType reduce_op, const std::string &group_name, void *stream = nullptr) override;

  bool Broadcast(const void *send_buff, void *recv_buff, size_t send_count, TypeId data_type, uint32_t root_rank,
                 const std::string &group_name, void *stream = nullptr) override;

  bool ReduceScatter(const void *send_buff, void *recv_buff, size_t recv_count, TypeId data_type,
                     CollectiveOpReduceType reduce_op, const std::string &group_name, void *stream = nullptr) override;

  bool Send(const void *send_buff, size_t count, TypeId data_type, uint32_t peer, const std::string &group_name,
            void *stream = nullptr) override;

  bool Recv(void *recv_buff, size_t count, TypeId data_type, uint32_t peer, const std::string &group_name,
            void *stream = nullptr) override;

  // A helper func to pass the host comm lib into CCOOL to assist in creating a ccool comm group
  void SetHelperCommLib(CollectiveCommunicationLib *comm_lib) override;

 private:
  CcoolCollectiveCommLib();
  ~CcoolCollectiveCommLib() override = default;
  bool InterClusterSimpleAllReduce(void *buff, size_t count, TypeId data_type, CollectiveOpReduceType reduce_op,
                                   CcoolCommunicationGroupPtr group, void *stream_ptr,
                                   const std::vector<uint32_t> &inter_cluster_ranks);
  bool InterClusterAllReduceCompute(const std::vector<void *> &ptr_vector, size_t count,
                                    const std::vector<uint32_t> &inter_cluster_ranks, CollectiveOpReduceType reduce_op,
                                    TypeId data_type, CcoolCommunicationGroupPtr group, const size_t &stream_id,
                                    void *stream_ptr, AscendEvent *mem_event);
  bool InterClusterAllReduce(void *buff, size_t count, TypeId data_type, CollectiveOpReduceType reduce_op,
                             CcoolCommunicationGroupPtr group, void *stream_ptr);
  bool InterClusterAllGather(void *send_buff, std::vector<void *> recv_buff_list, size_t size,
                             CcoolCommunicationGroupPtr group, void *stream_ptr);
  bool InterClusterReduceScatter(const std::vector<void *> &send_buff_list, void *recv_buff, size_t recv_count,
                                 TypeId data_type, CollectiveOpReduceType reduce_op, CcoolCommunicationGroupPtr group,
                                 void *stream_ptr);
  size_t GetDtypeSize(TypeId type);
  bool LaunchReduceOperations(void *dst_buff, void *src_buff, void *workspace_buff, size_t data_size, size_t count,
                              TypeId data_type, CollectiveOpReduceType reduce_op, void *stream_ptr);
  nlohmann::json global_rank_table_;
  std::vector<std::string> rank_az_map_;
  std::vector<std::string> rank_ip_map_;
  size_t inner_stream_id_;

  // A helper instance to pass the host comm lib into CCOOL to assist in creating a ccool comm group
  CollectiveCommunicationLib *helper_comm_lib_instance_;
};
}  // namespace ascend
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_HARDWARE_ASCEND_COOL_COLLECTIVE_COMM_LIB_H_
