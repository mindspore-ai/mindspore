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

#ifndef MINDSPORE_CCSRC_RUNTIME_HARDWARE_ASCEND_ASCEND_COLLECTIVE_COMM_LIB_H_
#define MINDSPORE_CCSRC_RUNTIME_HARDWARE_ASCEND_ASCEND_COLLECTIVE_COMM_LIB_H_

#include <map>
#include <memory>
#include <vector>
#include <string>
#include "runtime/collective/collective_communication_lib.h"
#include "plugin/device/ascend/hal/hardware/ascend_communication_group.h"

#ifndef EXPORT_WRAPPER
#define EXPORT_WRAPPER __attribute__((visibility("default")))
#endif

namespace mindspore {
namespace device {
namespace ascend {
static std::map<CollectiveOpReduceType, HcclReduceOp> kHcomOpReduceTypeMap = {
  {device::CollectiveOpReduceType::Reduce_Max, HCCL_REDUCE_MAX},
  {device::CollectiveOpReduceType::Reduce_Min, HCCL_REDUCE_MIN},
  {device::CollectiveOpReduceType::Reduce_Prod, HCCL_REDUCE_PROD},
  {device::CollectiveOpReduceType::Reduce_Sum, HCCL_REDUCE_SUM}};

static std::map<int64_t, HcclDataType> kConstOpHcomDataTypeMap = {
  {TypeId::kNumberTypeInt8, HCCL_DATA_TYPE_INT8},     {TypeId::kNumberTypeInt16, HCCL_DATA_TYPE_INT16},
  {TypeId::kNumberTypeInt32, HCCL_DATA_TYPE_INT32},   {TypeId::kNumberTypeFloat16, HCCL_DATA_TYPE_FP16},
  {TypeId::kNumberTypeFloat32, HCCL_DATA_TYPE_FP32},  {TypeId::kNumberTypeInt64, HCCL_DATA_TYPE_INT64},
  {TypeId::kNumberTypeUInt64, HCCL_DATA_TYPE_UINT64}, {TypeId::kNumberTypeUInt8, HCCL_DATA_TYPE_UINT8},
  {TypeId::kNumberTypeUInt16, HCCL_DATA_TYPE_UINT16}, {TypeId::kNumberTypeUInt32, HCCL_DATA_TYPE_UINT32},
  {TypeId::kNumberTypeFloat64, HCCL_DATA_TYPE_FP64},  {TypeId::kNumberTypeBFloat16, HCCL_DATA_TYPE_BFP16},
};

constexpr char kHCCLGlobalGroupName[] = "hccl_world_group";

class EXPORT_WRAPPER AscendCollectiveCommLib : public CollectiveCommunicationLib {
 public:
  static AscendCollectiveCommLib &GetInstance() {
    static AscendCollectiveCommLib instance;
    return instance;
  }

  bool Initialize(uint32_t global_rank, uint32_t global_rank_size, uint32_t local_rank_id) override;

  bool InitializeHccl();

  bool CreateCommunicationGroup(const std::string &group_name, const std::vector<uint32_t> &group_ranks,
                                uint32_t local_group_rank, uint32_t local_group_size) override;

  bool CreateDeviceCommunicationGroup(const std::string &group_name, const std::vector<uint32_t> &group_ranks) override;

  bool DestroyCommunicationGroup(const std::string &group_name) override;

  bool DestroyDeviceCommunicationGroup(const std::string &group_name) override;

  uint32_t GetRankId(const std::string &group_name) override;

  uint32_t GetGroupSize(const std::string &group_name) override;

  uint32_t GetLocalRankId(const std::string &group_name) override;

  uint32_t GetLocalGroupSize(const std::string &group_name) override;

  uint32_t GetWorldRankFromGroupRank(const std::string &group_name, uint32_t local_rank) override;

  uint32_t GetGroupRankFromWorldRank(uint32_t group_rank, const std::string &group_name) override;

  HcclComm HcclCommunicator(const std::string &group_name);

  std::string HcclInnerCommName(const std::string &group_name);

  bool DestroyHcclComm();

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

 private:
  AscendCollectiveCommLib();
  ~AscendCollectiveCommLib() override = default;
};
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_RUNTIME_HARDWARE_ASCEND_ASCEND_COLLECTIVE_COMM_LIB_H_
