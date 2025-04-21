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

#include "plugin/device/ascend/hal/hardware/lowlatency_collective_comm_lib.h"

namespace mindspore {
namespace device {
namespace ascend {
LowlatencyCollectiveCommLib::LowlatencyCollectiveCommLib() { global_group_name_ = kLCCLGlobalGroupName; }

bool LowlatencyCollectiveCommLib::Initialize(uint32_t global_rank, uint32_t global_rank_size, uint32_t local_rank_id) {
  if (initialized_) {
    return false;
  }

  global_rank_id_ = global_rank;
  global_rank_size_ = global_rank_size;
  local_rank_id_ = local_rank_id;
  initialized_ = true;
  finalized_ = false;
  return true;
}

bool LowlatencyCollectiveCommLib::CreateCommunicationGroup(const std::string &group_name,
                                                           const std::vector<uint32_t> &group_ranks,
                                                           uint32_t local_group_rank, uint32_t local_group_size) {
  CHECK_RET((groups_.count(group_name) == 0), true, "The LCCL group " + group_name + " has already existed.");

  LowlatencyCommunicationGroupPtr group = std::make_shared<LowlatencyCommunicationGroup>(
    group_name, group_ranks, global_rank_id_, local_group_rank, local_group_size);
  CHECK_IF_NULL(group);
  groups_[group_name] = group;
  return true;
}

int LowlatencyCollectiveCommLib::AllReduce(const LcclPtr &lccl_ptr, void *send_buff, void *recv_buff, size_t count,
                                           HcclDataType data_type, const HcclReduceOp reduce_op,
                                           const aclrtStream stream) {
  return lccl_ptr->AllReduce(send_buff, recv_buff, count, data_type, reduce_op, stream);
}

int LowlatencyCollectiveCommLib::AllGather(const LcclPtr &lccl_ptr, void *send_buff, void *recv_buff, size_t count,
                                           HcclDataType data_type, const aclrtStream stream) {
  return lccl_ptr->AllGather(send_buff, recv_buff, count, data_type, stream);
}

int LowlatencyCollectiveCommLib::ReduceScatter(const LcclPtr &lccl_ptr, void *send_buff, void *recv_buff, size_t count,
                                               HcclDataType data_type, const HcclReduceOp reduce_op,
                                               const aclrtStream stream) {
  return lccl_ptr->ReduceScatter(send_buff, recv_buff, count, data_type, reduce_op, stream);
}

int LowlatencyCollectiveCommLib::All2All(const LcclPtr &lccl_ptr, void *send_buff, void *recv_buff, size_t count,
                                         HcclDataType data_type, const aclrtStream stream) {
  return lccl_ptr->All2All(send_buff, recv_buff, count, data_type, stream);
}

int LowlatencyCollectiveCommLib::Broadcast(const LcclPtr &lccl_ptr, void *buff, size_t count, HcclDataType data_type,
                                           int root, const aclrtStream stream) {
  return lccl_ptr->Broadcast(buff, count, data_type, root, stream);
}

int LowlatencyCollectiveCommLib::MatmulAllReduce(const LcocPtr &lcoc_ptr, const CoCInputPkg &input_pkg,
                                                 const CoCOutputPkg &output_pkg, void *workspace,
                                                 const aclrtStream stream) {
  return lcoc_ptr->MatmulAllReduce(input_pkg, output_pkg, workspace, stream);
}

LcclPtr LowlatencyCollectiveCommLib::LcclCommunicator(const std::string &group_name) {
  CHECK_RET((groups_.count(group_name) != 0), true, "The LCCL group " + group_name + " does not existed.");
  auto group = std::dynamic_pointer_cast<LowlatencyCommunicationGroup>(groups_[group_name]);
  CHECK_IF_NULL(group);
  return group->lccl_communicator();
}

LcocPtr LowlatencyCollectiveCommLib::CreateLcocForOp(const std::string &group_name) {
  CHECK_RET((groups_.count(group_name) != 0), true, "The LCCL group " + group_name + " does not existed.");
  auto group = std::dynamic_pointer_cast<LowlatencyCommunicationGroup>(groups_[group_name]);
  CHECK_IF_NULL(group);

  LcalCommPtr lcal_comm = group->lcal_comm();
  CHECK_IF_NULL(lcal_comm);
  LcocPtr lcoc_ptr = std::make_shared<Lcoc>(*(lcal_comm.get()));
  return lcoc_ptr;
}

void LowlatencyCollectiveCommLib::SetParamForLcoc(const LcocPtr &lcoc_ptr, LcalType lcal_type, const CoCTiling &tiling,
                                                  const CoCParamDesc &param_desc) {
  lcoc_ptr->SetParam(lcal_type, tiling, param_desc);
}

int64_t LowlatencyCollectiveCommLib::GetLcocWorkspaceSize(const LcocPtr &lcoc_ptr) {
  return lcoc_ptr->GetWorkspaceSize();
}
}  // namespace ascend

using LowlatencyCollectiveCommLib = mindspore::device::ascend::LowlatencyCollectiveCommLib;

CollectiveCommunicationLib *communication_lib_instance() { return &LowlatencyCollectiveCommLib::GetInstance(); }

int AllReduce(const LcclPtr &lccl_ptr, void *send_buff, void *recv_buff, size_t count, HcclDataType data_type,
              const HcclReduceOp reduce_op, const aclrtStream stream) {
  return LowlatencyCollectiveCommLib::GetInstance().AllReduce(lccl_ptr, send_buff, recv_buff, count, data_type,
                                                              reduce_op, stream);
}

int AllGather(const LcclPtr &lccl_ptr, void *send_buff, void *recv_buff, size_t count, HcclDataType data_type,
              const aclrtStream stream) {
  return LowlatencyCollectiveCommLib::GetInstance().AllGather(lccl_ptr, send_buff, recv_buff, count, data_type, stream);
}

int ReduceScatter(const LcclPtr &lccl_ptr, void *send_buff, void *recv_buff, size_t count, HcclDataType data_type,
                  const HcclReduceOp reduce_op, const aclrtStream stream) {
  return LowlatencyCollectiveCommLib::GetInstance().ReduceScatter(lccl_ptr, send_buff, recv_buff, count, data_type,
                                                                  reduce_op, stream);
}

int All2All(const LcclPtr &lccl_ptr, void *send_buff, void *recv_buff, size_t count, HcclDataType data_type,
            const aclrtStream stream) {
  return LowlatencyCollectiveCommLib::GetInstance().All2All(lccl_ptr, send_buff, recv_buff, count, data_type, stream);
}

int Broadcast(const LcclPtr &lccl_ptr, void *buff, size_t count, HcclDataType data_type, int root,
              const aclrtStream stream) {
  return LowlatencyCollectiveCommLib::GetInstance().Broadcast(lccl_ptr, buff, count, data_type, root, stream);
}

int MatmulAllReduce(const LcocPtr &lcoc_ptr, const CoCInputPkg &input_pkg, const CoCOutputPkg &output_pkg,
                    void *workspace, const aclrtStream stream) {
  return LowlatencyCollectiveCommLib::GetInstance().MatmulAllReduce(lcoc_ptr, input_pkg, output_pkg, workspace, stream);
}

LcclPtr LcclCommunicator(const std::string &group_name) {
  return LowlatencyCollectiveCommLib::GetInstance().LcclCommunicator(group_name);
}

LcocPtr CreateLcocForOp(const std::string &group_name) {
  return LowlatencyCollectiveCommLib::GetInstance().CreateLcocForOp(group_name);
}

void SetParamForLcoc(const LcocPtr &lcoc_ptr, LcalType lcal_type, const CoCTiling &tiling,
                     const CoCParamDesc &param_desc) {
  LowlatencyCollectiveCommLib::GetInstance().SetParamForLcoc(lcoc_ptr, lcal_type, tiling, param_desc);
}

int64_t GetLcocWorkspaceSize(const LcocPtr &lcoc_ptr) {
  return LowlatencyCollectiveCommLib::GetInstance().GetLcocWorkspaceSize(lcoc_ptr);
}
}  // namespace device
}  // namespace mindspore
