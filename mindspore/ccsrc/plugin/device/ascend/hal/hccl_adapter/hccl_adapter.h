/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_RUNTIME_HCCL_ADAPTER_HCCL_ADAPTER_H
#define MINDSPORE_RUNTIME_HCCL_ADAPTER_HCCL_ADAPTER_H

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <mutex>
#include "mindspore/core/ir/anf.h"
#include "hccl/hccl_types.h"
#include "plugin/device/ascend/hal/hccl_adapter/plugin/hccl_plugin.h"
#include "plugin/device/ascend/hal/device/distribute/ascend_collective.h"
using HcclCollectiveGroup = mindspore::device::ascend::collective::HcclCollectiveGroup;

namespace ge {
class OpsKernelInfoStore;
class OpsKernelvBuilder;
}  // namespace ge

namespace mindspore::hccl {
struct HcclTaskInfo {
  std::string private_def;
  int64_t workspace_size;
  int64_t stream_num;
};

struct HcclAllToAllVParams {
  std::vector<uint64_t> sendcounts;
  std::vector<uint64_t> sdispls;
  std::vector<uint64_t> recvcounts;
  std::vector<uint64_t> rdispls;
};

enum HcclMode { kGraph, kPynative, kKernelByKernel };

class HcclAdapter {
 public:
  static HcclAdapter &GetInstance();

  // common
  bool InitHccl(uint32_t device_id, std::string_view rank_id, std::string_view rank_file, HcclMode hccl_mode);
  bool InitHccl(uint32_t device_id, std::string_view rank_id);
  bool FinalizeHccl();
  const bool Inited() const { return init_flag_; }
  void SetHcclComm(HcclComm comm) {
    if (hccl_comm_ == nullptr) hccl_comm_ = comm;
  }
  HcclComm get_hccl_comm() const { return hccl_comm_; }
  HcclResult HcclCreateGroup(const std::string &group, uint32_t rank_num, uint32_t *rank_ids) const;
  HcclResult HcclDestroyGroup(const std::string &group) const;
  HcclResult HcclGetRankId(const std::string &group, uint32_t *rank_id) const;
  HcclResult HcclGetRankSize(const std::string &group, uint32_t *rank_size) const;

  HcclResult HcclGetRankId(uint32_t *rank_id) const;
  HcclResult HcclGetRankSize(uint32_t *rank_size) const;

  HcclResult HcclGetLocalRankId(const std::string &group, uint32_t *lcoal_rank_id) const;
  HcclResult HcclGetLocalRankSize(const std::string &group, uint32_t *local_rank_size) const;
  HcclResult HcclGetWorldRankFromGroupRank(const std::string &group, uint32_t local_rank, uint32_t *world_rank) const;
  HcclResult HcclGetGroupRankFromWorldRank(uint32_t world_rank, const std::string &group, uint32_t *local_rank) const;

  // for ge node
  bool GenTask(const AnfNodePtr &node, HcclDataType datatype, std::vector<HcclTaskInfo> *task_info_lists) const;
  int64_t CalcWorkspaceSize(const AnfNodePtr &node, HcclDataType datatype) const;
  void *GetHcclOpsKernelInfoStore() const;
  static std::string GetHcclType(const AnfNodePtr &node);

  // for single op
  HcclResult HcclBroadcast(void *buf, uint64_t count, HcclDataType dataType, uint32_t root, aclrtStream stream,
                           const std::string &group = "") const;
  HcclResult HcclAllReduce(void *send_buf, void *recv_buf, uint64_t count, HcclDataType dataType, HcclReduceOp op,
                           const aclrtStream stream, const std::string &group = "") const;
  HcclResult HcclAllGather(void *send_buf, void *recv_buf, uint64_t count, HcclDataType dataType,
                           const aclrtStream stream, const std::string &group = "") const;
  HcclResult HcclReduceScatter(void *send_buf, void *recv_buf, uint64_t count, HcclDataType dataType, HcclReduceOp op,
                               const aclrtStream stream, const std::string &group = "") const;
  HcclResult HcclSend(void *send_buf, uint64_t count, HcclDataType dataType, uint32_t destRank,
                      const aclrtStream stream, const std::string &group = "") const;
  HcclResult HcclRecv(void *recv_buf, uint64_t count, HcclDataType dataType, uint32_t srcRank, const aclrtStream stream,
                      const std::string &group = "") const;
  HcclResult HcclAllToAll(void *send_buf, void *recv_buf, hccl::HcclAllToAllVParams params, HcclDataType dataType,
                          const aclrtStream stream, const std::string &group) const;

  // for enqueue op
  HcclResult HcclExecEnqueueOp(const ::HcomOperation &op_info, const HExecCallBack &callback) const;
  HcclResult HcclExecAllToAllv(const ::HcomAllToAllVParams &params, const HExecCallBack &callback) const;

 private:
  HcclAdapter() = default;
  ~HcclAdapter() = default;
  void InitPlugin();
  void FinalizePlugin();

  HcclComm GetHcomm(const std::string &group) const {
    if (common::UseMPI()) {
      return HcclCollectiveGroup::instance().GetGroupComm(group);
    } else if (hccl_comm_ != nullptr) {
      return hccl_comm_;
    } else {
      MS_LOG(EXCEPTION) << "Couldn't get correct hccl hcom with group " << group;
    }
  }

  bool InitKernelInfoStore(const std::map<std::string, std::string> options);
  bool FinalizeKernelInfoStore();

  bool InitHcclComm(std::string_view rank_id, std::string_view rank_file);
  bool FinalizeHcclComm();

  bool InitHcclExec();
  bool FinalizeHcclExec();

  HcclMode GetCurrentHcclMode() const;
  void CheckExcutionMode() const;
  static std::string GetHcclModeString(HcclMode hccl_mode);

  void *plugin_handle_ = nullptr;

  InitHcomGraphAdapterFunObj init_hcom_graph_adapter_ = nullptr;
  FinalizeHcomGraphAdapterFunObj finalize_hcom_graph_adapter_ = nullptr;
  GetHcclKernelInfoStoreFunObj get_hccl_kernel_info_store_ = nullptr;
  GetAllKernelBuilderFunObj get_all_kernel_builder_ = nullptr;

  HcclCommInitClusterInfoFunObj init_hccl_comm_ = nullptr;
  HcclCommDestroyFunObj finalize_hccl_comm_ = nullptr;
  HcclBroadcastFunObj launch_hccl_broadcast_ = nullptr;
  HcclAllReduceFunObj launch_hccl_all_reduce_ = nullptr;
  HcclReduceScatterFunObj launch_hccl_reduce_scatter_ = nullptr;
  HcclAllGatherFunObj launch_hccl_all_gather_ = nullptr;
  HcclSendFunObj launch_hccl_send_ = nullptr;
  HcclRecvFunObj launch_hccl_recv_ = nullptr;
  HcclGetRankIdFunObj single_op_hccl_get_rank_id_ = nullptr;
  HcclGetRankSizeFunObj single_op_hccl_get_rank_size_ = nullptr;
  HcclAlltoAllVFunObj launch_hccl_all_to_allv_ = nullptr;

  HcomCreateGroupFunObj hccl_create_group_ = nullptr;
  HcomDestroyGroupFunObj hccl_destroy_group_ = nullptr;
  HcomGetRankIdFunObj hccl_get_rank_id_ = nullptr;
  HcomGetRankSizeFunObj hccl_get_rank_size_ = nullptr;
  HcomGetLocalRankIdFunObj hccl_get_local_rank_id_ = nullptr;
  HcomGetLocalRankSizeFunObj hccl_get_local_rank_size_ = nullptr;
  HcomGetWorldRankFromGroupRankFunObj hccl_get_world_rank_by_group_rank_ = nullptr;
  HcomGetGroupRankFromWorldRankFunObj hccl_get_group_rank_by_world_rank_ = nullptr;

  HcomExecInitializeFunObj hccl_exec_initialize_ = nullptr;
  HcomExecFinalizeFunObj hccl_exec_finalize_ = nullptr;
  HcomExecEnqueueOperationFunObj hccl_exec_enqueue_op_ = nullptr;
  HcomExecEnqueueAllToAllVFunObj hccl_exec_enqueue_all_to_all_v_ = nullptr;

  HcclComm hccl_comm_ = nullptr;

  std::shared_ptr<::ge::OpsKernelInfoStore> ops_kernel_info_store_ = nullptr;
  std::shared_ptr<::ge::OpsKernelBuilder> ops_kernel_builder_ = nullptr;

  bool init_flag_ = false;
  HcclMode hccl_mode_ = HcclMode::kGraph;
  std::mutex init_mutex_;
};
}  // namespace mindspore::hccl
#endif  // MINDSPORE_RUNTIME_HCCL_ADAPTER_HCCL_ADAPTER_H
