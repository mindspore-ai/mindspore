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

#include "plugin/ascend/res_manager/hccl_adapter/plugin/hccl_plugin.h"
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <mutex>
#include "ir/anf.h"
#include "hccl/hccl_types.h"
#include "include/runtime/hardware_abstract/kernel_base/kernel.h"
#include "plugin/ascend/res_manager/visible.h"

using mindspore::kernel::KernelTensor;

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

struct HcclAllGatherVParams {
  uint64_t send_count;
  std::vector<uint64_t> recv_counts;
  std::vector<uint64_t> rdispls;
};

struct HcclReduceScatterVParams {
  std::vector<uint64_t> send_counts;
  std::vector<uint64_t> sdispls;
  uint64_t recv_count;
};

struct HcclAllToAllParams {
  uint64_t sendcount;
  uint64_t recvcount;
};

enum HcclMode { kGraph, kPynative, kKernelByKernel };

class ASCEND_RES_MANAGER_EXPORT HcclAdapter {
 public:
  static HcclAdapter &GetInstance();

  // common
  bool InitHccl(uint32_t device_id, std::string_view rank_id, std::string_view rank_file, HcclMode hccl_mode);
  bool InitHccl(uint32_t device_id, std::string_view rank_id);
  uint32_t HcclGetCommConfigCapability();
  HcclResult HcclCommInitClusterInfoConfig(const char *rank_table, uint32_t rank_id, HcclCommConfig *config,
                                           HcclComm *hccl_comm_);
  HcclResult HcclCommInitRootInfoConfig(uint32_t n_ranks, const HcclRootInfo *root_info, uint32_t rank,
                                        const HcclCommConfig *config, HcclComm *hccl_comm_);
  HcclResult HcclCreateSubCommConfig(HcclComm *global_comm, uint32_t rank_size, uint32_t *rank_ids, uint64_t comm_id,
                                     uint32_t rank_id, HcclCommConfig *config, HcclComm *hccl_comm_);
  bool FinalizeHccl();
  bool HcclWatchdogThread(HcclComm comm, std::string *error_info, bool *ret);
  const bool Inited() const { return init_flag_; }
  const HcclComm get_hccl_comm() const { return hccl_comm_; }
  HcclResult HcclCreateGroup(const std::string &group, uint32_t rank_num, uint32_t *rank_ids) const;
  HcclResult HcclDestroyGroup(const std::string &group) const;
  HcclResult HcclGetRankId(const std::string &group, uint32_t *rank_id) const;
  HcclResult HcclGetRankSize(const std::string &group, uint32_t *rank_size) const;
  HcclResult HcclGetLocalRankId(const std::string &group, uint32_t *lcoal_rank_id) const;
  HcclResult HcclGetLocalRankSize(const std::string &group, uint32_t *local_rank_size) const;
  HcclResult HcclGetWorldRankFromGroupRank(const std::string &group, uint32_t local_rank, uint32_t *world_rank) const;
  HcclResult HcclGetGroupRankFromWorldRank(uint32_t world_rank, const std::string &group, uint32_t *local_rank) const;
  // for single op
  HcclResult HcclBroadcast(void *buf, uint64_t count, HcclDataType dataType, uint32_t root, aclrtStream stream,
                           HcclComm comm) const;
  HcclResult HcclAllReduce(void *send_buf, void *recv_buf, uint64_t count, HcclDataType dataType, HcclReduceOp op,
                           const aclrtStream stream, HcclComm comm) const;
  HcclResult HcclReduce(void *send_buf, void *recv_buf, uint64_t count, HcclDataType dataType, HcclReduceOp op,
                        uint32_t root, const aclrtStream stream, HcclComm comm) const;
  HcclResult HcclScatter(void *send_buf, void *recv_buf, uint64_t count, HcclDataType dataType, uint32_t root,
                         HcclComm comm, aclrtStream stream) const;
  HcclResult HcclAllGather(void *send_buf, void *recv_buf, uint64_t count, HcclDataType dataType,
                           const aclrtStream stream, HcclComm comm) const;
  HcclResult HcclReduceScatter(void *send_buf, void *recv_buf, uint64_t count, HcclDataType dataType, HcclReduceOp op,
                               const aclrtStream stream, HcclComm comm) const;
  HcclResult HcclSend(void *send_buf, uint64_t count, HcclDataType dataType, uint32_t destRank,
                      const aclrtStream stream, HcclComm comm) const;
  HcclResult HcclRecv(void *recv_buf, uint64_t count, HcclDataType dataType, uint32_t srcRank, const aclrtStream stream,
                      HcclComm comm) const;
  HcclResult HcclAlltoAllV(void *send_buf, void *recv_buf, hccl::HcclAllToAllVParams params, HcclDataType dataType,
                           const aclrtStream stream, HcclComm comm) const;

  HcclResult HcclAlltoAllVC(void *send_buf, void *send_count_matrix, HcclDataType send_type, void *recv_buf,
                            HcclDataType recv_type, aclrtStream stream, HcclComm hccl_comm) const;

  HcclResult HcclReduceScatterV(void *send_buf, void *recv_buf, hccl::HcclReduceScatterVParams params,
                                HcclDataType data_type, const HcclReduceOp op, const aclrtStream stream,
                                HcclComm hccl_comm) const;

  HcclResult HcclAllGatherV(void *send_buf, void *recv_buf, hccl::HcclAllGatherVParams params, HcclDataType data_type,
                            const aclrtStream stream, HcclComm hccl_comm) const;

  HcclResult HcclAllToAll(void *send_buf, void *recv_buf, hccl::HcclAllToAllParams params, HcclDataType dataType,
                          const aclrtStream stream, HcclComm comm) const;
  HcclResult HcclBarrier(const aclrtStream stream, HcclComm comm) const;
  HcclResult HcclBatchISendIRecv(HcclSendRecvItem *sendRecvInfo, uint32_t itemNum, HcclComm comm,
                                 aclrtStream stream) const;

  // for enqueue op
  HcclResult HcclExecEnqueueOp(const ::HcomOperation &op_info, const HExecCallBack &callback) const;
  HcclResult HcclExecAlltoAllV(const ::HcomAllToAllVParams &params, const HExecCallBack &callback) const;

  HcclResult HcclCommResume(HcclComm comm) const;

  HcclResult HcclCommWorkingDevNicSet(HcclComm comm, uint32_t *ranks, bool *useBackup, uint32_t nRanks);

  // Return whether using CM to initialize HCCL.
  bool UseHcclCM() const;
  static void AddCMEnvToHcclOption(std::map<std::string, std::string> *hccl_opt_map);

  bool IsSameServer(const std::vector<uint32_t> &rank_ids) const;

  string GetHcomGroup(const CNodePtr &cnode) const;

 private:
  HcclAdapter() = default;
  ~HcclAdapter() = default;
  void InitPlugin();
  void FinalizePlugin();

  bool InitKernelInfoStore(const std::map<std::string, std::string> options);
  bool FinalizeKernelInfoStore();

  bool InitHcclComm(std::string_view rank_id, std::string_view rank_file);
  bool FinalizeHcclComm();

  bool InitHcclExec();
  bool FinalizeHcclExec();

  static std::string GetHcclModeString(HcclMode hccl_mode);
  string DoGetHcomGroup(const string &original_group, const std::vector<uint32_t> &rank_ids) const;

  static bool IsSimulation();
  void *plugin_handle_ = nullptr;

  InitHcomGraphAdapterFunObj init_hcom_graph_adapter_ = nullptr;
  FinalizeHcomGraphAdapterFunObj finalize_hcom_graph_adapter_ = nullptr;
  GetHcclKernelInfoStoreFunObj get_hccl_kernel_info_store_ = nullptr;
  GetAllKernelBuilderFunObj get_all_kernel_builder_ = nullptr;
  HcomDestroyFunObj hcom_destroy_ = nullptr;

  HcclGetCommConfigCapabilityFunObj get_hccl_comm_config_capability_ = nullptr;
  HcclCommInitClusterInfoFunObj init_hccl_comm_ = nullptr;
  HcclCommInitClusterInfoConfigFunObj init_hccl_global_comm_ranktable_ = nullptr;
  HcclCommInitRootInfoConfigFunObj init_hccl_root_info_config_ = nullptr;
  HcclCreateSubCommConfigFunObj init_hccl_sub_comm_ranktable_ = nullptr;
  HcclCommDestroyFunObj finalize_hccl_comm_ = nullptr;
  HcclBroadcastFunObj launch_hccl_broadcast_ = nullptr;
  HcclAllReduceFunObj launch_hccl_all_reduce_ = nullptr;
  HcclReduceFunObj launch_hccl_reduce_ = nullptr;
  HcclScatterFunObj launch_hccl_scatter_ = nullptr;
  HcclReduceScatterFunObj launch_hccl_reduce_scatter_ = nullptr;
  HcclAllGatherFunObj launch_hccl_all_gather_ = nullptr;
  HcclSendFunObj launch_hccl_send_ = nullptr;
  HcclRecvFunObj launch_hccl_recv_ = nullptr;
  HcclBarrierFunObj launch_hccl_barrier_ = nullptr;
  HcclGetRankIdFunObj single_op_hccl_get_rank_id_ = nullptr;
  HcclGetRankSizeFunObj single_op_hccl_get_rank_size_ = nullptr;
  HcclAlltoAllVFunObj launch_hccl_all_to_allv_ = nullptr;
  HcclAlltoAllVCFunObj launch_hccl_all_to_allvc_ = nullptr;
  HcclReduceScatterVFunObj launch_hccl_reduce_scatterv_ = nullptr;
  HcclAllGatherVFunObj launch_hccl_all_gatherv_ = nullptr;
  HcclAlltoAllFunObj launch_hccl_all_to_all_ = nullptr;
  HcclBatchSendRecvFunObj launch_hccl_batch_isend_irecv_ = nullptr;
  HcclCommResumeFunObj launch_hccl_comm_resume_ = nullptr;
  HcclGetCommAsyncErrorFunObj hccl_get_comm_async_error_ = nullptr;
  HcclGetErrorStringFunObj hccl_get_error_string_ = nullptr;
  HcomCreateGroupFunObj hccl_create_group_ = nullptr;
  HcomDestroyGroupFunObj hccl_destroy_group_ = nullptr;
  HcomGetRankIdFunObj hccl_get_rank_id_ = nullptr;
  HcomGetRankSizeFunObj hccl_get_rank_size_ = nullptr;
  HcomGetLocalRankIdFunObj hccl_get_local_rank_id_ = nullptr;
  HcomGetLocalRankSizeFunObj hccl_get_local_rank_size_ = nullptr;
  HcomGetWorldRankFromGroupRankFunObj hccl_get_world_rank_by_group_rank_ = nullptr;
  HcomGetGroupRankFromWorldRankFunObj hccl_get_group_rank_by_world_rank_ = nullptr;
  HcclCommWorkingDevNicSetFunObj hccl_comm_working_dev_nic_set_ = nullptr;

  HcomExecInitializeFunObj hccl_exec_initialize_ = nullptr;
  HcomExecFinalizeFunObj hccl_exec_finalize_ = nullptr;
  HcomExecEnqueueOperationFunObj hccl_exec_enqueue_op_ = nullptr;
  HcomExecEnqueueAllToAllVFunObj hccl_exec_enqueue_all_to_all_v_ = nullptr;

  HcclComm hccl_comm_ = nullptr;

  std::shared_ptr<::ge::OpsKernelInfoStore> ops_kernel_info_store_ = nullptr;
  std::shared_ptr<::ge::OpsKernelBuilder> ops_kernel_builder_ = nullptr;

  bool init_flag_ = false;
  bool init_kernel_info_store_ = false;
  bool init_hccl_exec_ = false;
  HcclMode hccl_mode_ = HcclMode::kGraph;
  std::mutex init_mutex_;
};
}  // namespace mindspore::hccl
#endif  // MINDSPORE_RUNTIME_HCCL_ADAPTER_HCCL_ADAPTER_H
