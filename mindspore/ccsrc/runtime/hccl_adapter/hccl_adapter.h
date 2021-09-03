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
#include <memory>
#include <mutex>
#include "mindspore/core/ir/anf.h"
#include "hccl/hccl_types.h"
#include "runtime/hccl_adapter/plugin/hccl_plugin.h"
#include "runtime/device/ascend/distribute/ascend_collective.h"
using HcclCollectiveGroup = mindspore::device::ascend::collective::HcclCollectiveGroup;

namespace ge {
class OpsKernelInfoStore;
class OpsKernelBuilder;
}  // namespace ge

namespace mindspore::hccl {
struct HcclTaskInfo {
  std::string private_def;
  int64_t workspace_size;
  int64_t stream_num;
};

class HcclAdapter {
 public:
  static HcclAdapter &GetInstance();

  // common
  bool InitHccl(uint32_t device_id, std::string_view rank_id, std::string_view rank_file, bool is_graph_mode);
  bool InitHccl();
  bool FinalizeHccl();
  const bool Inited() const { return init_flag_; }

  HcclResult HcclCreateGroup(const std::string &group, uint32_t rank_num, uint32_t *rank_ids) const;
  HcclResult HcclDestroyGroup(const std::string &group) const;
  HcclResult HcclGetRankId(const std::string &group, uint32_t *rank_id) const;
  HcclResult HcclGetRankSize(const std::string &group, uint32_t *rank_size) const;

  HcclResult HcclGetRankId(uint32_t *rank_id) const;
  HcclResult HcclGetRankSize(uint32_t *rank_size) const;

  // for ge node
  bool GenTask(const AnfNodePtr &node, HcclDataType datatype, std::vector<HcclTaskInfo> *task_info_lists) const;
  int64_t CalcWorkspaceSize(const AnfNodePtr &node, HcclDataType datatype) const;
  void *GetHcclOpsKernelInfoStore() const;
  static std::string GetHcclType(const AnfNodePtr &node);

  // for single op
  HcclResult HcclBroadcast(void *buf, uint64_t count, HcclDataType dataType, uint32_t root, aclrtStream stream) const;
  HcclResult HcclAllReduce(void *send_buf, void *recv_buf, uint64_t count, HcclDataType dataType, HcclReduceOp op,
                           aclrtStream stream, const std::string &group = "") const;
  HcclResult HcclAllGather(void *send_buf, void *recv_buf, uint64_t count, HcclDataType dataType, aclrtStream stream,
                           const std::string &group = "") const;
  HcclResult HcclReduceScatter(void *send_buf, void *recv_buf, uint64_t count, HcclDataType dataType, HcclReduceOp op,
                               aclrtStream stream, const std::string &group = "") const;
  HcclResult HcclSend(void *send_buf, uint64_t count, HcclDataType dataType, uint32_t destRank, aclrtStream stream,
                      const std::string &group = "") const;
  HcclResult HcclRecv(void *recv_buf, uint64_t count, HcclDataType dataType, uint32_t srcRank, aclrtStream stream,
                      const std::string &group = "") const;

  // for enqueue op
  HcclResult HcclExecEnqueueOp(const ::HcomOperation &op_info, const HExecCallBack &callback) const;
  HcclResult HcclExecAllToAllv(const ::HcomAllToAllVParams &params, const HExecCallBack &callback) const;

 private:
  HcclAdapter() = default;
  ~HcclAdapter() = default;
  void InitPlugin();
  void FinalizePlugin();

  HcclComm GetHcomm(const std::string &group) const {
    if (hccl_comm_ != nullptr) {
      return hccl_comm_;
    } else {
      return HcclCollectiveGroup::instance().GetGroupComm(group);
    }
  }

  bool InitKernelInfoStore(uint32_t device_id, std::string_view rank_id, std::string_view rank_file);
  bool FinalizeKernelInfoStore();

  bool InitHcclComm(std::string_view rank_id, std::string_view rank_file);
  bool FinalizeHcclComm();

  bool InitHcclExec();
  bool FinalizeHcclExec();

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

  HcomCreateGroupFunObj hccl_create_group_ = nullptr;
  HcomDestroyGroupFunObj hccl_destroy_group_ = nullptr;
  HcomGetRankIdFunObj hccl_get_rank_id_ = nullptr;
  HcomGetRankSizeFunObj hccl_get_rank_size_ = nullptr;

  HcomExecInitializeFunObj hccl_exec_initialize_ = nullptr;
  HcomExecFinalizeFunObj hccl_exec_finalize_ = nullptr;
  HcomExecEnqueueOperationFunObj hccl_exec_enqueue_op_ = nullptr;
  HcomExecEnqueueAllToAllVFunObj hccl_exec_enqueue_all_to_all_v_ = nullptr;

  HcclComm hccl_comm_ = nullptr;

  std::shared_ptr<::ge::OpsKernelInfoStore> ops_kernel_info_store_ = nullptr;
  std::shared_ptr<::ge::OpsKernelBuilder> ops_kernel_builder_ = nullptr;

  bool init_flag_ = false;
  bool is_graph_mode_ = false;
  std::mutex init_mutex_;
};
}  // namespace mindspore::hccl
#endif  // MINDSPORE_RUNTIME_HCCL_ADAPTER_HCCL_ADAPTER_H
