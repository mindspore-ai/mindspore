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
#include "runtime/hccl_adapter/plugin/hccl_plugin.h"
#define google ascend_private
#include "register/ops_kernel_builder_registry.h"
#include "common/opskernel/ops_kernel_info_store.h"
#undef google
#include "hccl/hcom.h"

static constexpr const char *kHcclOpsKernelInfoStore = "ops_kernel_info_hccl";

extern "C" {
ge::Status Initialize(const std::map<std::string, std::string> &);
ge::Status Finalize();
void GetOpsKernelInfoStores(std::map<std::string, std::shared_ptr<ge::OpsKernelInfoStore>> &);

ge::Status PluginInitHcomGraphAdapter(const std::map<std::string, std::string> &options) {
  return ::Initialize(options);
}

ge::Status PluginFinalizeHcomGraphAdapter() { return ::Finalize(); }

void PluginGetHcclKernelInfoStore(std::shared_ptr<ge::OpsKernelInfoStore> *hccl_kernel_info_store) {
  if (hccl_kernel_info_store == nullptr) {
    return;
  }

  std::map<std::string, std::shared_ptr<ge::OpsKernelInfoStore>> all_ops_kernel_info_stores;
  ::GetOpsKernelInfoStores(all_ops_kernel_info_stores);
  for (auto &[name, ptr] : all_ops_kernel_info_stores) {
    if (name == kHcclOpsKernelInfoStore) {
      *hccl_kernel_info_store = ptr;
      return;
    }
  }

  *hccl_kernel_info_store = nullptr;
}

void PluginGetAllKernelBuilder(std::map<std::string, ge::OpsKernelBuilderPtr> *all_ops_kernel_builder) {
  if (all_ops_kernel_builder == nullptr) {
    return;
  }

  *all_ops_kernel_builder = ge::OpsKernelBuilderRegistry::GetInstance().GetAll();
}

HcclResult PluginLaunchHcclBroadcast(void *buf, uint64_t count, HcclDataType data_type, uint32_t root, HcclComm comm,
                                     aclrtStream stream) {
  return HcclBroadcast(buf, count, data_type, root, comm, stream);
}

HcclResult PluginLaunchHcclAllReduce(void *send_buf, void *recv_buf, uint64_t count, HcclDataType data_type,
                                     HcclReduceOp op, HcclComm comm, aclrtStream stream) {
  return HcclAllReduce(send_buf, recv_buf, count, data_type, op, comm, stream);
}

HcclResult PluginInitHcclComm(const char *cluster_info, uint32_t rank, HcclComm *comm) {
  return HcclCommInitClusterInfo(cluster_info, rank, comm);
}

HcclResult PluginFinalizeHcclComm(HcclComm comm) { return HcclCommDestroy(comm); }

HcclResult PluginHcclCreateGroup(const char *group, uint32_t rank_num, uint32_t *rank_ids) {
  return HcomCreateGroup(group, rank_num, rank_ids);
}

HcclResult PluginHcclDestroyGroup(const char *group) { return HcomDestroyGroup(group); }
HcclResult PluginHcclGetRankId(const char *group, uint32_t *rank_id) { return HcomGetRankId(group, rank_id); }
HcclResult PluginHcclGetRankSize(const char *group, uint32_t *rank_size) { return HcomGetRankSize(group, rank_size); }

HcclResult PluginHcclExecInitialize() { return HcomExecInitialize(); }
HcclResult PluginHcclExecFinalize() { return HcomExecFinalize(); }
HcclResult PluginHcclExecEnqueueOp(const ::HcomOperation &op_info, HExecCallBack callback) {
  return HcomExecEnqueueOperation(op_info, callback);
}
};  // extern C
