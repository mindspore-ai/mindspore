/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include <vector>
#include "runtime/hccl_adapter/hccl_adapter.h"

namespace mindspore {
namespace hccl {
HcclAdapter &HcclAdapter::GetInstance() {
  static HcclAdapter instance;
  return instance;
}
bool HcclAdapter::InitHccl() { return true; }
bool HcclAdapter::InitHccl(uint32_t, std::string_view, std::string_view, bool) { return true; }
bool HcclAdapter::FinalizeHccl() { return true; }
HcclResult HcclAdapter::HcclCreateGroup(const std::string &, uint32_t, uint32_t *) const { return HCCL_SUCCESS; }
HcclResult HcclAdapter::HcclDestroyGroup(const std::string &) const { return HCCL_SUCCESS; }
HcclResult HcclAdapter::HcclGetRankId(const std::string &, uint32_t *) const { return HCCL_SUCCESS; }
HcclResult HcclAdapter::HcclGetRankSize(const std::string &, uint32_t *) const { return HCCL_SUCCESS; }
HcclResult HcclAdapter::HcclGetRankId(uint32_t *rank_id) const { return HCCL_SUCCESS; }
HcclResult HcclAdapter::HcclGetRankSize(uint32_t *rank_size) const { return HCCL_SUCCESS; }
bool HcclAdapter::GenTask(const AnfNodePtr &, HcclDataType, std::vector<HcclTaskInfo> *) const { return true; }
int64_t HcclAdapter::CalcWorkspaceSize(const AnfNodePtr &, HcclDataType) const { return 0; }
void *HcclAdapter::GetHcclOpsKernelInfoStore() const { return nullptr; }
std::string HcclAdapter::GetHcclType(const AnfNodePtr &) { return ""; }
HcclResult HcclAdapter::HcclBroadcast(void *, uint64_t, HcclDataType, uint32_t, aclrtStream) const {
  return HCCL_SUCCESS;
}
HcclResult HcclAdapter::HcclAllReduce(void *, void *, uint64_t, HcclDataType, HcclReduceOp, aclrtStream,
                                      const std::string &) const {
  return HCCL_SUCCESS;
}
HcclResult HcclAdapter::HcclAllGather(void *, void *, uint64_t, HcclDataType, aclrtStream, const std::string &) const {
  return HCCL_SUCCESS;
}
HcclResult HcclAdapter::HcclReduceScatter(void *, void *, uint64_t, HcclDataType, HcclReduceOp, aclrtStream,
                                          const std::string &) const {
  return HCCL_SUCCESS;
}
HcclResult HcclAdapter::HcclSend(void *, uint64_t, HcclDataType, uint32_t, aclrtStream, const std::string &) const {
  return HCCL_SUCCESS;
}
HcclResult HcclAdapter::HcclRecv(void *, uint64_t, HcclDataType, uint32_t, aclrtStream, const std::string &) const {
  return HCCL_SUCCESS;
}
HcclResult HcclAdapter::HcclExecEnqueueOp(const ::HcomOperation &op_info, const HExecCallBack &callback) const {
  return HCCL_SUCCESS;
}
}  // namespace hccl
}  // namespace mindspore
