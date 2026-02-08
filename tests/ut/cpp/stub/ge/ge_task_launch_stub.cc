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
#include "plugin/ascend/res_manager/hccl_adapter/hccl_adapter.h"

namespace mindspore {
namespace hccl {
HcclAdapter &HcclAdapter::GetInstance() {
  static HcclAdapter instance;
  return instance;
}
bool HcclAdapter::InitHccl(uint32_t, std::string_view) { return true; }
bool HcclAdapter::InitHccl(uint32_t, std::string_view, std::string_view, HcclMode) { return true; }
bool HcclAdapter::FinalizeHccl() { return true; }
bool HcclAdapter::HcclWatchdogThread(HcclComm comm, std::string *error_info, bool *disable) { return true; }
HcclResult HcclAdapter::HcclGetRankId(const std::string &, uint32_t *) const { return HCCL_SUCCESS; }
HcclResult HcclAdapter::HcclGetRankSize(const std::string &, uint32_t *) const { return HCCL_SUCCESS; }
HcclResult HcclAdapter::HcclBroadcast(void *, uint64_t, HcclDataType, uint32_t, aclrtStream, HcclComm) const {
  return HCCL_SUCCESS;
}
HcclResult HcclAdapter::HcclAllReduce(void *, void *, uint64_t, HcclDataType, HcclReduceOp, aclrtStream,
                                      HcclComm) const {
  return HCCL_SUCCESS;
}
HcclResult HcclAdapter::HcclAllGather(void *, void *, uint64_t, HcclDataType, aclrtStream, HcclComm) const {
  return HCCL_SUCCESS;
}
HcclResult HcclAdapter::HcclReduceScatter(void *, void *, uint64_t, HcclDataType, HcclReduceOp, aclrtStream,
                                          HcclComm) const {
  return HCCL_SUCCESS;
}
HcclResult HcclAdapter::HcclSend(void *, uint64_t, HcclDataType, uint32_t, aclrtStream, HcclComm) const {
  return HCCL_SUCCESS;
}
HcclResult HcclAdapter::HcclRecv(void *, uint64_t, HcclDataType, uint32_t, aclrtStream, HcclComm) const {
  return HCCL_SUCCESS;
}
HcclResult HcclAdapter::HcclAlltoAllV(void *, void *, hccl::HcclAllToAllVParams, HcclDataType, aclrtStream,
                                      HcclComm) const {
  return HCCL_SUCCESS;
}
HcclResult HcclAdapter::HcclAllToAll(void *, void *, hccl::HcclAllToAllParams, HcclDataType, aclrtStream,
                                     HcclComm) const {
  return HCCL_SUCCESS;
}

bool HcclAdapter::UseHcclCM() const { return false; }

bool HcclAdapter::IsSameServer(const std::vector<uint32_t> &rank_ids) const { return false; }

std::string HcclAdapter::GetHcomGroup(const CNodePtr &) const { return ""; }

HcclResult HcclAdapter::HcclCommWorkingDevNicSet(HcclComm comm, uint32_t *ranks, bool *use_backup, uint32_t nRanks) {
  return HCCL_SUCCESS;
}
}  // namespace hccl
}  // namespace mindspore
