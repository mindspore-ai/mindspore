/**
 * Copyright 2026 Huawei Technologies Co., Ltd
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

#include "plugin/ascend/res_manager/collective/ascend_collective_comm_lib.h"

namespace mindspore {
namespace device {
namespace ascend {
AscendCollectiveCommLib &AscendCollectiveCommLib::GetInstance() {
  static AscendCollectiveCommLib instance;
  return instance;
}

AscendCollectiveCommLib::AscendCollectiveCommLib() = default;

bool AscendCollectiveCommLib::Initialize(uint32_t, uint32_t, uint32_t) { return true; }
bool AscendCollectiveCommLib::Finalize() { return true; }
bool AscendCollectiveCommLib::InitializeHccl() { return true; }

bool AscendCollectiveCommLib::CreateCommunicationGroup(const std::string &, const std::vector<uint32_t> &, uint32_t,
                                                       uint32_t, const GroupOptions &) {
  return true;
}
bool AscendCollectiveCommLib::CreateDeviceCommunicationGroup(const std::string &, const std::vector<uint32_t> &) {
  return true;
}
bool AscendCollectiveCommLib::DestroyCommunicationGroup(const std::string &) { return true; }
bool AscendCollectiveCommLib::DestroyDeviceCommunicationGroup(const std::string &) { return true; }
uint32_t AscendCollectiveCommLib::GetRankId(const std::string &) { return 0; }
uint32_t AscendCollectiveCommLib::GetGroupSize(const std::string &) { return 1; }
uint32_t AscendCollectiveCommLib::GetLocalRankId(const std::string &) { return 0; }
uint32_t AscendCollectiveCommLib::GetLocalGroupSize(const std::string &) { return 1; }
uint32_t AscendCollectiveCommLib::GetWorldRankFromGroupRank(const std::string &, uint32_t local_rank) { return local_rank; }
uint32_t AscendCollectiveCommLib::GetGroupRankFromWorldRank(uint32_t group_rank, const std::string &) {
  return group_rank;
}
HcclComm AscendCollectiveCommLib::HcclCommunicator(const std::string &) { return nullptr; }
std::string AscendCollectiveCommLib::CommName(const std::string &group_name) { return group_name; }
void AscendCollectiveCommLib::SaveDefaultGroupInfo() {}
void AscendCollectiveCommLib::RestoreDefaultGroupInfo() {}
HcclComm AscendCollectiveCommLib::GetHcomByGroup(const std::string &) { return nullptr; }
bool AscendCollectiveCommLib::DestroyHcclComm() { return true; }
bool AscendCollectiveCommLib::ResumeHcclComm() { return true; }
bool AscendCollectiveCommLib::CommSwitchNic(const std::vector<uint32_t> &, const std::vector<bool> &) { return true; }
bool AscendCollectiveCommLib::AllGather(const void *, void *, size_t, TypeId, const std::string &, void *) { return true; }
bool AscendCollectiveCommLib::AllReduce(const void *, void *, size_t, TypeId, CollectiveOpReduceType, const std::string &,
                                        void *) {
  return true;
}
bool AscendCollectiveCommLib::Broadcast(const void *, void *, size_t, TypeId, uint32_t, const std::string &, void *) {
  return true;
}
bool AscendCollectiveCommLib::ReduceScatter(const void *, void *, size_t, TypeId, CollectiveOpReduceType,
                                            const std::string &, void *) {
  return true;
}
bool AscendCollectiveCommLib::Send(const void *, size_t, TypeId, uint32_t, const std::string &, void *) { return true; }
bool AscendCollectiveCommLib::Recv(void *, size_t, TypeId, uint32_t, const std::string &, void *) { return true; }
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
