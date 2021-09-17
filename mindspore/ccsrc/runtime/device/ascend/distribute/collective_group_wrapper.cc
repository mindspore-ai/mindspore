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

#include "runtime/device/ascend/distribute/collective_group_wrapper.h"

extern "C" {
void InitMPI() { (void)MPICollective::instance().Init(); }
void FinalizeMPI() { MPICollective::instance().FinalizeMPI(); }
int GetRankIdByGroup(const std::string &name) { return MPICollective::instance().GetRankIdByGroup(name); }
int GetGroupSize(const std::string &name) { return MPICollective::instance().GetGroupSize(name); }
int GetDeviceId() { return MPICollective::instance().GetDeviceId(); }
HcclComm GetGroupComm(const std::string &name) { return MPICollective::instance().GetGroupComm(name); }
bool CreateCommForGroup(const std::string &name, const std::vector<unsigned int> &ranks) {
  return MPICollective::instance().CreateCommGroup(name, ranks);
}
void DestroyHcclComm() { MPICollective::instance().DestroyHcclComm(); }
}
