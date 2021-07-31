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

#ifndef MINDSPORE_MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_DISTRIBUTE_COLLECTIVE_GROUP_WRAPPER_H
#define MINDSPORE_MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_DISTRIBUTE_COLLECTIVE_GROUP_WRAPPER_H

#include <vector>
#include <string>
#include "runtime/device/ascend/distribute/mpi_collective_group.h"
#ifndef EXPORT_WRAPPER
#define EXPORT_WRAPPER __attribute__((visibility("default")))
#endif
using MPICollective = mindspore::device::ascend::collective::MPICollective;

extern "C" EXPORT_WRAPPER void InitMPI();
extern "C" EXPORT_WRAPPER void FinalizeMPI();
extern "C" EXPORT_WRAPPER int GetRankIdByGroup(const std::string &name);
extern "C" EXPORT_WRAPPER int GetGroupSize(const std::string &name);
extern "C" EXPORT_WRAPPER int GetDeviceId();
extern "C" EXPORT_WRAPPER HcclComm GetGroupComm(const std::string &name);
extern "C" EXPORT_WRAPPER bool CreateCommForGroup(const std::string &name, const std::vector<unsigned int> &ranks);
extern "C" EXPORT_WRAPPER void DestroyHcclComm();
#endif  // MINDSPORE_MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_DISTRIBUTE_COLLECTIVE_GROUP_WRAPPER_H
