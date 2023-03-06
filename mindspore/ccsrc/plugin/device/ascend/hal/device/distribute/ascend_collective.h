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

#ifndef MINDSPORE_MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_DISTRIBUTE_ASCEND_COLLECTIVE_H
#define MINDSPORE_MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_DISTRIBUTE_ASCEND_COLLECTIVE_H

#include <vector>
#include <string>
#include <map>
#include "hccl/hccl_types.h"
#include "include/common/utils/utils.h"
#include "utils/dlopen_macro.h"
#include "include/backend/visible.h"

namespace mindspore {
namespace device {
namespace ascend {
namespace collective {

ORIGIN_METHOD(InitMPI, void);
ORIGIN_METHOD(FinalizeMPI, void);
ORIGIN_METHOD(GetGroupComm, HcclComm, const std::string &);
ORIGIN_METHOD(GetGroupSize, int, const std::string &);
ORIGIN_METHOD(GetRankIdByGroup, int, const std::string &);
ORIGIN_METHOD(GetDeviceId, int);
ORIGIN_METHOD(CreateCommForGroup, bool, const std::string &, const std::vector<unsigned int> &);
ORIGIN_METHOD(DestroyHcclComm, void);

class HcclCollectiveGroup {
 public:
  HcclCollectiveGroup(HcclCollectiveGroup const &) = delete;
  HcclCollectiveGroup &operator=(const HcclCollectiveGroup &) = delete;
  static HcclCollectiveGroup &instance();
  bool InitCollective();
  void FinalizeCollective();
  HcclComm GetGroupComm(const std::string &name);
  int GetDeviceId() const;
  int GetRankId(const std::string &name = kHcclWorldGroup) const;
  int GetRankSize(const std::string &name = kHcclWorldGroup) const;
  void CreateCommGroup(const std::string &name, const std::vector<unsigned int> &ranks);
  void DestroyCommGroup();
  const void *collective_handle() const { return collective_handle_; }

 private:
  HcclCollectiveGroup() = default;
  ~HcclCollectiveGroup() = default;
  bool inited_ = false;
  void *collective_handle_ = nullptr;
  InitMPIFunObj init_mpi_ = nullptr;
  FinalizeMPIFunObj finalize_mpi_ = nullptr;
  GetGroupCommFunObj get_group_comm_ = nullptr;
  GetGroupSizeFunObj get_group_size_ = nullptr;
  GetRankIdByGroupFunObj get_rank_id_by_group_ = nullptr;
  GetDeviceIdFunObj get_device_id_ = nullptr;
  CreateCommForGroupFunObj create_comm_for_group_ = nullptr;
  DestroyHcclCommFunObj destroy_hccl_comm_ = nullptr;
};
}  // namespace collective
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_DISTRIBUTE_ASCEND_COLLECTIVE_H
