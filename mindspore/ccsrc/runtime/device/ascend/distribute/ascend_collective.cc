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

#include "runtime/device/ascend/distribute/ascend_collective.h"
#include "utils/log_adapter.h"

static constexpr const char *kAscendCollectiveFileName = "libascend_collective.so";
namespace mindspore {
namespace device {
namespace ascend {
namespace collective {
HcclCollectiveGroup &HcclCollectiveGroup::instance() {
  static HcclCollectiveGroup instance = {};
  return instance;
}

void HcclCollectiveGroup::FinalizeCollective() {
  MS_LOG(INFO) << "Finalize Collective";
  if (collective_handle_ != nullptr) {
    MS_EXCEPTION_IF_NULL(finalize_mpi_);
    finalize_mpi_();
    if (dlclose(collective_handle_) != 0) {
      MS_LOG(EXCEPTION) << "Closing libascend_collective.so handle failed.";
    }
  }
}

bool HcclCollectiveGroup::InitCollective() {
  MS_LOG(INFO) << "InitCollective";
  if (inited_) {
    return true;
  }
  collective_handle_ = dlopen(kAscendCollectiveFileName, RTLD_NOW);
  if (collective_handle_ == nullptr) {
    MS_LOG(EXCEPTION)
      << "Loading libascend_collective.so failed. Many reasons could cause this:\n1.libascend_collective.so is not "
         "installed.\n2.hccl is not "
         "installed or found.\n3.mpi is not installed or found, please check if lib files of OpenMPI is added to "
         "LD_LIBRATY_PATH.";
  }
  init_mpi_ = DlsymFuncObj(InitMPI, collective_handle_);
  finalize_mpi_ = DlsymFuncObj(FinalizeMPI, collective_handle_);
  get_group_comm_ = DlsymFuncObj(GetGroupComm, collective_handle_);
  get_group_size_ = DlsymFuncObj(GetGroupSize, collective_handle_);
  get_rank_id_by_group_ = DlsymFuncObj(GetRankIdByGroup, collective_handle_);
  get_device_id_ = DlsymFuncObj(GetDeviceId, collective_handle_);
  create_comm_for_group_ = DlsymFuncObj(CreateCommForGroup, collective_handle_);
  destroy_hccl_comm_ = DlsymFuncObj(DestroyHcclComm, collective_handle_);
  MS_EXCEPTION_IF_NULL(init_mpi_);
  init_mpi_();
  inited_ = true;
  MS_LOG(INFO) << "InitCollective success";
  return true;
}
HcclComm HcclCollectiveGroup::GetGroupComm(const std::string &name) {
  MS_EXCEPTION_IF_NULL(get_group_comm_);
  return get_group_comm_(name);
}
int HcclCollectiveGroup::GetRankSize(const std::string &name) const {
  MS_EXCEPTION_IF_NULL(get_group_size_);
  return get_group_size_(name);
}
int HcclCollectiveGroup::GetRankId(const std::string &name) const {
  MS_EXCEPTION_IF_NULL(get_rank_id_by_group_);
  return get_rank_id_by_group_(name);
}
int HcclCollectiveGroup::GetDeviceId() const {
  MS_EXCEPTION_IF_NULL(get_device_id_);
  return get_device_id_();
}
void HcclCollectiveGroup::CreateCommGroup(const std::string &name, const std::vector<unsigned int> &ranks) {
  MS_EXCEPTION_IF_NULL(create_comm_for_group_);
  (void)create_comm_for_group_(name, ranks);
}
void HcclCollectiveGroup::DestroyCommGroup() {
  MS_EXCEPTION_IF_NULL(destroy_hccl_comm_);
  destroy_hccl_comm_();
}
}  // namespace collective
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
