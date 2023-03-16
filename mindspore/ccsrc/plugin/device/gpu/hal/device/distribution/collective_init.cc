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

#include "plugin/device/gpu/hal/device/distribution/collective_init.h"
#include "utils/log_adapter.h"
#include "utils/ms_utils.h"
#include "include/backend/distributed/init.h"

namespace mindspore {
namespace device {
namespace gpu {
CollectiveInitializer &CollectiveInitializer::instance() {
  static CollectiveInitializer instance = {};
  return instance;
}

bool CollectiveInitializer::collective_inited() const { return collective_inited_; }

const void *CollectiveInitializer::collective_handle() { return collective_handle_; }

void CollectiveInitializer::InitCollective() {
  if (common::UseMPI()) {
#ifndef _WIN32
    void *handle = dlopen("libgpu_collective.so", RTLD_LAZY);
    if (handle == nullptr) {
      MS_LOG(EXCEPTION)
        << "Loading libgpu_collective.so failed. Many reasons could cause this:\n"
           "1.libgpu_collective.so is not found, please check this MindSpore package is GPU version and built "
           "with distributed feature.\n"
           "2.NCCL is not found or the user-installed NCCL version installed is incompatible: MindSpore "
           "requires NCCL-2.7.6.\n"
           "3.OpenMPI is not found or the user-installed OpenMPI version is incompatible: MindSpore "
           "requires OpenMPI-4.0.3.\n";
    }
    auto mpi_init_funcptr = reinterpret_cast<InitMPI>(dlsym(handle, "InitMPI"));
    MS_EXCEPTION_IF_NULL(mpi_init_funcptr);
    (*mpi_init_funcptr)();

    // Because this method InitCollective is static, the non-static member variables should be accessed by
    // CollectiveInitializer::instance().
    CollectiveInitializer::instance().use_mpi_ = true;
    CollectiveInitializer::instance().collective_handle_ = handle;
#else
    MS_LOG(EXCEPTION) << "windows not support MPI.";
#endif
  } else {
    if (!distributed::Initialize()) {
      MS_LOG(EXCEPTION) << "Failed to initialize distributed execution for NCCL. Maybe the MindSpore cluster is not "
                           "successfully built. Please check schuduler and other nodes' log.";
    }
  }
  CollectiveInitializer::instance().collective_inited_ = true;
}

void CollectiveInitializer::FinalizeCollective() {
#ifndef _WIN32
  if (CollectiveInitializer::instance().collective_handle_ != nullptr) {
    if (dlclose(CollectiveInitializer::instance().collective_handle_) != 0) {
      MS_LOG(EXCEPTION) << "Closing libgpu_collective.so handle failed.";
    }
  }
#else
  MS_LOG(EXCEPTION) << "windows not support MPI.";
#endif
}

uint32_t CollectiveInitializer::GetRankID(const std::string &group_name) {
  return CollectiveInitializer::instance().GetRankIDByGroup(group_name);
}

uint32_t CollectiveInitializer::GetRankSize(const std::string &group_name) {
  return CollectiveInitializer::instance().GetGroupSize(group_name);
}

uint32_t CollectiveInitializer::local_rank_id() {
  uint32_t local_rank_id = 0;
  if (common::UseMPI()) {
#ifndef _WIN32
    MS_EXCEPTION_IF_NULL(collective_handle_);
    auto get_local_rank_funcptr =
      reinterpret_cast<GetLocalRankId>(dlsym(const_cast<void *>(collective_handle_), "local_rank_id"));
    MS_EXCEPTION_IF_NULL(get_local_rank_funcptr);
    local_rank_id = IntToUint((*get_local_rank_funcptr)());
#else
    MS_LOG(EXCEPTION) << "windows not support MPI.";
#endif
  } else {
    local_rank_id = distributed::collective::CollectiveManager::instance()->local_rank_id();
  }
  return local_rank_id;
}

bool CollectiveInitializer::CreateCommunicationGroup(const std::string &group_name,
                                                     const std::vector<uint32_t> &group_ranks) {
  if (common::UseMPI()) {
#ifndef _WIN32
    MS_EXCEPTION_IF_NULL(collective_handle_);
    auto create_comm_group_funcptr =
      reinterpret_cast<CreateCommGroupFunc>(dlsym(const_cast<void *>(collective_handle_), "CreateCommGroup"));
    MS_EXCEPTION_IF_NULL(create_comm_group_funcptr);
    return (*create_comm_group_funcptr)(group_name, group_ranks);
#else
    MS_LOG(EXCEPTION) << "windows not support MPI.";
#endif
  } else {
    // There's only one scheduler in cluster. It shouldn't have collective operations.
    if (distributed::cluster::ClusterContext::instance()->node_role() == distributed::kEnvRoleOfScheduler) {
      return true;
    }
    return distributed::collective::CollectiveManager::instance()->CreateCommunicationGroup(group_name, group_ranks);
  }
}

bool CollectiveInitializer::DestroyCommunicationGroup(const std::string &group_name) {
  if (common::UseMPI()) {
#ifndef _WIN32
    MS_EXCEPTION_IF_NULL(collective_handle_);
    auto destroy_group_funcptr =
      reinterpret_cast<DestroyGroupFunc>(dlsym(const_cast<void *>(collective_handle_), "DestroyGroup"));
    MS_EXCEPTION_IF_NULL(destroy_group_funcptr);
    return (*destroy_group_funcptr)(group_name);
#else
    MS_LOG(EXCEPTION) << "windows not support MPI.";
#endif
  } else {
    // There's only one scheduler in cluster. It shouldn't have collective operations.
    if (distributed::cluster::ClusterContext::instance()->node_role() == distributed::kEnvRoleOfScheduler) {
      return true;
    }
    return distributed::collective::CollectiveManager::instance()->DestroyCommunicationGroup(group_name);
  }
}

uint32_t CollectiveInitializer::GetRankIDByGroup(const std::string &group_name) {
  if (common::UseMPI()) {
#ifndef _WIN32
    MS_EXCEPTION_IF_NULL(collective_handle_);
    auto get_rank_id_funcptr =
      reinterpret_cast<GetRankIDByGroupFunc>(dlsym(const_cast<void *>(collective_handle_), "GetRankIDByGroup"));
    MS_EXCEPTION_IF_NULL(get_rank_id_funcptr);
    return IntToUint((*get_rank_id_funcptr)(group_name));
#else
    MS_LOG(EXCEPTION) << "windows not support MPI.";
#endif
  } else {
    // There's only one scheduler in cluster. It shouldn't have collective operations.
    if (distributed::cluster::ClusterContext::instance()->node_role() == distributed::kEnvRoleOfScheduler) {
      return 0;
    }
    return distributed::collective::CollectiveManager::instance()->GetRankId(group_name);
  }
}

uint32_t CollectiveInitializer::GetGroupSize(const std::string &group_name) {
  if (common::UseMPI()) {
#ifndef _WIN32
    MS_EXCEPTION_IF_NULL(collective_handle_);
    auto get_group_size_funcptr =
      reinterpret_cast<GetGroupSizeFunc>(dlsym(const_cast<void *>(collective_handle_), "GetGroupSize"));
    MS_EXCEPTION_IF_NULL(get_group_size_funcptr);
    return IntToUint((*get_group_size_funcptr)(group_name));
#else
    MS_LOG(EXCEPTION) << "windows not support MPI.";
#endif
  } else {
    // There's only one scheduler in cluster. It shouldn't have collective operations.
    if (distributed::cluster::ClusterContext::instance()->node_role() == distributed::kEnvRoleOfScheduler) {
      return 0;
    }
    return distributed::collective::CollectiveManager::instance()->GetGroupSize(group_name);
  }
}
}  // namespace gpu
}  // namespace device
}  // namespace mindspore
