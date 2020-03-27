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

#include "device/gpu/distribution/mpi_wrapper.h"

#include <cuda_runtime_api.h>
#include <string>
#include "device/gpu/distribution/nccl_wrapper.h"

namespace mindspore {
namespace device {
namespace gpu {
MPIWrapper::MPIWrapper() : rank_id_(0), rank_size_(0), local_rank_id_(0) { Init(); }

MPIWrapper::~MPIWrapper() {
  int finalized;
  MPI_Finalized(&finalized);
  if (finalized == 0) {
    MPI_Finalize();
  }
}

MPIWrapper &MPIWrapper::instance() {
  static MPIWrapper instance;
  return instance;
}

int MPIWrapper::local_rank_id() const { return local_rank_id_; }

void MPIWrapper::Init() {
  int initialized;
  CHECK_RET(MPI_Initialized(&initialized), MPI_SUCCESS, "Failed to check mpi initialization status.");

  if (initialized == 0) {
    MPI_Init(nullptr, nullptr);
  }
  CHECK_RET(MPI_Comm_rank(MPI_COMM_WORLD, &rank_id_), MPI_SUCCESS, "Failed to init mpi rank id.");
  CHECK_RET(MPI_Comm_size(MPI_COMM_WORLD, &rank_size_), MPI_SUCCESS, "Failed to init mpi rank size.");
  NCCLWrapper::instance().set_rank(rank_id_, rank_size_);
  AssignLocalRankId();

  ncclUniqueId unique_id;
  if (rank_id_ == 0) {
    unique_id = NCCLWrapper::instance().nccl_unique_id();
  }
  CHECK_RET(MPI_Bcast(reinterpret_cast<void *>(&unique_id), sizeof(unique_id), MPI_BYTE, 0, MPI_COMM_WORLD),
            MPI_SUCCESS, "Failed to broadcast nccl unique id.");
  NCCLWrapper::instance().set_nccl_unique_id(unique_id);
  return;
}

void MPIWrapper::AssignLocalRankId() {
  char host_name[MAX_HOSTNAME_LEN] = {0};
  CHECK_RET(gethostname(host_name, MAX_HOSTNAME_LEN), 0, "Getting host name failed.");
  size_t host_hash = std::hash<std::string>()(host_name);

  const int kRankSize = rank_size_;
  size_t all_host_hashs[kRankSize];
  all_host_hashs[rank_id_] = host_hash;
  CHECK_RET(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, all_host_hashs, sizeof(size_t), MPI_BYTE, MPI_COMM_WORLD),
            MPI_SUCCESS, "MPI_Allgather host hashs failed.");
  for (int global_rank = 0; global_rank < kRankSize; global_rank++) {
    if (global_rank == rank_id_) {
      break;
    }
    if (all_host_hashs[global_rank] == all_host_hashs[rank_id_]) {
      local_rank_id_++;
    }
  }
  return;
}
}  // namespace gpu
}  // namespace device
}  // namespace mindspore
