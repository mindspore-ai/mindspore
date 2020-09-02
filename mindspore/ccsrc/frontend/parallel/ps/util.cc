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

#include "frontend/parallel/ps/util.h"
#include <unordered_map>
#include "frontend/parallel/ps/common.h"
#include "utils/ms_utils.h"

namespace mindspore {
namespace parallel {
namespace ps {
int Util::rank_id_ = -1;

std::unordered_map<std::string, int> Util::optimizer_to_ids{
  {kApplyMomentum, 0},
  {kSparseAdam, 1},
  {kSparseLazyAdam, 2},
  {kSparseFtrl, 3},
};

std::unordered_map<int, std::string> Util::id_to_optimizers{
  {0, kApplyMomentum},
  {1, kSparseAdam},
  {2, kSparseLazyAdam},
  {3, kSparseFtrl},
};

std::unordered_map<int, std::string> Util::id_to_optimizer_nodes{
  {0, kApplyMomentumOp},
  {1, kSparseAdamOp},
  {2, kSparseLazyAdamOp},
  {3, kSparseFtrlOp},
};

bool Util::IsParamServerMode() { return IsRoleOfWorker() || IsRoleOfPServer() || IsRoleOfScheduler(); }

bool Util::IsRoleOfWorker() {
  auto role = common::GetEnv(kEnvRole);
  if (strcmp(role.c_str(), kEnvRoleOfWorker) == 0) {
    return true;
  } else {
    return false;
  }
}

bool Util::IsRoleOfPServer() {
  auto role = common::GetEnv(kEnvRole);
  if (strcmp(role.c_str(), kEnvRoleOfPServer) == 0) {
    return true;
  } else {
    return false;
  }
}

bool Util::IsRoleOfScheduler() {
  auto role = common::GetEnv(kEnvRole);
  if (strcmp(role.c_str(), kEnvRoleOfScheduler) == 0) {
    return true;
  } else {
    return false;
  }
}

void Util::SetInternalEnvVar() {
  if (IsParamServerMode()) {
    auto comm_type = common::GetEnv(kEnvCommType);
    if (comm_type.size() > 0) {
      (void)common::SetEnv(kDmlcCommType, comm_type.c_str());
    }
    auto interface = common::GetEnv(kEnvInterface);
    if (interface.size() > 0) {
      (void)common::SetEnv(kDmlcInterface, interface.c_str());
    }
    auto server_num = common::GetEnv(kEnvPServerNum);
    if (server_num.size() > 0) {
      (void)common::SetEnv(kDmlcPServerNum, server_num.c_str());
    }
    auto worker_num = common::GetEnv(kEnvWorkerNum);
    if (worker_num.size() > 0) {
      (void)common::SetEnv(kDmlcWorkerNum, worker_num.c_str());
    }
    if (IsRoleOfScheduler()) {
      (void)common::SetEnv(kDmlcRole, kRoleOfScheduler);
    } else if (IsRoleOfPServer()) {
      (void)common::SetEnv(kDmlcRole, kRoleOfPServer);
    } else if (IsRoleOfWorker()) {
      (void)common::SetEnv(kDmlcRole, kRoleOfWorker);
    }
    auto scheduler_host = common::GetEnv(kEnvSchedulerHost);
    if (scheduler_host.size() > 0) {
      (void)common::SetEnv(kDmlcSchedulerHost, scheduler_host.c_str());
    }
    auto scheduler_port = common::GetEnv(kEnvSchedulerPort);
    if (scheduler_port.size() > 0) {
      (void)common::SetEnv(kDmlcSchedulerPort, scheduler_port.c_str());
    }
  }
}

int Util::optimizer_id(std::string name) {
  if (optimizer_to_ids.count(name) > 0) {
    return optimizer_to_ids[name];
  }
  return -1;
}

std::string Util::optimizer_name(int id) {
  if (id_to_optimizers.count(id) > 0) {
    return id_to_optimizers[id];
  }
  return "";
}

std::string Util::optimizer_node_name(int id) {
  if (id_to_optimizer_nodes.count(id) > 0) {
    return id_to_optimizer_nodes[id];
  }
  return "";
}

bool Util::is_optimizer(std::string name) { return optimizer_to_ids.count(name) > 0; }

int Util::LocalShard(int first_dim, int rank_id, int server_num) {
  std::map<int, int> shard_dims = AllRankLocalShard(first_dim, rank_id, server_num);
  if (shard_dims.count(rank_id) == 0) {
    MS_LOG(EXCEPTION) << "Invalid rank id " << rank_id;
  }
  return shard_dims[rank_id];
}

std::map<int, int> Util::AllRankLocalShard(int first_dim, int rank_id, int server_num) {
  if (rank_id >= server_num) {
    MS_LOG(EXCEPTION) << "The rank ID " << rank_id << " should be less than the number of servers " << server_num;
  }
  std::map<int, int> shard_dims;
  for (int i = 0; i < server_num; i++) {
    shard_dims[i] = 0;
  }
  if (server_num != static_cast<int>(shard_dims.size())) {
    MS_LOG(EXCEPTION) << "Inconsistent server num " << server_num << " shard dims counter size " << shard_dims.size();
  }
  int server_index = -1;
  for (int i = 0; i < first_dim; i++) {
    server_index = (server_index + 1) % server_num;
    shard_dims[server_index] = shard_dims[server_index] + 1;
  }
  if (shard_dims.count(rank_id) == 0) {
    MS_LOG(EXCEPTION) << "Invalid rank id " << rank_id << ", total server num " << server_num;
  }
  return shard_dims;
}

void Util::SetRankId(int rank_id) { rank_id_ = rank_id; }

int Util::GetRankId() { return rank_id_; }

void Util::ReduceSparseGradient(float *gradients, int *indices, const size_t indices_size, size_t segment_size,
                                const size_t first_dim_size, const size_t outer_dim_size,
                                mindspore::kernel::SparseGradient *unique_sparse_grad) {
  size_t slice_segment_size = indices_size * segment_size;
  auto workspace_grad = new float[slice_segment_size];
  auto workspace_indices = new int[indices_size];

  MS_EXCEPTION_IF_NULL(gradients);
  MS_EXCEPTION_IF_NULL(indices);
  MS_EXCEPTION_IF_NULL(workspace_grad);
  MS_EXCEPTION_IF_NULL(workspace_indices);

  mindspore::kernel::SparseGradient workspace_sparse_grad({workspace_grad, workspace_indices, indices_size});
  mindspore::kernel::SparseGradient input_sparse_grad({gradients, indices, indices_size});
  mindspore::kernel::ReduceSparseGradientParam param;
  param.input_grad_ = &input_sparse_grad;
  param.workspace_grad_ = &workspace_sparse_grad;
  param.output_grad_ = unique_sparse_grad;
  param.max_index_ = first_dim_size;
  param.value_stride_ = outer_dim_size;

  BucketReduceSparseGradient(param);
  delete[] workspace_grad;
  delete[] workspace_indices;
}
}  // namespace ps
}  // namespace parallel
}  // namespace mindspore
