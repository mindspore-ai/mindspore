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

#include "device/cpu/mpi/mpi_adapter.h"
#ifdef ENABLE_MPI
#include <algorithm>
#include <sstream>
#include "pybind11/pybind11.h"
#endif  // ENABLE_MPI
#include "utils/log_adapter.h"

namespace mindspore {
namespace device {
namespace cpu {
std::shared_ptr<MPIAdapter> MPIAdapter::instance_ = nullptr;
std::shared_ptr<MPIAdapter> MPIAdapter::Instance() {
  if (instance_ == nullptr) {
    MS_LOG(DEBUG) << "Create new mpi adapter instance.";
    instance_.reset(new (std::nothrow) MPIAdapter());
  }
  return instance_;
}

#ifdef ENABLE_MPI

#define RAISE_EXCEPTION(message)                                    \
  {                                                                 \
    std::ostringstream oss;                                         \
    oss << "[" << __FILE__ << "] [" << __LINE__ << "] " << message; \
    pybind11::pybind11_fail(oss.str());                             \
  }

#define RAISE_EXCEPTION_WITH_PARAM(message, param)                           \
  {                                                                          \
    std::ostringstream oss;                                                  \
    oss << "[" << __FILE__ << "] [" << __LINE__ << "] " << message << param; \
    pybind11::pybind11_fail(oss.str());                                      \
  }

namespace {
MPI_Op GetMpiOp(const std::string &op_type) {
  if (op_type == "sum") {
    return MPI_SUM;
  } else if (op_type == "max") {
    return MPI_MAX;
  } else if (op_type == "min") {
    return MPI_MIN;
  } else if (op_type == "prod") {
    return MPI_PROD;
  }

  RAISE_EXCEPTION_WITH_PARAM("unsupport op_type: ", op_type);
  return MPI_SUM;
}

int GetScatterIndex(int rankid, const std::vector<int> &ranks_group) {
  int scatter_index = -1;
  for (size_t i = 0; i < ranks_group.size(); ++i) {
    if (ranks_group[i] == rankid) {
      scatter_index = static_cast<int>(i);
      break;
    }
  }
  if (scatter_index == -1) {
    RAISE_EXCEPTION_WITH_PARAM("local rankid does not in the input rank group!local rank id:", rankid);
  }
  return scatter_index;
}
}  // namespace

MPIAdapter::MPIAdapter() : comm_group_world_(MPI_GROUP_NULL) { Init(); }

MPIAdapter::~MPIAdapter() {
  int finalized;
  MPI_Finalized(&finalized);
  if (finalized != 0) {
    return;
  }

  for (auto iter = ranks_group_.begin(); iter != ranks_group_.end(); ++iter) {
    MPI_Group_free(&iter->second);
  }
  ranks_group_.clear();
  if (comm_group_world_ != MPI_GROUP_NULL) {
    MPI_Group_free(&comm_group_world_);
    comm_group_world_ = MPI_GROUP_NULL;
  }
  MPI_Finalize();
}

void MPIAdapter::Init() {
  static bool init = false;
  if (init) {
    return;
  }

  int init_flag = 0;
  if (MPI_Initialized(&init_flag) != MPI_SUCCESS) {
    RAISE_EXCEPTION("Check mpi initialized fail!");
  }
  if (init_flag == 0) {
    auto ret = MPI_Init(nullptr, nullptr);
    if (ret != MPI_SUCCESS) {
      RAISE_EXCEPTION("Failed to init mpi!");
    }
  }

  MPI_Comm_group(MPI_COMM_WORLD, &comm_group_world_);
  if (comm_group_world_ == MPI_GROUP_NULL) {
    RAISE_EXCEPTION("comm_group_world_ init fail!");
  }
  auto ret = MPI_Comm_rank(MPI_COMM_WORLD, &rank_id_);
  if (ret != MPI_SUCCESS) {
    RAISE_EXCEPTION("Failed to init mpi rank id!");
  }

  ret = MPI_Comm_size(MPI_COMM_WORLD, &rank_size_);
  if (ret != MPI_SUCCESS) {
    RAISE_EXCEPTION_WITH_PARAM("Failed to init mpi rank size!rankid:", rank_id_)
  }
  init = true;
}

MPI_Group MPIAdapter::AddGroup(const std::vector<int> &ranks) {
  if (ranks.size() > static_cast<size_t>(rank_size_) || ranks.empty()) {
    RAISE_EXCEPTION_WITH_PARAM("input rank size:", ranks.size());
  }

  if (std::find(ranks.begin(), ranks.end(), rank_id_) == ranks.end()) {
    RAISE_EXCEPTION_WITH_PARAM("local rankid does not in the input rank group!local rank id:", rank_id_);
  }
  std::lock_guard<std::mutex> lock(group_mutex_);
  auto iter = ranks_group_.find(ranks);
  if (iter != ranks_group_.end()) {
    return iter->second;
  }
  const auto ranks_size = ranks.size();
  std::vector<int> ranks_input(ranks_size, 0);
  for (size_t i = 0; i < ranks_size; ++i) {
    ranks_input[i] = ranks[i];
  }

  MPI_Group group = MPI_GROUP_NULL;
  MPI_Group_incl(comm_group_world_, ranks.size(), ranks_input.data(), &group);
  if (group == MPI_GROUP_NULL) {
    RAISE_EXCEPTION_WITH_PARAM("create mpi group fail!rankid:", rank_id_)
  }

  ranks_group_[ranks] = group;
  return group;
}

bool MPIAdapter::ReduceScatter(const float *input, float *output, const std::vector<int> &ranks_group, size_t data_num,
                               const std::string &op_type) {
  if (ranks_group.empty()) {
    RAISE_EXCEPTION("input rank group is empty!");
    return false;
  }

  auto group = AddGroup(ranks_group);
  if (group == MPI_GROUP_NULL) {
    RAISE_EXCEPTION_WITH_PARAM("Get mpi group fail!rankid:", rank_id_)
  }
  MPI_Comm comm;
  MPI_Comm_create_group(MPI_COMM_WORLD, group, 0, &comm);
  if (comm == MPI_COMM_NULL) {
    RAISE_EXCEPTION_WITH_PARAM("create mpi comm fail!rankid:", rank_id_);
  }
  std::vector<int> receive_count(ranks_group.size(), 0);
  for (size_t i = 0; i < ranks_group.size(); ++i) {
    receive_count[i] = data_num;
  }

  auto op = GetMpiOp(op_type);
  auto ret = MPI_Reduce_scatter(input, output, receive_count.data(), MPI_FLOAT, op, comm);
  bool result = true;
  if (ret != MPI_SUCCESS) {
    RAISE_EXCEPTION_WITH_PARAM("mpi reduce_scatter fail!ret = ", ret);
    result = false;
  }

  ret = MPI_Comm_free(&comm);
  if (ret != MPI_SUCCESS) {
    RAISE_EXCEPTION_WITH_PARAM("mpi comm free fail! ret = ", ret);
  }
  return result;
}

bool MPIAdapter::ReduceScatterOverwriteInput(float *input, const std::vector<int> &ranks_group, size_t input_data_num,
                                             size_t output_size, const std::string &op_type, float *output) {
  int scatter_index = GetScatterIndex(rank_id_, ranks_group);
  auto group = AddGroup(ranks_group);
  if (group == MPI_GROUP_NULL) {
    RAISE_EXCEPTION_WITH_PARAM("Get mpi group fail!rankid:", rank_id_);
  }
  MPI_Comm comm;
  MPI_Comm_create_group(MPI_COMM_WORLD, group, 0, &comm);
  if (comm == MPI_COMM_NULL) {
    RAISE_EXCEPTION_WITH_PARAM("create mpi comm fail!rankid:", rank_id_);
  }

  MPI_Win window;
  auto ret = MPI_Win_create(input, input_data_num * sizeof(float), sizeof(float), MPI_INFO_NULL, comm, &window);
  if (ret != MPI_SUCCESS) {
    RAISE_EXCEPTION_WITH_PARAM("mpi window create fail! ret = ", ret);
  }
  MPI_Win_fence(0, window);
  for (size_t i = 0; i < ranks_group.size(); ++i) {
    int remote_rank = ranks_group[i];
    if (rank_id_ == remote_rank) {
      continue;
    }
    auto op = GetMpiOp(op_type);
    ret = MPI_Accumulate(input + i * input_data_num, input_data_num, MPI_FLOAT, remote_rank, i * input_data_num,
                         input_data_num, MPI_FLOAT, op, window);
    if (ret != MPI_SUCCESS) {
      RAISE_EXCEPTION_WITH_PARAM("mpi accumulate fail!ret = ", ret);
    }
  }
  MPI_Win_fence(0, window);
  if (output != nullptr) {
    auto data_size = input_data_num * sizeof(float);
    if (output_size < data_size) {
      std::ostringstream exception_msg;
      exception_msg << "output buffer size " << output_size << " < input size " << data_size;
      RAISE_EXCEPTION(exception_msg.str())
    }
    auto copy_ret = memcpy_s(output, output_size, input + scatter_index * input_data_num, data_size);
    if (copy_ret != 0) {
      RAISE_EXCEPTION_WITH_PARAM("copy output memory fail!ret = ", copy_ret);
    }
  }
  MPI_Win_free(&window);
  MPI_Comm_free(&comm);
  return true;
}

bool MPIAdapter::AllGather(const float *input, float *output, const std::vector<int> &ranks_group, size_t data_num) {
  if (ranks_group.empty()) {
    RAISE_EXCEPTION("input rank group is empty!");
    return false;
  }
  auto group = AddGroup(ranks_group);
  if (group == MPI_GROUP_NULL) {
    RAISE_EXCEPTION_WITH_PARAM("Get mpi group fail! rankid:", rank_id_);
  }
  MPI_Comm comm;
  MPI_Comm_create_group(MPI_COMM_WORLD, group, 0, &comm);
  if (comm == MPI_COMM_NULL) {
    RAISE_EXCEPTION_WITH_PARAM("create mpi comm fail! rankid:", rank_id_);
  }

  auto ret = MPI_Allgather(input, data_num, MPI_FLOAT, output, data_num, MPI_FLOAT, comm);

  if (ret != MPI_SUCCESS) {
    RAISE_EXCEPTION_WITH_PARAM("mpi allgater fail!ret = ", ret);
  }
  ret = MPI_Comm_free(&comm);
  if (ret != MPI_SUCCESS) {
    RAISE_EXCEPTION_WITH_PARAM("mpi comm free fail!ret = ", ret);
  }
  return true;
}
#endif  // ENABLE_MPI
}  // namespace cpu
}  // namespace device
}  // namespace mindspore
