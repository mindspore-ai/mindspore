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

#ifndef MINDSPORE_CCSRC_DEVICE_CPU_MPI_MPI_ADAPTER_H_
#define MINDSPORE_CCSRC_DEVICE_CPU_MPI_MPI_ADAPTER_H_
#ifdef ENABLE_MPI
#include <mpi.h>
#include <vector>
#include <map>
#include <string>
#include <mutex>
#endif  // ENABLE_MPI
#include <memory>

namespace mindspore {
namespace device {
namespace cpu {
#ifndef FUNC_EXPORT
#define FUNC_EXPORT __attribute__((visibility("default")))
#endif

constexpr auto kOpTypeSum = "sum";
class MPIAdapter {
 public:
  FUNC_EXPORT static std::shared_ptr<MPIAdapter> Instance();
  FUNC_EXPORT int GetRankId() const { return rank_id_; }
  FUNC_EXPORT int GetRankSize() const { return rank_size_; }
#ifdef ENABLE_MPI
  FUNC_EXPORT ~MPIAdapter();
  FUNC_EXPORT bool ReduceScatter(const float *input, float *output, const std::vector<int> &ranks_group,
                                 size_t data_num, const std::string &op_type = kOpTypeSum);
  FUNC_EXPORT bool ReduceScatterOverwriteInput(float *input, const std::vector<int> &ranks_group, size_t in_data_num,
                                               size_t output_size, const std::string &op_type = kOpTypeSum,
                                               float *output = nullptr);
  FUNC_EXPORT bool AllGather(const float *input, float *output, const std::vector<int> &ranks_group, size_t data_num);
#else
  FUNC_EXPORT ~MPIAdapter() = default;
#endif  // ENABLE_MPI

 private:
#ifdef ENABLE_MPI
  MPIAdapter();
  void Init();
  MPI_Group AddGroup(const std::vector<int> &ranks);

  MPI_Group comm_group_world_;
  // key:ranks group, value: mpi group
  std::map<std::vector<int>, MPI_Group> ranks_group_;
  std::mutex group_mutex_;
#else
  MPIAdapter() = default;
#endif  // ENABLE_MPI
  int rank_id_{-1};
  int rank_size_{0};

  static std::shared_ptr<MPIAdapter> instance_;
};
}  // namespace cpu
}  // namespace device
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_DEVICE_CPU_MPI_MPI_ADAPTER_H_
