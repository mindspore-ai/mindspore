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

namespace mindspore {
namespace device {
namespace cpu {
constexpr auto kOpTypeSum = "sum";
class MPIAdapter {
 public:
  ~MPIAdapter();
  static MPIAdapter &Instance();
  int GetRankId() const;
  bool ReduceScatter(const float *input, float *output, const std::vector<int> &ranks_group, size_t data_num,
                     const std::string &op_type = kOpTypeSum);
  bool ReduceScatterOverwriteInput(float *input, const std::vector<int> &ranks_group, size_t input_data_num,
                                   size_t output_size, const std::string &op_type = kOpTypeSum,
                                   float *output = nullptr);
  bool AllGather(const float *input, float *output, const std::vector<int> &ranks_group, size_t data_num);

 private:
  MPIAdapter();
  void Init();
  MPI_Group AddGroup(const std::vector<int> &ranks);

  int rank_id_;
  int rank_size_;
  MPI_Group comm_group_world_;
  // key:ranks group, value: mpi group
  std::map<std::vector<int>, MPI_Group> ranks_group_;
  std::mutex group_mutex_;
};
}  // namespace cpu
}  // namespace device
}  // namespace mindspore
#endif  // ENABLE_MPI
#endif  // MINDSPORE_CCSRC_DEVICE_CPU_MPI_MPI_ADAPTER_H_
