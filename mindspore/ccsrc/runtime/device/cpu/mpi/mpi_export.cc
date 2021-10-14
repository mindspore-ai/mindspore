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
#include "runtime/device/cpu/mpi/mpi_export.h"
#include <vector>
#include "runtime/device/cpu/mpi/mpi_adapter.h"

extern "C" {
int GetMPIRankId() {
  auto inst = mindspore::device::cpu::MPIAdapter::Instance();
  if (inst == nullptr) {
    return 0;
  }
  return inst->GetRankId();
}

int GetMPIRankSize() {
  auto inst = mindspore::device::cpu::MPIAdapter::Instance();
  if (inst == nullptr) {
    return 0;
  }
  return inst->GetRankSize();
}

bool MPIReduceScatter(const float *input, float *output, const std::vector<int> &ranks_group, size_t data_num,
                      const std::string &op_type) {
  auto inst = mindspore::device::cpu::MPIAdapter::Instance();
  if (inst == nullptr) {
    return false;
  }
  return inst->ReduceScatter(input, output, ranks_group, data_num, op_type);
}

bool MPIReduceScatterOverwriteInput(float *input, const std::vector<int> &ranks_group, size_t in_data_num,
                                    size_t output_size, const std::string &op_type, float *output) {
  auto inst = mindspore::device::cpu::MPIAdapter::Instance();
  if (inst == nullptr) {
    return false;
  }
  return inst->ReduceScatterOverwriteInput(input, ranks_group, in_data_num, output_size, op_type, output);
}

bool MPIAllGather(const float *input, float *output, const std::vector<int> &ranks_group, size_t data_num) {
  auto inst = mindspore::device::cpu::MPIAdapter::Instance();
  if (inst == nullptr) {
    return false;
  }
  return inst->AllGather(input, output, ranks_group, data_num);
}
}
