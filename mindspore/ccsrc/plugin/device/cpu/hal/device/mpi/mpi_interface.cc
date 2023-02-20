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
#include "plugin/device/cpu/hal/device/mpi/mpi_interface.h"
#ifdef ENABLE_MPI
#include <dlfcn.h>
#include <vector>
#include <string>
#include "utils/log_adapter.h"

inline void *LoadLibrary(const char *name) {
  auto handle = dlopen(name, RTLD_LAZY | RTLD_LOCAL);
  if (handle == nullptr) {
    MS_LOG(EXCEPTION) << "Load lib " << name << " failed, make sure you have installed it!";
  }
  return handle;
}

inline void *GetMPIAdapterHandle() {
  void *handle = LoadLibrary("libmpi_adapter.so");
  return handle;
}

template <class T>
static T GetMPIAdapterFunc(const std::string &name) {
  void *handle = GetMPIAdapterHandle();
  if (handle == nullptr) {
    MS_LOG(EXCEPTION) << "Load lib " << name << " failed, make sure you have installed it!";
  }
  auto func = reinterpret_cast<T>(dlsym(handle, name.c_str()));
  if (func == nullptr) {
    MS_LOG(EXCEPTION) << "Load func " << name << " failed, make sure you have implied it!";
  }
  return func;
}

typedef int (*GetMPIRankIdFunc)();
typedef int (*GetMPIRankSizeFunc)();
typedef bool (*MPIReduceScatterFunc)(const float *input, float *output, const std::vector<int> &ranks_group,
                                     size_t data_num, const std::string &op_type);
typedef bool (*MPIReduceScatterOverwriteInputFunc)(float *input, const std::vector<int> &ranks_group,
                                                   size_t in_data_num, size_t output_size, const std::string &op_type,
                                                   float *output);
typedef bool (*MPIAllGatherFunc)(const float *input, float *output, const std::vector<int> &ranks_group,
                                 size_t data_num);

int GetMPIRankId() {
  auto func = GetMPIAdapterFunc<GetMPIRankIdFunc>("GetMPIRankId");
  return func();
}

int GetMPIRankSize() {
  auto func = GetMPIAdapterFunc<GetMPIRankSizeFunc>("GetMPIRankSize");
  return func();
}

bool MPIReduceScatter(const float *input, float *output, const std::vector<int> &ranks_group, size_t data_num,
                      const std::string &op_type) {
  auto func = GetMPIAdapterFunc<MPIReduceScatterFunc>("MPIReduceScatter");
  return func(input, output, ranks_group, data_num, op_type);
}

bool MPIReduceScatterOverwriteInput(float *input, const std::vector<int> &ranks_group, size_t in_data_num,
                                    size_t output_size, const std::string &op_type, float *output) {
  auto func = GetMPIAdapterFunc<MPIReduceScatterOverwriteInputFunc>("MPIReduceScatterOverwriteInput");
  return func(input, ranks_group, in_data_num, output_size, op_type, output);
}

bool MPIAllGather(const float *input, float *output, const std::vector<int> &ranks_group, size_t data_num) {
  auto func = GetMPIAdapterFunc<MPIAllGatherFunc>("MPIAllGather");
  return func(input, output, ranks_group, data_num);
}
#endif  // ENABLE_MPI
