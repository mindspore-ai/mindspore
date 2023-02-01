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

#ifndef MINDSPORE_CCSRC_RUNTIME_DEVICE_GPU_COLLECTIVE_COMMON_H_
#define MINDSPORE_CCSRC_RUNTIME_DEVICE_GPU_COLLECTIVE_COMMON_H_

#include <nccl.h>
#include <vector>
#include <sstream>
#ifndef LITE_CUDA_DISTRIBUTION
#include "pybind11/pybind11.h"
#endif

namespace mindspore {
namespace device {
namespace gpu {
constexpr int MAX_HOSTNAME_LEN = 1024;
constexpr char NCCL_WORLD_GROUP[] = "nccl_world_group";
struct NcclGroupInfo {
  int size;
  int rank;
  ncclUniqueId unique_id;
  ncclComm_t comm;
  std::vector<int> group_ranks;
};
#ifndef LITE_CUDA_DISTRIBUTION
#define CHECK_RET(expression, result, message)                                                                         \
  do {                                                                                                                 \
    auto ret = (expression);                                                                                           \
    if (ret != result) {                                                                                               \
      std::ostringstream oss;                                                                                          \
      oss << "Error in file " << __FILE__ << " | Error on line " << __LINE__ << " | GPU collective Error: " << message \
          << " | Error Number " << ret;                                                                                \
      pybind11::pybind11_fail(oss.str());                                                                              \
    }                                                                                                                  \
  } while (0)
#else
#define CHECK_RET(expression, result, message)                                                                         \
  do {                                                                                                                 \
    auto ret = (expression);                                                                                           \
    if (ret != result) {                                                                                               \
      std::ostringstream oss;                                                                                          \
      oss << "Error in file " << __FILE__ << " | Error on line " << __LINE__ << " | GPU collective Error: " << message \
          << " | Error Number " << ret;                                                                                \
      printf("%s", oss.str().c_str());                                                                                 \
      exit(EXIT_FAILURE);                                                                                              \
    }                                                                                                                  \
  } while (0)
#endif
}  // namespace gpu
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_DEVICE_GPU_COLLECTIVE_COMMON_H_
