/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_RUNTIME_DEVICE_GPU_QUEUE_COMMON_H_
#define MINDSPORE_CCSRC_RUNTIME_DEVICE_GPU_QUEUE_COMMON_H_

#include <iostream>
#include <vector>
#include <algorithm>
#include <map>
#include "utils/log_adapter.h"
#include "include/curand.h"

namespace mindspore {
namespace device {
namespace gpu {
#define CHECK_CUDA_RET_WITH_ERROR(expression, message)                                   \
  do {                                                                                   \
    cudaError_t status = (expression);                                                   \
    if (status != cudaSuccess) {                                                         \
      MS_LOG(ERROR) << "CUDA Error: " << message << " | Error Number: " << status << " " \
                    << cudaGetErrorString(status);                                       \
    }                                                                                    \
  } while (0);
}  // namespace gpu
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_DEVICE_GPU_QUEUE_COMMON_H_
