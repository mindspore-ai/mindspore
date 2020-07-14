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

#ifndef MINDSPORE_CCSRC_DEVICE_GPU_GPU_COMMON_H_
#define MINDSPORE_CCSRC_DEVICE_GPU_GPU_COMMON_H_

#include <iostream>
#include <vector>
#include <algorithm>
#include "utils/log_adapter.h"

namespace mindspore {
namespace device {
namespace gpu {
#define CHECK_OP_RET_WITH_EXCEPT(expression, message)                                 \
  {                                                                                   \
    bool success = (expression);                                                      \
    if (!success) {                                                                   \
      MS_LOG(EXCEPTION) << "Op Error: " << message << " | Error Number: " << success; \
    }                                                                                 \
  }

#define CHECK_OP_RET_WITH_ERROR(expression, message)                              \
  {                                                                               \
    bool success = (expression);                                                  \
    if (!success) {                                                               \
      MS_LOG(ERROR) << "Op Error: " << message << " | Error Number: " << success; \
    }                                                                             \
  }

#define CHECK_CUDA_RET_WITH_ERROR(expression, message)                                   \
  {                                                                                      \
    cudaError_t status = (expression);                                                   \
    if (status != cudaSuccess) {                                                         \
      MS_LOG(ERROR) << "CUDA Error: " << message << " | Error Number: " << status << " " \
                    << cudaGetErrorString(status);                                       \
    }                                                                                    \
  }

#define CHECK_CUDA_RET_WITH_EXCEPT(expression, message)                                      \
  {                                                                                          \
    cudaError_t status = (expression);                                                       \
    if (status != cudaSuccess) {                                                             \
      MS_LOG(EXCEPTION) << "CUDA Error: " << message << " | Error Number: " << status << " " \
                        << cudaGetErrorString(status);                                       \
    }                                                                                        \
  }

#define CHECK_CUDNN_RET_WITH_EXCEPT(expression, message)                                      \
  {                                                                                           \
    cudnnStatus_t status = (expression);                                                      \
    if (status != CUDNN_STATUS_SUCCESS) {                                                     \
      MS_LOG(EXCEPTION) << "cuDNN Error: " << message << " | Error Number: " << status << " " \
                        << cudnnGetErrorString(status);                                       \
    }                                                                                         \
  }

#define CHECK_CUDNN_RET_WITH_ERROR(expression, message)                                   \
  {                                                                                       \
    cudnnStatus_t status = (expression);                                                  \
    if (status != CUDNN_STATUS_SUCCESS) {                                                 \
      MS_LOG(ERROR) << "cuDNN Error: " << message << " | Error Number: " << status << " " \
                    << cudnnGetErrorString(status);                                       \
    }                                                                                     \
  }

#define CHECK_CUBLAS_RET_WITH_EXCEPT(expression, message)                                \
  {                                                                                      \
    cublasStatus_t status = (expression);                                                \
    if (status != CUBLAS_STATUS_SUCCESS) {                                               \
      MS_LOG(EXCEPTION) << "cuBLAS Error: " << message << " | Error Number: " << status; \
    }                                                                                    \
  }

#define CHECK_CUBLAS_RET_WITH_ERROR(expression, message)                             \
  {                                                                                  \
    cublasStatus_t status = (expression);                                            \
    if (status != CUBLAS_STATUS_SUCCESS) {                                           \
      MS_LOG(ERROR) << "cuBLAS Error: " << message << " | Error Number: " << status; \
    }                                                                                \
  }

#define CHECK_NCCL_RET_WITH_EXCEPT(expression, message)                                \
  {                                                                                    \
    int result = (expression);                                                         \
    if (result != ncclSuccess) {                                                       \
      MS_LOG(EXCEPTION) << "NCCL Error: " << message << " | Error Number: " << result; \
    }                                                                                  \
  }

#define VARIABLE_NOT_USED(var) \
  { (void)(var); }

inline bool CheckNullInput(std::vector<size_t> input_shape) {
  // If input_shape.size() == 0, it means a scalar input; If input_shape.size() != 0 and input_shape contains 0,
  // it means a null input. Just return a null output.
  if (input_shape.size() != 0) {
    if (std::any_of(input_shape.begin(), input_shape.end(), [](size_t i) { return i == 0; })) {
      return true;
    }
  }
  return false;
}
#define CHECK_NULL_INPUT(input_shape) mindspore::device::gpu::CheckNullInput(input_shape)
}  // namespace gpu
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_DEVICE_GPU_GPU_COMMON_H_
