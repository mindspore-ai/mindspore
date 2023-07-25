/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_DYNAMIC_RANGE_IMPL_CUH_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_DYNAMIC_RANGE_IMPL_CUH_
#include <cuda_runtime.h>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cuda_common.h"

enum class DynamicRangeErrorCode {
  kOk = 0,
  kDeltaIsZero,
  kInvalidPositiveDelta,
  kInvalidNegativeDelta,
  kMaxSizeExceeded
};

template <typename T>
CUDA_LIB_EXPORT cudaError_t CudaValidateInputAndInferShape(const T *range_start, const T *range_end,
                                                           const T *range_delta, int64_t *output_shape,
                                                           DynamicRangeErrorCode *error_code,
                                                           const int64_t max_output_size, cudaStream_t cuda_stream);

template <typename T>
CUDA_LIB_EXPORT cudaError_t CalRange(const T *range_start, const T *range_end, const T *range_delta, T *output,
                                     int64_t *output_shape, DynamicRangeErrorCode *error_code,
                                     const int64_t max_output_size, cudaStream_t cuda_stream);

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_DYNAMIC_RANGE_IMPL_CUH_
