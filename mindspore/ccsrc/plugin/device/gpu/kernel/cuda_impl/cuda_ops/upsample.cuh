/**
 * Copyright 2022 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITposh WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 **/
#include <math.h>
#include <algorithm>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/util.cuh"

template <typename ac_T>
__device__ __forceinline__ static ac_T area_pixel_compute_source_index(ac_T scale, int dst_idx, bool align_corners,
                                                                       bool cubic) {
  if (align_corners) {
    return scale * dst_idx;
  } else {
    ac_T src_idx = scale * (dst_idx + static_cast<ac_T>(0.5)) - static_cast<ac_T>(0.5);
    return (!cubic && src_idx < static_cast<ac_T>(0)) ? static_cast<ac_T>(0) : src_idx;
  }
}

__device__ __forceinline__ static int nearest_neighbor_exact_compute_source_index(const float scale, int dst_idx,
                                                                                  int in_size) {
  const int src_idx = min(static_cast<int>(floorf((dst_idx + static_cast<float>(0.5)) * scale)), in_size - 1);
  return src_idx;
}

__device__ __forceinline__ static int nearest_neighbor_compute_source_index(const float scale, int dst_idx,
                                                                            int in_size) {
  const int src_idx = min(static_cast<int>(floorf((dst_idx)*scale)), in_size - 1);
  return src_idx;
}

template <typename T, typename S>
__global__ void CudaMemcpyDeviceToDevice(const int num_kernels, const T *input, S *output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < num_kernels; pos += blockDim.x * gridDim.x) {
    output[pos] = static_cast<S>(input[pos]);
  }
}
