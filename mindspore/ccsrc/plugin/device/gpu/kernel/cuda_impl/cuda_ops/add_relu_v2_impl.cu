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

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/add_relu_v2_impl.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/util.cuh"

template <typename T>
__global__ void AddReluV2Kernel(const size_t num, const T *x1, const T *x2, T *y, uint32_t *mask) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < num; i += gridDim.x * blockDim.x) {
    T sum = x1[i] + x2[i];
    bool p = sum > static_cast<T>(0);
    y[i] = p ? sum : static_cast<T>(0);

    auto warp_predict = BallotSync(p, __activemask());
    if (LaneId() == 0) {
      mask[WarpId(i)] = warp_predict;
    }
  }
}

template <typename T>
cudaError_t AddReluV2(const size_t num, const T *x1, const T *x2, T *y, uint32_t *mask, cudaStream_t cuda_stream) {
  AddReluV2Kernel<<<kBlocksPerGrid(num), kThreadsPerBlock, 0, cuda_stream>>>(num, x1, x2, y, mask);
  CHECK_CUDA_LAUNCH_SUCCESS();
}

template <typename T>
__global__ void AddReluGradV2Kernel(const size_t num, const T *x1, const T *x2, const uint32_t *mask, T *dx) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < num; i += gridDim.x * blockDim.x) {
    bool positive = mask[WarpId(i)] & (1 << LaneId());
    dx[i] = positive ? x1[i] + x2[i] : static_cast<T>(0);
  }
}

template <typename T>
cudaError_t AddReluGradV2(const size_t num, const T *x1, const T *x2, const uint32_t *mask, T *dx,
                          cudaStream_t cuda_stream) {
  AddReluGradV2Kernel<<<kBlocksPerGrid(num), kThreadsPerBlock, 0, cuda_stream>>>(num, x1, x2, mask, dx);
  CHECK_CUDA_LAUNCH_SUCCESS();
}

template CUDA_LIB_EXPORT cudaError_t AddReluV2(const size_t num, const float *x1, const float *x2, float *y,
                                               uint32_t *mask, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t AddReluV2(const size_t num, const double *x1, const double *x2, double *y,
                                               uint32_t *mask, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t AddReluV2(const size_t num, const half *x1, const half *x2, half *y,
                                               uint32_t *mask, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t AddReluV2(const size_t num, const int32_t *x1, const int32_t *x2, int32_t *y,
                                               uint32_t *mask, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t AddReluV2(const size_t num, const int64_t *x1, const int64_t *x2, int64_t *y,
                                               uint32_t *mask, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t AddReluGradV2(const size_t num, const float *x1, const float *x2,
                                                   const uint32_t *mask, float *dx, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t AddReluGradV2(const size_t num, const double *x1, const double *x2,
                                                   const uint32_t *mask, double *dx, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t AddReluGradV2(const size_t num, const half *x1, const half *x2,
                                                   const uint32_t *mask, half *dx, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t AddReluGradV2(const size_t num, const int32_t *x1, const int32_t *x2,
                                                   const uint32_t *mask, int32_t *dx, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t AddReluGradV2(const size_t num, const int64_t *x1, const int64_t *x2,
                                                   const uint32_t *mask, int64_t *dx, cudaStream_t cuda_stream);
