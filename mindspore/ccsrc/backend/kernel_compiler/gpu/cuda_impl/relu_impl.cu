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

#include "backend/kernel_compiler/gpu/cuda_impl/relu_impl.cuh"
#include "backend/kernel_compiler/gpu/cuda_impl/util.cuh"
#include "runtime/device/gpu/cuda_common.h"

template <typename T>
__global__ void CalReLUKernel(int size, T *input_addr, T *output_addr) {
  for (int pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    output_addr[pos] = input_addr[pos] > static_cast<T>(0) ? input_addr[pos] : static_cast<T>(0);
  }
}

template <typename T>
void CalReLU(int size, T *input_addr, T *output_addr, cudaStream_t cuda_stream) {
  CalReLUKernel<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(size, input_addr, output_addr);
  return;
}

template void CalReLU(int size, double *input_addr, double *output_addr, cudaStream_t cuda_stream);
template void CalReLU(int size, float *input_addr, float *output_addr, cudaStream_t cuda_stream);
template void CalReLU(int size, half *input_addr, half *output_addr, cudaStream_t cuda_stream);
template void CalReLU(int size, int8_t *input_addr, int8_t *output_addr, cudaStream_t cuda_stream);
template void CalReLU(int size, int16_t *input_addr, int16_t *output_addr, cudaStream_t cuda_stream);
template void CalReLU(int size, int32_t *input_addr, int32_t *output_addr, cudaStream_t cuda_stream);
template void CalReLU(int size, int64_t *input_addr, int64_t *output_addr, cudaStream_t cuda_stream);
template void CalReLU(int size, uint8_t *input_addr, uint8_t *output_addr, cudaStream_t cuda_stream);

template <typename T>
__global__ void ReluV2Kernel(const size_t num, const T *x, T *y, uint32_t *mask) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < num; i += blockDim.x * gridDim.x) {
    T v = x[i];
    bool p = v > static_cast<T>(0);
    y[i] = p ? v : static_cast<T>(0);

    auto warp_predict = BallotSync(p, __activemask());
    if (LaneId() == 0) {
      mask[WarpId(i)] = warp_predict;
    }
  }
}

template <typename T>
void ReluV2(const size_t num, const T *x, T *y, uint32_t *mask, cudaStream_t cuda_stream) {
  ReluV2Kernel<<<kBlocksPerGrid(num), kThreadsPerBlock, 0, cuda_stream>>>(num, x, y, mask);
}

template <typename T>
__global__ void ReluGradV2Kernel(const size_t num, const T *dy, const uint32_t *mask, T *dx) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < num; i += blockDim.x * gridDim.x) {
    bool p = mask[WarpId(i)] & (1 << LaneId());
    dx[i] = p ? dy[i] : static_cast<T>(0);
  }
}

template <typename T>
void ReluGradV2(const size_t num, const T *dy, const uint32_t *mask, T *dx, cudaStream_t cuda_stream) {
  ReluGradV2Kernel<<<kBlocksPerGrid(num), kThreadsPerBlock, 0, cuda_stream>>>(num, dy, mask, dx);
}

template void ReluV2(const size_t num, const double *x, double *y, uint32_t *mask, cudaStream_t cuda_stream);
template void ReluV2(const size_t num, const float *x, float *y, uint32_t *mask, cudaStream_t cuda_stream);
template void ReluV2(const size_t num, const half *x, half *y, uint32_t *mask, cudaStream_t cuda_stream);
template void ReluV2(const size_t num, const int8_t *x, int8_t *y, uint32_t *mask, cudaStream_t cuda_stream);
template void ReluV2(const size_t num, const int16_t *x, int16_t *y, uint32_t *mask, cudaStream_t cuda_stream);
template void ReluV2(const size_t num, const int32_t *x, int32_t *y, uint32_t *mask, cudaStream_t cuda_stream);
template void ReluV2(const size_t num, const int64_t *x, int64_t *y, uint32_t *mask, cudaStream_t cuda_stream);
template void ReluV2(const size_t num, const uint8_t *x, uint8_t *y, uint32_t *mask, cudaStream_t cuda_stream);

template void ReluGradV2(const size_t num, const double *dy, const uint32_t *mask, double *dx,
        cudaStream_t cuda_stream);
template void ReluGradV2(const size_t num, const float *dy, const uint32_t *mask, float *dx, cudaStream_t cuda_stream);
template void ReluGradV2(const size_t num, const half *dy, const uint32_t *mask, half *dx, cudaStream_t cuda_stream);
template void ReluGradV2(const size_t num, const int8_t *dy, const uint32_t *mask, int8_t *dx,
        cudaStream_t cuda_stream);
template void ReluGradV2(const size_t num, const int16_t *dy, const uint32_t *mask, int16_t *dx,
        cudaStream_t cuda_stream);
template void ReluGradV2(const size_t num, const int32_t *dy, const uint32_t *mask, int32_t *dx,
        cudaStream_t cuda_stream);
template void ReluGradV2(const size_t num, const int64_t *dy, const uint32_t *mask, int64_t *dx,
        cudaStream_t cuda_stream);
template void ReluGradV2(const size_t num, const uint8_t *dy, const uint32_t *mask, uint8_t *dx,
        cudaStream_t cuda_stream);
