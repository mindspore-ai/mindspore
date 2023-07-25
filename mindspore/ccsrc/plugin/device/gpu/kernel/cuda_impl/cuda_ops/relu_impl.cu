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

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/relu_impl.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/util.cuh"

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
cudaError_t ReluV2(const size_t num, const T *x, T *y, uint32_t *mask, cudaStream_t cuda_stream) {
  ReluV2Kernel<<<kBlocksPerGrid(num), kThreadsPerBlock, 0, cuda_stream>>>(num, x, y, mask);
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t ReluV2(const size_t num, const double *x, double *y, uint32_t *mask,
                                            cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t ReluV2(const size_t num, const float *x, float *y, uint32_t *mask,
                                            cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t ReluV2(const size_t num, const half *x, half *y, uint32_t *mask,
                                            cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t ReluV2(const size_t num, const int8_t *x, int8_t *y, uint32_t *mask,
                                            cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t ReluV2(const size_t num, const int16_t *x, int16_t *y, uint32_t *mask,
                                            cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t ReluV2(const size_t num, const int32_t *x, int32_t *y, uint32_t *mask,
                                            cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t ReluV2(const size_t num, const int64_t *x, int64_t *y, uint32_t *mask,
                                            cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t ReluV2(const size_t num, const uint8_t *x, uint8_t *y, uint32_t *mask,
                                            cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t ReluV2(const size_t num, const uint16_t *x, uint16_t *y, uint32_t *mask,
                                            cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t ReluV2(const size_t num, const uint32_t *x, uint32_t *y, uint32_t *mask,
                                            cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t ReluV2(const size_t num, const uint64_t *x, uint64_t *y, uint32_t *mask,
                                            cudaStream_t cuda_stream);
