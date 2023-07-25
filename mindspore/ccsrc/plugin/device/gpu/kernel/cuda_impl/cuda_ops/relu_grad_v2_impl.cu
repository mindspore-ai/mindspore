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
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/relu_grad_v2_impl.cuh"
#include "plugin/device/cpu/kernel/nnacl/op_base.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/elementwise/elementswise_pub_impl.cuh"
constexpr uint kThreadsPerBlock = cuda::elementwise::kThreadsPerBlock;

enum : unsigned { warp_size = 32, log_wap_size = 5 };
template <uint vec_size>
__device__ __forceinline__ unsigned LaneId() {
  // cause run in a vectorized mode, lane num shrinks to 'warp_size / vec_size' in a wrap
  // then land_id should be calculated in this way:
  // get first land_id in a VectorizedCall, then += 1 in vectorized loop
  unsigned lane_id = threadIdx.x & (warp_size - 1);
  auto lane_num_in_a_warp = warp_size / vec_size;
  return (lane_id % lane_num_in_a_warp) * vec_size;
}
__device__ __forceinline__ unsigned WarpId(const unsigned &tid) { return tid >> log_wap_size; }

template <uint vec_size, typename T>
__device__ __forceinline__ void VectorizedCall(const T *dy, const uint32_t *mask, T *dx, uint offset) {
  uint tid = threadIdx.x;
  auto index = tid * vec_size + offset;

  using VecT = cuda::elementwise::AlignVec<T, vec_size>;
  using VecUint32 = cuda::elementwise::AlignVec<uint32_t, vec_size>;

  auto vec_dy = reinterpret_cast<const VecT *>(dy + offset);
  uint32_t single_mask = mask[WarpId(index)];
  auto land_id = LaneId<vec_size>();
  auto vec_dx = reinterpret_cast<VecT *>(dx + offset);
  VecT cache_dy = vec_dy[tid];
  VecT out_dx{0};

#pragma unroll
  for (uint j = 0; j < vec_size; j++) {
    bool p = single_mask & (1 << land_id);
    land_id += 1;
    out_dx.elements_[j] = p ? cache_dy.elements_[j] : static_cast<T>(0);
  }
  vec_dx[tid] = out_dx;
}

template <uint vec_size, typename T>
__device__ __forceinline__ void NormalCall(const T *dy, const uint32_t *mask, T *dx, uint offset, uint remaining) {
  uint loop = UP_DIV(remaining, vec_size);
  for (uint i = threadIdx.x; i < loop; i += blockDim.x) {
    auto lane_id = LaneId<vec_size>();
#pragma unroll
    for (uint j = 0; j < vec_size; j++) {
      uint index = i * vec_size + j;
      if (index >= remaining) {
        return;
      }
      index += offset;
      bool p = mask[WarpId(index)] & (1 << lane_id);
      lane_id += 1;
      dx[index] = p ? dy[index] : static_cast<T>(0);
    }
  }
}

template <uint vec_size, typename T>
__global__ void ReluGradV2Vectorized(const T *dy, const uint32_t *mask, T *dx, uint num_of_elements) {
  uint elements_per_block = kThreadsPerBlock * vec_size;
  for (uint offset = elements_per_block * blockIdx.x; offset < num_of_elements;
       offset += elements_per_block * gridDim.x) {
    uint remaining = num_of_elements - offset;
    if (remaining < elements_per_block) {
      NormalCall<vec_size, T>(dy, mask, dx, offset, remaining);
    } else {
      VectorizedCall<vec_size, T>(dy, mask, dx, offset);
    }
  }
}

template <typename T>
cudaError_t ReluGradV2(const size_t num, const T *dy, const uint32_t *mask, T *dx, cudaStream_t cuda_stream) {
  constexpr uint vec_size = cuda::elementwise::VecSize<T, uint32_t>();
  const auto block_x = uint(kThreadsPerBlock);
  const uint elements_per_block = kThreadsPerBlock * vec_size;
  const auto grid_x = uint(UP_DIV(num, elements_per_block));
  dim3 block{block_x};
  dim3 grid{grid_x};
  ReluGradV2Vectorized<vec_size, T><<<grid, block, 0, cuda_stream>>>(dy, mask, dx, num);
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t ReluGradV2(const size_t num, const double *dy, const uint32_t *mask, double *dx,
                                                cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t ReluGradV2(const size_t num, const float *dy, const uint32_t *mask, float *dx,
                                                cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t ReluGradV2(const size_t num, const half *dy, const uint32_t *mask, half *dx,
                                                cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t ReluGradV2(const size_t num, const int8_t *dy, const uint32_t *mask, int8_t *dx,
                                                cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t ReluGradV2(const size_t num, const int16_t *dy, const uint32_t *mask, int16_t *dx,
                                                cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t ReluGradV2(const size_t num, const int32_t *dy, const uint32_t *mask, int32_t *dx,
                                                cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t ReluGradV2(const size_t num, const int64_t *dy, const uint32_t *mask, int64_t *dx,
                                                cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t ReluGradV2(const size_t num, const uint8_t *dy, const uint32_t *mask, uint8_t *dx,
                                                cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t ReluGradV2(const size_t num, const uint16_t *dy, const uint32_t *mask,
                                                uint16_t *dx, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t ReluGradV2(const size_t num, const uint32_t *dy, const uint32_t *mask,
                                                uint32_t *dx, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t ReluGradV2(const size_t num, const uint64_t *dy, const uint32_t *mask,
                                                uint64_t *dx, cudaStream_t cuda_stream);
