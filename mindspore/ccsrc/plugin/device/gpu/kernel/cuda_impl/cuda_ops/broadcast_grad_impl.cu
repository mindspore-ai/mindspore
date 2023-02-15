/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/broadcast_grad_impl.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cuda_device_info.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/util.cuh"

template <typename T>
struct MinimumGradFunc {
  __device__ __forceinline__ void operator()(const T &x1, const T &x2, const bool &grad_x1, const bool &grad_x2,
                                             const T &dy, T *dx1, T *dx2) {
    if (grad_x1 && x1 < x2) {
      MsAtomicAdd(dx1, dy);
    } else if (grad_x2 && x1 > x2) {
      MsAtomicAdd(dx2, dy);
    } else if (grad_x1 && grad_x2 && x1 == x2) {
      T ddy = dy * (T) 0.5;
      MsAtomicAdd(dx1, ddy);
      MsAtomicAdd(dx2, ddy);
    }
  }
};

template <typename T>
struct MaximumGradFunc {
  __device__ __forceinline__ void operator()(const T &x1, const T &x2, const bool &grad_x1, const bool &grad_x2,
                                             const T &dy, T *dx1, T *dx2) {
    if (grad_x1 && x1 > x2) {
      MsAtomicAdd(dx1, dy);
    } else if (grad_x2 && x1 < x2) {
      MsAtomicAdd(dx2, dy);
    } else if (grad_x1 && grad_x2 && x1 == x2) {
      T ddy = dy * (T) 0.5;
      MsAtomicAdd(dx1, ddy);
      MsAtomicAdd(dx2, ddy);
    }
  }
};

__device__ __forceinline__ size_t Index(const size_t &index, const size_t &dim) { return dim == 1 ? 0 : index; }

template <typename T, typename Func>
__global__ void BroadcastGradOperator(const size_t l0, const size_t l1, const size_t l2, const size_t l3,
                                      const size_t l4, const size_t l5, const size_t l6, const size_t r0,
                                      const size_t r1, const size_t r2, const size_t r3, const size_t r4,
                                      const size_t r5, const size_t r6, const size_t d0, const size_t d1,
                                      const size_t d2, const size_t d3, const size_t d4, const size_t d5,
                                      const size_t d6, const size_t nums, const bool grad_x1, const bool grad_x2,
                                      const T *x1, const T *x2, const T *dy, T *dx1, T *dx2) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < nums; pos += blockDim.x * gridDim.x) {
    size_t i = pos / (d1 * d2 * d3 * d4 * d5 * d6) % d0;
    size_t j = pos / (d2 * d3 * d4 * d5 * d6) % d1;
    size_t k = pos / (d3 * d4 * d5 * d6) % d2;
    size_t l = pos / (d4 * d5 * d6) % d3;
    size_t m = pos / (d5 * d6) % d4;
    size_t n = pos / d6 % d5;
    size_t o = pos % d6;

    size_t l_index = Index(i, l0) * l1 * l2 * l3 * l4 * l5 * l6;
    l_index += Index(j, l1) * l2 * l3 * l4 * l5 * l6;
    l_index += Index(k, l2) * l3 * l4 * l5 * l6;
    l_index += Index(l, l3) * l4 * l5 * l6;
    l_index += Index(m, l4) * l5 * l6;
    l_index += Index(n, l5) * l6;
    l_index += Index(o, l6);

    size_t r_index = Index(i, r0) * r1 * r2 * r3 * r4 * r5 * r6;
    r_index += Index(j, r1) * r2 * r3 * r4 * r5 * r6;
    r_index += Index(k, r2) * r3 * r4 * r5 * r6;
    r_index += Index(l, r3) * r4 * r5 * r6;
    r_index += Index(m, r4) * r5 * r6;
    r_index += Index(n, r5) * r6;
    r_index += Index(o, r6);
    Func()(x1[l_index], x2[r_index], grad_x1, grad_x2, dy[pos], dx1 + l_index, dx2 + r_index);
  }
}

template <typename T>
void BroadcastGrad(const std::vector<size_t> &x1_shape, const std::vector<size_t> &x2_shape,
                   const std::vector<size_t> &dy_shape, const size_t &nums, const bool &grad_x1, const bool &grad_x2,
                   BroadcastGradOpType op, const T *x1, const T *x2, const T *dy, T *dx1, T *dx2,
                   const uint32_t &device_id, cudaStream_t stream) {
  switch (op) {
    case BROADCAST_GRAD_TYPE_MINIMUM:
      return BroadcastGradOperator<T, MinimumGradFunc<T>>
        <<<CUDA_BLOCKS(device_id, nums), CUDA_THREADS(device_id), 0, stream>>>(
          x1_shape[0], x1_shape[1], x1_shape[2], x1_shape[3], x1_shape[4], x1_shape[5], x1_shape[6], x2_shape[0],
          x2_shape[1], x2_shape[2], x2_shape[3], x2_shape[4], x2_shape[5], x2_shape[6], dy_shape[0], dy_shape[1],
          dy_shape[2], dy_shape[3], dy_shape[4], dy_shape[5], dy_shape[6], nums, grad_x1, grad_x2, x1, x2, dy, dx1,
          dx2);
    case BROADCAST_GRAD_TYPE_MAXIMUM:
      return BroadcastGradOperator<T, MaximumGradFunc<T>>
        <<<CUDA_BLOCKS(device_id, nums), CUDA_THREADS(device_id), 0, stream>>>(
          x1_shape[0], x1_shape[1], x1_shape[2], x1_shape[3], x1_shape[4], x1_shape[5], x1_shape[6], x2_shape[0],
          x2_shape[1], x2_shape[2], x2_shape[3], x2_shape[4], x2_shape[5], x2_shape[6], dy_shape[0], dy_shape[1],
          dy_shape[2], dy_shape[3], dy_shape[4], dy_shape[5], dy_shape[6], nums, grad_x1, grad_x2, x1, x2, dy, dx1,
          dx2);
    default:
      break;
  }
}

template <typename T, typename Func>
__global__ void NoBroadcastOperator(const size_t nums, const bool grad_x1, const bool grad_x2, const T *x1, const T *x2,
                                    const T *dy, T *dx1, T *dx2) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < nums; pos += blockDim.x * gridDim.x) {
    Func()(x1[pos], x2[pos], grad_x1, grad_x2, dy[pos], dx1 + pos, dx2 + pos);
  }
}

template <typename T>
void NoBroadcastGrad(const size_t &nums, const bool &grad_x1, const bool &grad_x2, BroadcastGradOpType op, const T *x1,
                     const T *x2, const T *dy, T *dx1, T *dx2, const uint32_t &device_id, cudaStream_t stream) {
  switch (op) {
    case BROADCAST_GRAD_TYPE_MINIMUM:
      return NoBroadcastOperator<T, MinimumGradFunc<T>>
        <<<CUDA_BLOCKS(device_id, nums), CUDA_THREADS(device_id), 0, stream>>>(nums, grad_x1, grad_x2, x1, x2, dy, dx1,
                                                                               dx2);
    case BROADCAST_GRAD_TYPE_MAXIMUM:
      return NoBroadcastOperator<T, MaximumGradFunc<T>>
        <<<CUDA_BLOCKS(device_id, nums), CUDA_THREADS(device_id), 0, stream>>>(nums, grad_x1, grad_x2, x1, x2, dy, dx1,
                                                                               dx2);
    default:
      break;
  }
}

template CUDA_LIB_EXPORT void NoBroadcastGrad(const size_t &nums, const bool &grad_x1, const bool &grad_x2,
                                              BroadcastGradOpType op, const half *x1, const half *x2, const half *dy,
                                              half *dx1, half *dx2, const uint32_t &device_id, cudaStream_t stream);
template CUDA_LIB_EXPORT void NoBroadcastGrad(const size_t &nums, const bool &grad_x1, const bool &grad_x2,
                                              BroadcastGradOpType op, const float *x1, const float *x2, const float *dy,
                                              float *dx1, float *dx2, const uint32_t &device_id, cudaStream_t stream);
template CUDA_LIB_EXPORT void NoBroadcastGrad(const size_t &nums, const bool &grad_x1, const bool &grad_x2,
                                              BroadcastGradOpType op, const double *x1, const double *x2,
                                              const double *dy, double *dx1, double *dx2, const uint32_t &device_id,
                                              cudaStream_t stream);
template CUDA_LIB_EXPORT void NoBroadcastGrad(const size_t &nums, const bool &grad_x1, const bool &grad_x2,
                                              BroadcastGradOpType op, const int *x1, const int *x2, const int *dy,
                                              int *dx1, int *dx2, const uint32_t &device_id, cudaStream_t stream);
template CUDA_LIB_EXPORT void NoBroadcastGrad(const size_t &nums, const bool &grad_x1, const bool &grad_x2,
                                              BroadcastGradOpType op, const int64_t *x1, const int64_t *x2,
                                              const int64_t *dy, int64_t *dx1, int64_t *dx2, const uint32_t &device_id,
                                              cudaStream_t stream);
template CUDA_LIB_EXPORT void NoBroadcastGrad(const size_t &nums, const bool &grad_x1, const bool &grad_x2,
                                              BroadcastGradOpType op, const int16_t *x1, const int16_t *x2,
                                              const int16_t *dy, int16_t *dx1, int16_t *dx2, const uint32_t &device_id,
                                              cudaStream_t stream);
template CUDA_LIB_EXPORT void NoBroadcastGrad(const size_t &nums, const bool &grad_x1, const bool &grad_x2,
                                              BroadcastGradOpType op, const uint16_t *x1, const uint16_t *x2,
                                              const uint16_t *dy, uint16_t *dx1, uint16_t *dx2,
                                              const uint32_t &device_id, cudaStream_t stream);
template CUDA_LIB_EXPORT void NoBroadcastGrad(const size_t &nums, const bool &grad_x1, const bool &grad_x2,
                                              BroadcastGradOpType op, const uint32_t *x1, const uint32_t *x2,
                                              const uint32_t *dy, uint32_t *dx1, uint32_t *dx2,
                                              const uint32_t &device_id, cudaStream_t stream);
template CUDA_LIB_EXPORT void NoBroadcastGrad(const size_t &nums, const bool &grad_x1, const bool &grad_x2,
                                              BroadcastGradOpType op, const uint64_t *x1, const uint64_t *x2,
                                              const uint64_t *dy, uint64_t *dx1, uint64_t *dx2,
                                              const uint32_t &device_id, cudaStream_t stream);

template CUDA_LIB_EXPORT void BroadcastGrad(const std::vector<size_t> &x1_shape, const std::vector<size_t> &x2_shape,
                                            const std::vector<size_t> &dy_shape, const size_t &nums,
                                            const bool &grad_x1, const bool &grad_x2, BroadcastGradOpType op,
                                            const half *x1, const half *x2, const half *dy, half *dx1, half *dx2,
                                            const uint32_t &device_id, cudaStream_t stream);
template CUDA_LIB_EXPORT void BroadcastGrad(const std::vector<size_t> &x1_shape, const std::vector<size_t> &x2_shape,
                                            const std::vector<size_t> &dy_shape, const size_t &nums,
                                            const bool &grad_x1, const bool &grad_x2, BroadcastGradOpType op,
                                            const float *x1, const float *x2, const float *dy, float *dx1, float *dx2,
                                            const uint32_t &device_id, cudaStream_t stream);
template CUDA_LIB_EXPORT void BroadcastGrad(const std::vector<size_t> &x1_shape, const std::vector<size_t> &x2_shape,
                                            const std::vector<size_t> &dy_shape, const size_t &nums,
                                            const bool &grad_x1, const bool &grad_x2, BroadcastGradOpType op,
                                            const double *x1, const double *x2, const double *dy, double *dx1,
                                            double *dx2, const uint32_t &device_id, cudaStream_t stream);
template CUDA_LIB_EXPORT void BroadcastGrad(const std::vector<size_t> &x1_shape, const std::vector<size_t> &x2_shape,
                                            const std::vector<size_t> &dy_shape, const size_t &nums,
                                            const bool &grad_x1, const bool &grad_x2, BroadcastGradOpType op,
                                            const int *x1, const int *x2, const int *dy, int *dx1, int *dx2,
                                            const uint32_t &device_id, cudaStream_t stream);
template CUDA_LIB_EXPORT void BroadcastGrad(const std::vector<size_t> &x1_shape, const std::vector<size_t> &x2_shape,
                                            const std::vector<size_t> &dy_shape, const size_t &nums,
                                            const bool &grad_x1, const bool &grad_x2, BroadcastGradOpType op,
                                            const int64_t *x1, const int64_t *x2, const int64_t *dy, int64_t *dx1,
                                            int64_t *dx2, const uint32_t &device_id, cudaStream_t stream);
template CUDA_LIB_EXPORT void BroadcastGrad(const std::vector<size_t> &x1_shape, const std::vector<size_t> &x2_shape,
                                            const std::vector<size_t> &dy_shape, const size_t &nums,
                                            const bool &grad_x1, const bool &grad_x2, BroadcastGradOpType op,
                                            const int16_t *x1, const int16_t *x2, const int16_t *dy, int16_t *dx1,
                                            int16_t *dx2, const uint32_t &device_id, cudaStream_t stream);
template CUDA_LIB_EXPORT void BroadcastGrad(const std::vector<size_t> &x1_shape, const std::vector<size_t> &x2_shape,
                                            const std::vector<size_t> &dy_shape, const size_t &nums,
                                            const bool &grad_x1, const bool &grad_x2, BroadcastGradOpType op,
                                            const uint16_t *x1, const uint16_t *x2, const uint16_t *dy, uint16_t *dx1,
                                            uint16_t *dx2, const uint32_t &device_id, cudaStream_t stream);
template CUDA_LIB_EXPORT void BroadcastGrad(const std::vector<size_t> &x1_shape, const std::vector<size_t> &x2_shape,
                                            const std::vector<size_t> &dy_shape, const size_t &nums,
                                            const bool &grad_x1, const bool &grad_x2, BroadcastGradOpType op,
                                            const uint32_t *x1, const uint32_t *x2, const uint32_t *dy, uint32_t *dx1,
                                            uint32_t *dx2, const uint32_t &device_id, cudaStream_t stream);
template CUDA_LIB_EXPORT void BroadcastGrad(const std::vector<size_t> &x1_shape, const std::vector<size_t> &x2_shape,
                                            const std::vector<size_t> &dy_shape, const size_t &nums,
                                            const bool &grad_x1, const bool &grad_x2, BroadcastGradOpType op,
                                            const uint64_t *x1, const uint64_t *x2, const uint64_t *dy, uint64_t *dx1,
                                            uint64_t *dx2, const uint32_t &device_id, cudaStream_t stream);
