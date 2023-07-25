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

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/broadcast_grad_grad_impl.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cuda_device_info.h"

template <typename T>
struct MinimumGradGradFunc {
  __device__ __forceinline__ T operator()(const T &x1, const T &x2, const T &dy1, const T &dy2) {
    if (x1 < x2) {
      return dy1;
    }
    return dy2;
  }
};

template <typename T>
struct MaximumGradGradFunc {
  __device__ __forceinline__ T operator()(const T &x1, const T &x2, const T &dy1, const T &dy2) {
    if (x1 > x2) {
      return dy1;
    }
    return dy2;
  }
};

__device__ __forceinline__ size_t Index(const size_t &index, const size_t &dim) { return dim == 1 ? 0 : index; }

template <typename T, typename Func>
__global__ void BroadcastGradGradOperator(const size_t l0, const size_t l1, const size_t l2, const size_t l3,
                                          const size_t l4, const size_t l5, const size_t l6, const size_t r0,
                                          const size_t r1, const size_t r2, const size_t r3, const size_t r4,
                                          const size_t r5, const size_t r6, const size_t d0, const size_t d1,
                                          const size_t d2, const size_t d3, const size_t d4, const size_t d5,
                                          const size_t d6, const size_t nums, const T *x1, const T *x2, const T *dy1,
                                          const T *dy2, T *sopd_grad) {
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
    sopd_grad[pos] = Func()(x1[l_index], x2[r_index], dy1[l_index], dy2[r_index]);
  }
}

template <typename T>
cudaError_t BroadcastGradGrad(const std::vector<size_t> &x1_shape, const std::vector<size_t> &x2_shape,
                              const std::vector<size_t> &grad_shape, const size_t &nums, BroadcastGradGradOpType op,
                              const T *x1, const T *x2, const T *dy1, const T *dy2, T *sopd_grad,
                              const uint32_t &device_id, cudaStream_t stream) {
  switch (op) {
    case BROADCAST_GRAD_GRAD_TYPE_MINIMUM:
      BroadcastGradGradOperator<T, MinimumGradGradFunc<T>>
        <<<CUDA_BLOCKS(device_id, nums), CUDA_THREADS(device_id), 0, stream>>>(
          x1_shape[0], x1_shape[1], x1_shape[2], x1_shape[3], x1_shape[4], x1_shape[5], x1_shape[6], x2_shape[0],
          x2_shape[1], x2_shape[2], x2_shape[3], x2_shape[4], x2_shape[5], x2_shape[6], grad_shape[0], grad_shape[1],
          grad_shape[2], grad_shape[3], grad_shape[4], grad_shape[5], grad_shape[6], nums, x1, x2, dy1, dy2, sopd_grad);
      break;
    case BROADCAST_GRAD_GRAD_TYPE_MAXIMUM:
      BroadcastGradGradOperator<T, MaximumGradGradFunc<T>>
        <<<CUDA_BLOCKS(device_id, nums), CUDA_THREADS(device_id), 0, stream>>>(
          x1_shape[0], x1_shape[1], x1_shape[2], x1_shape[3], x1_shape[4], x1_shape[5], x1_shape[6], x2_shape[0],
          x2_shape[1], x2_shape[2], x2_shape[3], x2_shape[4], x2_shape[5], x2_shape[6], grad_shape[0], grad_shape[1],
          grad_shape[2], grad_shape[3], grad_shape[4], grad_shape[5], grad_shape[6], nums, x1, x2, dy1, dy2, sopd_grad);
      break;
    default:
      break;
  }
  return GetCudaStatus();
}

template <typename T, typename Func>
__global__ void NoBroadcastGradGradOperator(const size_t nums, const T *x1, const T *x2, const T *dy1, const T *dy2,
                                            T *sopd_grad) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < nums; pos += blockDim.x * gridDim.x) {
    sopd_grad[pos] = Func()(x1[pos], x2[pos], dy1[pos], dy2[pos]);
  }
}

template <typename T>
cudaError_t NoBroadcastGradGrad(const size_t &nums, BroadcastGradGradOpType op, const T *x1, const T *x2, const T *dy1,
                                const T *dy2, T *sopd_grad, const uint32_t &device_id, cudaStream_t stream) {
  switch (op) {
    case BROADCAST_GRAD_GRAD_TYPE_MINIMUM:
      NoBroadcastGradGradOperator<T, MinimumGradGradFunc<T>>
        <<<CUDA_BLOCKS(device_id, nums), CUDA_THREADS(device_id), 0, stream>>>(nums, x1, x2, dy1, dy2, sopd_grad);
      break;
    case BROADCAST_GRAD_GRAD_TYPE_MAXIMUM:
      NoBroadcastGradGradOperator<T, MaximumGradGradFunc<T>>
        <<<CUDA_BLOCKS(device_id, nums), CUDA_THREADS(device_id), 0, stream>>>(nums, x1, x2, dy1, dy2, sopd_grad);
      break;
    default:
      break;
  }
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t NoBroadcastGradGrad(const size_t &nums, BroadcastGradGradOpType op, const half *x1,
                                                         const half *x2, const half *dy1, const half *dy2,
                                                         half *sopd_grad, const uint32_t &device_id,
                                                         cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t NoBroadcastGradGrad(const size_t &nums, BroadcastGradGradOpType op,
                                                         const float *x1, const float *x2, const float *dy1,
                                                         const float *dy2, float *sopd_grad, const uint32_t &device_id,
                                                         cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t NoBroadcastGradGrad(const size_t &nums, BroadcastGradGradOpType op,
                                                         const double *x1, const double *x2, const double *dy1,
                                                         const double *dy2, double *sopd_grad,
                                                         const uint32_t &device_id, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t NoBroadcastGradGrad(const size_t &nums, BroadcastGradGradOpType op, const int *x1,
                                                         const int *x2, const int *dy1, const int *dy2, int *sopd_grad,
                                                         const uint32_t &device_id, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t NoBroadcastGradGrad(const size_t &nums, BroadcastGradGradOpType op,
                                                         const int64_t *x1, const int64_t *x2, const int64_t *dy1,
                                                         const int64_t *dy2, int64_t *sopd_grad,
                                                         const uint32_t &device_id, cudaStream_t stream);

template CUDA_LIB_EXPORT cudaError_t BroadcastGradGrad(const std::vector<size_t> &x1_shape,
                                                       const std::vector<size_t> &x2_shape,
                                                       const std::vector<size_t> &grad_shape, const size_t &nums,
                                                       BroadcastGradGradOpType op, const half *x1, const half *x2,
                                                       const half *dy1, const half *dy2, half *sopd_grad,
                                                       const uint32_t &device_id, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t BroadcastGradGrad(const std::vector<size_t> &x1_shape,
                                                       const std::vector<size_t> &x2_shape,
                                                       const std::vector<size_t> &grad_shape, const size_t &nums,
                                                       BroadcastGradGradOpType op, const float *x1, const float *x2,
                                                       const float *dy1, const float *dy2, float *sopd_grad,
                                                       const uint32_t &device_id, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t BroadcastGradGrad(const std::vector<size_t> &x1_shape,
                                                       const std::vector<size_t> &x2_shape,
                                                       const std::vector<size_t> &grad_shape, const size_t &nums,
                                                       BroadcastGradGradOpType op, const double *x1, const double *x2,
                                                       const double *dy1, const double *dy2, double *sopd_grad,
                                                       const uint32_t &device_id, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t BroadcastGradGrad(const std::vector<size_t> &x1_shape,
                                                       const std::vector<size_t> &x2_shape,
                                                       const std::vector<size_t> &grad_shape, const size_t &nums,
                                                       BroadcastGradGradOpType op, const int *x1, const int *x2,
                                                       const int *dy1, const int *dy2, int *sopd_grad,
                                                       const uint32_t &device_id, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t BroadcastGradGrad(const std::vector<size_t> &x1_shape,
                                                       const std::vector<size_t> &x2_shape,
                                                       const std::vector<size_t> &grad_shape, const size_t &nums,
                                                       BroadcastGradGradOpType op, const int64_t *x1, const int64_t *x2,
                                                       const int64_t *dy1, const int64_t *dy2, int64_t *sopd_grad,
                                                       const uint32_t &device_id, cudaStream_t stream);
