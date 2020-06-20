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

#include "kernel/gpu/cuda_impl/broadcast_grad_impl.cuh"
#include "device/gpu/cuda_common.h"

template <typename T>
struct MinimumGradFunc {
  __device__ __forceinline__ void operator()(const T &x1, const T &x2, const T &dy, T *dx1, T *dx2) {
    if (x1 < x2) {
      atomicAdd(dx1, dy);
    } else {
      atomicAdd(dx2, dy);
    }
  }
};

template <typename T>
struct MaximumGradFunc {
  __device__ __forceinline__ void operator()(const T &x1, const T &x2, const T &dy, T *dx1, T *dx2) {
    if (x1 > x2) {
      atomicAdd(dx1, dy);
    } else {
      atomicAdd(dx2, dy);
    }
  }
};

__device__ __forceinline__ int Index(const int &index, const int &dim) { return dim == 1 ? 0 : index; }

template <typename T, typename Func>
__device__ __forceinline__ void BroadcastGradOperator(const int &l0, const int &l1, const int &l2, const int &l3,
                                                      const int &r0, const int &r1, const int &r2, const int &r3,
                                                      const int &d0, const int &d1, const int &d2, const int &d3,
                                                      const T *x1, const T *x2, const T *dy, T *dx1, T *dx2) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < d0 * d1 * d2 * d3; pos += blockDim.x * gridDim.x) {
    int i = pos / (d1 * d2 * d3) % d0;
    int j = pos / (d2 * d3) % d1;
    int k = pos / d3 % d2;
    int l = pos % d3;

    int l_index = Index(i, l0) * l1 * l2 * l3 + Index(j, l1) * l2 * l3 + Index(k, l2) * l3 + Index(l, l3);
    int r_index = Index(i, r0) * r1 * r2 * r3 + Index(j, r1) * r2 * r3 + Index(k, r2) * r3 + Index(l, r3);
    Func()(x1[l_index], x2[r_index], dy[pos], dx1 + l_index, dx2 + r_index);
  }
}

template <typename T>
__global__ void BroadcastGradKernel(const int l0, const int l1, const int l2, const int l3, const int r0, const int r1,
                                    const int r2, const int r3, const int d0, const int d1, const int d2, const int d3,
                                    enum BroadcastGradOpType op, const T *x1, const T *x2, const T *dy, T *dx1,
                                    T *dx2) {
  switch (op) {
    case BROADCAST_GRAD_TYPE_MINIMUM:
      return BroadcastGradOperator<T, MinimumGradFunc<T>>(l0, l1, l2, l3, r0, r1, r2, r3, d0, d1, d2, d3, x1, x2, dy,
                                                          dx1, dx2);
    case BROADCAST_GRAD_TYPE_MAXIMUM:
      return BroadcastGradOperator<T, MaximumGradFunc<T>>(l0, l1, l2, l3, r0, r1, r2, r3, d0, d1, d2, d3, x1, x2, dy,
                                                          dx1, dx2);
  }
}

template <typename T>
void BroadcastGrad(const int &l0, const int &l1, const int &l2, const int &l3, const int &r0, const int &r1,
                   const int &r2, const int &r3, const int &d0, const int &d1, const int &d2, const int &d3,
                   enum BroadcastGradOpType op, const T *x1, const T *x2, const T *dy, T *dx1, T *dx2,
                   cudaStream_t stream) {
  int size = d0 * d1 * d2 * d3;
  BroadcastGradKernel<<<GET_BLOCKS(size), GET_THREADS, 0, stream>>>(l0, l1, l2, l3, r0, r1, r2, r3, d0, d1, d2, d3, op,
                                                                    x1, x2, dy, dx1, dx2);
}

template <typename T, typename Func>
__device__ __forceinline__ void NoBroadcastOperator(const int &nums, const T *x1, const T *x2, const T *dy, T *dx1,
                                                    T *dx2) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < nums; pos += blockDim.x * gridDim.x) {
    Func()(x1[pos], x2[pos], dy[pos], dx1 + pos, dx2 + pos);
  }
}

template <typename T>
__global__ void NoBroadcastGradKernel(const int nums, enum BroadcastGradOpType op, const T *x1, const T *x2,
                                      const T *dy, T *dx1, T *dx2) {
  switch (op) {
    case BROADCAST_GRAD_TYPE_MINIMUM:
      return NoBroadcastOperator<T, MinimumGradFunc<T>>(nums, x1, x2, dy, dx1, dx2);
    case BROADCAST_GRAD_TYPE_MAXIMUM:
      return NoBroadcastOperator<T, MaximumGradFunc<T>>(nums, x1, x2, dy, dx1, dx2);
  }
}

template <typename T>
void NoBroadcastGrad(const int &nums, enum BroadcastGradOpType op, const T *x1, const T *x2, const T *dy, T *dx1,
                     T *dx2, cudaStream_t stream) {
  NoBroadcastGradKernel<<<GET_BLOCKS(nums), GET_THREADS, 0, stream>>>(nums, op, x1, x2, dy, dx1, dx2);
}

template void NoBroadcastGrad(const int &nums, enum BroadcastGradOpType op, const float *x1, const float *x2,
                              const float *dy, float *dx1, float *dx2, cudaStream_t stream);
template void NoBroadcastGrad(const int &nums, enum BroadcastGradOpType op, const int *x1, const int *x2,
                              const int *dy, int *dx1, int *dx2, cudaStream_t stream);
template void BroadcastGrad(const int &l0, const int &l1, const int &l2, const int &l3, const int &r0, const int &r1,
                            const int &r2, const int &r3, const int &d0, const int &d1, const int &d2, const int &d3,
                            enum BroadcastGradOpType op, const float *x1, const float *x2, const float *dy, float *dx1,
                            float *dx2, cudaStream_t stream);
template void BroadcastGrad(const int &l0, const int &l1, const int &l2, const int &l3, const int &r0, const int &r1,
                            const int &r2, const int &r3, const int &d0, const int &d1, const int &d2, const int &d3,
                            enum BroadcastGradOpType op, const int *x1, const int *x2, const int *dy, int *dx1,
                            int *dx2, cudaStream_t stream);
