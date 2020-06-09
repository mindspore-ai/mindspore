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

#include "kernel/gpu/cuda_impl/broadcast_impl.cuh"
#include "device/gpu/cuda_common.h"

template <typename T, typename S>
struct GreaterFunc {
  __device__ __forceinline__ S operator()(const T &lhs, const T &rhs) { return lhs > rhs ? true : false; }
};

template <typename T, typename S>
struct LessFunc {
  __device__ __forceinline__ S operator()(const T &lhs, const T &rhs) { return lhs < rhs ? true : false; }
};

template <typename T, typename S>
struct MinimumFunc {
  __device__ __forceinline__ S operator()(const T &lhs, const T &rhs) { return lhs < rhs ? lhs : rhs; }
};

template <typename T, typename S>
struct MaximumFunc {
  __device__ __forceinline__ S operator()(const T &lhs, const T &rhs) { return lhs > rhs ? lhs : rhs; }
};

template <typename T, typename S>
struct PowerFunc {
  __device__ __forceinline__ S operator()(const T &lhs, const T &rhs) { return pow(lhs, rhs); }
};

template <>
struct PowerFunc<half, half> {
  __device__ __forceinline__ half operator()(const half &lhs, const half &rhs) {
    return __float2half(pow(__half2float(lhs), __half2float(rhs)));
  }
};

template <typename T, typename S>
struct RealDivFunc {
  __device__ __forceinline__ S operator()(const T &lhs, const T &rhs) { return (lhs / rhs); }
};

template <typename T, typename S>
struct MulFunc {
  __device__ __forceinline__ S operator()(const T &lhs, const T &rhs) { return (lhs * rhs); }
};

template <typename T, typename S>
struct SubFunc {
  __device__ __forceinline__ S operator()(const T &lhs, const T &rhs) { return (lhs - rhs); }
};

template <typename T, typename S>
struct AddFunc {
  __device__ __forceinline__ S operator()(const T &lhs, const T &rhs) { return (lhs + rhs); }
};

template <>
struct PowerFunc<half, bool> {
  // invalid branch
  __device__ __forceinline__ half operator()(const half &lhs, const half &rhs) { return false; }
};

__device__ __forceinline__ int Index(const int &index, const int &dim) { return dim == 1 ? 0 : index; }

template <typename T, typename S, typename Func>
__device__ __forceinline__ void BroadcastOperator(const int &l0, const int &l1, const int &l2, const int &l3,
                                                  const int &r0, const int &r1, const int &r2, const int &r3,
                                                  const int &d0, const int &d1, const int &d2, const int &d3,
                                                  const T *input0, const T *input1, S *output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < d0 * d1 * d2 * d3; pos += blockDim.x * gridDim.x) {
    int i = pos / (d1 * d2 * d3) % d0;
    int j = pos / (d2 * d3) % d1;
    int k = pos / d3 % d2;
    int l = pos % d3;

    int l_index = Index(i, l0) * l1 * l2 * l3 + Index(j, l1) * l2 * l3 + Index(k, l2) * l3 + Index(l, l3);
    int r_index = Index(i, r0) * r1 * r2 * r3 + Index(j, r1) * r2 * r3 + Index(k, r2) * r3 + Index(l, r3);
    output[pos] = Func()(input0[l_index], input1[r_index]);
  }
}

template <typename T, typename S>
__global__ void BroadcastKernel(const int l0, const int l1, const int l2, const int l3, const int r0, const int r1,
                                const int r2, const int r3, const int d0, const int d1, const int d2, const int d3,
                                enum BroadcastOpType op, const T *input0, const T *input1, S *output) {
  switch (op) {
    case BROADCAST_TYPE_GREATER:
      return BroadcastOperator<T, S, GreaterFunc<T, S>>(l0, l1, l2, l3, r0, r1, r2, r3, d0, d1, d2, d3, input0, input1,
                                                        output);
    case BROADCAST_TYPE_LESS:
      return BroadcastOperator<T, S, LessFunc<T, S>>(l0, l1, l2, l3, r0, r1, r2, r3, d0, d1, d2, d3, input0, input1,
                                                     output);
    case BROADCAST_TYPE_MINIMUM:
      return BroadcastOperator<T, S, MinimumFunc<T, S>>(l0, l1, l2, l3, r0, r1, r2, r3, d0, d1, d2, d3, input0, input1,
                                                        output);
    case BROADCAST_TYPE_MAXIMUM:
      return BroadcastOperator<T, S, MaximumFunc<T, S>>(l0, l1, l2, l3, r0, r1, r2, r3, d0, d1, d2, d3, input0, input1,
                                                        output);
    case BROADCAST_TYPE_POWER:
      return BroadcastOperator<T, S, PowerFunc<T, S>>(l0, l1, l2, l3, r0, r1, r2, r3, d0, d1, d2, d3, input0, input1,
                                                      output);
    case BROADCAST_TYPE_REALDIV:
      return BroadcastOperator<T, S, RealDivFunc<T, S>>(l0, l1, l2, l3, r0, r1, r2, r3, d0, d1, d2, d3, input0, input1,
                                                      output);
    case BROADCAST_TYPE_MUL:
      return BroadcastOperator<T, S, MulFunc<T, S>>(l0, l1, l2, l3, r0, r1, r2, r3, d0, d1, d2, d3, input0, input1,
                                                      output);
    case BROADCAST_TYPE_SUB:
      return BroadcastOperator<T, S, SubFunc<T, S>>(l0, l1, l2, l3, r0, r1, r2, r3, d0, d1, d2, d3, input0, input1,
                                                      output);
    case BROADCAST_TYPE_ADD:
      return BroadcastOperator<T, S, AddFunc<T, S>>(l0, l1, l2, l3, r0, r1, r2, r3, d0, d1, d2, d3, input0, input1,
                                                      output);
  }
}

template <typename T, typename S>
void Broadcast(const int &l0, const int &l1, const int &l2, const int &l3, const int &r0, const int &r1, const int &r2,
               const int &r3, const int &d0, const int &d1, const int &d2, const int &d3, enum BroadcastOpType op,
               const T *input0, const T *input1, S *output, cudaStream_t stream) {
  int size = d0 * d1 * d2 * d3;
  BroadcastKernel<<<GET_BLOCKS(size), GET_THREADS, 0, stream>>>(l0, l1, l2, l3, r0, r1, r2, r3, d0, d1, d2, d3, op,
                                                                input0, input1, output);
}

template <typename T, typename S, typename Func>
__device__ __forceinline__ void NoBroadcastOperator(const int &nums, const T *input0, const T *input1, S *output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < nums; pos += blockDim.x * gridDim.x) {
    output[pos] = Func()(input0[pos], input1[pos]);
  }
}

template <typename T, typename S>
__global__ void NoBroadcastKernel(const int nums, enum BroadcastOpType op, const T *input0, const T *input1,
                                  S *output) {
  switch (op) {
    case BROADCAST_TYPE_GREATER:
      return NoBroadcastOperator<T, S, GreaterFunc<T, bool>>(nums, input0, input1, output);
    case BROADCAST_TYPE_LESS:
      return NoBroadcastOperator<T, S, LessFunc<T, bool>>(nums, input0, input1, output);
    case BROADCAST_TYPE_MINIMUM:
      return NoBroadcastOperator<T, S, MinimumFunc<T, S>>(nums, input0, input1, output);
    case BROADCAST_TYPE_MAXIMUM:
      return NoBroadcastOperator<T, S, MaximumFunc<T, S>>(nums, input0, input1, output);
    case BROADCAST_TYPE_POWER:
      return NoBroadcastOperator<T, S, PowerFunc<T, S>>(nums, input0, input1, output);
    case BROADCAST_TYPE_REALDIV:
      return NoBroadcastOperator<T, S, RealDivFunc<T, S>>(nums, input0, input1, output);
    case BROADCAST_TYPE_MUL:
      return NoBroadcastOperator<T, S, MulFunc<T, S>>(nums, input0, input1, output);
    case BROADCAST_TYPE_SUB:
      return NoBroadcastOperator<T, S, SubFunc<T, S>>(nums, input0, input1, output);
    case BROADCAST_TYPE_ADD:
      return NoBroadcastOperator<T, S, AddFunc<T, S>>(nums, input0, input1, output);
  }
}

template <typename T, typename S>
void NoBroadcast(const int &nums, enum BroadcastOpType op, const T *input0, const T *input1, S *output,
                 cudaStream_t stream) {
  NoBroadcastKernel<<<GET_BLOCKS(nums), GET_THREADS, 0, stream>>>(nums, op, input0, input1, output);
}

template void Broadcast(const int &l0, const int &l1, const int &l2, const int &l3, const int &r0, const int &r1,
                        const int &r2, const int &r3, const int &d0, const int &d1, const int &d2, const int &d3,
                        enum BroadcastOpType op, const float *input0, const float *input1, bool *output,
                        cudaStream_t stream);
template void Broadcast(const int &l0, const int &l1, const int &l2, const int &l3, const int &r0, const int &r1,
                        const int &r2, const int &r3, const int &d0, const int &d1, const int &d2, const int &d3,
                        enum BroadcastOpType op, const float *input0, const float *input1, float *output,
                        cudaStream_t stream);
template void Broadcast(const int &l0, const int &l1, const int &l2, const int &l3, const int &r0, const int &r1,
                        const int &r2, const int &r3, const int &d0, const int &d1, const int &d2, const int &d3,
                        enum BroadcastOpType op, const half *input0, const half *input1, bool *output,
                        cudaStream_t stream);
template void Broadcast(const int &l0, const int &l1, const int &l2, const int &l3, const int &r0, const int &r1,
                        const int &r2, const int &r3, const int &d0, const int &d1, const int &d2, const int &d3,
                        enum BroadcastOpType op, const half *input0, const half *input1, half *output,
                        cudaStream_t stream);
template void Broadcast(const int &l0, const int &l1, const int &l2, const int &l3, const int &r0, const int &r1,
                        const int &r2, const int &r3, const int &d0, const int &d1, const int &d2, const int &d3,
                        enum BroadcastOpType op, const int *input0, const int *input1, int *output,
                        cudaStream_t stream);
template void NoBroadcast(const int &nums, enum BroadcastOpType op, const float *input0, const float *input1,
                          bool *output, cudaStream_t stream);
template void NoBroadcast(const int &nums, enum BroadcastOpType op, const float *input0, const float *input1,
                          float *output, cudaStream_t stream);
template void NoBroadcast(const int &nums, enum BroadcastOpType op, const half *input0, const half *input1,
                          bool *output, cudaStream_t stream);
template void NoBroadcast(const int &nums, enum BroadcastOpType op, const half *input0, const half *input1,
                          half *output, cudaStream_t stream);
template void NoBroadcast(const int &nums, enum BroadcastOpType op, const int *input0, const int *input1,
                          int *output, cudaStream_t stream);
