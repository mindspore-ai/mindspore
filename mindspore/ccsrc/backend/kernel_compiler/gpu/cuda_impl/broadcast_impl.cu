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

#include <vector>
#include "backend/kernel_compiler/gpu/cuda_impl/broadcast_impl.cuh"
#include "runtime/device/gpu/cuda_common.h"

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
struct DivFunc {
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

template <typename T, typename S>
struct FloorDivFunc {
  __device__ __forceinline__ S operator()(const T &lhs, const T &rhs) { return floor(static_cast<float>(lhs / rhs)); }
};

template <>
struct FloorDivFunc<half, half> {
  __device__ __forceinline__ half operator()(const half &lhs, const half &rhs) {
    return __float2half(floor(__half2float(lhs) / __half2float(rhs)));
  }
};

template <>
struct FloorDivFunc<half, bool> {
  // invalid branch
  __device__ __forceinline__ half operator()(const half &lhs, const half &rhs) { return false; }
};

template <typename T, typename S>
struct AbsGradFunc {
  __device__ __forceinline__ S operator()(const T &lhs, const T &rhs) {
    T zero = 0.0;
    return lhs < zero ? -rhs : rhs;
  }
};

template <>
struct PowerFunc<half, bool> {
  // invalid branch
  __device__ __forceinline__ half operator()(const half &lhs, const half &rhs) { return false; }
};

__device__ __forceinline__ int Index(const int &index, const int &dim) { return dim == 1 ? 0 : index; }

template <typename T, typename S, typename Func>
__device__ __forceinline__ void BroadcastOperator(const int &l0, const int &l1, const int &l2, const int &l3,
                                                  const int &l4, const int &l5, const int &l6, const int &r0,
                                                  const int &r1, const int &r2, const int &r3, const int &r4,
                                                  const int &r5, const int &r6, const int &d0, const int &d1,
                                                  const int &d2, const int &d3, const int &d4, const int &d5,
                                                  const int &d6, const T *input0, const T *input1, S *output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < d0 * d1 * d2 * d3 * d4 * d5 * d6;
       pos += blockDim.x * gridDim.x) {
    int i = pos / (d1 * d2 * d3 * d4 * d5 * d6) % d0;
    int j = pos / (d2 * d3 * d4 * d5 * d6) % d1;
    int k = pos / (d3 * d4 * d5 * d6) % d2;
    int l = pos / (d4 * d5 * d6) % d3;
    int m = pos / (d5 * d6) % d4;
    int n = pos / d6 % d5;
    int o = pos % d6;

    int l_index = Index(i, l0) * l1 * l2 * l3 * l4 * l5 * l6;
    l_index += Index(j, l1) * l2 * l3 * l4 * l5 * l6;
    l_index += Index(k, l2) * l3 * l4 * l5 * l6;
    l_index += Index(l, l3) * l4 * l5 * l6;
    l_index += Index(m, l4) * l5 * l6;
    l_index += Index(n, l5) * l6;
    l_index += Index(o, l6);
    int r_index = Index(i, r0) * r1 * r2 * r3 * r4 * r5 * r6;
    r_index += Index(j, r1) * r2 * r3 * r4 * r5 * r6;
    r_index += Index(k, r2) * r3 * r4 * r5 * r6;
    r_index += Index(l, r3) * r4 * r5 * r6;
    r_index += Index(m, r4) * r5 * r6;
    r_index += Index(n, r5) * r6;
    r_index += Index(o, r6);
    output[pos] = Func()(input0[l_index], input1[r_index]);
  }
}

template <typename T, typename S>
__global__ void BroadcastKernel(const int l0, const int l1, const int l2, const int l3, const int l4, const int l5,
                                const int l6, const int r0, const int r1, const int r2, const int r3, const int r4,
                                const int r5, const int r6, const int d0, const int d1, const int d2, const int d3,
                                const int d4, const int d5, const int d6, enum BroadcastOpType op, const T *input0,
                                const T *input1, S *output) {
  switch (op) {
    case BROADCAST_TYPE_GREATER:
      return BroadcastOperator<T, S, GreaterFunc<T, S>>(l0, l1, l2, l3, l4, l5, l6, r0, r1, r2, r3, r4, r5, r6, d0, d1,
                                                        d2, d3, d4, d5, d6, input0, input1, output);
    case BROADCAST_TYPE_LESS:
      return BroadcastOperator<T, S, LessFunc<T, S>>(l0, l1, l2, l3, l4, l5, l6, r0, r1, r2, r3, r4, r5, r6, d0, d1, d2,
                                                     d3, d4, d5, d6, input0, input1, output);
    case BROADCAST_TYPE_MINIMUM:
      return BroadcastOperator<T, S, MinimumFunc<T, S>>(l0, l1, l2, l3, l4, l5, l6, r0, r1, r2, r3, r4, r5, r6, d0, d1,
                                                        d2, d3, d4, d5, d6, input0, input1, output);
    case BROADCAST_TYPE_MAXIMUM:
      return BroadcastOperator<T, S, MaximumFunc<T, S>>(l0, l1, l2, l3, l4, l5, l6, r0, r1, r2, r3, r4, r5, r6, d0, d1,
                                                        d2, d3, d4, d5, d6, input0, input1, output);
    case BROADCAST_TYPE_POWER:
      return BroadcastOperator<T, S, PowerFunc<T, S>>(l0, l1, l2, l3, l4, l5, l6, r0, r1, r2, r3, r4, r5, r6, d0, d1,
                                                      d2, d3, d4, d5, d6, input0, input1, output);
    case BROADCAST_TYPE_REALDIV:
      return BroadcastOperator<T, S, RealDivFunc<T, S>>(l0, l1, l2, l3, l4, l5, l6, r0, r1, r2, r3, r4, r5, r6, d0, d1,
                                                        d2, d3, d4, d5, d6, input0, input1, output);
    case BROADCAST_TYPE_MUL:
      return BroadcastOperator<T, S, MulFunc<T, S>>(l0, l1, l2, l3, l4, l5, l6, r0, r1, r2, r3, r4, r5, r6, d0, d1, d2,
                                                    d3, d4, d5, d6, input0, input1, output);
    case BROADCAST_TYPE_SUB:
      return BroadcastOperator<T, S, SubFunc<T, S>>(l0, l1, l2, l3, l4, l5, l6, r0, r1, r2, r3, r4, r5, r6, d0, d1, d2,
                                                    d3, d4, d5, d6, input0, input1, output);
    case BROADCAST_TYPE_ADD:
      return BroadcastOperator<T, S, AddFunc<T, S>>(l0, l1, l2, l3, l4, l5, l6, r0, r1, r2, r3, r4, r5, r6, d0, d1, d2,
                                                    d3, d4, d5, d6, input0, input1, output);
    case BROADCAST_TYPE_FLOORDIV:
      return BroadcastOperator<T, S, FloorDivFunc<T, S>>(l0, l1, l2, l3, l4, l5, l6, r0, r1, r2, r3, r4, r5, r6, d0, d1,
                                                         d2, d3, d4, d5, d6, input0, input1, output);
    case BROADCAST_TYPE_ABSGRAD:
      return BroadcastOperator<T, S, AbsGradFunc<T, S>>(l0, l1, l2, l3, l4, l5, l6, r0, r1, r2, r3, r4, r5, r6, d0, d1,
                                                        d2, d3, d4, d5, d6, input0, input1, output);
    case BROADCAST_TYPE_DIV:
      return BroadcastOperator<T, S, DivFunc<T, S>>(l0, l1, l2, l3, l4, l5, l6, r0, r1, r2, r3, r4, r5, r6, d0, d1, d2,
                                                    d3, d4, d5, d6, input0, input1, output);
  }
}

template <typename T, typename S>
void Broadcast(const std::vector<int> &lhs_shape, const std::vector<int> &rhs_shape,
               const std::vector<int> &output_shape, enum BroadcastOpType op, const T *input0, const T *input1,
               S *output, cudaStream_t stream) {
  int size = 1;
  for (auto d : output_shape) {
    size *= d;
  }
  BroadcastKernel<<<GET_BLOCKS(size), GET_THREADS, 0, stream>>>(
    lhs_shape[0], lhs_shape[1], lhs_shape[2], lhs_shape[3], lhs_shape[4], lhs_shape[5], lhs_shape[6], rhs_shape[0],
    rhs_shape[1], rhs_shape[2], rhs_shape[3], rhs_shape[4], rhs_shape[5], rhs_shape[6], output_shape[0],
    output_shape[1], output_shape[2], output_shape[3], output_shape[4], output_shape[5], output_shape[6], op, input0,
    input1, output);
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
    case BROADCAST_TYPE_FLOORDIV:
      return NoBroadcastOperator<T, S, FloorDivFunc<T, S>>(nums, input0, input1, output);
    case BROADCAST_TYPE_ABSGRAD:
      return NoBroadcastOperator<T, S, AbsGradFunc<T, S>>(nums, input0, input1, output);
    case BROADCAST_TYPE_DIV:
      return NoBroadcastOperator<T, S, DivFunc<T, S>>(nums, input0, input1, output);
  }
}

template <typename T, typename S>
void NoBroadcast(const int &nums, enum BroadcastOpType op, const T *input0, const T *input1, S *output,
                 cudaStream_t stream) {
  NoBroadcastKernel<<<GET_BLOCKS(nums), GET_THREADS, 0, stream>>>(nums, op, input0, input1, output);
}

template <typename T>
__global__ void BroadcastToKernel(const int i0, const int i1, const int i2, const int i3, const int o0, const int o1,
                                  const int o2, const int o3, const T *input_addr, T *output_addr) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < o0 * o1 * o2 * o3; pos += blockDim.x * gridDim.x) {
    int i = pos / (o1 * o2 * o3) % o0;
    int j = pos / (o2 * o3) % o1;
    int k = pos / o3 % o2;
    int l = pos % o3;

    int input_idx = Index(i, i0) * i1 * i2 * i3 + Index(j, i1) * i2 * i3 + Index(k, i2) * i3 + Index(l, i3);
    output_addr[pos] = input_addr[input_idx];
  }
}

template <typename T>
void BroadcastTo(const int &i0, const int &i1, const int &i2, const int &i3, const int &o0, const int &o1,
                 const int &o2, const int &o3, const T *input_addr, T *output_addr, cudaStream_t stream) {
  int nums = o0 * o1 * o2 * o3;
  BroadcastToKernel<<<GET_BLOCKS(nums), GET_THREADS, 0, stream>>>(i0, i1, i2, i3, o0, o1, o2, o3, input_addr,
                                                                  output_addr);
}

template void Broadcast(const std::vector<int> &lhs_shape, const std::vector<int> &rhs_shape,
                        const std::vector<int> &output_shape, enum BroadcastOpType op, const float *input0,
                        const float *input1, bool *output, cudaStream_t stream);
template void Broadcast(const std::vector<int> &lhs_shape, const std::vector<int> &rhs_shape,
                        const std::vector<int> &output_shape, enum BroadcastOpType op, const float *input0,
                        const float *input1, float *output, cudaStream_t stream);
template void Broadcast(const std::vector<int> &lhs_shape, const std::vector<int> &rhs_shape,
                        const std::vector<int> &output_shape, enum BroadcastOpType op, const half *input0,
                        const half *input1, bool *output, cudaStream_t stream);
template void Broadcast(const std::vector<int> &lhs_shape, const std::vector<int> &rhs_shape,
                        const std::vector<int> &output_shape, enum BroadcastOpType op, const half *input0,
                        const half *input1, half *output, cudaStream_t stream);
template void Broadcast(const std::vector<int> &lhs_shape, const std::vector<int> &rhs_shape,
                        const std::vector<int> &output_shape, enum BroadcastOpType op, const int *input0,
                        const int *input1, int *output, cudaStream_t stream);
template void Broadcast(const std::vector<int> &lhs_shape, const std::vector<int> &rhs_shape,
                        const std::vector<int> &output_shape, enum BroadcastOpType op, const int *input0,
                        const int *input1, bool *output, cudaStream_t stream);
template void NoBroadcast(const int &nums, enum BroadcastOpType op, const float *input0, const float *input1,
                          bool *output, cudaStream_t stream);
template void NoBroadcast(const int &nums, enum BroadcastOpType op, const float *input0, const float *input1,
                          float *output, cudaStream_t stream);
template void NoBroadcast(const int &nums, enum BroadcastOpType op, const half *input0, const half *input1,
                          bool *output, cudaStream_t stream);
template void NoBroadcast(const int &nums, enum BroadcastOpType op, const half *input0, const half *input1,
                          half *output, cudaStream_t stream);
template void NoBroadcast(const int &nums, enum BroadcastOpType op, const int *input0, const int *input1, int *output,
                          cudaStream_t stream);
template void NoBroadcast(const int &nums, enum BroadcastOpType op, const int *input0, const int *input1, bool *output,
                          cudaStream_t stream);
template void BroadcastTo(const int &i0, const int &i1, const int &i2, const int &i3, const int &o0, const int &o1,
                          const int &o2, const int &o3, const float *input_addr, float *output_addr,
                          cudaStream_t stream);
template void BroadcastTo(const int &i0, const int &i1, const int &i2, const int &i3, const int &o0, const int &o1,
                          const int &o2, const int &o3, const half *input_addr, half *output_addr, cudaStream_t stream);
