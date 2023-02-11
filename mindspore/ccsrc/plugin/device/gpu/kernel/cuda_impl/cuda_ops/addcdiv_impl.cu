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
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/addcdiv_impl.cuh"

// Broadcast comparison
__device__ __forceinline__ int64_t Index(const int64_t &index, const int64_t &dim) { return dim == 1 ? 0 : index; }

__global__ void Addcdiv_all_same(const half *input_data, const half *x1, const half *x2, const half *value,
                                 half *output, const int64_t size) {
  for (int64_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    output[pos] = input_data[pos] + value[pos] * x1[pos] / x2[pos];
  }
}
template <typename T, typename VT>
__global__ void Addcdiv_all_same(const T *input_data, const T *x1, const T *x2, const VT *value, T *output,
                                 const int64_t size) {
  for (int64_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    output[pos] = static_cast<T>(input_data[pos] + value[pos] * x1[pos] / x2[pos]);
  }
}
template <typename T>
__global__ void Addcdiv_all_same(const T *input_data, const T *x1, const T *x2, const half *value, T *output,
                                 const int64_t size) {
  for (int64_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    output[pos] = static_cast<T>(input_data[pos] + __half2float(value[pos]) * x1[pos] / x2[pos]);
  }
}
template <typename VT>
__global__ void Addcdiv_all_same(const half *input_data, const half *x1, const half *x2, const VT *value, half *output,
                                 const int64_t size) {
  for (int64_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    output[pos] = input_data[pos] + __float2half(static_cast<float>(value[pos])) * x1[pos] / x2[pos];
  }
}

template <typename T, typename VT>
__global__ void Addcdiv_all_same_value1(const T *input_data, const T *x1, const T *x2, const VT *value, T *output,
                                        const int64_t size) {
  for (int64_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    output[pos] = static_cast<T>(input_data[pos] + value[0] * x1[pos] / x2[pos]);
  }
}
template <typename T>
__global__ void Addcdiv_all_same_value1(const T *input_data, const T *x1, const T *x2, const half *value, T *output,
                                        const int64_t size) {
  for (int64_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    output[pos] = static_cast<T>(input_data[pos] + __half2float(value[0]) * x1[pos] / x2[pos]);
  }
}
template <typename VT>
__global__ void Addcdiv_all_same_value1(const half *input_data, const half *x1, const half *x2, const VT *value,
                                        half *output, const int64_t size) {
  for (int64_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    output[pos] = input_data[pos] + __float2half(static_cast<float>(value[0])) * x1[pos] / x2[pos];
  }
}

__global__ void Addcdiv_all_same_value1(const half *input_data, const half *x1, const half *x2, const half *value,
                                        half *output, const int64_t size) {
  for (int64_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    output[pos] = input_data[pos] + value[0] * x1[pos] / x2[pos];
  }
}
template <typename T, typename VT>
__global__ void Addcdiv(const int64_t l0, const int64_t l1, const int64_t l2, const int64_t l3, const int64_t l4,
                        const int64_t l5, const int64_t l6,

                        const int64_t r0, const int64_t r1, const int64_t r2, const int64_t r3, const int64_t r4,
                        const int64_t r5, const int64_t r6,

                        const int64_t u0, const int64_t u1, const int64_t u2, const int64_t u3, const int64_t u4,
                        const int64_t u5, const int64_t u6,

                        const int64_t v0, const int64_t v1, const int64_t v2, const int64_t v3, const int64_t v4,
                        const int64_t v5, const int64_t v6,

                        const int64_t d0, const int64_t d1, const int64_t d2, const int64_t d3, const int64_t d4,
                        const int64_t d5, const int64_t d6,

                        const int64_t d_1, const int64_t d_2, const int64_t d_3, const int64_t d_4, const int64_t d_5,
                        const int64_t l_1, const int64_t l_2, const int64_t l_3, const int64_t l_4, const int64_t l_5,
                        const int64_t r_1, const int64_t r_2, const int64_t r_3, const int64_t r_4, const int64_t r_5,
                        const int64_t u_1, const int64_t u_2, const int64_t u_3, const int64_t u_4, const int64_t u_5,
                        const int64_t v_1, const int64_t v_2, const int64_t v_3, const int64_t v_4, const int64_t v_5,

                        const T *input_data, const T *x1, const T *x2, const VT *value, T *output, const int64_t size) {
  for (int64_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    int64_t i = pos / d_1;
    int64_t j = pos / d_2 % d1;
    int64_t k = pos / d_3 % d2;
    int64_t l = pos / d_4 % d3;
    int64_t m = pos / d_5 % d4;
    int64_t n = pos / d6 % d5;
    int64_t o = pos % d6;

    int64_t l_index = Index(i, l0) * l_1;
    l_index += Index(j, l1) * l_2;
    l_index += Index(k, l2) * l_3;
    l_index += Index(l, l3) * l_4;
    l_index += Index(m, l4) * l_5;
    l_index += Index(n, l5) * l6;
    l_index += Index(o, l6);
    int64_t r_index = Index(i, r0) * r_1;
    r_index += Index(j, r1) * r_2;
    r_index += Index(k, r2) * r_3;
    r_index += Index(l, r3) * r_4;
    r_index += Index(m, r4) * r_5;
    r_index += Index(n, r5) * r6;
    r_index += Index(o, r6);
    int64_t u_index = Index(i, u0) * u_1;
    u_index += Index(j, u1) * u_2;
    u_index += Index(k, u2) * u_3;
    u_index += Index(l, u3) * u_4;
    u_index += Index(m, u4) * u_5;
    u_index += Index(n, u5) * u6;
    u_index += Index(o, u6);
    int64_t v_index = Index(i, v0) * v_1;
    v_index += Index(j, v1) * v_2;
    v_index += Index(k, v2) * v_3;
    v_index += Index(l, v3) * v_4;
    v_index += Index(m, v4) * v_5;
    v_index += Index(n, v5) * v6;
    v_index += Index(o, v6);

    output[pos] = static_cast<T>(input_data[l_index] + value[v_index] * x1[r_index] / x2[u_index]);
  }
}
template <typename T>
__global__ void Addcdiv(const int64_t l0, const int64_t l1, const int64_t l2, const int64_t l3, const int64_t l4,
                        const int64_t l5, const int64_t l6,

                        const int64_t r0, const int64_t r1, const int64_t r2, const int64_t r3, const int64_t r4,
                        const int64_t r5, const int64_t r6,

                        const int64_t u0, const int64_t u1, const int64_t u2, const int64_t u3, const int64_t u4,
                        const int64_t u5, const int64_t u6,

                        const int64_t v0, const int64_t v1, const int64_t v2, const int64_t v3, const int64_t v4,
                        const int64_t v5, const int64_t v6,

                        const int64_t d0, const int64_t d1, const int64_t d2, const int64_t d3, const int64_t d4,
                        const int64_t d5, const int64_t d6,

                        const int64_t d_1, const int64_t d_2, const int64_t d_3, const int64_t d_4, const int64_t d_5,
                        const int64_t l_1, const int64_t l_2, const int64_t l_3, const int64_t l_4, const int64_t l_5,
                        const int64_t r_1, const int64_t r_2, const int64_t r_3, const int64_t r_4, const int64_t r_5,
                        const int64_t u_1, const int64_t u_2, const int64_t u_3, const int64_t u_4, const int64_t u_5,
                        const int64_t v_1, const int64_t v_2, const int64_t v_3, const int64_t v_4, const int64_t v_5,

                        const T *input_data, const T *x1, const T *x2, const half *value, T *output,
                        const int64_t size) {
  for (int64_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    int64_t i = pos / d_1;
    int64_t j = pos / d_2 % d1;
    int64_t k = pos / d_3 % d2;
    int64_t l = pos / d_4 % d3;
    int64_t m = pos / d_5 % d4;
    int64_t n = pos / d6 % d5;
    int64_t o = pos % d6;

    int64_t l_index = Index(i, l0) * l_1;
    l_index += Index(j, l1) * l_2;
    l_index += Index(k, l2) * l_3;
    l_index += Index(l, l3) * l_4;
    l_index += Index(m, l4) * l_5;
    l_index += Index(n, l5) * l6;
    l_index += Index(o, l6);
    int64_t r_index = Index(i, r0) * r_1;
    r_index += Index(j, r1) * r_2;
    r_index += Index(k, r2) * r_3;
    r_index += Index(l, r3) * r_4;
    r_index += Index(m, r4) * r_5;
    r_index += Index(n, r5) * r6;
    r_index += Index(o, r6);
    int64_t u_index = Index(i, u0) * u_1;
    u_index += Index(j, u1) * u_2;
    u_index += Index(k, u2) * u_3;
    u_index += Index(l, u3) * u_4;
    u_index += Index(m, u4) * u_5;
    u_index += Index(n, u5) * u6;
    u_index += Index(o, u6);
    int64_t v_index = Index(i, v0) * v_1;
    v_index += Index(j, v1) * v_2;
    v_index += Index(k, v2) * v_3;
    v_index += Index(l, v3) * v_4;
    v_index += Index(m, v4) * v_5;
    v_index += Index(n, v5) * v6;
    v_index += Index(o, v6);

    output[pos] = static_cast<T>(input_data[l_index] + __half2float(value[v_index]) * x1[r_index] / x2[u_index]);
  }
}

template <typename VT>
__global__ void Addcdiv(const int64_t l0, const int64_t l1, const int64_t l2, const int64_t l3, const int64_t l4,
                        const int64_t l5, const int64_t l6,

                        const int64_t r0, const int64_t r1, const int64_t r2, const int64_t r3, const int64_t r4,
                        const int64_t r5, const int64_t r6,

                        const int64_t u0, const int64_t u1, const int64_t u2, const int64_t u3, const int64_t u4,
                        const int64_t u5, const int64_t u6,

                        const int64_t v0, const int64_t v1, const int64_t v2, const int64_t v3, const int64_t v4,
                        const int64_t v5, const int64_t v6,

                        const int64_t d0, const int64_t d1, const int64_t d2, const int64_t d3, const int64_t d4,
                        const int64_t d5, const int64_t d6,

                        const int64_t d_1, const int64_t d_2, const int64_t d_3, const int64_t d_4, const int64_t d_5,
                        const int64_t l_1, const int64_t l_2, const int64_t l_3, const int64_t l_4, const int64_t l_5,
                        const int64_t r_1, const int64_t r_2, const int64_t r_3, const int64_t r_4, const int64_t r_5,
                        const int64_t u_1, const int64_t u_2, const int64_t u_3, const int64_t u_4, const int64_t u_5,
                        const int64_t v_1, const int64_t v_2, const int64_t v_3, const int64_t v_4, const int64_t v_5,

                        const half *input_data, const half *x1, const half *x2, const VT *value, half *output,
                        const int64_t size) {
  for (int64_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    int64_t i = pos / d_1;
    int64_t j = pos / d_2 % d1;
    int64_t k = pos / d_3 % d2;
    int64_t l = pos / d_4 % d3;
    int64_t m = pos / d_5 % d4;
    int64_t n = pos / d6 % d5;
    int64_t o = pos % d6;

    int64_t l_index = Index(i, l0) * l_1;
    l_index += Index(j, l1) * l_2;
    l_index += Index(k, l2) * l_3;
    l_index += Index(l, l3) * l_4;
    l_index += Index(m, l4) * l_5;
    l_index += Index(n, l5) * l6;
    l_index += Index(o, l6);
    int64_t r_index = Index(i, r0) * r_1;
    r_index += Index(j, r1) * r_2;
    r_index += Index(k, r2) * r_3;
    r_index += Index(l, r3) * r_4;
    r_index += Index(m, r4) * r_5;
    r_index += Index(n, r5) * r6;
    r_index += Index(o, r6);
    int64_t u_index = Index(i, u0) * u_1;
    u_index += Index(j, u1) * u_2;
    u_index += Index(k, u2) * u_3;
    u_index += Index(l, u3) * u_4;
    u_index += Index(m, u4) * u_5;
    u_index += Index(n, u5) * u6;
    u_index += Index(o, u6);
    int64_t v_index = Index(i, v0) * v_1;
    v_index += Index(j, v1) * v_2;
    v_index += Index(k, v2) * v_3;
    v_index += Index(l, v3) * v_4;
    v_index += Index(m, v4) * v_5;
    v_index += Index(n, v5) * v6;
    v_index += Index(o, v6);

    output[pos] = input_data[l_index] + __float2half(static_cast<float>(value[v_index])) * x1[r_index] / x2[u_index];
  }
}

__global__ void Addcdiv(const int64_t l0, const int64_t l1, const int64_t l2, const int64_t l3, const int64_t l4,
                        const int64_t l5, const int64_t l6,

                        const int64_t r0, const int64_t r1, const int64_t r2, const int64_t r3, const int64_t r4,
                        const int64_t r5, const int64_t r6,

                        const int64_t u0, const int64_t u1, const int64_t u2, const int64_t u3, const int64_t u4,
                        const int64_t u5, const int64_t u6,

                        const int64_t v0, const int64_t v1, const int64_t v2, const int64_t v3, const int64_t v4,
                        const int64_t v5, const int64_t v6,

                        const int64_t d0, const int64_t d1, const int64_t d2, const int64_t d3, const int64_t d4,
                        const int64_t d5, const int64_t d6,

                        const int64_t d_1, const int64_t d_2, const int64_t d_3, const int64_t d_4, const int64_t d_5,
                        const int64_t l_1, const int64_t l_2, const int64_t l_3, const int64_t l_4, const int64_t l_5,
                        const int64_t r_1, const int64_t r_2, const int64_t r_3, const int64_t r_4, const int64_t r_5,
                        const int64_t u_1, const int64_t u_2, const int64_t u_3, const int64_t u_4, const int64_t u_5,
                        const int64_t v_1, const int64_t v_2, const int64_t v_3, const int64_t v_4, const int64_t v_5,

                        const half *input_data, const half *x1, const half *x2, const half *value, half *output,
                        const int64_t size) {
  for (int64_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    int64_t i = pos / d_1;
    int64_t j = pos / d_2 % d1;
    int64_t k = pos / d_3 % d2;
    int64_t l = pos / d_4 % d3;
    int64_t m = pos / d_5 % d4;
    int64_t n = pos / d6 % d5;
    int64_t o = pos % d6;

    int64_t l_index = Index(i, l0) * l_1;
    l_index += Index(j, l1) * l_2;
    l_index += Index(k, l2) * l_3;
    l_index += Index(l, l3) * l_4;
    l_index += Index(m, l4) * l_5;
    l_index += Index(n, l5) * l6;
    l_index += Index(o, l6);
    int64_t r_index = Index(i, r0) * r_1;
    r_index += Index(j, r1) * r_2;
    r_index += Index(k, r2) * r_3;
    r_index += Index(l, r3) * r_4;
    r_index += Index(m, r4) * r_5;
    r_index += Index(n, r5) * r6;
    r_index += Index(o, r6);
    int64_t u_index = Index(i, u0) * u_1;
    u_index += Index(j, u1) * u_2;
    u_index += Index(k, u2) * u_3;
    u_index += Index(l, u3) * u_4;
    u_index += Index(m, u4) * u_5;
    u_index += Index(n, u5) * u6;
    u_index += Index(o, u6);
    int64_t v_index = Index(i, v0) * v_1;
    v_index += Index(j, v1) * v_2;
    v_index += Index(k, v2) * v_3;
    v_index += Index(l, v3) * v_4;
    v_index += Index(m, v4) * v_5;
    v_index += Index(n, v5) * v6;
    v_index += Index(o, v6);

    output[pos] = input_data[l_index] + value[v_index] * x1[r_index] / x2[u_index];
  }
}
template <typename T, typename VT>
__global__ void Addcdiv_value1(const int64_t l0, const int64_t l1, const int64_t l2, const int64_t l3, const int64_t l4,
                               const int64_t l5, const int64_t l6,

                               const int64_t r0, const int64_t r1, const int64_t r2, const int64_t r3, const int64_t r4,
                               const int64_t r5, const int64_t r6,

                               const int64_t u0, const int64_t u1, const int64_t u2, const int64_t u3, const int64_t u4,
                               const int64_t u5, const int64_t u6,

                               const int64_t d0, const int64_t d1, const int64_t d2, const int64_t d3, const int64_t d4,
                               const int64_t d5, const int64_t d6,

                               const int64_t d_1, const int64_t d_2, const int64_t d_3, const int64_t d_4,
                               const int64_t d_5, const int64_t l_1, const int64_t l_2, const int64_t l_3,
                               const int64_t l_4, const int64_t l_5, const int64_t r_1, const int64_t r_2,
                               const int64_t r_3, const int64_t r_4, const int64_t r_5, const int64_t u_1,
                               const int64_t u_2, const int64_t u_3, const int64_t u_4, const int64_t u_5,

                               const T *input_data, const T *x1, const T *x2, const VT *value, T *output,
                               const int64_t size) {
  for (int64_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    int64_t i = pos / d_1;
    int64_t j = pos / d_2 % d1;
    int64_t k = pos / d_3 % d2;
    int64_t l = pos / d_4 % d3;
    int64_t m = pos / d_5 % d4;
    int64_t n = pos / d6 % d5;
    int64_t o = pos % d6;

    int64_t l_index = Index(i, l0) * l_1;
    l_index += Index(j, l1) * l_2;
    l_index += Index(k, l2) * l_3;
    l_index += Index(l, l3) * l_4;
    l_index += Index(m, l4) * l_5;
    l_index += Index(n, l5) * l6;
    l_index += Index(o, l6);
    int64_t r_index = Index(i, r0) * r_1;
    r_index += Index(j, r1) * r_2;
    r_index += Index(k, r2) * r_3;
    r_index += Index(l, r3) * r_4;
    r_index += Index(m, r4) * r_5;
    r_index += Index(n, r5) * r6;
    r_index += Index(o, r6);
    int64_t u_index = Index(i, u0) * u_1;
    u_index += Index(j, u1) * u_2;
    u_index += Index(k, u2) * u_3;
    u_index += Index(l, u3) * u_4;
    u_index += Index(m, u4) * u_5;
    u_index += Index(n, u5) * u6;
    u_index += Index(o, u6);

    VT v = value[0];
    output[pos] = static_cast<T>(input_data[l_index] + v * x1[r_index] / x2[u_index]);
  }
}

template <typename T>
__global__ void Addcdiv_value1(const int64_t l0, const int64_t l1, const int64_t l2, const int64_t l3, const int64_t l4,
                               const int64_t l5, const int64_t l6,

                               const int64_t r0, const int64_t r1, const int64_t r2, const int64_t r3, const int64_t r4,
                               const int64_t r5, const int64_t r6,

                               const int64_t u0, const int64_t u1, const int64_t u2, const int64_t u3, const int64_t u4,
                               const int64_t u5, const int64_t u6,

                               const int64_t d0, const int64_t d1, const int64_t d2, const int64_t d3, const int64_t d4,
                               const int64_t d5, const int64_t d6,

                               const int64_t d_1, const int64_t d_2, const int64_t d_3, const int64_t d_4,
                               const int64_t d_5, const int64_t l_1, const int64_t l_2, const int64_t l_3,
                               const int64_t l_4, const int64_t l_5, const int64_t r_1, const int64_t r_2,
                               const int64_t r_3, const int64_t r_4, const int64_t r_5, const int64_t u_1,
                               const int64_t u_2, const int64_t u_3, const int64_t u_4, const int64_t u_5,

                               const T *input_data, const T *x1, const T *x2, const half *value, T *output,
                               const int64_t size) {
  for (int64_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    int64_t i = pos / d_1;
    int64_t j = pos / d_2 % d1;
    int64_t k = pos / d_3 % d2;
    int64_t l = pos / d_4 % d3;
    int64_t m = pos / d_5 % d4;
    int64_t n = pos / d6 % d5;
    int64_t o = pos % d6;

    int64_t l_index = Index(i, l0) * l_1;
    l_index += Index(j, l1) * l_2;
    l_index += Index(k, l2) * l_3;
    l_index += Index(l, l3) * l_4;
    l_index += Index(m, l4) * l_5;
    l_index += Index(n, l5) * l6;
    l_index += Index(o, l6);
    int64_t r_index = Index(i, r0) * r_1;
    r_index += Index(j, r1) * r_2;
    r_index += Index(k, r2) * r_3;
    r_index += Index(l, r3) * r_4;
    r_index += Index(m, r4) * r_5;
    r_index += Index(n, r5) * r6;
    r_index += Index(o, r6);
    int64_t u_index = Index(i, u0) * u_1;
    u_index += Index(j, u1) * u_2;
    u_index += Index(k, u2) * u_3;
    u_index += Index(l, u3) * u_4;
    u_index += Index(m, u4) * u_5;
    u_index += Index(n, u5) * u6;
    u_index += Index(o, u6);

    output[pos] = static_cast<T>(input_data[l_index] + __half2float(value[0]) * x1[r_index] / x2[u_index]);
  }
}

template <typename VT>
__global__ void Addcdiv_value1(const int64_t l0, const int64_t l1, const int64_t l2, const int64_t l3, const int64_t l4,
                               const int64_t l5, const int64_t l6,

                               const int64_t r0, const int64_t r1, const int64_t r2, const int64_t r3, const int64_t r4,
                               const int64_t r5, const int64_t r6,

                               const int64_t u0, const int64_t u1, const int64_t u2, const int64_t u3, const int64_t u4,
                               const int64_t u5, const int64_t u6,

                               const int64_t d0, const int64_t d1, const int64_t d2, const int64_t d3, const int64_t d4,
                               const int64_t d5, const int64_t d6,

                               const int64_t d_1, const int64_t d_2, const int64_t d_3, const int64_t d_4,
                               const int64_t d_5, const int64_t l_1, const int64_t l_2, const int64_t l_3,
                               const int64_t l_4, const int64_t l_5, const int64_t r_1, const int64_t r_2,
                               const int64_t r_3, const int64_t r_4, const int64_t r_5, const int64_t u_1,
                               const int64_t u_2, const int64_t u_3, const int64_t u_4, const int64_t u_5,

                               const half *input_data, const half *x1, const half *x2, const VT *value, half *output,
                               const int64_t size) {
  for (int64_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    int64_t i = pos / d_1;
    int64_t j = pos / d_2 % d1;
    int64_t k = pos / d_3 % d2;
    int64_t l = pos / d_4 % d3;
    int64_t m = pos / d_5 % d4;
    int64_t n = pos / d6 % d5;
    int64_t o = pos % d6;

    int64_t l_index = Index(i, l0) * l_1;
    l_index += Index(j, l1) * l_2;
    l_index += Index(k, l2) * l_3;
    l_index += Index(l, l3) * l_4;
    l_index += Index(m, l4) * l_5;
    l_index += Index(n, l5) * l6;
    l_index += Index(o, l6);
    int64_t r_index = Index(i, r0) * r_1;
    r_index += Index(j, r1) * r_2;
    r_index += Index(k, r2) * r_3;
    r_index += Index(l, r3) * r_4;
    r_index += Index(m, r4) * r_5;
    r_index += Index(n, r5) * r6;
    r_index += Index(o, r6);
    int64_t u_index = Index(i, u0) * u_1;
    u_index += Index(j, u1) * u_2;
    u_index += Index(k, u2) * u_3;
    u_index += Index(l, u3) * u_4;
    u_index += Index(m, u4) * u_5;
    u_index += Index(n, u5) * u6;
    u_index += Index(o, u6);

    VT v = value[0];
    output[pos] = input_data[l_index] + __float2half(static_cast<float>(v)) * x1[r_index] / x2[u_index];
  }
}

__global__ void Addcdiv_value1(const int64_t l0, const int64_t l1, const int64_t l2, const int64_t l3, const int64_t l4,
                               const int64_t l5, const int64_t l6,

                               const int64_t r0, const int64_t r1, const int64_t r2, const int64_t r3, const int64_t r4,
                               const int64_t r5, const int64_t r6,

                               const int64_t u0, const int64_t u1, const int64_t u2, const int64_t u3, const int64_t u4,
                               const int64_t u5, const int64_t u6,

                               const int64_t d0, const int64_t d1, const int64_t d2, const int64_t d3, const int64_t d4,
                               const int64_t d5, const int64_t d6,

                               const int64_t d_1, const int64_t d_2, const int64_t d_3, const int64_t d_4,
                               const int64_t d_5, const int64_t l_1, const int64_t l_2, const int64_t l_3,
                               const int64_t l_4, const int64_t l_5, const int64_t r_1, const int64_t r_2,
                               const int64_t r_3, const int64_t r_4, const int64_t r_5, const int64_t u_1,
                               const int64_t u_2, const int64_t u_3, const int64_t u_4, const int64_t u_5,

                               const half *input_data, const half *x1, const half *x2, const half *value, half *output,
                               const int64_t size) {
  for (int64_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    int64_t i = pos / d_1;
    int64_t j = pos / d_2 % d1;
    int64_t k = pos / d_3 % d2;
    int64_t l = pos / d_4 % d3;
    int64_t m = pos / d_5 % d4;
    int64_t n = pos / d6 % d5;
    int64_t o = pos % d6;

    int64_t l_index = Index(i, l0) * l_1;
    l_index += Index(j, l1) * l_2;
    l_index += Index(k, l2) * l_3;
    l_index += Index(l, l3) * l_4;
    l_index += Index(m, l4) * l_5;
    l_index += Index(n, l5) * l6;
    l_index += Index(o, l6);
    int64_t r_index = Index(i, r0) * r_1;
    r_index += Index(j, r1) * r_2;
    r_index += Index(k, r2) * r_3;
    r_index += Index(l, r3) * r_4;
    r_index += Index(m, r4) * r_5;
    r_index += Index(n, r5) * r6;
    r_index += Index(o, r6);
    int64_t u_index = Index(i, u0) * u_1;
    u_index += Index(j, u1) * u_2;
    u_index += Index(k, u2) * u_3;
    u_index += Index(l, u3) * u_4;
    u_index += Index(m, u4) * u_5;
    u_index += Index(n, u5) * u6;
    u_index += Index(o, u6);

    half v = value[0];
    output[pos] = input_data[l_index] + v * x1[r_index] / x2[u_index];
  }
}

template <typename T, typename VT>
void CalAddcdiv(const std::vector<int64_t> &input_data_dims, const std::vector<int64_t> &x1_dims,
                const std::vector<int64_t> &x2_dims, const std::vector<int64_t> &value_dims,
                const std::vector<int64_t> &output_dims, const T *input_data, const T *x1, const T *x2, const VT *value,
                T *output, const uint32_t &device_id, cudaStream_t stream) {
  int64_t size = 1;

  int64_t size_value = 1;
  for (int d = 0; d < 7; ++d) {
    size *= output_dims[d];
    size_value *= value_dims[d];
  }
  int64_t output_broadcast_used[5];
  int64_t input_data_broadcast_used[5];
  int64_t x1_broadcast_used[5];
  int64_t x2_broadcast_used[5];
  int64_t value_broadcast_used[5];
  int64_t o = 1;
  int64_t inp = 1;
  int64_t x1_ = 1;
  int64_t x2_ = 1;
  int64_t v = 1;
  for (int64_t i = 0; i < 5; ++i) {
    for (int64_t j = i + 1; j < 7; ++j) {
      o *= output_dims[j];
      inp *= input_data_dims[j];
      x1_ *= x1_dims[j];
      x2_ *= x2_dims[j];
      v *= value_dims[j];
    }
    output_broadcast_used[i] = o;
    input_data_broadcast_used[i] = inp;
    x1_broadcast_used[i] = x1_;
    x2_broadcast_used[i] = x2_;
    value_broadcast_used[i] = v;
    o = 1;
    inp = 1;
    x1_ = 1;
    x2_ = 1;
    v = 1;
  }
  if (input_data_dims == x1_dims && x1_dims == x2_dims) {
    if (x2_dims == value_dims) {
      Addcdiv_all_same<<<CUDA_BLOCKS(device_id, size), CUDA_THREADS(device_id), 0, stream>>>(input_data, x1, x2, value,
                                                                                             output, size);
    } else if (size_value == 1) {
      Addcdiv_all_same_value1<<<CUDA_BLOCKS(device_id, size), CUDA_THREADS(device_id), 0, stream>>>(
        input_data, x1, x2, value, output, size);
    } else {
      Addcdiv<<<CUDA_BLOCKS(device_id, size), CUDA_THREADS(device_id), 0, stream>>>(
        input_data_dims[0], input_data_dims[1], input_data_dims[2], input_data_dims[3], input_data_dims[4],
        input_data_dims[5], input_data_dims[6], x1_dims[0], x1_dims[1], x1_dims[2], x1_dims[3], x1_dims[4], x1_dims[5],
        x1_dims[6], x2_dims[0], x2_dims[1], x2_dims[2], x2_dims[3], x2_dims[4], x2_dims[5], x2_dims[6], value_dims[0],
        value_dims[1], value_dims[2], value_dims[3], value_dims[4], value_dims[5], value_dims[6], output_dims[0],
        output_dims[1], output_dims[2], output_dims[3], output_dims[4], output_dims[5], output_dims[6],
        output_broadcast_used[0], output_broadcast_used[1], output_broadcast_used[2], output_broadcast_used[3],
        output_broadcast_used[4], input_data_broadcast_used[0], input_data_broadcast_used[1],
        input_data_broadcast_used[2], input_data_broadcast_used[3], input_data_broadcast_used[4], x1_broadcast_used[0],
        x1_broadcast_used[1], x1_broadcast_used[2], x1_broadcast_used[3], x1_broadcast_used[4], x2_broadcast_used[0],
        x2_broadcast_used[1], x2_broadcast_used[2], x2_broadcast_used[3], x2_broadcast_used[4], value_broadcast_used[0],
        value_broadcast_used[1], value_broadcast_used[2], value_broadcast_used[3], value_broadcast_used[4], input_data,
        x1, x2, value, output, size);
    }
  } else if (size_value == 1) {
    Addcdiv_value1<<<CUDA_BLOCKS(device_id, size), CUDA_THREADS(device_id), 0, stream>>>(
      input_data_dims[0], input_data_dims[1], input_data_dims[2], input_data_dims[3], input_data_dims[4],
      input_data_dims[5], input_data_dims[6], x1_dims[0], x1_dims[1], x1_dims[2], x1_dims[3], x1_dims[4], x1_dims[5],
      x1_dims[6], x2_dims[0], x2_dims[1], x2_dims[2], x2_dims[3], x2_dims[4], x2_dims[5], x2_dims[6], output_dims[0],
      output_dims[1], output_dims[2], output_dims[3], output_dims[4], output_dims[5], output_dims[6],
      output_broadcast_used[0], output_broadcast_used[1], output_broadcast_used[2], output_broadcast_used[3],
      output_broadcast_used[4], input_data_broadcast_used[0], input_data_broadcast_used[1],
      input_data_broadcast_used[2], input_data_broadcast_used[3], input_data_broadcast_used[4], x1_broadcast_used[0],
      x1_broadcast_used[1], x1_broadcast_used[2], x1_broadcast_used[3], x1_broadcast_used[4], x2_broadcast_used[0],
      x2_broadcast_used[1], x2_broadcast_used[2], x2_broadcast_used[3], x2_broadcast_used[4], input_data, x1, x2, value,
      output, size);
  } else {
    Addcdiv<<<CUDA_BLOCKS(device_id, size), CUDA_THREADS(device_id), 0, stream>>>(
      input_data_dims[0], input_data_dims[1], input_data_dims[2], input_data_dims[3], input_data_dims[4],
      input_data_dims[5], input_data_dims[6], x1_dims[0], x1_dims[1], x1_dims[2], x1_dims[3], x1_dims[4], x1_dims[5],
      x1_dims[6], x2_dims[0], x2_dims[1], x2_dims[2], x2_dims[3], x2_dims[4], x2_dims[5], x2_dims[6], value_dims[0],
      value_dims[1], value_dims[2], value_dims[3], value_dims[4], value_dims[5], value_dims[6], output_dims[0],
      output_dims[1], output_dims[2], output_dims[3], output_dims[4], output_dims[5], output_dims[6],
      output_broadcast_used[0], output_broadcast_used[1], output_broadcast_used[2], output_broadcast_used[3],
      output_broadcast_used[4], input_data_broadcast_used[0], input_data_broadcast_used[1],
      input_data_broadcast_used[2], input_data_broadcast_used[3], input_data_broadcast_used[4], x1_broadcast_used[0],
      x1_broadcast_used[1], x1_broadcast_used[2], x1_broadcast_used[3], x1_broadcast_used[4], x2_broadcast_used[0],
      x2_broadcast_used[1], x2_broadcast_used[2], x2_broadcast_used[3], x2_broadcast_used[4], value_broadcast_used[0],
      value_broadcast_used[1], value_broadcast_used[2], value_broadcast_used[3], value_broadcast_used[4], input_data,
      x1, x2, value, output, size);
  }
}

template CUDA_LIB_EXPORT void CalAddcdiv<half, half>(
  const std::vector<int64_t> &input_data_dims, const std::vector<int64_t> &x1_dims, const std::vector<int64_t> &x2_dims,
  const std::vector<int64_t> &value_dims, const std::vector<int64_t> &output_dims, const half *input_data,
  const half *x1, const half *x2, const half *value, half *output, const uint32_t &device_id, cudaStream_t stream);
template CUDA_LIB_EXPORT void CalAddcdiv<half, double>(
  const std::vector<int64_t> &input_data_dims, const std::vector<int64_t> &x1_dims, const std::vector<int64_t> &x2_dims,
  const std::vector<int64_t> &value_dims, const std::vector<int64_t> &output_dims, const half *input_data,
  const half *x1, const half *x2, const double *value, half *output, const uint32_t &device_id, cudaStream_t stream);
template CUDA_LIB_EXPORT void CalAddcdiv<half, float>(
  const std::vector<int64_t> &input_data_dims, const std::vector<int64_t> &x1_dims, const std::vector<int64_t> &x2_dims,
  const std::vector<int64_t> &value_dims, const std::vector<int64_t> &output_dims, const half *input_data,
  const half *x1, const half *x2, const float *value, half *output, const uint32_t &device_id, cudaStream_t stream);
template CUDA_LIB_EXPORT void CalAddcdiv<half, int>(
  const std::vector<int64_t> &input_data_dims, const std::vector<int64_t> &x1_dims, const std::vector<int64_t> &x2_dims,
  const std::vector<int64_t> &value_dims, const std::vector<int64_t> &output_dims, const half *input_data,
  const half *x1, const half *x2, const int *value, half *output, const uint32_t &device_id, cudaStream_t stream);
template CUDA_LIB_EXPORT void CalAddcdiv<half, int64_t>(
  const std::vector<int64_t> &input_data_dims, const std::vector<int64_t> &x1_dims, const std::vector<int64_t> &x2_dims,
  const std::vector<int64_t> &value_dims, const std::vector<int64_t> &output_dims, const half *input_data,
  const half *x1, const half *x2, const int64_t *value, half *output, const uint32_t &device_id, cudaStream_t stream);
template CUDA_LIB_EXPORT void CalAddcdiv<float, float>(
  const std::vector<int64_t> &input_data_dims, const std::vector<int64_t> &x1_dims, const std::vector<int64_t> &x2_dims,
  const std::vector<int64_t> &value_dims, const std::vector<int64_t> &output_dims, const float *input_data,
  const float *x1, const float *x2, const float *value, float *output, const uint32_t &device_id, cudaStream_t stream);
template CUDA_LIB_EXPORT void CalAddcdiv<float, double>(
  const std::vector<int64_t> &input_data_dims, const std::vector<int64_t> &x1_dims, const std::vector<int64_t> &x2_dims,
  const std::vector<int64_t> &value_dims, const std::vector<int64_t> &output_dims, const float *input_data,
  const float *x1, const float *x2, const double *value, float *output, const uint32_t &device_id, cudaStream_t stream);
template CUDA_LIB_EXPORT void CalAddcdiv<float, half>(
  const std::vector<int64_t> &input_data_dims, const std::vector<int64_t> &x1_dims, const std::vector<int64_t> &x2_dims,
  const std::vector<int64_t> &value_dims, const std::vector<int64_t> &output_dims, const float *input_data,
  const float *x1, const float *x2, const half *value, float *output, const uint32_t &device_id, cudaStream_t stream);
template CUDA_LIB_EXPORT void CalAddcdiv<float, int64_t>(
  const std::vector<int64_t> &input_data_dims, const std::vector<int64_t> &x1_dims, const std::vector<int64_t> &x2_dims,
  const std::vector<int64_t> &value_dims, const std::vector<int64_t> &output_dims, const float *input_data,
  const float *x1, const float *x2, const int64_t *value, float *output, const uint32_t &device_id,
  cudaStream_t stream);
template CUDA_LIB_EXPORT void CalAddcdiv<float, int>(
  const std::vector<int64_t> &input_data_dims, const std::vector<int64_t> &x1_dims, const std::vector<int64_t> &x2_dims,
  const std::vector<int64_t> &value_dims, const std::vector<int64_t> &output_dims, const float *input_data,
  const float *x1, const float *x2, const int *value, float *output, const uint32_t &device_id, cudaStream_t stream);
template CUDA_LIB_EXPORT void CalAddcdiv<double, double>(
  const std::vector<int64_t> &input_data_dims, const std::vector<int64_t> &x1_dims, const std::vector<int64_t> &x2_dims,
  const std::vector<int64_t> &value_dims, const std::vector<int64_t> &output_dims, const double *input_data,
  const double *x1, const double *x2, const double *y, double *output, const uint32_t &device_id, cudaStream_t stream);
template CUDA_LIB_EXPORT void CalAddcdiv<double, float>(
  const std::vector<int64_t> &input_data_dims, const std::vector<int64_t> &x1_dims, const std::vector<int64_t> &x2_dims,
  const std::vector<int64_t> &value_dims, const std::vector<int64_t> &output_dims, const double *input_data,
  const double *x1, const double *x2, const float *y, double *output, const uint32_t &device_id, cudaStream_t stream);
template CUDA_LIB_EXPORT void CalAddcdiv<double, int>(
  const std::vector<int64_t> &input_data_dims, const std::vector<int64_t> &x1_dims, const std::vector<int64_t> &x2_dims,
  const std::vector<int64_t> &value_dims, const std::vector<int64_t> &output_dims, const double *input_data,
  const double *x1, const double *x2, const int *y, double *output, const uint32_t &device_id, cudaStream_t stream);
template CUDA_LIB_EXPORT void CalAddcdiv<double, half>(
  const std::vector<int64_t> &input_data_dims, const std::vector<int64_t> &x1_dims, const std::vector<int64_t> &x2_dims,
  const std::vector<int64_t> &value_dims, const std::vector<int64_t> &output_dims, const double *input_data,
  const double *x1, const double *x2, const half *y, double *output, const uint32_t &device_id, cudaStream_t stream);
template CUDA_LIB_EXPORT void CalAddcdiv<double, int64_t>(
  const std::vector<int64_t> &input_data_dims, const std::vector<int64_t> &x1_dims, const std::vector<int64_t> &x2_dims,
  const std::vector<int64_t> &value_dims, const std::vector<int64_t> &output_dims, const double *input_data,
  const double *x1, const double *x2, const int64_t *y, double *output, const uint32_t &device_id, cudaStream_t stream);
template CUDA_LIB_EXPORT void CalAddcdiv<int64_t, int64_t>(
  const std::vector<int64_t> &input_data_dims, const std::vector<int64_t> &x1_dims, const std::vector<int64_t> &x2_dims,
  const std::vector<int64_t> &value_dims, const std::vector<int64_t> &output_dims, const int64_t *input_data,
  const int64_t *x1, const int64_t *x2, const int64_t *y, int64_t *output, const uint32_t &device_id,
  cudaStream_t stream);
template CUDA_LIB_EXPORT void CalAddcdiv<int64_t, half>(
  const std::vector<int64_t> &input_data_dims, const std::vector<int64_t> &x1_dims, const std::vector<int64_t> &x2_dims,
  const std::vector<int64_t> &value_dims, const std::vector<int64_t> &output_dims, const int64_t *input_data,
  const int64_t *x1, const int64_t *x2, const half *y, int64_t *output, const uint32_t &device_id, cudaStream_t stream);

template CUDA_LIB_EXPORT void CalAddcdiv<int64_t, double>(
  const std::vector<int64_t> &input_data_dims, const std::vector<int64_t> &x1_dims, const std::vector<int64_t> &x2_dims,
  const std::vector<int64_t> &value_dims, const std::vector<int64_t> &output_dims, const int64_t *input_data,
  const int64_t *x1, const int64_t *x2, const double *y, int64_t *output, const uint32_t &device_id,
  cudaStream_t stream);
template CUDA_LIB_EXPORT void CalAddcdiv<int64_t, float>(
  const std::vector<int64_t> &input_data_dims, const std::vector<int64_t> &x1_dims, const std::vector<int64_t> &x2_dims,
  const std::vector<int64_t> &value_dims, const std::vector<int64_t> &output_dims, const int64_t *input_data,
  const int64_t *x1, const int64_t *x2, const float *y, int64_t *output, const uint32_t &device_id,
  cudaStream_t stream);
template CUDA_LIB_EXPORT void CalAddcdiv<int64_t, int>(
  const std::vector<int64_t> &input_data_dims, const std::vector<int64_t> &x1_dims, const std::vector<int64_t> &x2_dims,
  const std::vector<int64_t> &value_dims, const std::vector<int64_t> &output_dims, const int64_t *input_data,
  const int64_t *x1, const int64_t *x2, const int *y, int64_t *output, const uint32_t &device_id, cudaStream_t stream);
