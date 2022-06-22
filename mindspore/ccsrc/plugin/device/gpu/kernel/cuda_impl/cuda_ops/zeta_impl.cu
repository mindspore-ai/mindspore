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

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/zeta_impl.cuh"
#include <limits>
#include "include/cuda_runtime.h"
#include "include/cuda_fp16.h"

template <typename T>
__global__ void ZetaKernel(const size_t size, const T *x, const T *dimension, T *output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    double p = static_cast<double>(x[pos]);
    double q = static_cast<double>(dimension[pos]);
    const double MACHEP = static_cast<double>(1.11022302462515654042E-16);
    constexpr double zero = static_cast<double>(0.0);
    constexpr double half = static_cast<double>(0.5);
    constexpr double one = static_cast<double>(1.0);
    static const double A[] = {
      12.0,
      -720.0,
      30240.0,
      -1209600.0,
      47900160.0,
      -1.8924375803183791606e9, /*1.307674368e12/691*/
      7.47242496e10,
      -2.950130727918164224e12, /*1.067062284288e16/3617*/
      1.1646782814350067249e14, /*5.109094217170944e18/43867*/
      -4.5979787224074726105e15, /*8.028576626982912e20/174611*/
      1.8152105401943546773e17, /*1.5511210043330985984e23/854513*/
      -7.1661652561756670113e18 /*1.6938241367317436694528e27/236364091*/
    };
    int i = 0;
    double a, b, k, s, t, w;
    bool flag = false;
    if (p == one) {
      output[pos] = std::numeric_limits<T>::infinity();
      continue;
    }
    if (p < one) {
      output[pos] = std::numeric_limits<T>::quiet_NaN();
      continue;
    }
    if (q <= zero) {
      if (q == std::floor(q)) {
        output[pos] = std::numeric_limits<T>::infinity();
        continue;
      }
      if (p != std::floor(p)) {
        output[pos] = std::numeric_limits<T>::quiet_NaN();
        continue;
      }
    }
    s = std::pow(q, -p);
    a = q;
    i = 0;
    b = zero;
    while ((i < 9) || (a <= T(9.0))) {
      i += 1;
      a += one;
      b = std::pow(a, -p);
      s += b;
      if ((-MACHEP * s < b) && (b < MACHEP * s)) {
        output[pos] = static_cast<T>(s);
        flag = true;
        break;
      }
    }
    if (flag) {
      continue;
    }
    w = a;
    s += b * w / (p - one);
    s -= half * b;
    a = one;
    k = zero;
    for (int i = 0; i < 12; i++) {
      a *= p + k;
      b /= w;
      t = a * b / A[i];
      s = s + t;
      t = std::fabs(t / s);
      if (t < MACHEP) {
        output[pos] = static_cast<T>(s);
        break;
      }
      k += one;
      a *= p + k;
      b /= w;
      k += one;
    }
    output[pos] = static_cast<T>(s);
  }
  return;
}

template <typename T>
void CalZeta(const size_t size, const T *x, const T *dimension, T *output, const uint32_t &device_id,
             cudaStream_t cuda_stream) {
  ZetaKernel<<<CUDA_BLOCKS(device_id, size), CUDA_THREADS(device_id), 0, cuda_stream>>>(size, x, dimension, output);
}

template CUDA_LIB_EXPORT void CalZeta<float>(const size_t size, const float *x, const float *dimension, float *output,
                                             const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalZeta<double>(const size_t size, const double *x, const double *dimension,
                                              double *output, const uint32_t &device_id, cudaStream_t cuda_stream);
