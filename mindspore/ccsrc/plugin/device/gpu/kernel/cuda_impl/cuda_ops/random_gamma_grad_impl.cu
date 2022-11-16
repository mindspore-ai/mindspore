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
#include "random_gamma_grad_impl.cuh"
#include <limits>
#include <algorithm>

template <typename T>
__device__ inline T polevl(const T x, const T A[], size_t len) {
  T result = 0;
  for (size_t i = 0; i <= len; i++) {
    result = result * x + A[i];
  }
  return result;
}

template <typename T>
__device__ inline T Digamma(T input) {
  const T PI = static_cast<T>(3.14159265358979323846);
  const T PSI_10 = static_cast<T>(2.25175258906672110764);

  T output = 0;
  if (input < 0) {
    if (input == trunc(input)) {
      return std::numeric_limits<T>::quiet_NaN();
    }
    output = static_cast<T>(-PI / tan(PI * input));
    input = 1 - input;
  }
  while (input < 10) {
    output -= 1 / input;
    input += 1;
  }
  if (input == 10) {
    return static_cast<T>(output + PSI_10);
  }

  const T A[] = {8.33333333333333333333E-2,  -2.10927960927960927961E-2, 7.57575757575757575758E-3,
                 -4.16666666666666666667E-3, 3.96825396825396825397E-3,  -8.33333333333333333333E-3,
                 8.33333333333333333333E-2};
  T y = 0;
  if (input < 1.0e17) {
    T z = 1.0 / (input * input);
    y = z * polevl(z, A, 6);
  }

  return static_cast<T>(log(input) - 0.5 / input - y + output);
}

template <typename dtype>
__device__ inline dtype IgammaSeries(dtype aSingle, dtype xSingle) {
  dtype enabled = 1, ans = 1, c_muti = 1;
  dtype r_plus_one = aSingle;
  dtype dc_da = 0, dans_da = 0, dlogax_da = 0;
  while (enabled != 0) {
    r_plus_one += 1;
    dc_da = dc_da * (xSingle / r_plus_one) + (-1 * c_muti * xSingle) / (r_plus_one * r_plus_one);
    dans_da += dc_da;
    c_muti *= (xSingle / r_plus_one);
    ans += c_muti;
    enabled = enabled && (fabs(dc_da / dans_da) > std::numeric_limits<dtype>::epsilon());
  }
  dlogax_da = log(xSingle) - Digamma(aSingle + 1);
  return static_cast<dtype>(-(dans_da + ans * dlogax_da) * xSingle / aSingle);
}

template <typename Intype>
__device__ inline Intype IgammacContinuedFraction(Intype aSingle, Intype xSingle) {
  Intype y_plus_one = 1 - aSingle;
  Intype z_plus_two = xSingle + y_plus_one + 1;
  Intype c_plus_one = 0, dpkm2_da = 0, dqkm2_da = 0, dpkm1_da = 0;
  Intype pkm2 = 1;
  Intype qkm2 = xSingle;
  Intype pkm1 = xSingle + 1;
  Intype qkm1 = z_plus_two * xSingle;
  Intype ans = pkm1 / qkm1;
  Intype dqkm1_da = - xSingle;
  Intype dans_da = (dpkm1_da - ans * dqkm1_da) / qkm1;
  for (size_t i = 0; i < 2000; i++) {
    c_plus_one += 1;
    y_plus_one += 1;
    z_plus_two += 2;
    Intype y_muti_c = y_plus_one * c_plus_one;
    Intype pk = pkm1 * z_plus_two - pkm2 * y_muti_c;
    Intype qk = qkm1 * z_plus_two - qkm2 * y_muti_c;
    Intype dpk_da = dpkm1_da * z_plus_two - pkm1 - dpkm2_da * y_muti_c + pkm2 * c_plus_one;
    Intype dqk_da = dqkm1_da * z_plus_two - qkm1 - dqkm2_da * y_muti_c + qkm2 * c_plus_one;
    if (qk != 0) {
      ans = pk / qk;
      Intype dans_da_new = dans_da;
      dans_da = (dpk_da - ans * dqk_da) / qk;
      if (fabs(dans_da - dans_da_new) < std::numeric_limits<Intype>::epsilon()) {
        break;
      }
    }
    pkm2 = pkm1;
    pkm1 = pk;
    qkm2 = qkm1;
    qkm1 = qk;

    dpkm2_da = dpkm1_da;
    dqkm2_da = dqkm1_da;
    dpkm1_da = dpk_da;
    dqkm1_da = dqk_da;
    bool rescale = fabs(pk) > (1 / std::numeric_limits<Intype>::epsilon());
    pkm2 = rescale ? pkm2 * std::numeric_limits<Intype>::epsilon() : pkm2;
    pkm1 = rescale ? pkm1 * std::numeric_limits<Intype>::epsilon() : pkm1;
    qkm2 = rescale ? qkm2 * std::numeric_limits<Intype>::epsilon() : qkm2;
    qkm1 = rescale ? qkm1 * std::numeric_limits<Intype>::epsilon() : qkm1;
    dpkm2_da = rescale ? dpkm2_da * std::numeric_limits<Intype>::epsilon() : dpkm2_da;
    dqkm2_da = rescale ? dqkm2_da * std::numeric_limits<Intype>::epsilon() : dqkm2_da;
    dpkm1_da = rescale ? dpkm1_da * std::numeric_limits<Intype>::epsilon() : dpkm1_da;
    dqkm1_da = rescale ? dqkm1_da * std::numeric_limits<Intype>::epsilon() : dqkm1_da;
  }
  Intype dlogax_da = log(xSingle) - Digamma(aSingle);
  return static_cast<Intype>((dans_da + ans * dlogax_da) * xSingle);
}

template <typename type>
__device__ inline type GammaSingle(type aSingle, type xSingle) {
  type ax = aSingle * log(xSingle) - xSingle - lgamma(aSingle);
  bool is_nonzero = (xSingle < 0) || (aSingle <= 0);
  bool is_nan = isnan(aSingle) || isnan(xSingle);
  bool underflow = ax < -log(std::numeric_limits<type>::max());
  bool x_is_zero = xSingle == 0;
  if (is_nan || is_nonzero || underflow || x_is_zero) {
    return std::numeric_limits<type>::quiet_NaN();
  }
  bool use_igammac = (xSingle > 1) && (xSingle > aSingle);
  type result;
  if (use_igammac) {
    result = IgammacContinuedFraction<type>(aSingle, xSingle);
  } else {
    result = IgammaSeries<type>(aSingle, xSingle);
  }
  return result;
}

template <typename T>
__global__ void RandomGammaGradKernel(const T *alpha, const T *sample, T *output, int elements) {
  for (int pos = blockIdx.x * blockDim.x + threadIdx.x; pos < elements; pos += blockDim.x * gridDim.x) {
    output[pos] = GammaSingle<T>(alpha[pos], sample[pos]);
  }
}

__device__ __forceinline__ size_t Index(const size_t &index, const size_t &dim) { return dim == 1 ? 0 : index; }

template <typename T>
__global__ void BroadcastRandomGammaGradKernel(size_t i0, size_t i1, size_t i2, size_t i3, size_t i4, size_t i5,
                                               size_t i6, size_t j0, size_t j1, size_t j2, size_t j3, size_t j4,
                                               size_t j5, size_t j6, size_t o0, size_t o1, size_t o2, size_t o3,
                                               size_t o4, size_t o5, size_t o6, const T *alpha, const T *sample,
                                               T *output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < o0 * o1 * o2 * o3 * o4 * o5 * o6;
       pos += blockDim.x * gridDim.x) {
    size_t i = pos / (o1 * o2 * o3 * o4 * o5 * o6) % o0;
    size_t j = pos / (o2 * o3 * o4 * o5 * o6) % o1;
    size_t k = pos / (o3 * o4 * o5 * o6) % o2;
    size_t l = pos / (o4 * o5 * o6) % o3;
    size_t m = pos / (o5 * o6) % o4;
    size_t n = pos / o6 % o5;
    size_t o = pos % o6;

    size_t inputx_idx = Index(i, i0) * i1 * i2 * i3 * i4 * i5 * i6;
    inputx_idx += Index(j, i1) * i2 * i3 * i4 * i5 * i6;
    inputx_idx += Index(k, i2) * i3 * i4 * i5 * i6;
    inputx_idx += Index(l, i3) * i4 * i5 * i6;
    inputx_idx += Index(m, i4) * i5 * i6;
    inputx_idx += Index(n, i5) * i6;
    inputx_idx += Index(o, i6);

    size_t inputy_idx = Index(i, j0) * j1 * j2 * j3 * j4 * j5 * j6;
    inputy_idx += Index(j, j1) * j2 * j3 * j4 * j5 * j6;
    inputy_idx += Index(k, j2) * j3 * j4 * j5 * j6;
    inputy_idx += Index(l, j3) * j4 * j5 * j6;
    inputy_idx += Index(m, j4) * j5 * j6;
    inputy_idx += Index(n, j5) * j6;
    inputy_idx += Index(o, j6);
    output[pos] = GammaSingle<T>(alpha[inputx_idx], sample[inputy_idx]);
  }
}

template <typename T>
void CalRandomGammaGrad(const T *alpha, const T *sample, T *output, int elements, const uint32_t &device_id,
                        cudaStream_t cuda_stream) {
  int thread_num = 1024 < elements ? 1024 : elements;
  cudaDeviceProp prop;
  (void)cudaGetDeviceProperties(&prop, device_id);
  int max_blocks = prop.multiProcessorCount;
  int block_num = std::min(static_cast<int>(((elements - 1) / thread_num) + 1), max_blocks);
  RandomGammaGradKernel<<<block_num, thread_num, 0, cuda_stream>>>(alpha, sample,
    output, elements);
}

template <typename T>
void BroadcastRandomGammaGrad(const std::vector<size_t> &alpha_shape, const std::vector<size_t> &sample_shape,
                              const std::vector<size_t> &output_shape, const T *alpha, const T *sample, T *output,
                              const uint32_t &device_id, cudaStream_t cuda_stream) {
  size_t size = 1;
  for (auto d : output_shape) {
    size *= d;
  }
  int thread_num = 1024 < size ? 1024 : size;
  cudaDeviceProp prop;
  (void)cudaGetDeviceProperties(&prop, device_id);
  int max_blocks = prop.multiProcessorCount;
  int block_num = std::min(static_cast<int>(((size - 1) / thread_num) + 1), max_blocks);
  BroadcastRandomGammaGradKernel<<<block_num, thread_num, 0, cuda_stream>>>(
    alpha_shape[0], alpha_shape[1], alpha_shape[2], alpha_shape[3], alpha_shape[4], alpha_shape[5], alpha_shape[6],
    sample_shape[0], sample_shape[1], sample_shape[2], sample_shape[3], sample_shape[4], sample_shape[5],
    sample_shape[6], output_shape[0], output_shape[1], output_shape[2], output_shape[3], output_shape[4],
    output_shape[5], output_shape[6], alpha, sample, output);
}

template CUDA_LIB_EXPORT void CalRandomGammaGrad<double>(const double *alpha, const double *sample, double *output,
                                                         int elements, const uint32_t &device_id,
                                                         cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalRandomGammaGrad<float>(const float *alpha, const float *sample, float *output,
                                                        int elements, const uint32_t &device_id,
                                                        cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void BroadcastRandomGammaGrad<double>(const std::vector<size_t> &, const std::vector<size_t> &,
                                                               const std::vector<size_t> &, const double *,
                                                               const double *, double *, const uint32_t &,
                                                               cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void BroadcastRandomGammaGrad<float>(const std::vector<size_t> &, const std::vector<size_t> &,
                                                              const std::vector<size_t> &, const float *, const float *,
                                                              float *, const uint32_t &, cudaStream_t cuda_stream);
