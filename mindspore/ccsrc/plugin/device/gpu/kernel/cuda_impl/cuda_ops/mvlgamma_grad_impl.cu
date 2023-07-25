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

#include <math.h>
#include <limits>
#include <algorithm>
#include "mvlgamma_grad_impl.cuh"
#define PI 3.141592653589793

__constant__ double kLanczosCoefficientsd[8] = {676.520368121885098567009190444019, -1259.13921672240287047156078755283,
                                                771.3234287776530788486528258894,   -176.61502916214059906584551354,
                                                12.507343278686904814458936853,     -0.13857109526572011689554707,
                                                9.984369578019570859563e-6,         1.50563273514931155834e-7};
template <typename T>
__device__ __forceinline__ T CalNumDivDenom(T x) {
  T num = 0;
  T denom = 0.99999999999980993227684700473478;
  for (int j = 0; j < 8; ++j) {
    num -= kLanczosCoefficientsd[j] / ((x + j + 1) * (x + j + 1));
    denom += kLanczosCoefficientsd[j] / (x + j + 1);
  }
  return num / denom;
}
template <typename T>
__global__ void MvlgammaGrad(const size_t size, const T *y_grad, const T *x, const int p, T *output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    T kLanczosGamma = 7;
    T log_lanczos_gamma_plus_one_half = log(7.5);
    T temp = 0;
    T cur_input = 0;
    T num_div_denom = 0;
    for (int i = 0; i < p; i++) {
      cur_input = x[pos] - 0.5 * i;
      if (cur_input < 0 && cur_input == floor(cur_input)) {
        temp += std::numeric_limits<T>::quiet_NaN();
        break;
      }
      if (cur_input < 0.5) {
        num_div_denom = CalNumDivDenom(-cur_input);
        temp += (log_lanczos_gamma_plus_one_half + log1pf((-cur_input) / (kLanczosGamma + 0.5))) + num_div_denom -
                kLanczosGamma / (kLanczosGamma + 0.5 - cur_input);
        temp -= PI / tan(PI * (cur_input + abs(floor(cur_input + 0.5))));
      } else {
        num_div_denom = CalNumDivDenom(cur_input - 1);
        temp += (log_lanczos_gamma_plus_one_half + log1pf((cur_input - 1) / (kLanczosGamma + 0.5))) + num_div_denom -
                kLanczosGamma / (kLanczosGamma + 0.5 + cur_input - 1);
      }
    }
    output[pos] = temp * y_grad[pos];
  }
}

template <typename T>
cudaError_t CalMvlgammaGrad(const size_t size, const T *y_grad, const T *x, const int p, T *output,
                            const uint32_t &device_id, cudaStream_t cuda_stream) {
  int thread_num = 256 < size ? 256 : size;
  cudaDeviceProp prop;
  (void)cudaGetDeviceProperties(&prop, device_id);
  int max_blocks = prop.multiProcessorCount;
  int block_num = std::min(static_cast<int>(((size - 1) / thread_num) + 1), max_blocks);
  MvlgammaGrad<<<block_num, thread_num, 0, cuda_stream>>>(size, y_grad, x, p, output);
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t CalMvlgammaGrad<float>(const size_t size, const float *y_grad, const float *x,
                                                            const int p, float *output, const uint32_t &device_id,
                                                            cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalMvlgammaGrad<double>(const size_t size, const double *y_grad, const double *x,
                                                             const int p, double *output, const uint32_t &device_id,
                                                             cudaStream_t cuda_stream);
