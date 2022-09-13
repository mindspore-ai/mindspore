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

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/fractional_max_pool_grad_with_fixed_ksize_impl.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/util.cuh"

template <typename T, typename S>
__global__ void Fractionalmaxpoolgradwithfixedksize(const T *origin_input, const T *out_backprop, S *argmax,
                                                      T *output, int64_t outputH, int64_t outputW,
                                                      int64_t N, int64_t C, int64_t inputH,
                                                      int64_t inputW, const int64_t out_backprop_size) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < out_backprop_size; pos += blockDim.x * gridDim.x) {
    const int posn = pos / (C * outputH * outputW);
    const int posc = pos / (outputH * outputW) % C;

    S maxind = argmax[pos];
    MsAtomicAdd(output + (posn * C + posc) * inputH * inputW + maxind, out_backprop[pos]);

    return;
  }
}

template <typename T, typename S>
void CalFractionalmaxpoolgradwithfixedksize(const T *origin_input, const T *out_backprop, S *argmax, T *output,
                                            int64_t outputH, int64_t outputW, int64_t inputN,
                                            int64_t inputC,  int64_t inputH, int64_t inputW,
                                            const int64_t outer_size, const int64_t out_backprop_size,
                                            const uint32_t &device_id, cudaStream_t cuda_stream) {
  Fractionalmaxpoolgradwithfixedksize<<<CUDA_BLOCKS(device_id, out_backprop_size), CUDA_THREADS(device_id), 0,
                                          cuda_stream>>>(origin_input, out_backprop, argmax, output, outputH,
                                                         outputW, inputN, inputC, inputH, inputW,
                                                         out_backprop_size);
  return;
}

template CUDA_LIB_EXPORT void CalFractionalmaxpoolgradwithfixedksize<half, int32_t>(
  const half *origin_input, const half *out_backprop, int32_t *argmax, half *output,  int64_t outputH,
  int64_t outputW, int64_t inputN, int64_t inputC, int64_t inputH, int64_t inputW,
  const int64_t outer_size, const int64_t out_backprop_size, const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalFractionalmaxpoolgradwithfixedksize<float, int32_t>(
  const float *origin_input, const float *out_backprop, int32_t *argmax, float *output,
  int64_t outputH, int64_t outputW, int64_t inputN, int64_t inputC, int64_t inputH, int64_t inputW,
  const int64_t outer_size, const int64_t out_backprop_size, const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalFractionalmaxpoolgradwithfixedksize<double, int32_t>(
  const double *origin_input, const double *out_backprop, int32_t *argmax, double *output,
  int64_t outputH, int64_t outputW, int64_t inputN, int64_t inputC, int64_t inputH, int64_t inputW,
  const int64_t outer_size, const int64_t out_backprop_size, const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalFractionalmaxpoolgradwithfixedksize<int32_t, int32_t>(
  const int32_t *origin_input, const int32_t *out_backprop, int32_t *argmax, int32_t *output,
  int64_t outputH, int64_t outputW, int64_t inputN, int64_t inputC, int64_t inputH, int64_t inputW,
  const int64_t outer_size, const int64_t out_backprop_size, const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalFractionalmaxpoolgradwithfixedksize<int64_t, int32_t>(
  const int64_t *origin_input, const int64_t *out_backprop, int32_t *argmax, int64_t *output,
  int64_t outputH, int64_t outputW, int64_t inputN, int64_t inputC, int64_t inputH, int64_t inputW,
  const int64_t outer_size, const int64_t out_backprop_size, const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalFractionalmaxpoolgradwithfixedksize<half, int64_t>(
  const half *origin_input, const half *out_backprop, int64_t *argmax, half *output,  int64_t outputH,
  int64_t outputW, int64_t inputN, int64_t inputC, int64_t inputH, int64_t inputW,
  const int64_t outer_size, const int64_t out_backprop_size, const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalFractionalmaxpoolgradwithfixedksize<float, int64_t>(
  const float *origin_input, const float *out_backprop, int64_t *argmax, float *output,
  int64_t outputH, int64_t outputW, int64_t inputN, int64_t inputC, int64_t inputH, int64_t inputW,
  const int64_t outer_size, const int64_t out_backprop_size, const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalFractionalmaxpoolgradwithfixedksize<double, int64_t>(
  const double *origin_input, const double *out_backprop, int64_t *argmax, double *output,
  int64_t outputH, int64_t outputW, int64_t inputN, int64_t inputC, int64_t inputH, int64_t inputW,
  const int64_t outer_size, const int64_t out_backprop_size, const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalFractionalmaxpoolgradwithfixedksize<int32_t, int64_t>(
  const int32_t *origin_input, const int32_t *out_backprop, int64_t *argmax, int32_t *output,
  int64_t outputH, int64_t outputW, int64_t inputN, int64_t inputC, int64_t inputH, int64_t inputW,
  const int64_t outer_size, const int64_t out_backprop_size, const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalFractionalmaxpoolgradwithfixedksize<int64_t, int64_t>(
  const int64_t *origin_input, const int64_t *out_backprop, int64_t *argmax, int64_t *output,
  int64_t outputH, int64_t outputW, int64_t inputN, int64_t inputC, int64_t inputH, int64_t inputW,
  const int64_t outer_size, const int64_t out_backprop_size, const uint32_t &device_id, cudaStream_t cuda_stream);
