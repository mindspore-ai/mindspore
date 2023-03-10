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

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/adaptive_max_pool2d_impl.cuh"
#include "include/cuda_fp16.h"

__device__ inline size_t start_index(size_t a, size_t b, size_t c) {
  return floorf(__uint2float_rn(a * c) / __uint2float_rn(b));
}

__device__ inline size_t end_index(size_t a, size_t b, size_t c) {
  return ceilf(__uint2float_rn((a + 1) * c) / __uint2float_rn(b));
}

template <typename T>
__device__ __forceinline__ void IsNan(const T *val, bool *out) {
  *out = isnan(*val);
  return;
}
template <>
__device__ __forceinline__ void IsNan(const half *val, bool *out) {
  *out = __hisnan(*val);
  return;
}

template <typename T>
__global__ void AdaptiveMaxPool2DKernel(const size_t size, const size_t input_height, const size_t input_width,
                                        const size_t output_height, const size_t output_width, T *input_data,
                                        T *output_data, int64_t *indices_data) {
  T *input_ptr = input_data + blockIdx.x * input_height * input_width;
  T *output_ptr = output_data + blockIdx.x * output_height * output_width;
  int64_t *indices_ptr = indices_data + blockIdx.x * output_height * output_width;

  for (size_t oh = blockDim.y * blockIdx.y + threadIdx.y; oh < output_height; oh += blockDim.y * gridDim.y) {
    size_t ih0 = start_index(oh, output_height, input_height);
    size_t ih1 = end_index(oh, output_height, input_height);

    for (size_t ow = threadIdx.x; ow < output_width; ow += blockDim.x) {
      size_t iw0 = start_index(ow, output_width, input_width);
      size_t iw1 = end_index(ow, output_width, input_width);

      T *sub_input_ptr = input_ptr + ih0 * input_width + iw0;
      int64_t indice = ih0 * input_width + iw0;
      T max = sub_input_ptr[0];

      for (size_t ih = 0; ih < ih1 - ih0; ih++) {
        for (size_t iw = 0; iw < iw1 - iw0; iw++) {
          T val = sub_input_ptr[iw];
          bool is_nan = false;
          IsNan(&val, &is_nan);
          if ((val > max) || is_nan) {
            max = val;
            indice = (ih + ih0) * input_width + iw + iw0;
          }
        }
        sub_input_ptr += input_width;
      }
      output_ptr[oh * output_width + ow] = max;
      indices_ptr[oh * output_width + ow] = indice;
    }
  }
}

template <typename T>
cudaError_t ApplyAdaptiveMaxPool2D(const size_t size, const size_t input_height, const size_t input_width,
                                   const size_t output_height, const size_t output_width, T *input_data, T *output_data,
                                   int64_t *indices_data, cudaStream_t cuda_stream) {
  int blocksH = size > 16 ? 1 : (16L / size);
  dim3 blocks(size, blocksH);
  dim3 threads(32, 8);
  AdaptiveMaxPool2DKernel<<<blocks, threads, 0, cuda_stream>>>(size, input_height, input_width, output_height,
                                                               output_width, input_data, output_data, indices_data);
  CHECK_CUDA_LAUNCH_SUCCESS();
}

template CUDA_LIB_EXPORT cudaError_t ApplyAdaptiveMaxPool2D<float>(const size_t size, const size_t input_height,
                                                                   const size_t input_width, const size_t output_height,
                                                                   const size_t output_width, float *input_data,
                                                                   float *output_data, int64_t *indices_data,
                                                                   cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t ApplyAdaptiveMaxPool2D<half>(const size_t size, const size_t input_height,
                                                                  const size_t input_width, const size_t output_height,
                                                                  const size_t output_width, half *input_data,
                                                                  half *output_data, int64_t *indices_data,
                                                                  cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t ApplyAdaptiveMaxPool2D<double>(
  const size_t size, const size_t input_height, const size_t input_width, const size_t output_height,
  const size_t output_width, double *input_data, double *output_data, int64_t *indices_data, cudaStream_t cuda_stream);
