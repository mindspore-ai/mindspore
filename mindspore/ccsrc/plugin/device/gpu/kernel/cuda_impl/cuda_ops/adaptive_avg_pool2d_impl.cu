/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/adaptive_avg_pool2d_impl.cuh"
#include "include/cuda_fp16.h"

__device__ inline uint start_index(uint a, uint b, uint c) {
  return floorf(__uint2float_rn(a * c) / __uint2float_rn(b));
}

__device__ inline uint end_index(uint a, uint b, uint c) {
  return ceilf(__uint2float_rn((a + 1) * c) / __uint2float_rn(b));
}

template <typename T>
__global__ void AdaptiveAvgPool2DKernel(const uint size, const uint input_height, const uint input_width,
                                        const uint output_height, const uint output_width, T *input_data,
                                        T *output_data) {
  for (uint c = blockIdx.x * blockDim.x + threadIdx.x; c < size; c += gridDim.x * blockDim.x) {
    T *input_ptr = input_data + c * input_height * input_width;
    T *output_ptr = output_data + c * output_height * output_width;

    for (uint oh = 0; oh < output_height; oh++) {
      uint ih0 = start_index(oh, output_height, input_height);
      uint ih1 = end_index(oh, output_height, input_height);
      uint kh = ih1 - ih0;

      for (uint ow = 0; ow < output_width; ow++) {
        uint iw0 = start_index(ow, output_width, input_width);
        uint iw1 = end_index(ow, output_width, input_width);
        uint kw = iw1 - iw0;

        // compute local average
        T sum = 0;
        for (uint ih = ih0; ih < ih1; ih++) {
          for (uint iw = iw0; iw < iw1; iw++) {
            sum += input_ptr[ih * input_width + iw];
          }
        }
        output_ptr[oh * output_width + ow] = sum / kh / kw;
      }
    }
  }
}

template <>
__global__ void AdaptiveAvgPool2DKernel(const uint size, const uint input_height, const uint input_width,
                                        const uint output_height, const uint output_width, float *input_data,
                                        float *output_data) {
  for (uint c = blockIdx.x * blockDim.x + threadIdx.x; c < size; c += gridDim.x * blockDim.x) {
    float *input_ptr = input_data + c * input_height * input_width;
    float *output_ptr = output_data + c * output_height * output_width;

    for (uint oh = 0; oh < output_height; oh++) {
      uint ih0 = start_index(oh, output_height, input_height);
      uint ih1 = end_index(oh, output_height, input_height);
      uint kh = ih1 - ih0;

      for (uint ow = 0; ow < output_width; ow++) {
        uint iw0 = start_index(ow, output_width, input_width);
        uint iw1 = end_index(ow, output_width, input_width);
        uint kw = iw1 - iw0;

        // compute local average
        float sum = 0;
        for (uint ih = ih0; ih < ih1; ih++) {
          for (uint iw = iw0; iw < iw1; iw++) {
            sum += input_ptr[ih * input_width + iw];
          }
        }
        output_ptr[oh * output_width + ow] = sum / __uint2float_rn(kh * kw);
      }
    }
  }
}

template <>
__global__ void AdaptiveAvgPool2DKernel(const uint size, const uint input_height, const uint input_width,
                                        const uint output_height, const uint output_width, half *input_data,
                                        half *output_data) {
  for (uint c = blockIdx.x * blockDim.x + threadIdx.x; c < size; c += gridDim.x * blockDim.x) {
    half *input_ptr = input_data + c * input_height * input_width;
    half *output_ptr = output_data + c * output_height * output_width;

    for (uint oh = 0; oh < output_height; oh++) {
      uint ih0 = start_index(oh, output_height, input_height);
      uint ih1 = end_index(oh, output_height, input_height);
      uint kh = ih1 - ih0;

      for (uint ow = 0; ow < output_width; ow++) {
        uint iw0 = start_index(ow, output_width, input_width);
        uint iw1 = end_index(ow, output_width, input_width);
        uint kw = iw1 - iw0;

        // compute local average
        half sum = 0;
        for (uint ih = ih0; ih < ih1; ih++) {
          for (uint iw = iw0; iw < iw1; iw++) {
            sum += input_ptr[ih * input_width + iw];
          }
        }
        output_ptr[oh * output_width + ow] = sum / __uint2half_rn(kh * kw);
      }
    }
  }
}

template <>
__global__ void AdaptiveAvgPool2DKernel(const uint size, const uint input_height, const uint input_width,
                                        const uint output_height, const uint output_width, double *input_data,
                                        double *output_data) {
  for (uint c = blockIdx.x * blockDim.x + threadIdx.x; c < size; c += gridDim.x * blockDim.x) {
    double *input_ptr = input_data + c * input_height * input_width;
    double *output_ptr = output_data + c * output_height * output_width;

    for (uint oh = 0; oh < output_height; oh++) {
      uint ih0 = start_index(oh, output_height, input_height);
      uint ih1 = end_index(oh, output_height, input_height);
      uint kh = ih1 - ih0;

      for (uint ow = 0; ow < output_width; ow++) {
        uint iw0 = start_index(ow, output_width, input_width);
        uint iw1 = end_index(ow, output_width, input_width);
        uint kw = iw1 - iw0;

        // compute local average
        double sum = 0;
        for (uint ih = ih0; ih < ih1; ih++) {
          for (uint iw = iw0; iw < iw1; iw++) {
            sum += input_ptr[ih * input_width + iw];
          }
        }
        output_ptr[oh * output_width + ow] = sum / __uint2double_rn(kh * kw);
      }
    }
  }
}

template <typename T>
cudaError_t ApplyAdaptiveAvgPool2D(const uint size, const uint input_height, const uint input_width,
                                   const uint output_height, const uint output_width, T *input_data, T *output_data,
                                   cudaStream_t cuda_stream) {
  AdaptiveAvgPool2DKernel<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(
    size, input_height, input_width, output_height, output_width, input_data, output_data);
  CHECK_CUDA_LAUNCH_SUCCESS();
}

template CUDA_LIB_EXPORT cudaError_t ApplyAdaptiveAvgPool2D<float>(const uint size, const uint input_height,
                                                                   const uint input_width, const uint output_height,
                                                                   const uint output_width, float *input_data,
                                                                   float *output_data, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t ApplyAdaptiveAvgPool2D<half>(const uint size, const uint input_height,
                                                                  const uint input_width, const uint output_height,
                                                                  const uint output_width, half *input_data,
                                                                  half *output_data, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t ApplyAdaptiveAvgPool2D<double>(const uint size, const uint input_height,
                                                                    const uint input_width, const uint output_height,
                                                                    const uint output_width, double *input_data,
                                                                    double *output_data, cudaStream_t cuda_stream);
