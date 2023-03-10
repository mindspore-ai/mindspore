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

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/adaptive_avg_pool2d_grad_impl.cuh"
#include "include/cuda_fp16.h"

__device__ inline uint start_index(uint a, uint b, uint c) {
  return floorf(__uint2float_rn(a * c) / __uint2float_rn(b));
}

__device__ inline uint end_index(uint a, uint b, uint c) {
  return ceilf(__uint2float_rn((a + 1) * c) / __uint2float_rn(b));
}

template <typename T>
__global__ void AdaptiveAvgPool2DGradKernel(const uint size, const uint input_height, const uint input_width,
                                            const uint output_height, const uint output_width, T *input_data,
                                            T *output_data, float *) {
  for (uint c = blockIdx.x * blockDim.x + threadIdx.x; c < size; c += gridDim.x * blockDim.x) {
    T *input_ptr = input_data + c * input_height * input_width;
    T *output_ptr = output_data + c * output_height * output_width;

    size_t output_size = output_height * output_width;
    for (size_t i = 0; i < output_size; i++) {
      output_ptr[i] = 0;
    }

    for (uint ih = 0; ih < input_height; ih++) {
      uint oh0 = start_index(ih, input_height, output_height);
      uint oh1 = end_index(ih, input_height, output_height);
      uint kh = oh1 - oh0;

      for (uint iw = 0; iw < input_width; iw++) {
        uint ow0 = start_index(iw, input_width, output_width);
        uint ow1 = end_index(iw, input_width, output_width);
        uint kw = ow1 - ow0;

        T delta = input_ptr[ih * input_width + iw] / (kh * kw);
        for (uint oh = oh0; oh < oh1; oh++) {
          for (uint ow = ow0; ow < ow1; ow++) {
            output_ptr[oh * output_width + ow] += delta;
          }
        }
      }
    }
  }
}

template <>
__global__ void AdaptiveAvgPool2DGradKernel(const uint size, const uint input_height, const uint input_width,
                                            const uint output_height, const uint output_width, float *input_data,
                                            float *output_data, float *) {
  for (uint c = blockIdx.x * blockDim.x + threadIdx.x; c < size; c += gridDim.x * blockDim.x) {
    float *input_ptr = input_data + c * input_height * input_width;
    float *output_ptr = output_data + c * output_height * output_width;

    size_t output_size = output_height * output_width;
    for (size_t i = 0; i < output_size; i++) {
      output_ptr[i] = 0;
    }

    for (uint ih = 0; ih < input_height; ih++) {
      uint oh0 = start_index(ih, input_height, output_height);
      uint oh1 = end_index(ih, input_height, output_height);
      uint kh = oh1 - oh0;

      for (uint iw = 0; iw < input_width; iw++) {
        uint ow0 = start_index(iw, input_width, output_width);
        uint ow1 = end_index(iw, input_width, output_width);
        uint kw = ow1 - ow0;

        float delta = input_ptr[ih * input_width + iw] / __uint2float_rn(kh * kw);
        for (uint oh = oh0; oh < oh1; oh++) {
          for (uint ow = ow0; ow < ow1; ow++) {
            output_ptr[oh * output_width + ow] += delta;
          }
        }
      }
    }
  }
}

template <>
__global__ void AdaptiveAvgPool2DGradKernel(const uint size, const uint input_height, const uint input_width,
                                            const uint output_height, const uint output_width, half *input_data,
                                            half *output_data, float *workspace) {
  for (uint c = blockIdx.x * blockDim.x + threadIdx.x; c < size; c += gridDim.x * blockDim.x) {
    half *input_ptr = input_data + c * input_height * input_width;
    half *output_ptr = output_data + c * output_height * output_width;
    float *workspace_ptr = workspace + c * output_height * output_width;

    size_t output_size = output_height * output_width;
    for (size_t i = 0; i < output_size; i++) {
      output_ptr[i] = 0;
      workspace_ptr[i] = 0;
    }

    for (uint ih = 0; ih < input_height; ih++) {
      uint oh0 = start_index(ih, input_height, output_height);
      uint oh1 = end_index(ih, input_height, output_height);
      uint kh = oh1 - oh0;

      for (uint iw = 0; iw < input_width; iw++) {
        uint ow0 = start_index(iw, input_width, output_width);
        uint ow1 = end_index(iw, input_width, output_width);
        uint kw = ow1 - ow0;

        half delta = input_ptr[ih * input_width + iw] / __uint2half_rn(kh * kw);
        for (uint oh = oh0; oh < oh1; oh++) {
          for (uint ow = ow0; ow < ow1; ow++) {
            // avoid accumulating out of range of fp16
            workspace_ptr[oh * output_width + ow] += static_cast<float>(delta);
          }
        }
      }
    }
    for (size_t i = 0; i < output_size; i++) {
      output_ptr[i] = static_cast<half>(workspace_ptr[i]);
    }
  }
}

template <>
__global__ void AdaptiveAvgPool2DGradKernel(const uint size, const uint input_height, const uint input_width,
                                            const uint output_height, const uint output_width, double *input_data,
                                            double *output_data, float *) {
  for (uint c = blockIdx.x * blockDim.x + threadIdx.x; c < size; c += gridDim.x * blockDim.x) {
    double *input_ptr = input_data + c * input_height * input_width;
    double *output_ptr = output_data + c * output_height * output_width;

    size_t output_size = output_height * output_width;
    for (size_t i = 0; i < output_size; i++) {
      output_ptr[i] = 0;
    }

    for (uint ih = 0; ih < input_height; ih++) {
      uint oh0 = start_index(ih, input_height, output_height);
      uint oh1 = end_index(ih, input_height, output_height);
      uint kh = oh1 - oh0;

      for (uint iw = 0; iw < input_width; iw++) {
        uint ow0 = start_index(iw, input_width, output_width);
        uint ow1 = end_index(iw, input_width, output_width);
        uint kw = ow1 - ow0;

        double delta = input_ptr[ih * input_width + iw] / __uint2double_rn(kh * kw);
        for (uint oh = oh0; oh < oh1; oh++) {
          for (uint ow = ow0; ow < ow1; ow++) {
            output_ptr[oh * output_width + ow] += delta;
          }
        }
      }
    }
  }
}

template <typename T>
cudaError_t ApplyAdaptiveAvgPool2DGrad(const uint size, const uint input_height, const uint input_width,
                                       const uint output_height, const uint output_width, T *input_data, T *output_data,
                                       float *workspace, cudaStream_t cuda_stream) {
  AdaptiveAvgPool2DGradKernel<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(
    size, input_height, input_width, output_height, output_width, input_data, output_data, workspace);
  CHECK_CUDA_LAUNCH_SUCCESS();
}

template CUDA_LIB_EXPORT cudaError_t ApplyAdaptiveAvgPool2DGrad<float>(const uint size, const uint input_height,
                                                                       const uint input_width, const uint output_height,
                                                                       const uint output_width, float *input_data,
                                                                       float *output_data, float *workspace,
                                                                       cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t ApplyAdaptiveAvgPool2DGrad<half>(const uint size, const uint input_height,
                                                                      const uint input_width, const uint output_height,
                                                                      const uint output_width, half *input_data,
                                                                      half *output_data, float *workspace,
                                                                      cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t ApplyAdaptiveAvgPool2DGrad<double>(
  const uint size, const uint input_height, const uint input_width, const uint output_height, const uint output_width,
  double *input_data, double *output_data, float *workspace, cudaStream_t cuda_stream);
