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

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/adaptive_avg_pool3d_grad_impl.cuh"
#include "include/cuda_fp16.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/util.cuh"

template <typename T>
__global__ void InitOutput(T *output_ptr, const uint out_size) {
  T zero = 0;
  for (size_t id = blockIdx.x * blockDim.x + threadIdx.x; id < out_size; id += blockDim.x * gridDim.x) {
    output_ptr[id] = zero;
  }
  return;
}

template <typename T>
__global__ void AdaptiveAvgPool3DGradKernel(const uint in_size, const uint out_size, const uint input_channel,
                                            const uint input_height, const uint input_width, const uint input_depth,
                                            const uint output_channel, const uint output_height,
                                            const uint output_width, const uint output_depth, T *input_data,
                                            T *output_data) {
  for (uint pos = blockIdx.x * blockDim.x + threadIdx.x; pos < in_size; pos += gridDim.x * blockDim.x) {
    const uint in = pos / (input_channel * input_height * input_width * input_depth);
    const uint ic = pos / (input_height * input_width * input_depth) % input_channel;
    const uint ih = pos / (input_width * input_depth) % input_height;
    const uint iw = pos / input_depth % input_width;
    const uint id = pos % input_depth;
    const uint on = in;
    const uint oc = ic;

    uint oh0 = floorf(__uint2float_rn(ih * output_height) / __uint2float_rn(input_height));
    uint oh1 = ceilf(__uint2float_rn((ih + 1) * output_height) / __uint2float_rn(input_height));
    uint kh = oh1 - oh0;

    uint ow0 = floorf(__uint2float_rn(iw * output_width) / __uint2float_rn(input_width));
    uint ow1 = ceilf(__uint2float_rn((iw + 1) * output_width) / __uint2float_rn(input_width));
    uint kw = ow1 - ow0;

    uint od0 = floorf(__uint2float_rn(id * output_depth) / __uint2float_rn(input_depth));
    uint od1 = ceilf(__uint2float_rn((id + 1) * output_depth) / __uint2float_rn(input_depth));
    uint kd = od1 - od0;

    uint in_index = (((in * input_channel + ic) * input_height + ih) * input_width + iw) * input_depth + id;
    uint out_index = 0;
    for (uint oh = oh0; oh < oh1; oh++) {
      for (uint ow = ow0; ow < ow1; ow++) {
        for (uint od = od0; od < od1; od++) {
          out_index = (((on * output_channel + oc) * output_height + oh) * output_width + ow) * output_depth + od;
          MsAtomicAdd(output_data + out_index, input_data[in_index] / static_cast<T>(kh * kw * kd));
        }
      }
    }
  }
}

template <typename T>
cudaError_t ApplyAdaptiveAvgPool3DGrad(const uint in_size, const uint out_size, const uint input_channel,
                                       const uint input_height, const uint input_width, const uint input_depth,
                                       const uint output_channel, const uint output_height, const uint output_width,
                                       const uint output_depth, T *input_data, T *output_data,
                                       cudaStream_t cuda_stream) {
  InitOutput<<<GET_BLOCKS(out_size), GET_THREADS, 0, cuda_stream>>>(output_data, out_size);
  AdaptiveAvgPool3DGradKernel<<<GET_BLOCKS(in_size), GET_THREADS, 0, cuda_stream>>>(
    in_size, out_size, input_channel, input_height, input_width, input_depth, output_channel, output_height,
    output_width, output_depth, input_data, output_data);
  CHECK_CUDA_LAUNCH_SUCCESS();
}

template CUDA_LIB_EXPORT cudaError_t ApplyAdaptiveAvgPool3DGrad<float>(
  const uint in_size, const uint out_size, const uint input_channel, const uint input_height, const uint input_width,
  const uint input_depth, const uint output_channel, const uint output_height, const uint output_width,
  const uint output_depth, float *input_data, float *output_data, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t ApplyAdaptiveAvgPool3DGrad<half>(
  const uint in_size, const uint out_size, const uint input_channel, const uint input_height, const uint input_width,
  const uint input_depth, const uint output_channel, const uint output_height, const uint output_width,
  const uint output_depth, half *input_data, half *output_data, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t ApplyAdaptiveAvgPool3DGrad<double>(
  const uint in_size, const uint out_size, const uint input_channel, const uint input_height, const uint input_width,
  const uint input_depth, const uint output_channel, const uint output_height, const uint output_width,
  const uint output_depth, double *input_data, double *output_data, cudaStream_t cuda_stream);
