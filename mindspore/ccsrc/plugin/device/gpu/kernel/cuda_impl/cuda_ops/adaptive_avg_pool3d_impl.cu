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

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/adaptive_avg_pool3d_impl.cuh"
#include "include/cuda_fp16.h"

template <typename T>
__global__ void AdaptiveAvgPool3DKernel(const uint out_size, const uint input_channel, const uint input_height,
                                        const uint input_width, const uint input_depth, const uint output_channel,
                                        const uint output_height, const uint output_width, const uint output_depth,
                                        T *input_data, T *output_data) {
  for (uint pos = blockIdx.x * blockDim.x + threadIdx.x; pos < out_size; pos += gridDim.x * blockDim.x) {
    const uint on = pos / (output_channel * output_height * output_width * output_depth);
    const uint oc = pos / (output_height * output_width * output_depth) % output_channel;
    const uint oh = pos / (output_width * output_depth) % output_height;
    const uint ow = pos / output_depth % output_width;
    const uint od = pos % output_depth;
    const uint in = on;
    const uint ic = oc;

    uint ih0 = floorf(__uint2float_rn(oh * input_height) / __uint2float_rn(output_height));
    uint ih1 = ceilf(__uint2float_rn((oh + 1) * input_height) / __uint2float_rn(output_height));
    uint kh = ih1 - ih0;

    uint iw0 = floorf(__uint2float_rn(ow * input_width) / __uint2float_rn(output_width));
    uint iw1 = ceilf(__uint2float_rn((ow + 1) * input_width) / __uint2float_rn(output_width));
    uint kw = iw1 - iw0;

    uint id0 = floorf(__uint2float_rn(od * input_depth) / __uint2float_rn(output_depth));
    uint id1 = ceilf(__uint2float_rn((od + 1) * input_depth) / __uint2float_rn(output_depth));
    uint kd = id1 - id0;

    T sum = 0;
    uint in_index = 0;
    for (uint ih = ih0; ih < ih1; ih++) {
      for (uint iw = iw0; iw < iw1; iw++) {
        for (uint id = id0; id < id1; id++) {
          in_index = (((in * input_channel + ic) * input_height + ih) * input_width + iw) * input_depth + id;
          sum += input_data[in_index];
        }
      }
    }
    uint out_index = (((on * output_channel + oc) * output_height + oh) * output_width + ow) * output_depth + od;
    output_data[out_index] = sum / static_cast<T>(kh * kw * kd);
  }
}

template <typename T>
cudaError_t ApplyAdaptiveAvgPool3D(const uint out_size, const uint input_channel, const uint input_height,
                                   const uint input_width, const uint input_depth, const uint output_channel,
                                   const uint output_height, const uint output_width, const uint output_depth,
                                   T *input_data, T *output_data, cudaStream_t cuda_stream) {
  AdaptiveAvgPool3DKernel<<<GET_BLOCKS(out_size), GET_THREADS, 0, cuda_stream>>>(
    out_size, input_channel, input_height, input_width, input_depth, output_channel, output_height, output_width,
    output_depth, input_data, output_data);
  CHECK_CUDA_LAUNCH_SUCCESS();
}

template CUDA_LIB_EXPORT cudaError_t ApplyAdaptiveAvgPool3D<float>(const uint out_size, const uint input_channel,
                                                                   const uint input_height, const uint input_width,
                                                                   const uint input_depth, const uint output_channel,
                                                                   const uint output_height, const uint output_width,
                                                                   const uint output_depth, float *input_data,
                                                                   float *output_data, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t ApplyAdaptiveAvgPool3D<half>(const uint out_size, const uint input_channel,
                                                                  const uint input_height, const uint input_width,
                                                                  const uint input_depth, const uint output_channel,
                                                                  const uint output_height, const uint output_width,
                                                                  const uint output_depth, half *input_data,
                                                                  half *output_data, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t ApplyAdaptiveAvgPool3D<double>(const uint out_size, const uint input_channel,
                                                                    const uint input_height, const uint input_width,
                                                                    const uint input_depth, const uint output_channel,
                                                                    const uint output_height, const uint output_width,
                                                                    const uint output_depth, double *input_data,
                                                                    double *output_data, cudaStream_t cuda_stream);
