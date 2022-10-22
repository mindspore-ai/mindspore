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

#include <cuda_runtime.h>
#include "spacetodepth_impl.cuh"
#include "include/cuda_fp16.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/complex.h"

template <typename T>
__global__ void SpaceToDepth(const size_t size, const T *input, const size_t in,
                             const size_t ic, const size_t ih, const size_t iw,
                             const size_t on, const size_t oc, const size_t oh,
                             const size_t ow, const size_t r, T *output) {
  size_t temp_stride = 0;
  size_t temp_pos = 0;
  size_t output_pos = 0;
  size_t input_pos_array[SPACETODEPTH_BUFFER_DIMENSION];

  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size;
       pos += blockDim.x * gridDim.x) {
    temp_stride = ic * ih * iw;
    input_pos_array[0] = pos / temp_stride;
    temp_pos = pos % temp_stride;

    temp_stride /= ic;
    input_pos_array[1] = temp_pos / temp_stride;
    temp_pos = pos % temp_stride;

    temp_stride /= ih;
    input_pos_array[2] = temp_pos / temp_stride;
    temp_pos = pos % temp_stride;

    temp_stride /= iw;
    input_pos_array[3] = temp_pos / temp_stride;

    output_pos += input_pos_array[0];
    output_pos = (output_pos * oc) +
                 (input_pos_array[1] +
                  (r * (input_pos_array[2] % r) + input_pos_array[3] % r) * ic);
    output_pos = (output_pos * oh) + (input_pos_array[2] / r);
    output_pos = (output_pos * ow) + (input_pos_array[3] / r);

    output[output_pos] = input[pos];
    output_pos = 0;
  }
  return;
}

template <typename T>
void CalSpaceToDepth(const size_t size, const T *input, const size_t in,
                     const size_t ic, const size_t ih, const size_t iw,
                     const size_t on, const size_t oc, const size_t oh,
                     const size_t ow, const size_t r, T *output,
                     cudaStream_t cuda_stream) {
  SpaceToDepth<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(
      size, input, in, ic, ih, iw, on, oc, oh, ow, r, output);
  return;
}

template CUDA_LIB_EXPORT void CalSpaceToDepth<float>(const size_t size, const float *input,
                                                     const size_t in, const size_t ic,
                                                     const size_t ih, const size_t iw,
                                                     const size_t on, const size_t oc,
                                                     const size_t oh, const size_t ow,
                                                     const size_t r, float *output,
                                                     cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalSpaceToDepth<half>(const size_t size, const half *input,
                                                    const size_t in, const size_t ic,
                                                    const size_t ih, const size_t iw,
                                                    const size_t on, const size_t oc,
                                                    const size_t oh, const size_t ow,
                                                    const size_t r, half *output,
                                                    cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalSpaceToDepth<int>(const size_t size, const int *input,
                                                   const size_t in, const size_t ic,
                                                   const size_t ih, const size_t iw,
                                                   const size_t on, const size_t oc,
                                                   const size_t oh, const size_t ow,
                                                   const size_t r, int *output,
                                                   cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalSpaceToDepth<int64_t>(const size_t size, const int64_t *input,
                                                       const size_t in, const size_t ic,
                                                       const size_t ih, const size_t iw,
                                                       const size_t on, const size_t oc,
                                                       const size_t oh, const size_t ow,
                                                       const size_t r, int64_t *output,
                                                       cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalSpaceToDepth<int16_t>(const size_t size, const int16_t *input,
                                                       const size_t in, const size_t ic,
                                                       const size_t ih, const size_t iw,
                                                       const size_t on, const size_t oc,
                                                       const size_t oh, const size_t ow,
                                                       const size_t r, int16_t *output,
                                                       cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalSpaceToDepth<int8_t>(const size_t size, const int8_t *input,
                                                      const size_t in, const size_t ic,
                                                      const size_t ih, const size_t iw,
                                                      const size_t on, const size_t oc,
                                                      const size_t oh, const size_t ow,
                                                      const size_t r, int8_t *output,
                                                      cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalSpaceToDepth<uint8_t>(const size_t size, const uint8_t *input,
                                                       const size_t in, const size_t ic,
                                                       const size_t ih, const size_t iw,
                                                       const size_t on, const size_t oc,
                                                       const size_t oh, const size_t ow,
                                                       const size_t r, uint8_t *output,
                                                       cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void
CalSpaceToDepth<uint16_t>(const size_t size, const uint16_t *input,
                          const size_t in, const size_t ic, const size_t ih,
                          const size_t iw, const size_t on, const size_t oc,
                          const size_t oh, const size_t ow, const size_t r,
                          uint16_t *output, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void
CalSpaceToDepth<uint32_t>(const size_t size, const uint32_t *input,
                          const size_t in, const size_t ic, const size_t ih,
                          const size_t iw, const size_t on, const size_t oc,
                          const size_t oh, const size_t ow, const size_t r,
                          uint32_t *output, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void
CalSpaceToDepth<uint64_t>(const size_t size, const uint64_t *input,
                          const size_t in, const size_t ic, const size_t ih,
                          const size_t iw, const size_t on, const size_t oc,
                          const size_t oh, const size_t ow, const size_t r,
                          uint64_t *output, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalSpaceToDepth<Complex<float>>(const size_t size, const Complex<float> *input,
                                                              const size_t in, const size_t ic, const size_t ih,
                                                              const size_t iw, const size_t on, const size_t oc,
                                                              const size_t oh, const size_t ow, const size_t r,
                                                              Complex<float> *output, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalSpaceToDepth<Complex<double>>(const size_t size, const Complex<double> *input,
                                                               const size_t in, const size_t ic, const size_t ih,
                                                               const size_t iw, const size_t on, const size_t oc,
                                                               const size_t oh, const size_t ow, const size_t r,
                                                               Complex<double> *output, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalSpaceToDepth<double>(const size_t size, const double *input,
                                                      const size_t in, const size_t ic,
                                                      const size_t ih, const size_t iw,
                                                      const size_t on, const size_t oc,
                                                      const size_t oh, const size_t ow,
                                                      const size_t r, double *output,
                                                      cudaStream_t cuda_stream);
