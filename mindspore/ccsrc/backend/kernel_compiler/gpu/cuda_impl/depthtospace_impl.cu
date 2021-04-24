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
#include "depthtospace_impl.cuh"
#include "runtime/device/gpu/cuda_common.h"

template <typename T>
__global__ void DepthToSpace(const size_t size, const T *input, const size_t in,
                             const size_t ic, const size_t ih, const size_t iw,
                             const size_t on, const size_t oc, const size_t oh,
                             const size_t ow, const size_t r, T *output) {
  size_t temp_stride = 0;
  size_t temp_pos = 0;
  size_t input_pos = 0;
  size_t output_pos_array[DEPTHTOSPACE_BUFFER_DIMENSION];

  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size;
       pos += blockDim.x * gridDim.x) {
    temp_stride = oc * oh * ow;
    output_pos_array[0] = pos / temp_stride;
    temp_pos = pos % temp_stride;

    temp_stride /= oc;
    output_pos_array[1] = temp_pos / temp_stride;
    temp_pos = pos % temp_stride;

    temp_stride /= oh;
    output_pos_array[2] = temp_pos / temp_stride;
    temp_pos = pos % temp_stride;

    temp_stride /= ow;
    output_pos_array[3] = temp_pos / temp_stride;

    input_pos += output_pos_array[0];
    input_pos =
        (input_pos * ic) +
        (output_pos_array[1] +
         (r * (output_pos_array[2] % r) + output_pos_array[3] % r) * oc);
    input_pos = (input_pos * ih) + (output_pos_array[2] / r);
    input_pos = (input_pos * iw) + (output_pos_array[3] / r);

    output[pos] = input[input_pos];
    input_pos = 0;
  }
  return;
}

template <typename T>
void CalDepthToSpace(const size_t size, const T *input, const size_t in,
                     const size_t ic, const size_t ih, const size_t iw,
                     const size_t on, const size_t oc, const size_t oh,
                     const size_t ow, const size_t r, T *output,
                     cudaStream_t cuda_stream) {
  DepthToSpace<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(
      size, input, in, ic, ih, iw, on, oc, oh, ow, r, output);
  return;
}

template void CalDepthToSpace<float>(const size_t size, const float *input,
                                     const size_t in, const size_t ic,
                                     const size_t ih, const size_t iw,
                                     const size_t on, const size_t oc,
                                     const size_t oh, const size_t ow,
                                     const size_t r, float *output,
                                     cudaStream_t cuda_stream);
template void CalDepthToSpace<half>(const size_t size, const half *input,
                                    const size_t in, const size_t ic,
                                    const size_t ih, const size_t iw,
                                    const size_t on, const size_t oc,
                                    const size_t oh, const size_t ow,
                                    const size_t r, half *output,
                                    cudaStream_t cuda_stream);
template void CalDepthToSpace<int>(const size_t size, const int *input,
                                   const size_t in, const size_t ic,
                                   const size_t ih, const size_t iw,
                                   const size_t on, const size_t oc,
                                   const size_t oh, const size_t ow,
                                   const size_t r, int *output,
                                   cudaStream_t cuda_stream);
template void CalDepthToSpace<int64_t>(const size_t size, const int64_t *input,
                                       const size_t in, const size_t ic,
                                       const size_t ih, const size_t iw,
                                       const size_t on, const size_t oc,
                                       const size_t oh, const size_t ow,
                                       const size_t r, int64_t *output,
                                       cudaStream_t cuda_stream);
template void CalDepthToSpace<int16_t>(const size_t size, const int16_t *input,
                                       const size_t in, const size_t ic,
                                       const size_t ih, const size_t iw,
                                       const size_t on, const size_t oc,
                                       const size_t oh, const size_t ow,
                                       const size_t r, int16_t *output,
                                       cudaStream_t cuda_stream);
template void CalDepthToSpace<int8_t>(const size_t size, const int8_t *input,
                                      const size_t in, const size_t ic,
                                      const size_t ih, const size_t iw,
                                      const size_t on, const size_t oc,
                                      const size_t oh, const size_t ow,
                                      const size_t r, int8_t *output,
                                      cudaStream_t cuda_stream);
template void CalDepthToSpace<uint8_t>(const size_t size, const uint8_t *input,
                                       const size_t in, const size_t ic,
                                       const size_t ih, const size_t iw,
                                       const size_t on, const size_t oc,
                                       const size_t oh, const size_t ow,
                                       const size_t r, uint8_t *output,
                                       cudaStream_t cuda_stream);
template void
CalDepthToSpace<uint16_t>(const size_t size, const uint16_t *input,
                          const size_t in, const size_t ic, const size_t ih,
                          const size_t iw, const size_t on, const size_t oc,
                          const size_t oh, const size_t ow, const size_t r,
                          uint16_t *output, cudaStream_t cuda_stream);
template void
CalDepthToSpace<uint32_t>(const size_t size, const uint32_t *input,
                          const size_t in, const size_t ic, const size_t ih,
                          const size_t iw, const size_t on, const size_t oc,
                          const size_t oh, const size_t ow, const size_t r,
                          uint32_t *output, cudaStream_t cuda_stream);
template void
CalDepthToSpace<uint64_t>(const size_t size, const uint64_t *input,
                          const size_t in, const size_t ic, const size_t ih,
                          const size_t iw, const size_t on, const size_t oc,
                          const size_t oh, const size_t ow, const size_t r,
                          uint64_t *output, cudaStream_t cuda_stream);
