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
#include "spacetobatch_impl.cuh"
#include "include/cuda_fp16.h"

template <typename T>
__global__ void SpaceToBatch(const size_t size, const T *input, const size_t in, const size_t ih, const size_t iw,
                             const size_t ic, const size_t on, const size_t oh, const size_t ow, const size_t oc,
                             const size_t pad_up, const size_t pad_dn, const size_t pad_lft, const size_t pad_rht,
                             const size_t block_num, T *output) {
  size_t temp_stride = 0;
  size_t temp_pos = 0;
  size_t idx_in = 0;
  size_t idx_ic = 0;
  size_t idx_ih = 0;
  size_t idx_iw = 0;
  size_t idx_on = 0;
  size_t output_pos = 0;
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    temp_stride = ic * ih * iw;
    idx_in = pos / temp_stride;
    temp_pos = pos % temp_stride;

    temp_stride /= ic;
    idx_ic = temp_pos / temp_stride;
    temp_pos = pos % temp_stride;

    temp_stride /= ih;
    idx_ih = temp_pos / temp_stride;
    temp_pos = pos % temp_stride;

    temp_stride /= iw;
    idx_iw = temp_pos / temp_stride;

    idx_on = (((idx_ih + pad_up) % block_num) * block_num + ((idx_iw + pad_lft) % block_num)) * in + idx_in;
    output_pos = idx_on * oc;
    output_pos = (output_pos + idx_ic) * oh;
    output_pos = (output_pos + ((idx_ih + pad_up) - (idx_on / (in * block_num))) / block_num) * ow;
    output_pos = (output_pos + ((idx_iw + pad_lft) - ((idx_on / in) % block_num)) / block_num);
    output[output_pos] = input[pos];
  }
  return;
}

template <typename T>
cudaError_t CalSpaceToBatch(const size_t size, const T *input, const size_t in, const size_t ih, const size_t iw,
                            const size_t ic, const size_t on, const size_t oh, const size_t ow, const size_t oc,
                            const size_t pad_up, const size_t pad_dn, const size_t pad_lft, const size_t pad_rht,
                            const size_t block_num, T *output, const uint32_t &device_id, cudaStream_t cuda_stream) {
  cudaMemset(output, 0, on * oc * oh * ow * sizeof(T));
  SpaceToBatch<<<CUDA_BLOCKS(device_id, size), CUDA_THREADS(device_id), 0, cuda_stream>>>(
    size, input, in, ih, iw, ic, on, oh, ow, oc, pad_up, pad_dn, pad_lft, pad_rht, block_num, output);
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t CalSpaceToBatch<float>(const size_t size, const float *input, const size_t in,
                                                            const size_t ih, const size_t iw, const size_t ic,
                                                            const size_t on, const size_t oh, const size_t ow,
                                                            const size_t oc, const size_t pad_up, const size_t pad_dn,
                                                            const size_t pad_lft, const size_t pad_rht,
                                                            const size_t block_num, float *output,
                                                            const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalSpaceToBatch<half>(const size_t size, const half *input, const size_t in,
                                                           const size_t ih, const size_t iw, const size_t ic,
                                                           const size_t on, const size_t oh, const size_t ow,
                                                           const size_t oc, const size_t pad_up, const size_t pad_dn,
                                                           const size_t pad_lft, const size_t pad_rht,
                                                           const size_t block_num, half *output,
                                                           const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalSpaceToBatch<int>(const size_t size, const int *input, const size_t in,
                                                          const size_t ih, const size_t iw, const size_t ic,
                                                          const size_t on, const size_t oh, const size_t ow,
                                                          const size_t oc, const size_t pad_up, const size_t pad_dn,
                                                          const size_t pad_lft, const size_t pad_rht,
                                                          const size_t block_num, int *output,
                                                          const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalSpaceToBatch<int64_t>(const size_t size, const int64_t *input, const size_t in,
                                                              const size_t ih, const size_t iw, const size_t ic,
                                                              const size_t on, const size_t oh, const size_t ow,
                                                              const size_t oc, const size_t pad_up, const size_t pad_dn,
                                                              const size_t pad_lft, const size_t pad_rht,
                                                              const size_t block_num, int64_t *output,
                                                              const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalSpaceToBatch<int16_t>(const size_t size, const int16_t *input, const size_t in,
                                                              const size_t ih, const size_t iw, const size_t ic,
                                                              const size_t on, const size_t oh, const size_t ow,
                                                              const size_t oc, const size_t pad_up, const size_t pad_dn,
                                                              const size_t pad_lft, const size_t pad_rht,
                                                              const size_t block_num, int16_t *output,
                                                              const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalSpaceToBatch<int8_t>(const size_t size, const int8_t *input, const size_t in,
                                                             const size_t ih, const size_t iw, const size_t ic,
                                                             const size_t on, const size_t oh, const size_t ow,
                                                             const size_t oc, const size_t pad_up, const size_t pad_dn,
                                                             const size_t pad_lft, const size_t pad_rht,
                                                             const size_t block_num, int8_t *output,
                                                             const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalSpaceToBatch<uint8_t>(const size_t size, const uint8_t *input, const size_t in,
                                                              const size_t ih, const size_t iw, const size_t ic,
                                                              const size_t on, const size_t oh, const size_t ow,
                                                              const size_t oc, const size_t pad_up, const size_t pad_dn,
                                                              const size_t pad_lft, const size_t pad_rht,
                                                              const size_t block_num, uint8_t *output,
                                                              const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalSpaceToBatch<uint16_t>(
  const size_t size, const uint16_t *input, const size_t in, const size_t ih, const size_t iw, const size_t ic,
  const size_t on, const size_t oh, const size_t ow, const size_t oc, const size_t pad_up, const size_t pad_dn,
  const size_t pad_lft, const size_t pad_rht, const size_t block_num, uint16_t *output, const uint32_t &device_id,
  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalSpaceToBatch<uint32_t>(
  const size_t size, const uint32_t *input, const size_t in, const size_t ih, const size_t iw, const size_t ic,
  const size_t on, const size_t oh, const size_t ow, const size_t oc, const size_t pad_up, const size_t pad_dn,
  const size_t pad_lft, const size_t pad_rht, const size_t block_num, uint32_t *output, const uint32_t &device_id,
  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalSpaceToBatch<uint64_t>(
  const size_t size, const uint64_t *input, const size_t in, const size_t ih, const size_t iw, const size_t ic,
  const size_t on, const size_t oh, const size_t ow, const size_t oc, const size_t pad_up, const size_t pad_dn,
  const size_t pad_lft, const size_t pad_rht, const size_t block_num, uint64_t *output, const uint32_t &device_id,
  cudaStream_t cuda_stream);
