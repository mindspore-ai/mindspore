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
#include "batchtospace_impl.cuh"
#include "include/cuda_fp16.h"

template <typename T>
__global__ void BatchToSpace(const size_t size, const T *input, const size_t in, const size_t ih, const size_t iw,
                             const size_t ic, const size_t on, const size_t oh, const size_t ow, const size_t oc,
                             const size_t crop_up, const size_t crop_dn, const size_t crop_lft, const size_t crop_rht,
                             const size_t block_num, T *output) {
  size_t temp_stride = 0;
  size_t temp_pos = 0;
  size_t idx_on = 0;
  size_t idx_oc = 0;
  size_t idx_oh = 0;
  size_t idx_ow = 0;
  size_t idx_in = 0;
  size_t input_pos = 0;
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    temp_stride = oc * oh * ow;
    idx_on = pos / temp_stride;
    temp_pos = pos % temp_stride;

    temp_stride /= oc;
    idx_oc = temp_pos / temp_stride;
    temp_pos = pos % temp_stride;

    temp_stride /= oh;
    idx_oh = temp_pos / temp_stride;
    temp_pos = pos % temp_stride;

    temp_stride /= ow;
    idx_ow = temp_pos / temp_stride;

    idx_in = (((idx_oh + crop_up) % block_num) * block_num + ((idx_ow + crop_lft) % block_num)) * on + idx_on;
    input_pos = idx_in * ic;
    input_pos = (input_pos + idx_oc) * ih;
    input_pos = (input_pos + ((idx_oh + crop_up) - (idx_in / (on * block_num))) / block_num) * iw;
    input_pos = (input_pos + ((idx_ow + crop_lft) - ((idx_in / on) % block_num)) / block_num);
    output[pos] = input[input_pos];
  }
  return;
}

template <typename T>
cudaError_t CalBatchToSpace(const size_t size, const T *input, const size_t in, const size_t ih, const size_t iw,
                            const size_t ic, const size_t on, const size_t oh, const size_t ow, const size_t oc,
                            const size_t crop_up, const size_t crop_dn, const size_t crop_lft, const size_t crop_rht,
                            const size_t block_num, T *output, const uint32_t &device_id, cudaStream_t cuda_stream) {
  BatchToSpace<<<CUDA_BLOCKS(device_id, size), CUDA_THREADS(device_id), 0, cuda_stream>>>(
    size, input, in, ih, iw, ic, on, oh, ow, oc, crop_up, crop_dn, crop_lft, crop_rht, block_num, output);
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t CalBatchToSpace<float>(const size_t size, const float *input, const size_t in,
                                                            const size_t ih, const size_t iw, const size_t ic,
                                                            const size_t on, const size_t oh, const size_t ow,
                                                            const size_t oc, const size_t crop_up, const size_t crop_dn,
                                                            const size_t crop_lft, const size_t crop_rht,
                                                            const size_t block_num, float *output,
                                                            const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalBatchToSpace<half>(const size_t size, const half *input, const size_t in,
                                                           const size_t ih, const size_t iw, const size_t ic,
                                                           const size_t on, const size_t oh, const size_t ow,
                                                           const size_t oc, const size_t crop_up, const size_t crop_dn,
                                                           const size_t crop_lft, const size_t crop_rht,
                                                           const size_t block_num, half *output,
                                                           const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalBatchToSpace<int>(const size_t size, const int *input, const size_t in,
                                                          const size_t ih, const size_t iw, const size_t ic,
                                                          const size_t on, const size_t oh, const size_t ow,
                                                          const size_t oc, const size_t crop_up, const size_t crop_dn,
                                                          const size_t crop_lft, const size_t crop_rht,
                                                          const size_t block_num, int *output,
                                                          const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t
CalBatchToSpace<int64_t>(const size_t size, const int64_t *input, const size_t in, const size_t ih, const size_t iw,
                         const size_t ic, const size_t on, const size_t oh, const size_t ow, const size_t oc,
                         const size_t crop_up, const size_t crop_dn, const size_t crop_lft, const size_t crop_rht,
                         const size_t block_num, int64_t *output, const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t
CalBatchToSpace<int16_t>(const size_t size, const int16_t *input, const size_t in, const size_t ih, const size_t iw,
                         const size_t ic, const size_t on, const size_t oh, const size_t ow, const size_t oc,
                         const size_t crop_up, const size_t crop_dn, const size_t crop_lft, const size_t crop_rht,
                         const size_t block_num, int16_t *output, const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t
CalBatchToSpace<int8_t>(const size_t size, const int8_t *input, const size_t in, const size_t ih, const size_t iw,
                        const size_t ic, const size_t on, const size_t oh, const size_t ow, const size_t oc,
                        const size_t crop_up, const size_t crop_dn, const size_t crop_lft, const size_t crop_rht,
                        const size_t block_num, int8_t *output, const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t
CalBatchToSpace<uint8_t>(const size_t size, const uint8_t *input, const size_t in, const size_t ih, const size_t iw,
                         const size_t ic, const size_t on, const size_t oh, const size_t ow, const size_t oc,
                         const size_t crop_up, const size_t crop_dn, const size_t crop_lft, const size_t crop_rht,
                         const size_t block_num, uint8_t *output, const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalBatchToSpace<uint16_t>(
  const size_t size, const uint16_t *input, const size_t in, const size_t ih, const size_t iw, const size_t ic,
  const size_t on, const size_t oh, const size_t ow, const size_t oc, const size_t crop_up, const size_t crop_dn,
  const size_t crop_lft, const size_t crop_rht, const size_t block_num, uint16_t *output, const uint32_t &device_id,
  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalBatchToSpace<uint32_t>(
  const size_t size, const uint32_t *input, const size_t in, const size_t ih, const size_t iw, const size_t ic,
  const size_t on, const size_t oh, const size_t ow, const size_t oc, const size_t crop_up, const size_t crop_dn,
  const size_t crop_lft, const size_t crop_rht, const size_t block_num, uint32_t *output, const uint32_t &device_id,
  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalBatchToSpace<uint64_t>(
  const size_t size, const uint64_t *input, const size_t in, const size_t ih, const size_t iw, const size_t ic,
  const size_t on, const size_t oh, const size_t ow, const size_t oc, const size_t crop_up, const size_t crop_dn,
  const size_t crop_lft, const size_t crop_rht, const size_t block_num, uint64_t *output, const uint32_t &device_id,
  cudaStream_t cuda_stream);
