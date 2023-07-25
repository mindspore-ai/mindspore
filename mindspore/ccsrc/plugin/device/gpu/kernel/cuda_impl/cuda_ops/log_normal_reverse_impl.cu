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

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/log_normal_reverse_impl.cuh"
#include <cub/cub.cuh>

__global__ void LogNormalReverseHalf(const half *input, half *output, const size_t elem_num, float *mask_h) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < elem_num; i += blockDim.x * gridDim.x) {
    output[i] = __float2half(mask_h[i]);
  }
}

cudaError_t CalLogNormalReverseHalf(const half *input, half *output, const size_t elem_num, float *mask_h,
                                    cudaStream_t cuda_stream_) {
  LogNormalReverseHalf<<<GET_BLOCKS(elem_num), GET_THREADS, 0, cuda_stream_>>>(input, output, elem_num, mask_h);
  return GetCudaStatus();
}
