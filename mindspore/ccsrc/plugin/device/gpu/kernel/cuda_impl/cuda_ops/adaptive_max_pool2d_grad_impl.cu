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

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/adaptive_max_pool2d_grad_impl.cuh"
#include "include/cuda_fp16.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/util.cuh"

template <typename T, typename S>
__global__ void AdaptiveMaxPool2DGradKernel(const T *input_data, const S *max_index, const int input_nchw,
                                            const int input_hw, const int output_hw, T *output_data) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < input_nchw; pos += blockDim.x * gridDim.x) {
    const S idx = max_index[pos];
    const int posn = pos / input_hw;
    MsAtomicAdd(output_data + posn * output_hw + static_cast<int>(idx), input_data[pos]);
  }
  return;
}

template <typename T, typename S>
cudaError_t CalAdaptiveMaxPool2DGrad(const T *input_data, const S *max_index, const int n, const int c,
                                     const uint input_height, const uint input_width, const uint output_height,
                                     const uint output_width, T *output_data, const uint32_t &device_id,
                                     cudaStream_t cuda_stream) {
  const int input_hw = input_height * input_width;
  const int input_chw = c * input_hw;
  const int input_nchw = n * input_chw;
  const int output_hw = output_height * output_width;

  AdaptiveMaxPool2DGradKernel<<<CUDA_BLOCKS(device_id, input_nchw), CUDA_THREADS(device_id), 0, cuda_stream>>>(
    input_data, max_index, input_nchw, input_hw, output_hw, output_data);
  CHECK_CUDA_LAUNCH_SUCCESS();
}

#define REG_ADAPTIVE_MAX_POOL2D_GRAD_CUDA(type1, type2)                                                 \
  template CUDA_LIB_EXPORT cudaError_t CalAdaptiveMaxPool2DGrad<type1, type2>(                          \
    const type1 *input_data, const type2 *max_index, const int n, const int c, const uint input_height, \
    const uint input_width, const uint output_height, const uint output_width, type1 *output_data,      \
    const uint32_t &device_id, cudaStream_t cuda_stream)

REG_ADAPTIVE_MAX_POOL2D_GRAD_CUDA(half, int32_t);
REG_ADAPTIVE_MAX_POOL2D_GRAD_CUDA(float, int32_t);
REG_ADAPTIVE_MAX_POOL2D_GRAD_CUDA(double, int32_t);
REG_ADAPTIVE_MAX_POOL2D_GRAD_CUDA(int8_t, int32_t);
REG_ADAPTIVE_MAX_POOL2D_GRAD_CUDA(int16_t, int32_t);
REG_ADAPTIVE_MAX_POOL2D_GRAD_CUDA(int32_t, int32_t);
REG_ADAPTIVE_MAX_POOL2D_GRAD_CUDA(int64_t, int32_t);
REG_ADAPTIVE_MAX_POOL2D_GRAD_CUDA(uint8_t, int32_t);
REG_ADAPTIVE_MAX_POOL2D_GRAD_CUDA(uint16_t, int32_t);
REG_ADAPTIVE_MAX_POOL2D_GRAD_CUDA(uint32_t, int32_t);
REG_ADAPTIVE_MAX_POOL2D_GRAD_CUDA(uint64_t, int32_t);

REG_ADAPTIVE_MAX_POOL2D_GRAD_CUDA(half, int64_t);
REG_ADAPTIVE_MAX_POOL2D_GRAD_CUDA(float, int64_t);
REG_ADAPTIVE_MAX_POOL2D_GRAD_CUDA(double, int64_t);
REG_ADAPTIVE_MAX_POOL2D_GRAD_CUDA(int8_t, int64_t);
REG_ADAPTIVE_MAX_POOL2D_GRAD_CUDA(int16_t, int64_t);
REG_ADAPTIVE_MAX_POOL2D_GRAD_CUDA(int32_t, int64_t);
REG_ADAPTIVE_MAX_POOL2D_GRAD_CUDA(int64_t, int64_t);
REG_ADAPTIVE_MAX_POOL2D_GRAD_CUDA(uint8_t, int64_t);
REG_ADAPTIVE_MAX_POOL2D_GRAD_CUDA(uint16_t, int64_t);
REG_ADAPTIVE_MAX_POOL2D_GRAD_CUDA(uint32_t, int64_t);
REG_ADAPTIVE_MAX_POOL2D_GRAD_CUDA(uint64_t, int64_t);
