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

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/adaptive_max_pool3d_grad_impl.cuh"
#include "include/cuda_fp16.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/util.cuh"

template <typename T, typename S>
__global__ void AdaptiveMaxPool3DGradKernel(const T *input_grad, const S *input_argmax, const int output_stride,
                                            const int argmax_stride, const int batch, T *output_data) {
  for (size_t n = blockIdx.x * blockDim.x + threadIdx.x; n < batch; n += blockDim.x * gridDim.x) {
    for (int64_t i = 0; i < argmax_stride; ++i) {
      int32_t maxp = input_argmax[i + n * argmax_stride] + n * output_stride;
      MsAtomicAdd(output_data + static_cast<int>(maxp), input_grad[i + n * argmax_stride]);
    }
  }
  return;
}

template <typename T, typename S>
cudaError_t CalAdaptiveMaxPool3DGrad(const T *input_grad, const S *input_argmax, const int output_stride,
                                     const int argmax_stride, const int batch, T *output_data,
                                     const uint32_t &device_id, cudaStream_t cuda_stream) {
  AdaptiveMaxPool3DGradKernel<<<CUDA_BLOCKS(device_id, batch), CUDA_THREADS(device_id), 0, cuda_stream>>>(
    input_grad, input_argmax, output_stride, argmax_stride, batch, output_data);
  CHECK_CUDA_LAUNCH_SUCCESS();
}

#define REG_ADAPTIVE_MAX_POOL3D_GRAD_CUDA(type1, type2)                                                   \
  template CUDA_LIB_EXPORT cudaError_t CalAdaptiveMaxPool3DGrad<type1, type2>(                            \
    const type1 *input_grad, const type2 *input_argmax, const int output_stride, const int argmax_stride, \
    const int batch, type1 *output_data, const uint32_t &device_id, cudaStream_t cuda_stream);

REG_ADAPTIVE_MAX_POOL3D_GRAD_CUDA(half, int32_t);
REG_ADAPTIVE_MAX_POOL3D_GRAD_CUDA(float, int32_t);
REG_ADAPTIVE_MAX_POOL3D_GRAD_CUDA(double, int32_t);
REG_ADAPTIVE_MAX_POOL3D_GRAD_CUDA(int8_t, int32_t);
REG_ADAPTIVE_MAX_POOL3D_GRAD_CUDA(int16_t, int32_t);
REG_ADAPTIVE_MAX_POOL3D_GRAD_CUDA(int32_t, int32_t);
REG_ADAPTIVE_MAX_POOL3D_GRAD_CUDA(int64_t, int32_t);
REG_ADAPTIVE_MAX_POOL3D_GRAD_CUDA(uint8_t, int32_t);
REG_ADAPTIVE_MAX_POOL3D_GRAD_CUDA(uint16_t, int32_t);
REG_ADAPTIVE_MAX_POOL3D_GRAD_CUDA(uint32_t, int32_t);
REG_ADAPTIVE_MAX_POOL3D_GRAD_CUDA(uint64_t, int32_t);

REG_ADAPTIVE_MAX_POOL3D_GRAD_CUDA(half, int64_t);
REG_ADAPTIVE_MAX_POOL3D_GRAD_CUDA(float, int64_t);
REG_ADAPTIVE_MAX_POOL3D_GRAD_CUDA(double, int64_t);
REG_ADAPTIVE_MAX_POOL3D_GRAD_CUDA(int8_t, int64_t);
REG_ADAPTIVE_MAX_POOL3D_GRAD_CUDA(int16_t, int64_t);
REG_ADAPTIVE_MAX_POOL3D_GRAD_CUDA(int32_t, int64_t);
REG_ADAPTIVE_MAX_POOL3D_GRAD_CUDA(int64_t, int64_t);
REG_ADAPTIVE_MAX_POOL3D_GRAD_CUDA(uint8_t, int64_t);
REG_ADAPTIVE_MAX_POOL3D_GRAD_CUDA(uint16_t, int64_t);
REG_ADAPTIVE_MAX_POOL3D_GRAD_CUDA(uint32_t, int64_t);
REG_ADAPTIVE_MAX_POOL3D_GRAD_CUDA(uint64_t, int64_t);
