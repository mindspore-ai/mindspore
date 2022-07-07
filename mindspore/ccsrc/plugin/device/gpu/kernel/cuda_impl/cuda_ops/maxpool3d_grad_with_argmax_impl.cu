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

#include <algorithm>
#include "maxpool3d_grad_with_argmax_impl.cuh"
#include "include/cuda_fp16.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/util.cuh"

template <typename T, typename S>
__global__ void MaxPool3DGradWithArgmax(const T *dy, const S *index, const int64_t x_dhw, const int64_t dy_dhw,
                                        const int64_t dy_ncdhw, T *dx) {
  for (int64_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < dy_ncdhw; pos += blockDim.x * gridDim.x) {
    const S idx = index[pos];
    int64_t offset = (pos / dy_dhw) * x_dhw;
    MsAtomicAdd(dx + offset + idx, dy[pos]);
  }
  return;
}

template <typename T, typename S>
void CalMaxPool3DGradWithArgmax(const T *dy, const S *index, const int64_t x_dhw, const int64_t dy_dhw,
                                const int64_t dy_ncdhw, T *dx, const uint32_t device_id, cudaStream_t cuda_stream) {
  dim3 grid = CUDA_GRIDS_MAXSIZE(device_id);
  int64_t thread_num = CUDA_THREADS(device_id);
  int64_t size = std::min(static_cast<int64_t>(grid.x), dy_ncdhw);
  size = (size + thread_num - 1) / thread_num;
  MaxPool3DGradWithArgmax<<<size, thread_num, 0, cuda_stream>>>(dy, index, x_dhw, dy_dhw, dy_ncdhw, dx);
  return;
}

template CUDA_LIB_EXPORT void CalMaxPool3DGradWithArgmax<half, int32_t>(const half *dy, const int32_t *index,
                                                                        const int64_t x_dhw, const int64_t dy_dhw,
                                                                        const int64_t dy_ncdhw, half *dx,
                                                                        const uint32_t device_id,
                                                                        cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalMaxPool3DGradWithArgmax<float, int32_t>(const float *dy, const int32_t *index,
                                                                         const int64_t x_dhw, const int64_t dy_dhw,
                                                                         const int64_t dy_ncdhw, float *dx,
                                                                         const uint32_t device_id,
                                                                         cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalMaxPool3DGradWithArgmax<double, int32_t>(const double *dy, const int32_t *index,
                                                                          const int64_t x_dhw, const int64_t dy_dhw,
                                                                          const int64_t dy_ncdhw, double *dx,
                                                                          const uint32_t device_id,
                                                                          cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalMaxPool3DGradWithArgmax<int8_t, int32_t>(const int8_t *dy, const int32_t *index,
                                                                          const int64_t x_dhw, const int64_t dy_dhw,
                                                                          const int64_t dy_ncdhw, int8_t *dx,
                                                                          const uint32_t device_id,
                                                                          cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalMaxPool3DGradWithArgmax<int16_t, int32_t>(const int16_t *dy, const int32_t *index,
                                                                           const int64_t x_dhw, const int64_t dy_dhw,
                                                                           const int64_t dy_ncdhw, int16_t *dx,
                                                                           const uint32_t device_id,
                                                                           cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalMaxPool3DGradWithArgmax<int32_t, int32_t>(const int32_t *dy, const int32_t *index,
                                                                           const int64_t x_dhw, const int64_t dy_dhw,
                                                                           const int64_t dy_ncdhw, int32_t *dx,
                                                                           const uint32_t device_id,
                                                                           cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalMaxPool3DGradWithArgmax<int64_t, int32_t>(const int64_t *dy, const int32_t *index,
                                                                           const int64_t x_dhw, const int64_t dy_dhw,
                                                                           const int64_t dy_ncdhw, int64_t *dx,
                                                                           const uint32_t device_id,
                                                                           cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalMaxPool3DGradWithArgmax<uint8_t, int32_t>(const uint8_t *dy, const int32_t *index,
                                                                           const int64_t x_dhw, const int64_t dy_dhw,
                                                                           const int64_t dy_ncdhw, uint8_t *dx,
                                                                           const uint32_t device_id,
                                                                           cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalMaxPool3DGradWithArgmax<uint16_t, int32_t>(const uint16_t *dy, const int32_t *index,
                                                                            const int64_t x_dhw, const int64_t dy_dhw,
                                                                            const int64_t dy_ncdhw, uint16_t *dx,
                                                                            const uint32_t device_id,
                                                                            cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalMaxPool3DGradWithArgmax<uint32_t, int32_t>(const uint32_t *dy, const int32_t *index,
                                                                            const int64_t x_dhw, const int64_t dy_dhw,
                                                                            const int64_t dy_ncdhw, uint32_t *dx,
                                                                            const uint32_t device_id,
                                                                            cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalMaxPool3DGradWithArgmax<uint64_t, int32_t>(const uint64_t *dy, const int32_t *index,
                                                                            const int64_t x_dhw, const int64_t dy_dhw,
                                                                            const int64_t dy_ncdhw, uint64_t *dx,
                                                                            const uint32_t device_id,
                                                                            cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalMaxPool3DGradWithArgmax<half, int64_t>(const half *dy, const int64_t *index,
                                                                        const int64_t x_dhw, const int64_t dy_dhw,
                                                                        const int64_t dy_ncdhw, half *dx,
                                                                        const uint32_t device_id,
                                                                        cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalMaxPool3DGradWithArgmax<float, int64_t>(const float *dy, const int64_t *index,
                                                                         const int64_t x_dhw, const int64_t dy_dhw,
                                                                         const int64_t dy_ncdhw, float *dx,
                                                                         const uint32_t device_id,
                                                                         cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalMaxPool3DGradWithArgmax<double, int64_t>(const double *dy, const int64_t *index,
                                                                          const int64_t x_dhw, const int64_t dy_dhw,
                                                                          const int64_t dy_ncdhw, double *dx,
                                                                          const uint32_t device_id,
                                                                          cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalMaxPool3DGradWithArgmax<int8_t, int64_t>(const int8_t *dy, const int64_t *index,
                                                                          const int64_t x_dhw, const int64_t dy_dhw,
                                                                          const int64_t dy_ncdhw, int8_t *dx,
                                                                          const uint32_t device_id,
                                                                          cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalMaxPool3DGradWithArgmax<int16_t, int64_t>(const int16_t *dy, const int64_t *index,
                                                                           const int64_t x_dhw, const int64_t dy_dhw,
                                                                           const int64_t dy_ncdhw, int16_t *dx,
                                                                           const uint32_t device_id,
                                                                           cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalMaxPool3DGradWithArgmax<int32_t, int64_t>(const int32_t *dy, const int64_t *index,
                                                                           const int64_t x_dhw, const int64_t dy_dhw,
                                                                           const int64_t dy_ncdhw, int32_t *dx,
                                                                           const uint32_t device_id,
                                                                           cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalMaxPool3DGradWithArgmax<int64_t, int64_t>(const int64_t *dy, const int64_t *index,
                                                                           const int64_t x_dhw, const int64_t dy_dhw,
                                                                           const int64_t dy_ncdhw, int64_t *dx,
                                                                           const uint32_t device_id,
                                                                           cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalMaxPool3DGradWithArgmax<uint8_t, int64_t>(const uint8_t *dy, const int64_t *index,
                                                                           const int64_t x_dhw, const int64_t dy_dhw,
                                                                           const int64_t dy_ncdhw, uint8_t *dx,
                                                                           const uint32_t device_id,
                                                                           cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalMaxPool3DGradWithArgmax<uint16_t, int64_t>(const uint16_t *dy, const int64_t *index,
                                                                            const int64_t x_dhw, const int64_t dy_dhw,
                                                                            const int64_t dy_ncdhw, uint16_t *dx,
                                                                            const uint32_t device_id,
                                                                            cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalMaxPool3DGradWithArgmax<uint32_t, int64_t>(const uint32_t *dy, const int64_t *index,
                                                                            const int64_t x_dhw, const int64_t dy_dhw,
                                                                            const int64_t dy_ncdhw, uint32_t *dx,
                                                                            const uint32_t device_id,
                                                                            cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalMaxPool3DGradWithArgmax<uint64_t, int64_t>(const uint64_t *dy, const int64_t *index,
                                                                            const int64_t x_dhw, const int64_t dy_dhw,
                                                                            const int64_t dy_ncdhw, uint64_t *dx,
                                                                            const uint32_t device_id,
                                                                            cudaStream_t cuda_stream);
