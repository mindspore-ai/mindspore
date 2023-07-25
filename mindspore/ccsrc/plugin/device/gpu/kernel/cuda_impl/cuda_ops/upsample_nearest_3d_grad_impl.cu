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
 * WITposh WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 **/

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/upsample_nearest_3d_grad_impl.cuh"
#include <algorithm>
#include <iostream>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/util.cuh"

#define MAX_THREADS 512

__inline__ __device__ int compute_source_index(const float scale, int dst_index, int output_size, int input_size) {
  int src_index = 0;
  if (output_size == input_size) {
    src_index = min(static_cast<int>(dst_index * scale), output_size);
  } else {
    float gap = ceilf(dst_index * scale) - dst_index * scale;
    if (fabs(gap - 0.0) < 1e-6 || fabs(gap - 1.0) < 1e-6) {
      src_index = min(static_cast<int>(dst_index * scale), output_size);
    } else {
      src_index = min(static_cast<int>(ceilf(dst_index * scale)), output_size);
    }
  }
  return src_index;
}

template <typename T, typename S>
__global__ void UpsampleNearest3DGradKernel(const T *dy, const int dim_n, const int dim_c, const int in_d,
                                            const int in_h, const int in_w, const int in_cdhw, const int in_dhw,
                                            const int in_hw, const int out_d, const int out_h, const int out_w,
                                            const int out_cdhw, const int out_dhw, const int out_hw,
                                            const float d_scale, const float h_scale, const float w_scale, T *dx) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < out_cdhw; pos += blockDim.x * gridDim.x) {
    const int c = pos / out_dhw;
    const int dx_z = pos / out_hw % out_d;
    const int dx_y = pos / out_w % out_h;
    const int dx_x = pos % out_w;
    int dy_z = compute_source_index(d_scale, dx_z, in_d, out_d);
    int dy_z_up = compute_source_index(d_scale, dx_z + 1, in_d, out_d);
    int dy_y = compute_source_index(h_scale, dx_y, in_h, out_h);
    int dy_y_up = compute_source_index(h_scale, dx_y + 1, in_h, out_h);
    int dy_x = compute_source_index(w_scale, dx_x, in_w, out_w);
    int dy_x_up = compute_source_index(w_scale, dx_x + 1, in_w, out_w);
    int src_offset = c * in_dhw;
    int dst_offset = 0;
    for (int b = 0; b < dim_n; ++b) {
      S grad = 0;
      for (int z = dy_z; z < dy_z_up; z++) {
        for (int y = dy_y; y < dy_y_up; y++) {
          for (int x = dy_x; x < dy_x_up; x++) {
            const int src_idx = src_offset + z * in_hw + y * in_w + x;
            grad += static_cast<S>(dy[src_idx]);
          }
        }
      }
      dx[dst_offset + pos] = static_cast<T>(grad);
      src_offset += in_cdhw;
      dst_offset += out_cdhw;
    }
  }
}

template <typename T>
cudaError_t CalUpsampleNearest3DGrad(const T *dy, const int n, const int c, const int dy_d, const int dy_h,
                                     const int dy_w, const int dx_d, const int dx_h, const int dx_w,
                                     const float d_scale, const float h_scale, const float w_scale, T *dx,
                                     const uint32_t device_id, cudaStream_t cuda_stream) {
  const int dx_hw = dx_h * dx_w;
  const int dx_dhw = dx_d * dx_hw;
  const int dx_cdhw = c * dx_dhw;
  const int dy_hw = dy_h * dy_w;
  const int dy_dhw = dy_d * dy_hw;
  const int dy_cdhw = c * dy_dhw;
  const int blockSize = std::min(CUDA_THREADS(device_id), static_cast<int>(MAX_THREADS));
  const int gridSize = (dx_cdhw + blockSize - 1) / blockSize;
  UpsampleNearest3DGradKernel<T, T>
    <<<gridSize, blockSize, 0, cuda_stream>>>(dy, n, c, dy_d, dy_h, dy_w, dy_cdhw, dy_dhw, dy_hw, dx_d, dx_h, dx_w,
                                              dx_cdhw, dx_dhw, dx_hw, d_scale, h_scale, w_scale, dx);
  return GetCudaStatus();
}

template <>
cudaError_t CalUpsampleNearest3DGrad(const half *dy, const int n, const int c, const int dy_d, const int dy_h,
                                     const int dy_w, const int dx_d, const int dx_h, const int dx_w,
                                     const float d_scale, const float h_scale, const float w_scale, half *dx,
                                     const uint32_t device_id, cudaStream_t cuda_stream) {
  const int dx_hw = dx_h * dx_w;
  const int dx_dhw = dx_d * dx_hw;
  const int dx_cdhw = c * dx_dhw;
  const int dy_hw = dy_h * dy_w;
  const int dy_dhw = dy_d * dy_hw;
  const int dy_cdhw = c * dy_dhw;
  const int blockSize = std::min(CUDA_THREADS(device_id), static_cast<int>(MAX_THREADS));
  const int gridSize = (dx_cdhw + blockSize - 1) / blockSize;
  UpsampleNearest3DGradKernel<half, float>
    <<<gridSize, blockSize, 0, cuda_stream>>>(dy, n, c, dy_d, dy_h, dy_w, dy_cdhw, dy_dhw, dy_hw, dx_d, dx_h, dx_w,
                                              dx_cdhw, dx_dhw, dx_hw, d_scale, h_scale, w_scale, dx);
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t CalUpsampleNearest3DGrad(const half *dy, const int n, const int c, const int dy_d,
                                                              const int dy_h, const int dy_w, const int dx_d,
                                                              const int dx_h, const int dx_w, const float d_scale,
                                                              const float h_scale, const float w_scale, half *dx,
                                                              const uint32_t device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalUpsampleNearest3DGrad(const float *dy, const int n, const int c, const int dy_d,
                                                              const int dy_h, const int dy_w, const int dx_d,
                                                              const int dx_h, const int dx_w, const float d_scale,
                                                              const float h_scale, const float w_scale, float *dx,
                                                              const uint32_t device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalUpsampleNearest3DGrad(const double *dy, const int n, const int c,
                                                              const int dy_d, const int dy_h, const int dy_w,
                                                              const int dx_d, const int dx_h, const int dx_w,
                                                              const float d_scale, const float h_scale,
                                                              const float w_scale, double *dx, const uint32_t device_id,
                                                              cudaStream_t cuda_stream);
