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

template <typename T>
__global__ void UpsampleNearest3DGrad(const T *dy, const size_t dim_n, const size_t dim_c, const size_t in_d,
                                      const size_t in_h, const size_t in_w, const size_t in_cdhw, const size_t in_dhw,
                                      const size_t in_hw, const size_t out_d, const size_t out_h, const size_t out_w,
                                      const size_t out_ncdhw, const size_t out_cdhw, const size_t out_dhw,
                                      const size_t out_hw, const float d_scale, const float h_scale,
                                      const float w_scale, T *dx) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < out_ncdhw; pos += blockDim.x * gridDim.x) {
    const size_t dx_n = pos / out_cdhw;
    const size_t dx_c = pos / out_dhw % dim_c;
    const size_t dx_z = pos / out_hw % out_d;
    const size_t dx_y = pos / out_w % out_h;
    const size_t dx_x = pos % out_w;
    int dy_z = compute_source_index(d_scale, dx_z, in_d, out_d);
    int dy_z_up = compute_source_index(d_scale, dx_z + 1, in_d, out_d);
    int dy_y = compute_source_index(h_scale, dx_y, in_h, out_h);
    int dy_y_up = compute_source_index(h_scale, dx_y + 1, in_h, out_h);
    int dy_x = compute_source_index(w_scale, dx_x, in_w, out_w);
    int dy_x_up = compute_source_index(w_scale, dx_x + 1, in_w, out_w);
    float grad = 0;
    for (int z = dy_z; z < dy_z_up; z++) {
      for (int y = dy_y; y < dy_y_up; y++) {
        for (int x = dy_x; x < dy_x_up; x++) {
          const size_t src_idx = dx_n * in_cdhw + dx_c * in_dhw + z * in_hw + y * in_w + x;
          grad += static_cast<float>(dy[src_idx]);
        }
      }
    }
    dx[pos] = static_cast<T>(grad);
  }
}

template <typename T>
void CalUpsampleNearest3DGrad(const T *dy, const size_t n, const size_t c, const size_t dy_d, const size_t dy_h,
                              const size_t dy_w, const size_t dx_d, const size_t dx_h, const size_t dx_w,
                              const float d_scale, const float h_scale, const float w_scale, T *dx,
                              cudaStream_t cuda_stream) {
  const size_t dx_hw = dx_h * dx_w;
  const size_t dx_dhw = dx_d * dx_hw;
  const size_t dx_cdhw = c * dx_dhw;
  const size_t dx_ncdhw = n * dx_cdhw;
  const size_t dy_hw = dy_h * dy_w;
  const size_t dy_dhw = dy_d * dy_hw;
  const size_t dy_cdhw = c * dy_dhw;
  UpsampleNearest3DGrad<<<GET_BLOCKS(dx_ncdhw), GET_THREADS, 0, cuda_stream>>>(
    dy, n, c, dy_d, dy_h, dy_w, dy_cdhw, dy_dhw, dy_hw, dx_d, dx_h, dx_w, dx_ncdhw, dx_cdhw, dx_dhw, dx_hw, d_scale,
    h_scale, w_scale, dx);
  return;
}

template CUDA_LIB_EXPORT void CalUpsampleNearest3DGrad<half>(const half *dy, const size_t n, const size_t c,
                                                             const size_t dy_d, const size_t dy_h, const size_t dy_w,
                                                             const size_t dx_d, const size_t dx_h, const size_t dx_w,
                                                             const float d_scale, const float h_scale,
                                                             const float w_scale, half *dx, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalUpsampleNearest3DGrad<float>(const float *dy, const size_t n, const size_t c,
                                                              const size_t dy_d, const size_t dy_h, const size_t dy_w,
                                                              const size_t dx_d, const size_t dx_h, const size_t dx_w,
                                                              const float d_scale, const float h_scale,
                                                              const float w_scale, float *dx, cudaStream_t cuda_stream);
