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

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/upsample_nearest_3d_impl.cuh"
#include <algorithm>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/util.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/upsample.cuh"

#define MAX_THREADS 512

template <typename T>
__global__ void UpsampleNearest3dKernel(const int num_kernels, const T *input, const int n, const int c, const int in_d,
                                        const int in_h, const int in_w, const int out_d, const int out_h,
                                        const int out_w, const int in_cdhw, const int out_cdhw, const int in_dhw,
                                        const int out_dhw, const int out_hw, const float d_scale, const float h_scale,
                                        const float w_scale, T *output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < num_kernels; pos += blockDim.x * gridDim.x) {
    const int posc = (pos / out_dhw) % c;
    const int posd = pos / out_hw % out_d;
    const int src_posd = nearest_neighbor_compute_source_index(d_scale, posd, in_d);
    const int posh = pos / out_w % out_h;
    const int src_posh = nearest_neighbor_compute_source_index(h_scale, posh, in_h);
    const int posw = pos % out_w;
    const int src_posw = nearest_neighbor_compute_source_index(w_scale, posw, in_w);
    int src_pos = posc * in_dhw + (src_posd * in_h + src_posh) * in_w + src_posw;
    int dst_pos = pos;
    for (int b = 0; b < n; ++b) {
      output[dst_pos] = input[src_pos];
      src_pos += in_cdhw;
      dst_pos += out_cdhw;
    }
  }
  return;
}

template <typename T>
cudaError_t CalUpsampleNearest3d(const T *input, const int n, const int c, const int in_d, const int in_h,
                                 const int in_w, const int out_d, const int out_h, const int out_w, const float d_scale,
                                 const float h_scale, const float w_scale, T *output, const uint32_t device_id,
                                 cudaStream_t cuda_stream) {
  if (in_d == out_d && in_h == out_h && in_w == out_w) {
    const int num_kernels = out_d * out_h * out_w * n * c;
    CudaMemcpyDeviceToDevice<T, T>
      <<<CUDA_BLOCKS(device_id, num_kernels), CUDA_THREADS(device_id), 0, cuda_stream>>>(num_kernels, input, output);
  } else {
    const int out_hw = out_h * out_w;
    const int out_dhw = out_d * out_hw;
    const int out_cdhw = out_dhw * c;
    const int in_dhw = in_d * in_h * in_w;
    const int in_cdhw = in_dhw * c;
    const int num_kernels = c * out_d * out_h * out_w;
    const int blockSize = std::min(CUDA_THREADS(device_id), static_cast<int>(MAX_THREADS));
    const int gridSize = (num_kernels + blockSize - 1) / blockSize;
    UpsampleNearest3dKernel<<<gridSize, blockSize, 0, cuda_stream>>>(num_kernels, input, n, c, in_d, in_h, in_w, out_d,
                                                                     out_h, out_w, in_cdhw, out_cdhw, in_dhw, out_dhw,
                                                                     out_hw, d_scale, h_scale, w_scale, output);
  }
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t CalUpsampleNearest3d<double>(const double *input, const int n, const int c,
                                                                  const int in_d, const int in_h, const int in_w,
                                                                  const int out_d, const int out_h, const int out_w,
                                                                  const float d_scale, const float h_scale,
                                                                  const float w_scale, double *output,
                                                                  const uint32_t device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalUpsampleNearest3d<float>(const float *input, const int n, const int c,
                                                                 const int in_d, const int in_h, const int in_w,
                                                                 const int out_d, const int out_h, const int out_w,
                                                                 const float d_scale, const float h_scale,
                                                                 const float w_scale, float *output,
                                                                 const uint32_t device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalUpsampleNearest3d<half>(const half *input, const int n, const int c,
                                                                const int in_d, const int in_h, const int in_w,
                                                                const int out_d, const int out_h, const int out_w,
                                                                const float d_scale, const float h_scale,
                                                                const float w_scale, half *output,
                                                                const uint32_t device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalUpsampleNearest3d<uint8_t>(const uint8_t *input, const int n, const int c,
                                                                   const int in_d, const int in_h, const int in_w,
                                                                   const int out_d, const int out_h, const int out_w,
                                                                   const float d_scale, const float h_scale,
                                                                   const float w_scale, uint8_t *output,
                                                                   const uint32_t device_id, cudaStream_t cuda_stream);
