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

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/avg_pool3d_helper_impl.cuh"
#include "include/cuda_fp16.h"

template <typename T>
__global__ void RealKernelSize(const size_t size, T *kernel, const int64_t kernel_size, const int64_t shape_d,
                               const int64_t shape_h, const int64_t shape_w, const int64_t kernel_d,
                               const int64_t kernel_h, const int64_t kernel_w, const int64_t edge_kernel_d,
                               const int64_t edge_kernel_h, const int64_t edge_kernel_w) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < (size); pos += blockDim.x * gridDim.x) {
    const int64_t d_max = shape_d - 1;
    const int64_t h_max = shape_h - 1;
    const int64_t w_max = shape_w - 1;
    for (int64_t d = 0; d < shape_d; ++d) {
      for (int64_t h = 0; h < shape_h; ++h) {
        for (int64_t w = 0; w < shape_w; ++w) {
          const int64_t valid_d = ((d == d_max) ? edge_kernel_d : kernel_d);
          const int64_t valid_h = ((h == h_max) ? edge_kernel_h : kernel_h);
          const int64_t valid_w = ((w == w_max) ? edge_kernel_w : kernel_w);
          const int64_t cur_kernel_size = valid_d * valid_h * valid_w;
          if (cur_kernel_size != kernel_size) {
            const int64_t index = pos * shape_d * shape_h * shape_w + d * shape_h * shape_w + h * shape_w + w;
            kernel[index] =
              kernel[index] * static_cast<T>(static_cast<float>(cur_kernel_size) / static_cast<float>(kernel_size));
          }
        }
      }
    }
  }
}

template <typename T>
cudaError_t CalRealKernelSize(const std::vector<int64_t> &input_shape, const std::vector<int64_t> &kernel_size,
                              const std::vector<int64_t> &edge_kernel_size, T *kernel, const uint32_t &device_id,
                              cudaStream_t cuda_stream) {
  const int64_t kernel_prod = kernel_size[2] * kernel_size[1] * kernel_size[2];
  const int64_t nc_size = input_shape[0] * input_shape[1];
  RealKernelSize<<<CUDA_BLOCKS(device_id, nc_size), CUDA_THREADS(device_id), 0, cuda_stream>>>(
    nc_size, kernel, kernel_prod, input_shape[2], input_shape[3], input_shape[4], kernel_size[0], kernel_size[1],
    kernel_size[2], edge_kernel_size[0], edge_kernel_size[1], edge_kernel_size[2]);
  CHECK_CUDA_LAUNCH_SUCCESS();
}

template CUDA_LIB_EXPORT cudaError_t CalRealKernelSize<double>(const std::vector<int64_t> &input_shape,
                                                               const std::vector<int64_t> &kernel_size,
                                                               const std::vector<int64_t> &edge_kernel_size,
                                                               double *kernel, const uint32_t &device_id,
                                                               cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalRealKernelSize<float>(const std::vector<int64_t> &input_shape,
                                                              const std::vector<int64_t> &kernel_size,
                                                              const std::vector<int64_t> &edge_kernel_size,
                                                              float *kernel, const uint32_t &device_id,
                                                              cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalRealKernelSize<half>(const std::vector<int64_t> &input_shape,
                                                             const std::vector<int64_t> &kernel_size,
                                                             const std::vector<int64_t> &edge_kernel_size, half *kernel,
                                                             const uint32_t &device_id, cudaStream_t cuda_stream);
