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
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/complex.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/im2col_impl.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/util.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_public/occupancy.h"

template <typename T>
using Complex = mindspore::utils::Complex<T>;
using Complex64 = Complex<float>;
using Complex128 = Complex<double>;

template <typename T>
__global__ void Im2ColKernel(const int n, T *data_x, T *data_y, const int inner_size_x, const int inner_size_y,
                             const int x_height, const int x_width, const int kernel_height, const int kernel_width,
                             const int pad_height, const int pad_width, const int stride_height, const int stride_width,
                             const int dilation_height, const int dilation_width, const int y_height, const int y_width,
                             const int inner_size_c) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < n; index += blockDim.x * gridDim.x) {
    int batch = index / inner_size_c, idx = index % inner_size_c;
    int w_out = idx % y_width, id = idx / y_width;
    int h_out = id % y_height, channel_in = id / y_height;
    int channel_out = channel_in * kernel_height * kernel_width;
    int h_in = h_out * stride_height - pad_height;
    int w_in = w_out * stride_width - pad_width;
    T *out = data_y + batch * inner_size_y + (channel_out * y_height + h_out) * y_width + w_out;
    T *in = data_x + batch * inner_size_x + (channel_in * x_height + h_in) * x_width + w_in;
    for (int i = 0; i < kernel_height; ++i) {
      for (int j = 0; j < kernel_width; ++j) {
        int h = h_in + i * dilation_height;
        int w = w_in + j * dilation_width;
        *out = (h >= 0 && w >= 0 && h < x_height && w < x_width)
                 ? in[i * dilation_height * x_width + j * dilation_width]
                 : static_cast<T>(0);
        out += y_height * y_width;
      }
    }
  }
}

template <typename T>
cudaError_t CudaIm2Col(const int batches, const int x_channel, const int x_height, const int x_width,
                       const int y_out_plane, const int y_height, const int y_width, const int kernel_height,
                       const int kernel_width, const int stride_height, const int stride_width,
                       const int dilation_height, const int dilation_width, const int pad_height, const int pad_width,
                       T *x, T *y, int *const maxBlockSize, const uint32_t device_id, cudaStream_t stream) {
  const int inner_size_y = y_out_plane * y_height * y_width;
  const int inner_size_x = x_channel * x_height * x_width;
  const int inner_size_c = x_channel * y_height * y_width;
  const int num_kernels = batches * inner_size_c;
  if (*maxBlockSize <= static_cast<int>(0)) {
    *maxBlockSize = FetchMaxBlokcSize(Im2ColKernel<T>, 0);
  }
  const int blockSize = std::max(std::min(*maxBlockSize, num_kernels), static_cast<int>(1));
  const int gridSize = (num_kernels + blockSize - 1) / blockSize;
  Im2ColKernel<T><<<gridSize, blockSize, 0, stream>>>(
    num_kernels, x, y, inner_size_x, inner_size_y, x_height, x_width, kernel_height, kernel_width, pad_height,
    pad_width, stride_height, stride_width, dilation_height, dilation_width, y_height, y_width, inner_size_c);
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t CudaIm2Col(const int batches, const int x_channel, const int x_height,
                                                const int x_width, const int y_out_plane, const int y_height,
                                                const int y_width, const int kernel_height, const int kernel_width,
                                                const int stride_height, const int stride_width,
                                                const int dilation_height, const int dilation_width,
                                                const int pad_height, const int pad_width, half *x, half *y,
                                                int *const maxBlockSize, const uint32_t device_id, cudaStream_t stream);

template CUDA_LIB_EXPORT cudaError_t CudaIm2Col(const int batches, const int x_channel, const int x_height,
                                                const int x_width, const int y_out_plane, const int y_height,
                                                const int y_width, const int kernel_height, const int kernel_width,
                                                const int stride_height, const int stride_width,
                                                const int dilation_height, const int dilation_width,
                                                const int pad_height, const int pad_width, float *x, float *y,
                                                int *const maxBlockSize, const uint32_t device_id, cudaStream_t stream);

template CUDA_LIB_EXPORT cudaError_t CudaIm2Col(const int batches, const int x_channel, const int x_height,
                                                const int x_width, const int y_out_plane, const int y_height,
                                                const int y_width, const int kernel_height, const int kernel_width,
                                                const int stride_height, const int stride_width,
                                                const int dilation_height, const int dilation_width,
                                                const int pad_height, const int pad_width, double *x, double *y,
                                                int *const maxBlockSize, const uint32_t device_id, cudaStream_t stream);

template CUDA_LIB_EXPORT cudaError_t CudaIm2Col(const int batches, const int x_channel, const int x_height,
                                                const int x_width, const int y_out_plane, const int y_height,
                                                const int y_width, const int kernel_height, const int kernel_width,
                                                const int stride_height, const int stride_width,
                                                const int dilation_height, const int dilation_width,
                                                const int pad_height, const int pad_width, Complex64 *x, Complex64 *y,
                                                int *const maxBlockSize, const uint32_t device_id, cudaStream_t stream);

template CUDA_LIB_EXPORT cudaError_t CudaIm2Col(const int batches, const int x_channel, const int x_height,
                                                const int x_width, const int y_out_plane, const int y_height,
                                                const int y_width, const int kernel_height, const int kernel_width,
                                                const int stride_height, const int stride_width,
                                                const int dilation_height, const int dilation_width,
                                                const int pad_height, const int pad_width, Complex128 *x, Complex128 *y,
                                                int *const maxBlockSize, const uint32_t device_id, cudaStream_t stream);

template CUDA_LIB_EXPORT cudaError_t CudaIm2Col(const int batches, const int x_channel, const int x_height,
                                                const int x_width, const int y_out_plane, const int y_height,
                                                const int y_width, const int kernel_height, const int kernel_width,
                                                const int stride_height, const int stride_width,
                                                const int dilation_height, const int dilation_width,
                                                const int pad_height, const int pad_width, uint8_t *x, uint8_t *y,
                                                int *const maxBlockSize, const uint32_t device_id, cudaStream_t stream);

template CUDA_LIB_EXPORT cudaError_t CudaIm2Col(const int batches, const int x_channel, const int x_height,
                                                const int x_width, const int y_out_plane, const int y_height,
                                                const int y_width, const int kernel_height, const int kernel_width,
                                                const int stride_height, const int stride_width,
                                                const int dilation_height, const int dilation_width,
                                                const int pad_height, const int pad_width, int8_t *x, int8_t *y,
                                                int *const maxBlockSize, const uint32_t device_id, cudaStream_t stream);

template CUDA_LIB_EXPORT cudaError_t CudaIm2Col(const int batches, const int x_channel, const int x_height,
                                                const int x_width, const int y_out_plane, const int y_height,
                                                const int y_width, const int kernel_height, const int kernel_width,
                                                const int stride_height, const int stride_width,
                                                const int dilation_height, const int dilation_width,
                                                const int pad_height, const int pad_width, int16_t *x, int16_t *y,
                                                int *const maxBlockSize, const uint32_t device_id, cudaStream_t stream);

template CUDA_LIB_EXPORT cudaError_t CudaIm2Col(const int batches, const int x_channel, const int x_height,
                                                const int x_width, const int y_out_plane, const int y_height,
                                                const int y_width, const int kernel_height, const int kernel_width,
                                                const int stride_height, const int stride_width,
                                                const int dilation_height, const int dilation_width,
                                                const int pad_height, const int pad_width, int32_t *x, int32_t *y,
                                                int *const maxBlockSize, const uint32_t device_id, cudaStream_t stream);

template CUDA_LIB_EXPORT cudaError_t CudaIm2Col(const int batches, const int x_channel, const int x_height,
                                                const int x_width, const int y_out_plane, const int y_height,
                                                const int y_width, const int kernel_height, const int kernel_width,
                                                const int stride_height, const int stride_width,
                                                const int dilation_height, const int dilation_width,
                                                const int pad_height, const int pad_width, int64_t *x, int64_t *y,
                                                int *const maxBlockSize, const uint32_t device_id, cudaStream_t stream);
