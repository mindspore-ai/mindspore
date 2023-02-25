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

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/im2col_impl.cuh"
#include <iostream>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/util.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/complex.h"
template <typename T>
using Complex = mindspore::utils::Complex<T>;
using Complex64 = Complex<float>;
using Complex128 = Complex<double>;

template <typename T>
__global__ void Im2ColKernel(const int64_t n, T *data_x, T *data_y, const int64_t inner_size_x,
                             const int64_t inner_size_y, const int64_t x_height, const int64_t x_width,
                             const int64_t kernel_height, const int64_t kernel_width, const int64_t pad_height,
                             const int64_t pad_width, const int64_t stride_height, const int64_t stride_width,
                             const int64_t dilation_height, const int64_t dilation_width, const int64_t y_height,
                             const int64_t y_width, const int64_t batches) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < n; index += blockDim.x * gridDim.x) {
    int64_t w_out = index % y_width, idx = index / y_width;
    int64_t h_out = idx % y_height, channel_in = idx / y_height;
    int64_t channel_out = channel_in * kernel_height * kernel_width;
    int64_t h_in = h_out * stride_height - pad_height;
    int64_t w_in = w_out * stride_width - pad_width;
    for (int batch = 0; batch < batches; ++batch) {
      T *out = data_y + batch * inner_size_y + (channel_out * y_height + h_out) * y_width + w_out;
      T *in = data_x + batch * inner_size_x + (channel_in * x_height + h_in) * x_width + w_in;
      for (int64_t i = 0; i < kernel_height; ++i) {
        for (int64_t j = 0; j < kernel_width; ++j) {
          int64_t h = h_in + i * dilation_height;
          int64_t w = w_in + j * dilation_width;
          *out = (h >= 0 && w >= 0 && h < x_height && w < x_width)
                   ? in[i * dilation_height * x_width + j * dilation_width]
                   : static_cast<T>(0);
          out += y_height * y_width;
        }
      }
    }
  }
}

template <typename T>
void CudaIm2Col(const int64_t batches, const int64_t x_channel, const int64_t x_height, const int64_t x_width,
                const int64_t y_out_plane, const int64_t y_height, const int64_t y_width, const int64_t kernel_height,
                const int64_t kernel_width, const int64_t stride_height, const int64_t stride_width,
                const int64_t dilation_height, const int64_t dilation_width, const int64_t pad_height,
                const int64_t pad_width, T *x, T *y, const uint32_t device_id, cudaStream_t stream) {
  const int64_t inner_size_y = y_out_plane * y_height * y_width;
  const int64_t inner_size_x = x_channel * x_height * x_width;
  const int64_t num_kernels = x_channel * y_height * y_width;
  Im2ColKernel<T><<<CUDA_BLOCKS(device_id, num_kernels), CUDA_THREADS(device_id), 0, stream>>>(
    num_kernels, x, y, inner_size_x, inner_size_y, x_height, x_width, kernel_height, kernel_width, pad_height,
    pad_width, stride_height, stride_width, dilation_height, dilation_width, y_height, y_width, batches);
}

template CUDA_LIB_EXPORT void CudaIm2Col(const int64_t batches, const int64_t x_channel, const int64_t x_height,
                                         const int64_t x_width, const int64_t y_out_plane, const int64_t y_height,
                                         const int64_t y_width, const int64_t kernel_height, const int64_t kernel_width,
                                         const int64_t stride_height, const int64_t stride_width,
                                         const int64_t dilation_height, const int64_t dilation_width,
                                         const int64_t pad_height, const int64_t pad_width, int8_t *x, int8_t *y,
                                         const uint32_t device_id, cudaStream_t stream);

template CUDA_LIB_EXPORT void CudaIm2Col(const int64_t batches, const int64_t x_channel, const int64_t x_height,
                                         const int64_t x_width, const int64_t y_out_plane, const int64_t y_height,
                                         const int64_t y_width, const int64_t kernel_height, const int64_t kernel_width,
                                         const int64_t stride_height, const int64_t stride_width,
                                         const int64_t dilation_height, const int64_t dilation_width,
                                         const int64_t pad_height, const int64_t pad_width, int16_t *x, int16_t *y,
                                         const uint32_t device_id, cudaStream_t stream);

template CUDA_LIB_EXPORT void CudaIm2Col(const int64_t batches, const int64_t x_channel, const int64_t x_height,
                                         const int64_t x_width, const int64_t y_out_plane, const int64_t y_height,
                                         const int64_t y_width, const int64_t kernel_height, const int64_t kernel_width,
                                         const int64_t stride_height, const int64_t stride_width,
                                         const int64_t dilation_height, const int64_t dilation_width,
                                         const int64_t pad_height, const int64_t pad_width, int32_t *x, int32_t *y,
                                         const uint32_t device_id, cudaStream_t stream);

template CUDA_LIB_EXPORT void CudaIm2Col(const int64_t batches, const int64_t x_channel, const int64_t x_height,
                                         const int64_t x_width, const int64_t y_out_plane, const int64_t y_height,
                                         const int64_t y_width, const int64_t kernel_height, const int64_t kernel_width,
                                         const int64_t stride_height, const int64_t stride_width,
                                         const int64_t dilation_height, const int64_t dilation_width,
                                         const int64_t pad_height, const int64_t pad_width, int64_t *x, int64_t *y,
                                         const uint32_t device_id, cudaStream_t stream);

template CUDA_LIB_EXPORT void CudaIm2Col(const int64_t batches, const int64_t x_channel, const int64_t x_height,
                                         const int64_t x_width, const int64_t y_out_plane, const int64_t y_height,
                                         const int64_t y_width, const int64_t kernel_height, const int64_t kernel_width,
                                         const int64_t stride_height, const int64_t stride_width,
                                         const int64_t dilation_height, const int64_t dilation_width,
                                         const int64_t pad_height, const int64_t pad_width, uint8_t *x, uint8_t *y,
                                         const uint32_t device_id, cudaStream_t stream);

template CUDA_LIB_EXPORT void CudaIm2Col(const int64_t batches, const int64_t x_channel, const int64_t x_height,
                                         const int64_t x_width, const int64_t y_out_plane, const int64_t y_height,
                                         const int64_t y_width, const int64_t kernel_height, const int64_t kernel_width,
                                         const int64_t stride_height, const int64_t stride_width,
                                         const int64_t dilation_height, const int64_t dilation_width,
                                         const int64_t pad_height, const int64_t pad_width, uint16_t *x, uint16_t *y,
                                         const uint32_t device_id, cudaStream_t stream);

template CUDA_LIB_EXPORT void CudaIm2Col(const int64_t batches, const int64_t x_channel, const int64_t x_height,
                                         const int64_t x_width, const int64_t y_out_plane, const int64_t y_height,
                                         const int64_t y_width, const int64_t kernel_height, const int64_t kernel_width,
                                         const int64_t stride_height, const int64_t stride_width,
                                         const int64_t dilation_height, const int64_t dilation_width,
                                         const int64_t pad_height, const int64_t pad_width, uint32_t *x, uint32_t *y,
                                         const uint32_t device_id, cudaStream_t stream);

template CUDA_LIB_EXPORT void CudaIm2Col(const int64_t batches, const int64_t x_channel, const int64_t x_height,
                                         const int64_t x_width, const int64_t y_out_plane, const int64_t y_height,
                                         const int64_t y_width, const int64_t kernel_height, const int64_t kernel_width,
                                         const int64_t stride_height, const int64_t stride_width,
                                         const int64_t dilation_height, const int64_t dilation_width,
                                         const int64_t pad_height, const int64_t pad_width, uint64_t *x, uint64_t *y,
                                         const uint32_t device_id, cudaStream_t stream);

template CUDA_LIB_EXPORT void CudaIm2Col(const int64_t batches, const int64_t x_channel, const int64_t x_height,
                                         const int64_t x_width, const int64_t y_out_plane, const int64_t y_height,
                                         const int64_t y_width, const int64_t kernel_height, const int64_t kernel_width,
                                         const int64_t stride_height, const int64_t stride_width,
                                         const int64_t dilation_height, const int64_t dilation_width,
                                         const int64_t pad_height, const int64_t pad_width, half *x, half *y,
                                         const uint32_t device_id, cudaStream_t stream);

template CUDA_LIB_EXPORT void CudaIm2Col(const int64_t batches, const int64_t x_channel, const int64_t x_height,
                                         const int64_t x_width, const int64_t y_out_plane, const int64_t y_height,
                                         const int64_t y_width, const int64_t kernel_height, const int64_t kernel_width,
                                         const int64_t stride_height, const int64_t stride_width,
                                         const int64_t dilation_height, const int64_t dilation_width,
                                         const int64_t pad_height, const int64_t pad_width, float *x, float *y,
                                         const uint32_t device_id, cudaStream_t stream);

template CUDA_LIB_EXPORT void CudaIm2Col(const int64_t batches, const int64_t x_channel, const int64_t x_height,
                                         const int64_t x_width, const int64_t y_out_plane, const int64_t y_height,
                                         const int64_t y_width, const int64_t kernel_height, const int64_t kernel_width,
                                         const int64_t stride_height, const int64_t stride_width,
                                         const int64_t dilation_height, const int64_t dilation_width,
                                         const int64_t pad_height, const int64_t pad_width, double *x, double *y,
                                         const uint32_t device_id, cudaStream_t stream);

template CUDA_LIB_EXPORT void CudaIm2Col(const int64_t batches, const int64_t x_channel, const int64_t x_height,
                                         const int64_t x_width, const int64_t y_out_plane, const int64_t y_height,
                                         const int64_t y_width, const int64_t kernel_height, const int64_t kernel_width,
                                         const int64_t stride_height, const int64_t stride_width,
                                         const int64_t dilation_height, const int64_t dilation_width,
                                         const int64_t pad_height, const int64_t pad_width, Complex64 *x, Complex64 *y,
                                         const uint32_t device_id, cudaStream_t stream);

template CUDA_LIB_EXPORT void CudaIm2Col(const int64_t batches, const int64_t x_channel, const int64_t x_height,
                                         const int64_t x_width, const int64_t y_out_plane, const int64_t y_height,
                                         const int64_t y_width, const int64_t kernel_height, const int64_t kernel_width,
                                         const int64_t stride_height, const int64_t stride_width,
                                         const int64_t dilation_height, const int64_t dilation_width,
                                         const int64_t pad_height, const int64_t pad_width, Complex128 *x,
                                         Complex128 *y, const uint32_t device_id, cudaStream_t stream);
