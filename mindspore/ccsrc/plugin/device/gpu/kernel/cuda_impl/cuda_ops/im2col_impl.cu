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
__global__ void Im2ColKernel(const int64_t output_size, const int64_t x_channel, const int64_t x_height,
                             const int64_t x_width, const int64_t y_channel, const int64_t y_height,
                             const int64_t y_width, const int64_t kernel_height, const int64_t kernel_width,
                             const int64_t stride_height, const int64_t stride_width, const int64_t dilation_height,
                             const int64_t dilation_width, const int64_t pad_height, const int64_t pad_width,
                             const int64_t inner_size_y, const int64_t inner_size_x, T *x, T *y) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < output_size; i += blockDim.x * gridDim.x) {
    const int64_t batch_idx = i / inner_size_y, index = i % inner_size_y;
    const int64_t w_col = index % y_width, idx = index / y_width;
    const int64_t h_col = idx % y_height, c_col = idx / y_height;
    const int64_t w_offset = c_col % kernel_width;
    const int64_t h_offset = (c_col / kernel_width) % kernel_height;
    const int64_t c_im = c_col / kernel_height / kernel_width;
    const int64_t h_im = h_col * stride_height - pad_height + h_offset * dilation_height;
    const int64_t w_im = w_col * stride_width - pad_width + w_offset * dilation_width;
    y[batch_idx * inner_size_y + (c_col * y_height + h_col) * y_width + w_col] =
      (h_im >= 0 && w_im >= 0 && h_im < x_height && w_im < x_width)
        ? x[batch_idx * inner_size_x + (c_im * x_height + h_im) * x_width + w_im]
        : static_cast<T>(0);
  }
}

template <typename T>
void CudaIm2Col(const int64_t output_size, const int64_t x_channel, const int64_t x_height, const int64_t x_width,
                const int64_t y_channel, const int64_t y_height, const int64_t y_width, const int64_t kernel_height,
                const int64_t kernel_width, const int64_t stride_height, const int64_t stride_width,
                const int64_t dilation_height, const int64_t dilation_width, const int64_t pad_height,
                const int64_t pad_width, T *x, T *y, const uint32_t device_id, cudaStream_t stream) {
  const int64_t inner_size_y = y_channel * y_height * y_width;
  const int64_t inner_size_x = x_channel * x_height * x_width;
  cudaMemset(static_cast<void *>(y), 0, static_cast<size_t>(output_size) * sizeof(T));
  Im2ColKernel<T><<<CUDA_BLOCKS(device_id, output_size), CUDA_THREADS(device_id), 0, stream>>>(
    output_size, x_channel, x_height, x_width, y_channel, y_height, y_width, kernel_height, kernel_width, stride_height,
    stride_width, dilation_height, dilation_width, pad_height, pad_width, inner_size_y, inner_size_x, x, y);
}

template CUDA_LIB_EXPORT void CudaIm2Col(const int64_t output_size, const int64_t x_channel, const int64_t x_height,
                                         const int64_t x_width, const int64_t y_channel, const int64_t y_height,
                                         const int64_t y_width, const int64_t kernel_height, const int64_t kernel_width,
                                         const int64_t stride_height, const int64_t stride_width,
                                         const int64_t dilation_height, const int64_t dilation_width,
                                         const int64_t pad_height, const int64_t pad_width, int8_t *x, int8_t *y,
                                         const uint32_t device_id, cudaStream_t stream);

template CUDA_LIB_EXPORT void CudaIm2Col(const int64_t output_size, const int64_t x_channel, const int64_t x_height,
                                         const int64_t x_width, const int64_t y_channel, const int64_t y_height,
                                         const int64_t y_width, const int64_t kernel_height, const int64_t kernel_width,
                                         const int64_t stride_height, const int64_t stride_width,
                                         const int64_t dilation_height, const int64_t dilation_width,
                                         const int64_t pad_height, const int64_t pad_width, int16_t *x, int16_t *y,
                                         const uint32_t device_id, cudaStream_t stream);

template CUDA_LIB_EXPORT void CudaIm2Col(const int64_t output_size, const int64_t x_channel, const int64_t x_height,
                                         const int64_t x_width, const int64_t y_channel, const int64_t y_height,
                                         const int64_t y_width, const int64_t kernel_height, const int64_t kernel_width,
                                         const int64_t stride_height, const int64_t stride_width,
                                         const int64_t dilation_height, const int64_t dilation_width,
                                         const int64_t pad_height, const int64_t pad_width, int32_t *x, int32_t *y,
                                         const uint32_t device_id, cudaStream_t stream);

template CUDA_LIB_EXPORT void CudaIm2Col(const int64_t output_size, const int64_t x_channel, const int64_t x_height,
                                         const int64_t x_width, const int64_t y_channel, const int64_t y_height,
                                         const int64_t y_width, const int64_t kernel_height, const int64_t kernel_width,
                                         const int64_t stride_height, const int64_t stride_width,
                                         const int64_t dilation_height, const int64_t dilation_width,
                                         const int64_t pad_height, const int64_t pad_width, int64_t *x, int64_t *y,
                                         const uint32_t device_id, cudaStream_t stream);

template CUDA_LIB_EXPORT void CudaIm2Col(const int64_t output_size, const int64_t x_channel, const int64_t x_height,
                                         const int64_t x_width, const int64_t y_channel, const int64_t y_height,
                                         const int64_t y_width, const int64_t kernel_height, const int64_t kernel_width,
                                         const int64_t stride_height, const int64_t stride_width,
                                         const int64_t dilation_height, const int64_t dilation_width,
                                         const int64_t pad_height, const int64_t pad_width, uint8_t *x, uint8_t *y,
                                         const uint32_t device_id, cudaStream_t stream);

template CUDA_LIB_EXPORT void CudaIm2Col(const int64_t output_size, const int64_t x_channel, const int64_t x_height,
                                         const int64_t x_width, const int64_t y_channel, const int64_t y_height,
                                         const int64_t y_width, const int64_t kernel_height, const int64_t kernel_width,
                                         const int64_t stride_height, const int64_t stride_width,
                                         const int64_t dilation_height, const int64_t dilation_width,
                                         const int64_t pad_height, const int64_t pad_width, uint16_t *x, uint16_t *y,
                                         const uint32_t device_id, cudaStream_t stream);

template CUDA_LIB_EXPORT void CudaIm2Col(const int64_t output_size, const int64_t x_channel, const int64_t x_height,
                                         const int64_t x_width, const int64_t y_channel, const int64_t y_height,
                                         const int64_t y_width, const int64_t kernel_height, const int64_t kernel_width,
                                         const int64_t stride_height, const int64_t stride_width,
                                         const int64_t dilation_height, const int64_t dilation_width,
                                         const int64_t pad_height, const int64_t pad_width, uint32_t *x, uint32_t *y,
                                         const uint32_t device_id, cudaStream_t stream);

template CUDA_LIB_EXPORT void CudaIm2Col(const int64_t output_size, const int64_t x_channel, const int64_t x_height,
                                         const int64_t x_width, const int64_t y_channel, const int64_t y_height,
                                         const int64_t y_width, const int64_t kernel_height, const int64_t kernel_width,
                                         const int64_t stride_height, const int64_t stride_width,
                                         const int64_t dilation_height, const int64_t dilation_width,
                                         const int64_t pad_height, const int64_t pad_width, uint64_t *x, uint64_t *y,
                                         const uint32_t device_id, cudaStream_t stream);

template CUDA_LIB_EXPORT void CudaIm2Col(const int64_t output_size, const int64_t x_channel, const int64_t x_height,
                                         const int64_t x_width, const int64_t y_channel, const int64_t y_height,
                                         const int64_t y_width, const int64_t kernel_height, const int64_t kernel_width,
                                         const int64_t stride_height, const int64_t stride_width,
                                         const int64_t dilation_height, const int64_t dilation_width,
                                         const int64_t pad_height, const int64_t pad_width, half *x, half *y,
                                         const uint32_t device_id, cudaStream_t stream);

template CUDA_LIB_EXPORT void CudaIm2Col(const int64_t output_size, const int64_t x_channel, const int64_t x_height,
                                         const int64_t x_width, const int64_t y_channel, const int64_t y_height,
                                         const int64_t y_width, const int64_t kernel_height, const int64_t kernel_width,
                                         const int64_t stride_height, const int64_t stride_width,
                                         const int64_t dilation_height, const int64_t dilation_width,
                                         const int64_t pad_height, const int64_t pad_width, float *x, float *y,
                                         const uint32_t device_id, cudaStream_t stream);

template CUDA_LIB_EXPORT void CudaIm2Col(const int64_t output_size, const int64_t x_channel, const int64_t x_height,
                                         const int64_t x_width, const int64_t y_channel, const int64_t y_height,
                                         const int64_t y_width, const int64_t kernel_height, const int64_t kernel_width,
                                         const int64_t stride_height, const int64_t stride_width,
                                         const int64_t dilation_height, const int64_t dilation_width,
                                         const int64_t pad_height, const int64_t pad_width, double *x, double *y,
                                         const uint32_t device_id, cudaStream_t stream);

template CUDA_LIB_EXPORT void CudaIm2Col(const int64_t output_size, const int64_t x_channel, const int64_t x_height,
                                         const int64_t x_width, const int64_t y_channel, const int64_t y_height,
                                         const int64_t y_width, const int64_t kernel_height, const int64_t kernel_width,
                                         const int64_t stride_height, const int64_t stride_width,
                                         const int64_t dilation_height, const int64_t dilation_width,
                                         const int64_t pad_height, const int64_t pad_width, Complex64 *x, Complex64 *y,
                                         const uint32_t device_id, cudaStream_t stream);

template CUDA_LIB_EXPORT void CudaIm2Col(const int64_t output_size, const int64_t x_channel, const int64_t x_height,
                                         const int64_t x_width, const int64_t y_channel, const int64_t y_height,
                                         const int64_t y_width, const int64_t kernel_height, const int64_t kernel_width,
                                         const int64_t stride_height, const int64_t stride_width,
                                         const int64_t dilation_height, const int64_t dilation_width,
                                         const int64_t pad_height, const int64_t pad_width, Complex128 *x,
                                         Complex128 *y, const uint32_t device_id, cudaStream_t stream);
