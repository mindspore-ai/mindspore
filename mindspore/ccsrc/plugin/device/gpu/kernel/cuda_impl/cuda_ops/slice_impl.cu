/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>
#include <algorithm>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/slice_impl.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cuda_device_info.h"
#include "include/cuda_fp16.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/complex.h"

template <typename T>
using Complex = mindspore::utils::Complex<T>;

template <typename T>
__global__ void Slice1D(const size_t s1, const size_t l1, const size_t d1, const T *input, T *output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < l1; pos += blockDim.x * gridDim.x) {
    output[pos] = input[pos + s1];
  }
}

template <typename T>
__global__ void Slice2D(const size_t s1, const size_t s2, const size_t l1, const size_t l2, const size_t d1,
                        const size_t d2, const T *input, T *output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < l1 * l2; pos += blockDim.x * gridDim.x) {
    size_t i = pos / l2 % l1;
    size_t j = pos % l2;

    size_t offset = (i + s1) * d2 + (j + s2);
    output[pos] = input[offset];
  }
}

template <typename T>
__global__ void Slice3D(const size_t s1, const size_t s2, const size_t s3, const size_t l1, const size_t l2,
                        const size_t l3, const size_t d1, const size_t d2, const size_t d3, const T *input, T *output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < l1 * l2 * l3; pos += blockDim.x * gridDim.x) {
    size_t i = pos / (l2 * l3) % l1;
    size_t j = pos / l3 % l2;
    size_t k = pos % l3;

    size_t offset = (i + s1) * (d2 * d3) + (j + s2) * d3 + (k + s3);
    output[pos] = input[offset];
  }
}

template <typename T>
__global__ void Slice4D(const size_t s1, const size_t s2, const size_t s3, const size_t s4, const size_t l1,
                        const size_t l2, const size_t l3, const size_t l4, const size_t d1, const size_t d2,
                        const size_t d3, const size_t d4, const T *input, T *output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < l1 * l2 * l3 * l4; pos += blockDim.x * gridDim.x) {
    size_t i = pos / (l2 * l3 * l4) % l1;
    size_t j = pos / (l3 * l4) % l2;
    size_t k = pos / l4 % l3;
    size_t o = pos % l4;

    size_t offset = (i + s1) * (d2 * d3 * d4) + (j + s2) * (d3 * d4) + (k + s3) * d4 + (o + s4);
    output[pos] = input[offset];
  }
}

template <typename T>
__global__ void Slice5D(const size_t s1, const size_t s2, const size_t s3, const size_t s4, const size_t s5,
                        const size_t l1, const size_t l2, const size_t l3, const size_t l4, const size_t l5,
                        const size_t d1, const size_t d2, const size_t d3, const size_t d4, const size_t d5,
                        const T *input, T *output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < l1 * l2 * l3 * l4 * l5;
       pos += blockDim.x * gridDim.x) {
    size_t i = pos / (l2 * l3 * l4 * l5) % l1;
    size_t j = pos / (l3 * l4 * l5) % l2;
    size_t k = pos / (l4 * l5) % l3;
    size_t o = pos / l5 % l4;
    size_t q = pos % l5;

    size_t offset =
      (i + s1) * (d2 * d3 * d4 * d5) + (j + s2) * (d3 * d4 * d5) + (k + s3) * (d4 * d5) + (o + s4) * d5 + (q + s5);
    output[pos] = input[offset];
  }
}

template <typename T>
__global__ void Slice6D(const size_t s1, const size_t s2, const size_t s3, const size_t s4, const size_t s5,
                        const size_t s6, const size_t l1, const size_t l2, const size_t l3, const size_t l4,
                        const size_t l5, const size_t l6, const size_t d1, const size_t d2, const size_t d3,
                        const size_t d4, const size_t d5, const size_t d6, const T *input, T *output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < l1 * l2 * l3 * l4 * l5 * l6;
       pos += blockDim.x * gridDim.x) {
    size_t i = pos / (l2 * l3 * l4 * l5 * l6) % l1;
    size_t j = pos / (l3 * l4 * l5 * l6) % l2;
    size_t k = pos / (l4 * l5 * l6) % l3;
    size_t o = pos / (l5 * l6) % l4;
    size_t q = pos / l6 % l5;
    size_t r = pos % l6;

    size_t offset = (i + s1) * (d2 * d3 * d4 * d5 * d6) + (j + s2) * (d3 * d4 * d5 * d6) + (k + s3) * (d4 * d5 * d6) +
                    (o + s4) * (d5 * d6) + (q + s5) * d6 + (r + s6);
    output[pos] = input[offset];
  }
}

template <typename T>
__global__ void Slice7D(const size_t s1, const size_t s2, const size_t s3, const size_t s4, const size_t s5,
                        const size_t s6, const size_t s7, const size_t l1, const size_t l2, const size_t l3,
                        const size_t l4, const size_t l5, const size_t l6, const size_t l7, const size_t d1,
                        const size_t d2, const size_t d3, const size_t d4, const size_t d5, const size_t d6,
                        const size_t d7, const T *input, T *output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < l1 * l2 * l3 * l4 * l5 * l6 * l7;
       pos += blockDim.x * gridDim.x) {
    size_t i = pos / (l2 * l3 * l4 * l5 * l6 * l7) % l1;
    size_t j = pos / (l3 * l4 * l5 * l6 * l7) % l2;
    size_t k = pos / (l4 * l5 * l6 * l7) % l3;
    size_t o = pos / (l5 * l6 * l7) % l4;
    size_t q = pos / (l6 * l7) % l5;
    size_t r = pos / l7 % l6;
    size_t s = pos % l7;

    size_t offset = (i + s1) * (d2 * d3 * d4 * d5 * d6 * d7) + (j + s2) * (d3 * d4 * d5 * d6 * d7) +
                    (k + s3) * (d4 * d5 * d6 * d7) + (o + s4) * (d5 * d6 * d7) + (q + s5) * (d6 * d7) + (r + s6) * d7 +
                    (s + s7);
    output[pos] = input[offset];
  }
}

template <typename T>
__global__ void Slice4DGrad(const size_t s1, const size_t s2, const size_t s3, const size_t s4, const size_t l1,
                            const size_t l2, const size_t l3, const size_t l4, const size_t d1, const size_t d2,
                            const size_t d3, const size_t d4, const T *dy, T *dx) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < (l1 * l2 * l3 * l4); pos += blockDim.x * gridDim.x) {
    size_t i = pos / (l2 * l3 * l4) % l1;
    size_t j = pos / (l3 * l4) % l2;
    size_t k = pos / l4 % l3;
    size_t o = pos % l4;
    size_t input_idx = (i + s1) * (d2 * d3 * d4) + (j + s2) * (d3 * d4) + (k + s3) * d4 + (o + s4);
    dx[input_idx] = dy[pos];
  }
}

template <typename T>
__global__ void Slice7DGrad(const size_t s1, const size_t s2, const size_t s3, const size_t s4, const size_t s5,
                            const size_t s6, const size_t s7, const size_t l1, const size_t l2, const size_t l3,
                            const size_t l4, const size_t l5, const size_t l6, const size_t l7, const size_t d1,
                            const size_t d2, const size_t d3, const size_t d4, const size_t d5, const size_t d6,
                            const size_t d7, const T *dy, T *dx) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < (l1 * l2 * l3 * l4 * l5 * l6 * l7);
       pos += blockDim.x * gridDim.x) {
    size_t i = pos / (l2 * l3 * l4 * l5 * l6 * l7) % l1;
    size_t j = pos / (l3 * l4 * l5 * l6 * l7) % l2;
    size_t k = pos / (l4 * l5 * l6 * l7) % l3;
    size_t o = pos / (l5 * l6 * l7) % l4;
    size_t q = pos / (l6 * l7) % l5;
    size_t r = pos / l7 % l6;
    size_t s = pos % l7;
    size_t input_idx = (i + s1) * (d2 * d3 * d4 * d5 * d6 * d7) + (j + s2) * (d3 * d4 * d5 * d6 * d7) +
                       (k + s3) * (d4 * d5 * d6 * d7) + (o + s4) * (d5 * d6 * d7) + (q + s5) * (d6 * d7) +
                       (r + s6) * d7 + (s + s7);
    dx[input_idx] = dy[pos];
  }
}

template <typename T>
__global__ void FillArray(T *addr, const size_t len, const float value) {
  T value_ = static_cast<T>(value);
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < len; pos += blockDim.x * gridDim.x) {
    addr[pos] = value_;
  }
  return;
}
template <typename T>
cudaError_t FillDeviceArray(const size_t input_size, T *addr, const float value, cudaStream_t cuda_stream) {
  FillArray<<<GET_BLOCKS(input_size), GET_THREADS, 0, cuda_stream>>>(addr, input_size, value);
  return GetCudaStatus();
}
template <typename T>
cudaError_t Slice1DKernel(const size_t s1, const size_t l1, const size_t d1, const T *input, T *output,
                          const uint32_t &device_id, cudaStream_t stream) {
  Slice1D<<<CUDA_BLOCKS(device_id, l1), CUDA_THREADS(device_id), 0, stream>>>(s1, l1, d1, input, output);
  return GetCudaStatus();
}
template <typename T>
cudaError_t Slice2DKernel(const size_t s1, const size_t s2, const size_t l1, const size_t l2, const size_t d1,
                          const size_t d2, const T *input, T *output, const uint32_t &device_id, cudaStream_t stream) {
  Slice2D<<<CUDA_BLOCKS(device_id, l1 * l2), CUDA_THREADS(device_id), 0, stream>>>(s1, s2, l1, l2, d1, d2, input,
                                                                                   output);
  return GetCudaStatus();
}
template <typename T>
cudaError_t Slice3DKernel(const size_t s1, const size_t s2, const size_t s3, const size_t l1, const size_t l2,
                          const size_t l3, const size_t d1, const size_t d2, const size_t d3, const T *input, T *output,
                          const uint32_t &device_id, cudaStream_t stream) {
  Slice3D<<<CUDA_BLOCKS(device_id, l1 * l2 * l3), CUDA_THREADS(device_id), 0, stream>>>(s1, s2, s3, l1, l2, l3, d1, d2,
                                                                                        d3, input, output);
  return GetCudaStatus();
}
template <typename T>
cudaError_t Slice4DKernel(const size_t s1, const size_t s2, const size_t s3, const size_t s4, const size_t l1,
                          const size_t l2, const size_t l3, const size_t l4, const size_t d1, const size_t d2,
                          const size_t d3, const size_t d4, const T *input, T *output, const uint32_t &device_id,
                          cudaStream_t stream) {
  Slice4D<<<CUDA_BLOCKS(device_id, l1 * l2 * l3 * l4), CUDA_THREADS(device_id), 0, stream>>>(
    s1, s2, s3, s4, l1, l2, l3, l4, d1, d2, d3, d4, input, output);
  return GetCudaStatus();
}
template <typename T>
cudaError_t Slice5DKernel(const size_t s1, const size_t s2, const size_t s3, const size_t s4, const size_t s5,
                          const size_t l1, const size_t l2, const size_t l3, const size_t l4, const size_t l5,
                          const size_t d1, const size_t d2, const size_t d3, const size_t d4, const size_t d5,
                          const T *input, T *output, const uint32_t &device_id, cudaStream_t stream) {
  Slice5D<<<CUDA_BLOCKS(device_id, l1 * l2 * l3 * l4 * l5), CUDA_THREADS(device_id), 0, stream>>>(
    s1, s2, s3, s4, s5, l1, l2, l3, l4, l5, d1, d2, d3, d4, d5, input, output);
  return GetCudaStatus();
}
template <typename T>
cudaError_t Slice6DKernel(const size_t s1, const size_t s2, const size_t s3, const size_t s4, const size_t s5,
                          const size_t s6, const size_t l1, const size_t l2, const size_t l3, const size_t l4,
                          const size_t l5, const size_t l6, const size_t d1, const size_t d2, const size_t d3,
                          const size_t d4, const size_t d5, const size_t d6, const T *input, T *output,
                          const uint32_t &device_id, cudaStream_t stream) {
  Slice6D<<<CUDA_BLOCKS(device_id, l1 * l2 * l3 * l4 * l5 * l6), CUDA_THREADS(device_id), 0, stream>>>(
    s1, s2, s3, s4, s5, s6, l1, l2, l3, l4, l5, l6, d1, d2, d3, d4, d5, d6, input, output);
  return GetCudaStatus();
}
template <typename T>
cudaError_t Slice7DKernel(const size_t s1, const size_t s2, const size_t s3, const size_t s4, const size_t s5,
                          const size_t s6, const size_t s7, const size_t l1, const size_t l2, const size_t l3,
                          const size_t l4, const size_t l5, const size_t l6, const size_t l7, const size_t d1,
                          const size_t d2, const size_t d3, const size_t d4, const size_t d5, const size_t d6,
                          const size_t d7, const T *input, T *output, const uint32_t &device_id, cudaStream_t stream) {
  Slice7D<<<CUDA_BLOCKS(device_id, l1 * l2 * l3 * l4 * l5 * l6 * l7), CUDA_THREADS(device_id), 0, stream>>>(
    s1, s2, s3, s4, s5, s6, s7, l1, l2, l3, l4, l5, l6, l7, d1, d2, d3, d4, d5, d6, d7, input, output);
  return GetCudaStatus();
}
template <typename T>
cudaError_t CalSlice4DGrad(const size_t s1, const size_t s2, const size_t s3, const size_t s4, const size_t l1,
                           const size_t l2, const size_t l3, const size_t l4, const size_t d1, const size_t d2,
                           const size_t d3, const size_t d4, const T *dy, T *dx, cudaStream_t stream) {
  Slice4DGrad<<<GET_BLOCKS(l1 * l2 * l3 * l4), GET_THREADS, 0, stream>>>(s1, s2, s3, s4, l1, l2, l3, l4, d1, d2, d3, d4,
                                                                         dy, dx);
  return GetCudaStatus();
}

template <typename T>
cudaError_t CalSlice7DGrad(const size_t s1, const size_t s2, const size_t s3, const size_t s4, const size_t s5,
                           const size_t s6, const size_t s7, const size_t l1, const size_t l2, const size_t l3,
                           const size_t l4, const size_t l5, const size_t l6, const size_t l7, const size_t d1,
                           const size_t d2, const size_t d3, const size_t d4, const size_t d5, const size_t d6,
                           const size_t d7, const T *dy, T *dx, cudaStream_t stream) {
  Slice7DGrad<<<GET_BLOCKS(l1 * l2 * l3 * l4 * l5 * l6 * l7), GET_THREADS, 0, stream>>>(
    s1, s2, s3, s4, s5, s6, s7, l1, l2, l3, l4, l5, l6, l7, d1, d2, d3, d4, d5, d6, d7, dy, dx);
  return GetCudaStatus();
}

const size_t DIM_SIZE = 8;
struct SliceInfo {
  size_t input_stride[DIM_SIZE];
  size_t output_stride[DIM_SIZE];
  size_t begin[DIM_SIZE];
  size_t strides[DIM_SIZE];
};

template <typename T>
__global__ void StridedSliceKernel(size_t output_num, size_t dim_size, SliceInfo sliceInfo, const T *input, T *output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < output_num; pos += blockDim.x * gridDim.x) {
    int64_t cur_out_idx = 0;
    size_t cur_pos = pos;
    size_t input_idx = 0;
    for (int idx = 0; idx < dim_size; idx++) {
      cur_out_idx = cur_pos / sliceInfo.output_stride[idx];
      cur_pos -= cur_out_idx * sliceInfo.output_stride[idx];
      input_idx += (cur_out_idx * sliceInfo.strides[idx] + sliceInfo.begin[idx]) * sliceInfo.input_stride[idx];
    }
    output[pos] = input[input_idx];
  }
}

template <typename T>
cudaError_t StridedSlice(const std::vector<size_t> &input_shape, const std::vector<int64_t> &begin,
                         const std::vector<int64_t> &strides, const std::vector<size_t> &output_shape, const T *input,
                         T *output, cudaStream_t cuda_stream) {
  auto dim_size = DIM_SIZE;

  SliceInfo sliceInfo;
  sliceInfo.input_stride[dim_size - 1] = 1;
  sliceInfo.output_stride[dim_size - 1] = 1;
  for (int i = dim_size - 2; i >= 0; i--) {
    sliceInfo.input_stride[i] = sliceInfo.input_stride[i + 1] * input_shape[i + 1];
  }
  for (int i = dim_size - 2; i >= 0; i--) {
    sliceInfo.output_stride[i] = sliceInfo.output_stride[i + 1] * output_shape[i + 1];
  }
  size_t output_num = 1;
  for (size_t i = 0; i < output_shape.size(); i++) {
    output_num *= output_shape[i];
  }
  for (size_t idx = 0; idx < dim_size; ++idx) {
    sliceInfo.input_stride[idx] = (input_shape[idx] == 1) ? 0 : sliceInfo.input_stride[idx];
  }
  for (size_t idx = 0; idx < dim_size; ++idx) {
    sliceInfo.begin[idx] = begin[idx];
  }
  for (size_t idx = 0; idx < dim_size; ++idx) {
    sliceInfo.strides[idx] = strides[idx];
  }
  StridedSliceKernel<<<GET_BLOCKS(output_num), GET_THREADS, 0, cuda_stream>>>(output_num, dim_size, sliceInfo, input,
                                                                              output);
  return GetCudaStatus();
}

template <typename T>
__global__ void StridedSliceGradKernel(size_t output_num, size_t dim_size, SliceInfo sliceInfo, const T *dy, T *dx) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < output_num; pos += blockDim.x * gridDim.x) {
    int64_t cur_out_idx = 0;
    size_t cur_pos = pos;
    size_t input_idx = 0;
    for (int idx = 0; idx < dim_size; idx++) {
      cur_out_idx = cur_pos / sliceInfo.output_stride[idx];
      cur_pos -= cur_out_idx * sliceInfo.output_stride[idx];
      input_idx += (cur_out_idx * sliceInfo.strides[idx] + sliceInfo.begin[idx]) * sliceInfo.input_stride[idx];
    }
    dx[input_idx] = dy[pos];
  }
}

template <typename T>
cudaError_t StridedSliceGrad(const std::vector<size_t> &dy_shape, const std::vector<int64_t> &begin,
                             const std::vector<int64_t> &strides, const std::vector<size_t> &dx_shape, const T *dy,
                             T *dx, cudaStream_t cuda_stream) {
  auto dim_size = DIM_SIZE;

  SliceInfo sliceInfo;
  sliceInfo.input_stride[dim_size - 1] = 1;
  sliceInfo.output_stride[dim_size - 1] = 1;
  for (int i = dim_size - 2; i >= 0; i--) {
    sliceInfo.input_stride[i] = sliceInfo.input_stride[i + 1] * dx_shape[i + 1];
  }
  for (int i = dim_size - 2; i >= 0; i--) {
    sliceInfo.output_stride[i] = sliceInfo.output_stride[i + 1] * dy_shape[i + 1];
  }
  size_t output_num = 1;
  for (size_t i = 0; i < dy_shape.size(); i++) {
    output_num *= dy_shape[i];
  }
  for (size_t idx = 0; idx < dim_size; ++idx) {
    sliceInfo.input_stride[idx] = (dx_shape[idx] == 1) ? 0 : sliceInfo.input_stride[idx];
  }
  for (size_t idx = 0; idx < dim_size; ++idx) {
    sliceInfo.begin[idx] = begin[idx];
  }
  for (size_t idx = 0; idx < dim_size; ++idx) {
    sliceInfo.strides[idx] = strides[idx];
  }
  StridedSliceGradKernel<<<GET_BLOCKS(output_num), GET_THREADS, 0, cuda_stream>>>(output_num, dim_size, sliceInfo, dy,
                                                                                  dx);
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t Slice1DKernel(const size_t s1, const size_t l1, const size_t d1,
                                                   const Complex<float> *input, Complex<float> *output,
                                                   const uint32_t &device_id, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Slice1DKernel(const size_t s1, const size_t l1, const size_t d1,
                                                   const Complex<double> *input, Complex<double> *output,
                                                   const uint32_t &device_id, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Slice1DKernel(const size_t s1, const size_t l1, const size_t d1,
                                                   const double *input, double *output, const uint32_t &device_id,
                                                   cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Slice1DKernel(const size_t s1, const size_t l1, const size_t d1,
                                                   const float *input, float *output, const uint32_t &device_id,
                                                   cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Slice1DKernel(const size_t s1, const size_t l1, const size_t d1, const half *input,
                                                   half *output, const uint32_t &device_id, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Slice1DKernel(const size_t s1, const size_t l1, const size_t d1, const int *input,
                                                   int *output, const uint32_t &device_id, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Slice1DKernel(const size_t s1, const size_t l1, const size_t d1,
                                                   const int16_t *input, int16_t *output, const uint32_t &device_id,
                                                   cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Slice1DKernel(const size_t s1, const size_t l1, const size_t d1, const char *input,
                                                   char *output, const uint32_t &device_id, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Slice1DKernel(const size_t s1, const size_t l1, const size_t d1,
                                                   const uint64_t *input, uint64_t *output, const uint32_t &device_id,
                                                   cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Slice1DKernel(const size_t s1, const size_t l1, const size_t d1,
                                                   const uint32_t *input, uint32_t *output, const uint32_t &device_id,
                                                   cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Slice1DKernel(const size_t s1, const size_t l1, const size_t d1,
                                                   const uint16_t *input, uint16_t *output, const uint32_t &device_id,
                                                   cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Slice1DKernel(const size_t s1, const size_t l1, const size_t d1,
                                                   const unsigned char *input, unsigned char *output,
                                                   const uint32_t &device_id, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Slice1DKernel(const size_t s1, const size_t l1, const size_t d1,
                                                   const int64_t *input, int64_t *output, const uint32_t &device_id,
                                                   cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Slice1DKernel(const size_t s1, const size_t l1, const size_t d1, const bool *input,
                                                   bool *output, const uint32_t &device_id, cudaStream_t stream);

template CUDA_LIB_EXPORT cudaError_t Slice2DKernel(const size_t s1, const size_t s2, const size_t l1, const size_t l2,
                                                   const size_t d1, const size_t d2, const Complex<float> *input,
                                                   Complex<float> *output, const uint32_t &device_id,
                                                   cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Slice2DKernel(const size_t s1, const size_t s2, const size_t l1, const size_t l2,
                                                   const size_t d1, const size_t d2, const Complex<double> *input,
                                                   Complex<double> *output, const uint32_t &device_id,
                                                   cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Slice2DKernel(const size_t s1, const size_t s2, const size_t l1, const size_t l2,
                                                   const size_t d1, const size_t d2, const double *input,
                                                   double *output, const uint32_t &device_id, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Slice2DKernel(const size_t s1, const size_t s2, const size_t l1, const size_t l2,
                                                   const size_t d1, const size_t d2, const float *input, float *output,
                                                   const uint32_t &device_id, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Slice2DKernel(const size_t s1, const size_t s2, const size_t l1, const size_t l2,
                                                   const size_t d1, const size_t d2, const half *input, half *output,
                                                   const uint32_t &device_id, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Slice2DKernel(const size_t s1, const size_t s2, const size_t l1, const size_t l2,
                                                   const size_t d1, const size_t d2, const int *input, int *output,
                                                   const uint32_t &device_id, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Slice2DKernel(const size_t s1, const size_t s2, const size_t l1, const size_t l2,
                                                   const size_t d1, const size_t d2, const int16_t *input,
                                                   int16_t *output, const uint32_t &device_id, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Slice2DKernel(const size_t s1, const size_t s2, const size_t l1, const size_t l2,
                                                   const size_t d1, const size_t d2, const char *input, char *output,
                                                   const uint32_t &device_id, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Slice2DKernel(const size_t s1, const size_t s2, const size_t l1, const size_t l2,
                                                   const size_t d1, const size_t d2, const uint64_t *input,
                                                   uint64_t *output, const uint32_t &device_id, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Slice2DKernel(const size_t s1, const size_t s2, const size_t l1, const size_t l2,
                                                   const size_t d1, const size_t d2, const uint32_t *input,
                                                   uint32_t *output, const uint32_t &device_id, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Slice2DKernel(const size_t s1, const size_t s2, const size_t l1, const size_t l2,
                                                   const size_t d1, const size_t d2, const uint16_t *input,
                                                   uint16_t *output, const uint32_t &device_id, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Slice2DKernel(const size_t s1, const size_t s2, const size_t l1, const size_t l2,
                                                   const size_t d1, const size_t d2, const unsigned char *input,
                                                   unsigned char *output, const uint32_t &device_id,
                                                   cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Slice2DKernel(const size_t s1, const size_t s2, const size_t l1, const size_t l2,
                                                   const size_t d1, const size_t d2, const int64_t *input,
                                                   int64_t *output, const uint32_t &device_id, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Slice2DKernel(const size_t s1, const size_t s2, const size_t l1, const size_t l2,
                                                   const size_t d1, const size_t d2, const bool *input, bool *output,
                                                   const uint32_t &device_id, cudaStream_t stream);

template CUDA_LIB_EXPORT cudaError_t Slice3DKernel(const size_t s1, const size_t s2, const size_t s3, const size_t l1,
                                                   const size_t l2, const size_t l3, const size_t d1, const size_t d2,
                                                   const size_t d3, const Complex<float> *input, Complex<float> *output,
                                                   const uint32_t &device_id, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Slice3DKernel(const size_t s1, const size_t s2, const size_t s3, const size_t l1,
                                                   const size_t l2, const size_t l3, const size_t d1, const size_t d2,
                                                   const size_t d3, const Complex<double> *input,
                                                   Complex<double> *output, const uint32_t &device_id,
                                                   cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Slice3DKernel(const size_t s1, const size_t s2, const size_t s3, const size_t l1,
                                                   const size_t l2, const size_t l3, const size_t d1, const size_t d2,
                                                   const size_t d3, const double *input, double *output,
                                                   const uint32_t &device_id, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Slice3DKernel(const size_t s1, const size_t s2, const size_t s3, const size_t l1,
                                                   const size_t l2, const size_t l3, const size_t d1, const size_t d2,
                                                   const size_t d3, const float *input, float *output,
                                                   const uint32_t &device_id, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Slice3DKernel(const size_t s1, const size_t s2, const size_t s3, const size_t l1,
                                                   const size_t l2, const size_t l3, const size_t d1, const size_t d2,
                                                   const size_t d3, const half *input, half *output,
                                                   const uint32_t &device_id, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Slice3DKernel(const size_t s1, const size_t s2, const size_t s3, const size_t l1,
                                                   const size_t l2, const size_t l3, const size_t d1, const size_t d2,
                                                   const size_t d3, const int *input, int *output,
                                                   const uint32_t &device_id, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Slice3DKernel(const size_t s1, const size_t s2, const size_t s3, const size_t l1,
                                                   const size_t l2, const size_t l3, const size_t d1, const size_t d2,
                                                   const size_t d3, const int16_t *input, int16_t *output,
                                                   const uint32_t &device_id, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Slice3DKernel(const size_t s1, const size_t s2, const size_t s3, const size_t l1,
                                                   const size_t l2, const size_t l3, const size_t d1, const size_t d2,
                                                   const size_t d3, const char *input, char *output,
                                                   const uint32_t &device_id, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Slice3DKernel(const size_t s1, const size_t s2, const size_t s3, const size_t l1,
                                                   const size_t l2, const size_t l3, const size_t d1, const size_t d2,
                                                   const size_t d3, const uint64_t *input, uint64_t *output,
                                                   const uint32_t &device_id, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Slice3DKernel(const size_t s1, const size_t s2, const size_t s3, const size_t l1,
                                                   const size_t l2, const size_t l3, const size_t d1, const size_t d2,
                                                   const size_t d3, const uint32_t *input, uint32_t *output,
                                                   const uint32_t &device_id, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Slice3DKernel(const size_t s1, const size_t s2, const size_t s3, const size_t l1,
                                                   const size_t l2, const size_t l3, const size_t d1, const size_t d2,
                                                   const size_t d3, const uint16_t *input, uint16_t *output,
                                                   const uint32_t &device_id, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Slice3DKernel(const size_t s1, const size_t s2, const size_t s3, const size_t l1,
                                                   const size_t l2, const size_t l3, const size_t d1, const size_t d2,
                                                   const size_t d3, const unsigned char *input, unsigned char *output,
                                                   const uint32_t &device_id, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Slice3DKernel(const size_t s1, const size_t s2, const size_t s3, const size_t l1,
                                                   const size_t l2, const size_t l3, const size_t d1, const size_t d2,
                                                   const size_t d3, const int64_t *input, int64_t *output,
                                                   const uint32_t &device_id, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Slice3DKernel(const size_t s1, const size_t s2, const size_t s3, const size_t l1,
                                                   const size_t l2, const size_t l3, const size_t d1, const size_t d2,
                                                   const size_t d3, const bool *input, bool *output,
                                                   const uint32_t &device_id, cudaStream_t stream);

template CUDA_LIB_EXPORT cudaError_t Slice4DKernel(const size_t s1, const size_t s2, const size_t s3, const size_t s4,
                                                   const size_t l1, const size_t l2, const size_t l3, const size_t l4,
                                                   const size_t d1, const size_t d2, const size_t d3, const size_t d4,
                                                   const Complex<float> *input, Complex<float> *output,
                                                   const uint32_t &device_id, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Slice4DKernel(const size_t s1, const size_t s2, const size_t s3, const size_t s4,
                                                   const size_t l1, const size_t l2, const size_t l3, const size_t l4,
                                                   const size_t d1, const size_t d2, const size_t d3, const size_t d4,
                                                   const Complex<double> *input, Complex<double> *output,
                                                   const uint32_t &device_id, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Slice4DKernel(const size_t s1, const size_t s2, const size_t s3, const size_t s4,
                                                   const size_t l1, const size_t l2, const size_t l3, const size_t l4,
                                                   const size_t d1, const size_t d2, const size_t d3, const size_t d4,
                                                   const double *input, double *output, const uint32_t &device_id,
                                                   cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Slice4DKernel(const size_t s1, const size_t s2, const size_t s3, const size_t s4,
                                                   const size_t l1, const size_t l2, const size_t l3, const size_t l4,
                                                   const size_t d1, const size_t d2, const size_t d3, const size_t d4,
                                                   const float *input, float *output, const uint32_t &device_id,
                                                   cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Slice4DKernel(const size_t s1, const size_t s2, const size_t s3, const size_t s4,
                                                   const size_t l1, const size_t l2, const size_t l3, const size_t l4,
                                                   const size_t d1, const size_t d2, const size_t d3, const size_t d4,
                                                   const half *input, half *output, const uint32_t &device_id,
                                                   cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Slice4DKernel(const size_t s1, const size_t s2, const size_t s3, const size_t s4,
                                                   const size_t l1, const size_t l2, const size_t l3, const size_t l4,
                                                   const size_t d1, const size_t d2, const size_t d3, const size_t d4,
                                                   const int *input, int *output, const uint32_t &device_id,
                                                   cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Slice4DKernel(const size_t s1, const size_t s2, const size_t s3, const size_t s4,
                                                   const size_t l1, const size_t l2, const size_t l3, const size_t l4,
                                                   const size_t d1, const size_t d2, const size_t d3, const size_t d4,
                                                   const int16_t *input, int16_t *output, const uint32_t &device_id,
                                                   cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Slice4DKernel(const size_t s1, const size_t s2, const size_t s3, const size_t s4,
                                                   const size_t l1, const size_t l2, const size_t l3, const size_t l4,
                                                   const size_t d1, const size_t d2, const size_t d3, const size_t d4,
                                                   const char *input, char *output, const uint32_t &device_id,
                                                   cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Slice4DKernel(const size_t s1, const size_t s2, const size_t s3, const size_t s4,
                                                   const size_t l1, const size_t l2, const size_t l3, const size_t l4,
                                                   const size_t d1, const size_t d2, const size_t d3, const size_t d4,
                                                   const uint64_t *input, uint64_t *output, const uint32_t &device_id,
                                                   cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Slice4DKernel(const size_t s1, const size_t s2, const size_t s3, const size_t s4,
                                                   const size_t l1, const size_t l2, const size_t l3, const size_t l4,
                                                   const size_t d1, const size_t d2, const size_t d3, const size_t d4,
                                                   const uint32_t *input, uint32_t *output, const uint32_t &device_id,
                                                   cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Slice4DKernel(const size_t s1, const size_t s2, const size_t s3, const size_t s4,
                                                   const size_t l1, const size_t l2, const size_t l3, const size_t l4,
                                                   const size_t d1, const size_t d2, const size_t d3, const size_t d4,
                                                   const uint16_t *input, uint16_t *output, const uint32_t &device_id,
                                                   cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Slice4DKernel(const size_t s1, const size_t s2, const size_t s3, const size_t s4,
                                                   const size_t l1, const size_t l2, const size_t l3, const size_t l4,
                                                   const size_t d1, const size_t d2, const size_t d3, const size_t d4,
                                                   const unsigned char *input, unsigned char *output,
                                                   const uint32_t &device_id, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Slice4DKernel(const size_t s1, const size_t s2, const size_t s3, const size_t s4,
                                                   const size_t l1, const size_t l2, const size_t l3, const size_t l4,
                                                   const size_t d1, const size_t d2, const size_t d3, const size_t d4,
                                                   const int64_t *input, int64_t *output, const uint32_t &device_id,
                                                   cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Slice4DKernel(const size_t s1, const size_t s2, const size_t s3, const size_t s4,
                                                   const size_t l1, const size_t l2, const size_t l3, const size_t l4,
                                                   const size_t d1, const size_t d2, const size_t d3, const size_t d4,
                                                   const bool *input, bool *output, const uint32_t &device_id,
                                                   cudaStream_t stream);

template CUDA_LIB_EXPORT cudaError_t Slice5DKernel(const size_t s1, const size_t s2, const size_t s3, const size_t s4,
                                                   const size_t s5, const size_t l1, const size_t l2, const size_t l3,
                                                   const size_t l4, const size_t l5, const size_t d1, const size_t d2,
                                                   const size_t d3, const size_t d4, const size_t d5,
                                                   const Complex<float> *input, Complex<float> *output,
                                                   const uint32_t &device_id, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Slice5DKernel(const size_t s1, const size_t s2, const size_t s3, const size_t s4,
                                                   const size_t s5, const size_t l1, const size_t l2, const size_t l3,
                                                   const size_t l4, const size_t l5, const size_t d1, const size_t d2,
                                                   const size_t d3, const size_t d4, const size_t d5,
                                                   const Complex<double> *input, Complex<double> *output,
                                                   const uint32_t &device_id, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Slice5DKernel(const size_t s1, const size_t s2, const size_t s3, const size_t s4,
                                                   const size_t s5, const size_t l1, const size_t l2, const size_t l3,
                                                   const size_t l4, const size_t l5, const size_t d1, const size_t d2,
                                                   const size_t d3, const size_t d4, const size_t d5,
                                                   const double *input, double *output, const uint32_t &device_id,
                                                   cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Slice5DKernel(const size_t s1, const size_t s2, const size_t s3, const size_t s4,
                                                   const size_t s5, const size_t l1, const size_t l2, const size_t l3,
                                                   const size_t l4, const size_t l5, const size_t d1, const size_t d2,
                                                   const size_t d3, const size_t d4, const size_t d5,
                                                   const float *input, float *output, const uint32_t &device_id,
                                                   cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Slice5DKernel(const size_t s1, const size_t s2, const size_t s3, const size_t s4,
                                                   const size_t s5, const size_t l1, const size_t l2, const size_t l3,
                                                   const size_t l4, const size_t l5, const size_t d1, const size_t d2,
                                                   const size_t d3, const size_t d4, const size_t d5, const half *input,
                                                   half *output, const uint32_t &device_id, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Slice5DKernel(const size_t s1, const size_t s2, const size_t s3, const size_t s4,
                                                   const size_t s5, const size_t l1, const size_t l2, const size_t l3,
                                                   const size_t l4, const size_t l5, const size_t d1, const size_t d2,
                                                   const size_t d3, const size_t d4, const size_t d5,
                                                   const int64_t *input, int64_t *output, const uint32_t &device_id,
                                                   cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Slice5DKernel(const size_t s1, const size_t s2, const size_t s3, const size_t s4,
                                                   const size_t s5, const size_t l1, const size_t l2, const size_t l3,
                                                   const size_t l4, const size_t l5, const size_t d1, const size_t d2,
                                                   const size_t d3, const size_t d4, const size_t d5, const int *input,
                                                   int *output, const uint32_t &device_id, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Slice5DKernel(const size_t s1, const size_t s2, const size_t s3, const size_t s4,
                                                   const size_t s5, const size_t l1, const size_t l2, const size_t l3,
                                                   const size_t l4, const size_t l5, const size_t d1, const size_t d2,
                                                   const size_t d3, const size_t d4, const size_t d5,
                                                   const int16_t *input, int16_t *output, const uint32_t &device_id,
                                                   cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Slice5DKernel(const size_t s1, const size_t s2, const size_t s3, const size_t s4,
                                                   const size_t s5, const size_t l1, const size_t l2, const size_t l3,
                                                   const size_t l4, const size_t l5, const size_t d1, const size_t d2,
                                                   const size_t d3, const size_t d4, const size_t d5, const char *input,
                                                   char *output, const uint32_t &device_id, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Slice5DKernel(const size_t s1, const size_t s2, const size_t s3, const size_t s4,
                                                   const size_t s5, const size_t l1, const size_t l2, const size_t l3,
                                                   const size_t l4, const size_t l5, const size_t d1, const size_t d2,
                                                   const size_t d3, const size_t d4, const size_t d5,
                                                   const uint64_t *input, uint64_t *output, const uint32_t &device_id,
                                                   cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Slice5DKernel(const size_t s1, const size_t s2, const size_t s3, const size_t s4,
                                                   const size_t s5, const size_t l1, const size_t l2, const size_t l3,
                                                   const size_t l4, const size_t l5, const size_t d1, const size_t d2,
                                                   const size_t d3, const size_t d4, const size_t d5,
                                                   const uint32_t *input, uint32_t *output, const uint32_t &device_id,
                                                   cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Slice5DKernel(const size_t s1, const size_t s2, const size_t s3, const size_t s4,
                                                   const size_t s5, const size_t l1, const size_t l2, const size_t l3,
                                                   const size_t l4, const size_t l5, const size_t d1, const size_t d2,
                                                   const size_t d3, const size_t d4, const size_t d5,
                                                   const uint16_t *input, uint16_t *output, const uint32_t &device_id,
                                                   cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Slice5DKernel(const size_t s1, const size_t s2, const size_t s3, const size_t s4,
                                                   const size_t s5, const size_t l1, const size_t l2, const size_t l3,
                                                   const size_t l4, const size_t l5, const size_t d1, const size_t d2,
                                                   const size_t d3, const size_t d4, const size_t d5,
                                                   const unsigned char *input, unsigned char *output,
                                                   const uint32_t &device_id, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Slice5DKernel(const size_t s1, const size_t s2, const size_t s3, const size_t s4,
                                                   const size_t s5, const size_t l1, const size_t l2, const size_t l3,
                                                   const size_t l4, const size_t l5, const size_t d1, const size_t d2,
                                                   const size_t d3, const size_t d4, const size_t d5, const bool *input,
                                                   bool *output, const uint32_t &device_id, cudaStream_t stream);

template CUDA_LIB_EXPORT cudaError_t Slice6DKernel(const size_t s1, const size_t s2, const size_t s3, const size_t s4,
                                                   const size_t s5, const size_t s6, const size_t l1, const size_t l2,
                                                   const size_t l3, const size_t l4, const size_t l5, const size_t l6,
                                                   const size_t d1, const size_t d2, const size_t d3, const size_t d4,
                                                   const size_t d5, const size_t d6, const Complex<float> *input,
                                                   Complex<float> *output, const uint32_t &device_id,
                                                   cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Slice6DKernel(const size_t s1, const size_t s2, const size_t s3, const size_t s4,
                                                   const size_t s5, const size_t s6, const size_t l1, const size_t l2,
                                                   const size_t l3, const size_t l4, const size_t l5, const size_t l6,
                                                   const size_t d1, const size_t d2, const size_t d3, const size_t d4,
                                                   const size_t d5, const size_t d6, const Complex<double> *input,
                                                   Complex<double> *output, const uint32_t &device_id,
                                                   cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Slice6DKernel(const size_t s1, const size_t s2, const size_t s3, const size_t s4,
                                                   const size_t s5, const size_t s6, const size_t l1, const size_t l2,
                                                   const size_t l3, const size_t l4, const size_t l5, const size_t l6,
                                                   const size_t d1, const size_t d2, const size_t d3, const size_t d4,
                                                   const size_t d5, const size_t d6, const double *input,
                                                   double *output, const uint32_t &device_id, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Slice6DKernel(const size_t s1, const size_t s2, const size_t s3, const size_t s4,
                                                   const size_t s5, const size_t s6, const size_t l1, const size_t l2,
                                                   const size_t l3, const size_t l4, const size_t l5, const size_t l6,
                                                   const size_t d1, const size_t d2, const size_t d3, const size_t d4,
                                                   const size_t d5, const size_t d6, const float *input, float *output,
                                                   const uint32_t &device_id, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Slice6DKernel(const size_t s1, const size_t s2, const size_t s3, const size_t s4,
                                                   const size_t s5, const size_t s6, const size_t l1, const size_t l2,
                                                   const size_t l3, const size_t l4, const size_t l5, const size_t l6,
                                                   const size_t d1, const size_t d2, const size_t d3, const size_t d4,
                                                   const size_t d5, const size_t d6, const half *input, half *output,
                                                   const uint32_t &device_id, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Slice6DKernel(const size_t s1, const size_t s2, const size_t s3, const size_t s4,
                                                   const size_t s5, const size_t s6, const size_t l1, const size_t l2,
                                                   const size_t l3, const size_t l4, const size_t l5, const size_t l6,
                                                   const size_t d1, const size_t d2, const size_t d3, const size_t d4,
                                                   const size_t d5, const size_t d6, const int64_t *input,
                                                   int64_t *output, const uint32_t &device_id, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Slice6DKernel(const size_t s1, const size_t s2, const size_t s3, const size_t s4,
                                                   const size_t s5, const size_t s6, const size_t l1, const size_t l2,
                                                   const size_t l3, const size_t l4, const size_t l5, const size_t l6,
                                                   const size_t d1, const size_t d2, const size_t d3, const size_t d4,
                                                   const size_t d5, const size_t d6, const int *input, int *output,
                                                   const uint32_t &device_id, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Slice6DKernel(const size_t s1, const size_t s2, const size_t s3, const size_t s4,
                                                   const size_t s5, const size_t s6, const size_t l1, const size_t l2,
                                                   const size_t l3, const size_t l4, const size_t l5, const size_t l6,
                                                   const size_t d1, const size_t d2, const size_t d3, const size_t d4,
                                                   const size_t d5, const size_t d6, const int16_t *input,
                                                   int16_t *output, const uint32_t &device_id, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Slice6DKernel(const size_t s1, const size_t s2, const size_t s3, const size_t s4,
                                                   const size_t s5, const size_t s6, const size_t l1, const size_t l2,
                                                   const size_t l3, const size_t l4, const size_t l5, const size_t l6,
                                                   const size_t d1, const size_t d2, const size_t d3, const size_t d4,
                                                   const size_t d5, const size_t d6, const char *input, char *output,
                                                   const uint32_t &device_id, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Slice6DKernel(const size_t s1, const size_t s2, const size_t s3, const size_t s4,
                                                   const size_t s5, const size_t s6, const size_t l1, const size_t l2,
                                                   const size_t l3, const size_t l4, const size_t l5, const size_t l6,
                                                   const size_t d1, const size_t d2, const size_t d3, const size_t d4,
                                                   const size_t d5, const size_t d6, const uint64_t *input,
                                                   uint64_t *output, const uint32_t &device_id, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Slice6DKernel(const size_t s1, const size_t s2, const size_t s3, const size_t s4,
                                                   const size_t s5, const size_t s6, const size_t l1, const size_t l2,
                                                   const size_t l3, const size_t l4, const size_t l5, const size_t l6,
                                                   const size_t d1, const size_t d2, const size_t d3, const size_t d4,
                                                   const size_t d5, const size_t d6, const uint32_t *input,
                                                   uint32_t *output, const uint32_t &device_id, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Slice6DKernel(const size_t s1, const size_t s2, const size_t s3, const size_t s4,
                                                   const size_t s5, const size_t s6, const size_t l1, const size_t l2,
                                                   const size_t l3, const size_t l4, const size_t l5, const size_t l6,
                                                   const size_t d1, const size_t d2, const size_t d3, const size_t d4,
                                                   const size_t d5, const size_t d6, const uint16_t *input,
                                                   uint16_t *output, const uint32_t &device_id, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Slice6DKernel(const size_t s1, const size_t s2, const size_t s3, const size_t s4,
                                                   const size_t s5, const size_t s6, const size_t l1, const size_t l2,
                                                   const size_t l3, const size_t l4, const size_t l5, const size_t l6,
                                                   const size_t d1, const size_t d2, const size_t d3, const size_t d4,
                                                   const size_t d5, const size_t d6, const unsigned char *input,
                                                   unsigned char *output, const uint32_t &device_id,
                                                   cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Slice6DKernel(const size_t s1, const size_t s2, const size_t s3, const size_t s4,
                                                   const size_t s5, const size_t s6, const size_t l1, const size_t l2,
                                                   const size_t l3, const size_t l4, const size_t l5, const size_t l6,
                                                   const size_t d1, const size_t d2, const size_t d3, const size_t d4,
                                                   const size_t d5, const size_t d6, const bool *input, bool *output,
                                                   const uint32_t &device_id, cudaStream_t stream);

template CUDA_LIB_EXPORT cudaError_t Slice7DKernel(const size_t s1, const size_t s2, const size_t s3, const size_t s4,
                                                   const size_t s5, const size_t s6, const size_t s7, const size_t l1,
                                                   const size_t l2, const size_t l3, const size_t l4, const size_t l5,
                                                   const size_t l6, const size_t l7, const size_t d1, const size_t d2,
                                                   const size_t d3, const size_t d4, const size_t d5, const size_t d6,
                                                   const size_t d7, const Complex<float> *input, Complex<float> *output,
                                                   const uint32_t &device_id, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Slice7DKernel(
  const size_t s1, const size_t s2, const size_t s3, const size_t s4, const size_t s5, const size_t s6, const size_t s7,
  const size_t l1, const size_t l2, const size_t l3, const size_t l4, const size_t l5, const size_t l6, const size_t l7,
  const size_t d1, const size_t d2, const size_t d3, const size_t d4, const size_t d5, const size_t d6, const size_t d7,
  const Complex<double> *input, Complex<double> *output, const uint32_t &device_id, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Slice7DKernel(const size_t s1, const size_t s2, const size_t s3, const size_t s4,
                                                   const size_t s5, const size_t s6, const size_t s7, const size_t l1,
                                                   const size_t l2, const size_t l3, const size_t l4, const size_t l5,
                                                   const size_t l6, const size_t l7, const size_t d1, const size_t d2,
                                                   const size_t d3, const size_t d4, const size_t d5, const size_t d6,
                                                   const size_t d7, const double *input, double *output,
                                                   const uint32_t &device_id, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Slice7DKernel(const size_t s1, const size_t s2, const size_t s3, const size_t s4,
                                                   const size_t s5, const size_t s6, const size_t s7, const size_t l1,
                                                   const size_t l2, const size_t l3, const size_t l4, const size_t l5,
                                                   const size_t l6, const size_t l7, const size_t d1, const size_t d2,
                                                   const size_t d3, const size_t d4, const size_t d5, const size_t d6,
                                                   const size_t d7, const float *input, float *output,
                                                   const uint32_t &device_id, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Slice7DKernel(const size_t s1, const size_t s2, const size_t s3, const size_t s4,
                                                   const size_t s5, const size_t s6, const size_t s7, const size_t l1,
                                                   const size_t l2, const size_t l3, const size_t l4, const size_t l5,
                                                   const size_t l6, const size_t l7, const size_t d1, const size_t d2,
                                                   const size_t d3, const size_t d4, const size_t d5, const size_t d6,
                                                   const size_t d7, const half *input, half *output,
                                                   const uint32_t &device_id, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Slice7DKernel(const size_t s1, const size_t s2, const size_t s3, const size_t s4,
                                                   const size_t s5, const size_t s6, const size_t s7, const size_t l1,
                                                   const size_t l2, const size_t l3, const size_t l4, const size_t l5,
                                                   const size_t l6, const size_t l7, const size_t d1, const size_t d2,
                                                   const size_t d3, const size_t d4, const size_t d5, const size_t d6,
                                                   const size_t d7, const int64_t *input, int64_t *output,
                                                   const uint32_t &device_id, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Slice7DKernel(const size_t s1, const size_t s2, const size_t s3, const size_t s4,
                                                   const size_t s5, const size_t s6, const size_t s7, const size_t l1,
                                                   const size_t l2, const size_t l3, const size_t l4, const size_t l5,
                                                   const size_t l6, const size_t l7, const size_t d1, const size_t d2,
                                                   const size_t d3, const size_t d4, const size_t d5, const size_t d6,
                                                   const size_t d7, const int *input, int *output,
                                                   const uint32_t &device_id, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Slice7DKernel(const size_t s1, const size_t s2, const size_t s3, const size_t s4,
                                                   const size_t s5, const size_t s6, const size_t s7, const size_t l1,
                                                   const size_t l2, const size_t l3, const size_t l4, const size_t l5,
                                                   const size_t l6, const size_t l7, const size_t d1, const size_t d2,
                                                   const size_t d3, const size_t d4, const size_t d5, const size_t d6,
                                                   const size_t d7, const int16_t *input, int16_t *output,
                                                   const uint32_t &device_id, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Slice7DKernel(const size_t s1, const size_t s2, const size_t s3, const size_t s4,
                                                   const size_t s5, const size_t s6, const size_t s7, const size_t l1,
                                                   const size_t l2, const size_t l3, const size_t l4, const size_t l5,
                                                   const size_t l6, const size_t l7, const size_t d1, const size_t d2,
                                                   const size_t d3, const size_t d4, const size_t d5, const size_t d6,
                                                   const size_t d7, const char *input, char *output,
                                                   const uint32_t &device_id, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Slice7DKernel(const size_t s1, const size_t s2, const size_t s3, const size_t s4,
                                                   const size_t s5, const size_t s6, const size_t s7, const size_t l1,
                                                   const size_t l2, const size_t l3, const size_t l4, const size_t l5,
                                                   const size_t l6, const size_t l7, const size_t d1, const size_t d2,
                                                   const size_t d3, const size_t d4, const size_t d5, const size_t d6,
                                                   const size_t d7, const uint64_t *input, uint64_t *output,
                                                   const uint32_t &device_id, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Slice7DKernel(const size_t s1, const size_t s2, const size_t s3, const size_t s4,
                                                   const size_t s5, const size_t s6, const size_t s7, const size_t l1,
                                                   const size_t l2, const size_t l3, const size_t l4, const size_t l5,
                                                   const size_t l6, const size_t l7, const size_t d1, const size_t d2,
                                                   const size_t d3, const size_t d4, const size_t d5, const size_t d6,
                                                   const size_t d7, const uint32_t *input, uint32_t *output,
                                                   const uint32_t &device_id, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Slice7DKernel(const size_t s1, const size_t s2, const size_t s3, const size_t s4,
                                                   const size_t s5, const size_t s6, const size_t s7, const size_t l1,
                                                   const size_t l2, const size_t l3, const size_t l4, const size_t l5,
                                                   const size_t l6, const size_t l7, const size_t d1, const size_t d2,
                                                   const size_t d3, const size_t d4, const size_t d5, const size_t d6,
                                                   const size_t d7, const uint16_t *input, uint16_t *output,
                                                   const uint32_t &device_id, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Slice7DKernel(const size_t s1, const size_t s2, const size_t s3, const size_t s4,
                                                   const size_t s5, const size_t s6, const size_t s7, const size_t l1,
                                                   const size_t l2, const size_t l3, const size_t l4, const size_t l5,
                                                   const size_t l6, const size_t l7, const size_t d1, const size_t d2,
                                                   const size_t d3, const size_t d4, const size_t d5, const size_t d6,
                                                   const size_t d7, const unsigned char *input, unsigned char *output,
                                                   const uint32_t &device_id, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t Slice7DKernel(const size_t s1, const size_t s2, const size_t s3, const size_t s4,
                                                   const size_t s5, const size_t s6, const size_t s7, const size_t l1,
                                                   const size_t l2, const size_t l3, const size_t l4, const size_t l5,
                                                   const size_t l6, const size_t l7, const size_t d1, const size_t d2,
                                                   const size_t d3, const size_t d4, const size_t d5, const size_t d6,
                                                   const size_t d7, const bool *input, bool *output,
                                                   const uint32_t &device_id, cudaStream_t stream);

template CUDA_LIB_EXPORT cudaError_t CalSlice4DGrad<Complex<float>>(const size_t s1, const size_t s2, const size_t s3,
                                                                    const size_t s4, const size_t l1, const size_t l2,
                                                                    const size_t l3, const size_t l4, const size_t d1,
                                                                    const size_t d2, const size_t d3, const size_t d4,
                                                                    const Complex<float> *dy, Complex<float> *dx,
                                                                    cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t CalSlice4DGrad<Complex<double>>(const size_t s1, const size_t s2, const size_t s3,
                                                                     const size_t s4, const size_t l1, const size_t l2,
                                                                     const size_t l3, const size_t l4, const size_t d1,
                                                                     const size_t d2, const size_t d3, const size_t d4,
                                                                     const Complex<double> *dy, Complex<double> *dx,
                                                                     cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t CalSlice4DGrad<double>(const size_t s1, const size_t s2, const size_t s3,
                                                            const size_t s4, const size_t l1, const size_t l2,
                                                            const size_t l3, const size_t l4, const size_t d1,
                                                            const size_t d2, const size_t d3, const size_t d4,
                                                            const double *dy, double *dx, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t CalSlice4DGrad<float>(const size_t s1, const size_t s2, const size_t s3,
                                                           const size_t s4, const size_t l1, const size_t l2,
                                                           const size_t l3, const size_t l4, const size_t d1,
                                                           const size_t d2, const size_t d3, const size_t d4,
                                                           const float *dy, float *dx, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t CalSlice4DGrad<half>(const size_t s1, const size_t s2, const size_t s3,
                                                          const size_t s4, const size_t l1, const size_t l2,
                                                          const size_t l3, const size_t l4, const size_t d1,
                                                          const size_t d2, const size_t d3, const size_t d4,
                                                          const half *dy, half *dx, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t CalSlice4DGrad<int>(const size_t s1, const size_t s2, const size_t s3,
                                                         const size_t s4, const size_t l1, const size_t l2,
                                                         const size_t l3, const size_t l4, const size_t d1,
                                                         const size_t d2, const size_t d3, const size_t d4,
                                                         const int *dy, int *dx, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t CalSlice4DGrad<int16_t>(const size_t s1, const size_t s2, const size_t s3,
                                                             const size_t s4, const size_t l1, const size_t l2,
                                                             const size_t l3, const size_t l4, const size_t d1,
                                                             const size_t d2, const size_t d3, const size_t d4,
                                                             const int16_t *dy, int16_t *dx, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t CalSlice4DGrad<unsigned char>(const size_t s1, const size_t s2, const size_t s3,
                                                                   const size_t s4, const size_t l1, const size_t l2,
                                                                   const size_t l3, const size_t l4, const size_t d1,
                                                                   const size_t d2, const size_t d3, const size_t d4,
                                                                   const unsigned char *dy, unsigned char *dx,
                                                                   cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t CalSlice4DGrad<int64_t>(const size_t s1, const size_t s2, const size_t s3,
                                                             const size_t s4, const size_t l1, const size_t l2,
                                                             const size_t l3, const size_t l4, const size_t d1,
                                                             const size_t d2, const size_t d3, const size_t d4,
                                                             const int64_t *dy, int64_t *dx, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t CalSlice4DGrad<bool>(const size_t s1, const size_t s2, const size_t s3,
                                                          const size_t s4, const size_t l1, const size_t l2,
                                                          const size_t l3, const size_t l4, const size_t d1,
                                                          const size_t d2, const size_t d3, const size_t d4,
                                                          const bool *dy, bool *dx, cudaStream_t stream);

template CUDA_LIB_EXPORT cudaError_t CalSlice7DGrad<double>(
  const size_t s1, const size_t s2, const size_t s3, const size_t s4, const size_t s5, const size_t s6, const size_t s7,
  const size_t l1, const size_t l2, const size_t l3, const size_t l4, const size_t l5, const size_t l6, const size_t l7,
  const size_t d1, const size_t d2, const size_t d3, const size_t d4, const size_t d5, const size_t d6, const size_t d7,
  const double *dy, double *dx, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t CalSlice7DGrad<float>(
  const size_t s1, const size_t s2, const size_t s3, const size_t s4, const size_t s5, const size_t s6, const size_t s7,
  const size_t l1, const size_t l2, const size_t l3, const size_t l4, const size_t l5, const size_t l6, const size_t l7,
  const size_t d1, const size_t d2, const size_t d3, const size_t d4, const size_t d5, const size_t d6, const size_t d7,
  const float *dy, float *dx, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t CalSlice7DGrad<half>(
  const size_t s1, const size_t s2, const size_t s3, const size_t s4, const size_t s5, const size_t s6, const size_t s7,
  const size_t l1, const size_t l2, const size_t l3, const size_t l4, const size_t l5, const size_t l6, const size_t l7,
  const size_t d1, const size_t d2, const size_t d3, const size_t d4, const size_t d5, const size_t d6, const size_t d7,
  const half *dy, half *dx, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t CalSlice7DGrad<int>(
  const size_t s1, const size_t s2, const size_t s3, const size_t s4, const size_t s5, const size_t s6, const size_t s7,
  const size_t l1, const size_t l2, const size_t l3, const size_t l4, const size_t l5, const size_t l6, const size_t l7,
  const size_t d1, const size_t d2, const size_t d3, const size_t d4, const size_t d5, const size_t d6, const size_t d7,
  const int *dy, int *dx, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t CalSlice7DGrad<int16_t>(
  const size_t s1, const size_t s2, const size_t s3,
  const size_t s4, const size_t s5, const size_t s6, const size_t s7, const size_t l1, const size_t l2, const size_t l3,
  const size_t l4, const size_t l5, const size_t l6, const size_t l7, const size_t d1, const size_t d2, const size_t d3,
  const size_t d4, const size_t d5, const size_t d6, const size_t d7, const int16_t *dy, int16_t *dx,
  cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t CalSlice7DGrad<unsigned char>(
  const size_t s1, const size_t s2, const size_t s3, const size_t s4, const size_t s5, const size_t s6, const size_t s7,
  const size_t l1, const size_t l2, const size_t l3, const size_t l4, const size_t l5, const size_t l6, const size_t l7,
  const size_t d1, const size_t d2, const size_t d3, const size_t d4, const size_t d5, const size_t d6, const size_t d7,
  const unsigned char *dy, unsigned char *dx, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t CalSlice7DGrad<int64_t>(
  const size_t s1, const size_t s2, const size_t s3, const size_t s4, const size_t s5, const size_t s6, const size_t s7,
  const size_t l1, const size_t l2, const size_t l3, const size_t l4, const size_t l5, const size_t l6, const size_t l7,
  const size_t d1, const size_t d2, const size_t d3, const size_t d4, const size_t d5, const size_t d6, const size_t d7,
  const int64_t *dy, int64_t *dx, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t CalSlice7DGrad<bool>(
  const size_t s1, const size_t s2, const size_t s3, const size_t s4, const size_t s5, const size_t s6, const size_t s7,
  const size_t l1, const size_t l2, const size_t l3, const size_t l4, const size_t l5, const size_t l6, const size_t l7,
  const size_t d1, const size_t d2, const size_t d3, const size_t d4, const size_t d5, const size_t d6, const size_t d7,
  const bool *dy, bool *dx, cudaStream_t stream);

template CUDA_LIB_EXPORT cudaError_t FillDeviceArray<bool>(const size_t input_size, bool *addr, const float value,
                                                           cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t FillDeviceArray<int64_t>(const size_t input_size, int64_t *addr, const float value,
                                                              cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t FillDeviceArray<int>(const size_t input_size, int *addr, const float value,
                                                          cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t FillDeviceArray<int16_t>(const size_t input_size, int16_t *addr, const float value,
                                                            cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t FillDeviceArray<int8_t>(const size_t input_size, int8_t *addr, const float value,
                                                             cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t FillDeviceArray<uint64_t>(const size_t input_size, uint64_t *addr,
                                                               const float value, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t FillDeviceArray<uint32_t>(const size_t input_size, uint32_t *addr,
                                                               const float value, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t FillDeviceArray<uint16_t>(const size_t input_size, uint16_t *addr,
                                                               const float value, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t FillDeviceArray<unsigned char>(const size_t input_size, unsigned char *addr,
                                                                    const float value, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t FillDeviceArray<half>(const size_t input_size, half *addr, const float value,
                                                           cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t FillDeviceArray<float>(const size_t input_size, float *addr, const float value,
                                                            cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t FillDeviceArray<double>(const size_t input_size, double *addr, const float value,
                                                             cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t FillDeviceArray<Complex<float>>(const size_t input_size, Complex<float> *addr,
                                                                     const float value, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t FillDeviceArray<Complex<double>>(const size_t input_size, Complex<double> *addr,
                                                                      const float value, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t StridedSlice(const std::vector<size_t> &input_shape,
                                                  const std::vector<int64_t> &begin,
                                                  const std::vector<int64_t> &strides,
                                                  const std::vector<size_t> &output_shape, const Complex<float> *input,
                                                  Complex<float> *output, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t StridedSlice(const std::vector<size_t> &input_shape,
                                                  const std::vector<int64_t> &begin,
                                                  const std::vector<int64_t> &strides,
                                                  const std::vector<size_t> &output_shape, const Complex<double> *input,
                                                  Complex<double> *output, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t StridedSlice(const std::vector<size_t> &input_shape,
                                                  const std::vector<int64_t> &begin,
                                                  const std::vector<int64_t> &strides,
                                                  const std::vector<size_t> &output_shape, const bool *input,
                                                  bool *output, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t StridedSlice(const std::vector<size_t> &input_shape,
                                                  const std::vector<int64_t> &begin,
                                                  const std::vector<int64_t> &strides,
                                                  const std::vector<size_t> &output_shape, const double *input,
                                                  double *output, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t StridedSlice(const std::vector<size_t> &input_shape,
                                                  const std::vector<int64_t> &begin,
                                                  const std::vector<int64_t> &strides,
                                                  const std::vector<size_t> &output_shape, const float *input,
                                                  float *output, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t StridedSlice(const std::vector<size_t> &input_shape,
                                                  const std::vector<int64_t> &begin,
                                                  const std::vector<int64_t> &strides,
                                                  const std::vector<size_t> &output_shape, const half *input,
                                                  half *output, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t StridedSlice(const std::vector<size_t> &input_shape,
                                                  const std::vector<int64_t> &begin,
                                                  const std::vector<int64_t> &strides,
                                                  const std::vector<size_t> &output_shape, const int64_t *input,
                                                  int64_t *output, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t StridedSlice(const std::vector<size_t> &input_shape,
                                                  const std::vector<int64_t> &begin,
                                                  const std::vector<int64_t> &strides,
                                                  const std::vector<size_t> &output_shape, const int *input,
                                                  int *output, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t StridedSlice(const std::vector<size_t> &input_shape,
                                                  const std::vector<int64_t> &begin,
                                                  const std::vector<int64_t> &strides,
                                                  const std::vector<size_t> &output_shape, const int16_t *input,
                                                  int16_t *output, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t StridedSlice(const std::vector<size_t> &input_shape,
                                                  const std::vector<int64_t> &begin,
                                                  const std::vector<int64_t> &strides,
                                                  const std::vector<size_t> &output_shape, const int8_t *input,
                                                  int8_t *output, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t StridedSlice(const std::vector<size_t> &input_shape,
                                                  const std::vector<int64_t> &begin,
                                                  const std::vector<int64_t> &strides,
                                                  const std::vector<size_t> &output_shape, const uint64_t *input,
                                                  uint64_t *output, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t StridedSlice(const std::vector<size_t> &input_shape,
                                                  const std::vector<int64_t> &begin,
                                                  const std::vector<int64_t> &strides,
                                                  const std::vector<size_t> &output_shape, const uint32_t *input,
                                                  uint32_t *output, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t StridedSlice(const std::vector<size_t> &input_shape,
                                                  const std::vector<int64_t> &begin,
                                                  const std::vector<int64_t> &strides,
                                                  const std::vector<size_t> &output_shape, const uint16_t *input,
                                                  uint16_t *output, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t StridedSlice(const std::vector<size_t> &input_shape,
                                                  const std::vector<int64_t> &begin,
                                                  const std::vector<int64_t> &strides,
                                                  const std::vector<size_t> &output_shape, const unsigned char *input,
                                                  unsigned char *output, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t StridedSliceGrad(const std::vector<size_t> &dy_shape,
                                                      const std::vector<int64_t> &begin,
                                                      const std::vector<int64_t> &strides,
                                                      const std::vector<size_t> &dx_shape, const Complex<float> *dy,
                                                      Complex<float> *dx, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t StridedSliceGrad(const std::vector<size_t> &dy_shape,
                                                      const std::vector<int64_t> &begin,
                                                      const std::vector<int64_t> &strides,
                                                      const std::vector<size_t> &dx_shape, const Complex<double> *dy,
                                                      Complex<double> *dx, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t StridedSliceGrad(const std::vector<size_t> &dy_shape,
                                                      const std::vector<int64_t> &begin,
                                                      const std::vector<int64_t> &strides,
                                                      const std::vector<size_t> &dx_shape, const bool *dy, bool *dx,
                                                      cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t StridedSliceGrad(const std::vector<size_t> &dy_shape,
                                                      const std::vector<int64_t> &begin,
                                                      const std::vector<int64_t> &strides,
                                                      const std::vector<size_t> &dx_shape, const double *dy, double *dx,
                                                      cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t StridedSliceGrad(const std::vector<size_t> &dy_shape,
                                                      const std::vector<int64_t> &begin,
                                                      const std::vector<int64_t> &strides,
                                                      const std::vector<size_t> &dx_shape, const float *dy, float *dx,
                                                      cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t StridedSliceGrad(const std::vector<size_t> &dy_shape,
                                                      const std::vector<int64_t> &begin,
                                                      const std::vector<int64_t> &strides,
                                                      const std::vector<size_t> &dx_shape, const half *dy, half *dx,
                                                      cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t StridedSliceGrad(const std::vector<size_t> &dy_shape,
                                                      const std::vector<int64_t> &begin,
                                                      const std::vector<int64_t> &strides,
                                                      const std::vector<size_t> &dx_shape, const int64_t *dy,
                                                      int64_t *dx, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t StridedSliceGrad(const std::vector<size_t> &dy_shape,
                                                      const std::vector<int64_t> &begin,
                                                      const std::vector<int64_t> &strides,
                                                      const std::vector<size_t> &dx_shape, const int *dy, int *dx,
                                                      cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t StridedSliceGrad(const std::vector<size_t> &dy_shape,
                                                      const std::vector<int64_t> &begin,
                                                      const std::vector<int64_t> &strides,
                                                      const std::vector<size_t> &dx_shape, const int16_t *dy,
                                                      int16_t *dx, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t StridedSliceGrad(const std::vector<size_t> &dy_shape,
                                                      const std::vector<int64_t> &begin,
                                                      const std::vector<int64_t> &strides,
                                                      const std::vector<size_t> &dx_shape, const int8_t *dy, int8_t *dx,
                                                      cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t StridedSliceGrad(const std::vector<size_t> &dy_shape,
                                                      const std::vector<int64_t> &begin,
                                                      const std::vector<int64_t> &strides,
                                                      const std::vector<size_t> &dx_shape, const uint64_t *dy,
                                                      uint64_t *dx, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t StridedSliceGrad(const std::vector<size_t> &dy_shape,
                                                      const std::vector<int64_t> &begin,
                                                      const std::vector<int64_t> &strides,
                                                      const std::vector<size_t> &dx_shape, const uint32_t *dy,
                                                      uint32_t *dx, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t StridedSliceGrad(const std::vector<size_t> &dy_shape,
                                                      const std::vector<int64_t> &begin,
                                                      const std::vector<int64_t> &strides,
                                                      const std::vector<size_t> &dx_shape, const uint16_t *dy,
                                                      uint16_t *dx, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t StridedSliceGrad(const std::vector<size_t> &dy_shape,
                                                      const std::vector<int64_t> &begin,
                                                      const std::vector<int64_t> &strides,
                                                      const std::vector<size_t> &dx_shape, const unsigned char *dy,
                                                      unsigned char *dx, cudaStream_t cuda_stream);
