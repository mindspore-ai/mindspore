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
#include "backend/kernel_compiler/gpu/cuda_impl/slice_impl.cuh"

template <typename T>
__global__ void Slice4D(const size_t s1, const size_t s2, const size_t s3, const size_t s4, const size_t l1,
                        const size_t l2, const size_t l3, const size_t l4, const size_t d1, const size_t d2,
                        const size_t d3, const size_t d4, const T *input, T *output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < (l1 * l2 * l3 * l4); pos += blockDim.x * gridDim.x) {
    size_t i = pos / (l2 * l3 * l4) % l1;
    size_t j = pos / (l3 * l4) % l2;
    size_t k = pos / l4 % l3;
    size_t o = pos % l4;

    size_t offset = (i + s1) * (d2 * d3 * d4) + (j + s2) * (d3 * d4) + (k + s3) * d4 + (o + s4);
    output[pos] = input[offset];
  }
}
template <typename T>
__global__ void SliceGrad(const T *dy, int64_t p, int64_t start, int64_t length, T *output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < (length); pos += blockDim.x * gridDim.x) {
    output[start + pos] = dy[p + pos];
  }
  return;
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
void FillDeviceArray(const size_t input_size, T *addr, const float value, cudaStream_t cuda_stream) {
  FillArray<<<GET_BLOCKS(input_size), GET_THREADS, 0, cuda_stream>>>(addr, input_size, value);
  return;
}
template <typename T>
void Slice4DKernel(const size_t s1, const size_t s2, const size_t s3, const size_t s4, const size_t l1, const size_t l2,
                   const size_t l3, const size_t l4, const size_t d1, const size_t d2, const size_t d3, const size_t d4,
                   const T *input, T *output, cudaStream_t stream) {
  Slice4D<<<GET_BLOCKS(l1 * l2 * l3 * l4), GET_THREADS, 0, stream>>>(s1, s2, s3, s4, l1, l2, l3, l4, d1, d2, d3, d4,
                                                                     input, output);
}
template <typename T>
void CalSliceGrad(const size_t input_size, const T *dy, const std::vector<size_t> in_shape,
                  const std::vector<int64_t> begin, const std::vector<int64_t> size, T *output,
                  cudaStream_t cuda_stream) {
  size_t block = in_shape[1] * in_shape[2] * in_shape[3];
  size_t map = in_shape[2] * in_shape[3];
  size_t w = in_shape[3];
  int64_t length = size[3];
  int64_t p = 0;
  for (int64_t i = begin[0]; i < size[0] + begin[0]; i++) {
    for (int64_t j = begin[1]; j < size[1] + begin[1]; j++) {
      for (int64_t k = begin[2]; k < size[2] + begin[2]; k++) {
        SliceGrad<<<GET_BLOCKS(input_size), GET_THREADS, 0, cuda_stream>>>(
          dy, p, i * block + j * map + k * w + begin[3], length, output);
        p = p + size[3];
      }
    }
  }
}

template <typename T>
__global__ void StridedSliceKernel(const size_t b0, const size_t b1, const size_t b2, const size_t b3, const size_t b4,
                                   const size_t b5, const size_t b6, const size_t s0, const size_t s1, const size_t s2,
                                   const size_t s3, const size_t s4, const size_t s5, const size_t s6, const size_t i0,
                                   const size_t i1, const size_t i2, const size_t i3, const size_t i4, const size_t i5,
                                   const size_t i6, const size_t o0, const size_t o1, const size_t o2, const size_t o3,
                                   const size_t o4, const size_t o5, const size_t o6, const T *input_addr,
                                   T *output_addr) {
  size_t output_num = o0 * o1 * o2 * o3 * o4 * o5 * o6;
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < output_num; pos += blockDim.x * gridDim.x) {
    size_t i = pos / (o1 * o2 * o3 * o4 * o5 * o6) % o0;
    size_t j = pos / (o2 * o3 * o4 * o5 * o6) % o1;
    size_t k = pos / (o3 * o4 * o5 * o6) % o2;
    size_t l = pos / (o4 * o5 * o6) % o3;
    size_t m = pos / (o5 * o6) % o4;
    size_t n = pos / (o6) % o5;
    size_t o = pos % o6;

    size_t input_idx = (i * s0 + b0) * i1 * i2 * i3 * i4 * i5 * i6 + (j * s1 + b1) * i2 * i3 * i4 * i5 * i6 +
                       (k * s2 + b2) * i3 * i4 * i5 * i6 + (l * s3 + b3) * i4 * i5 * i6 + (m * s4 + b4) * i5 * i6 +
                       (n * s5 + b5) * i6 + (o * s6 + b6);
    output_addr[pos] = input_addr[input_idx];
  }
}

template <typename T>
void StridedSlice(const std::vector<size_t> &input_shape, const std::vector<int64_t> &begin,
                  const std::vector<int64_t> &strides, const std::vector<size_t> &output_shape, const T *input,
                  T *output, cudaStream_t cuda_stream) {
  size_t size = output_shape[0] * output_shape[1] * output_shape[2] * output_shape[3] * output_shape[4] *
                output_shape[5] * output_shape[6];
  StridedSliceKernel<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(
    begin[0], begin[1], begin[2], begin[3], begin[4], begin[5], begin[6], strides[0], strides[1], strides[2],
    strides[3], strides[4], strides[5], strides[6], input_shape[0], input_shape[1], input_shape[2], input_shape[3],
    input_shape[4], input_shape[5], input_shape[6], output_shape[0], output_shape[1], output_shape[2], output_shape[3],
    output_shape[4], output_shape[5], output_shape[6], input, output);
}

template <typename T>
__global__ void StridedSliceGradKernel(const size_t b0, const size_t b1, const size_t b2, const size_t b3,
                                       const size_t b4, const size_t b5, const size_t b6, const size_t s0,
                                       const size_t s1, const size_t s2, const size_t s3, const size_t s4,
                                       const size_t s5, const size_t s6, const size_t i0, const size_t i1,
                                       const size_t i2, const size_t i3, const size_t i4, const size_t i5,
                                       const size_t i6, const size_t o0, const size_t o1, const size_t o2,
                                       const size_t o3, const size_t o4, const size_t o5, const size_t o6, const T *dy,
                                       T *dx) {
  size_t output_num = o0 * o1 * o2 * o3 * o4 * o5 * o6;
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < output_num; pos += blockDim.x * gridDim.x) {
    size_t i = pos / (o1 * o2 * o3 * o4 * o5 * o6) % o0;
    size_t j = pos / (o2 * o3 * o4 * o5 * o6) % o1;
    size_t k = pos / (o3 * o4 * o5 * o6) % o2;
    size_t l = pos / (o4 * o5 * o6) % o3;
    size_t m = pos / (o5 * o6) % o4;
    size_t n = pos / (o6) % o5;
    size_t o = pos % o6;

    size_t input_idx = (i * s0 + b0) * i1 * i2 * i3 * i4 * i5 * i6 + (j * s1 + b1) * i2 * i3 * i4 * i5 * i6 +
                       (k * s2 + b2) * i3 * i4 * i5 * i6 + (l * s3 + b3) * i4 * i5 * i6 + (m * s4 + b4) * i5 * i6 +
                       (n * s5 + b5) * i6 + (o * s6 + b6);
    dx[input_idx] = dy[pos];
  }
  return;
}

template <typename T>
void StridedSliceGrad(const std::vector<size_t> &dy_shape, const std::vector<int64_t> &begin,
                      const std::vector<int64_t> &strides, const std::vector<size_t> &dx_shape, const T *dy, T *dx,
                      cudaStream_t cuda_stream) {
  size_t size = dy_shape[0] * dy_shape[1] * dy_shape[2] * dy_shape[3] * dy_shape[4] * dy_shape[5] * dy_shape[6];
  StridedSliceGradKernel<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(
    begin[0], begin[1], begin[2], begin[3], begin[4], begin[5], begin[6], strides[0], strides[1], strides[2],
    strides[3], strides[4], strides[5], strides[6], dx_shape[0], dx_shape[1], dx_shape[2], dx_shape[3], dx_shape[4],
    dx_shape[5], dx_shape[6], dy_shape[0], dy_shape[1], dy_shape[2], dy_shape[3], dy_shape[4], dy_shape[5], dy_shape[6],
    dy, dx);
}

template void Slice4DKernel(const size_t s1, const size_t s2, const size_t s3, const size_t s4, const size_t l1,
                            const size_t l2, const size_t l3, const size_t l4, const size_t d1, const size_t d2,
                            const size_t d3, const size_t d4, const double *input, double *output, cudaStream_t stream);
template void Slice4DKernel(const size_t s1, const size_t s2, const size_t s3, const size_t s4, const size_t l1,
                            const size_t l2, const size_t l3, const size_t l4, const size_t d1, const size_t d2,
                            const size_t d3, const size_t d4, const float *input, float *output, cudaStream_t stream);
template void Slice4DKernel(const size_t s1, const size_t s2, const size_t s3, const size_t s4, const size_t l1,
                            const size_t l2, const size_t l3, const size_t l4, const size_t d1, const size_t d2,
                            const size_t d3, const size_t d4, const half *input, half *output, cudaStream_t stream);
template void Slice4DKernel(const size_t s1, const size_t s2, const size_t s3, const size_t s4, const size_t l1,
                            const size_t l2, const size_t l3, const size_t l4, const size_t d1, const size_t d2,
                            const size_t d3, const size_t d4, const int64_t *input, int64_t *output,
                            cudaStream_t stream);
template void Slice4DKernel(const size_t s1, const size_t s2, const size_t s3, const size_t s4, const size_t l1,
                            const size_t l2, const size_t l3, const size_t l4, const size_t d1, const size_t d2,
                            const size_t d3, const size_t d4, const int *input, int *output, cudaStream_t stream);
template void Slice4DKernel(const size_t s1, const size_t s2, const size_t s3, const size_t s4, const size_t l1,
                            const size_t l2, const size_t l3, const size_t l4, const size_t d1, const size_t d2,
                            const size_t d3, const size_t d4, const short *input, short *output,  // NOLINT
                            cudaStream_t stream);
template void Slice4DKernel(const size_t s1, const size_t s2, const size_t s3, const size_t s4, const size_t l1,
                            const size_t l2, const size_t l3, const size_t l4, const size_t d1, const size_t d2,
                            const size_t d3, const size_t d4, const unsigned char *input, unsigned char *output,
                            cudaStream_t stream);
template void Slice4DKernel(const size_t s1, const size_t s2, const size_t s3, const size_t s4, const size_t l1,
                            const size_t l2, const size_t l3, const size_t l4, const size_t d1, const size_t d2,
                            const size_t d3, const size_t d4, const bool *input, bool *output, cudaStream_t stream);

template void CalSliceGrad<double>(const size_t input_size, const double *dy, const std::vector<size_t> in_shape,
                                   const std::vector<int64_t> begin, const std::vector<int64_t> size, double *output,
                                   cudaStream_t cuda_stream);
template void CalSliceGrad<float>(const size_t input_size, const float *dy, const std::vector<size_t> in_shape,
                                  const std::vector<int64_t> begin, const std::vector<int64_t> size, float *output,
                                  cudaStream_t cuda_stream);
template void CalSliceGrad<half>(const size_t input_size, const half *dy, const std::vector<size_t> in_shape,
                                 const std::vector<int64_t> begin, const std::vector<int64_t> size, half *output,
                                 cudaStream_t cuda_stream);
template void CalSliceGrad<int64_t>(const size_t input_size, const int64_t *dy, const std::vector<size_t> in_shape,
                                    const std::vector<int64_t> begin, const std::vector<int64_t> size, int64_t *output,
                                    cudaStream_t cuda_stream);
template void CalSliceGrad<int>(const size_t input_size, const int *dy, const std::vector<size_t> in_shape,
                                const std::vector<int64_t> begin, const std::vector<int64_t> size, int *output,
                                cudaStream_t cuda_stream);
template void CalSliceGrad<short>(const size_t input_size, const short *dy,  // NOLINT
                                  const std::vector<size_t> in_shape, const std::vector<int64_t> begin,
                                  const std::vector<int64_t> size, short *output,  // NOLINT
                                  cudaStream_t cuda_stream);
template void CalSliceGrad<unsigned char>(const size_t input_size, const unsigned char *dy,
                                          const std::vector<size_t> in_shape, const std::vector<int64_t> begin,
                                          const std::vector<int64_t> size, unsigned char *output,
                                          cudaStream_t cuda_stream);
template void CalSliceGrad<bool>(const size_t input_size, const bool *dy, const std::vector<size_t> in_shape,
                                 const std::vector<int64_t> begin, const std::vector<int64_t> size, bool *output,
                                 cudaStream_t cuda_stream);

template void FillDeviceArray<bool>(const size_t input_size, bool *addr, const float value, cudaStream_t cuda_stream);
template void FillDeviceArray<int64_t>(const size_t input_size, int64_t *addr, const float value,
                                       cudaStream_t cuda_stream);
template void FillDeviceArray<int>(const size_t input_size, int *addr, const float value, cudaStream_t cuda_stream);
template void FillDeviceArray<short>(const size_t input_size, short *addr, const float value,  // NOLINT
                                     cudaStream_t cuda_stream);
template void FillDeviceArray<int8_t>(const size_t input_size, int8_t *addr, const float value,
                                      cudaStream_t cuda_stream);
template void FillDeviceArray<uint64_t>(const size_t input_size, uint64_t *addr, const float value,
                                        cudaStream_t cuda_stream);
template void FillDeviceArray<uint32_t>(const size_t input_size, uint32_t *addr, const float value,
                                        cudaStream_t cuda_stream);
template void FillDeviceArray<uint16_t>(const size_t input_size, uint16_t *addr, const float value,
                                        cudaStream_t cuda_stream);
template void FillDeviceArray<unsigned char>(const size_t input_size, unsigned char *addr, const float value,
                                             cudaStream_t cuda_stream);
template void FillDeviceArray<half>(const size_t input_size, half *addr, const float value, cudaStream_t cuda_stream);
template void FillDeviceArray<float>(const size_t input_size, float *addr, const float value, cudaStream_t cuda_stream);
template void FillDeviceArray<double>(const size_t input_size, double *addr, const float value,
                                      cudaStream_t cuda_stream);

template void StridedSlice(const std::vector<size_t> &input_shape, const std::vector<int64_t> &begin,
                           const std::vector<int64_t> &strides, const std::vector<size_t> &output_shape,
                           const bool *input, bool *output, cudaStream_t cuda_stream);
template void StridedSlice(const std::vector<size_t> &input_shape, const std::vector<int64_t> &begin,
                           const std::vector<int64_t> &strides, const std::vector<size_t> &output_shape,
                           const double *input, double *output, cudaStream_t cuda_stream);
template void StridedSlice(const std::vector<size_t> &input_shape, const std::vector<int64_t> &begin,
                           const std::vector<int64_t> &strides, const std::vector<size_t> &output_shape,
                           const float *input, float *output, cudaStream_t cuda_stream);
template void StridedSlice(const std::vector<size_t> &input_shape, const std::vector<int64_t> &begin,
                           const std::vector<int64_t> &strides, const std::vector<size_t> &output_shape,
                           const half *input, half *output, cudaStream_t cuda_stream);
template void StridedSlice(const std::vector<size_t> &input_shape, const std::vector<int64_t> &begin,
                           const std::vector<int64_t> &strides, const std::vector<size_t> &output_shape,
                           const int64_t *input, int64_t *output, cudaStream_t cuda_stream);
template void StridedSlice(const std::vector<size_t> &input_shape, const std::vector<int64_t> &begin,
                           const std::vector<int64_t> &strides, const std::vector<size_t> &output_shape,
                           const int *input, int *output, cudaStream_t cuda_stream);
template void StridedSlice(const std::vector<size_t> &input_shape, const std::vector<int64_t> &begin,
                           const std::vector<int64_t> &strides, const std::vector<size_t> &output_shape,
                           const short *input, short *output, cudaStream_t cuda_stream);  // NOLINT
template void StridedSlice(const std::vector<size_t> &input_shape, const std::vector<int64_t> &begin,
                           const std::vector<int64_t> &strides, const std::vector<size_t> &output_shape,
                           const int8_t *input, int8_t *output, cudaStream_t cuda_stream);
template void StridedSlice(const std::vector<size_t> &input_shape, const std::vector<int64_t> &begin,
                           const std::vector<int64_t> &strides, const std::vector<size_t> &output_shape,
                           const uint64_t *input, uint64_t *output, cudaStream_t cuda_stream);
template void StridedSlice(const std::vector<size_t> &input_shape, const std::vector<int64_t> &begin,
                           const std::vector<int64_t> &strides, const std::vector<size_t> &output_shape,
                           const uint32_t *input, uint32_t *output, cudaStream_t cuda_stream);
template void StridedSlice(const std::vector<size_t> &input_shape, const std::vector<int64_t> &begin,
                           const std::vector<int64_t> &strides, const std::vector<size_t> &output_shape,
                           const uint16_t *input, uint16_t *output, cudaStream_t cuda_stream);
template void StridedSlice(const std::vector<size_t> &input_shape, const std::vector<int64_t> &begin,
                           const std::vector<int64_t> &strides, const std::vector<size_t> &output_shape,
                           const unsigned char *input, unsigned char *output, cudaStream_t cuda_stream);

template void StridedSliceGrad(const std::vector<size_t> &dy_shape, const std::vector<int64_t> &begin,
                               const std::vector<int64_t> &strides, const std::vector<size_t> &dx_shape, const bool *dy,
                               bool *dx, cudaStream_t cuda_stream);
template void StridedSliceGrad(const std::vector<size_t> &dy_shape, const std::vector<int64_t> &begin,
                               const std::vector<int64_t> &strides, const std::vector<size_t> &dx_shape,
                               const double *dy, double *dx, cudaStream_t cuda_stream);
template void StridedSliceGrad(const std::vector<size_t> &dy_shape, const std::vector<int64_t> &begin,
                               const std::vector<int64_t> &strides, const std::vector<size_t> &dx_shape,
                               const float *dy, float *dx, cudaStream_t cuda_stream);
template void StridedSliceGrad(const std::vector<size_t> &dy_shape, const std::vector<int64_t> &begin,
                               const std::vector<int64_t> &strides, const std::vector<size_t> &dx_shape, const half *dy,
                               half *dx, cudaStream_t cuda_stream);
template void StridedSliceGrad(const std::vector<size_t> &dy_shape, const std::vector<int64_t> &begin,
                               const std::vector<int64_t> &strides, const std::vector<size_t> &dx_shape,
                               const int64_t *dy, int64_t *dx, cudaStream_t cuda_stream);
template void StridedSliceGrad(const std::vector<size_t> &dy_shape, const std::vector<int64_t> &begin,
                               const std::vector<int64_t> &strides, const std::vector<size_t> &dx_shape, const int *dy,
                               int *dx, cudaStream_t cuda_stream);
template void StridedSliceGrad(const std::vector<size_t> &dy_shape, const std::vector<int64_t> &begin,
                               const std::vector<int64_t> &strides, const std::vector<size_t> &dx_shape,
                               const short *dy,                       // NOLINT
                               short *dx, cudaStream_t cuda_stream);  // NOLINT
template void StridedSliceGrad(const std::vector<size_t> &dy_shape, const std::vector<int64_t> &begin,
                               const std::vector<int64_t> &strides, const std::vector<size_t> &dx_shape,
                               const int8_t *dy, int8_t *dx, cudaStream_t cuda_stream);
template void StridedSliceGrad(const std::vector<size_t> &dy_shape, const std::vector<int64_t> &begin,
                               const std::vector<int64_t> &strides, const std::vector<size_t> &dx_shape,
                               const uint64_t *dy, uint64_t *dx, cudaStream_t cuda_stream);
template void StridedSliceGrad(const std::vector<size_t> &dy_shape, const std::vector<int64_t> &begin,
                               const std::vector<int64_t> &strides, const std::vector<size_t> &dx_shape,
                               const uint32_t *dy, uint32_t *dx, cudaStream_t cuda_stream);
template void StridedSliceGrad(const std::vector<size_t> &dy_shape, const std::vector<int64_t> &begin,
                               const std::vector<int64_t> &strides, const std::vector<size_t> &dx_shape,
                               const uint16_t *dy, uint16_t *dx, cudaStream_t cuda_stream);
template void StridedSliceGrad(const std::vector<size_t> &dy_shape, const std::vector<int64_t> &begin,
                               const std::vector<int64_t> &strides, const std::vector<size_t> &dx_shape,
                               const unsigned char *dy, unsigned char *dx, cudaStream_t cuda_stream);
