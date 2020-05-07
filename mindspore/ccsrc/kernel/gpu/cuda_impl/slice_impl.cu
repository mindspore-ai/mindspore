/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#include "kernel/gpu/cuda_impl/slice_impl.cuh"

template <typename T>
__global__ void Slice4D(const int s1, const int s2, const int s3, const int s4,
                       const int l1, const int l2, const int l3, const int l4,
                       const int d1, const int d2, const int d3, const int d4,
                       const T *input, T *output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < (l1 * l2 * l3 * l4); pos += blockDim.x * gridDim.x) {
    int i = pos / (l2 * l3 * l4) % l1;
    int j = pos / (l3 * l4) % l2;
    int k = pos / l4 % l3;
    int o = pos % l4;

    int offset = (i + s1) * (d2 * d3 * d4) +
                 (j + s2) * (d3 * d4) +
                 (k + s3) * d4 +
                 (o + s4);
    output[pos] = input[offset];
  }
}
template <typename T>
__global__ void SliceGrad(const T* dy, int p, int start, int length, T* output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < (length); pos += blockDim.x * gridDim.x) {
    output[start + pos] = dy[p + pos];
  }
  return;
}
template <typename T>
__global__ void StridedSlice(const T* input, int p, int start, int begin, int stride, int ended, T* output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < ((ended - 1 - begin) / stride) + 1;
       pos += blockDim.x * gridDim.x) {
    output[p + pos] = input[start + pos * stride];
  }
  return;
}
template <typename T>
__global__ void StridedSliceGrad(const T* dy, int p, int start, int begin, int stride, int ended, T* dx) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < ((ended - 1 - begin) / stride) + 1;
       pos += blockDim.x * gridDim.x) {
    dx[start + pos * stride] = dy[p + pos];
  }
  return;
}
template <typename T>
__global__ void FillArray(T* addr, const size_t len, const float value) {
  T value_ = static_cast<T>(value);
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < len; pos += blockDim.x * gridDim.x) {
    addr[pos] = value_;
  }
  return;
}
template <typename T>
void FillDeviceArray(const size_t input_size, T* addr, const float value, cudaStream_t cuda_stream) {
  FillArray<<<GET_BLOCKS(input_size), GET_THREADS, 0, cuda_stream>>>(addr, input_size, value);
  return;
}
template <typename T>
void Slice4DKernel(const int s1, const int s2, const int s3, const int s4,
                  const int l1, const int l2, const int l3, const int l4,
                  const int d1, const int d2, const int d3, const int d4,
                  const T *input, T *output, cudaStream_t stream) {
  Slice4D<<<GET_BLOCKS(l1 * l2 * l3 * l4), GET_THREADS, 0, stream>>>(s1, s2, s3, s4, l1, l2, l3, l4,
                                                                     d1, d2, d3, d4, input, output);
}
template <typename T>
void CalSliceGrad(const size_t input_size, const T* dy, const std::vector<int> in_shape, const std::vector<int> begin,
                  const std::vector<int> size, T* output, cudaStream_t cuda_stream) {
  int block = in_shape[1] * in_shape[2] * in_shape[3];
  int map = in_shape[2] * in_shape[3];
  int w = in_shape[3];
  int length = size[3];
  int p = 0;
  for (int i = begin[0]; i < size[0] + begin[0]; i++) {
    for (int j = begin[1]; j < size[1] + begin[1]; j++) {
      for (int k = begin[2]; k < size[2] + begin[2]; k++) {
        SliceGrad<<<GET_BLOCKS(input_size), GET_THREADS, 0, cuda_stream>>>(
          dy, p, i * block + j * map + k * w + begin[3], length, output);
        p = p + size[3];
      }
    }
  }
}
template <typename T>
void CalStridedSlice(const size_t input_size, const T* input, const std::vector<int> in_shape,
                     const std::vector<int> begin, const std::vector<int> end, const std::vector<int> strides,
                     T* output, cudaStream_t cuda_stream) {
  int block = in_shape[1] * in_shape[2] * in_shape[3];
  int map = in_shape[2] * in_shape[3];
  int w = in_shape[3];
  int ended = end[3];
  int p = 0;
  int start = 0;
  for (int i = begin[0]; i < ((end[0] > begin[0]) ? end[0] : (2 * begin[0] - end[0])); i += std::abs(strides[0])) {
    for (int j = begin[1]; j < ((end[1] > begin[1]) ? end[1] : (2 * begin[1] - end[1])); j += std::abs(strides[1])) {
      for (int k = begin[2]; k < ((end[2] > begin[2]) ? end[2] : (2 * begin[2] - end[2])); k += std::abs(strides[2])) {
        start = (strides[0] > 0 ? i : 2 * begin[0] - i) * block + (strides[1] > 0 ? j : 2 * begin[1] - j) * map +
                (strides[2] > 0 ? k : 2 * begin[2] - k) * w + begin[3];
        StridedSlice<<<GET_BLOCKS(input_size), GET_THREADS, 0, cuda_stream>>>(input, p, start, begin[3], strides[3],
                                                                              ended, output);
        p = p + (end[3] - 1 - begin[3]) / strides[3] + 1;
      }
    }
  }
}
template <typename T>
void CalStridedSliceGrad(const size_t input_size, const T* dy, const std::vector<int> in_shape,
                         const std::vector<int> begin, const std::vector<int> end, const std::vector<int> strides,
                         T* dx, cudaStream_t cuda_stream) {
  int block = in_shape[1] * in_shape[2] * in_shape[3];
  int map = in_shape[2] * in_shape[3];
  int w = in_shape[3];
  int ended = end[3];
  int p = 0;
  int start = 0;
  for (int i = begin[0]; i < ((end[0] > begin[0]) ? end[0] : (2 * begin[0] - end[0] + 1)); i += std::abs(strides[0])) {
    for (int j = begin[1]; j < ((end[1] > begin[1]) ? end[1] : (2 * begin[1] - end[1] + 1));
         j += std::abs(strides[1])) {
      for (int k = begin[2]; k < ((end[2] > begin[2]) ? end[2] : (2 * begin[2] - end[2] + 1));
           k += std::abs(strides[2])) {
        start = (strides[0] > 0 ? i : 2 * begin[0] - i) * block + (strides[1] > 0 ? j : 2 * begin[1] - j) * map +
                (strides[2] > 0 ? k : 2 * begin[2] - k) * w + begin[3];
        StridedSliceGrad<<<GET_BLOCKS(input_size), GET_THREADS, 0, cuda_stream>>>(dy, p, start, begin[3], strides[3],
                                                                                  ended, dx);
        p = p + (end[3] - 1 - begin[3]) / strides[3] + 1;
      }
    }
  }
}

template void FillDeviceArray<float>(const size_t input_size, float* addr, const float value, cudaStream_t cuda_stream);
template void Slice4DKernel(const int s1, const int s2, const int s3, const int s4,
                            const int l1, const int l2, const int l3, const int l4,
                            const int d1, const int d2, const int d3, const int d4,
                            const float *input, float *output, cudaStream_t stream);
template void CalSliceGrad<float>(const size_t input_size, const float* dy, const std::vector<int> in_shape,
                                  const std::vector<int> begin, const std::vector<int> size, float* output,
                                  cudaStream_t cuda_stream);
template void CalStridedSlice<float>(const size_t input_size, const float* input, const std::vector<int> in_shape,
                                     const std::vector<int> begin, const std::vector<int> end,
                                     const std::vector<int> strides, float* output, cudaStream_t cuda_stream);
template void CalStridedSliceGrad<float>(const size_t input_size, const float* dy, const std::vector<int> in_shape,
                                         const std::vector<int> begin, const std::vector<int> end,
                                         const std::vector<int> strides, float* dx, cudaStream_t cuda_stream);
template void FillDeviceArray<half>(const size_t input_size, half* addr, const float value, cudaStream_t cuda_stream);
template void Slice4DKernel(const int s1, const int s2, const int s3, const int s4,
                            const int l1, const int l2, const int l3, const int l4,
                            const int d1, const int d2, const int d3, const int d4,
                            const half *input, half *output, cudaStream_t stream);
template void CalSliceGrad<half>(const size_t input_size, const half* dy, const std::vector<int> in_shape,
                                 const std::vector<int> begin, const std::vector<int> size, half* output,
                                 cudaStream_t cuda_stream);
template void CalStridedSlice<half>(const size_t input_size, const half* input, const std::vector<int> in_shape,
                                    const std::vector<int> begin, const std::vector<int> end,
                                    const std::vector<int> strides, half* output, cudaStream_t cuda_stream);
template void CalStridedSliceGrad<half>(const size_t input_size, const half* dy, const std::vector<int> in_shape,
                                        const std::vector<int> begin, const std::vector<int> end,
                                        const std::vector<int> strides, half* dx, cudaStream_t cuda_stream);
template void FillDeviceArray<int>(const size_t input_size, int* addr, const float value, cudaStream_t cuda_stream);
template void Slice4DKernel(const int s1, const int s2, const int s3, const int s4,
                            const int l1, const int l2, const int l3, const int l4,
                            const int d1, const int d2, const int d3, const int d4,
                            const int *input, int *output, cudaStream_t stream);
template void CalSliceGrad<int>(const size_t input_size, const int* dy, const std::vector<int> in_shape,
                                const std::vector<int> begin, const std::vector<int> size, int* output,
                                cudaStream_t cuda_stream);
template void CalStridedSlice<int>(const size_t input_size, const int* input, const std::vector<int> in_shape,
                                   const std::vector<int> begin, const std::vector<int> end,
                                   const std::vector<int> strides, int* output, cudaStream_t cuda_stream);
template void CalStridedSliceGrad<int>(const size_t input_size, const int* dy, const std::vector<int> in_shape,
                                       const std::vector<int> begin, const std::vector<int> end,
                                       const std::vector<int> strides, int* dx, cudaStream_t cuda_stream);
