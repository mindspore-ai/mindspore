/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include <numeric>
#include <functional>
#include "backend/kernel_compiler/gpu/cuda_impl/slice_copy_impl.cuh"

namespace {
constexpr size_t kMaxDim = 8;
}

template <typename T, size_t N>
class VectorWrapper {
 public:
  explicit VectorWrapper(const std::vector<T> &v) { std::copy(v.begin(), v.end(), data); }
  ~VectorWrapper() {}
  __device__ T& operator[](size_t index) { return data[index]; }

 private:
  T data[N];
};

template <typename T>
__global__ void CopySlicesKernel(VectorWrapper<int64_t, kMaxDim> begins, VectorWrapper<int64_t, kMaxDim> stride,
                                 VectorWrapper<size_t, kMaxDim> u, VectorWrapper<size_t, kMaxDim> u_offset,
                                 VectorWrapper<size_t, kMaxDim> o_offset, const T *update_addr, T *output_addr) {
  size_t update_num = u[0] * u_offset[0];

  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < update_num; pos += blockDim.x * gridDim.x) {
    size_t i = pos / (u_offset[0]) % u[0];
    size_t j = pos / (u_offset[1]) % u[1];
    size_t k = pos / (u_offset[2]) % u[2];
    size_t l = pos / (u_offset[3]) % u[3];
    size_t m = pos / (u_offset[4]) % u[4];
    size_t n = pos / (u_offset[5]) % u[5];
    size_t o = pos / (u[7]) % u[6];
    size_t p = pos % u[7];

    size_t output_idx = (i * stride[0] + begins[0]) * o_offset[0] + (j * stride[1] + begins[1]) * o_offset[1] +
                        (k * stride[2] + begins[2]) * o_offset[2] + (l * stride[3] + begins[3]) * o_offset[3] +
                        (m * stride[4] + begins[4]) * o_offset[4] + (n * stride[5] + begins[5]) * o_offset[5] +
                        (o * stride[6] + begins[6]) * o_offset[6] + (p * stride[7] + begins[7]);
    output_addr[output_idx] = update_addr[pos];
  }
}

std::vector<size_t> CalculateOffset(const std::vector<size_t> &shape) {
  std::vector<size_t> offset(kMaxDim);
  offset[7] = 1;
  offset[6] = offset[7] * shape[7];
  offset[5] = offset[6] * shape[6];
  offset[4] = offset[5] * shape[5];
  offset[3] = offset[4] * shape[4];
  offset[2] = offset[3] * shape[3];
  offset[1] = offset[2] * shape[2];
  offset[0] = offset[1] * shape[1];
  return offset;
}

template <typename T>
void CopySlices(const std::vector<size_t> &update_shape, const std::vector<int64_t> &begin,
                const std::vector<int64_t> &stride, const std::vector<size_t> &output_shape, const T *update, T *output,
                cudaStream_t cuda_stream) {
  size_t size = std::accumulate(update_shape.begin(), update_shape.end(), 1, std::multiplies<size_t>());

  VectorWrapper<size_t, kMaxDim> o_offset(CalculateOffset(output_shape));
  VectorWrapper<size_t, kMaxDim> u_offset(CalculateOffset(update_shape));

  VectorWrapper<int64_t, kMaxDim> begins(begin);
  VectorWrapper<int64_t, kMaxDim> strides(stride);
  VectorWrapper<size_t, kMaxDim> update_shapes(update_shape);

  CopySlicesKernel<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(begins, strides, update_shapes, u_offset,
                                                                      o_offset, update, output);
}

template void CopySlices(const std::vector<size_t> &update_shape, const std::vector<int64_t> &begin,
                         const std::vector<int64_t> &stride, const std::vector<size_t> &output_shape,
                         const bool *update, bool *output, cudaStream_t cuda_stream);
template void CopySlices(const std::vector<size_t> &update_shape, const std::vector<int64_t> &begin,
                         const std::vector<int64_t> &stride, const std::vector<size_t> &output_shape,
                         const double *update, double *output, cudaStream_t cuda_stream);
template void CopySlices(const std::vector<size_t> &update_shape, const std::vector<int64_t> &begin,
                         const std::vector<int64_t> &stride, const std::vector<size_t> &output_shape,
                         const float *update, float *output, cudaStream_t cuda_stream);
template void CopySlices(const std::vector<size_t> &update_shape, const std::vector<int64_t> &begin,
                         const std::vector<int64_t> &stride, const std::vector<size_t> &output_shape,
                         const half *update, half *output, cudaStream_t cuda_stream);
template void CopySlices(const std::vector<size_t> &update_shape, const std::vector<int64_t> &begin,
                         const std::vector<int64_t> &stride, const std::vector<size_t> &output_shape,
                         const int64_t *update, int64_t *output, cudaStream_t cuda_stream);
template void CopySlices(const std::vector<size_t> &update_shape, const std::vector<int64_t> &begin,
                         const std::vector<int64_t> &stride, const std::vector<size_t> &output_shape, const int *update,
                         int *output, cudaStream_t cuda_stream);
template void CopySlices(const std::vector<size_t> &update_shape, const std::vector<int64_t> &begin,
                         const std::vector<int64_t> &stride, const std::vector<size_t> &output_shape,
                         const short *update, short *output, cudaStream_t cuda_stream);  // NOLINT
template void CopySlices(const std::vector<size_t> &update_shape, const std::vector<int64_t> &begin,
                         const std::vector<int64_t> &stride, const std::vector<size_t> &output_shape,
                         const int8_t *update, int8_t *output, cudaStream_t cuda_stream);
template void CopySlices(const std::vector<size_t> &update_shape, const std::vector<int64_t> &begin,
                         const std::vector<int64_t> &stride, const std::vector<size_t> &output_shape,
                         const uint64_t *update, uint64_t *output, cudaStream_t cuda_stream);
template void CopySlices(const std::vector<size_t> &update_shape, const std::vector<int64_t> &begin,
                         const std::vector<int64_t> &stride, const std::vector<size_t> &output_shape,
                         const uint32_t *update, uint32_t *output, cudaStream_t cuda_stream);
template void CopySlices(const std::vector<size_t> &update_shape, const std::vector<int64_t> &begin,
                         const std::vector<int64_t> &stride, const std::vector<size_t> &output_shape,
                         const uint16_t *update, uint16_t *output, cudaStream_t cuda_stream);
template void CopySlices(const std::vector<size_t> &update_shape, const std::vector<int64_t> &begin,
                         const std::vector<int64_t> &stride, const std::vector<size_t> &output_shape,
                         const unsigned char *update, unsigned char *output, cudaStream_t cuda_stream);
template void CopySlices(const std::vector<size_t> &update_shape, const std::vector<int64_t> &begin,
                         const std::vector<int64_t> &stride, const std::vector<size_t> &output_shape,
                         const char *update, char *output, cudaStream_t cuda_stream);
