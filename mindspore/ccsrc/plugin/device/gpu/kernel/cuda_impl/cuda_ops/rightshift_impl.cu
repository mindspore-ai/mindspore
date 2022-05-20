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

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/rightshift_impl.cuh"

template <typename T>
__global__ void CalRightShiftKernel(size_t size, const T *inputx, const T *inputy, T *output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    T y1 = inputy[pos] <= static_cast<T>(0) ? static_cast<T>(0) : inputy[pos];
    output[pos] = inputx[pos] >> y1;
  }
}

__device__ __forceinline__ size_t Index(const size_t &index, const size_t &dim) { return dim == 1 ? 0 : index; }

template <typename T>
__global__ void BroadcastRightShiftKernel(const size_t l0, const size_t l1, const size_t l2, const size_t l3,
                                          const size_t l4, const size_t l5, const size_t l6, const size_t r0,
                                          const size_t r1, const size_t r2, const size_t r3, const size_t r4,
                                          const size_t r5, const size_t r6, const size_t d0, const size_t d1,
                                          const size_t d2, const size_t d3, const size_t d4, const size_t d5,
                                          const size_t d6, const T *inputx, const T *inputy, T *output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < d0 * d1 * d2 * d3 * d4 * d5 * d6;
       pos += blockDim.x * gridDim.x) {
    size_t i = pos / (d1 * d2 * d3 * d4 * d5 * d6) % d0;
    size_t j = pos / (d2 * d3 * d4 * d5 * d6) % d1;
    size_t k = pos / (d3 * d4 * d5 * d6) % d2;
    size_t l = pos / (d4 * d5 * d6) % d3;
    size_t m = pos / (d5 * d6) % d4;
    size_t n = pos / d6 % d5;
    size_t o = pos % d6;

    size_t l_index = Index(i, l0) * l1 * l2 * l3 * l4 * l5 * l6;
    l_index += Index(j, l1) * l2 * l3 * l4 * l5 * l6;
    l_index += Index(k, l2) * l3 * l4 * l5 * l6;
    l_index += Index(l, l3) * l4 * l5 * l6;
    l_index += Index(m, l4) * l5 * l6;
    l_index += Index(n, l5) * l6;
    l_index += Index(o, l6);
    size_t r_index = Index(i, r0) * r1 * r2 * r3 * r4 * r5 * r6;
    r_index += Index(j, r1) * r2 * r3 * r4 * r5 * r6;
    r_index += Index(k, r2) * r3 * r4 * r5 * r6;
    r_index += Index(l, r3) * r4 * r5 * r6;
    r_index += Index(m, r4) * r5 * r6;
    r_index += Index(n, r5) * r6;
    r_index += Index(o, r6);
    T y1 = inputy[r_index] <= static_cast<T>(0) ? static_cast<T>(0) : inputy[r_index];
    output[pos] = inputx[l_index] >> y1;
  }
}

template <typename T>
void CalRightShift(size_t size, const T *inputx, const T *inputy, T *output, const uint32_t &device_id,
                   cudaStream_t cuda_stream) {
  CalRightShiftKernel<<<CUDA_BLOCKS(device_id, size), CUDA_THREADS(device_id), 0, cuda_stream>>>(size, inputx, inputy,
                                                                                                 output);
}

template <typename T>
void BroadcastRightShift(const std::vector<size_t> &inputx_shape, const std::vector<size_t> &inputy_shape,
                         const std::vector<size_t> &output_shape, const T *inputx, const T *inputy, T *output,
                         const uint32_t &device_id, cudaStream_t cuda_stream) {
  size_t size = 1;
  for (auto d : output_shape) {
    size *= d;
  }
  BroadcastRightShiftKernel<<<CUDA_BLOCKS(device_id, size), CUDA_THREADS(device_id), 0, cuda_stream>>>(
    inputx_shape[0], inputx_shape[1], inputx_shape[2], inputx_shape[3], inputx_shape[4], inputx_shape[5],
    inputx_shape[6], inputy_shape[0], inputy_shape[1], inputy_shape[2], inputy_shape[3], inputy_shape[4],
    inputy_shape[5], inputy_shape[6], output_shape[0], output_shape[1], output_shape[2], output_shape[3],
    output_shape[4], output_shape[5], output_shape[6], inputx, inputy, output);
}

template CUDA_LIB_EXPORT void CalRightShift<int8_t>(size_t, const int8_t *, const int8_t *, int8_t *, const uint32_t &,
                                                    cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalRightShift<int16_t>(size_t, const int16_t *, const int16_t *, int16_t *,
                                                     const uint32_t &, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalRightShift<int32_t>(size_t, const int32_t *, const int32_t *, int32_t *,
                                                     const uint32_t &, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalRightShift<int64_t>(size_t, const int64_t *, const int64_t *, int64_t *,
                                                     const uint32_t &, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalRightShift<uint8_t>(size_t, const uint8_t *, const uint8_t *, uint8_t *,
                                                     const uint32_t &, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalRightShift<uint16_t>(size_t, const uint16_t *, const uint16_t *, uint16_t *,
                                                      const uint32_t &, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalRightShift<uint32_t>(size_t, const uint32_t *, const uint32_t *, uint32_t *,
                                                      const uint32_t &, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalRightShift<uint64_t>(size_t, const uint64_t *, const uint64_t *, uint64_t *,
                                                      const uint32_t &, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void BroadcastRightShift<int8_t>(const std::vector<size_t> &, const std::vector<size_t> &,
                                                          const std::vector<size_t> &, const int8_t *, const int8_t *,
                                                          int8_t *, const uint32_t &, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void BroadcastRightShift<int16_t>(const std::vector<size_t> &, const std::vector<size_t> &,
                                                           const std::vector<size_t> &, const int16_t *,
                                                           const int16_t *, int16_t *, const uint32_t &,
                                                           cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void BroadcastRightShift<int32_t>(const std::vector<size_t> &, const std::vector<size_t> &,
                                                           const std::vector<size_t> &, const int32_t *,
                                                           const int32_t *, int32_t *, const uint32_t &,
                                                           cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void BroadcastRightShift<int64_t>(const std::vector<size_t> &, const std::vector<size_t> &,
                                                           const std::vector<size_t> &, const int64_t *,
                                                           const int64_t *, int64_t *, const uint32_t &,
                                                           cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void BroadcastRightShift<uint8_t>(const std::vector<size_t> &, const std::vector<size_t> &,
                                                           const std::vector<size_t> &, const uint8_t *,
                                                           const uint8_t *, uint8_t *, const uint32_t &,
                                                           cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void BroadcastRightShift<uint16_t>(const std::vector<size_t> &, const std::vector<size_t> &,
                                                            const std::vector<size_t> &, const uint16_t *,
                                                            const uint16_t *, uint16_t *, const uint32_t &,
                                                            cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void BroadcastRightShift<uint32_t>(const std::vector<size_t> &, const std::vector<size_t> &,
                                                            const std::vector<size_t> &, const uint32_t *,
                                                            const uint32_t *, uint32_t *, const uint32_t &,
                                                            cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void BroadcastRightShift<uint64_t>(const std::vector<size_t> &, const std::vector<size_t> &,
                                                            const std::vector<size_t> &, const uint64_t *,
                                                            const uint64_t *, uint64_t *, const uint32_t &,
                                                            cudaStream_t cuda_stream);
