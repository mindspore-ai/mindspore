/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include "addv2_impl.cuh"

template <typename T>
struct AddFunc {
  __device__ __host__ __forceinline__ T operator()(const T &lhs, const T &rhs) { return (lhs + rhs); }
  __device__ __host__ __forceinline__ Complex<T> operator()(const Complex<T> &lhs, const Complex<T> &rhs) {
    return (lhs + rhs);
  }
};

__device__ __forceinline__ int Index(const int &index, const int &dim) { return dim == 1 ? 0 : index; }
template <typename T, typename Func>
__global__ void Broadcast(const size_t l0, const size_t l1, const size_t l2, const size_t l3, const size_t l4,
                          const size_t l5, const size_t l6, const size_t r0, const size_t r1, const size_t r2,
                          const size_t r3, const size_t r4, const size_t r5, const size_t r6, const size_t d0,
                          const size_t d1, const size_t d2, const size_t d3, const size_t d4, const size_t d5,
                          const size_t d6, const T *x0, const T *x1, T *y) {
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
    y[pos] = Func()(x0[l_index], x1[r_index]);
  }
}

template <typename T>
cudaError_t CalAddV2(const size_t size, const std::vector<size_t> &x0_dims, const std::vector<size_t> &x1_dims,
              const std::vector<size_t> &y_dims, const T *x0, const T *x1, T *y, const uint32_t &device_id,
              cudaStream_t cuda_stream) {
  size_t size1 = 1;
  for (auto d : y_dims) {
    size1 *= d;
  }

  Broadcast<T, AddFunc<T>><<<(size1 + 255) / 256, 256, 0, cuda_stream>>>(
    x0_dims[0], x0_dims[1], x0_dims[2], x0_dims[3], x0_dims[4], x0_dims[5], x0_dims[6], x1_dims[0], x1_dims[1],
    x1_dims[2], x1_dims[3], x1_dims[4], x1_dims[5], x1_dims[6], y_dims[0], y_dims[1], y_dims[2], y_dims[3], y_dims[4],
    y_dims[5], y_dims[6], x0, x1, y);
  CHECK_CUDA_LAUNCH_SUCCESS();
}

template <typename T>
using Complex = mindspore::utils::Complex<T>;
template CUDA_LIB_EXPORT cudaError_t CalAddV2<float>(const size_t size, const std::vector<size_t> &x0_dims,
                                                     const std::vector<size_t> &x1_dims,
                                                     const std::vector<size_t> &y_dims, const float *x0,
                                                     const float *x1, float *y, const uint32_t &device_id,
                                                     cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalAddV2<double>(const size_t size, const std::vector<size_t> &x0_dims,
                                                      const std::vector<size_t> &x1_dims,
                                                      const std::vector<size_t> &y_dims, const double *x0,
                                                      const double *x1, double *y, const uint32_t &device_id,
                                                      cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalAddV2<half>(const size_t size, const std::vector<size_t> &x0_dims,
                                                    const std::vector<size_t> &x1_dims,
                                                    const std::vector<size_t> &y_dims, const half *x0, const half *x1,
                                                    half *y, const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalAddV2<int8_t>(const size_t size, const std::vector<size_t> &x0_dims,
                                                      const std::vector<size_t> &x1_dims,
                                                      const std::vector<size_t> &y_dims, const int8_t *x0,
                                                      const int8_t *x1, int8_t *y, const uint32_t &device_id,
                                                      cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalAddV2<int64_t>(const size_t size, const std::vector<size_t> &x0_dims,
                                                       const std::vector<size_t> &x1_dims,
                                                       const std::vector<size_t> &y_dims, const int64_t *x0,
                                                       const int64_t *x1, int64_t *y, const uint32_t &device_id,
                                                       cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalAddV2<int32_t>(const size_t size, const std::vector<size_t> &x0_dims,
                                                       const std::vector<size_t> &x1_dims,
                                                       const std::vector<size_t> &y_dims, const int32_t *x0,
                                                       const int32_t *x1, int32_t *y, const uint32_t &device_id,
                                                       cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalAddV2<int16_t>(const size_t size, const std::vector<size_t> &x0_dims,
                                                       const std::vector<size_t> &x1_dims,
                                                       const std::vector<size_t> &y_dims, const int16_t *x0,
                                                       const int16_t *x1, int16_t *y, const uint32_t &device_id,
                                                       cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalAddV2<uint8_t>(const size_t size, const std::vector<size_t> &x0_dims,
                                                       const std::vector<size_t> &x1_dims,
                                                       const std::vector<size_t> &y_dims, const uint8_t *x0,
                                                       const uint8_t *x1, uint8_t *y, const uint32_t &device_id,
                                                       cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalAddV2<uint64_t>(const size_t size, const std::vector<size_t> &x0_dims,
                                                        const std::vector<size_t> &x1_dims,
                                                        const std::vector<size_t> &y_dims, const uint64_t *x0,
                                                        const uint64_t *x1, uint64_t *y, const uint32_t &device_id,
                                                        cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalAddV2<uint32_t>(const size_t size, const std::vector<size_t> &x0_dims,
                                                        const std::vector<size_t> &x1_dims,
                                                        const std::vector<size_t> &y_dims, const uint32_t *x0,
                                                        const uint32_t *x1, uint32_t *y, const uint32_t &device_id,
                                                        cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalAddV2<uint16_t>(const size_t size, const std::vector<size_t> &x0_dims,
                                                        const std::vector<size_t> &x1_dims,
                                                        const std::vector<size_t> &y_dims, const uint16_t *x0,
                                                        const uint16_t *x1, uint16_t *y, const uint32_t &device_id,
                                                        cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalAddV2<Complex<float>>(const size_t size, const std::vector<size_t> &x0_dims,
                                                              const std::vector<size_t> &x1_dims,
                                                              const std::vector<size_t> &y_dims,
                                                              const Complex<float> *x0, const Complex<float> *x1,
                                                              Complex<float> *y, const uint32_t &device_id,
                                                              cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalAddV2<Complex<double>>(const size_t size, const std::vector<size_t> &x0_dims,
                                                               const std::vector<size_t> &x1_dims,
                                                               const std::vector<size_t> &y_dims,
                                                               const Complex<double> *x0, const Complex<double> *x1,
                                                               Complex<double> *y, const uint32_t &device_id,
                                                               cudaStream_t cuda_stream);

template <typename T, typename Func>
__global__ void ElewiseAddV2Kernel(const int nums, const T *x0, const T *x1, T *y) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < nums; pos += blockDim.x * gridDim.x) {
    y[pos] = Func()(x0[pos], x1[pos]);
  }
}

template <typename T>
cudaError_t ElewiseAddV2(const int &nums, const T *x0, const T *x1, T *y, cudaStream_t stream) {
  ElewiseAddV2Kernel<T, AddFunc<T>><<<(nums + 255) / 256, 256, 0, stream>>>(nums, x0, x1, y);
  CHECK_CUDA_LAUNCH_SUCCESS();
}

template CUDA_LIB_EXPORT cudaError_t ElewiseAddV2(const int &nums, const double *x0, const double *x1, double *y,
                                                  cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t ElewiseAddV2(const int &nums, const float *x0, const float *x1, float *y,
                                                  cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t ElewiseAddV2(const int &nums, const half *x0, const half *x1, half *y,
                                                  cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t ElewiseAddV2(const int &nums, const int32_t *x0, const int32_t *x1, int32_t *y,
                                                  cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t ElewiseAddV2(const int &nums, const int8_t *x0, const int8_t *x1, int8_t *y,
                                                  cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t ElewiseAddV2(const int &nums, const int64_t *x0, const int64_t *x1, int64_t *y,
                                                  cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t ElewiseAddV2(const int &nums, const int16_t *x0, const int16_t *x1, int16_t *y,
                                                  cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t ElewiseAddV2(const int &nums, const uint8_t *x0, const uint8_t *x1, uint8_t *y,
                                                  cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t ElewiseAddV2(const int &nums, const uint16_t *x0, const uint16_t *x1, uint16_t *y,
                                                  cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t ElewiseAddV2(const int &nums, const uint32_t *x0, const uint32_t *x1, uint32_t *y,
                                                  cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t ElewiseAddV2(const int &nums, const uint64_t *x0, const uint64_t *x1, uint64_t *y,
                                                  cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t ElewiseAddV2(const int &nums, const Complex<float> *x0, const Complex<float> *x1,
                                                  Complex<float> *y, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t ElewiseAddV2(const int &nums, const Complex<double> *x0, const Complex<double> *x1,
                                                  Complex<double> *y, cudaStream_t stream);
