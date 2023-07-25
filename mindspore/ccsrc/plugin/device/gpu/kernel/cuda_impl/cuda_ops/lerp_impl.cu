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

#include <cuda_runtime.h>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include "lerp_impl.cuh"

__constant__ size_t start_dim_cal[5];
__constant__ size_t end_dim_cal[5];
__constant__ size_t weight_dim_cal[5];
__constant__ size_t output_dim_cal[5];

template <typename T, typename S>
__global__ void LerpFloatKernel(const size_t size, const T *start, const T *end, const S *weight, T *output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += gridDim.x * blockDim.x) {
    output[pos] = start[pos] + (*weight) * (end[pos] - start[pos]);
  }
}

template <>
__global__ void LerpFloatKernel(const size_t size, const half *start, const half *end, const float *weight,
                                half *output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += gridDim.x * blockDim.x) {
    output[pos] = __half2float(start[pos]) + (*weight) * (__half2float(end[pos]) - __half2float(start[pos]));
  }
}

__device__ __forceinline__ size_t Index(const size_t &index, const size_t &dim) { return dim == 1 ? 0 : index; }

template <typename T, typename S>
__global__ void BroadcastLerpWeightFloatKernel(const size_t l0, const size_t l1, const size_t l2, const size_t l3,
                                               const size_t l4, const size_t l5, const size_t l6, const size_t r0,
                                               const size_t r1, const size_t r2, const size_t r3, const size_t r4,
                                               const size_t r5, const size_t r6, const size_t d0, const size_t d1,
                                               const size_t d2, const size_t d3, const size_t d4, const size_t d5,
                                               const size_t d6, const T *inputx, const T *inputy, const S *weight,
                                               T *output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < d0 * d1 * d2 * d3 * d4 * d5 * d6;
       pos += blockDim.x * gridDim.x) {
    size_t i = pos / output_dim_cal[0] % d0;
    size_t j = pos / output_dim_cal[1] % d1;
    size_t k = pos / output_dim_cal[2] % d2;
    size_t l = pos / output_dim_cal[3] % d3;
    size_t m = pos / output_dim_cal[4] % d4;
    size_t n = pos / d6 % d5;
    size_t o = pos % d6;

    size_t l_index = Index(i, l0) * start_dim_cal[0];
    l_index += Index(j, l1) * start_dim_cal[1];
    l_index += Index(k, l2) * start_dim_cal[2];
    l_index += Index(l, l3) * start_dim_cal[3];
    l_index += Index(m, l4) * start_dim_cal[4];
    l_index += Index(n, l5) * l6;
    l_index += Index(o, l6);
    size_t r_index = Index(i, r0) * end_dim_cal[0];
    r_index += Index(j, r1) * end_dim_cal[1];
    r_index += Index(k, r2) * end_dim_cal[2];
    r_index += Index(l, r3) * end_dim_cal[3];
    r_index += Index(m, r4) * end_dim_cal[4];
    r_index += Index(n, r5) * r6;
    r_index += Index(o, r6);
    output[pos] = inputx[l_index] + (*weight) * (inputy[r_index] - inputx[l_index]);
  }
}

template <>
__global__ void BroadcastLerpWeightFloatKernel(const size_t l0, const size_t l1, const size_t l2, const size_t l3,
                                               const size_t l4, const size_t l5, const size_t l6, const size_t r0,
                                               const size_t r1, const size_t r2, const size_t r3, const size_t r4,
                                               const size_t r5, const size_t r6, const size_t d0, const size_t d1,
                                               const size_t d2, const size_t d3, const size_t d4, const size_t d5,
                                               const size_t d6, const half *inputx, const half *inputy,
                                               const float *weight, half *output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < d0 * d1 * d2 * d3 * d4 * d5 * d6;
       pos += blockDim.x * gridDim.x) {
    size_t i = pos / output_dim_cal[0] % d0;
    size_t j = pos / output_dim_cal[1] % d1;
    size_t k = pos / output_dim_cal[2] % d2;
    size_t l = pos / output_dim_cal[3] % d3;
    size_t m = pos / output_dim_cal[4] % d4;
    size_t n = pos / d6 % d5;
    size_t o = pos % d6;

    size_t l_index = Index(i, l0) * start_dim_cal[0];
    l_index += Index(j, l1) * start_dim_cal[1];
    l_index += Index(k, l2) * start_dim_cal[2];
    l_index += Index(l, l3) * start_dim_cal[3];
    l_index += Index(m, l4) * start_dim_cal[4];
    l_index += Index(n, l5) * l6;
    l_index += Index(o, l6);
    size_t r_index = Index(i, r0) * end_dim_cal[0];
    r_index += Index(j, r1) * end_dim_cal[1];
    r_index += Index(k, r2) * end_dim_cal[2];
    r_index += Index(l, r3) * end_dim_cal[3];
    r_index += Index(m, r4) * end_dim_cal[4];
    r_index += Index(n, r5) * r6;
    r_index += Index(o, r6);
    output[pos] =
      __half2float(inputx[l_index]) + (*weight) * (__half2float(inputy[r_index]) - __half2float(inputx[l_index]));
  }
}

template <typename T, typename S>
__global__ void LerpTensorKernel(const size_t size, const T *start, const T *end, const S *weight, T *output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += gridDim.x * blockDim.x) {
    output[pos] = start[pos] + weight[pos] * (end[pos] - start[pos]);
  }
}

// this condition is actually wrong and won't be invoked(after ops/lerp.cc),just for kernel_attr's unification
template <>
__global__ void LerpTensorKernel(const size_t size, const half *start, const half *end, const float *weight,
                                 half *output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += gridDim.x * blockDim.x) {
    output[pos] = __half2float(start[pos]) + weight[pos] * (__half2float(end[pos]) - __half2float(start[pos]));
  }
}
template <typename T, typename S>
__global__ void BroadcastLerpWeightTensorKernel(const size_t l0, const size_t l1, const size_t l2, const size_t l3,
                                                const size_t l4, const size_t l5, const size_t l6, const size_t r0,
                                                const size_t r1, const size_t r2, const size_t r3, const size_t r4,
                                                const size_t r5, const size_t r6, const size_t w0, const size_t w1,
                                                const size_t w2, const size_t w3, const size_t w4, const size_t w5,
                                                const size_t w6, const size_t d0, const size_t d1, const size_t d2,
                                                const size_t d3, const size_t d4, const size_t d5, const size_t d6,
                                                const T *inputx, const T *inputy, const S *weight, T *output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < d0 * d1 * d2 * d3 * d4 * d5 * d6;
       pos += blockDim.x * gridDim.x) {
    size_t i = pos / output_dim_cal[0] % d0;
    size_t j = pos / output_dim_cal[1] % d1;
    size_t k = pos / output_dim_cal[2] % d2;
    size_t l = pos / output_dim_cal[3] % d3;
    size_t m = pos / output_dim_cal[4] % d4;
    size_t n = pos / d6 % d5;
    size_t o = pos % d6;

    size_t l_index = Index(i, l0) * start_dim_cal[0];
    l_index += Index(j, l1) * start_dim_cal[1];
    l_index += Index(k, l2) * start_dim_cal[2];
    l_index += Index(l, l3) * start_dim_cal[3];
    l_index += Index(m, l4) * start_dim_cal[4];
    l_index += Index(n, l5) * l6;
    l_index += Index(o, l6);
    size_t r_index = Index(i, r0) * end_dim_cal[0];
    r_index += Index(j, r1) * end_dim_cal[1];
    r_index += Index(k, r2) * end_dim_cal[2];
    r_index += Index(l, r3) * end_dim_cal[3];
    r_index += Index(m, r4) * end_dim_cal[4];
    r_index += Index(n, r5) * r6;
    r_index += Index(o, r6);
    size_t w_index = Index(i, w0) * weight_dim_cal[0];
    w_index += Index(j, w1) * weight_dim_cal[1];
    w_index += Index(k, w2) * weight_dim_cal[2];
    w_index += Index(l, w3) * weight_dim_cal[3];
    w_index += Index(m, w4) * weight_dim_cal[4];
    w_index += Index(n, w5) * w6;
    w_index += Index(o, w6);
    output[pos] = inputx[l_index] + weight[w_index] * (inputy[r_index] - inputx[l_index]);
  }
}

// same as LerpTensorkernel<half, float>, it won't be invoked, just for template unification;
template <>
__global__ void BroadcastLerpWeightTensorKernel(
  const size_t l0, const size_t l1, const size_t l2, const size_t l3, const size_t l4, const size_t l5, const size_t l6,
  const size_t r0, const size_t r1, const size_t r2, const size_t r3, const size_t r4, const size_t r5, const size_t r6,
  const size_t w0, const size_t w1, const size_t w2, const size_t w3, const size_t w4, const size_t w5, const size_t w6,
  const size_t d0, const size_t d1, const size_t d2, const size_t d3, const size_t d4, const size_t d5, const size_t d6,
  const half *inputx, const half *inputy, const float *weight, half *output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < d0 * d1 * d2 * d3 * d4 * d5 * d6;
       pos += blockDim.x * gridDim.x) {
    size_t i = pos / output_dim_cal[0] % d0;
    size_t j = pos / output_dim_cal[1] % d1;
    size_t k = pos / output_dim_cal[2] % d2;
    size_t l = pos / output_dim_cal[3] % d3;
    size_t m = pos / output_dim_cal[4] % d4;
    size_t n = pos / d6 % d5;
    size_t o = pos % d6;

    size_t l_index = Index(i, l0) * start_dim_cal[0];
    l_index += Index(j, l1) * start_dim_cal[1];
    l_index += Index(k, l2) * start_dim_cal[2];
    l_index += Index(l, l3) * start_dim_cal[3];
    l_index += Index(m, l4) * start_dim_cal[4];
    l_index += Index(n, l5) * l6;
    l_index += Index(o, l6);
    size_t r_index = Index(i, r0) * end_dim_cal[0];
    r_index += Index(j, r1) * end_dim_cal[1];
    r_index += Index(k, r2) * end_dim_cal[2];
    r_index += Index(l, r3) * end_dim_cal[3];
    r_index += Index(m, r4) * end_dim_cal[4];
    r_index += Index(n, r5) * r6;
    r_index += Index(o, r6);
    size_t w_index = Index(i, w0) * weight_dim_cal[0];
    w_index += Index(j, w1) * weight_dim_cal[1];
    w_index += Index(k, w2) * weight_dim_cal[2];
    w_index += Index(l, w3) * weight_dim_cal[3];
    w_index += Index(m, w4) * weight_dim_cal[4];
    w_index += Index(n, w5) * w6;
    w_index += Index(o, w6);
    output[pos] =
      __half2float(inputx[l_index]) + weight[w_index] * (__half2float(inputy[r_index]) - __half2float(inputx[l_index]));
  }
}

void CalDimData(const std::vector<size_t> &start_shape, size_t *output) {
  output[4] = start_shape[5] * start_shape[6];
  output[3] = output[4] * start_shape[4];
  output[2] = output[3] * start_shape[3];
  output[1] = output[2] * start_shape[2];
  output[0] = output[1] * start_shape[1];
}

template <typename T, typename S>
cudaError_t LerpWeightFloat(const size_t input_size, const T *start, const T *end, const S *weight, T *output,
                            const uint32_t &device_id, cudaStream_t cuda_stream) {
  LerpFloatKernel<<<CUDA_BLOCKS(device_id, input_size), CUDA_THREADS(device_id), 0, cuda_stream>>>(input_size, start,
                                                                                                   end, weight, output);
  return GetCudaStatus();
}

template <typename T, typename S>
cudaError_t BroadcastLerpWeightFloat(const std::vector<size_t> &start_shape, const std::vector<size_t> &end_shape,
                              const std::vector<size_t> &output_shape, const T *start, const T *end, const S *weight,
                              T *output, const uint32_t &device_id, cudaStream_t cuda_stream) {
  size_t size = 1;
  for (auto d : output_shape) {
    size *= d;
  }
  size_t start_dim[5];
  size_t end_dim[5];
  size_t output_dim[5];
  CalDimData(start_shape, start_dim);
  CalDimData(end_shape, end_dim);
  CalDimData(output_shape, output_dim);
  cudaMemcpyToSymbol(start_dim_cal, start_dim, sizeof(size_t) * 5);
  cudaMemcpyToSymbol(end_dim_cal, end_dim, sizeof(size_t) * 5);
  cudaMemcpyToSymbol(output_dim_cal, output_dim, sizeof(size_t) * 5);
  BroadcastLerpWeightFloatKernel<<<CUDA_BLOCKS(device_id, size), CUDA_THREADS(device_id), 0, cuda_stream>>>(
    start_shape[0], start_shape[1], start_shape[2], start_shape[3], start_shape[4], start_shape[5], start_shape[6],
    end_shape[0], end_shape[1], end_shape[2], end_shape[3], end_shape[4], end_shape[5], end_shape[6], output_shape[0],
    output_shape[1], output_shape[2], output_shape[3], output_shape[4], output_shape[5], output_shape[6], start, end,
    weight, output);
  return GetCudaStatus();
}

template <typename T, typename S>
cudaError_t LerpWeightTensor(const size_t input_size, const T *start, const T *end, const S *weight, T *output,
                      const uint32_t &device_id, cudaStream_t cuda_stream) {
  LerpTensorKernel<<<CUDA_BLOCKS(device_id, input_size), CUDA_THREADS(device_id), 0, cuda_stream>>>(
    input_size, start, end, weight, output);
  return GetCudaStatus();
}

template <typename T, typename S>
cudaError_t BroadcastLerpWeightTensor(const std::vector<size_t> &start_shape, const std::vector<size_t> &end_shape,
                               const std::vector<size_t> &weight_shape, const std::vector<size_t> &output_shape,
                               const T *start, const T *end, const S *weight, T *output, const uint32_t &device_id,
                               cudaStream_t cuda_stream) {
  size_t size = 1;
  for (auto d : output_shape) {
    size *= d;
  }
  size_t start_dim[5];
  size_t end_dim[5];
  size_t weight_dim[5];
  size_t output_dim[5];
  CalDimData(start_shape, start_dim);
  CalDimData(end_shape, end_dim);
  CalDimData(weight_shape, weight_dim);
  CalDimData(output_shape, output_dim);
  cudaMemcpyToSymbol(start_dim_cal, start_dim, sizeof(size_t) * 5);
  cudaMemcpyToSymbol(end_dim_cal, end_dim, sizeof(size_t) * 5);
  cudaMemcpyToSymbol(output_dim_cal, output_dim, sizeof(size_t) * 5);
  cudaMemcpyToSymbol(weight_dim_cal, weight_dim, sizeof(size_t) * 5);
  BroadcastLerpWeightTensorKernel<<<CUDA_BLOCKS(device_id, size), CUDA_THREADS(device_id), 0, cuda_stream>>>(
    start_shape[0], start_shape[1], start_shape[2], start_shape[3], start_shape[4], start_shape[5], start_shape[6],
    end_shape[0], end_shape[1], end_shape[2], end_shape[3], end_shape[4], end_shape[5], end_shape[6], weight_shape[0],
    weight_shape[1], weight_shape[2], weight_shape[3], weight_shape[4], weight_shape[5], weight_shape[6],
    output_shape[0], output_shape[1], output_shape[2], output_shape[3], output_shape[4], output_shape[5],
    output_shape[6], start, end, weight, output);
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t LerpWeightFloat<float, float>(const size_t input_size, const float *start,
                                                            const float *end, const float *weight, float *output,
                                                            const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t LerpWeightFloat<half, float>(const size_t input_size, const half *start,
                                                           const half *end, const float *weight, half *output,
                                                           const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t LerpWeightFloat<double, float>(const size_t input_size, const double *start,
                                                             const double *end, const float *weight, double *output,
                                                             const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t LerpWeightFloat<half, half>(const size_t input_size, const half *start,
                                                           const half *end, const half *weight, half *output,
                                                           const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t LerpWeightFloat<double, double>(const size_t input_size, const double *start,
                                                              const double *end, const double *weight, double *output,
                                                              const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t BroadcastLerpWeightFloat<float, float>(
  const std::vector<size_t> &start_shape, const std::vector<size_t> &end_shape, const std::vector<size_t> &output_shape,
  const float *start, const float *end, const float *weight, float *output, const uint32_t &device_id,
  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t BroadcastLerpWeightFloat<half, float>(
  const std::vector<size_t> &start_shape, const std::vector<size_t> &end_shape, const std::vector<size_t> &output_shape,
  const half *start, const half *end, const float *weight, half *output, const uint32_t &device_id,
  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t BroadcastLerpWeightFloat<double, float>(
  const std::vector<size_t> &start_shape, const std::vector<size_t> &end_shape, const std::vector<size_t> &output_shape,
  const double *start, const double *end, const float *weight, double *output, const uint32_t &device_id,
  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t BroadcastLerpWeightFloat<half, half>(const std::vector<size_t> &start_shape,
                                                                   const std::vector<size_t> &end_shape,
                                                                   const std::vector<size_t> &output_shape,
                                                                   const half *start, const half *end,
                                                                   const half *weight, half *output,
                                                                   const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t BroadcastLerpWeightFloat<double, double>(
  const std::vector<size_t> &start_shape, const std::vector<size_t> &end_shape, const std::vector<size_t> &output_shape,
  const double *start, const double *end, const double *weight, double *output, const uint32_t &device_id,
  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t LerpWeightTensor<half, float>(const size_t input_size, const half *start,
                                                            const half *end, const float *weight, half *output,
                                                            const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t LerpWeightTensor<double, float>(const size_t input_size, const double *start,
                                                              const double *end, const float *weight, double *output,
                                                              const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t LerpWeightTensor<float, float>(const size_t input_size, const float *start,
                                                             const float *end, const float *weight, float *output,
                                                             const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t LerpWeightTensor<half, half>(const size_t input_size, const half *start,
                                                            const half *end, const half *weight, half *output,
                                                            const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t LerpWeightTensor<double, double>(const size_t input_size, const double *start,
                                                               const double *end, const double *weight, double *output,
                                                               const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t BroadcastLerpWeightTensor<half, float>(
  const std::vector<size_t> &start_shape, const std::vector<size_t> &end_shape, const std::vector<size_t> &weight_shape,
  const std::vector<size_t> &output_shape, const half *start, const half *end, const float *weight, half *output,
  const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t BroadcastLerpWeightTensor<double, float>(
  const std::vector<size_t> &start_shape, const std::vector<size_t> &end_shape, const std::vector<size_t> &weight_shape,
  const std::vector<size_t> &output_shape, const double *start, const double *end, const float *weight, double *output,
  const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t BroadcastLerpWeightTensor<float, float>(
  const std::vector<size_t> &start_shape, const std::vector<size_t> &end_shape, const std::vector<size_t> &weight_shape,
  const std::vector<size_t> &output_shape, const float *start, const float *end, const float *weight, float *output,
  const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t BroadcastLerpWeightTensor<half, half>(
  const std::vector<size_t> &start_shape, const std::vector<size_t> &end_shape, const std::vector<size_t> &weight_shape,
  const std::vector<size_t> &output_shape, const half *start, const half *end, const half *weight, half *output,
  const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t BroadcastLerpWeightTensor<double, double>(
  const std::vector<size_t> &start_shape, const std::vector<size_t> &end_shape, const std::vector<size_t> &weight_shape,
  const std::vector<size_t> &output_shape, const double *start, const double *end, const double *weight, double *output,
  const uint32_t &device_id, cudaStream_t cuda_stream);
