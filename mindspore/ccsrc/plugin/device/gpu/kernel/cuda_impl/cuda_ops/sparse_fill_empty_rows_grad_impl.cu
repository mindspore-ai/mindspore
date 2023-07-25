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
#include "sparse_fill_empty_rows_grad_impl.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/complex.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/util.cuh"

template <typename T>
__global__ void AssignValue(const size_t map_num, const int64_t *reverse_map, const T *grad_values, T *d_values,
                            bool *flag) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < map_num; pos += blockDim.x * gridDim.x) {
    d_values[pos] = grad_values[reverse_map[pos]];
    flag[reverse_map[pos]] = false;
  }
}

template <>
__global__ void AssignValue(const size_t map_num, const int64_t *reverse_map, const Complex<float> *grad_values,
                            Complex<float> *d_values, bool *flag) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < map_num; pos += blockDim.x * gridDim.x) {
    d_values[pos] = grad_values[reverse_map[pos]];
    flag[reverse_map[pos]] = false;
  }
}

template <>
__global__ void AssignValue(const size_t map_num, const int64_t *reverse_map, const Complex<double> *grad_values,
                            Complex<double> *d_values, bool *flag) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < map_num; pos += blockDim.x * gridDim.x) {
    d_values[pos] = grad_values[reverse_map[pos]];
    flag[reverse_map[pos]] = false;
  }
}

template <typename T>
__global__ void CmpValue(const size_t map_num, const size_t values_num, const T *grad_values, bool *flag,
                         void *temp_sum_val) {
  __shared__ T sum_val[1];
  if (threadIdx.x == 0) {
    sum_val[0] = static_cast<T>(0.0);
  }
  __syncthreads();
  T *sum_ptr = static_cast<T *>(temp_sum_val);
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < values_num; pos += blockDim.x * gridDim.x) {
    if (flag[pos]) {
      MsAtomicAdd(sum_val, grad_values[pos]);
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    sum_ptr[blockIdx.x] = *sum_val;
  }
  return;
}

template <>
__global__ void CmpValue(const size_t map_num, const size_t values_num, const half *grad_values, bool *flag,
                         void *temp_sum_val) {
  __shared__ float sum_val[1];
  if (threadIdx.x == 0) {
    sum_val[0] = 0;
  }
  __syncthreads();
  float *sum_ptr = static_cast<float *>(temp_sum_val);
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < values_num; pos += blockDim.x * gridDim.x) {
    if (flag[pos]) {
      MsAtomicAdd(sum_val, __half2float(grad_values[pos]));
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    sum_ptr[blockIdx.x] = *sum_val;
  }
  return;
}

template <>
__global__ void CmpValue(const size_t map_num, const size_t values_num, const float *grad_values, bool *flag,
                         void *temp_sum_val) {
  __shared__ double sum_val[1];
  if (threadIdx.x == 0) {
    sum_val[0] = 0.0;
  }
  __syncthreads();
  double *sum_ptr = static_cast<double *>(temp_sum_val);
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < values_num; pos += blockDim.x * gridDim.x) {
    if (flag[pos]) {
      MsAtomicAdd(sum_val, static_cast<double>(grad_values[pos]));
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    sum_ptr[blockIdx.x] = *sum_val;
  }
  return;
}

template <>
__global__ void CmpValue(const size_t map_num, const size_t values_num, const Complex<double> *grad_values, bool *flag,
                         void *temp_sum_val) {
  __shared__ Complex<double> sum_val[1];
  if (threadIdx.x == 0) {
    *sum_val = Complex<double>(0, 0);
  }
  __syncthreads();
  Complex<double> *sum_ptr = static_cast<Complex<double> *>(temp_sum_val);

  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < values_num; pos += blockDim.x * gridDim.x) {
    if (flag[pos]) {
      MsAtomicAdd(sum_val, grad_values[pos]);
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    sum_ptr[blockIdx.x] = *sum_val;
  }
  return;
}

template <>
__global__ void CmpValue(const size_t map_num, const size_t values_num, const Complex<float> *grad_values, bool *flag,
                         void *temp_sum_val) {
  __shared__ Complex<double> sum_val[1];
  if (threadIdx.x == 0) {
    *sum_val = Complex<double>(0, 0);
  }
  __syncthreads();
  Complex<double> *sum_ptr = static_cast<Complex<double> *>(temp_sum_val);
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < values_num; pos += blockDim.x * gridDim.x) {
    if (flag[pos]) {
      Complex<double> temp_grad_value;
      temp_grad_value.real(static_cast<double>(grad_values[pos].real()));
      temp_grad_value.imag(static_cast<double>(grad_values[pos].imag()));
      MsAtomicAdd(sum_val, temp_grad_value);
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    sum_ptr[blockIdx.x] = *sum_val;
  }
  return;
}

template <typename T>
__global__ void SumAll(T *d_default_value, void *sum_ptr, size_t length) {
  T *temp_sum = static_cast<T *>(sum_ptr);
  T sum_all = 0.0;
  for (size_t i = 0; i < length; i += 1) {
    sum_all += temp_sum[i];
  }
  *d_default_value = sum_all;
}

template <>
__global__ void SumAll(half *d_default_value, void *sum_ptr, size_t length) {
  float *temp_sum = static_cast<float *>(sum_ptr);
  float sum_all = 0;
  for (size_t i = 0; i < length; i += 1) {
    sum_all += temp_sum[i];
  }
  *d_default_value = __float2half(sum_all);
}

template <>
__global__ void SumAll(float *d_default_value, void *sum_ptr, size_t length) {
  double *temp_sum = static_cast<double *>(sum_ptr);
  double sum_all = 0.0;
  for (size_t i = 0; i < length; i += 1) {
    sum_all += temp_sum[i];
  }
  *d_default_value = static_cast<float>(sum_all);
}

template <>
__global__ void SumAll(Complex<float> *d_default_value, void *sum_ptr, size_t length) {
  Complex<double> *temp_sum = static_cast<Complex<double> *>(sum_ptr);
  Complex<double> sum_all = {0.0, 0.0};
  for (size_t i = 0; i < length; i += 1) {
    sum_all += temp_sum[i];
  }
  *d_default_value = static_cast<Complex<float>>(sum_all);
}

template <typename T>
cudaError_t CalFillRowsGrad(const size_t map_num, const size_t values_num, const int64_t *reverse_map,
                            const T *grad_values, T *d_values, T *d_default_value, bool *workspace_flag,
                            void *workspace_sum_val, const uint32_t &device_id, cudaStream_t cuda_stream) {
  cudaMemset(d_default_value, 0, sizeof(T));
  cudaMemset(workspace_flag, true, values_num * sizeof(bool));
  int thread_num_map_num = 256 < map_num ? 256 : map_num;
  cudaDeviceProp prop;
  (void)cudaGetDeviceProperties(&prop, device_id);
  int max_blocks = prop.multiProcessorCount;
  int block_num = 0;
  if (map_num > 0) {
    block_num = std::min(static_cast<int>(((map_num - 1) / thread_num_map_num) + 1), max_blocks);
    AssignValue<<<block_num, thread_num_map_num, 0, cuda_stream>>>(map_num, reverse_map, grad_values, d_values,
                                                                   workspace_flag);
  }
  int thread_num_values_num = 256 < values_num ? 256 : values_num;
  block_num = std::min(static_cast<int>(((values_num - 1) / thread_num_values_num) + 1), max_blocks);
  CmpValue<<<block_num, thread_num_values_num, 0, cuda_stream>>>(map_num, values_num, grad_values, workspace_flag,
                                                                 workspace_sum_val);
  SumAll<<<1, 1, 0, cuda_stream>>>(d_default_value, workspace_sum_val, block_num);
  return GetCudaStatus();
}
#define TEMPLATE_INSTANCE(DTYPE)                                                                                       \
  template CUDA_LIB_EXPORT cudaError_t CalFillRowsGrad<DTYPE>(                                                         \
    const size_t map_num, const size_t values_num, const int64_t *reverse_map, const DTYPE *grad_values,               \
    DTYPE *d_values, DTYPE *d_default_value, bool *workspace_flag, void *workspace_sum_val, const uint32_t &device_id, \
    cudaStream_t cuda_stream);

TEMPLATE_INSTANCE(bool)
TEMPLATE_INSTANCE(int8_t)
TEMPLATE_INSTANCE(int16_t)
TEMPLATE_INSTANCE(int)
TEMPLATE_INSTANCE(int64_t)
TEMPLATE_INSTANCE(uint8_t)
TEMPLATE_INSTANCE(uint16_t)
TEMPLATE_INSTANCE(uint32_t)
TEMPLATE_INSTANCE(uint64_t)
TEMPLATE_INSTANCE(half)
TEMPLATE_INSTANCE(float)
TEMPLATE_INSTANCE(double)
TEMPLATE_INSTANCE(Complex<float>)
TEMPLATE_INSTANCE(Complex<double>)
