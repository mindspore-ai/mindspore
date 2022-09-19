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

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/scatter_impl.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/util.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/complex.h"

// Specializations of atomic div for complex types
__device__ inline Complex<float> ScatterDivComplex(Complex<float>* address, Complex<float> val) {
  auto ptr_addr = reinterpret_cast<float*>(address);
  float addr_real = (*address).real();
  float addr_imag = (*address).imag();
  float temp = (pow(val.real(), static_cast<float>(2)) + pow(val.imag(), static_cast<float>(2)));

  MsAtomicMul(ptr_addr, val.real());
  MsAtomicAdd(ptr_addr, addr_imag * val.imag());
  MsAtomicMul(ptr_addr + 1, val.real());
  MsAtomicSub(ptr_addr + 1, addr_real * val.imag());
  return Complex<float>(MsAtomicDiv(ptr_addr, temp),
                        MsAtomicDiv(ptr_addr + 1, temp));
}

__device__ inline Complex<double> ScatterDivComplex(Complex<double>* address, Complex<double> val) {
  auto ptr_addr = reinterpret_cast<double*>(address);
  double addr_real = (*address).real();
  double addr_imag = (*address).imag();
  double temp = (pow(val.real(), static_cast<double>(2)) + pow(val.imag(), static_cast<double>(2)));

  MsAtomicMul(ptr_addr, val.real());
  MsAtomicAdd(ptr_addr, addr_imag * val.imag());
  MsAtomicMul(ptr_addr + 1, val.real());
  MsAtomicSub(ptr_addr + 1, addr_real * val.imag());
  return Complex<double>(MsAtomicDiv(ptr_addr, temp),
                         MsAtomicDiv(ptr_addr + 1, temp));
}

// Specializations of atomic mul for complex types
__device__ inline Complex<float> ScatterMulComplex(Complex<float>* address, Complex<float> val) {
  auto ptr_addr = reinterpret_cast<float*>(address);
  float addr_real = (*address).real();
  float addr_imag = (*address).imag();
  MsAtomicMul(ptr_addr, val.real());
  MsAtomicMul(ptr_addr + 1, val.real());
  return Complex<float>(MsAtomicSub(ptr_addr, addr_imag * val.imag()),
                        MsAtomicAdd(ptr_addr + 1, addr_real * val.imag()));
}

__device__ inline Complex<double> ScatterMulComplex(Complex<double>* address, Complex<double> val) {
  auto ptr_addr = reinterpret_cast<double*>(address);
  double addr_real = (*address).real();
  double addr_imag = (*address).imag();
  MsAtomicMul(ptr_addr, val.real());
  MsAtomicMul(ptr_addr + 1, val.real());
  return Complex<double>(MsAtomicSub(ptr_addr, addr_imag * val.imag()),
                         MsAtomicAdd(ptr_addr + 1, addr_real * val.imag()));
}

template <typename T, typename S>
  __global__ void ScatterDivKernel(S size_limit, const size_t inner_size, const size_t updates_size, const S *indices,
                                  const T *updates, T *input) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < updates_size; pos += blockDim.x * gridDim.x) {
     const size_t index = pos / inner_size;
     const size_t offset = pos % inner_size;
     if (indices[index] < 0 || indices[index] >= size_limit) {
        continue;
      }
     const size_t current_pos = indices[index] * inner_size + offset;
     MsAtomicDiv(&input[current_pos], updates[pos]);
  }
}

__global__ void ScatterDivKernel(int size_limit, const size_t inner_size, const size_t updates_size, const int *indices,
                                  const Complex<float> *updates, Complex<float> *input) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < updates_size; pos += blockDim.x * gridDim.x) {
     const size_t index = pos / inner_size;
     const size_t offset = pos % inner_size;
     if (indices[index] < 0 || indices[index] >= size_limit) {
        continue;
      }
     const size_t current_pos = indices[index] * inner_size + offset;
     ScatterDivComplex(&input[current_pos], updates[pos]);
  }
}

__global__ void ScatterDivKernel(int64_t size_limit, const size_t inner_size, const size_t updates_size,
                                 const int64_t *indices,  const Complex<float> *updates, Complex<float> *input) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < updates_size; pos += blockDim.x * gridDim.x) {
     const size_t index = pos / inner_size;
     const size_t offset = pos % inner_size;
     if (indices[index] < 0 || indices[index] >= size_limit) {
         continue;
      }
     const size_t current_pos = indices[index] * inner_size + offset;
     ScatterDivComplex(&input[current_pos], updates[pos]);
  }
}

__global__ void ScatterDivKernel(int size_limit, const size_t inner_size, const size_t updates_size, const int *indices,
                                 const Complex<double> *updates, Complex<double> *input) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < updates_size; pos += blockDim.x * gridDim.x) {
     const size_t index = pos / inner_size;
     const size_t offset = pos % inner_size;
     if (indices[index] < 0 || indices[index] >= size_limit) {
        continue;
      }
     const size_t current_pos = indices[index] * inner_size + offset;
     ScatterDivComplex(&input[current_pos], updates[pos]);
  }
}

__global__ void ScatterDivKernel(int64_t size_limit, const size_t inner_size, const size_t updates_size,
                                 const int64_t *indices, const Complex<double> *updates, Complex<double> *input) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < updates_size; pos += blockDim.x * gridDim.x) {
     const size_t index = pos / inner_size;
     const size_t offset = pos % inner_size;
     if (indices[index] < 0 || indices[index] >= size_limit) {
        continue;
      }
     const size_t current_pos = indices[index] * inner_size + offset;
     ScatterDivComplex(&input[current_pos], updates[pos]);
  }
}

template <typename T, typename S>
  __global__ void ScatterMulKernel(S size_limit, const size_t inner_size, const size_t updates_size, const S *indices,
                                  const T *updates, T *input) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < updates_size; pos += blockDim.x * gridDim.x) {
     const size_t index = pos / inner_size;
     const size_t offset = pos % inner_size;
     if (indices[index] < 0 || indices[index] >= size_limit) {
        continue;
     }
     const size_t current_pos = indices[index] * inner_size + offset;
     MsAtomicMul(&input[current_pos], updates[pos]);
  }
}

__global__ void ScatterMulKernel(int size_limit, const size_t inner_size, const size_t updates_size, const int *indices,
                                 const Complex<float> *updates, Complex<float> *input) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < updates_size; pos += blockDim.x * gridDim.x) {
     const size_t index = pos / inner_size;
     const size_t offset = pos % inner_size;
     if (indices[index] < 0 || indices[index] >= size_limit) {
        continue;
      }
     const size_t current_pos = indices[index] * inner_size + offset;
     ScatterMulComplex(&input[current_pos], updates[pos]);
  }
}

__global__ void ScatterMulKernel(int64_t size_limit, const size_t inner_size, const size_t updates_size,
                                 const int64_t *indices, const Complex<float> *updates, Complex<float> *input) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < updates_size; pos += blockDim.x * gridDim.x) {
     const size_t index = pos / inner_size;
     const size_t offset = pos % inner_size;
     if (indices[index] < 0 || indices[index] >= size_limit) {
        continue;
      }
     const size_t current_pos = indices[index] * inner_size + offset;
     ScatterMulComplex(&input[current_pos], updates[pos]);
  }
}

__global__ void ScatterMulKernel(int size_limit, const size_t inner_size, const size_t updates_size, const int *indices,
                                 const Complex<double> *updates, Complex<double> *input) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < updates_size; pos += blockDim.x * gridDim.x) {
     const size_t index = pos / inner_size;
     const size_t offset = pos % inner_size;
     if (indices[index] < 0 || indices[index] >= size_limit) {
        continue;
      }
     const size_t current_pos = indices[index] * inner_size + offset;
     ScatterMulComplex(&input[current_pos], updates[pos]);
  }
}

__global__ void ScatterMulKernel(int64_t size_limit, const size_t inner_size, const size_t updates_size,
                                 const int64_t *indices, const Complex<double> *updates, Complex<double> *input) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < updates_size; pos += blockDim.x * gridDim.x) {
     const size_t index = pos / inner_size;
     const size_t offset = pos % inner_size;
     if (indices[index] < 0 || indices[index] >= size_limit) {
        continue;
      }
     const size_t current_pos = indices[index] * inner_size + offset;
     ScatterMulComplex(&input[current_pos], updates[pos]);
  }
}

template <typename T, typename S>
void Scatter(enum ScatterType func_type, S size_limit, const size_t &inner_size, const size_t &indices_size,
             const S *indices, const T *updates, T *input, const uint32_t &device_id, cudaStream_t cuda_stream) {
  const size_t updates_size = inner_size * indices_size;
  switch (func_type) {
    case SCATTER_DIV:
      return ScatterDivKernel<<<CUDA_BLOCKS(device_id, updates_size), CUDA_THREADS(device_id), 0, cuda_stream>>>(
        size_limit, inner_size, updates_size, indices, updates, input);
    case SCATTER_MUL:
      return ScatterMulKernel<<<CUDA_BLOCKS(device_id, updates_size), CUDA_THREADS(device_id), 0, cuda_stream>>>(
        size_limit, inner_size, updates_size, indices, updates, input);
    default:
      break;
  }
}

template <typename S>
void Scatter(enum ScatterType func_type, S size_limit, const size_t &inner_size, const size_t &indices_size,
             const S *indices, const Complex<float> *updates, Complex<float> *input, const uint32_t &device_id,
             cudaStream_t cuda_stream) {
  const size_t updates_size = inner_size * indices_size;
  switch (func_type) {
    case SCATTER_DIV:
      return ScatterDivKernel<<<CUDA_BLOCKS(device_id, updates_size), CUDA_THREADS(device_id), 0, cuda_stream>>>(
        size_limit, inner_size, updates_size, indices, updates, input);
    case SCATTER_MUL:
      return ScatterMulKernel<<<CUDA_BLOCKS(device_id, updates_size), CUDA_THREADS(device_id), 0, cuda_stream>>>(
        size_limit, inner_size, updates_size, indices, updates, input);
    default:
      break;
  }
}

template <typename S>
void Scatter(enum ScatterType func_type, S size_limit, const size_t &inner_size, const size_t &indices_size,
             const S *indices, const Complex<double> *updates, Complex<double> *input, const uint32_t &device_id,
             cudaStream_t cuda_stream) {
  const size_t updates_size = inner_size * indices_size;
  switch (func_type) {
    case SCATTER_DIV:
      return ScatterDivKernel<<<CUDA_BLOCKS(device_id, updates_size), CUDA_THREADS(device_id), 0, cuda_stream>>>(
        size_limit, inner_size, updates_size, indices, updates, input);
    case SCATTER_MUL:
      return ScatterMulKernel<<<CUDA_BLOCKS(device_id, updates_size), CUDA_THREADS(device_id), 0, cuda_stream>>>(
        size_limit, inner_size, updates_size, indices, updates, input);
    default:
      break;
  }
}

template CUDA_LIB_EXPORT void Scatter<float, int>(enum ScatterType func_type, int size_limit,
  const size_t &inner_size, const size_t &indices_size,
  const int *indices, const float *updates, float *input,
  const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Scatter<float, int64_t>(enum ScatterType func_type, int64_t size_limit,
      const size_t &inner_size, const size_t &indices_size,
      const int64_t *indices, const float *updates, float *input,
      const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Scatter<half, int>(enum ScatterType func_type, int size_limit,
 const size_t &inner_size, const size_t &indices_size,
 const int *indices, const half *updates, half *input,
 const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Scatter<half, int64_t>(enum ScatterType func_type, int64_t size_limit,
     const size_t &inner_size, const size_t &indices_size,
     const int64_t *indices, const half *updates, half *input,
     const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Scatter<double, int>(enum ScatterType func_type, int size_limit,
  const size_t &inner_size, const size_t &indices_size,
  const int *indices, const double *updates, double *input,
  const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Scatter<double, int64_t>(enum ScatterType func_type, int64_t size_limit,
      const size_t &inner_size, const size_t &indices_size,
      const int64_t *indices, const double *updates, double *input,
      const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Scatter<int8_t, int>(enum ScatterType func_type, int size_limit,
   const size_t &inner_size, const size_t &indices_size,
   const int *indices, const int8_t *updates, int8_t *input,
   const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Scatter<int8_t, int64_t>(enum ScatterType func_type, int64_t size_limit,
       const size_t &inner_size, const size_t &indices_size,
       const int64_t *indices, const int8_t *updates, int8_t *input,
       const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Scatter<unsigned char, int>(enum ScatterType func_type, int size_limit,
          const size_t &inner_size, const size_t &indices_size,
          const int *indices, const unsigned char *updates,
          unsigned char *input, const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Scatter<unsigned char, int64_t>(enum ScatterType func_type, int64_t size_limit,
              const size_t &inner_size, const size_t &indices_size,
              const int64_t *indices, const unsigned char *updates,
              unsigned char *input, const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Scatter<int16_t, int>(enum ScatterType func_type, int size_limit,
   const size_t &inner_size, const size_t &indices_size,
   const int *indices, const int16_t *updates, int16_t *input,
   const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Scatter<int16_t, int64_t>(enum ScatterType func_type, int64_t size_limit,
       const size_t &inner_size, const size_t &indices_size,
       const int64_t *indices, const int16_t *updates, int16_t *input,
       const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Scatter<uint16_t, int>(enum ScatterType func_type, int size_limit,
   const size_t &inner_size, const size_t &indices_size,
   const int *indices, const uint16_t *updates, uint16_t *input,
   const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Scatter<uint16_t, int64_t>(enum ScatterType func_type, int64_t size_limit,
       const size_t &inner_size, const size_t &indices_size,
       const int64_t *indices, const uint16_t *updates, uint16_t *input,
       const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Scatter<int, int>(enum ScatterType func_type, int size_limit,
   const size_t &inner_size, const size_t &indices_size,
   const int *indices, const int *updates, int *input,
   const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Scatter<int, int64_t>(enum ScatterType func_type, int64_t size_limit,
       const size_t &inner_size, const size_t &indices_size,
       const int64_t *indices, const int *updates, int *input,
       const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Scatter<uint32_t, int>(enum ScatterType func_type, int size_limit,
   const size_t &inner_size, const size_t &indices_size,
   const int *indices, const uint32_t *updates, uint32_t *input,
   const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Scatter<uint32_t, int64_t>(enum ScatterType func_type, int64_t size_limit,
       const size_t &inner_size, const size_t &indices_size,
       const int64_t *indices, const uint32_t *updates, uint32_t *input,
       const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Scatter<int64_t, int>(enum ScatterType func_type, int size_limit,
   const size_t &inner_size, const size_t &indices_size,
   const int *indices, const int64_t *updates, int64_t *input,
   const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Scatter<int64_t, int64_t>(enum ScatterType func_type, int64_t size_limit,
       const size_t &inner_size, const size_t &indices_size,
       const int64_t *indices, const int64_t *updates, int64_t *input,
       const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Scatter<uint64_t, int>(enum ScatterType func_type, int size_limit,
   const size_t &inner_size, const size_t &indices_size,
   const int *indices, const uint64_t *updates, uint64_t *input,
   const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Scatter<uint64_t, int64_t>(enum ScatterType func_type, int64_t size_limit,
       const size_t &inner_size, const size_t &indices_size,
       const int64_t *indices, const uint64_t *updates, uint64_t *input,
       const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Scatter<Complex<float>, int>(enum ScatterType func_type, int size_limit,
   const size_t &inner_size, const size_t &indices_size,
   const int *indices, const Complex<float> *updates, Complex<float> *input,
   const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Scatter<Complex<float>, int64_t>(enum ScatterType func_type, int64_t size_limit,
       const size_t &inner_size, const size_t &indices_size,
       const int64_t *indices, const Complex<float> *updates, Complex<float> *input,
       const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Scatter<Complex<double>, int>(enum ScatterType func_type, int size_limit,
   const size_t &inner_size, const size_t &indices_size,
   const int *indices, const Complex<double> *updates, Complex<double> *input,
   const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Scatter<Complex<double>, int64_t>(enum ScatterType func_type, int64_t size_limit,
       const size_t &inner_size, const size_t &indices_size,
       const int64_t *indices, const Complex<double> *updates, Complex<double> *input,
       const uint32_t &device_id, cudaStream_t cuda_stream);
