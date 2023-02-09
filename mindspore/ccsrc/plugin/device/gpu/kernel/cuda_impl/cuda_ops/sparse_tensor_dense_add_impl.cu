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

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/sparse_tensor_dense_add_impl.cuh"
#include "include/cuda_fp16.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/complex.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/util.cuh"

template <typename T>
using Complex = mindspore::utils::Complex<T>;

template <typename T>
__device__ void AtomicFunc(T *y_addr, T x1_values_addr) {
  MsAtomicAdd(y_addr, x1_values_addr);
}

template <>
__device__ void AtomicFunc(Complex<double> *y_addr, Complex<double> x1_values_addr) {
  auto real_new_byte = reinterpret_cast<double *>(y_addr);
  auto image_new_byte = real_new_byte + 1;
  MsAtomicAdd(real_new_byte, x1_values_addr.real());
  MsAtomicAdd(image_new_byte, x1_values_addr.imag());
}

template <typename T, typename I>
__global__ void SparseTensorDenseAddKernelFunc(size_t input_elements, size_t rank, size_t *x2_shape, I *x1_indices_addr,
                                               T *x1_values_addr, I *x1_shape_addr, T *x2_values_addr, T *y_addr) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < (input_elements); pos += blockDim.x * gridDim.x) {
    int out_index = 0;
    for (size_t j = 0; j < rank; j++) {
      int index = x1_indices_addr[pos * rank + j];
      int count = 1;
      for (size_t k = j + 1; k < rank; k++) {
        count *= x1_shape_addr[k];
      }
      out_index += index * count;
    }
    AtomicFunc(&y_addr[out_index], x1_values_addr[pos]);
  }
}

template <typename T, typename I>
void SparseTensorDenseAddKernel(size_t input_elements, size_t rank, size_t *x2_shape, I *x1_indices_addr,
                                T *x1_values_addr, I *x1_shape_addr, T *x2_values_addr, T *y_addr,
                                const uint32_t &device_id, cudaStream_t cuda_stream) {
  SparseTensorDenseAddKernelFunc<<<CUDA_BLOCKS(device_id, input_elements), CUDA_THREADS(device_id), 0,
   cuda_stream>>>(input_elements, rank, x2_shape, x1_indices_addr, x1_values_addr, x1_shape_addr, x2_values_addr,
                  y_addr);
}

template CUDA_LIB_EXPORT void SparseTensorDenseAddKernel(size_t input_elements, size_t rank, size_t *x2_shape,
                                                         int64_t *x1_indices_addr, uint8_t *x1_values_addr,
                                                         int64_t *x1_shape_addr, uint8_t *x2_values_addr,
                                                         uint8_t *y_addr, const uint32_t &device_id,
                                                         cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void SparseTensorDenseAddKernel(size_t input_elements, size_t rank, size_t *x2_shape,
                                                         int32_t *x1_indices_addr, uint8_t *x1_values_addr,
                                                         int32_t *x1_shape_addr, uint8_t *x2_values_addr,
                                                         uint8_t *y_addr, const uint32_t &device_id,
                                                         cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void SparseTensorDenseAddKernel(size_t input_elements, size_t rank, size_t *x2_shape,
                                                         int64_t *x1_indices_addr, int8_t *x1_values_addr,
                                                         int64_t *x1_shape_addr, int8_t *x2_values_addr,
                                                         int8_t *y_addr, const uint32_t &device_id,
                                                         cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void SparseTensorDenseAddKernel(size_t input_elements, size_t rank, size_t *x2_shape,
                                                         int32_t *x1_indices_addr, int8_t *x1_values_addr,
                                                         int32_t *x1_shape_addr, int8_t *x2_values_addr,
                                                         int8_t *y_addr, const uint32_t &device_id,
                                                         cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void SparseTensorDenseAddKernel(size_t input_elements, size_t rank, size_t *x2_shape,
                                                         int64_t *x1_indices_addr, uint16_t *x1_values_addr,
                                                         int64_t *x1_shape_addr, uint16_t *x2_values_addr,
                                                         uint16_t *y_addr, const uint32_t &device_id,
                                                         cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void SparseTensorDenseAddKernel(size_t input_elements, size_t rank, size_t *x2_shape,
                                                         int32_t *x1_indices_addr, uint16_t *x1_values_addr,
                                                         int32_t *x1_shape_addr, uint16_t *x2_values_addr,
                                                         uint16_t *y_addr, const uint32_t &device_id,
                                                         cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void SparseTensorDenseAddKernel(size_t input_elements, size_t rank, size_t *x2_shape,
                                                         int64_t *x1_indices_addr, int16_t *x1_values_addr,
                                                         int64_t *x1_shape_addr, int16_t *x2_values_addr,
                                                         int16_t *y_addr, const uint32_t &device_id,
                                                         cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void SparseTensorDenseAddKernel(size_t input_elements, size_t rank, size_t *x2_shape,
                                                         int32_t *x1_indices_addr, int16_t *x1_values_addr,
                                                         int32_t *x1_shape_addr, int16_t *x2_values_addr,
                                                         int16_t *y_addr, const uint32_t &device_id,
                                                         cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void SparseTensorDenseAddKernel(size_t input_elements, size_t rank, size_t *x2_shape,
                                                         int64_t *x1_indices_addr, int32_t *x1_values_addr,
                                                         int64_t *x1_shape_addr, int32_t *x2_values_addr,
                                                         int32_t *y_addr, const uint32_t &device_id,
                                                         cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void SparseTensorDenseAddKernel(size_t input_elements, size_t rank, size_t *x2_shape,
                                                         int32_t *x1_indices_addr, int32_t *x1_values_addr,
                                                         int32_t *x1_shape_addr, int32_t *x2_values_addr,
                                                         int32_t *y_addr, const uint32_t &device_id,
                                                         cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void SparseTensorDenseAddKernel(size_t input_elements, size_t rank, size_t *x2_shape,
                                                         int64_t *x1_indices_addr, int64_t *x1_values_addr,
                                                         int64_t *x1_shape_addr, int64_t *x2_values_addr,
                                                         int64_t *y_addr, const uint32_t &device_id,
                                                         cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void SparseTensorDenseAddKernel(size_t input_elements, size_t rank, size_t *x2_shape,
                                                         int32_t *x1_indices_addr, int64_t *x1_values_addr,
                                                         int32_t *x1_shape_addr, int64_t *x2_values_addr,
                                                         int64_t *y_addr, const uint32_t &device_id,
                                                         cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void SparseTensorDenseAddKernel(size_t input_elements, size_t rank, size_t *x2_shape,
                                                         int64_t *x1_indices_addr, double *x1_values_addr,
                                                         int64_t *x1_shape_addr, double *x2_values_addr,
                                                         double *y_addr, const uint32_t &device_id,
                                                         cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void SparseTensorDenseAddKernel(size_t input_elements, size_t rank, size_t *x2_shape,
                                                         int32_t *x1_indices_addr, double *x1_values_addr,
                                                         int32_t *x1_shape_addr, double *x2_values_addr,
                                                         double *y_addr, const uint32_t &device_id,
                                                         cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void SparseTensorDenseAddKernel(size_t input_elements, size_t rank, size_t *x2_shape,
                                                         int64_t *x1_indices_addr, Complex<float> *x1_values_addr,
                                                         int64_t *x1_shape_addr, Complex<float> *x2_values_addr,
                                                         Complex<float> *y_addr, const uint32_t &device_id,
                                                         cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void SparseTensorDenseAddKernel(size_t input_elements, size_t rank, size_t *x2_shape,
                                                         int32_t *x1_indices_addr, Complex<float> *x1_values_addr,
                                                         int32_t *x1_shape_addr, Complex<float> *x2_values_addr,
                                                         Complex<float> *y_addr, const uint32_t &device_id,
                                                         cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void SparseTensorDenseAddKernel(size_t input_elements, size_t rank, size_t *x2_shape,
                                                         int64_t *x1_indices_addr, Complex<double> *x1_values_addr,
                                                         int64_t *x1_shape_addr, Complex<double> *x2_values_addr,
                                                         Complex<double> *y_addr, const uint32_t &device_id,
                                                         cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void SparseTensorDenseAddKernel(size_t input_elements, size_t rank, size_t *x2_shape,
                                                         int32_t *x1_indices_addr, Complex<double> *x1_values_addr,
                                                         int32_t *x1_shape_addr, Complex<double> *x2_values_addr,
                                                         Complex<double> *y_addr, const uint32_t &device_id,
                                                         cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void SparseTensorDenseAddKernel(size_t input_elements, size_t rank, size_t *x2_shape,
                                                         int32_t *x1_indices_addr, float *x1_values_addr,
                                                         int32_t *x1_shape_addr, float *x2_values_addr,
                                                         float *y_addr, const uint32_t &device_id,
                                                         cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void SparseTensorDenseAddKernel(size_t input_elements, size_t rank, size_t *x2_shape,
                                                         int32_t *x1_indices_addr, half *x1_values_addr,
                                                         int32_t *x1_shape_addr, half *x2_values_addr,
                                                         half *y_addr, const uint32_t &device_id,
                                                         cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void SparseTensorDenseAddKernel(size_t input_elements, size_t rank, size_t *x2_shape,
                                                         int64_t *x1_indices_addr, float *x1_values_addr,
                                                         int64_t *x1_shape_addr, float *x2_values_addr,
                                                         float *y_addr, const uint32_t &device_id,
                                                         cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void SparseTensorDenseAddKernel(size_t input_elements, size_t rank, size_t *x2_shape,
                                                         int64_t *x1_indices_addr, half *x1_values_addr,
                                                         int64_t *x1_shape_addr, half *x2_values_addr,
                                                         half *y_addr, const uint32_t &device_id,
                                                         cudaStream_t cuda_stream);
