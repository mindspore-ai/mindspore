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

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/complex.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/sparse_dense_cwise_operation_impl.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/util.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/broadcast_to_impl.cuh"

template <typename T>
using Complex = mindspore::utils::Complex<T>;

__device__ __forceinline__ size_t Index(const size_t &index, const size_t &dim) { return dim == 1 ? 0 : index; }

__device__ int BroadcastToKernel(size_t dim_size, UnaryBroadcastStrideInfo strides, int index_out) {
  int64_t cur_out_idx = 0;
  size_t cur_pos = index_out;
  size_t inp_pos = 0;
  for (int idx = 0; idx < dim_size; ++idx) {
    cur_out_idx = cur_pos / strides.output_stride[idx];
    inp_pos += cur_out_idx * strides.input_stride[idx];
    cur_pos -= cur_out_idx * strides.output_stride[idx];
  }
  return static_cast<int>(inp_pos);
}

// specSpecializations of complex types
__global__ void SparseDenseCwiseAddNoBcastGpuKernel(const int64_t *x1_indices, const Complex<float> *x1_values,
                                                    const int64_t *x1_shape, const Complex<float> *x2,
                                                    Complex<float> *y, const int64_t dimension,
                                                    const int64_t value_nums) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < value_nums; i += blockDim.x * gridDim.x) {
    int index = 0;
    for (int64_t j = 0; j < dimension - 1; j++) {
      int c = 1;
      for (int64_t k = j + 1; k < dimension; k++) {
        c = c * x1_shape[k];
      }
      index += c * x1_indices[j + i * dimension];
    }
    index += x1_indices[(i + 1) * dimension - 1];
    y[i] = Complex<float>(x1_values[i].real() + x2[index].real(), x1_values[i].imag() + x2[index].imag());
  }
}

__global__ void SparseDenseCwiseMulNoBcastGpuKernel(const int64_t *x1_indices, const Complex<float> *x1_values,
                                                    const int64_t *x1_shape, const Complex<float> *x2,
                                                    Complex<float> *y, const int64_t dimension,
                                                    const int64_t value_nums) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < value_nums; i += blockDim.x * gridDim.x) {
    int index = 0;
    for (int64_t j = 0; j < dimension - 1; j++) {
      int c = 1;
      for (int64_t k = j + 1; k < dimension; k++) {
        c = c * x1_shape[k];
      }
      index += c * x1_indices[j + i * dimension];
    }
    index += x1_indices[(i + 1) * dimension - 1];
    y[i] = Complex<float>(x1_values[i].real() + x2[index].real(), x1_values[i].imag() + x2[index].imag());
  }
}

__global__ void SparseDenseCwiseDivNoBcastGpuKernel(const int64_t *x1_indices, const Complex<float> *x1_values,
                                                    const int64_t *x1_shape, const Complex<float> *x2,
                                                    Complex<float> *y, const int64_t dimension,
                                                    const int64_t value_nums) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < value_nums; i += blockDim.x * gridDim.x) {
    int index = 0;
    for (int64_t j = 0; j < dimension - 1; j++) {
      int c = 1;
      for (int64_t k = j + 1; k < dimension; k++) {
        c = c * x1_shape[k];
      }
      index += c * x1_indices[j + i * dimension];
    }
    index += x1_indices[(i + 1) * dimension - 1];
    y[i] = Complex<float>(x1_values[i].real() + x2[index].real(), x1_values[i].imag() + x2[index].imag());
  }
}

__global__ void SparseDenseCwiseAddNoBcastGpuKernel(const int64_t *x1_indices, const Complex<double> *x1_values,
                                                    const int64_t *x1_shape, const Complex<double> *x2,
                                                    Complex<double> *y, const int64_t dimension,
                                                    const int64_t value_nums) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < value_nums; i += blockDim.x * gridDim.x) {
    int index = 0;
    for (int64_t j = 0; j < dimension - 1; j++) {
      int c = 1;
      for (int64_t k = j + 1; k < dimension; k++) {
        c = c * x1_shape[k];
      }
      index += c * x1_indices[j + i * dimension];
    }
    index += x1_indices[(i + 1) * dimension - 1];
    y[i] = Complex<double>(x1_values[i].real() + x2[index].real(), x1_values[i].imag() + x2[index].imag());
  }
}

__global__ void SparseDenseCwiseMulNoBcastGpuKernel(const int64_t *x1_indices, const Complex<double> *x1_values,
                                                    const int64_t *x1_shape, const Complex<double> *x2,
                                                    Complex<double> *y, const int64_t dimension,
                                                    const int64_t value_nums) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < value_nums; i += blockDim.x * gridDim.x) {
    int index = 0;
    for (int64_t j = 0; j < dimension - 1; j++) {
      int c = 1;
      for (int64_t k = j + 1; k < dimension; k++) {
        c = c * x1_shape[k];
      }
      index += c * x1_indices[j + i * dimension];
    }
    index += x1_indices[(i + 1) * dimension - 1];
    y[i] = Complex<double>(x1_values[i].real() + x2[index].real(), x1_values[i].imag() + x2[index].imag());
  }
}

__global__ void SparseDenseCwiseDivNoBcastGpuKernel(const int64_t *x1_indices, const Complex<double> *x1_values,
                                                    const int64_t *x1_shape, const Complex<double> *x2,
                                                    Complex<double> *y, const int64_t dimension,
                                                    const int64_t value_nums) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < value_nums; i += blockDim.x * gridDim.x) {
    int index = 0;
    for (int64_t j = 0; j < dimension - 1; j++) {
      int c = 1;
      for (int64_t k = j + 1; k < dimension; k++) {
        c = c * x1_shape[k];
      }
      index += c * x1_indices[j + i * dimension];
    }
    index += x1_indices[(i + 1) * dimension - 1];
    y[i] = Complex<double>(x1_values[i].real() + x2[index].real(), x1_values[i].imag() + x2[index].imag());
  }
}

__global__ void SparseDenseCwiseAddBcastGpuKernel(const int64_t *x1_indices, const Complex<float> *x1_values,
                                                  const int64_t *x1_shape, const Complex<float> *x2, Complex<float> *y,
                                                  const int64_t dimension, const int64_t value_nums, size_t dim_size,
                                                  UnaryBroadcastStrideInfo strides) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < value_nums; i += blockDim.x * gridDim.x) {
    int index = 0;
    for (int64_t j = 0; j < dimension - 1; j++) {
      int c = 1;
      for (int64_t k = j + 1; k < dimension; k++) {
        c = c * x1_shape[k];
      }
      index += c * x1_indices[j + i * dimension];
    }
    index += x1_indices[(i + 1) * dimension - 1];
    index = BroadcastToKernel(dim_size, strides, index);
    y[i] = Complex<float>(x1_values[i].real() + x2[index].real(), x1_values[i].imag() + x2[index].imag());
  }
}

__global__ void SparseDenseCwiseMulBcastGpuKernel(const int64_t *x1_indices, const Complex<float> *x1_values,
                                                  const int64_t *x1_shape, const Complex<float> *x2, Complex<float> *y,
                                                  const int64_t dimension, const int64_t value_nums, size_t dim_size,
                                                  UnaryBroadcastStrideInfo strides) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < value_nums; i += blockDim.x * gridDim.x) {
    int index = 0;
    for (int64_t j = 0; j < dimension - 1; j++) {
      int c = 1;
      for (int64_t k = j + 1; k < dimension; k++) {
        c = c * x1_shape[k];
      }
      index += c * x1_indices[j + i * dimension];
    }
    index += x1_indices[(i + 1) * dimension - 1];
    index = BroadcastToKernel(dim_size, strides, index);
    y[i] = Complex<float>(x1_values[i].real() + x2[index].real(), x1_values[i].imag() + x2[index].imag());
  }
}

__global__ void SparseDenseCwiseDivBcastGpuKernel(const int64_t *x1_indices, const Complex<float> *x1_values,
                                                  const int64_t *x1_shape, const Complex<float> *x2, Complex<float> *y,
                                                  const int64_t dimension, const int64_t value_nums, size_t dim_size,
                                                  UnaryBroadcastStrideInfo strides) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < value_nums; i += blockDim.x * gridDim.x) {
    int index = 0;
    for (int64_t j = 0; j < dimension - 1; j++) {
      int c = 1;
      for (int64_t k = j + 1; k < dimension; k++) {
        c = c * x1_shape[k];
      }
      index += c * x1_indices[j + i * dimension];
    }
    index += x1_indices[(i + 1) * dimension - 1];
    index = BroadcastToKernel(dim_size, strides, index);
    y[i] = Complex<float>(x1_values[i].real() + x2[index].real(), x1_values[i].imag() + x2[index].imag());
  }
}

__global__ void SparseDenseCwiseAddBcastGpuKernel(const int64_t *x1_indices, const Complex<double> *x1_values,
                                                  const int64_t *x1_shape, const Complex<double> *x2,
                                                  Complex<double> *y, const int64_t dimension, const int64_t value_nums,
                                                  size_t dim_size, UnaryBroadcastStrideInfo strides) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < value_nums; i += blockDim.x * gridDim.x) {
    int index = 0;
    for (int64_t j = 0; j < dimension - 1; j++) {
      int c = 1;
      for (int64_t k = j + 1; k < dimension; k++) {
        c = c * x1_shape[k];
      }
      index += c * x1_indices[j + i * dimension];
    }
    index += x1_indices[(i + 1) * dimension - 1];
    index = BroadcastToKernel(dim_size, strides, index);
    y[i] = Complex<double>(x1_values[i].real() + x2[index].real(), x1_values[i].imag() + x2[index].imag());
  }
}

__global__ void SparseDenseCwiseMulBcastGpuKernel(const int64_t *x1_indices, const Complex<double> *x1_values,
                                                  const int64_t *x1_shape, const Complex<double> *x2,
                                                  Complex<double> *y, const int64_t dimension, const int64_t value_nums,
                                                  size_t dim_size, UnaryBroadcastStrideInfo strides) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < value_nums; i += blockDim.x * gridDim.x) {
    int index = 0;
    for (int64_t j = 0; j < dimension - 1; j++) {
      int c = 1;
      for (int64_t k = j + 1; k < dimension; k++) {
        c = c * x1_shape[k];
      }
      index += c * x1_indices[j + i * dimension];
    }
    index += x1_indices[(i + 1) * dimension - 1];
    index = BroadcastToKernel(dim_size, strides, index);
    y[i] = Complex<double>(x1_values[i].real() + x2[index].real(), x1_values[i].imag() + x2[index].imag());
  }
}

__global__ void SparseDenseCwiseDivBcastGpuKernel(const int64_t *x1_indices, const Complex<double> *x1_values,
                                                  const int64_t *x1_shape, const Complex<double> *x2,
                                                  Complex<double> *y, const int64_t dimension, const int64_t value_nums,
                                                  size_t dim_size, UnaryBroadcastStrideInfo strides) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < value_nums; i += blockDim.x * gridDim.x) {
    int index = 0;
    for (int64_t j = 0; j < dimension - 1; j++) {
      int c = 1;
      for (int64_t k = j + 1; k < dimension; k++) {
        c = c * x1_shape[k];
      }
      index += c * x1_indices[j + i * dimension];
    }
    index += x1_indices[(i + 1) * dimension - 1];
    index = BroadcastToKernel(dim_size, strides, index);
    y[i] = Complex<double>(x1_values[i].real() + x2[index].real(), x1_values[i].imag() + x2[index].imag());
  }
}

// normal

template <typename T>
__global__ void SparseDenseCwiseAddNoBcastGpuKernel(const int64_t *x1_indices, const T *x1_values,
                                                    const int64_t *x1_shape, const T *x2, T *y, const int64_t dimension,
                                                    const int64_t value_nums) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < value_nums; i += blockDim.x * gridDim.x) {
    int index = 0;
    for (int64_t j = 0; j < dimension - 1; j++) {
      int c = 1;
      for (int64_t k = j + 1; k < dimension; k++) {
        c = c * x1_shape[k];
      }
      index += c * x1_indices[j + i * dimension];
    }
    index += x1_indices[(i + 1) * dimension - 1];
    y[i] = static_cast<T>(x1_values[i] + x2[index]);
  }
}

template <typename T>
__global__ void SparseDenseCwiseMulNoBcastGpuKernel(const int64_t *x1_indices, const T *x1_values,
                                                    const int64_t *x1_shape, const T *x2, T *y, const int64_t dimension,
                                                    const int64_t value_nums) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < value_nums; i += blockDim.x * gridDim.x) {
    int index = 0;
    for (int64_t j = 0; j < dimension - 1; j++) {
      int c = 1;
      for (int64_t k = j + 1; k < dimension; k++) {
        c = c * x1_shape[k];
      }
      index += c * x1_indices[j + i * dimension];
    }
    index += x1_indices[(i + 1) * dimension - 1];
    y[i] = static_cast<T>(x1_values[i] * x2[index]);
  }
}

template <typename T>
__global__ void SparseDenseCwiseDivNoBcastGpuKernel(const int64_t *x1_indices, const T *x1_values,
                                                    const int64_t *x1_shape, const T *x2, T *y, const int64_t dimension,
                                                    const int64_t value_nums) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < value_nums; i += blockDim.x * gridDim.x) {
    int index = 0;
    for (int64_t j = 0; j < dimension - 1; j++) {
      int c = 1;
      for (int64_t k = j + 1; k < dimension; k++) {
        c = c * x1_shape[k];
      }
      index += c * x1_indices[j + i * dimension];
    }
    index += x1_indices[(i + 1) * dimension - 1];
    y[i] = static_cast<T>(x1_values[i] / x2[index]);
  }
}

template <typename T>
__global__ void SparseDenseCwiseAddBcastGpuKernel(const int64_t *x1_indices, const T *x1_values,
                                                  const int64_t *x1_shape, const T *x2, T *y, const int64_t dimension,
                                                  const int64_t value_nums, size_t dim_size,
                                                  UnaryBroadcastStrideInfo strides) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < value_nums; i += blockDim.x * gridDim.x) {
    int index = 0;
    for (int64_t j = 0; j < dimension - 1; j++) {
      int c = 1;
      for (int64_t k = j + 1; k < dimension; k++) {
        c = c * x1_shape[k];
      }
      index += c * x1_indices[j + i * dimension];
    }
    index += x1_indices[(i + 1) * dimension - 1];
    index = BroadcastToKernel(dim_size, strides, index);
    y[i] = static_cast<T>(x1_values[i] + x2[index]);
  }
}

template <typename T>
__global__ void SparseDenseCwiseMulBcastGpuKernel(const int64_t *x1_indices, const T *x1_values,
                                                  const int64_t *x1_shape, const T *x2, T *y, const int64_t dimension,
                                                  const int64_t value_nums, size_t dim_size,
                                                  UnaryBroadcastStrideInfo strides) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < value_nums; i += blockDim.x * gridDim.x) {
    int index = 0;
    for (int64_t j = 0; j < dimension - 1; j++) {
      int c = 1;
      for (int64_t k = j + 1; k < dimension; k++) {
        c = c * x1_shape[k];
      }
      index += c * x1_indices[j + i * dimension];
    }
    index += x1_indices[(i + 1) * dimension - 1];
    index = BroadcastToKernel(dim_size, strides, index);
    y[i] = static_cast<T>(x1_values[i] * x2[index]);
  }
}

template <typename T>
__global__ void SparseDenseCwiseDivBcastGpuKernel(const int64_t *x1_indices, const T *x1_values,
                                                  const int64_t *x1_shape, const T *x2, T *y, const int64_t dimension,
                                                  const int64_t value_nums, size_t dim_size,
                                                  UnaryBroadcastStrideInfo strides) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < value_nums; i += blockDim.x * gridDim.x) {
    int index = 0;
    for (int64_t j = 0; j < dimension - 1; j++) {
      int c = 1;
      for (int64_t k = j + 1; k < dimension; k++) {
        c = c * x1_shape[k];
      }
      index += c * x1_indices[j + i * dimension];
    }
    index += x1_indices[(i + 1) * dimension - 1];
    index = BroadcastToKernel(dim_size, strides, index);
    y[i] = static_cast<T>(x1_values[i] / x2[index]);
  }
}

template <typename T>
__global__ void SparseDenseCwiseAddDenseDim1GpuKernel(const int64_t *x1_indices, const T *x1_values,
                                                      const int64_t *x1_shape, const T *x2, T *y,
                                                      const int64_t dimension, const int64_t value_nums) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < value_nums; i += blockDim.x * gridDim.x) {
    y[i] = static_cast<T>(x1_values[i] + *(x2));
  }
}

template <typename T>
__global__ void SparseDenseCwiseMulDenseDim1GpuKernel(const int64_t *x1_indices, const T *x1_values,
                                                      const int64_t *x1_shape, const T *x2, T *y,
                                                      const int64_t dimension, const int64_t value_nums) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < value_nums; i += blockDim.x * gridDim.x) {
    y[i] = static_cast<T>(x1_values[i] * *(x2));
  }
}

template <typename T>
__global__ void SparseDenseCwiseDivDenseDim1GpuKernel(const int64_t *x1_indices, const T *x1_values,
                                                      const int64_t *x1_shape, const T *x2, T *y,
                                                      const int64_t dimension, const int64_t value_nums) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < value_nums; i += blockDim.x * gridDim.x) {
    y[i] = static_cast<T>(x1_values[i] / *(x2));
  }
}

template <typename T>
cudaError_t CalSparseDenseCwiseOperationNoBcastCompute(const enum SparseDenseCwiseOperationFunctionType &func_type,
                                                       const int64_t *x1_indices, const T *x1_values,
                                                       const int64_t *x1_shape, const T *x2, T *y,
                                                       const int64_t dimension, const int64_t value_nums,
                                                       const int64_t dense_dim, const uint32_t &device_id,
                                                       cudaStream_t cuda_stream) {
  if (dimension == dense_dim) {
    switch (func_type) {
      case SPARSE_DENSE_CWISE_OPERATION_FUNC_ADD:
        SparseDenseCwiseAddNoBcastGpuKernel<<<CUDA_BLOCKS(device_id, value_nums), CUDA_THREADS(device_id), 0,
                                              cuda_stream>>>(x1_indices, x1_values, x1_shape, x2, y, dimension,
                                                             value_nums);
        break;
      case SPARSE_DENSE_CWISE_OPERATION_FUNC_MUL:
        SparseDenseCwiseMulNoBcastGpuKernel<<<CUDA_BLOCKS(device_id, value_nums), CUDA_THREADS(device_id), 0,
                                              cuda_stream>>>(x1_indices, x1_values, x1_shape, x2, y, dimension,
                                                             value_nums);
        break;
      case SPARSE_DENSE_CWISE_OPERATION_FUNC_DIV:
        SparseDenseCwiseDivNoBcastGpuKernel<<<CUDA_BLOCKS(device_id, value_nums), CUDA_THREADS(device_id), 0,
                                              cuda_stream>>>(x1_indices, x1_values, x1_shape, x2, y, dimension,
                                                             value_nums);
        break;
      default:
        break;
    }
  } else {
    switch (func_type) {
      case SPARSE_DENSE_CWISE_OPERATION_FUNC_ADD:
        SparseDenseCwiseAddDenseDim1GpuKernel<<<CUDA_BLOCKS(device_id, value_nums), CUDA_THREADS(device_id), 0,
                                                cuda_stream>>>(x1_indices, x1_values, x1_shape, x2, y, dimension,
                                                               value_nums);
        break;
      case SPARSE_DENSE_CWISE_OPERATION_FUNC_MUL:
        SparseDenseCwiseMulDenseDim1GpuKernel<<<CUDA_BLOCKS(device_id, value_nums), CUDA_THREADS(device_id), 0,
                                                cuda_stream>>>(x1_indices, x1_values, x1_shape, x2, y, dimension,
                                                               value_nums);
        break;
      case SPARSE_DENSE_CWISE_OPERATION_FUNC_DIV:
        SparseDenseCwiseDivDenseDim1GpuKernel<<<CUDA_BLOCKS(device_id, value_nums), CUDA_THREADS(device_id), 0,
                                                cuda_stream>>>(x1_indices, x1_values, x1_shape, x2, y, dimension,
                                                               value_nums);
        break;
      default:
        break;
    }
  }
  return GetCudaStatus();
}

template <typename T>
cudaError_t CalSparseDenseCwiseOperationBcastCompute(const enum SparseDenseCwiseOperationFunctionType &func_type,
                                              const int64_t *x1_indices, const T *x1_values, const int64_t *x1_shape,
                                              const T *x2, T *y, const std::vector<int64_t> i,
                                              const std::vector<int64_t> o, const int64_t dimension,
                                              const int64_t value_nums, const uint32_t &device_id,
                                              cudaStream_t cuda_stream) {
  const size_t dim_size = o.size();
  UnaryBroadcastStrideInfo strides = UnaryBroadcastCalStride(dim_size, i, o);
  switch (func_type) {
    case SPARSE_DENSE_CWISE_OPERATION_FUNC_ADD:
      SparseDenseCwiseAddBcastGpuKernel<<<CUDA_BLOCKS(device_id, value_nums), CUDA_THREADS(device_id), 0,
                                          cuda_stream>>>(x1_indices, x1_values, x1_shape, x2, y, dimension, value_nums,
                                                         dim_size, strides);
      break;
    case SPARSE_DENSE_CWISE_OPERATION_FUNC_MUL:
      SparseDenseCwiseMulBcastGpuKernel<<<CUDA_BLOCKS(device_id, value_nums), CUDA_THREADS(device_id), 0,
                                          cuda_stream>>>(x1_indices, x1_values, x1_shape, x2, y, dimension, value_nums,
                                                         dim_size, strides);
      break;
    case SPARSE_DENSE_CWISE_OPERATION_FUNC_DIV:
      SparseDenseCwiseDivBcastGpuKernel<<<CUDA_BLOCKS(device_id, value_nums), CUDA_THREADS(device_id), 0,
                                          cuda_stream>>>(x1_indices, x1_values, x1_shape, x2, y, dimension, value_nums,
                                                         dim_size, strides);
      break;
    default:
      break;
  }
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t CalSparseDenseCwiseOperationNoBcastCompute<int8_t>(
  const enum SparseDenseCwiseOperationFunctionType &func_type, const int64_t *x1_indices, const int8_t *x1_values,
  const int64_t *x1_shape, const int8_t *x2, int8_t *y, const int64_t dimension, const int64_t value_nums,
  const int64_t dense_dim, const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalSparseDenseCwiseOperationNoBcastCompute<int16_t>(
  const enum SparseDenseCwiseOperationFunctionType &func_type, const int64_t *x1_indices, const int16_t *x1_values,
  const int64_t *x1_shape, const int16_t *x2, int16_t *y, const int64_t dimension, const int64_t value_nums,
  const int64_t dense_dim, const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalSparseDenseCwiseOperationNoBcastCompute<int32_t>(
  const enum SparseDenseCwiseOperationFunctionType &func_type, const int64_t *x1_indices, const int32_t *x1_values,
  const int64_t *x1_shape, const int32_t *x2, int32_t *y, const int64_t dimension, const int64_t value_nums,
  const int64_t dense_dim, const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalSparseDenseCwiseOperationNoBcastCompute<int64_t>(
  const enum SparseDenseCwiseOperationFunctionType &func_type, const int64_t *x1_indices, const int64_t *x1_values,
  const int64_t *x1_shape, const int64_t *x2, int64_t *y, const int64_t dimension, const int64_t value_nums,
  const int64_t dense_dim, const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalSparseDenseCwiseOperationNoBcastCompute<uint8_t>(
  const enum SparseDenseCwiseOperationFunctionType &func_type, const int64_t *x1_indices, const uint8_t *x1_values,
  const int64_t *x1_shape, const uint8_t *x2, uint8_t *y, const int64_t dimension, const int64_t value_nums,
  const int64_t dense_dim, const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalSparseDenseCwiseOperationNoBcastCompute<uint16_t>(
  const enum SparseDenseCwiseOperationFunctionType &func_type, const int64_t *x1_indices, const uint16_t *x1_values,
  const int64_t *x1_shape, const uint16_t *x2, uint16_t *y, const int64_t dimension, const int64_t value_nums,
  const int64_t dense_dim, const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalSparseDenseCwiseOperationNoBcastCompute<uint32_t>(
  const enum SparseDenseCwiseOperationFunctionType &func_type, const int64_t *x1_indices, const uint32_t *x1_values,
  const int64_t *x1_shape, const uint32_t *x2, uint32_t *y, const int64_t dimension, const int64_t value_nums,
  const int64_t dense_dim, const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalSparseDenseCwiseOperationNoBcastCompute<uint64_t>(
  const enum SparseDenseCwiseOperationFunctionType &func_type, const int64_t *x1_indices, const uint64_t *x1_values,
  const int64_t *x1_shape, const uint64_t *x2, uint64_t *y, const int64_t dimension, const int64_t value_nums,
  const int64_t dense_dim, const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalSparseDenseCwiseOperationNoBcastCompute<half>(
  const enum SparseDenseCwiseOperationFunctionType &func_type, const int64_t *x1_indices, const half *x1_values,
  const int64_t *x1_shape, const half *x2, half *y, const int64_t dimension, const int64_t value_nums,
  const int64_t dense_dim, const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalSparseDenseCwiseOperationNoBcastCompute<float>(
  const enum SparseDenseCwiseOperationFunctionType &func_type, const int64_t *x1_indices, const float *x1_values,
  const int64_t *x1_shape, const float *x2, float *y, const int64_t dimension, const int64_t value_nums,
  const int64_t dense_dim, const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalSparseDenseCwiseOperationNoBcastCompute<double>(
  const enum SparseDenseCwiseOperationFunctionType &func_type, const int64_t *x1_indices, const double *x1_values,
  const int64_t *x1_shape, const double *x2, double *y, const int64_t dimension, const int64_t value_nums,
  const int64_t dense_dim, const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalSparseDenseCwiseOperationNoBcastCompute<Complex<float>>(
  const enum SparseDenseCwiseOperationFunctionType &func_type, const int64_t *x1_indices,
  const Complex<float> *x1_values, const int64_t *x1_shape, const Complex<float> *x2, Complex<float> *y,
  const int64_t dimension, const int64_t value_nums, const int64_t dense_dim, const uint32_t &device_id,
  cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalSparseDenseCwiseOperationNoBcastCompute<Complex<double>>(
  const enum SparseDenseCwiseOperationFunctionType &func_type, const int64_t *x1_indices,
  const Complex<double> *x1_values, const int64_t *x1_shape, const Complex<double> *x2, Complex<double> *y,
  const int64_t dimension, const int64_t value_nums, const int64_t dense_dim, const uint32_t &device_id,
  cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalSparseDenseCwiseOperationBcastCompute<int8_t>(
  const enum SparseDenseCwiseOperationFunctionType &func_type, const int64_t *x1_indices, const int8_t *x1_values,
  const int64_t *x1_shape, const int8_t *x2, int8_t *y, const std::vector<int64_t> i, const std::vector<int64_t> o,
  const int64_t dimension, const int64_t value_nums, const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalSparseDenseCwiseOperationBcastCompute<int16_t>(
  const enum SparseDenseCwiseOperationFunctionType &func_type, const int64_t *x1_indices, const int16_t *x1_values,
  const int64_t *x1_shape, const int16_t *x2, int16_t *y, const std::vector<int64_t> i, const std::vector<int64_t> o,
  const int64_t dimension, const int64_t value_nums, const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalSparseDenseCwiseOperationBcastCompute<int32_t>(
  const enum SparseDenseCwiseOperationFunctionType &func_type, const int64_t *x1_indices, const int32_t *x1_values,
  const int64_t *x1_shape, const int32_t *x2, int32_t *y, const std::vector<int64_t> i, const std::vector<int64_t> o,
  const int64_t dimension, const int64_t value_nums, const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalSparseDenseCwiseOperationBcastCompute<int64_t>(
  const enum SparseDenseCwiseOperationFunctionType &func_type, const int64_t *x1_indices, const int64_t *x1_values,
  const int64_t *x1_shape, const int64_t *x2, int64_t *y, const std::vector<int64_t> i, const std::vector<int64_t> o,
  const int64_t dimension, const int64_t value_nums, const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalSparseDenseCwiseOperationBcastCompute<uint8_t>(
  const enum SparseDenseCwiseOperationFunctionType &func_type, const int64_t *x1_indices, const uint8_t *x1_values,
  const int64_t *x1_shape, const uint8_t *x2, uint8_t *y, const std::vector<int64_t> i, const std::vector<int64_t> o,
  const int64_t dimension, const int64_t value_nums, const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalSparseDenseCwiseOperationBcastCompute<uint16_t>(
  const enum SparseDenseCwiseOperationFunctionType &func_type, const int64_t *x1_indices, const uint16_t *x1_values,
  const int64_t *x1_shape, const uint16_t *x2, uint16_t *y, const std::vector<int64_t> i, const std::vector<int64_t> o,
  const int64_t dimension, const int64_t value_nums, const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalSparseDenseCwiseOperationBcastCompute<uint32_t>(
  const enum SparseDenseCwiseOperationFunctionType &func_type, const int64_t *x1_indices, const uint32_t *x1_values,
  const int64_t *x1_shape, const uint32_t *x2, uint32_t *y, const std::vector<int64_t> i, const std::vector<int64_t> o,
  const int64_t dimension, const int64_t value_nums, const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalSparseDenseCwiseOperationBcastCompute<uint64_t>(
  const enum SparseDenseCwiseOperationFunctionType &func_type, const int64_t *x1_indices, const uint64_t *x1_values,
  const int64_t *x1_shape, const uint64_t *x2, uint64_t *y, const std::vector<int64_t> i, const std::vector<int64_t> o,
  const int64_t dimension, const int64_t value_nums, const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalSparseDenseCwiseOperationBcastCompute<half>(
  const enum SparseDenseCwiseOperationFunctionType &func_type, const int64_t *x1_indices, const half *x1_values,
  const int64_t *x1_shape, const half *x2, half *y, const std::vector<int64_t> i, const std::vector<int64_t> o,
  const int64_t dimension, const int64_t value_nums, const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalSparseDenseCwiseOperationBcastCompute<float>(
  const enum SparseDenseCwiseOperationFunctionType &func_type, const int64_t *x1_indices, const float *x1_values,
  const int64_t *x1_shape, const float *x2, float *y, const std::vector<int64_t> i, const std::vector<int64_t> o,
  const int64_t dimension, const int64_t value_nums, const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalSparseDenseCwiseOperationBcastCompute<double>(
  const enum SparseDenseCwiseOperationFunctionType &func_type, const int64_t *x1_indices, const double *x1_values,
  const int64_t *x1_shape, const double *x2, double *y, const std::vector<int64_t> i, const std::vector<int64_t> o,
  const int64_t dimension, const int64_t value_nums, const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalSparseDenseCwiseOperationBcastCompute<Complex<float>>(
  const enum SparseDenseCwiseOperationFunctionType &func_type, const int64_t *x1_indices,
  const Complex<float> *x1_values, const int64_t *x1_shape, const Complex<float> *x2, Complex<float> *y,
  const std::vector<int64_t> i, const std::vector<int64_t> o, const int64_t dimension, const int64_t value_nums,
  const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalSparseDenseCwiseOperationBcastCompute<Complex<double>>(
  const enum SparseDenseCwiseOperationFunctionType &func_type, const int64_t *x1_indices,
  const Complex<double> *x1_values, const int64_t *x1_shape, const Complex<double> *x2, Complex<double> *y,
  const std::vector<int64_t> i, const std::vector<int64_t> o, const int64_t dimension, const int64_t value_nums,
  const uint32_t &device_id, cudaStream_t cuda_stream);
