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

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/tensor_scatter_arithmetic.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/util.cuh"

template <typename T, typename S>
__global__ void TensorScatterUpdateKernel(const T *input, const S *indices, const T *update, T *output,
                                          const size_t block_size, const size_t input_size, const size_t output_size,
                                          const size_t indices_dim_0, const size_t indices_dim_1, S *indices_stride,
                                          S *work_shape) {
  int i, j;
  for (size_t read_index = blockIdx.x * blockDim.x + threadIdx.x; read_index < input_size;
       read_index += blockDim.x * gridDim.x) {
    size_t write_index = 0;
    bool out_bound = false;

    i = static_cast<int>(read_index / block_size);
    j = static_cast<int>(read_index % block_size);

    for (size_t k = 0; k < indices_dim_1; k++) {
      S indices_i = indices[i * indices_dim_1 + k];
      out_bound |= indices_i >= work_shape[k];
      out_bound |= indices_i < 0;
      write_index += indices_i * indices_stride[k];
    }
    write_index += j;
    out_bound |= write_index >= output_size;
    if (!out_bound) {
      output[write_index] = update[read_index];
    }
  }
}

template <typename T, typename S>
__global__ void TensorScatterMinKernel(const T *input, const S *indices, const T *update, T *output,
                                       const size_t block_size, const size_t input_size, const size_t output_size,
                                       const size_t indices_dim_0, const size_t indices_dim_1, S *indices_stride,
                                       S *work_shape) {
  int i, j;
  for (size_t read_index = blockIdx.x * blockDim.x + threadIdx.x; read_index < input_size;
       read_index += blockDim.x * gridDim.x) {
    size_t write_index = 0;
    bool out_bound = false;

    i = static_cast<int>(read_index / block_size);
    j = static_cast<int>(read_index % block_size);

    for (size_t k = 0; k < indices_dim_1; k++) {
      S indices_i = indices[i * indices_dim_1 + k];
      out_bound |= indices_i >= work_shape[k];
      out_bound |= indices_i < 0;
      write_index += indices_i * indices_stride[k];
    }
    write_index += j;
    out_bound |= write_index >= output_size;
    if (!out_bound) {
      (void)MsAtomicMin(&output[write_index], update[read_index]);
    }
  }
}

template <typename T, typename S>
__global__ void TensorScatterMaxKernel(const T *input, const S *indices, const T *update, T *output,
                                       const size_t block_size, const size_t input_size, const size_t output_size,
                                       const size_t indices_dim_0, const size_t indices_dim_1, S *indices_stride,
                                       S *work_shape) {
  int i, j;
  for (size_t read_index = blockIdx.x * blockDim.x + threadIdx.x; read_index < input_size;
       read_index += blockDim.x * gridDim.x) {
    size_t write_index = 0;
    bool out_bound = false;

    i = static_cast<int>(read_index / block_size);
    j = static_cast<int>(read_index % block_size);

    for (size_t k = 0; k < indices_dim_1; k++) {
      S indices_i = indices[i * indices_dim_1 + k];
      out_bound |= indices_i >= work_shape[k];
      out_bound |= indices_i < 0;
      write_index += indices_i * indices_stride[k];
    }
    write_index += j;
    out_bound |= write_index >= output_size;
    if (!out_bound) {
      (void)MsAtomicMax(&output[write_index], update[read_index]);
    }
  }
}

template <typename T, typename S>
__global__ void TensorScatterAddKernel(const T *input, const S *indices, const T *update, T *output,
                                       const size_t block_size, const size_t input_size, const size_t output_size,
                                       const size_t indices_dim_0, const size_t indices_dim_1, S *indices_stride,
                                       S *work_shape) {
  int i, j;
  for (size_t read_index = blockIdx.x * blockDim.x + threadIdx.x; read_index < input_size;
       read_index += blockDim.x * gridDim.x) {
    size_t write_index = 0;
    bool out_bound = false;

    i = static_cast<int>(read_index / block_size);
    j = static_cast<int>(read_index % block_size);

    for (size_t k = 0; k < indices_dim_1; k++) {
      S indices_i = indices[i * indices_dim_1 + k];
      out_bound |= indices_i >= work_shape[k];
      out_bound |= indices_i < 0;
      write_index += indices_i * indices_stride[k];
    }
    write_index += j;
    out_bound |= write_index >= output_size;
    if (!out_bound) {
      (void)MsAtomicAdd(&output[write_index], update[read_index]);
    }
  }
}

template <typename T, typename S>
__global__ void TensorScatterSubKernel(const T *input, const S *indices, const T *update, T *output,
                                       const size_t block_size, const size_t input_size, const size_t output_size,
                                       const size_t indices_dim_0, const size_t indices_dim_1, S *indices_stride,
                                       S *work_shape) {
  int i, j;
  for (size_t read_index = blockIdx.x * blockDim.x + threadIdx.x; read_index < input_size;
       read_index += blockDim.x * gridDim.x) {
    size_t write_index = 0;
    bool out_bound = false;

    i = static_cast<int>(read_index / block_size);
    j = static_cast<int>(read_index % block_size);

    for (size_t k = 0; k < indices_dim_1; k++) {
      S indices_i = indices[i * indices_dim_1 + k];
      out_bound |= indices_i >= work_shape[k];
      out_bound |= indices_i < 0;
      write_index += indices_i * indices_stride[k];
    }
    write_index += j;
    out_bound |= write_index >= output_size;
    if (!out_bound) {
      (void)MsAtomicSub(&output[write_index], update[read_index]);
    }
  }
}

template <typename T, typename S>
__global__ void TensorScatterMulKernel(const T *input, const S *indices, const T *update, T *output,
                                       const size_t block_size, const size_t input_size, const size_t output_size,
                                       const size_t indices_dim_0, const size_t indices_dim_1, S *indices_stride,
                                       S *work_shape) {
  int i, j;
  for (size_t read_index = blockIdx.x * blockDim.x + threadIdx.x; read_index < input_size;
       read_index += blockDim.x * gridDim.x) {
    size_t write_index = 0;
    bool out_bound = false;

    i = static_cast<int>(read_index / block_size);
    j = static_cast<int>(read_index % block_size);

    for (size_t k = 0; k < indices_dim_1; k++) {
      S indices_i = indices[i * indices_dim_1 + k];
      out_bound |= indices_i >= work_shape[k];
      out_bound |= indices_i < 0;
      write_index += indices_i * indices_stride[k];
    }
    write_index += j;
    out_bound |= write_index >= output_size;
    if (!out_bound) {
      (void)MsAtomicMul(&output[write_index], update[read_index]);
    }
  }
}

template <typename T, typename S>
__global__ void TensorScatterDivKernel(const T *input, const S *indices, const T *update, T *output,
                                       const size_t block_size, const size_t input_size, const size_t output_size,
                                       const size_t indices_dim_0, const size_t indices_dim_1, S *indices_stride,
                                       S *work_shape) {
  int i, j;
  for (size_t read_index = blockIdx.x * blockDim.x + threadIdx.x; read_index < input_size;
       read_index += blockDim.x * gridDim.x) {
    size_t write_index = 0;
    bool out_bound = false;

    i = static_cast<int>(read_index / block_size);
    j = static_cast<int>(read_index % block_size);

    for (size_t k = 0; k < indices_dim_1; k++) {
      S indices_i = indices[i * indices_dim_1 + k];
      out_bound |= indices_i >= work_shape[k];
      out_bound |= indices_i < 0;
      write_index += indices_i * indices_stride[k];
    }
    write_index += j;
    out_bound |= write_index >= output_size;
    if (!out_bound) {
      (void)MsAtomicDiv(&output[write_index], update[read_index]);
    }
  }
}

template <typename T, typename S>
void TensorScatterArithmetic(const enum TensorScatterArithmeticFunctionType &func_type, const T *input,
                             const S *indices, const T *update, T *output, const size_t &block_size,
                             const size_t &input_size, const size_t &output_size, const size_t &indices_dim_0,
                             const size_t &indices_dim_1, S *indices_stride, S *work_shape, uint32_t device_id,
                             cudaStream_t stream) {
  switch (func_type) {
    case TENSOR_SCATTER_FUNC_UPDATE:
      return TensorScatterUpdateKernel<<<CUDA_BLOCKS(device_id, output_size), CUDA_THREADS(device_id), 0, stream>>>(
        input, indices, update, output, block_size, input_size, output_size, indices_dim_0, indices_dim_1,
        indices_stride, work_shape);
    case TENSOR_SCATTER_FUNC_MIN:
      return TensorScatterMinKernel<<<CUDA_BLOCKS(device_id, output_size), CUDA_THREADS(device_id), 0, stream>>>(
        input, indices, update, output, block_size, input_size, output_size, indices_dim_0, indices_dim_1,
        indices_stride, work_shape);
    case TENSOR_SCATTER_FUNC_MAX:
      return TensorScatterMaxKernel<<<CUDA_BLOCKS(device_id, output_size), CUDA_THREADS(device_id), 0, stream>>>(
        input, indices, update, output, block_size, input_size, output_size, indices_dim_0, indices_dim_1,
        indices_stride, work_shape);
    case TENSOR_SCATTER_FUNC_ADD:
      return TensorScatterAddKernel<<<CUDA_BLOCKS(device_id, output_size), CUDA_THREADS(device_id), 0, stream>>>(
        input, indices, update, output, block_size, input_size, output_size, indices_dim_0, indices_dim_1,
        indices_stride, work_shape);
    case TENSOR_SCATTER_FUNC_SUB:
      return TensorScatterSubKernel<<<CUDA_BLOCKS(device_id, output_size), CUDA_THREADS(device_id), 0, stream>>>(
        input, indices, update, output, block_size, input_size, output_size, indices_dim_0, indices_dim_1,
        indices_stride, work_shape);
    case TENSOR_SCATTER_FUNC_MUL:
      return TensorScatterMulKernel<<<CUDA_BLOCKS(device_id, output_size), CUDA_THREADS(device_id), 0, stream>>>(
        input, indices, update, output, block_size, input_size, output_size, indices_dim_0, indices_dim_1,
        indices_stride, work_shape);
    case TENSOR_SCATTER_FUNC_DIV:
      return TensorScatterDivKernel<<<CUDA_BLOCKS(device_id, output_size), CUDA_THREADS(device_id), 0, stream>>>(
        input, indices, update, output, block_size, input_size, output_size, indices_dim_0, indices_dim_1,
        indices_stride, work_shape);
    default:
      break;
  }
}

template <typename T, typename S>
void CallTensorScatterUpdate(const T *input, const S *indices, const T *update, T *output, const size_t &block_size,
                             const size_t &input_size, const size_t &output_size, const size_t &indices_dim_0,
                             const size_t &indices_dim_1, S *indices_stride, S *work_shape, uint32_t device_id,
                             cudaStream_t stream) {
  TensorScatterUpdateKernel<<<CUDA_BLOCKS(device_id, output_size), CUDA_THREADS(device_id), 0, stream>>>(
    input, indices, update, output, block_size, input_size, output_size, indices_dim_0, indices_dim_1, indices_stride,
    work_shape);
}

template CUDA_LIB_EXPORT void TensorScatterArithmetic<half, int>(
  const enum TensorScatterArithmeticFunctionType &func_type, const half *input, const int *indices, const half *update,
  half *output, const size_t &block_size, const size_t &input_size, const size_t &output_size,
  const size_t &indices_dim_0, const size_t &indices_dim_1, int *indices_stride, int *work_shape, uint32_t device_id,
  cudaStream_t stream);

template CUDA_LIB_EXPORT void TensorScatterArithmetic<float, int>(
  const enum TensorScatterArithmeticFunctionType &func_type, const float *input, const int *indices,
  const float *update, float *output, const size_t &block_size, const size_t &input_size, const size_t &output_size,
  const size_t &indices_dim_0, const size_t &indices_dim_1, int *indices_stride, int *work_shape, uint32_t device_id,
  cudaStream_t stream);

template CUDA_LIB_EXPORT void TensorScatterArithmetic<double, int>(
  const enum TensorScatterArithmeticFunctionType &func_type, const double *input, const int *indices,
  const double *update, double *output, const size_t &block_size, const size_t &input_size, const size_t &output_size,
  const size_t &indices_dim_0, const size_t &indices_dim_1, int *indices_stride, int *work_shape, uint32_t device_id,
  cudaStream_t stream);

template CUDA_LIB_EXPORT void TensorScatterArithmetic<char, int>(
  const enum TensorScatterArithmeticFunctionType &func_type, const char *input, const int *indices, const char *update,
  char *output, const size_t &block_size, const size_t &input_size, const size_t &output_size,
  const size_t &indices_dim_0, const size_t &indices_dim_1, int *indices_stride, int *work_shape, uint32_t device_id,
  cudaStream_t stream);

template CUDA_LIB_EXPORT void TensorScatterArithmetic<unsigned char, int>(
  const enum TensorScatterArithmeticFunctionType &func_type, const unsigned char *input, const int *indices,
  const unsigned char *update, unsigned char *output, const size_t &block_size, const size_t &input_size,
  const size_t &output_size, const size_t &indices_dim_0, const size_t &indices_dim_1, int *indices_stride,
  int *work_shape, uint32_t device_id, cudaStream_t stream);

template CUDA_LIB_EXPORT void TensorScatterArithmetic<int16_t, int>(
  const enum TensorScatterArithmeticFunctionType &func_type, const int16_t *input, const int *indices,
  const int16_t *update, int16_t *output, const size_t &block_size, const size_t &input_size, const size_t &output_size,
  const size_t &indices_dim_0, const size_t &indices_dim_1, int *indices_stride, int *work_shape, uint32_t device_id,
  cudaStream_t stream);

template CUDA_LIB_EXPORT void TensorScatterArithmetic<uint16_t, int>(
  const enum TensorScatterArithmeticFunctionType &func_type, const uint16_t *input, const int *indices,
  const uint16_t *update, uint16_t *output, const size_t &block_size, const size_t &input_size,
  const size_t &output_size, const size_t &indices_dim_0, const size_t &indices_dim_1, int *indices_stride,
  int *work_shape, uint32_t device_id, cudaStream_t stream);

template CUDA_LIB_EXPORT void TensorScatterArithmetic<int, int>(
  const enum TensorScatterArithmeticFunctionType &func_type, const int *input, const int *indices, const int *update,
  int *output, const size_t &block_size, const size_t &input_size, const size_t &output_size,
  const size_t &indices_dim_0, const size_t &indices_dim_1, int *indices_stride, int *work_shape, uint32_t device_id,
  cudaStream_t stream);

template CUDA_LIB_EXPORT void TensorScatterArithmetic<uint32_t, int>(
  const enum TensorScatterArithmeticFunctionType &func_type, const uint32_t *input, const int *indices,
  const uint32_t *update, uint32_t *output, const size_t &block_size, const size_t &input_size,
  const size_t &output_size, const size_t &indices_dim_0, const size_t &indices_dim_1, int *indices_stride,
  int *work_shape, uint32_t device_id, cudaStream_t stream);

template CUDA_LIB_EXPORT void TensorScatterArithmetic<int64_t, int>(
  const enum TensorScatterArithmeticFunctionType &func_type, const int64_t *input, const int *indices,
  const int64_t *update, int64_t *output, const size_t &block_size, const size_t &input_size, const size_t &output_size,
  const size_t &indices_dim_0, const size_t &indices_dim_1, int *indices_stride, int *work_shape, uint32_t device_id,
  cudaStream_t stream);

template CUDA_LIB_EXPORT void TensorScatterArithmetic<uint64_t, int>(
  const enum TensorScatterArithmeticFunctionType &func_type, const uint64_t *input, const int *indices,
  const uint64_t *update, uint64_t *output, const size_t &block_size, const size_t &input_size,
  const size_t &output_size, const size_t &indices_dim_0, const size_t &indices_dim_1, int *indices_stride,
  int *work_shape, uint32_t device_id, cudaStream_t stream);

template CUDA_LIB_EXPORT void TensorScatterArithmetic<bool, int>(
  const enum TensorScatterArithmeticFunctionType &func_type, const bool *input, const int *indices, const bool *update,
  bool *output, const size_t &block_size, const size_t &input_size, const size_t &output_size,
  const size_t &indices_dim_0, const size_t &indices_dim_1, int *indices_stride, int *work_shape, uint32_t device_id,
  cudaStream_t stream);

template CUDA_LIB_EXPORT void TensorScatterArithmetic<half, int64_t>(
  const enum TensorScatterArithmeticFunctionType &func_type, const half *input, const int64_t *indices,
  const half *update, half *output, const size_t &block_size, const size_t &input_size, const size_t &output_size,
  const size_t &indices_dim_0, const size_t &indices_dim_1, int64_t *indices_stride, int64_t *work_shape,
  uint32_t device_id, cudaStream_t stream);

template CUDA_LIB_EXPORT void TensorScatterArithmetic<float, int64_t>(
  const enum TensorScatterArithmeticFunctionType &func_type, const float *input, const int64_t *indices,
  const float *update, float *output, const size_t &block_size, const size_t &input_size, const size_t &output_size,
  const size_t &indices_dim_0, const size_t &indices_dim_1, int64_t *indices_stride, int64_t *work_shape,
  uint32_t device_id, cudaStream_t stream);

template CUDA_LIB_EXPORT void TensorScatterArithmetic<double, int64_t>(
  const enum TensorScatterArithmeticFunctionType &func_type, const double *input, const int64_t *indices,
  const double *update, double *output, const size_t &block_size, const size_t &input_size, const size_t &output_size,
  const size_t &indices_dim_0, const size_t &indices_dim_1, int64_t *indices_stride, int64_t *work_shape,
  uint32_t device_id, cudaStream_t stream);

template CUDA_LIB_EXPORT void TensorScatterArithmetic<char, int64_t>(
  const enum TensorScatterArithmeticFunctionType &func_type, const char *input, const int64_t *indices,
  const char *update, char *output, const size_t &block_size, const size_t &input_size, const size_t &output_size,
  const size_t &indices_dim_0, const size_t &indices_dim_1, int64_t *indices_stride, int64_t *work_shape,
  uint32_t device_id, cudaStream_t stream);

template CUDA_LIB_EXPORT void TensorScatterArithmetic<unsigned char, int64_t>(
  const enum TensorScatterArithmeticFunctionType &func_type, const unsigned char *input, const int64_t *indices,
  const unsigned char *update, unsigned char *output, const size_t &block_size, const size_t &input_size,
  const size_t &output_size, const size_t &indices_dim_0, const size_t &indices_dim_1, int64_t *indices_stride,
  int64_t *work_shape, uint32_t device_id, cudaStream_t stream);

template CUDA_LIB_EXPORT void TensorScatterArithmetic<int16_t, int64_t>(
  const enum TensorScatterArithmeticFunctionType &func_type, const int16_t *input, const int64_t *indices,
  const int16_t *update, int16_t *output, const size_t &block_size, const size_t &input_size, const size_t &output_size,
  const size_t &indices_dim_0, const size_t &indices_dim_1, int64_t *indices_stride, int64_t *work_shape,
  uint32_t device_id, cudaStream_t stream);

template CUDA_LIB_EXPORT void TensorScatterArithmetic<uint16_t, int64_t>(
  const enum TensorScatterArithmeticFunctionType &func_type, const uint16_t *input, const int64_t *indices,
  const uint16_t *update, uint16_t *output, const size_t &block_size, const size_t &input_size,
  const size_t &output_size, const size_t &indices_dim_0, const size_t &indices_dim_1, int64_t *indices_stride,
  int64_t *work_shape, uint32_t device_id, cudaStream_t stream);

template CUDA_LIB_EXPORT void TensorScatterArithmetic<int, int64_t>(
  const enum TensorScatterArithmeticFunctionType &func_type, const int *input, const int64_t *indices,
  const int *update, int *output, const size_t &block_size, const size_t &input_size, const size_t &output_size,
  const size_t &indices_dim_0, const size_t &indices_dim_1, int64_t *indices_stride, int64_t *work_shape,
  uint32_t device_id, cudaStream_t stream);

template CUDA_LIB_EXPORT void TensorScatterArithmetic<uint32_t, int64_t>(
  const enum TensorScatterArithmeticFunctionType &func_type, const uint32_t *input, const int64_t *indices,
  const uint32_t *update, uint32_t *output, const size_t &block_size, const size_t &input_size,
  const size_t &output_size, const size_t &indices_dim_0, const size_t &indices_dim_1, int64_t *indices_stride,
  int64_t *work_shape, uint32_t device_id, cudaStream_t stream);

template CUDA_LIB_EXPORT void TensorScatterArithmetic<int64_t, int64_t>(
  const enum TensorScatterArithmeticFunctionType &func_type, const int64_t *input, const int64_t *indices,
  const int64_t *update, int64_t *output, const size_t &block_size, const size_t &input_size, const size_t &output_size,
  const size_t &indices_dim_0, const size_t &indices_dim_1, int64_t *indices_stride, int64_t *work_shape,
  uint32_t device_id, cudaStream_t stream);

template CUDA_LIB_EXPORT void TensorScatterArithmetic<uint64_t, int64_t>(
  const enum TensorScatterArithmeticFunctionType &func_type, const uint64_t *input, const int64_t *indices,
  const uint64_t *update, uint64_t *output, const size_t &block_size, const size_t &input_size,
  const size_t &output_size, const size_t &indices_dim_0, const size_t &indices_dim_1, int64_t *indices_stride,
  int64_t *work_shape, uint32_t device_id, cudaStream_t stream);

template CUDA_LIB_EXPORT void TensorScatterArithmetic<bool, int64_t>(
  const enum TensorScatterArithmeticFunctionType &func_type, const bool *input, const int64_t *indices,
  const bool *update, bool *output, const size_t &block_size, const size_t &input_size, const size_t &output_size,
  const size_t &indices_dim_0, const size_t &indices_dim_1, int64_t *indices_stride, int64_t *work_shape,
  uint32_t device_id, cudaStream_t stream);

template CUDA_LIB_EXPORT void CallTensorScatterUpdate<Complex<float>, int64_t>(
  const Complex<float> *input, const int64_t *indices, const Complex<float> *update, Complex<float> *output,
  const size_t &block_size, const size_t &input_size, const size_t &output_size, const size_t &indices_dim_0,
  const size_t &indices_dim_1, int64_t *indices_stride, int64_t *work_shape, uint32_t device_id, cudaStream_t stream);

template CUDA_LIB_EXPORT void CallTensorScatterUpdate<Complex<float>, int>(
  const Complex<float> *input, const int *indices, const Complex<float> *update, Complex<float> *output,
  const size_t &block_size, const size_t &input_size, const size_t &output_size, const size_t &indices_dim_0,
  const size_t &indices_dim_1, int *indices_stride, int *work_shape, uint32_t device_id, cudaStream_t stream);

template CUDA_LIB_EXPORT void CallTensorScatterUpdate<Complex<double>, int64_t>(
  const Complex<double> *input, const int64_t *indices, const Complex<double> *update, Complex<double> *output,
  const size_t &block_size, const size_t &input_size, const size_t &output_size, const size_t &indices_dim_0,
  const size_t &indices_dim_1, int64_t *indices_stride, int64_t *work_shape, uint32_t device_id, cudaStream_t stream);

template CUDA_LIB_EXPORT void CallTensorScatterUpdate<Complex<double>, int>(
  const Complex<double> *input, const int *indices, const Complex<double> *update, Complex<double> *output,
  const size_t &block_size, const size_t &input_size, const size_t &output_size, const size_t &indices_dim_0,
  const size_t &indices_dim_1, int *indices_stride, int *work_shape, uint32_t device_id, cudaStream_t stream);
