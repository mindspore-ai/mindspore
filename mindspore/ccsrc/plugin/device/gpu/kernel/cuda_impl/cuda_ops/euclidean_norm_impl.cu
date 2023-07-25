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

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/euclidean_norm_impl.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/util.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/complex.h"
#include "include/cuda_fp16.h"

template <typename T>
__device__ __forceinline__ T SqrtFunc(T input) {
  return static_cast<T>(sqrt(static_cast<float>(input)));
}

template <>
__device__ __forceinline__ half SqrtFunc(half input) {
  return hsqrt(input);
}

template <typename T>
__inline__ __device__ void EuclideanNormCoreOp(const T *input, size_t input_index, T *output, size_t output_index) {
  MsAtomicAdd(output + output_index, static_cast<T>(input[input_index] * input[input_index]));
}

template <>
__inline__ __device__ void EuclideanNormCoreOp(const Complex<float> *input, size_t input_index, Complex<float> *output,
                                               size_t output_index) {
  float abs_value_real =
    input[input_index].real() >= static_cast<float>(0) ? input[input_index].real() : -input[input_index].real();
  float abs_value_imag =
    input[input_index].imag() >= static_cast<float>(0) ? input[input_index].imag() : -input[input_index].imag();
  MsAtomicAdd(output + output_index,
              static_cast<Complex<float>>(abs_value_real * abs_value_real + abs_value_imag * abs_value_imag));
}

template <>
__inline__ __device__ void EuclideanNormCoreOp(const Complex<double> *input, size_t input_index,
                                               Complex<double> *output, size_t output_index) {
  float abs_value_real =
    input[input_index].real() >= static_cast<float>(0) ? input[input_index].real() : -input[input_index].real();
  float abs_value_imag =
    input[input_index].imag() >= static_cast<float>(0) ? input[input_index].imag() : -input[input_index].imag();
  MsAtomicAdd(output + output_index,
              static_cast<Complex<double>>(abs_value_real * abs_value_real + abs_value_imag * abs_value_imag));
}

template <typename T>
__inline__ __device__ void EuclideanNormHighPrecisionCoreOp(const T *input, size_t input_index, float *output,
                                                            size_t output_index) {
  MsAtomicAdd(output + output_index, static_cast<float>(input[input_index] * input[input_index]));
}

template <typename T>
__global__ void EuclideanCalKernel(const T *input, const size_t *input_shape, size_t input_shape_length,
                                   size_t input_elements, const size_t *output_axis, const size_t *output_stride,
                                   size_t output_shape_length, T *output) {
  for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < (input_elements);
       index += blockDim.x * gridDim.x) {
    size_t flatten_index = index;
    size_t output_index = 0;
    for (int i = static_cast<int>(input_shape_length - 1); i >= 0; --i) {
      for (int j = static_cast<int>(output_shape_length - 1); j >= 0; --j) {
        if (i == output_axis[j]) {
          size_t coordinate = flatten_index % input_shape[i];
          output_index += coordinate * output_stride[j];
          break;
        }
      }
      flatten_index = flatten_index / input_shape[i];
    }
    EuclideanNormCoreOp(input, index, output, output_index);
  }
}

template <typename T>
__global__ void EuclideanCalHighPrecisionKernel(const T *input, const size_t *input_shape, size_t input_shape_length,
                                                size_t input_elements, const size_t *output_axis,
                                                const size_t *output_stride, size_t output_shape_length,
                                                float *output) {
  for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < (input_elements);
       index += blockDim.x * gridDim.x) {
    size_t flatten_index = index;
    size_t output_index = 0;
    for (int i = static_cast<int>(input_shape_length - 1); i >= 0; --i) {
      for (int j = static_cast<int>(output_shape_length - 1); j >= 0; --j) {
        if (i == output_axis[j]) {
          size_t coordinate = flatten_index % input_shape[i];
          output_index += coordinate * output_stride[j];
          break;
        }
      }
      flatten_index = flatten_index / input_shape[i];
    }
    EuclideanNormHighPrecisionCoreOp(input, index, output, output_index);
  }
}

template <typename T>
__global__ void NormCalKernel(T *output, size_t output_elements) {
  for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < (output_elements);
       index += blockDim.x * gridDim.x) {
    output[index] = SqrtFunc(output[index]);
  }
}

template <>
__global__ void NormCalKernel(Complex<float> *output, size_t output_elements) {
  for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < (output_elements);
       index += blockDim.x * gridDim.x) {
    output[index] = static_cast<Complex<float>>(SqrtFunc(output[index].real()));
  }
}

template <>
__global__ void NormCalKernel(Complex<double> *output, size_t output_elements) {
  for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < (output_elements);
       index += blockDim.x * gridDim.x) {
    output[index] = static_cast<Complex<double>>(SqrtFunc(output[index].real()));
  }
}

template <typename T>
__global__ void NormCalHighPrecisionKernel(const float *middle_output, T *output, size_t output_elements) {
  for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < (output_elements);
       index += blockDim.x * gridDim.x) {
    output[index] = SqrtFunc(middle_output[index]);
  }
}

template <typename T>
CUDA_LIB_EXPORT cudaError_t CalEuclideanNorm(const T *input, const size_t *input_shape, size_t input_shape_length,
                                             size_t input_elements, const size_t *output_axis,
                                             const size_t *output_stride, size_t output_shape_length,
                                             size_t output_elements, float *middle_output, T *output,
                                             const uint32_t &device_id, cudaStream_t cuda_stream) {
  EuclideanCalKernel<<<CUDA_BLOCKS(device_id, input_elements), CUDA_THREADS(device_id), 0, cuda_stream>>>(
    input, input_shape, input_shape_length, input_elements, output_axis, output_stride, output_shape_length, output);
  NormCalKernel<<<CUDA_BLOCKS(device_id, output_elements), CUDA_THREADS(device_id), 0, cuda_stream>>>(output,
                                                                                                      output_elements);
  return GetCudaStatus();
}

template <>
CUDA_LIB_EXPORT cudaError_t CalEuclideanNorm(const int8_t *input, const size_t *input_shape, size_t input_shape_length,
                                             size_t input_elements, const size_t *output_axis,
                                             const size_t *output_stride, size_t output_shape_length,
                                             size_t output_elements, float *middle_output, int8_t *output,
                                             const uint32_t &device_id, cudaStream_t cuda_stream) {
  EuclideanCalHighPrecisionKernel<<<CUDA_BLOCKS(device_id, input_elements), CUDA_THREADS(device_id), 0, cuda_stream>>>(
    input, input_shape, input_shape_length, input_elements, output_axis, output_stride, output_shape_length,
    middle_output);
  NormCalHighPrecisionKernel<<<CUDA_BLOCKS(device_id, output_elements), CUDA_THREADS(device_id), 0, cuda_stream>>>(
    middle_output, output, output_elements);
  return GetCudaStatus();
}

template <>
CUDA_LIB_EXPORT cudaError_t CalEuclideanNorm(const int16_t *input, const size_t *input_shape, size_t input_shape_length,
                                             size_t input_elements, const size_t *output_axis,
                                             const size_t *output_stride, size_t output_shape_length,
                                             size_t output_elements, float *middle_output, int16_t *output,
                                             const uint32_t &device_id, cudaStream_t cuda_stream) {
  EuclideanCalHighPrecisionKernel<<<CUDA_BLOCKS(device_id, input_elements), CUDA_THREADS(device_id), 0, cuda_stream>>>(
    input, input_shape, input_shape_length, input_elements, output_axis, output_stride, output_shape_length,
    middle_output);
  NormCalHighPrecisionKernel<<<CUDA_BLOCKS(device_id, output_elements), CUDA_THREADS(device_id), 0, cuda_stream>>>(
    middle_output, output, output_elements);
  return GetCudaStatus();
}

template <>
CUDA_LIB_EXPORT cudaError_t CalEuclideanNorm(const uint8_t *input, const size_t *input_shape, size_t input_shape_length,
                                             size_t input_elements, const size_t *output_axis,
                                             const size_t *output_stride, size_t output_shape_length,
                                             size_t output_elements, float *middle_output, uint8_t *output,
                                             const uint32_t &device_id, cudaStream_t cuda_stream) {
  EuclideanCalHighPrecisionKernel<<<CUDA_BLOCKS(device_id, input_elements), CUDA_THREADS(device_id), 0, cuda_stream>>>(
    input, input_shape, input_shape_length, input_elements, output_axis, output_stride, output_shape_length,
    middle_output);
  NormCalHighPrecisionKernel<<<CUDA_BLOCKS(device_id, output_elements), CUDA_THREADS(device_id), 0, cuda_stream>>>(
    middle_output, output, output_elements);
  return GetCudaStatus();
}

template <>
CUDA_LIB_EXPORT cudaError_t CalEuclideanNorm(const uint16_t *input, const size_t *input_shape,
                                             size_t input_shape_length, size_t input_elements,
                                             const size_t *output_axis, const size_t *output_stride,
                                             size_t output_shape_length, size_t output_elements, float *middle_output,
                                             uint16_t *output, const uint32_t &device_id, cudaStream_t cuda_stream) {
  EuclideanCalHighPrecisionKernel<<<CUDA_BLOCKS(device_id, input_elements), CUDA_THREADS(device_id), 0, cuda_stream>>>(
    input, input_shape, input_shape_length, input_elements, output_axis, output_stride, output_shape_length,
    middle_output);
  NormCalHighPrecisionKernel<<<CUDA_BLOCKS(device_id, output_elements), CUDA_THREADS(device_id), 0, cuda_stream>>>(
    middle_output, output, output_elements);
  return GetCudaStatus();
}

template <>
CUDA_LIB_EXPORT cudaError_t CalEuclideanNorm(const half *input, const size_t *input_shape, size_t input_shape_length,
                                             size_t input_elements, const size_t *output_axis,
                                             const size_t *output_stride, size_t output_shape_length,
                                             size_t output_elements, float *middle_output, half *output,
                                             const uint32_t &device_id, cudaStream_t cuda_stream) {
  EuclideanCalHighPrecisionKernel<<<CUDA_BLOCKS(device_id, input_elements), CUDA_THREADS(device_id), 0, cuda_stream>>>(
    input, input_shape, input_shape_length, input_elements, output_axis, output_stride, output_shape_length,
    middle_output);
  NormCalHighPrecisionKernel<<<CUDA_BLOCKS(device_id, output_elements), CUDA_THREADS(device_id), 0, cuda_stream>>>(
    middle_output, output, output_elements);
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t CalEuclideanNorm<int8_t>(const int8_t *input, const size_t *input_shape,
                                                              size_t input_shape_length, size_t input_elements,
                                                              const size_t *output_axis, const size_t *output_stride,
                                                              size_t output_shape_length, size_t output_elements,
                                                              float *middle_output, int8_t *output,
                                                              const uint32_t &device_id, cudaStream_t cuda_stream_);
template CUDA_LIB_EXPORT cudaError_t CalEuclideanNorm<int16_t>(const int16_t *input, const size_t *input_shape,
                                                               size_t input_shape_length, size_t input_elements,
                                                               const size_t *output_axis, const size_t *output_stride,
                                                               size_t output_shape_length, size_t output_elements,
                                                               float *middle_output, int16_t *output,
                                                               const uint32_t &device_id, cudaStream_t cuda_stream_);
template CUDA_LIB_EXPORT cudaError_t CalEuclideanNorm<int>(const int *input, const size_t *input_shape,
                                                           size_t input_shape_length, size_t input_elements,
                                                           const size_t *output_axis, const size_t *output_stride,
                                                           size_t output_shape_length, size_t output_elements,
                                                           float *middle_output, int *output, const uint32_t &device_id,
                                                           cudaStream_t cuda_stream_);
template CUDA_LIB_EXPORT cudaError_t CalEuclideanNorm<int64_t>(const int64_t *input, const size_t *input_shape,
                                                               size_t input_shape_length, size_t input_elements,
                                                               const size_t *output_axis, const size_t *output_stride,
                                                               size_t output_shape_length, size_t output_elements,
                                                               float *middle_output, int64_t *output,
                                                               const uint32_t &device_id, cudaStream_t cuda_stream_);

template CUDA_LIB_EXPORT cudaError_t CalEuclideanNorm<uint8_t>(const uint8_t *input, const size_t *input_shape,
                                                               size_t input_shape_length, size_t input_elements,
                                                               const size_t *output_axis, const size_t *output_stride,
                                                               size_t output_shape_length, size_t output_elements,
                                                               float *middle_output, uint8_t *output,
                                                               const uint32_t &device_id, cudaStream_t cuda_stream_);
template CUDA_LIB_EXPORT cudaError_t CalEuclideanNorm<uint16_t>(const uint16_t *input, const size_t *input_shape,
                                                                size_t input_shape_length, size_t input_elements,
                                                                const size_t *output_axis, const size_t *output_stride,
                                                                size_t output_shape_length, size_t output_elements,
                                                                float *middle_output, uint16_t *output,
                                                                const uint32_t &device_id, cudaStream_t cuda_stream_);
template CUDA_LIB_EXPORT cudaError_t CalEuclideanNorm<uint32_t>(const uint32_t *input, const size_t *input_shape,
                                                                size_t input_shape_length, size_t input_elements,
                                                                const size_t *output_axis, const size_t *output_stride,
                                                                size_t output_shape_length, size_t output_elements,
                                                                float *middle_output, uint32_t *output,
                                                                const uint32_t &device_id, cudaStream_t cuda_stream_);
template CUDA_LIB_EXPORT cudaError_t CalEuclideanNorm<uint64_t>(const uint64_t *input, const size_t *input_shape,
                                                                size_t input_shape_length, size_t input_elements,
                                                                const size_t *output_axis, const size_t *output_stride,
                                                                size_t output_shape_length, size_t output_elements,
                                                                float *middle_output, uint64_t *output,
                                                                const uint32_t &device_id, cudaStream_t cuda_stream_);

template CUDA_LIB_EXPORT cudaError_t CalEuclideanNorm<half>(const half *input, const size_t *input_shape,
                                                            size_t input_shape_length, size_t input_elements,
                                                            const size_t *output_axis, const size_t *output_stride,
                                                            size_t output_shape_length, size_t output_elements,
                                                            float *middle_output, half *output,
                                                            const uint32_t &device_id, cudaStream_t cuda_stream_);

template CUDA_LIB_EXPORT cudaError_t CalEuclideanNorm<float>(const float *input, const size_t *input_shape,
                                                             size_t input_shape_length, size_t input_elements,
                                                             const size_t *output_axis, const size_t *output_stride,
                                                             size_t output_shape_length, size_t output_elements,
                                                             float *middle_output, float *output,
                                                             const uint32_t &device_id, cudaStream_t cuda_stream_);
template CUDA_LIB_EXPORT cudaError_t CalEuclideanNorm<double>(const double *input, const size_t *input_shape,
                                                              size_t input_shape_length, size_t input_elements,
                                                              const size_t *output_axis, const size_t *output_stride,
                                                              size_t output_shape_length, size_t output_elements,
                                                              float *middle_output, double *output,
                                                              const uint32_t &device_id, cudaStream_t cuda_stream_);

template CUDA_LIB_EXPORT cudaError_t CalEuclideanNorm<Complex<float>>(
  const Complex<float> *input, const size_t *input_shape, size_t input_shape_length, size_t input_elements,
  const size_t *output_axis, const size_t *output_stride, size_t output_shape_length, size_t output_elements,
  float *middle_output, Complex<float> *output, const uint32_t &device_id, cudaStream_t cuda_stream_);

template CUDA_LIB_EXPORT cudaError_t CalEuclideanNorm<Complex<double>>(
  const Complex<double> *input, const size_t *input_shape, size_t input_shape_length, size_t input_elements,
  const size_t *output_axis, const size_t *output_stride, size_t output_shape_length, size_t output_elements,
  float *middle_output, Complex<double> *output, const uint32_t &device_id, cudaStream_t cuda_stream_);
