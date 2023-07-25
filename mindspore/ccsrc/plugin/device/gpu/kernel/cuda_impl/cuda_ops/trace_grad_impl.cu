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
#include <math.h>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/trace_grad_impl.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/util.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/complex.h"

template <typename T, typename S>
__global__ void TraceGrad(S size, const T *y_grad, const S *input_shape, T *output) {
  S matrix_col = input_shape[1];
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += gridDim.x * blockDim.x) {
    size_t row_num = pos / matrix_col;
    size_t col_num = pos % matrix_col;
    if (row_num < size && row_num == col_num) {
      output[pos] = *y_grad;
    } else {
      output[pos] = 0;
    }
  }
  return;
}

template <typename T, typename S>
cudaError_t CalTraceGrad(S size, const T *y_grad, const S *input_shape, T *output, const uint32_t &device_id,
                         cudaStream_t cuda_stream) {
  TraceGrad<T, S>
    <<<CUDA_BLOCKS(device_id, size), CUDA_THREADS(device_id), 0, cuda_stream>>>(size, y_grad, input_shape, output);
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t CalTraceGrad<uint8_t, int32_t>(int32_t size, const uint8_t *y_grad,
                                                                    const int32_t *input_shape, uint8_t *output,
                                                                    const uint32_t &device_id,
                                                                    cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalTraceGrad<uint16_t, int32_t>(int32_t size, const uint16_t *y_grad,
                                                                     const int32_t *input_shape, uint16_t *output,
                                                                     const uint32_t &device_id,
                                                                     cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalTraceGrad<uint32_t, int32_t>(int32_t size, const uint32_t *y_grad,
                                                                     const int32_t *input_shape, uint32_t *output,
                                                                     const uint32_t &device_id,
                                                                     cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalTraceGrad<uint64_t, int32_t>(int32_t size, const uint64_t *y_grad,
                                                                     const int32_t *input_shape, uint64_t *output,
                                                                     const uint32_t &device_id,
                                                                     cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalTraceGrad<int8_t, int32_t>(int32_t size, const int8_t *y_grad,
                                                                   const int32_t *input_shape, int8_t *output,
                                                                   const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalTraceGrad<int16_t, int32_t>(int32_t size, const int16_t *y_grad,
                                                                    const int32_t *input_shape, int16_t *output,
                                                                    const uint32_t &device_id,
                                                                    cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalTraceGrad<int32_t, int32_t>(int32_t size, const int32_t *y_grad,
                                                                    const int32_t *input_shape, int32_t *output,
                                                                    const uint32_t &device_id,
                                                                    cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalTraceGrad<int64_t, int32_t>(int32_t size, const int64_t *y_grad,
                                                                    const int32_t *input_shape, int64_t *output,
                                                                    const uint32_t &device_id,
                                                                    cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalTraceGrad<half, int32_t>(int32_t size, const half *y_grad,
                                                                 const int32_t *input_shape, half *output,
                                                                 const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalTraceGrad<float, int32_t>(int32_t size, const float *y_grad,
                                                                  const int32_t *input_shape, float *output,
                                                                  const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalTraceGrad<double, int32_t>(int32_t size, const double *y_grad,
                                                                   const int32_t *input_shape, double *output,
                                                                   const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalTraceGrad<Complex<float>, int32_t>(int32_t size, const Complex<float> *y_grad,
                                                                           const int32_t *input_shape,
                                                                           Complex<float> *output,
                                                                           const uint32_t &device_id,
                                                                           cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalTraceGrad<Complex<double>, int32_t>(int32_t size, const Complex<double> *y_grad,
                                                                            const int32_t *input_shape,
                                                                            Complex<double> *output,
                                                                            const uint32_t &device_id,
                                                                            cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalTraceGrad<uint8_t, int64_t>(int64_t size, const uint8_t *y_grad,
                                                                    const int64_t *input_shape, uint8_t *output,
                                                                    const uint32_t &device_id,
                                                                    cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalTraceGrad<uint16_t, int64_t>(int64_t size, const uint16_t *y_grad,
                                                                     const int64_t *input_shape, uint16_t *output,
                                                                     const uint32_t &device_id,
                                                                     cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalTraceGrad<uint32_t, int64_t>(int64_t size, const uint32_t *y_grad,
                                                                     const int64_t *input_shape, uint32_t *output,
                                                                     const uint32_t &device_id,
                                                                     cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalTraceGrad<uint64_t, int64_t>(int64_t size, const uint64_t *y_grad,
                                                                     const int64_t *input_shape, uint64_t *output,
                                                                     const uint32_t &device_id,
                                                                     cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalTraceGrad<int8_t, int64_t>(int64_t size, const int8_t *y_grad,
                                                                   const int64_t *input_shape, int8_t *output,
                                                                   const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalTraceGrad<int16_t, int64_t>(int64_t size, const int16_t *y_grad,
                                                                    const int64_t *input_shape, int16_t *output,
                                                                    const uint32_t &device_id,
                                                                    cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalTraceGrad<int32_t, int64_t>(int64_t size, const int32_t *y_grad,
                                                                    const int64_t *input_shape, int32_t *output,
                                                                    const uint32_t &device_id,
                                                                    cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalTraceGrad<int64_t, int64_t>(int64_t size, const int64_t *y_grad,
                                                                    const int64_t *input_shape, int64_t *output,
                                                                    const uint32_t &device_id,
                                                                    cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalTraceGrad<half, int64_t>(int64_t size, const half *y_grad,
                                                                 const int64_t *input_shape, half *output,
                                                                 const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalTraceGrad<float, int64_t>(int64_t size, const float *y_grad,
                                                                  const int64_t *input_shape, float *output,
                                                                  const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalTraceGrad<double, int64_t>(int64_t size, const double *y_grad,
                                                                   const int64_t *input_shape, double *output,
                                                                   const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalTraceGrad<Complex<float>, int64_t>(int64_t size, const Complex<float> *y_grad,
                                                                           const int64_t *input_shape,
                                                                           Complex<float> *output,
                                                                           const uint32_t &device_id,
                                                                           cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalTraceGrad<Complex<double>, int64_t>(int64_t size, const Complex<double> *y_grad,
                                                                            const int64_t *input_shape,
                                                                            Complex<double> *output,
                                                                            const uint32_t &device_id,
                                                                            cudaStream_t cuda_stream);
