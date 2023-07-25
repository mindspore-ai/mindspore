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

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/fft_with_size_impl.cuh"
#include "include/cuda_runtime.h"

// cublas exec scale
#ifndef CUBLAS_EXEC_SCALE
#define CUBLAS_EXEC_SCALE(cublas_exec, real, cu_complex)                \
  do {                                                                  \
    if (scale_factor != 1.0) {                                          \
      auto alpha = static_cast<real>(scale_factor);                     \
      auto out = static_cast<cu_complex *>(y_ptr);                      \
      CUBLAS_CALL(cublas_exec(scale_plan, y_elements, &alpha, out, 1)); \
    }                                                                   \
  } while (0)
#endif  // CUBLAS_EXEC_SCALE

cudaError_t CalculateFFT(cufftComplex *x_ptr, cufftComplex *y_ptr, const double &scale_factor, const int &y_elements,
                         cufftHandle cufft_plan, cublasHandle_t scale_plan, const uint32_t &device_id,
                         cudaStream_t cuda_stream) {
  CUFFT_CALL(cufftSetStream(cufft_plan, cuda_stream));
  CUBLAS_CALL(cublasSetStream_v2(scale_plan, cuda_stream));
  CUFFT_CALL(cufftExecC2C(cufft_plan, x_ptr, y_ptr, CUFFT_FORWARD));
  CUBLAS_EXEC_SCALE(cublasCsscal_v2, float, cuComplex);
  return GetCudaStatus();
}

cudaError_t CalculateFFT(cufftDoubleComplex *x_ptr, cufftDoubleComplex *y_ptr, const double &scale_factor,
                         const int &y_elements, cufftHandle cufft_plan, cublasHandle_t scale_plan,
                         const uint32_t &device_id, cudaStream_t cuda_stream) {
  CUFFT_CALL(cufftSetStream(cufft_plan, cuda_stream));
  CUBLAS_CALL(cublasSetStream_v2(scale_plan, cuda_stream));
  CUFFT_CALL(cufftExecZ2Z(cufft_plan, x_ptr, y_ptr, CUFFT_FORWARD));
  CUBLAS_EXEC_SCALE(cublasZdscal_v2, double, cuDoubleComplex);
  return GetCudaStatus();
}

cudaError_t CalculateIFFT(cufftComplex *x_ptr, cufftComplex *y_ptr, const double &scale_factor, const int &y_elements,
                          cufftHandle cufft_plan, cublasHandle_t scale_plan, const uint32_t &device_id,
                          cudaStream_t cuda_stream) {
  CUFFT_CALL(cufftSetStream(cufft_plan, cuda_stream));
  CUBLAS_CALL(cublasSetStream_v2(scale_plan, cuda_stream));
  CUFFT_CALL(cufftExecC2C(cufft_plan, x_ptr, y_ptr, CUFFT_INVERSE));
  CUBLAS_EXEC_SCALE(cublasCsscal_v2, float, cuComplex);
  return GetCudaStatus();
}

cudaError_t CalculateIFFT(cufftDoubleComplex *x_ptr, cufftDoubleComplex *y_ptr, const double &scale_factor,
                          const int &y_elements, cufftHandle cufft_plan, cublasHandle_t scale_plan,
                          const uint32_t &device_id, cudaStream_t cuda_stream) {
  CUFFT_CALL(cufftSetStream(cufft_plan, cuda_stream));
  CUBLAS_CALL(cublasSetStream_v2(scale_plan, cuda_stream));
  CUFFT_CALL(cufftExecZ2Z(cufft_plan, x_ptr, y_ptr, CUFFT_INVERSE));
  CUBLAS_EXEC_SCALE(cublasZdscal_v2, double, cuDoubleComplex);
  return GetCudaStatus();
}

__global__ void Float2FloatComplex(const float *input_addr, cufftComplex *output_addr, const int len) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < len; pos += blockDim.x * gridDim.x) {
    output_addr[pos] = make_cuFloatComplex(input_addr[pos], 0.);
  }
}

__global__ void Double2DoubleComplex(const double *input_addr, cufftDoubleComplex *output_addr, const int len) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < len; pos += blockDim.x * gridDim.x) {
    output_addr[pos] = make_cuDoubleComplex(input_addr[pos], 0.);
  }
}

__global__ void FloatComplex2Float(const cufftComplex *input_addr, float *output_addr, const int len) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < len; pos += blockDim.x * gridDim.x) {
    output_addr[pos] = cuCrealf(input_addr[pos]);
  }
}

__global__ void DoubleComplex2Double(const cufftDoubleComplex *input_addr, double *output_addr, const int len) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < len; pos += blockDim.x * gridDim.x) {
    output_addr[pos] = cuCreal(input_addr[pos]);
  }
}

cudaError_t CalculateRFFT(float *x_ptr, cufftComplex *w_ptr, cufftComplex *y_ptr, const bool &is_onesided,
                          const double &scale_factor, const int &x_elements, const int &y_elements,
                          cufftHandle cufft_plan, cublasHandle_t scale_plan, const uint32_t &device_id,
                          cudaStream_t cuda_stream) {
  CUFFT_CALL(cufftSetStream(cufft_plan, cuda_stream));
  CUBLAS_CALL(cublasSetStream_v2(scale_plan, cuda_stream));
  if (is_onesided) {  // onesided use native cufft r2c
    CUFFT_CALL(cufftExecR2C(cufft_plan, x_ptr, y_ptr));
  } else {  // full freq use [casting + c2c], cast real input buffer to complex workspace buffer
    Float2FloatComplex<<<CUDA_BLOCKS(device_id, x_elements), CUDA_THREADS(device_id), 0, cuda_stream>>>(x_ptr, w_ptr,
                                                                                                        x_elements);
    CUFFT_CALL(cufftExecC2C(cufft_plan, w_ptr, y_ptr, CUFFT_FORWARD));
  }
  CUBLAS_EXEC_SCALE(cublasCsscal_v2, float, cuComplex);
  return GetCudaStatus();
}

cudaError_t CalculateRFFT(double *x_ptr, cufftDoubleComplex *w_ptr, cufftDoubleComplex *y_ptr, const bool &is_onesided,
                          const double &scale_factor, const int &x_elements, const int &y_elements,
                          cufftHandle cufft_plan, cublasHandle_t scale_plan, const uint32_t &device_id,
                          cudaStream_t cuda_stream) {
  CUFFT_CALL(cufftSetStream(cufft_plan, cuda_stream));
  CUBLAS_CALL(cublasSetStream_v2(scale_plan, cuda_stream));
  if (is_onesided) {  // onesided use native cufft r2c
    CUFFT_CALL(cufftExecD2Z(cufft_plan, x_ptr, y_ptr));
  } else {  // full freq use [casting + c2c], cast real input buffer to complex workspace buffer
    Double2DoubleComplex<<<CUDA_BLOCKS(device_id, x_elements), CUDA_THREADS(device_id), 0, cuda_stream>>>(x_ptr, w_ptr,
                                                                                                          x_elements);
    CUFFT_CALL(cufftExecZ2Z(cufft_plan, w_ptr, y_ptr, CUFFT_FORWARD));
  }
  CUBLAS_EXEC_SCALE(cublasZdscal_v2, double, cuDoubleComplex);
  return GetCudaStatus();
}

cudaError_t CalculateIRFFT(cufftComplex *x_ptr, cufftComplex *w_ptr, float *y_ptr, const bool &is_onesided,
                    const double &scale_factor, const int &x_elements, const int &y_elements, cufftHandle cufft_plan,
                    cublasHandle_t scale_plan, const uint32_t &device_id, cudaStream_t cuda_stream) {
  CUFFT_CALL(cufftSetStream(cufft_plan, cuda_stream));
  CUBLAS_CALL(cublasSetStream_v2(scale_plan, cuda_stream));
  if (is_onesided) {  // onesided use native cufft c2r
    // complex-to-real need to copy input buffer to tmp buffer to avoid cufft overwriting.
    CUDA_RT_CALL(cudaMemcpyAsync(w_ptr, x_ptr, x_elements * sizeof(cufftComplex), cudaMemcpyDeviceToDevice));
    CUFFT_CALL(cufftExecC2R(cufft_plan, w_ptr, y_ptr));
  } else {  // full freq use [c2c + casting]
    // dump c2c result to workspace buffer, then cast complex workspace buffer to real output buffer.
    CUFFT_CALL(cufftExecC2C(cufft_plan, x_ptr, w_ptr, CUFFT_INVERSE));
    FloatComplex2Float<<<CUDA_BLOCKS(device_id, y_elements), CUDA_THREADS(device_id), 0, cuda_stream>>>(w_ptr, y_ptr,
                                                                                                        y_elements);
  }
  CUBLAS_EXEC_SCALE(cublasSscal_v2, float, float);
  return GetCudaStatus();
}

cudaError_t CalculateIRFFT(cufftDoubleComplex *x_ptr, cufftDoubleComplex *w_ptr, double *y_ptr, const bool &is_onesided,
                           const double &scale_factor, const int &x_elements, const int &y_elements,
                           cufftHandle cufft_plan, cublasHandle_t scale_plan, const uint32_t &device_id,
                           cudaStream_t cuda_stream) {
  CUFFT_CALL(cufftSetStream(cufft_plan, cuda_stream));
  CUBLAS_CALL(cublasSetStream_v2(scale_plan, cuda_stream));
  if (is_onesided) {  // onesided use native cufft r2c
    // complex-to-real need to copy input buffer to tmp buffer to avoid cufft overwriting.
    CUDA_RT_CALL(cudaMemcpyAsync(w_ptr, x_ptr, x_elements * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToDevice));
    CUFFT_CALL(cufftExecZ2D(cufft_plan, w_ptr, y_ptr));
  } else {  // full freq use [c2c + casting]
    // dump c2c result to workspace buffer, then cast complex workspace buffer to real output buffer.
    CUFFT_CALL(cufftExecZ2Z(cufft_plan, x_ptr, w_ptr, CUFFT_INVERSE));
    DoubleComplex2Double<<<CUDA_BLOCKS(device_id, y_elements), CUDA_THREADS(device_id), 0, cuda_stream>>>(w_ptr, y_ptr,
                                                                                                          y_elements);
  }
  CUBLAS_EXEC_SCALE(cublasDscal_v2, double, double);
  return GetCudaStatus();
}
