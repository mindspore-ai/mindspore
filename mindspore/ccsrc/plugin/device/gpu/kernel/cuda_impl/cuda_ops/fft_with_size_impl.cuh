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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_FFT_WITH_SIZE_IMPL_CUH_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_FFT_WITH_SIZE_IMPL_CUH_
#include <cufft.h>
#include <cublas_v2.h>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cuda_device_info.h"

// CUDA API error checking
#ifndef CUDA_RT_CALL
#define CUDA_RT_CALL(call)                                                                                          \
  do {                                                                                                              \
    auto status = static_cast<cudaError_t>(call);                                                                   \
    if (status != cudaSuccess) {                                                                                    \
      fprintf(stderr, "ERROR: CUDA RT call \"%s\" in line %d of file %s failed with code (%d).\n", #call, __LINE__, \
              __FILE__, status);                                                                                    \
      return cudaErrorNotReady;                                                                                     \
    }                                                                                                               \
  } while (0)
#endif  // CUDA_RT_CALL

// cublas API error checking
#ifndef CUBLAS_CALL
#define CUBLAS_CALL(call)                                                                                          \
  do {                                                                                                             \
    auto status = static_cast<cublasStatus_t>(call);                                                               \
    if (status != CUBLAS_STATUS_SUCCESS) {                                                                         \
      fprintf(stderr, "ERROR: CUBLAS call \"%s\" in line %d of file %s failed with code (%d).\n", #call, __LINE__, \
              __FILE__, status);                                                                                   \
      return cudaErrorNotReady;                                                                                    \
    }                                                                                                              \
  } while (0)
#endif  // CUBLAS_CALL

// cufft API error checking
#ifndef CUFFT_CALL
#define CUFFT_CALL(call)                                                                                          \
  do {                                                                                                            \
    auto status = static_cast<cufftResult>(call);                                                                 \
    if (status != CUFFT_SUCCESS) {                                                                                \
      fprintf(stderr, "ERROR: CUFFT call \"%s\" in line %d of file %s failed with code (%d).\n", #call, __LINE__, \
              __FILE__, status);                                                                                  \
      return cudaErrorNotReady;                                                                                   \
    }                                                                                                             \
  } while (0)
#endif  // CUFFT_CALL

CUDA_LIB_EXPORT cudaError_t CalculateFFT(cufftComplex *x_ptr, cufftComplex *y_ptr, const double &scale_factor,
                                         const int &y_elements, cufftHandle cufft_plan, cublasHandle_t scale_plan,
                                         const uint32_t &device_id, cudaStream_t cuda_stream);

CUDA_LIB_EXPORT cudaError_t CalculateFFT(cufftDoubleComplex *x_ptr, cufftDoubleComplex *y_ptr,
                                         const double &scale_factor, const int &y_elements, cufftHandle cufft_plan,
                                         cublasHandle_t scale_plan, const uint32_t &device_id,
                                         cudaStream_t cuda_stream);

CUDA_LIB_EXPORT cudaError_t CalculateIFFT(cufftComplex *x_ptr, cufftComplex *y_ptr, const double &scale_factor,
                                          const int &y_elements, cufftHandle cufft_plan, cublasHandle_t scale_plan,
                                          const uint32_t &device_id, cudaStream_t cuda_stream);

CUDA_LIB_EXPORT cudaError_t CalculateIFFT(cufftDoubleComplex *x_ptr, cufftDoubleComplex *y_ptr,
                                          const double &scale_factor, const int &y_elements, cufftHandle cufft_plan,
                                          cublasHandle_t scale_plan, const uint32_t &device_id,
                                          cudaStream_t cuda_stream);

CUDA_LIB_EXPORT cudaError_t CalculateRFFT(float *x_ptr, cufftComplex *w_ptr, cufftComplex *y_ptr,
                                          const bool &is_onesided, const double &scale_factor, const int &x_elements,
                                          const int &y_elements, cufftHandle cufft_plan, cublasHandle_t scale_plan,
                                          const uint32_t &device_id, cudaStream_t cuda_stream);

CUDA_LIB_EXPORT cudaError_t CalculateRFFT(double *x_ptr, cufftDoubleComplex *w_ptr, cufftDoubleComplex *y_ptr,
                                          const bool &is_onesided, const double &scale_factor, const int &x_elements,
                                          const int &y_elements, cufftHandle cufft_plan, cublasHandle_t scale_plan,
                                          const uint32_t &device_id, cudaStream_t cuda_stream);

CUDA_LIB_EXPORT cudaError_t CalculateIRFFT(cufftComplex *x_ptr, cufftComplex *w_ptr, float *y_ptr,
                                           const bool &is_onesided, const double &scale_factor, const int &x_elements,
                                           const int &y_elements, cufftHandle cufft_plan, cublasHandle_t scale_plan,
                                           const uint32_t &device_id, cudaStream_t cuda_stream);

CUDA_LIB_EXPORT cudaError_t CalculateIRFFT(cufftDoubleComplex *x_ptr, cufftDoubleComplex *w_ptr, double *y_ptr,
                                           const bool &is_onesided, const double &scale_factor, const int &x_elements,
                                           const int &y_elements, cufftHandle cufft_plan, cublasHandle_t scale_plan,
                                           const uint32_t &device_id, cudaStream_t cuda_stream);

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_FFT_WITH_SIZE_IMPL_CUH_
