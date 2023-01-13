/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_PUBLIC_CUSOLVER_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_PUBLIC_CUSOLVER_H_
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cusolverDn.h>
#include "plugin/device/gpu/hal/device/gpu_common.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/complex.h"
namespace mindspore {
namespace cusolver {
// 1. ormqr
template <typename Dtype>
void ormqr_buffersize(cusolverDnHandle_t handle, cublasSideMode_t side, cublasOperation_t trans, int m, int n, int k,
                      Dtype *A, int lda, Dtype *tau, Dtype *C, int ldc, int *lwork) {
  return;
}
template <>
void ormqr_buffersize<float>(cusolverDnHandle_t handle, cublasSideMode_t side, cublasOperation_t trans, int m, int n,
                             int k, float *A, int lda, float *tau, float *C, int ldc, int *lwork);
template <>
void ormqr_buffersize<double>(cusolverDnHandle_t handle, cublasSideMode_t side, cublasOperation_t trans, int m, int n,
                              int k, double *A, int lda, double *tau, double *C, int ldc, int *lwork);
template <>
void ormqr_buffersize<utils::Complex<float>>(cusolverDnHandle_t handle, cublasSideMode_t side, cublasOperation_t trans,
                                             int m, int n, int k, utils::Complex<float> *A, int lda,
                                             utils::Complex<float> *tau, utils::Complex<float> *C, int ldc, int *lwork);
template <>
void ormqr_buffersize<utils::Complex<double>>(cusolverDnHandle_t handle, cublasSideMode_t side, cublasOperation_t trans,
                                              int m, int n, int k, utils::Complex<double> *A, int lda,
                                              utils::Complex<double> *tau, utils::Complex<double> *C, int ldc,
                                              int *lwork);
// run
template <typename Dtype>
void ormqr(cusolverDnHandle_t handle, cublasSideMode_t side, cublasOperation_t trans, int m, int n, int k, Dtype *A,
           int lda, Dtype *tau, Dtype *C, int ldc, Dtype *work, int lwork, int *dev_info) {
  return;
}
template <>
void ormqr<float>(cusolverDnHandle_t handle, cublasSideMode_t side, cublasOperation_t trans, int m, int n, int k,
                  float *A, int lda, float *tau, float *C, int ldc, float *work, int lwork, int *dev_info);
template <>
void ormqr<double>(cusolverDnHandle_t handle, cublasSideMode_t side, cublasOperation_t trans, int m, int n, int k,
                   double *A, int lda, double *tau, double *C, int ldc, double *work, int lwork, int *dev_info);
template <>
void ormqr<utils::Complex<float>>(cusolverDnHandle_t handle, cublasSideMode_t side, cublasOperation_t trans, int m,
                                  int n, int k, utils::Complex<float> *A, int lda, utils::Complex<float> *tau,
                                  utils::Complex<float> *C, int ldc, utils::Complex<float> *work, int lwork,
                                  int *dev_info);
template <>
void ormqr<utils::Complex<double>>(cusolverDnHandle_t handle, cublasSideMode_t side, cublasOperation_t trans, int m,
                                   int n, int k, utils::Complex<double> *A, int lda, utils::Complex<double> *tau,
                                   utils::Complex<double> *C, int ldc, utils::Complex<double> *work, int lwork,
                                   int *dev_info);
// 2. orgqr
// buffersize
template <typename Dtype>
void orgqr_buffersize(cusolverDnHandle_t handle, int m, int n, int k, Dtype *A, int lda, Dtype *tau, int *lwork) {
  return;
}
template <>
void orgqr_buffersize<float>(cusolverDnHandle_t handle, int m, int n, int k, float *A, int lda, float *tau, int *lwork);
template <>
void orgqr_buffersize<double>(cusolverDnHandle_t handle, int m, int n, int k, double *A, int lda, double *tau,
                              int *lwork);
template <>
void orgqr_buffersize<utils::Complex<float>>(cusolverDnHandle_t handle, int m, int n, int k, utils::Complex<float> *A,
                                             int lda, utils::Complex<float> *tau, int *lwork);
template <>
void orgqr_buffersize<utils::Complex<double>>(cusolverDnHandle_t handle, int m, int n, int k, utils::Complex<double> *A,
                                              int lda, utils::Complex<double> *tau, int *lwork);
// run
template <typename Dtype>
void orgqr(cusolverDnHandle_t handle, int m, int n, int k, Dtype *A, int lda, Dtype *tau, Dtype *work, int lwork,
           int *dev_info) {
  return;
}
template <>
void orgqr<float>(cusolverDnHandle_t handle, int m, int n, int k, float *A, int lda, float *tau, float *work, int lwork,
                  int *dev_info);
template <>
void orgqr<double>(cusolverDnHandle_t handle, int m, int n, int k, double *A, int lda, double *tau, double *work,
                   int lwork, int *dev_info);
template <>
void orgqr<utils::Complex<float>>(cusolverDnHandle_t handle, int m, int n, int k, utils::Complex<float> *A, int lda,
                                  utils::Complex<float> *tau, utils::Complex<float> *work, int lwork, int *dev_info);
template <>
void orgqr<utils::Complex<double>>(cusolverDnHandle_t handle, int m, int n, int k, utils::Complex<double> *A, int lda,
                                   utils::Complex<double> *tau, utils::Complex<double> *work, int lwork, int *dev_info);

// 3. geqrf
// buffersize
template <typename Dtype>
void geqrf_buffersize(cusolverDnHandle_t handle, int m, int n, Dtype *A, int lda, int *lwork) {
  return;
}
template <>
void geqrf_buffersize<float>(cusolverDnHandle_t handle, int m, int n, float *A, int lda, int *lwork);
template <>
void geqrf_buffersize<double>(cusolverDnHandle_t handle, int m, int n, double *A, int lda, int *lwork);
template <>
void geqrf_buffersize<utils::Complex<float>>(cusolverDnHandle_t handle, int m, int n, utils::Complex<float> *A, int lda,
                                             int *lwork);
template <>
void geqrf_buffersize<utils::Complex<double>>(cusolverDnHandle_t handle, int m, int n, utils::Complex<double> *A,
                                              int lda, int *lwork);
// run
template <typename Dtype>
void geqrf(cusolverDnHandle_t handle, int m, int n, Dtype *A, int lda, Dtype *output_tau, Dtype *work, int lwork,
           int *dev_info) {
  return;
}
template <>
void geqrf<float>(cusolverDnHandle_t handle, int m, int n, float *A, int lda, float *output_tau, float *work, int lwork,
                  int *dev_info);
template <>
void geqrf<double>(cusolverDnHandle_t handle, int m, int n, double *A, int lda, double *output_tau, double *work,
                   int lwork, int *dev_info);
template <>
void geqrf<utils::Complex<float>>(cusolverDnHandle_t handle, int m, int n, utils::Complex<float> *A, int lda,
                                  utils::Complex<float> *output_tau, utils::Complex<float> *work, int lwork,
                                  int *dev_info);
template <>
void geqrf<utils::Complex<double>>(cusolverDnHandle_t handle, int m, int n, utils::Complex<double> *A, int lda,
                                   utils::Complex<double> *output_tau, utils::Complex<double> *work, int lwork,
                                   int *dev_info);
}  // namespace cusolver
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_PUBLIC_CUSOLVER_H_
