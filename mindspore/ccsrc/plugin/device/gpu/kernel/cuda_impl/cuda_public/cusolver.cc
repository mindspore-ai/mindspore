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

#include "plugin/device/gpu/kernel/cuda_impl/cuda_public/cusolver.h"
namespace mindspore {
namespace cusolver {
template <>
void ormqr_buffersize<float>(cusolverDnHandle_t handle, cublasSideMode_t side, cublasOperation_t trans, int m, int n,
                             int k, float *A, int lda, float *tau, float *C, int ldc, int *lwork) {
  CHECK_CUSOLVER_RET_WITH_EXCEPT_NOTRACE(
    cusolverDnSormqr_bufferSize(handle, side, trans, m, n, k, A, lda, tau, C, ldc, lwork),
    "cusolver query ormqr work size failed.");
}
template <>
void ormqr_buffersize<double>(cusolverDnHandle_t handle, cublasSideMode_t side, cublasOperation_t trans, int m, int n,
                              int k, double *A, int lda, double *tau, double *C, int ldc, int *lwork) {
  CHECK_CUSOLVER_RET_WITH_EXCEPT_NOTRACE(
    cusolverDnDormqr_bufferSize(handle, side, trans, m, n, k, A, lda, tau, C, ldc, lwork),
    "cusolver query ormqr work size failed.");
}
template <>
void ormqr_buffersize<utils::Complex<float>>(cusolverDnHandle_t handle, cublasSideMode_t side, cublasOperation_t trans,
                                             int m, int n, int k, utils::Complex<float> *A, int lda,
                                             utils::Complex<float> *tau, utils::Complex<float> *C, int ldc,
                                             int *lwork) {
  CHECK_CUSOLVER_RET_WITH_EXCEPT_NOTRACE(
    cusolverDnCunmqr_bufferSize(handle, side, trans, m, n, k, reinterpret_cast<cuComplex *>(A), lda,
                                reinterpret_cast<cuComplex *>(tau), reinterpret_cast<cuComplex *>(C), ldc, lwork),
    "cusolver query ormqr work size failed.");
}
template <>
void ormqr_buffersize<utils::Complex<double>>(cusolverDnHandle_t handle, cublasSideMode_t side, cublasOperation_t trans,
                                              int m, int n, int k, utils::Complex<double> *A, int lda,
                                              utils::Complex<double> *tau, utils::Complex<double> *C, int ldc,
                                              int *lwork) {
  CHECK_CUSOLVER_RET_WITH_EXCEPT_NOTRACE(
    cusolverDnZunmqr_bufferSize(handle, side, trans, m, n, k, reinterpret_cast<cuDoubleComplex *>(A), lda,
                                reinterpret_cast<cuDoubleComplex *>(tau), reinterpret_cast<cuDoubleComplex *>(C), ldc,
                                lwork),
    "cusolver query ormqr work size failed.");
}
template <>
void ormqr<float>(cusolverDnHandle_t handle, cublasSideMode_t side, cublasOperation_t trans, int m, int n, int k,
                  float *A, int lda, float *tau, float *C, int ldc, float *work, int lwork, int *dev_info) {
  CHECK_CUSOLVER_RET_WITH_EXCEPT_NOTRACE(
    cusolverDnSormqr(handle, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, dev_info),
    "cusolver ormqr failed.");
}
template <>
void ormqr<double>(cusolverDnHandle_t handle, cublasSideMode_t side, cublasOperation_t trans, int m, int n, int k,
                   double *A, int lda, double *tau, double *C, int ldc, double *work, int lwork, int *dev_info) {
  CHECK_CUSOLVER_RET_WITH_EXCEPT_NOTRACE(
    cusolverDnDormqr(handle, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, dev_info),
    "cusolver ormqr failed.");
}
template <>
void ormqr<utils::Complex<float>>(cusolverDnHandle_t handle, cublasSideMode_t side, cublasOperation_t trans, int m,
                                  int n, int k, utils::Complex<float> *A, int lda, utils::Complex<float> *tau,
                                  utils::Complex<float> *C, int ldc, utils::Complex<float> *work, int lwork,
                                  int *dev_info) {
  CHECK_CUSOLVER_RET_WITH_EXCEPT_NOTRACE(
    cusolverDnCunmqr(handle, side, trans, m, n, k, reinterpret_cast<cuComplex *>(A), lda,
                     reinterpret_cast<cuComplex *>(tau), reinterpret_cast<cuComplex *>(C), ldc,
                     reinterpret_cast<cuComplex *>(work), lwork, dev_info),
    "cusolver ormqr failed.");
}
template <>
void ormqr<utils::Complex<double>>(cusolverDnHandle_t handle, cublasSideMode_t side, cublasOperation_t trans, int m,
                                   int n, int k, utils::Complex<double> *A, int lda, utils::Complex<double> *tau,
                                   utils::Complex<double> *C, int ldc, utils::Complex<double> *work, int lwork,
                                   int *dev_info) {
  CHECK_CUSOLVER_RET_WITH_EXCEPT_NOTRACE(
    cusolverDnZunmqr(handle, side, trans, m, n, k, reinterpret_cast<cuDoubleComplex *>(A), lda,
                     reinterpret_cast<cuDoubleComplex *>(tau), reinterpret_cast<cuDoubleComplex *>(C), ldc,
                     reinterpret_cast<cuDoubleComplex *>(work), lwork, dev_info),
    "cusolver ormqr failed.");
}

template <>
void orgqr_buffersize<float>(cusolverDnHandle_t handle, int m, int n, int k, float *A, int lda, float *tau,
                             int *lwork) {
  CHECK_CUSOLVER_RET_WITH_EXCEPT_NOTRACE(cusolverDnSorgqr_bufferSize(handle, m, n, k, A, lda, tau, lwork),
                                         "cusolver query orgqr work size failed.");
}
template <>
void orgqr_buffersize<double>(cusolverDnHandle_t handle, int m, int n, int k, double *A, int lda, double *tau,
                              int *lwork) {
  CHECK_CUSOLVER_RET_WITH_EXCEPT_NOTRACE(cusolverDnDorgqr_bufferSize(handle, m, n, k, A, lda, tau, lwork),
                                         "cusolver query orgqr work size failed.");
}
template <>
void orgqr_buffersize<utils::Complex<float>>(cusolverDnHandle_t handle, int m, int n, int k, utils::Complex<float> *A,
                                             int lda, utils::Complex<float> *tau, int *lwork) {
  CHECK_CUSOLVER_RET_WITH_EXCEPT_NOTRACE(cusolverDnCungqr_bufferSize(handle, m, n, k, reinterpret_cast<cuComplex *>(A),
                                                                     lda, reinterpret_cast<cuComplex *>(tau), lwork),
                                         "cusolver query orgqr work size failed.");
}
template <>
void orgqr_buffersize<utils::Complex<double>>(cusolverDnHandle_t handle, int m, int n, int k, utils::Complex<double> *A,
                                              int lda, utils::Complex<double> *tau, int *lwork) {
  CHECK_CUSOLVER_RET_WITH_EXCEPT_NOTRACE(
    cusolverDnZungqr_bufferSize(handle, m, n, k, reinterpret_cast<cuDoubleComplex *>(A), lda,
                                reinterpret_cast<cuDoubleComplex *>(tau), lwork),
    "cusolver query orgqr work size failed.");
}
template <>
void orgqr<float>(cusolverDnHandle_t handle, int m, int n, int k, float *A, int lda, float *tau, float *work, int lwork,
                  int *dev_info) {
  CHECK_CUSOLVER_RET_WITH_EXCEPT_NOTRACE(cusolverDnSorgqr(handle, m, n, k, A, lda, tau, work, lwork, dev_info),
                                         "cusolver orgqr failed.");
}
template <>
void orgqr<double>(cusolverDnHandle_t handle, int m, int n, int k, double *A, int lda, double *tau, double *work,
                   int lwork, int *dev_info) {
  CHECK_CUSOLVER_RET_WITH_EXCEPT_NOTRACE(cusolverDnDorgqr(handle, m, n, k, A, lda, tau, work, lwork, dev_info),
                                         "cusolver orgqr failed.");
}
template <>
void orgqr<utils::Complex<float>>(cusolverDnHandle_t handle, int m, int n, int k, utils::Complex<float> *A, int lda,
                                  utils::Complex<float> *tau, utils::Complex<float> *work, int lwork, int *dev_info) {
  CHECK_CUSOLVER_RET_WITH_EXCEPT_NOTRACE(
    cusolverDnCungqr(handle, m, n, k, reinterpret_cast<cuComplex *>(A), lda, reinterpret_cast<cuComplex *>(tau),
                     reinterpret_cast<cuComplex *>(work), lwork, dev_info),
    "cusolver orgqr failed.");
}
template <>
void orgqr<utils::Complex<double>>(cusolverDnHandle_t handle, int m, int n, int k, utils::Complex<double> *A, int lda,
                                   utils::Complex<double> *tau, utils::Complex<double> *work, int lwork,
                                   int *dev_info) {
  CHECK_CUSOLVER_RET_WITH_EXCEPT_NOTRACE(cusolverDnZungqr(handle, m, n, k, reinterpret_cast<cuDoubleComplex *>(A), lda,
                                                          reinterpret_cast<cuDoubleComplex *>(tau),
                                                          reinterpret_cast<cuDoubleComplex *>(work), lwork, dev_info),
                                         "cusolver orgqr failed.");
}

template <>
void geqrf_buffersize<float>(cusolverDnHandle_t handle, int m, int n, float *A, int lda, int *lwork) {
  CHECK_CUSOLVER_RET_WITH_EXCEPT_NOTRACE(cusolverDnSgeqrf_bufferSize(handle, m, n, A, lda, lwork),
                                         "cusolver query geqrf work size failed.");
}
template <>
void geqrf_buffersize<double>(cusolverDnHandle_t handle, int m, int n, double *A, int lda, int *lwork) {
  CHECK_CUSOLVER_RET_WITH_EXCEPT_NOTRACE(cusolverDnDgeqrf_bufferSize(handle, m, n, A, lda, lwork),
                                         "cusolver query geqrf work size failed.");
}
template <>
void geqrf_buffersize<utils::Complex<float>>(cusolverDnHandle_t handle, int m, int n, utils::Complex<float> *A, int lda,
                                             int *lwork) {
  CHECK_CUSOLVER_RET_WITH_EXCEPT_NOTRACE(
    cusolverDnCgeqrf_bufferSize(handle, m, n, reinterpret_cast<cuComplex *>(A), lda, lwork),
    "cusolver query geqrf work size failed.");
}
template <>
void geqrf_buffersize<utils::Complex<double>>(cusolverDnHandle_t handle, int m, int n, utils::Complex<double> *A,
                                              int lda, int *lwork) {
  CHECK_CUSOLVER_RET_WITH_EXCEPT_NOTRACE(
    cusolverDnZgeqrf_bufferSize(handle, m, n, reinterpret_cast<cuDoubleComplex *>(A), lda, lwork),
    "cusolver query geqrf work size failed.");
}
template <>
void geqrf<float>(cusolverDnHandle_t handle, int m, int n, float *A, int lda, float *output_tau, float *work, int lwork,
                  int *dev_info) {
  CHECK_CUSOLVER_RET_WITH_EXCEPT_NOTRACE(cusolverDnSgeqrf(handle, m, n, A, lda, output_tau, work, lwork, dev_info),
                                         "cusolver geqrf failed.");
}
template <>
void geqrf<double>(cusolverDnHandle_t handle, int m, int n, double *A, int lda, double *output_tau, double *work,
                   int lwork, int *dev_info) {
  CHECK_CUSOLVER_RET_WITH_EXCEPT_NOTRACE(cusolverDnDgeqrf(handle, m, n, A, lda, output_tau, work, lwork, dev_info),
                                         "cusolver geqrf failed.");
}
template <>
void geqrf<utils::Complex<float>>(cusolverDnHandle_t handle, int m, int n, utils::Complex<float> *A, int lda,
                                  utils::Complex<float> *output_tau, utils::Complex<float> *work, int lwork,
                                  int *dev_info) {
  CHECK_CUSOLVER_RET_WITH_EXCEPT_NOTRACE(
    cusolverDnCgeqrf(handle, m, n, reinterpret_cast<cuComplex *>(A), lda, reinterpret_cast<cuComplex *>(output_tau),
                     reinterpret_cast<cuComplex *>(work), lwork, dev_info),
    "cusolver geqrf failed.");
}
template <>
void geqrf<utils::Complex<double>>(cusolverDnHandle_t handle, int m, int n, utils::Complex<double> *A, int lda,
                                   utils::Complex<double> *output_tau, utils::Complex<double> *work, int lwork,
                                   int *dev_info) {
  CHECK_CUSOLVER_RET_WITH_EXCEPT_NOTRACE(cusolverDnZgeqrf(handle, m, n, reinterpret_cast<cuDoubleComplex *>(A), lda,
                                                          reinterpret_cast<cuDoubleComplex *>(output_tau),
                                                          reinterpret_cast<cuDoubleComplex *>(work), lwork, dev_info),
                                         "cusolver geqrf failed.");
}
}  // namespace cusolver
}  // namespace mindspore
