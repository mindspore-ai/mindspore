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

#include "plugin/device/gpu/kernel/math/matrix_triangular_solve_gpu_kernel.h"
#include <utility>
#include <memory>
#include <algorithm>
#include <cmath>
#include <string>
#include <functional>
#include <vector>
#include <map>
#include "abstract/utils.h"
#include "kernel/common_utils.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/complex.h"

namespace mindspore {
namespace kernel {
namespace {
using mindspore::utils::Complex;
using KernelRunFunc = MatrixTriangularSolveGpuKernelMod::KernelRunFunc;

inline cublasStatus_t Trsm(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans,
                           cublasDiagType_t diag, int m, int n, const float *alpha, const float *A, int lda, float *B,
                           int ldb) {
  return cublasStrsm(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb);
}
inline cublasStatus_t Trsm(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans,
                           cublasDiagType_t diag, int m, int n, const double *alpha, const double *A, int lda,
                           double *B, int ldb) {
  return cublasDtrsm(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb);
}
inline cublasStatus_t Trsm(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans,
                           cublasDiagType_t diag, int m, int n, Complex<float> *alpha, Complex<float> *A, int lda,
                           Complex<float> *B, int ldb) {
  return cublasCtrsm(handle, side, uplo, trans, diag, m, n, reinterpret_cast<cuComplex *>(alpha),
                     reinterpret_cast<cuComplex *>(A), lda, reinterpret_cast<cuComplex *>(B), ldb);
}
inline cublasStatus_t Trsm(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans,
                           cublasDiagType_t diag, int m, int n, Complex<double> *alpha, Complex<double> *A, int lda,
                           Complex<double> *B, int ldb) {
  return cublasZtrsm(handle, side, uplo, trans, diag, m, n, reinterpret_cast<cuDoubleComplex *>(alpha),
                     reinterpret_cast<cuDoubleComplex *>(A), lda, reinterpret_cast<cuDoubleComplex *>(B), ldb);
}

inline cublasStatus_t TrsmBatched(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo,
                                  cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const float *alpha,
                                  const float *const A[], int lda, float *B[], int ldb, int batch_size) {
  return cublasStrsmBatched(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, batch_size);
}
inline cublasStatus_t TrsmBatched(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo,
                                  cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const double *alpha,
                                  const double *const A[], int lda, double *B[], int ldb, int batch_size) {
  return cublasDtrsmBatched(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, batch_size);
}
inline cublasStatus_t TrsmBatched(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo,
                                  cublasOperation_t trans, cublasDiagType_t diag, int m, int n, Complex<float> *alpha,
                                  Complex<float> *A[], int lda, Complex<float> *B[], int ldb, int batch_size) {
  return cublasCtrsmBatched(handle, side, uplo, trans, diag, m, n, reinterpret_cast<cuComplex *>(alpha),
                            reinterpret_cast<cuComplex **>(A), lda, reinterpret_cast<cuComplex **>(B), ldb, batch_size);
}
inline cublasStatus_t TrsmBatched(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo,
                                  cublasOperation_t trans, cublasDiagType_t diag, int m, int n, Complex<double> *alpha,
                                  Complex<double> *A[], int lda, Complex<double> *B[], int ldb, int batch_size) {
  return cublasZtrsmBatched(handle, side, uplo, trans, diag, m, n, reinterpret_cast<cuDoubleComplex *>(alpha),
                            reinterpret_cast<cuDoubleComplex **>(A), lda, reinterpret_cast<cuDoubleComplex **>(B), ldb,
                            batch_size);
}

void ComputeBatchIndicesHelper(const size_t batch_elements, const std::vector<int64_t> &reshape,
                               const std::vector<int64_t> &bcast, std::vector<int64_t> *broadcast_indices) {
  broadcast_indices->resize(batch_elements);
  int64_t num_output_elements = 1;
  int64_t num_input_elements = 1;
  for (int64_t i = reshape.size() - 1; i >= 0; --i) {
    const int64_t dim = std::max(reshape[i], bcast[i]);
    const int64_t incr = bcast[i] > 1 ? 0 : num_input_elements;
    for (int64_t k = 0; k < (dim - 1) * num_output_elements; ++k) {
      (*broadcast_indices)[num_output_elements + k] = (*broadcast_indices)[k] + incr;
    }
    num_output_elements *= dim;
    num_input_elements *= reshape[i];
  }
}

void BroadcastBatchIndices(const std::vector<int64_t> &a_batch_shape, const std::vector<int64_t> &b_batch_shape,
                           std::vector<int64_t> batch_shape, std::vector<int64_t> *a_broadcast_indices,
                           std::vector<int64_t> *b_broadcast_indices) {
  // compute intermediate variables
  size_t batch_elements = std::accumulate(batch_shape.begin(), batch_shape.end(), 1, std::multiplies<int64_t>());
  size_t rank = batch_shape.size();
  std::vector<int64_t> a_reshape = a_batch_shape;
  std::vector<int64_t> b_reshape = b_batch_shape;
  reverse(a_reshape.begin(), a_reshape.end());
  reverse(b_reshape.begin(), b_reshape.end());
  reverse(batch_shape.begin(), batch_shape.end());
  if (a_reshape.size() != rank) {
    a_reshape.resize(rank, 1);
  }
  if (b_reshape.size() != rank) {
    b_reshape.resize(rank, 1);
  }
  std::vector<int64_t> a_bcast(rank);
  std::vector<int64_t> b_bcast(rank);
  for (size_t i = 0; i < rank; i++) {
    a_bcast[i] = a_reshape[i] == 1 ? batch_shape[i] : 1;
    b_bcast[i] = b_reshape[i] == 1 ? batch_shape[i] : 1;
  }
  reverse(a_reshape.begin(), a_reshape.end());
  reverse(b_reshape.begin(), b_reshape.end());
  reverse(a_bcast.begin(), a_bcast.end());
  reverse(b_bcast.begin(), b_bcast.end());

  // compute batch indices for mapping
  ComputeBatchIndicesHelper(batch_elements, a_reshape, a_bcast, a_broadcast_indices);
  ComputeBatchIndicesHelper(batch_elements, b_reshape, b_bcast, b_broadcast_indices);
}
}  // namespace

bool MatrixTriangularSolveGpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                             const std::vector<KernelTensorPtr> &inputs,
                                             const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it got empty inputs or outputs, which is invalid.";
    return false;
  }
  if (!MatchKernelFunc(base_operator, inputs, outputs)) {
    return false;
  }

  // the size of per element of args
  unit_size_ = abstract::TypeIdSize(inputs[kIndex0]->GetDtype());

  // set mode and operation
  lower_ = GetValue<bool>(base_operator->GetAttr("lower"));
  adjoint_ = GetValue<bool>(base_operator->GetAttr("adjoint"));
  uplo_ = lower_ ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;
  trans_ = adjoint_ ? CUBLAS_OP_C : CUBLAS_OP_N;
  blas_handle_ = device::gpu::GPUDeviceManager::GetInstance().GetCublasHandle();

  return true;
}

int MatrixTriangularSolveGpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                              const std::vector<KernelTensorPtr> &inputs,
                                              const std::vector<KernelTensorPtr> &outputs,
                                              const std::map<uint32_t, tensor::TensorPtr> &) {
  // If any input shape contains -1, means input shape is dynamic, so just return do nothing.
  for (const auto &input : inputs) {
    auto input_shape = input->GetShapeVector();
    if (!IsValidShape(input_shape)) {
      return KRET_UNKNOWN_SHAPE;
    }
  }

  std::vector<int64_t> output_shape = std::vector<int64_t>(outputs.at(kIndex0)->GetShapeVector());
  if (!IsValidShape(output_shape)) {
    return KRET_UNKNOWN_SHAPE;
  }

  std::vector<int64_t> a_shape = std::vector<int64_t>(inputs.at(kIndex0)->GetShapeVector());
  std::vector<int64_t> b_shape = std::vector<int64_t>(inputs.at(kIndex1)->GetShapeVector());
  size_t a_elements = std::accumulate(a_shape.begin(), a_shape.end(), 1, std::multiplies<int64_t>());
  size_t b_elements = std::accumulate(b_shape.begin(), b_shape.end(), 1, std::multiplies<int64_t>());
  size_t output_elements = std::accumulate(output_shape.begin(), output_shape.end(), 1, std::multiplies<int64_t>());
  m_ = a_shape[a_shape.size() - 1];
  n_ = b_shape[b_shape.size() - 1];
  a_batch_num_ = static_cast<int64_t>(a_elements / (m_ * m_));
  b_batch_num_ = static_cast<int64_t>(b_elements / (m_ * n_));
  output_batch_num_ = static_cast<int64_t>(output_elements / (m_ * n_));

  if (output_elements == 0) {
    is_null_input_ = true;
  }

  // check whether we need to broadcast batch indices
  if (a_shape.size() == kSizeTwo && b_shape.size() == kSizeTwo) {
    is_bcast_required_ = false;
  } else if (a_shape.size() > kSizeTwo && b_shape.size() == kSizeTwo) {
    is_bcast_required_ = true;
    std::vector<int64_t> a_batch_shape(a_shape.begin(), a_shape.end() - kIndex2);
    std::vector<int64_t> b_batch_shape = {1};
    std::vector<int64_t> batch_shape(output_shape.begin(), output_shape.end() - kIndex2);
    BroadcastBatchIndices(a_batch_shape, b_batch_shape, batch_shape, &a_broadcast_indices_, &b_broadcast_indices_);
  } else if (a_shape.size() == kSizeTwo && b_shape.size() > kSizeTwo) {
    is_bcast_required_ = true;
    std::vector<int64_t> a_batch_shape = {1};
    std::vector<int64_t> b_batch_shape(b_shape.begin(), b_shape.end() - kIndex2);
    std::vector<int64_t> batch_shape(output_shape.begin(), output_shape.end() - kIndex2);
    BroadcastBatchIndices(a_batch_shape, b_batch_shape, batch_shape, &a_broadcast_indices_, &b_broadcast_indices_);
  } else {
    std::vector<int64_t> a_batch_shape(a_shape.begin(), a_shape.end() - kIndex2);
    std::vector<int64_t> b_batch_shape(b_shape.begin(), b_shape.end() - kIndex2);
    if (a_batch_shape == b_batch_shape) {
      is_bcast_required_ = false;
    } else {
      is_bcast_required_ = true;
      std::vector<int64_t> batch_shape(output_shape.begin(), output_shape.end() - kIndex2);
      BroadcastBatchIndices(a_batch_shape, b_batch_shape, batch_shape, &a_broadcast_indices_, &b_broadcast_indices_);
    }
  }

  input_size_list_.clear();
  output_size_list_.clear();
  workspace_size_list_.clear();
  input_size_list_.emplace_back(a_elements * unit_size_);
  input_size_list_.emplace_back(b_elements * unit_size_);
  output_size_list_.emplace_back(output_elements * unit_size_);
  workspace_size_list_.emplace_back(output_batch_num_ * sizeof(void *));
  workspace_size_list_.emplace_back(output_batch_num_ * sizeof(void *));

  return KRET_OK;
}

template <typename T>
bool MatrixTriangularSolveGpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                                     const std::vector<AddressPtr> &workspace,
                                                     const std::vector<kernel::AddressPtr> &outputs) {
  T *a = GetDeviceAddress<T>(inputs, kIndex0);
  T *b = GetDeviceAddress<T>(inputs, kIndex1);
  T *output = GetDeviceAddress<T>(outputs, kIndex0);
  MS_EXCEPTION_IF_NULL(a);
  MS_EXCEPTION_IF_NULL(b);
  MS_EXCEPTION_IF_NULL(output);

  auto a_device_array = GetDeviceAddress<T *>(workspace, kIndex0);  // broadcasted
  auto output_device_array = GetDeviceAddress<T *>(workspace, kIndex1);
  MS_EXCEPTION_IF_NULL(a_device_array);
  MS_EXCEPTION_IF_NULL(output_device_array);

  const uint64_t m = m_;
  const uint64_t n = n_;
  const int64_t batch_num = output_batch_num_;

  CHECK_CUBLAS_RET_WITH_EXCEPT_NOTRACE(cublasSetStream(blas_handle_, cuda_stream_),
                                       "For 'MatrixTriangularSolveGpuKernelMod', cublasSetStream Fail");

  if (!is_bcast_required_) {
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
      cudaMemcpyAsync(output, b, inputs[kIndex1]->size, cudaMemcpyDeviceToDevice, cuda_stream_),
      "For 'MatrixTriangularSolveGpuKernelMod', copy 'rhs' from device to device failed.");
  } else {
    for (int64_t i = 0; i < batch_num; i++) {
      CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
        cudaMemcpyAsync((output + i * m * n), (b + b_broadcast_indices_[i] * m * n), m * n * sizeof(T),
                        cudaMemcpyDeviceToDevice, cuda_stream_),
        "For 'MatrixTriangularSolveGpuKernelMod', copy 'rhs' from device to device failed.");
    }
  }

  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaStreamSynchronize(cuda_stream_),
                                     "For 'MatrixTriangularSolveGpuKernelMod', launch cudaStreamSynchronized failed");

  std::vector<T *> a_ptrs;
  std::vector<T *> out_ptrs;
  a_ptrs.reserve(batch_num);
  out_ptrs.reserve(batch_num);
  if (!is_bcast_required_) {
    for (int64_t i = 0; i < batch_num; ++i) {
      a_ptrs[i] = (a + i * m * m);
      out_ptrs[i] = (output + i * m * n);
    }
  } else {
    for (int64_t i = 0; i < batch_num; i++) {
      a_ptrs[i] = (a + a_broadcast_indices_[i] * m * m);
      out_ptrs[i] = (output + i * m * n);
    }
  }

  T alpha = 1;

  // device memory required
  if (batch_num == 1) {
    CHECK_CUBLAS_RET_WITH_EXCEPT_NOTRACE(
      Trsm(blas_handle_, side_, uplo_, trans_, diag_, n, m, &alpha, a_ptrs[0], m, out_ptrs[0], n),
      "For 'MatrixTriangularSolveGpuKernelMod', launch Trsm failed.");
  } else {
    const int kMaxMatrixSizeToBatchSizeRatio = 128;
    const bool use_batched_solver = m_ <= kMaxMatrixSizeToBatchSizeRatio * batch_num;
    if (use_batched_solver) {
      CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
        cudaMemcpyAsync(a_device_array, a_ptrs.data(), batch_num * sizeof(T *), cudaMemcpyHostToDevice, cuda_stream_),
        "For 'MatrixTriangularSolveGpuKernelMod', copy data from host to device failed.");
      CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
        cudaMemcpyAsync(output_device_array, out_ptrs.data(), batch_num * sizeof(T *), cudaMemcpyHostToDevice,
                        cuda_stream_),
        "For 'MatrixTriangularSolveGpuKernelMod', copy data from host to device failed.");
      CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
        cudaStreamSynchronize(cuda_stream_),
        "For 'MatrixTriangularSolveGpuKernelMod', launch cudaStreamSynchronized failed");
      CHECK_CUBLAS_RET_WITH_EXCEPT_NOTRACE(TrsmBatched(blas_handle_, side_, uplo_, trans_, diag_, n, m, &alpha,
                                                       a_device_array, m, output_device_array, n, batch_num),
                                           "For 'MatrixSolveGpuKernelMod', launch TrsmBatched failed.");
    } else {
      for (int batch = 0; batch < batch_num; ++batch) {
        CHECK_CUBLAS_RET_WITH_EXCEPT_NOTRACE(
          Trsm(blas_handle_, side_, uplo_, trans_, diag_, n, m, &alpha, a_ptrs[batch], m, out_ptrs[batch], n),
          "For 'MatrixSolveGpuKernelMod', launch Trsm failed.");
      }
    }
  }

  return true;
}

const std::vector<std::pair<KernelAttr, KernelRunFunc>> &MatrixTriangularSolveGpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, KernelRunFunc>> func_list = {
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     &MatrixTriangularSolveGpuKernelMod::LaunchKernel<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
     &MatrixTriangularSolveGpuKernelMod::LaunchKernel<double>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeComplex64)
       .AddInputAttr(kNumberTypeComplex64)
       .AddOutputAttr(kNumberTypeComplex64),
     &MatrixTriangularSolveGpuKernelMod::LaunchKernel<Complex<float>>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeComplex128)
       .AddInputAttr(kNumberTypeComplex128)
       .AddOutputAttr(kNumberTypeComplex128),
     &MatrixTriangularSolveGpuKernelMod::LaunchKernel<Complex<double>>},
  };

  return func_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, MatrixTriangularSolve, MatrixTriangularSolveGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
