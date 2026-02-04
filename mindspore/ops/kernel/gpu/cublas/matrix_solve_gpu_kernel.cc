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

#include "kernel/gpu/cublas/matrix_solve_gpu_kernel.h"
#include <vector>
#include "mindspore/ops/infer/matrix_solve.h"
#include "kernel/gpu/cuda_impl/cuda_ops/matrix_transpose_impl.cuh"
#include "kernel/gpu/cuda_impl/cuda_ops/complex.h"

namespace mindspore {
namespace kernel {
namespace {
using mindspore::utils::Complex;
using KernelRunFunc = MatrixSolveGpuKernelMod::KernelRunFunc;
constexpr size_t kShape3D = 3;
inline cublasStatus_t cublasXgetrfBatched(cublasHandle_t handle, int m, float *const matrix_array[], int *pivot_array,
                                          int *info_array, int batch_size) {
  return cublasSgetrfBatched(handle, m, matrix_array, m, pivot_array, info_array, batch_size);
}
inline cublasStatus_t cublasXgetrfBatched(cublasHandle_t handle, int m, double *const matrix_array[], int *pivot_array,
                                          int *info_array, int batch_size) {
  return cublasDgetrfBatched(handle, m, matrix_array, m, pivot_array, info_array, batch_size);
}
inline cublasStatus_t cublasXgetrfBatched(cublasHandle_t handle, int m, Complex<float> *matrix_array[],
                                          int *pivot_array, int *info_array, int batch_size) {
  return cublasCgetrfBatched(handle, m, reinterpret_cast<cuComplex **>(matrix_array), m, pivot_array, info_array,
                             batch_size);
}
inline cublasStatus_t cublasXgetrfBatched(cublasHandle_t handle, int m, Complex<double> *matrix_array[],
                                          int *pivot_array, int *info_array, int batch_size) {
  return cublasZgetrfBatched(handle, m, reinterpret_cast<cuDoubleComplex **>(matrix_array), m, pivot_array, info_array,
                             batch_size);
}
inline cublasStatus_t cublasXgetrsBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int k,
                                          const float *const matrix_array[], const int *pivot_array,
                                          float *const rhs_array[], int *info, int batch_size) {
  return cublasSgetrsBatched(handle, trans, m, k, matrix_array, m, pivot_array, rhs_array, m, info, batch_size);
}
inline cublasStatus_t cublasXgetrsBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int k,
                                          const double *const matrix_array[], const int *pivot_array,
                                          double *const rhs_array[], int *info, int batch_size) {
  return cublasDgetrsBatched(handle, trans, m, k, matrix_array, m, pivot_array, rhs_array, m, info, batch_size);
}
inline cublasStatus_t cublasXgetrsBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int k,
                                          Complex<float> *matrix_array[], const int *pivot_array,
                                          Complex<float> *rhs_array[], int *info, int batch_size) {
  auto cu_matrix_array = reinterpret_cast<cuComplex **>(matrix_array);
  auto cu_rhs_array = reinterpret_cast<cuComplex **>(rhs_array);
  return cublasCgetrsBatched(handle, trans, m, k, cu_matrix_array, m, pivot_array, cu_rhs_array, m, info, batch_size);
}
inline cublasStatus_t cublasXgetrsBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int k,
                                          Complex<double> *matrix_array[], const int *pivot_array,
                                          Complex<double> *rhs_array[], int *info, int batch_size) {
  auto cu_matrix_array = reinterpret_cast<cuDoubleComplex **>(matrix_array);
  auto cu_rhs_array = reinterpret_cast<cuDoubleComplex **>(rhs_array);
  return cublasZgetrsBatched(handle, trans, m, k, cu_matrix_array, m, pivot_array, cu_rhs_array, m, info, batch_size);
}
}  // namespace
bool MatrixSolveGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                   const std::vector<KernelTensor *> &outputs) {
  const auto dtype = inputs.at(kIndex0)->dtype_id();

  bool adjoint = GetValue<bool>(primitive_->GetAttr("adjoint"));

  if (dtype == kNumberTypeComplex64 || dtype == kNumberTypeComplex128) {
    blas_option_ = adjoint ? CUBLAS_OP_C : CUBLAS_OP_T;
    trans_ = adjoint;
  } else {
    // Converting row major to col major is the same as reverting the trans flag
    blas_option_ = adjoint ? CUBLAS_OP_N : CUBLAS_OP_T;
    trans_ = false;
  }

  blas_handle_ = device::gpu::GPUDeviceManager::GetInstance().GetCublasHandle();

  if (!MatchKernelFunc(kernel_name_, inputs, outputs)) {
    return false;
  }

  return true;
}

int MatrixSolveGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                    const std::vector<KernelTensor *> &outputs) {
  if (auto ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }

  const auto matrix_shape = inputs.at(kIndex0)->GetShapeVector();
  const auto rhs_shape = inputs.at(kIndex1)->GetShapeVector();

  is_null_input_ = CHECK_SHAPE_NULL(LongVecToSizeVec(matrix_shape), kernel_name_, "matrix") ||
                   CHECK_SHAPE_NULL(LongVecToSizeVec(rhs_shape), kernel_name_, "rhs");

  batch_num_ = std::accumulate(matrix_shape.begin(), matrix_shape.end() - kIndex2, int64_t(1), std::multiplies{});
  m_ = matrix_shape.back();
  k_ = rhs_shape.back();

  const size_t matrix_size =
    LongToSize(std::accumulate(matrix_shape.begin(), matrix_shape.end(), int64_t(1), std::multiplies{}));
  const size_t rhs_size =
    LongToSize(std::accumulate(rhs_shape.begin(), rhs_shape.end(), int64_t(1), std::multiplies{}));
  const size_t type_size = GetTypeByte(TypeIdToType(inputs.at(kIndex1)->dtype_id()));

  workspace_size_list_.clear();
  workspace_size_list_ = {
    matrix_size * type_size,        // matrix column major
    rhs_size * type_size,           // rhs column major
    batch_num_ * m_ * sizeof(int),  // pivoting sequence
    batch_num_ * sizeof(int),       // info
    batch_num_ * sizeof(float *),   // matrix_array, the size of float* and double* are the same
    batch_num_ * sizeof(float *)    // matrix_array, the size of float* and double* are the same
  };

  return KRET_OK;
}

template <typename T>
bool MatrixSolveGpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                           const std::vector<KernelTensor *> &workspace,
                                           const std::vector<KernelTensor *> &outputs) {
  T *matrix = GetDeviceAddress<T>(inputs, kIndex0);
  T *rhs = GetDeviceAddress<T>(inputs, kIndex1);

  auto matrix_col_major = GetDeviceAddress<T>(workspace, kIndex0);
  auto rhs_col_major = GetDeviceAddress<T>(workspace, kIndex1);
  auto piv_array = GetDeviceAddress<int>(workspace, kIndex2);
  auto info_array = GetDeviceAddress<int>(workspace, kIndex3);
  auto matrix_device_array = GetDeviceAddress<T *>(workspace, kIndex4);
  auto rhs_device_array = GetDeviceAddress<T *>(workspace, kIndex5);

  T *output = GetDeviceAddress<T>(outputs, kIndex0);

  CHECK_CUBLAS_RET_WITH_EXCEPT_NOTRACE(cublasSetStream(blas_handle_, cuda_stream_),
                                       "For MatrixSolveGpuKernelMod cublasSetStream Fail");

  // 1. Convert matrix and rhs to column major
  // Transpose matrix if complex adjoint
  cudaError_t status = cudaErrorNotReady;
  if (trans_) {
    status = MatrixTranspose(matrix, LongToSize(batch_num_ * m_ * m_), SizeToInt(m_), SizeToInt(m_), matrix_col_major,
                             device_id_, cuda_stream_);
    CHECK_CUDA_STATUS(status, kernel_name_);
  } else {
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
      cudaMemcpyAsync(matrix_col_major, matrix, inputs[kIndex0]->size(), cudaMemcpyDeviceToDevice, cuda_stream_),
      "cudaMemcpyAsync dst failed");
  }
  status = MatrixTranspose(rhs, LongToSize(batch_num_ * m_ * k_), SizeToInt(m_), SizeToInt(k_), rhs_col_major,
                           device_id_, cuda_stream_);
  CHECK_CUDA_STATUS(status, kernel_name_);

  // 2. LU factorization
  // Prepare matrix_array
  std::vector<T *> matrix_host_array(batch_num_);
  for (int64_t i = 0; i < batch_num_; i++) {
    matrix_host_array[i] = matrix_col_major + i * m_ * m_;
  }
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaMemcpyAsync(matrix_device_array, matrix_host_array.data(),
                                                     batch_num_ * sizeof(T *), cudaMemcpyHostToDevice, cuda_stream_),
                                     "For 'MatrixSolveGpuKernelMod', it launch memcopy failed.");
  // Call cublasXgetrfBatched
  CHECK_CUBLAS_RET_WITH_EXCEPT_NOTRACE(
    cublasXgetrfBatched(blas_handle_, m_, matrix_device_array, piv_array, info_array, batch_num_),
    "For 'MatrixSolveGpuKernelMod', it launch cublasXgetrfBatched failed");
  // Check invertible
  std::vector<int> infos(batch_num_);
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(infos.data(), info_array, batch_num_ * sizeof(int), cudaMemcpyDeviceToHost, cuda_stream_),
    "For 'MatrixSolveGpuKernelMod', it launch cudaStreamSynchronized failed");
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaStreamSynchronize(cuda_stream_),
                                     "For 'MatrixSolveGpuKernelMod', it launch cudaStreamSynchronized failed");
  if (std::any_of(infos.begin(), infos.end(), [](int info) { return info < 0; })) {
    MS_EXCEPTION(ValueError) << "For '" << kernel_name_ << "', the input 'matrix' has illegal value.";
  }
  if (std::any_of(infos.begin(), infos.end(), [](int info) { return info > 0; })) {
    MS_EXCEPTION(ValueError) << "For '" << kernel_name_ << "', the input 'matrix' is not invertible.";
  }

  // 3. Solve Matrix
  // Prepare matrix_array and rhs_array
  std::vector<T *> rhs_host_array(batch_num_);
  for (int64_t i = 0; i < batch_num_; i++) {
    matrix_host_array[i] = matrix_col_major + i * m_ * m_;
    rhs_host_array[i] = rhs_col_major + i * m_ * k_;
  }
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaMemcpyAsync(matrix_device_array, matrix_host_array.data(),
                                                     batch_num_ * sizeof(T *), cudaMemcpyHostToDevice, cuda_stream_),
                                     "For 'MatrixSolveGpuKernelMod', it launch memcopy failed.");
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaMemcpyAsync(rhs_device_array, rhs_host_array.data(), batch_num_ * sizeof(T *),
                                                     cudaMemcpyHostToDevice, cuda_stream_),
                                     "For 'MatrixSolveGpuKernelMod', it launch memcopy failed.");
  int info = 0;
  CHECK_CUBLAS_RET_WITH_EXCEPT_NOTRACE(cublasXgetrsBatched(blas_handle_, blas_option_, m_, k_, matrix_device_array,
                                                           piv_array, rhs_device_array, &info, batch_num_),
                                       "For 'MatrixSolveGpuKernelMod', it launch cublasXgetrfBatched failed");

  // 4. Convert matrix and rhs to row major
  const std::vector<size_t> rhs_col_shape = {LongToSize(batch_num_), LongToSize(k_), LongToSize(m_)};
  status = MatrixTranspose(rhs_col_major, LongToSize(batch_num_ * m_ * k_), SizeToInt(k_), SizeToInt(m_), output,
                           device_id_, cuda_stream_);
  CHECK_CUDA_STATUS(status, kernel_name_);

  return true;
}

const std::vector<std::pair<KernelAttr, KernelRunFunc>> &MatrixSolveGpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, KernelRunFunc>> func_list = {
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     &MatrixSolveGpuKernelMod::LaunchKernel<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
     &MatrixSolveGpuKernelMod::LaunchKernel<double>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeComplex64)
       .AddInputAttr(kNumberTypeComplex64)
       .AddOutputAttr(kNumberTypeComplex64),
     &MatrixSolveGpuKernelMod::LaunchKernel<Complex<float>>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeComplex128)
       .AddInputAttr(kNumberTypeComplex128)
       .AddOutputAttr(kNumberTypeComplex128),
     &MatrixSolveGpuKernelMod::LaunchKernel<Complex<double>>},
  };
  return func_list;
}
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, MatrixSolve, MatrixSolveGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
