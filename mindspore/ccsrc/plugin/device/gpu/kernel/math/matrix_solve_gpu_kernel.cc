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

#include "plugin/device/gpu/kernel/math/matrix_solve_gpu_kernel.h"
#include <vector>
#include "mindspore/core/ops/matrix_solve.h"
#include "plugin/device/gpu/kernel/gpu_kernel_utils.h"

namespace mindspore {
namespace kernel {
namespace {
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
inline cublasStatus_t cublasXgetrfBatched(cublasHandle_t handle, int m, cuComplex *const matrix_array[],
                                          int *pivot_array, int *info_array, int batch_size) {
  return cublasCgetrfBatched(handle, m, matrix_array, m, pivot_array, info_array, batch_size);
}
inline cublasStatus_t cublasXgetrfBatched(cublasHandle_t handle, int m, cuDoubleComplex *const matrix_array[],
                                          int *pivot_array, int *info_array, int batch_size) {
  return cublasZgetrfBatched(handle, m, matrix_array, m, pivot_array, info_array, batch_size);
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
                                          const cuComplex *const matrix_array[], const int *pivot_array,
                                          cuComplex *const rhs_array[], int *info, int batch_size) {
  return cublasCgetrsBatched(handle, trans, m, k, matrix_array, m, pivot_array, rhs_array, m, info, batch_size);
}
inline cublasStatus_t cublasXgetrsBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int k,
                                          const cuDoubleComplex *const matrix_array[], const int *pivot_array,
                                          cuDoubleComplex *const rhs_array[], int *info, int batch_size) {
  return cublasZgetrsBatched(handle, trans, m, k, matrix_array, m, pivot_array, rhs_array, m, info, batch_size);
}
}  // namespace
bool MatrixSolveGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                   const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();

  const auto dtype = inputs.at(kIndex0)->GetDtype();

  auto kernel_ptr = std::make_shared<ops::MatrixSolve>(base_operator->GetPrim());
  bool adjoint = kernel_ptr->get_adjoint();

  if (dtype == kNumberTypeComplex64 || dtype == kNumberTypeComplex128) {
    blas_option_ = adjoint ? CUBLAS_OP_C : CUBLAS_OP_T;
    trans_ = adjoint;
  } else {
    // Converting row major to col major is the same as reverting the trans flag
    blas_option_ = adjoint ? CUBLAS_OP_N : CUBLAS_OP_T;
    trans_ = false;
  }

  blas_handle_ = device::gpu::GPUDeviceManager::GetInstance().GetCublasHandle();

  if (!MatchKernelFunc(base_operator, inputs, outputs)) {
    return false;
  }

  return true;
}

int MatrixSolveGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                    const std::vector<KernelTensorPtr> &outputs,
                                    const std::map<uint32_t, tensor::TensorPtr> &) {
  if (auto ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
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
  const size_t type_size = GetTypeByte(TypeIdToType(inputs.at(kIndex1)->GetDtype()));

  workspace_size_list_.clear();
  workspace_size_list_ = {
    kShape3D * sizeof(size_t),      // dev_shape
    kShape3D * sizeof(size_t),      // dev_axis
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
bool MatrixSolveGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                           const std::vector<AddressPtr> &workspace,
                                           const std::vector<AddressPtr> &outputs) {
  T *matrix = GetDeviceAddress<T>(inputs, kIndex0);
  T *rhs = GetDeviceAddress<T>(inputs, kIndex1);

  auto dev_shape = GetDeviceAddress<size_t>(workspace, kIndex0);
  auto dev_axis = GetDeviceAddress<size_t>(workspace, kIndex1);
  auto matrix_col_major = GetDeviceAddress<T>(workspace, kIndex2);
  auto rhs_col_major = GetDeviceAddress<T>(workspace, kIndex3);
  auto piv_array = GetDeviceAddress<int>(workspace, kIndex4);
  auto info_array = GetDeviceAddress<int>(workspace, kIndex5);
  auto matrix_device_array = GetDeviceAddress<T *>(workspace, kIndex6);
  auto rhs_device_array = GetDeviceAddress<T *>(workspace, kIndex7);

  T *output = GetDeviceAddress<T>(outputs, kIndex0);

  // 1. Convert matrix and rhs to column major
  // Transpose matrix if complex adjoint
  if (trans_) {
    const std::vector<size_t> matrix_shape = {LongToSize(batch_num_), LongToSize(m_), LongToSize(m_)};
    MatrixTransposeND(matrix, matrix_shape, {kDim0, kDim2, kDim1}, dev_shape, dev_axis, matrix_col_major, cuda_stream_,
                      kernel_name_);
  } else {
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
      cudaMemcpyAsync(matrix_col_major, matrix, inputs[kIndex0]->size, cudaMemcpyDeviceToDevice, cuda_stream_),
      "cudaMemcpyAsync dst failed");
  }
  // Transpose rhs to column major
  const std::vector<size_t> rhs_shape = {LongToSize(batch_num_), LongToSize(m_), LongToSize(k_)};
  MatrixTransposeND(rhs, rhs_shape, {kDim0, kDim2, kDim1}, dev_shape, dev_axis, rhs_col_major, cuda_stream_,
                    kernel_name_);

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
  MatrixTransposeND(rhs_col_major, rhs_col_shape, {kDim0, kDim2, kDim1}, dev_shape, dev_axis, output, cuda_stream_,
                    kernel_name_);

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
     &MatrixSolveGpuKernelMod::LaunchKernel<cuComplex>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeComplex128)
       .AddInputAttr(kNumberTypeComplex128)
       .AddOutputAttr(kNumberTypeComplex128),
     &MatrixSolveGpuKernelMod::LaunchKernel<cuDoubleComplex>},
  };
  return func_list;
}
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, MatrixSolve, MatrixSolveGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
