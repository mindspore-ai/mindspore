/**
 * Copyright 2020-2024 Huawei Technologies Co., Ltd
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

#include "kernel/gpu/cublas/solve_triangular_gpu_kernel.h"
#include <map>
#include <utility>
#include <memory>

namespace mindspore {
namespace kernel {
constexpr size_t kIndexA = 0;
constexpr size_t kIndexB = 1;
constexpr size_t kIndexX = 0;
constexpr size_t kIndexTrans = 2;
constexpr size_t kIndexLower = 3;
constexpr size_t kIndexUnitDiagonal = 4;
constexpr size_t kSquareSize = 2;
constexpr int64_t kTransN = 0;
constexpr int64_t kTransT = 1;
constexpr int64_t kTransC = 2;
using KernelRunFunc = SolveTriangularGpuKernelMod::KernelRunFunc;
bool SolveTriangularGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                       const std::vector<KernelTensor *> &outputs) {
  blas_handle_ = device::gpu::GPUDeviceManager::GetInstance().GetCublasHandle();

  bool lower = inputs[kIndexLower]->GetValueWithCheck<bool>();
  // reverting the trans flag by default, so also flip the lower flag
  lower = !lower;
  uplo_ = lower ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;

  bool unit_diagonal = inputs[kIndexUnitDiagonal]->GetValueWithCheck<bool>();
  unit_diagonal_ = unit_diagonal ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT;

  int64_t trans = inputs[kIndexTrans]->GetValueWithCheck<int64_t>();
  if (trans == kTransN) {
    trans_ = CUBLAS_OP_T;
  } else if (trans == kTransT) {
    trans_ = CUBLAS_OP_N;
  } else if (trans == kTransC) {
    // currently does not support complex.
    trans_ = CUBLAS_OP_N;
  } else {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', 'trans' must be in ['N', 'T', 'C'], but got [" << trans << "].";
  }

  if (!MatchKernelFunc(kernel_name_, inputs, outputs)) {
    return false;
  }

  return true;
}
int SolveTriangularGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                        const std::vector<KernelTensor *> &outputs) {
  if (auto ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }

  auto a_shape = LongVecToSizeVec(inputs.at(kIndexA)->GetShapeVector());
  auto b_shape = LongVecToSizeVec(inputs.at(kIndexB)->GetShapeVector());

  is_null_input_ =
    CHECK_SHAPE_NULL(a_shape, kernel_name_, "input_a") || CHECK_SHAPE_NULL(b_shape, kernel_name_, "input_b");
  // Since the shape check is done in frontend, we can suppose that the shape of a, b here is valid.
  size_t a_dims = a_shape.size();
  size_t b_dims = b_shape.size();
  m_ = a_shape[a_dims - kSquareSize];
  n_ = (b_dims == a_dims - 1) ? 1 : b_shape[b_dims - 1];
  batch_ = std::accumulate(a_shape.begin(), a_shape.end() - kSquareSize, int64_t(1), std::multiplies{});

  lda_ = SizeToInt(m_);
  ldb_ = SizeToInt(m_);

  const size_t unit_size = GetTypeByte(TypeIdToType(inputs.at(kIndexA)->dtype_id()));
  constexpr size_t pointer_size = sizeof(float *);
  size_t b_size = batch_ * m_ * n_ * unit_size;
  workspace_size_list_.clear();
  if (n_ != 1) {
    workspace_size_list_ = {
      // workspace for batched a
      batch_ * pointer_size,
      // workspace for batched b
      batch_ * pointer_size,
      // workspace for transposed b
      b_size,
      // workspace for b transpose shape
      kShape3D * sizeof(size_t *),
      // workspace for b transpose axis
      kShape3D * sizeof(size_t *),
    };
  } else {
    workspace_size_list_ = {
      // workspace for batched a
      batch_ * pointer_size,
      // workspace for batched b
      batch_ * pointer_size,
    };
  }

  return KRET_OK;
}

template <typename T>
bool SolveTriangularGpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                               const std::vector<KernelTensor *> &workspace,
                                               const std::vector<KernelTensor *> &outputs) {
  CHECK_CUBLAS_RET_WITH_ERROR(cublasSetStream(blas_handle_, cuda_stream_), "cublasSetStream failed");
  auto inputa_addr = GetDeviceAddress<T>(inputs, kIndexA);
  auto inputb_addr = GetDeviceAddress<T>(inputs, kIndexB);
  auto output_addr = GetDeviceAddress<T>(outputs, kIndexX);

  std::vector<T *> host_a_array(batch_);
  std::vector<T *> host_dst_array(batch_);

  // if b is not a vector, solve b in the workspace
  T *dst = nullptr;
  if (n_ == 1) {
    dst = output_addr;
  } else {
    dst = GetDeviceAddress<T>(workspace, kIndexBBuffer);
  }

  const size_t batched_b_size = batch_ * m_ * n_;
  if (n_ == 1) {
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
      cudaMemcpyAsync(dst, inputb_addr, batched_b_size * sizeof(T), cudaMemcpyDeviceToDevice, cuda_stream_),
      "cudaMemcpyAsync dst failed");
  } else {
    // No matter how many batch dimensions the batched matrix b has, use their cumulative multiplication batch.
    // In order to convert row major matrix b(batch, m, n) to col major matrix b'(batch, m, n),
    // the following operation is equivalent to:
    // b' = b.tarnspose(batch, n, m).reshape(batch, m, n)
    auto dev_transpose_b_shape = GetDeviceAddress<size_t>(workspace, kIndexBTransposeShape);
    auto dev_transpose_b_axis = GetDeviceAddress<size_t>(workspace, kIndexBTransposeAxis);
    MatrixTransposeND(inputb_addr, {batch_, m_, n_}, {kDim0, kDim2, kDim1}, dev_transpose_b_shape, dev_transpose_b_axis,
                      dst, cuda_stream_, kernel_name_);
  }

  // index calculation
  auto device_a_array_addr = GetDeviceAddress<T *>(workspace, kIndexAArray);
  auto device_dst_array_addr = GetDeviceAddress<T *>(workspace, kIndexDstArray);
  for (size_t i = 0; i < batch_; i++) {
    host_a_array[i] = inputa_addr + i * m_ * m_;
    host_dst_array[i] = dst + i * m_ * n_;
  }

  CHECK_CUDA_RET_WITH_ERROR_NOTRACE(cudaMemcpyAsync(device_a_array_addr, host_a_array.data(), sizeof(T *) * batch_,
                                                    cudaMemcpyHostToDevice, cuda_stream_),
                                    "cuda memcopy Fail");
  CHECK_CUDA_RET_WITH_ERROR_NOTRACE(cudaMemcpyAsync(device_dst_array_addr, host_dst_array.data(), sizeof(T *) * batch_,
                                                    cudaMemcpyHostToDevice, cuda_stream_),
                                    "cuda memcopy Fail");

  T alpha = 1;
  if constexpr (std::is_same_v<T, float>) {
    CHECK_CUBLAS_RET_WITH_EXCEPT_NOTRACE(
      cublasStrsmBatched(blas_handle_, CUBLAS_SIDE_LEFT, uplo_, trans_, unit_diagonal_, m_, n_, &alpha,
                         device_a_array_addr, lda_, device_dst_array_addr, ldb_, batch_),
      "cublas trsm Fail");
  } else {
    CHECK_CUBLAS_RET_WITH_EXCEPT_NOTRACE(
      cublasDtrsmBatched(blas_handle_, CUBLAS_SIDE_LEFT, uplo_, trans_, unit_diagonal_, m_, n_, &alpha,
                         device_a_array_addr, lda_, device_dst_array_addr, ldb_, batch_),
      "cublas trsm Fail");
  }

  // if x is not a vector, do transpose
  if (n_ != 1) {
    // in order to convert col major matrix x'(m x n) to row major matrix x'(m x n),
    // the following operation is equivalent to:
    // x = x'.reshape(n, m).T
    auto dev_transpose_b_shape = GetDeviceAddress<size_t>(workspace, kIndexBTransposeShape);
    auto dev_transpose_b_axis = GetDeviceAddress<size_t>(workspace, kIndexBTransposeAxis);
    MatrixTransposeND(dst, {batch_, n_, m_}, {kDim0, kDim2, kDim1}, dev_transpose_b_shape, dev_transpose_b_axis,
                      output_addr, cuda_stream_, kernel_name_);
  }
  return true;
}

const std::vector<std::pair<KernelAttr, KernelRunFunc>> &SolveTriangularGpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, KernelRunFunc>> func_list = {
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
       .AddOutputAttr(kNumberTypeFloat32),
     &SolveTriangularGpuKernelMod::LaunchKernel<float>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
       .AddOutputAttr(kNumberTypeFloat64),
     &SolveTriangularGpuKernelMod::LaunchKernel<double>},
  };
  return func_list;
}
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, SolveTriangular, SolveTriangularGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
