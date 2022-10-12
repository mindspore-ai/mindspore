/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/gpu/kernel/math/solve_triangular_gpu_kernel.h"
#include <map>
#include <utility>
#include <memory>
#include "mindspore/core/ops/solve_triangular.h"

namespace mindspore {
namespace kernel {
using KernelRunFunc = SolveTriangularGpuKernelMod::KernelRunFunc;
bool SolveTriangularGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                       const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();
  blas_handle_ = device::gpu::GPUDeviceManager::GetInstance().GetCublasHandle();

  auto kernel_ptr = std::make_shared<ops::SolveTriangular>(base_operator->GetPrim());

  bool lower = kernel_ptr->get_lower();
  // reverting the trans flag by default, so also flip the lower flag
  lower = !lower;
  uplo_ = lower ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;

  bool unit_diagonal = kernel_ptr->get_unit_diagonal();
  unit_diagonal_ = unit_diagonal ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT;

  const std::string trans = kernel_ptr->get_trans();
  if (trans == "N") {
    trans_ = CUBLAS_OP_T;
  } else if (trans == "T") {
    trans_ = CUBLAS_OP_N;
  } else if (trans == "C") {
    // currently does not support complex.
    trans_ = CUBLAS_OP_N;
  } else {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', 'trans' must be in ['N', 'T', 'C'], but got [" << trans << "].";
  }

  if (!MatchKernelFunc(base_operator, inputs, outputs)) {
    return false;
  }

  return true;
}
int SolveTriangularGpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                        const std::vector<KernelTensorPtr> &inputs,
                                        const std::vector<KernelTensorPtr> &outputs,
                                        const std::map<uint32_t, tensor::TensorPtr> &) {
  if (auto ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
    return ret;
  }

  auto a_shape = LongVecToSizeVec(inputs.at(kIndex0)->GetShapeVector());
  auto b_shape = LongVecToSizeVec(inputs.at(kIndex1)->GetShapeVector());

  is_null_input_ =
    CHECK_SHAPE_NULL(a_shape, kernel_name_, "input_a") || CHECK_SHAPE_NULL(b_shape, kernel_name_, "input_b");
  // Since the shape check is done in frontend, we can suppose that the shape of a, b here is valid.
  size_t a_dims = a_shape.size();
  size_t b_dims = b_shape.size();
  m_ = a_shape[a_dims - kIndex2];
  n_ = (b_dims == a_dims - 1) ? 1 : b_shape[b_dims - 1];
  batch_ = std::accumulate(a_shape.begin(), a_shape.end() - kIndex2, int64_t(1), std::multiplies{});

  lda_ = SizeToInt(m_);
  ldb_ = SizeToInt(m_);

  const size_t unit_size = GetTypeByte(TypeIdToType(inputs.at(kIndex0)->GetDtype()));
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
bool SolveTriangularGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                               const std::vector<AddressPtr> &workspace,
                                               const std::vector<AddressPtr> &outputs) {
  CHECK_CUBLAS_RET_WITH_ERROR(cublasSetStream(blas_handle_, cuda_stream_), "cublasSetStream failed");
  auto inputa_addr = GetDeviceAddress<T>(inputs, 0);
  auto inputb_addr = GetDeviceAddress<T>(inputs, 1);
  auto output_addr = GetDeviceAddress<T>(outputs, 0);

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
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     &SolveTriangularGpuKernelMod::LaunchKernel<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
     &SolveTriangularGpuKernelMod::LaunchKernel<double>},
  };
  return func_list;
}
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, SolveTriangular, SolveTriangularGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
