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

#include "kernel/gpu/cublas/lu_solve_gpu_kernel.h"
#include <vector>
#include "mindspore/ops/infer/lu_solve_.h"
#include "kernel/gpu/cuda_impl/cuda_ops/matrix_transpose_impl.cuh"
#include "kernel/gpu/cuda_impl/cuda_ops/broadcast_to_impl.cuh"
#include "kernel/gpu/cuda/math/broadcast_public.h"
#include "kernel/gpu/cuda_impl/cuda_ops/complex.h"

namespace mindspore {
namespace kernel {
namespace {
using KernelRunFunc = LuSolveGpuKernelMod::KernelRunFunc;
constexpr int MAX_DIMS = 8;
inline cublasStatus_t cublasXgetrsBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int k,
                                          const float *const a_array[], const int *pivot_array, float *const b_array[],
                                          int *info, int batch_size) {
  return cublasSgetrsBatched(handle, trans, m, k, a_array, m, pivot_array, b_array, m, info, batch_size);
}
inline cublasStatus_t cublasXgetrsBatched(cublasHandle_t handle, cublasOperation_t trans, int m, int k,
                                          const double *const a_array[], const int *pivot_array,
                                          double *const b_array[], int *info, int batch_size) {
  return cublasDgetrsBatched(handle, trans, m, k, a_array, m, pivot_array, b_array, m, info, batch_size);
}
}  // namespace
bool LuSolveGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  blas_handle_ = device::gpu::GPUDeviceManager::GetInstance().GetCublasHandle();

  if (!MatchKernelFunc(kernel_name_, inputs, outputs)) {
    return false;
  }

  return true;
}

int LuSolveGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  if (auto ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }

  const auto b_shape = inputs.at(kIndex0)->GetShapeVector();
  const auto a_shape = inputs.at(kIndex1)->GetShapeVector();
  auto out_shape = outputs.at(kIndex0)->GetShapeVector();
  a_shape_len_ = a_shape.size();
  b_shape_len_ = b_shape.size();
  out_shape_len_ = out_shape.size();
  is_null_input_ = CHECK_SHAPE_NULL(LongVecToSizeVec(a_shape), kernel_name_, "a") ||
                   CHECK_SHAPE_NULL(LongVecToSizeVec(b_shape), kernel_name_, "b");
  need_broadcast_ = false;
  if (a_shape.size() != b_shape.size()) {
    need_broadcast_ = true;
  }
  for (size_t i = 0; i < a_shape.size() - kIndex2; i++) {
    if (a_shape[i] != b_shape[i]) {
      need_broadcast_ = true;
    }
  }

  lhs_shape_.resize(MAX_DIMS, 1);
  rhs_shape_.resize(MAX_DIMS, 1);
  output_shape_.resize(MAX_DIMS, 1);

  if (need_broadcast_) {
    for (size_t i = 0; i < out_shape.size(); i++) {
      output_shape_[i] = out_shape[i];
    }
    size_t lhs_offset = out_shape.size() - a_shape.size();
    size_t rhs_offset = out_shape.size() - b_shape.size();
    for (size_t j = 0; j < a_shape.size(); j++) {
      if ((j + lhs_offset) < MAX_DIMS) {
        lhs_shape_[j + lhs_offset] = a_shape[j];
      }
    }
    for (size_t k = 0; k < b_shape.size(); k++) {
      if ((k + rhs_offset) < MAX_DIMS) {
        rhs_shape_[k + rhs_offset] = b_shape[k];
      }
    }
  }

  batch_num_a_ = std::accumulate(a_shape.begin(), a_shape.end() - kIndex2, int64_t(1), std::multiplies{});
  batch_num_b_ = std::accumulate(b_shape.begin(), b_shape.end() - kIndex2, int64_t(1), std::multiplies{});
  batch_num_out_ = std::accumulate(out_shape.begin(), out_shape.end() - kIndex2, int64_t(1), std::multiplies{});
  m_ = a_shape.back();
  k_ = b_shape.back();

  const size_t a_size = LongToSize(std::accumulate(a_shape.begin(), a_shape.end(), int64_t(1), std::multiplies{}));
  const size_t b_size = LongToSize(std::accumulate(b_shape.begin(), b_shape.end(), int64_t(1), std::multiplies{}));
  const size_t type_size = GetTypeByte(TypeIdToType(inputs.at(kIndex0)->dtype_id()));

  workspace_size_list_.clear();
  workspace_size_list_ = {
    a_size * type_size,                // a column major
    b_size * type_size,                // b column major
    batch_num_out_ * sizeof(float *),  // a_device_array, the size of float* and double* are the same
    batch_num_out_ * sizeof(float *),  // b_device_array, the size of float* and double* are the same
    1,                                 // a_broadcast
    1,                                 // b_broadcast
    1,                                 // pivoting_broadcast sequence
  };
  if (need_broadcast_) {
    workspace_size_list_[kIndex4] = batch_num_out_ * m_ * m_ * type_size;
    workspace_size_list_[kIndex5] = batch_num_out_ * m_ * k_ * type_size;
    workspace_size_list_[kIndex6] = batch_num_out_ * m_ * sizeof(int);
  }
  return KRET_OK;
}

template <typename T>
bool LuSolveGpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                       const std::vector<KernelTensor *> &workspace,
                                       const std::vector<KernelTensor *> &outputs) {
  T *b = GetDeviceAddress<T>(inputs, kIndex0);
  T *a = GetDeviceAddress<T>(inputs, kIndex1);
  int *piv_array = GetDeviceAddress<int>(inputs, kIndex2);
  MS_EXCEPTION_IF_NULL(b);
  MS_EXCEPTION_IF_NULL(a);
  MS_EXCEPTION_IF_NULL(piv_array);

  auto a_col_major = GetDeviceAddress<T>(workspace, kIndex0);
  auto b_col_major = GetDeviceAddress<T>(workspace, kIndex1);
  auto a_device_array = GetDeviceAddress<T *>(workspace, kIndex2);
  auto b_device_array = GetDeviceAddress<T *>(workspace, kIndex3);
  auto a_broadcast = GetDeviceAddress<T>(workspace, kIndex4);
  auto b_broadcast = GetDeviceAddress<T>(workspace, kIndex5);
  auto piv_broadcast = GetDeviceAddress<int>(workspace, kIndex6);
  MS_EXCEPTION_IF_NULL(a_col_major);
  MS_EXCEPTION_IF_NULL(b_col_major);
  MS_EXCEPTION_IF_NULL(a_device_array);
  MS_EXCEPTION_IF_NULL(b_device_array);
  MS_EXCEPTION_IF_NULL(a_broadcast);
  MS_EXCEPTION_IF_NULL(b_broadcast);
  MS_EXCEPTION_IF_NULL(piv_broadcast);

  T *output = GetDeviceAddress<T>(outputs, kIndex0);
  MS_EXCEPTION_IF_NULL(output);

  CHECK_CUBLAS_RET_WITH_EXCEPT_NOTRACE(cublasSetStream(blas_handle_, cuda_stream_),
                                       "For LuSolveGpuKernelMod cublasSetStream Fail");
  cudaError_t status = cudaErrorNotReady;
  status = MatrixTranspose(a, LongToSize(batch_num_a_ * m_ * m_), SizeToInt(m_), SizeToInt(m_), a_col_major, device_id_,
                           cuda_stream_);
  CHECK_CUDA_STATUS(status, kernel_name_);
  status = MatrixTranspose(b, LongToSize(batch_num_b_ * m_ * k_), SizeToInt(m_), SizeToInt(k_), b_col_major, device_id_,
                           cuda_stream_);
  CHECK_CUDA_STATUS(status, kernel_name_);
  if (need_broadcast_) {
    std::vector<int64_t> simplified_inp_shape;
    std::vector<int64_t> simplified_out_shape;
    // expand_size :(*,m,m)
    auto origin_size = lhs_shape_;
    auto expand_size = output_shape_;
    expand_size[out_shape_len_ - kIndex1] = m_;
    SimplifyBroadcastToShape(origin_size, expand_size, &simplified_inp_shape, &simplified_out_shape);
    status =
      BroadcastTo(simplified_inp_shape, simplified_out_shape, a_col_major, a_broadcast, device_id_, cuda_stream_);
    CHECK_CUDA_STATUS(status, kernel_name_);

    // expand_size :(*,k,m)
    origin_size = rhs_shape_;
    expand_size = output_shape_;
    std::swap(origin_size[out_shape_len_ - kIndex1], origin_size[out_shape_len_ - kIndex2]);
    std::swap(expand_size[out_shape_len_ - kIndex1], expand_size[out_shape_len_ - kIndex2]);
    SimplifyBroadcastToShape(origin_size, expand_size, &simplified_inp_shape, &simplified_out_shape);
    status =
      BroadcastTo(simplified_inp_shape, simplified_out_shape, b_col_major, b_broadcast, device_id_, cuda_stream_);
    CHECK_CUDA_STATUS(status, kernel_name_);

    // origin_size:(*,m,1)
    // expand_size :(*,m,1)
    origin_size = lhs_shape_;
    origin_size[out_shape_len_ - kIndex1] = 1;
    expand_size = output_shape_;
    expand_size[out_shape_len_ - kIndex1] = 1;
    SimplifyBroadcastToShape(origin_size, expand_size, &simplified_inp_shape, &simplified_out_shape);
    status =
      BroadcastTo(simplified_inp_shape, simplified_out_shape, piv_array, piv_broadcast, device_id_, cuda_stream_);
    CHECK_CUDA_STATUS(status, kernel_name_);
  } else {
    a_broadcast = a_col_major;
    b_broadcast = b_col_major;
    piv_broadcast = piv_array;
  }
  std::vector<T *> a_host_array(batch_num_out_);
  std::vector<T *> b_host_array(batch_num_out_);
  for (int64_t i = 0; i < batch_num_out_; i++) {
    a_host_array[i] = a_broadcast + i * m_ * m_;
    b_host_array[i] = b_broadcast + i * m_ * k_;
  }
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaMemcpyAsync(a_device_array, a_host_array.data(), batch_num_out_ * sizeof(T *),
                                                     cudaMemcpyHostToDevice, cuda_stream_),
                                     "For 'LuSolveGpuKernelMod', it launch memcopy failed.");
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaMemcpyAsync(b_device_array, b_host_array.data(), batch_num_out_ * sizeof(T *),
                                                     cudaMemcpyHostToDevice, cuda_stream_),
                                     "For 'LuSolveGpuKernelMod', it launch memcopy failed.");
  int info = 0;
  CHECK_CUBLAS_RET_WITH_EXCEPT_NOTRACE(cublasXgetrsBatched(blas_handle_, CUBLAS_OP_N, m_, k_, a_device_array,
                                                           piv_broadcast, b_device_array, &info, batch_num_out_),
                                       "For 'LuSolveGpuKernelMod', it launch cublasXgetrfBatched failed");

  status = MatrixTranspose(b_broadcast, LongToSize(batch_num_out_ * m_ * k_), SizeToInt(k_), SizeToInt(m_), output,
                           device_id_, cuda_stream_);
  CHECK_CUDA_STATUS(status, kernel_name_);
  return true;
}

const std::vector<std::pair<KernelAttr, KernelRunFunc>> &LuSolveGpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, KernelRunFunc>> func_list = {
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeFloat32),
     &LuSolveGpuKernelMod::LaunchKernel<float>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeFloat64),
     &LuSolveGpuKernelMod::LaunchKernel<double>},
  };
  return func_list;
}
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, LuSolve, LuSolveGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
