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
#include "kernel/gpu/cuda/sparse/sparse_matrix_mul_gpu_kernel.h"
#include <algorithm>
#include <map>
#include <set>
#include <utility>
#include "include/utils/thread_pool.h"
#include "mindspore/ops/infer/sparse_matrix_mul.h"
#include "kernel/gpu/cuda_impl/cuda_ops/sparse_matrix_mul.cuh"

namespace mindspore {
namespace kernel {
namespace {
constexpr int kSparseMatrixMulAddIndex0 = 0;
constexpr int kSparseMatrixMulAddIndex1 = 1;
constexpr int kSparseMatrixMulAddIndex2 = 2;
constexpr int kSparseMatrixMulAddIndex3 = 3;
}  // namespace
bool SparseMatrixMulGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                       const std::vector<KernelTensor *> &outputs) {
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << kernel_name_ << " does not support this kernel data type: " << kernel_attr << ".";
    return false;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int SparseMatrixMulGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                        const std::vector<KernelTensor *> &outputs) {
  int ret = KRET_OK;
  if ((ret = KernelMod::Resize(inputs, outputs)) != 0) {
    MS_LOG(ERROR) << kernel_name_ << " reinit failed.";
    return ret;
  }
  std::vector<int64_t> input_shape_2 = inputs[kSparseMatrixMulAddIndex2]->GetShapeVector();
  std::vector<int64_t> input_shape_3 = inputs[kSparseMatrixMulAddIndex3]->GetShapeVector();
  std::vector<int64_t> input_shape_0 = inputs[kSparseMatrixMulAddIndex0]->GetShapeVector();
  std::vector<int64_t> input_shape_1 = inputs[kSparseMatrixMulAddIndex1]->GetShapeVector();
  row_ = input_shape_2[0];
  col_ = input_shape_3[0];
  dense_size_ = input_shape_0[0];
  batch_size_ = input_shape_1[0];
  return ret;
}

template <typename T, typename S>
bool SparseMatrixMulGpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                               const std::vector<KernelTensor *> &,
                                               const std::vector<KernelTensor *> &outputs) {
  T *a_shape_addr = GetDeviceAddress<T>(inputs, kIndex0);
  T *a_batch_pointers_addr = GetDeviceAddress<T>(inputs, kIndex1);
  T *a_indptr_addr = GetDeviceAddress<T>(inputs, kIndex2);
  T *a_indices_addr = GetDeviceAddress<T>(inputs, kIndex3);
  S *a_values_addr = GetDeviceAddress<S>(inputs, kIndex4);
  S *b_dense_addr = GetDeviceAddress<S>(inputs, kIndex5);

  T *c_shape_addr = GetDeviceAddress<T>(outputs, kIndex0);
  T *c_batch_pointers_addr = GetDeviceAddress<T>(outputs, kIndex1);
  T *c_indptr_addr = GetDeviceAddress<T>(outputs, kIndex2);
  T *c_indices_addr = GetDeviceAddress<T>(outputs, kIndex3);
  S *c_values_addr = GetDeviceAddress<S>(outputs, kIndex4);

  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(c_shape_addr, a_shape_addr, dense_size_ * sizeof(T), cudaMemcpyDeviceToDevice,
                    reinterpret_cast<cudaStream_t>(cuda_stream_)),
    "Cuda memcopy input to output Fail");

  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(c_batch_pointers_addr, a_batch_pointers_addr, batch_size_ * sizeof(T), cudaMemcpyDeviceToDevice,
                    reinterpret_cast<cudaStream_t>(cuda_stream_)),
    "Cuda memcopy input to output Fail");

  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(c_indptr_addr, a_indptr_addr, row_ * sizeof(T), cudaMemcpyDeviceToDevice,
                    reinterpret_cast<cudaStream_t>(cuda_stream_)),
    "Cuda memcopy input to output Fail");

  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(c_indices_addr, a_indices_addr, col_ * sizeof(T), cudaMemcpyDeviceToDevice,
                    reinterpret_cast<cudaStream_t>(cuda_stream_)),
    "Cuda memcopy input to output Fail");

  auto status = CalSparseMatrixMul(a_shape_addr, a_batch_pointers_addr, a_indptr_addr, a_indices_addr, a_values_addr,
                                   b_dense_addr, c_shape_addr, c_batch_pointers_addr, c_indptr_addr, c_indices_addr,
                                   c_values_addr, row_, col_, device_id_, reinterpret_cast<cudaStream_t>(cuda_stream_));
  CHECK_CUDA_STATUS(status, kernel_name_);
  bool is_nullptr = (a_shape_addr == nullptr) || (a_batch_pointers_addr == nullptr) || (a_indptr_addr == nullptr) ||
                    (a_indices_addr == nullptr) || (a_values_addr == nullptr) || (b_dense_addr == nullptr) ||
                    (c_shape_addr == nullptr) || (c_batch_pointers_addr == nullptr) || (c_indptr_addr == nullptr) ||
                    (c_indices_addr == nullptr) || (c_values_addr == nullptr);
  if (is_nullptr) {
    return false;
  }
  return true;
}

std::vector<std::pair<KernelAttr, SparseMatrixMulGpuKernelMod::SparseMatrixMulLaunchFunc>>
  SparseMatrixMulGpuKernelMod::func_list_ = {
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeFloat32),
     &SparseMatrixMulGpuKernelMod::LaunchKernel<int64_t, float>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeFloat64)
       .AddOutputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeFloat64),
     &SparseMatrixMulGpuKernelMod::LaunchKernel<int64_t, double>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeFloat32),
     &SparseMatrixMulGpuKernelMod::LaunchKernel<int32_t, float>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeFloat64)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeFloat64),
     &SparseMatrixMulGpuKernelMod::LaunchKernel<int32_t, double>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt64),
     &SparseMatrixMulGpuKernelMod::LaunchKernel<int64_t, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt32),
     &SparseMatrixMulGpuKernelMod::LaunchKernel<int64_t, int32_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt64),
     &SparseMatrixMulGpuKernelMod::LaunchKernel<int32_t, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32),
     &SparseMatrixMulGpuKernelMod::LaunchKernel<int32_t, int32_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt16)
       .AddInputAttr(kNumberTypeInt16)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt16),
     &SparseMatrixMulGpuKernelMod::LaunchKernel<int32_t, int16_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt16)
       .AddInputAttr(kNumberTypeInt16)
       .AddOutputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt16),
     &SparseMatrixMulGpuKernelMod::LaunchKernel<int64_t, int16_t>},
};

std::vector<KernelAttr> SparseMatrixMulGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, SparseMatrixMulGpuKernelMod::SparseMatrixMulLaunchFunc> &pair) {
                         return pair.first;
                       });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, SparseMatrixMul, SparseMatrixMulGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
