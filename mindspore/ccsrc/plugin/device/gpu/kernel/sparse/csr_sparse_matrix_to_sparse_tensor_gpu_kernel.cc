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

#include "plugin/device/gpu/kernel/sparse/csr_sparse_matrix_to_sparse_tensor_gpu_kernel.h"
#include <algorithm>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/complex.h"

namespace mindspore {
namespace kernel {
bool CSRSparseMatrixToSparseTensorGpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                                     const std::vector<KernelTensorPtr> &inputs,
                                                     const std::vector<KernelTensorPtr> &outputs) {
  auto kernel_ptr = std::dynamic_pointer_cast<ops::CSRSparseMatrixToSparseTensor>(base_operator);
  if (!kernel_ptr) {
    MS_LOG(ERROR) << "cast CSRSparseMatrixToSparseTensor ops failed!";
    return false;
  }
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kCSRSparseMatrixToSparseTensorInputsNum, kernel_ptr->name());
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kCSRSparseMatrixToSparseTensorOutputsNum, kernel_ptr->name());

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(EXCEPTION) << "For '" << kernel_ptr->name()
                      << "', it does not support this kernel data type: " << kernel_attr;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

void CSRSparseMatrixToSparseTensorGpuKernelMod::ResetResource() noexcept {
  is_null_input_ = false;
  input_dense_shape_size_ = 0;
  input_batch_pointers_size_ = 0;
  input_row_pointers_size_ = 0;
  input_col_indices_size_ = 0;
  input_values_size_ = 0;
  output_indices_size_ = 0;
  output_values_size_ = 0;
  output_dense_shape_size_ = 0;
  input_size_list_.clear();
  workspace_size_list_.clear();
  output_size_list_.clear();
}

int CSRSparseMatrixToSparseTensorGpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                                      const std::vector<KernelTensorPtr> &inputs,
                                                      const std::vector<KernelTensorPtr> &outputs,
                                                      const std::map<uint32_t, tensor::TensorPtr> &others) {
  ResetResource();
  input_dense_shape_shapes_ = inputs[kIndex0]->GetShapeVector();
  input_batch_pointers_shapes_ = inputs[kIndex1]->GetShapeVector();
  input_row_pointers_shapes_ = inputs[kIndex2]->GetShapeVector();
  input_col_indices_shapes_ = inputs[kIndex3]->GetShapeVector();
  input_values_shapes_ = inputs[kIndex4]->GetShapeVector();
  output_indices_shapes_ = outputs[kIndex0]->GetShapeVector();
  output_values_shapes_ = outputs[kIndex1]->GetShapeVector();
  output_dense_shape_shapes_ = outputs[kIndex2]->GetShapeVector();
  if (!(CHECK_SHAPE_POSITIVE(input_dense_shape_shapes_) && CHECK_SHAPE_POSITIVE(input_batch_pointers_shapes_) &&
        CHECK_SHAPE_POSITIVE(input_row_pointers_shapes_) && CHECK_SHAPE_POSITIVE(input_col_indices_shapes_) &&
        CHECK_SHAPE_POSITIVE(input_values_shapes_) && CHECK_SHAPE_POSITIVE(output_indices_shapes_) &&
        CHECK_SHAPE_POSITIVE(output_values_shapes_) && CHECK_SHAPE_POSITIVE(output_dense_shape_shapes_))) {
    is_null_input_ = true;
    InitSizeLists();
    return 0;
  }

  MS_EXCEPTION_IF_CHECK_FAIL(!input_dense_shape_shapes_.empty(), "input_dense_shape_ should not be empty!");
  MS_EXCEPTION_IF_CHECK_FAIL(!input_batch_pointers_shapes_.empty(), "input_batch_pointers_ should not be empty!");
  MS_EXCEPTION_IF_CHECK_FAIL(!input_row_pointers_shapes_.empty(), "input_row_pointers_ should not be empty!");
  MS_EXCEPTION_IF_CHECK_FAIL(!output_dense_shape_shapes_.empty(), "output_dense_shapes_ should not be empty!");
  rank_ = input_dense_shape_shapes_[kIndex0];
  is_batch_csr_ = (rank_ == kBatchCSR) ? true : false;

  auto GetNums = [](const std::vector<int64_t> &shape) {
    size_t res = 1;
    for (const auto &sh : shape) {
      res *= LongToSize(sh);
    }
    return res;
  };
  input_dense_shape_size_ = abstract::TypeIdSize(inputs[kIndex0]->GetDtype()) * GetNums(input_dense_shape_shapes_);
  input_batch_pointers_size_ =
    abstract::TypeIdSize(inputs[kIndex1]->GetDtype()) * GetNums(input_batch_pointers_shapes_);
  input_row_pointers_size_ = abstract::TypeIdSize(inputs[kIndex2]->GetDtype()) * GetNums(input_row_pointers_shapes_);
  input_col_indices_size_ = abstract::TypeIdSize(inputs[kIndex3]->GetDtype()) * GetNums(input_col_indices_shapes_);
  input_values_size_ = abstract::TypeIdSize(inputs[kIndex4]->GetDtype()) * GetNums(input_values_shapes_);
  output_indices_size_ = abstract::TypeIdSize(outputs[kIndex0]->GetDtype()) * GetNums(output_indices_shapes_);
  output_values_size_ = abstract::TypeIdSize(outputs[kIndex1]->GetDtype()) * GetNums(output_values_shapes_);
  output_dense_shape_size_ = abstract::TypeIdSize(outputs[kIndex2]->GetDtype()) * GetNums(output_dense_shape_shapes_);
  InitSizeLists();
  return 0;
}

void CSRSparseMatrixToSparseTensorGpuKernelMod::InitSizeLists() {
  input_size_list_.push_back(input_dense_shape_size_);
  input_size_list_.push_back(input_batch_pointers_size_);
  input_size_list_.push_back(input_row_pointers_size_);
  input_size_list_.push_back(input_col_indices_size_);
  input_size_list_.push_back(input_values_size_);
  workspace_size_list_.push_back(input_col_indices_size_);
  output_size_list_.push_back(output_indices_size_);
  output_size_list_.push_back(output_values_size_);
  output_size_list_.push_back(output_dense_shape_size_);
}

template <typename T, typename S>
bool CSRSparseMatrixToSparseTensorGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                                             const std::vector<AddressPtr> &workspace,
                                                             const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  if (is_null_input_) {
    return true;
  }
  S *csr_dense_shape_addr = GetDeviceAddress<S>(inputs, kIndex0);
  S *csr_batch_pointers_addr = GetDeviceAddress<S>(inputs, kIndex1);
  S *csr_row_pointers_addr = GetDeviceAddress<S>(inputs, kIndex2);
  S *csr_col_indices_addr = GetDeviceAddress<S>(inputs, kIndex3);
  T *csr_values_addr = GetDeviceAddress<T>(inputs, kIndex4);
  S *sparse_row_indices_addr = GetDeviceAddress<S>(workspace, kIndex0);
  S *sparse_indices_addr = GetDeviceAddress<S>(outputs, kIndex0);
  T *sparse_values_addr = GetDeviceAddress<T>(outputs, kIndex1);
  S *sparse_dense_shape_addr = GetDeviceAddress<S>(outputs, kIndex2);

  std::vector<S> host_shape_pointers(input_dense_shape_shapes_[kIndex0], 0);
  device::gpu::CudaDriver::CopyDeviceMemToHost(host_shape_pointers.data(), csr_dense_shape_addr, sizeof(S) * rank_);
  size_t num_batches = (is_batch_csr_) ? host_shape_pointers[kIndex0] : 1;
  auto total_nnz = input_col_indices_shapes_[kIndex0];
  auto row_dim = is_batch_csr_ ? kIndex1 : kIndex0;
  auto row_size = host_shape_pointers[row_dim];

  if (!is_batch_csr_) {
    cusparseXcsr2coo(handle_, csr_row_pointers_addr, total_nnz, row_size, sparse_row_indices_addr,
                     CUSPARSE_INDEX_BASE_ZERO);
    CallStackIndices2D<S>(sparse_row_indices_addr, csr_col_indices_addr, sparse_indices_addr, total_nnz,
                          reinterpret_cast<cudaStream_t>(stream_ptr));
  } else {
    std::vector<S> host_batch_pointers(input_batch_pointers_shapes_[kIndex0], 0);
    device::gpu::CudaDriver::CopyDeviceMemToHost(host_batch_pointers.data(), csr_batch_pointers_addr,
                                                 sizeof(S) * (num_batches + 1));
    int accum_nnz = 0;
    for (size_t i = 0; i < num_batches; ++i) {
      S *row_ind_ptr = csr_row_pointers_addr + i * (row_size + 1);
      S nnz = host_batch_pointers[i + 1] - host_batch_pointers[i];
      if (nnz != 0) {
        cusparseXcsr2coo(handle_, row_ind_ptr, nnz, row_size, sparse_row_indices_addr + accum_nnz,
                         CUSPARSE_INDEX_BASE_ZERO);
      }
      accum_nnz += nnz;
    }
    if (accum_nnz > 0) {
      size_t shared_memory_size = sizeof(S) * (num_batches + 1);
      CallStackIndices3D<S>(csr_batch_pointers_addr, sparse_row_indices_addr, csr_col_indices_addr, sparse_indices_addr,
                            num_batches, total_nnz, shared_memory_size, reinterpret_cast<cudaStream_t>(stream_ptr));
    }
  }
  device::gpu::CudaDriver::CopyDeviceMemToDeviceAsync(sparse_values_addr, csr_values_addr, sizeof(T) * total_nnz,
                                                      reinterpret_cast<cudaStream_t>(stream_ptr));
  device::gpu::CudaDriver::CopyDeviceMemToDeviceAsync(sparse_dense_shape_addr, csr_dense_shape_addr, sizeof(S) * rank_,
                                                      reinterpret_cast<cudaStream_t>(stream_ptr));
  return true;
}

std::vector<KernelAttr> CSRSparseMatrixToSparseTensorGpuKernelMod::GetOpSupport() {
  static std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, CSRSparseMatrixToSparseTensorFunc> &pair) { return pair.first; });
  return support_list;
}

template <typename T>
using Complex = mindspore::utils::Complex<T>;

std::vector<std::pair<KernelAttr, CSRSparseMatrixToSparseTensorGpuKernelMod::CSRSparseMatrixToSparseTensorFunc>>
  CSRSparseMatrixToSparseTensorGpuKernelMod::func_list_ = {
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeInt32),
     &CSRSparseMatrixToSparseTensorGpuKernelMod::LaunchKernel<float, int>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeFloat64)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeFloat64)
       .AddOutputAttr(kNumberTypeInt32),
     &CSRSparseMatrixToSparseTensorGpuKernelMod::LaunchKernel<double, int>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeFloat16)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeFloat16)
       .AddOutputAttr(kNumberTypeInt32),
     &CSRSparseMatrixToSparseTensorGpuKernelMod::LaunchKernel<half, int>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeComplex64)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeComplex64)
       .AddOutputAttr(kNumberTypeInt32),
     &CSRSparseMatrixToSparseTensorGpuKernelMod::LaunchKernel<Complex<float>, int>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeComplex128)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeComplex128)
       .AddOutputAttr(kNumberTypeInt32),
     &CSRSparseMatrixToSparseTensorGpuKernelMod::LaunchKernel<Complex<double>, int>},
};

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, CSRSparseMatrixToSparseTensor, CSRSparseMatrixToSparseTensorGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
