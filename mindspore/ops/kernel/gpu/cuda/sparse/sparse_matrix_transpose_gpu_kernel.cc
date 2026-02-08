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

#include <algorithm>
#include <functional>
#include <string>
#include <utility>

#include "include/runtime/hardware_abstract/kernel_base/common_utils.h"
#include "mindspore/ops/infer/sparse_matrix_transpose.h"
#include "kernel/gpu/cuda_impl/cuda_ops/complex.h"
#include "kernel/gpu/cuda_impl/cuda_ops/sparse_matrix_transpose_impl.cuh"
#include "kernel/gpu/cuda/sparse/sparse_matrix_transpose_gpu_kernel.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr int kSparseMatrixTransposeInputsNum = 5;
constexpr int kSparseMatrixTransposeOutputsNum = 5;
constexpr int kBatchCSR = 3;
#define ToVoid(x) reinterpret_cast<void *>(x)

std::map<TypeId, cudaDataType> get_cuda_type{{kNumberTypeFloat32, CUDA_R_32F},
                                             {kNumberTypeFloat64, CUDA_R_64F},
                                             {kNumberTypeComplex64, CUDA_C_32F},
                                             {kNumberTypeComplex128, CUDA_C_64F}};

void CuSparseGetBufferSize(TypeId ms_type, cusparseHandle_t handle, int m, int n, int nnz, int *x_row_ptrs,
                           int *x_col_inds, int *y_row_ptrs, int *y_col_inds, size_t *buffer_size) {
  CHECK_CUSPARSE_RET_WITH_EXCEPT(
    cusparseCsr2cscEx2_bufferSize(handle, m, n, nnz, nullptr, x_row_ptrs, x_col_inds, nullptr, y_row_ptrs, y_col_inds,
                                  get_cuda_type[ms_type], CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO,
                                  CUSPARSE_CSR2CSC_ALG2, buffer_size),
    "Failed to allocate buffer");
}

void Csr2Csc(TypeId ms_type, cusparseHandle_t handle, int m, int n, int nnz, void *x_values, int *x_row_ptrs,
             int *x_col_inds, void *y_values, int *y_row_ptrs, int *y_col_inds, void *buffer) {
  CHECK_CUSPARSE_RET_WITH_EXCEPT(
    cusparseCsr2cscEx2(handle, m, n, nnz, x_values, x_row_ptrs, x_col_inds, y_values, y_row_ptrs, y_col_inds,
                       get_cuda_type[ms_type], CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG2,
                       buffer),
    "Failed to call cusparse function.");
}
}  // namespace

bool SparseMatrixTransposeGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                             const std::vector<KernelTensor *> &outputs) {
  conjugate = GetValue<bool>(primitive_->GetAttr("conjugate"));
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kSparseMatrixTransposeInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kSparseMatrixTransposeOutputsNum, kernel_name_);
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

template <class S>
size_t SparseMatrixTransposeGpuKernelMod::GetBufferSize(size_t num_batches, int rows, int cols,
                                                        const std::vector<S> &host_batch_pointers, S *x_row_ptrs,
                                                        S *x_col_inds, S *y_row_ptrs, S *y_col_inds) {
  size_t max_buffer_size = 0;
  for (size_t i = 0; i < num_batches; i++) {
    S offset = host_batch_pointers[i];  // get the single matrix
    S nnz = host_batch_pointers[i + 1] - offset;
    S *x_col_ind = x_col_inds + offset;
    S *y_col_ind = y_col_inds + offset;
    S *x_row_ptr = x_row_ptrs + (rows + 1) * i;
    S *y_row_ptr = y_row_ptrs + (cols + 1) * i;
    size_t buffer_size;
    CuSparseGetBufferSize(ms_type_, handle_, rows, cols, nnz, x_row_ptr, x_col_ind, y_row_ptr, y_col_ind, &buffer_size);
    max_buffer_size = std::max(max_buffer_size, buffer_size);
  }

  return max_buffer_size;
}

int SparseMatrixTransposeGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                              const std::vector<KernelTensor *> &outputs) {
  if (auto ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  input_dense_shape_shapes_ = inputs[kIndex0]->GetShapeVector();
  input_batch_pointers_shapes_ = inputs[kIndex1]->GetShapeVector();
  input_row_pointers_shapes_ = inputs[kIndex2]->GetShapeVector();
  input_col_indices_shapes_ = inputs[kIndex3]->GetShapeVector();
  input_values_shapes_ = inputs[kIndex4]->GetShapeVector();

  output_dense_shape_shapes_ = outputs[kIndex0]->GetShapeVector();
  output_batch_pointers_shapes_ = outputs[kIndex1]->GetShapeVector();
  output_row_pointers_shapes_ = outputs[kIndex2]->GetShapeVector();
  output_col_indices_shapes_ = outputs[kIndex3]->GetShapeVector();
  output_values_shapes_ = outputs[kIndex4]->GetShapeVector();

  if (!(CHECK_SHAPE_POSITIVE(input_dense_shape_shapes_) && CHECK_SHAPE_POSITIVE(input_batch_pointers_shapes_) &&
        CHECK_SHAPE_POSITIVE(input_row_pointers_shapes_) && CHECK_SHAPE_POSITIVE(output_dense_shape_shapes_) &&
        CHECK_SHAPE_POSITIVE(output_batch_pointers_shapes_) && CHECK_SHAPE_POSITIVE(output_row_pointers_shapes_))) {
    is_null_input_ = true;
    return 0;
  }

  if (!(CHECK_SHAPE_POSITIVE(input_col_indices_shapes_) && CHECK_SHAPE_POSITIVE(input_values_shapes_) &&
        CHECK_SHAPE_POSITIVE(output_col_indices_shapes_) && CHECK_SHAPE_POSITIVE(output_values_shapes_))) {
    is_empty_matrix = true;
  }

  MS_EXCEPTION_IF_CHECK_FAIL(!input_dense_shape_shapes_.empty(), "input_dense_shape_ should not be empty!");
  MS_EXCEPTION_IF_CHECK_FAIL(!input_batch_pointers_shapes_.empty(), "input_batch_pointers_ should not be empty!");
  MS_EXCEPTION_IF_CHECK_FAIL(!input_row_pointers_shapes_.empty(), "input_row_pointers_ should not be empty!");

  MS_EXCEPTION_IF_CHECK_FAIL(!output_dense_shape_shapes_.empty(), "output_dense_shape_ should not be empty!");
  MS_EXCEPTION_IF_CHECK_FAIL(!output_batch_pointers_shapes_.empty(), "output_batch_pointers_ should not be empty!");
  MS_EXCEPTION_IF_CHECK_FAIL(!output_row_pointers_shapes_.empty(), "output_row_pointers_ should not be empty!");

  rank_ = input_dense_shape_shapes_[kIndex0];
  batched = (rank_ == kBatchCSR) ? true : false;
  ms_type_ = inputs[kIndex4]->dtype_id();
  output_values_size_ = abstract::TypeIdSize(outputs[kIndex4]->dtype_id()) * SizeOf(output_values_shapes_);
  return 0;
}

template <class S, class T>
bool SparseMatrixTransposeGpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                                     const std::vector<KernelTensor *> &workspace,
                                                     const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  if (is_null_input_) {
    return true;
  }
  MS_EXCEPTION_IF_NULL(stream_ptr);
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kSparseMatrixTransposeInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kSparseMatrixTransposeOutputsNum, kernel_name_);
  S *csr_dense_shape_addr = GetDeviceAddress<S>(inputs, kIndex0);
  S *csr_batch_pointers_addr = GetDeviceAddress<S>(inputs, kIndex1);
  S *csr_row_pointers_addr = GetDeviceAddress<S>(inputs, kIndex2);
  S *csc_dense_shape_addr = GetDeviceAddress<S>(outputs, kIndex0);
  S *csc_batch_pointers_addr = GetDeviceAddress<S>(outputs, kIndex1);
  S *csc_col_pointers_addr = GetDeviceAddress<S>(outputs, kIndex2);
  std::vector<S> host_shape_pointers(input_dense_shape_shapes_[kIndex0], 0);
  device::gpu::CudaDriver::CopyDeviceMemToHost(host_shape_pointers.data(), csr_dense_shape_addr, sizeof(S) * rank_);
  size_t num_batches = (batched) ? host_shape_pointers[kIndex0] : 1;
  auto row_dim = batched ? kIndex1 : kIndex0;
  auto row_size = host_shape_pointers[row_dim];
  auto col_size = host_shape_pointers[row_dim + 1];

  if (is_empty_matrix) {
    // copy batch ptrs from x to y
    device::gpu::CudaDriver::CopyDeviceMemToDeviceAsync(csc_batch_pointers_addr, csr_batch_pointers_addr,
                                                        sizeof(S) * (num_batches + 1),
                                                        reinterpret_cast<cudaStream_t>(stream_ptr));
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
      cudaMemsetAsync(csc_col_pointers_addr, 0, sizeof(S) * num_batches * (col_size + 1),
                      reinterpret_cast<cudaStream_t>(stream_ptr)),
      "For 'SparseMatrixTranspose', it's cudaMemsetAsync failed.");
    // copy shape to y
    std::swap(host_shape_pointers[rank_ - kIndex1], host_shape_pointers[rank_ - kIndex2]);
    device::gpu::CudaDriver::CopyHostMemToDeviceAsync(csc_dense_shape_addr, host_shape_pointers.data(),
                                                      sizeof(S) * rank_, reinterpret_cast<cudaStream_t>(stream_ptr));
    return true;
  }

  S *csr_col_indices_addr = GetDeviceAddress<S>(inputs, kIndex3);
  T *csr_values_addr = GetDeviceAddress<T>(inputs, kIndex4);
  S *csc_row_indices_addr = GetDeviceAddress<S>(outputs, kIndex3);
  T *csc_values_addr = GetDeviceAddress<T>(outputs, kIndex4);

  auto total_nnz = input_col_indices_shapes_[kIndex0];

  std::vector<S> host_batch_pointers(input_batch_pointers_shapes_[kIndex0], 0);
  device::gpu::CudaDriver::CopyDeviceMemToHost(host_batch_pointers.data(), csr_batch_pointers_addr,
                                               sizeof(S) * (num_batches + 1));
  size_t buffer_size = GetBufferSize<S>(num_batches, row_size, col_size, host_batch_pointers, csr_row_pointers_addr,
                                        csr_col_indices_addr, csc_col_pointers_addr, csc_row_indices_addr);
  auto &allocator = device::gpu::GPUMemoryAllocator::GetInstance();
  void *buffer = allocator.AllocTensorMem(buffer_size);
  MS_EXCEPTION_IF_NULL(buffer);
  if (!batched) {
    Csr2Csc(ms_type_, handle_, row_size, col_size, total_nnz, ToVoid(csr_values_addr), csr_row_pointers_addr,
            csr_col_indices_addr, ToVoid(csc_values_addr), csc_col_pointers_addr, csc_row_indices_addr, buffer);
  } else {
    for (size_t i = 0; i < num_batches; i++) {
      S offset = host_batch_pointers[i];  // get the single matrix
      S nnz = host_batch_pointers[i + 1] - offset;
      T *x_value = csr_values_addr + offset;
      T *y_value = csc_values_addr + offset;
      S *x_col_ind = csr_col_indices_addr + offset;
      S *y_col_ind = csc_row_indices_addr + offset;
      S *x_row_ptr = csr_row_pointers_addr + (row_size + 1) * i;
      S *y_row_ptr = csc_col_pointers_addr + (col_size + 1) * i;
      Csr2Csc(ms_type_, handle_, row_size, col_size, nnz, ToVoid(x_value), x_row_ptr, x_col_ind, ToVoid(y_value),
              y_row_ptr, y_col_ind, buffer);
    }
  }
  allocator.FreeTensorMem(buffer);
  // copy batch ptrs from x to y
  device::gpu::CudaDriver::CopyDeviceMemToDeviceAsync(csc_batch_pointers_addr, csr_batch_pointers_addr,
                                                      sizeof(S) * (num_batches + 1),
                                                      reinterpret_cast<cudaStream_t>(stream_ptr));
  // copy shape to y
  int kOne = 1;
  int kTwo = 2;
  std::swap(host_shape_pointers[rank_ - kOne], host_shape_pointers[rank_ - kTwo]);
  device::gpu::CudaDriver::CopyHostMemToDeviceAsync(csc_dense_shape_addr, host_shape_pointers.data(), sizeof(S) * rank_,
                                                    reinterpret_cast<cudaStream_t>(stream_ptr));

  if constexpr (std::is_same<T, std::complex<float>>::value) {
    if (conjugate) {
      auto status = Conj(output_values_size_, reinterpret_cast<cuComplex *>(csc_values_addr),
                         reinterpret_cast<cudaStream_t>(stream_ptr));
      CHECK_CUDA_STATUS(status, kernel_name_);
    }
  }
  if constexpr (std::is_same<T, std::complex<double>>::value) {
    if (conjugate) {
      auto status = Conj(output_values_size_, reinterpret_cast<cuDoubleComplex *>(csc_values_addr),
                         reinterpret_cast<cudaStream_t>(stream_ptr));
      CHECK_CUDA_STATUS(status, kernel_name_);
    }
  }
  return true;
}

std::vector<std::pair<KernelAttr, SparseMatrixTransposeGpuKernelMod::SparseMatrixTransposeLaunchFunc>>
  SparseMatrixTransposeGpuKernelMod::func_list_ = {
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeFloat32),
     &SparseMatrixTransposeGpuKernelMod::LaunchKernel<int32_t, float>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeFloat64)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeFloat64),
     &SparseMatrixTransposeGpuKernelMod::LaunchKernel<int32_t, double>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeComplex64)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeComplex64),
     &SparseMatrixTransposeGpuKernelMod::LaunchKernel<int32_t, std::complex<float>>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeComplex128)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeComplex128),
     &SparseMatrixTransposeGpuKernelMod::LaunchKernel<int32_t, std::complex<double>>}};

std::vector<KernelAttr> SparseMatrixTransposeGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(
    func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
    [](const std::pair<KernelAttr, SparseMatrixTransposeGpuKernelMod::SparseMatrixTransposeLaunchFunc> &pair) {
      return pair.first;
    });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, SparseMatrixTranspose, SparseMatrixTransposeGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
