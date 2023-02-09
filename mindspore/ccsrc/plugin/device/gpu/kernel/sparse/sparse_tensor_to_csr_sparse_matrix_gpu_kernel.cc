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

#include <complex>
#include <algorithm>
#include <functional>
#include <utility>
#include <map>
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/sparse/sparse_tensor_to_csr_sparse_matrix_gpu_kernel.h"
namespace mindspore {
namespace kernel {
constexpr size_t kRankWithoutBatch = 2;
constexpr size_t kRankWithBatch = 3;
constexpr size_t kZero = 0;
constexpr size_t kOne = 1;
constexpr size_t kTwo = 2;
bool SparseTensorToCSRSparseMatrixGpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                                     const std::vector<KernelTensorPtr> &inputs,
                                                     const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();
  if (kernel_name_ != prim::kPrimSparseTensorToCSRSparseMatrix->name()) {
    MS_LOG(ERROR) << "For 'SparseTensorToCSRSparseMatrixGpuKernelMod',"
                     "the kernel name must be 'SparseTensorToCSRSparseMatrix', but got "
                  << kernel_name_;
    return false;
  }
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  if (inputs.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' got empty inputs, which is invalid.";
    return false;
  }
  handle_ = device::gpu::GPUDeviceManager::GetInstance().GetCuSparseHandle();
  cusparseSetPointerMode(handle_, CUSPARSE_POINTER_MODE_HOST);
  return true;
}

int SparseTensorToCSRSparseMatrixGpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                                      const std::vector<KernelTensorPtr> &inputs,
                                                      const std::vector<KernelTensorPtr> &outputs,
                                                      const std::map<uint32_t, tensor::TensorPtr> &) {
  bapt = 0;
  elements[kZero] = 0;
  elements[kOne] = 0;
  elements[kTwo] = 0;
  input_size_list_.clear();
  output_size_list_.clear();
  workspace_size_list_.clear();
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  for (size_t i = 0; i < inputs.size(); i++) {
    std::vector<int64_t> input_shape = std::vector<int64_t>(inputs.at(i)->GetDeviceShapeAdaptively().begin(),
                                                            inputs.at(i)->GetDeviceShapeAdaptively().end());
    size_t input_elements_ = std::accumulate(input_shape.begin(), input_shape.end(), 1, std::multiplies<int64_t>());
    elements[i] = input_elements_;
    size_t unit_size_ = abstract::TypeIdSize(kernel_attr.GetInputAttr(i).dtype);
    input_size_list_.push_back(input_elements_ * unit_size_);
  }
  unit_size_ = abstract::TypeIdSize(kernel_attr.GetInputAttr(0).dtype);
  workspace_size_list_.push_back(elements[kOne] * unit_size_);
  for (size_t i = 0; i < outputs.size(); i++) {
    std::vector<int64_t> output_shape = std::vector<int64_t>(outputs.at(i)->GetDeviceShapeAdaptively().begin(),
                                                             outputs.at(i)->GetDeviceShapeAdaptively().end());
    size_t output_elements_ = std::accumulate(output_shape.begin(), output_shape.end(), 1, std::multiplies<int64_t>());
    size_t unit_size_ = abstract::TypeIdSize(kernel_attr.GetOutputAttr(i).dtype);
    output_size_list_.push_back(output_elements_ * unit_size_);
    if (i == 1) {
      bapt = output_elements_;
    }
  }
  return KRET_OK;
}

template <typename IndiceType, typename DataType>
bool SparseTensorToCSRSparseMatrixGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                                             const std::vector<AddressPtr> &workspace,
                                                             const std::vector<AddressPtr> &outputs) {
  IndiceType *x_indices_ptr = GetDeviceAddress<IndiceType>(inputs, kIndex0);
  DataType *x_value_ptr = GetDeviceAddress<DataType>(inputs, kIndex1);
  IndiceType *x_dense_shape_ptr = GetDeviceAddress<IndiceType>(inputs, kIndex2);
  IndiceType *y_dense_shape_ptr = GetDeviceAddress<IndiceType>(outputs, kIndex0);
  IndiceType *y_batch_pointers_ptr = GetDeviceAddress<IndiceType>(outputs, kIndex1);
  IndiceType *y_row_pointers_ptr = GetDeviceAddress<IndiceType>(outputs, kIndex2);
  IndiceType *y_col_indices_ptr = GetDeviceAddress<IndiceType>(outputs, kIndex3);
  DataType *y_value_ptr = GetDeviceAddress<DataType>(outputs, kIndex4);
  IndiceType *x_row_indices_ptr = GetDeviceAddress<IndiceType>(workspace, kIndex0);

  std::vector<IndiceType> y_batch_pointers_ptr_test(bapt);
  std::vector<IndiceType> x_dense_shape_ptr_test(elements[kTwo]);

  cudaMemsetAsync(y_batch_pointers_ptr, 0, outputs[kIndex1]->size, stream);

  cudaMemcpyAsync(x_dense_shape_ptr_test.data(), x_dense_shape_ptr, elements[kTwo] * sizeof(IndiceType),
                  cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
  SparseTensorToCSRSparseMatrix<IndiceType>(x_indices_ptr, x_row_indices_ptr, y_col_indices_ptr, y_batch_pointers_ptr,
                                            elements[kOne], elements[kTwo], stream, device_id_);
  if (elements[kTwo] == kRankWithoutBatch) {
    row_num = x_dense_shape_ptr_test[kZero];
    cusparseXcoo2csr(handle_, x_row_indices_ptr, elements[kOne], row_num, y_row_pointers_ptr, CUSPARSE_INDEX_BASE_ZERO);
  } else {
    batch_size = x_dense_shape_ptr_test[kZero];
    row_num = x_dense_shape_ptr_test[kOne];
    cudaMemcpyAsync(y_batch_pointers_ptr_test.data(), y_batch_pointers_ptr, (batch_size + 1) * sizeof(IndiceType),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    for (int i = 0; i < batch_size; ++i) {
      y_batch_pointers_ptr_test[i + 1] = std::max(y_batch_pointers_ptr_test[i + 1], y_batch_pointers_ptr_test[i]);
      int *temp_row_indices_addr = x_row_indices_ptr + y_batch_pointers_ptr_test[i];
      int *temp_row_pointers_addr = y_row_pointers_ptr + i * (row_num + 1);
      temp_nnz = y_batch_pointers_ptr_test[i + 1] - y_batch_pointers_ptr_test[i];
      if (temp_nnz != 0) {
        cusparseXcoo2csr(handle_, temp_row_indices_addr, temp_nnz, row_num, temp_row_pointers_addr,
                         CUSPARSE_INDEX_BASE_ZERO);
      }
    }
    cudaMemcpyAsync(y_batch_pointers_ptr, y_batch_pointers_ptr_test.data(), (batch_size + 1) * sizeof(IndiceType),
                    cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);
  }
  cudaMemcpyAsync(y_dense_shape_ptr, x_dense_shape_ptr, elements[kTwo] * sizeof(DataType), cudaMemcpyDeviceToDevice,
                  stream);
  cudaStreamSynchronize(stream);
  cudaMemcpyAsync(y_value_ptr, x_value_ptr, elements[kOne] * sizeof(DataType), cudaMemcpyDeviceToDevice, stream);
  cudaStreamSynchronize(stream);
  return true;
}

std::vector<std::pair<KernelAttr, SparseTensorToCSRSparseMatrixGpuKernelMod::SparseTensorToCSRSparseMatrixFunc>>
  SparseTensorToCSRSparseMatrixGpuKernelMod::func_list_ = {
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeFloat32),
     &SparseTensorToCSRSparseMatrixGpuKernelMod::LaunchKernel<int32_t, float>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeFloat64),
     &SparseTensorToCSRSparseMatrixGpuKernelMod::LaunchKernel<int32_t, double>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeComplex64)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeComplex64),
     &SparseTensorToCSRSparseMatrixGpuKernelMod::LaunchKernel<int32_t, std::complex<float>>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeComplex128)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeComplex128),
     &SparseTensorToCSRSparseMatrixGpuKernelMod::LaunchKernel<int32_t, std::complex<double>>},
};

std::vector<KernelAttr> SparseTensorToCSRSparseMatrixGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, SparseTensorToCSRSparseMatrixFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, SparseTensorToCSRSparseMatrix, SparseTensorToCSRSparseMatrixGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
