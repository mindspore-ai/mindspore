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
#include <functional>
#include <algorithm>
#include "plugin/device/gpu/kernel/sparse/csr_sparse_matrix_to_dense_gpu_kernel.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/complex.h"
#include "mindspore/core/ops/csr_sparse_matrix_to_dense.h"

namespace mindspore {
namespace kernel {
constexpr size_t kISRSparseMatrixToDenseInputsNum = 5;
constexpr size_t kISRSparseMatrixToDenseOutputsNum = 1;

bool CSRSparseMatrixToDenseGpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                              const std::vector<KernelTensorPtr> &inputs,
                                              const std::vector<KernelTensorPtr> &outputs) {
  auto kernel_ptr = std::dynamic_pointer_cast<ops::CSRSparseMatrixToDense>(base_operator);
  if (!kernel_ptr) {
    MS_LOG(ERROR) << "cast CSRSparseMatrixToDense ops failed!";
    return false;
  }
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kISRSparseMatrixToDenseInputsNum, kernel_ptr->name());
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kISRSparseMatrixToDenseOutputsNum, kernel_ptr->name());

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(EXCEPTION) << "For '" << kernel_ptr->name()
                      << "', it does not support this kernel data type: " << kernel_attr;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int CSRSparseMatrixToDenseGpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                               const std::vector<KernelTensorPtr> &inputs,
                                               const std::vector<KernelTensorPtr> &outputs,
                                               const std::map<uint32_t, tensor::TensorPtr> &others) {
  ResetResource();
  dense_shape_shape_ = inputs[kIndex0]->GetShapeVector();
  batch_ptr_shape_ = inputs[kIndex1]->GetShapeVector();
  row_ptr_shape_ = inputs[kIndex2]->GetShapeVector();
  col_indices_shape_ = inputs[kIndex3]->GetShapeVector();
  values_shape_ = inputs[kIndex4]->GetShapeVector();
  output_shape_ = outputs[kIndex0]->GetShapeVector();
  if (!(CHECK_SHAPE_POSITIVE(dense_shape_shape_) && CHECK_SHAPE_POSITIVE(batch_ptr_shape_) &&
        CHECK_SHAPE_POSITIVE(row_ptr_shape_) && CHECK_SHAPE_POSITIVE(col_indices_shape_) &&
        CHECK_SHAPE_POSITIVE(values_shape_) && CHECK_SHAPE_POSITIVE(output_shape_))) {
    is_null_input_ = true;
    InitSizeLists();
    return 0;
  }

  MS_EXCEPTION_IF_CHECK_FAIL(!dense_shape_shape_.empty(), "dense_shape_shape_ should not be empty!");
  MS_EXCEPTION_IF_CHECK_FAIL(!row_ptr_shape_.empty(), "row_ptr_shape_ should not be empty!");
  MS_EXCEPTION_IF_CHECK_FAIL(!output_shape_.empty(), "output_shape_ should not be empty!");
  ndim = dense_shape_shape_[kIndex0];
  rows = row_ptr_shape_[kIndex0] - 1;
  nums = std::accumulate(output_shape_.begin(), output_shape_.end(), 1, std::multiplies<size_t>());

  auto GetNums = [](const std::vector<int64_t> &shape) {
    size_t res = 1;
    for (const auto &sh : shape) {
      res *= LongToSize(sh);
    }
    return res;
  };
  dense_shape_size_ = abstract::TypeIdSize(inputs[kIndex0]->GetDtype()) * GetNums(dense_shape_shape_);
  batch_ptr_size_ = abstract::TypeIdSize(inputs[kIndex1]->GetDtype()) * GetNums(batch_ptr_shape_);
  row_ptr_size_ = abstract::TypeIdSize(inputs[kIndex2]->GetDtype()) * GetNums(row_ptr_shape_);
  col_indices_size_ = abstract::TypeIdSize(inputs[kIndex3]->GetDtype()) * GetNums(col_indices_shape_);
  values_size_ = abstract::TypeIdSize(inputs[kIndex4]->GetDtype()) * GetNums(values_shape_);
  output_size_ = abstract::TypeIdSize(outputs[kIndex0]->GetDtype()) * GetNums(output_shape_);

  InitSizeLists();
  return 0;
}  // namespace kernel

void CSRSparseMatrixToDenseGpuKernelMod::ResetResource() noexcept {
  is_null_input_ = false;
  dense_shape_size_ = 0;
  batch_ptr_size_ = 0;
  row_ptr_size_ = 0;
  col_indices_size_ = 0;
  values_size_ = 0;
  output_size_ = 0;
  input_size_list_.clear();
  output_size_list_.clear();
}

void CSRSparseMatrixToDenseGpuKernelMod::InitSizeLists() {
  input_size_list_.push_back(dense_shape_size_);
  input_size_list_.push_back(batch_ptr_size_);
  input_size_list_.push_back(row_ptr_size_);
  input_size_list_.push_back(col_indices_size_);
  input_size_list_.push_back(values_size_);
  output_size_list_.push_back(output_size_);
}

template <typename T, typename S>
bool CSRSparseMatrixToDenseGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                                      const std::vector<AddressPtr> &,
                                                      const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  if (is_null_input_) {
    return true;
  }
  T *dense_shape_addr = GetDeviceAddress<T>(inputs, kIndex0);
  T *batch_ptr_addr = GetDeviceAddress<T>(inputs, kIndex1);
  T *row_ptr_addr = GetDeviceAddress<T>(inputs, kIndex2);
  T *col_indices_addr = GetDeviceAddress<T>(inputs, kIndex3);
  S *values_addr = GetDeviceAddress<S>(inputs, kIndex4);
  S *output_addr = GetDeviceAddress<S>(outputs, kIndex0);

  CalCSRSparseMatrixToDense(dense_shape_addr, batch_ptr_addr, row_ptr_addr, col_indices_addr, values_addr, output_addr,
                            ndim, rows, nums, reinterpret_cast<cudaStream_t>(stream_ptr));
  return true;
}

template <typename T>
using Complex = mindspore::utils::Complex<T>;

std::vector<std::pair<KernelAttr, CSRSparseMatrixToDenseGpuKernelMod::CSRSparseMatrixToDenseFunc>>
  CSRSparseMatrixToDenseGpuKernelMod::func_list_ = {
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeFloat16)
       .AddOutputAttr(kNumberTypeFloat16),
     &CSRSparseMatrixToDenseGpuKernelMod::LaunchKernel<int, half>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeFloat32),
     &CSRSparseMatrixToDenseGpuKernelMod::LaunchKernel<int, float>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeFloat64)
       .AddOutputAttr(kNumberTypeFloat64),
     &CSRSparseMatrixToDenseGpuKernelMod::LaunchKernel<int, double>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeComplex64)
       .AddOutputAttr(kNumberTypeComplex64),
     &CSRSparseMatrixToDenseGpuKernelMod::LaunchKernel<int, Complex<float>>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeComplex128)
       .AddOutputAttr(kNumberTypeComplex128),
     &CSRSparseMatrixToDenseGpuKernelMod::LaunchKernel<int, Complex<double>>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeFloat16)
       .AddOutputAttr(kNumberTypeFloat16),
     &CSRSparseMatrixToDenseGpuKernelMod::LaunchKernel<int64_t, half>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeFloat32),
     &CSRSparseMatrixToDenseGpuKernelMod::LaunchKernel<int64_t, float>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeFloat64)
       .AddOutputAttr(kNumberTypeFloat64),
     &CSRSparseMatrixToDenseGpuKernelMod::LaunchKernel<int64_t, double>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeComplex64)
       .AddOutputAttr(kNumberTypeComplex64),
     &CSRSparseMatrixToDenseGpuKernelMod::LaunchKernel<int64_t, Complex<float>>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeComplex128)
       .AddOutputAttr(kNumberTypeComplex128),
     &CSRSparseMatrixToDenseGpuKernelMod::LaunchKernel<int64_t, Complex<double>>},
};

std::vector<KernelAttr> CSRSparseMatrixToDenseGpuKernelMod::GetOpSupport() {
  static std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, CSRSparseMatrixToDenseFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, CSRSparseMatrixToDense, CSRSparseMatrixToDenseGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
