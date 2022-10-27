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

#include <iostream>
#include "plugin/device/cpu/kernel/csr_sparse_matrix_to_sparse_tensor_cpu_kernel.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kRankWithoutBatch = 2;
constexpr size_t kRankWithBatch = 3;
constexpr size_t kZero = 0;
constexpr size_t kOne = 1;
constexpr size_t kTwo = 2;
constexpr size_t kCSRSparseMatrixToSparseTensorInputsNum = 5;
constexpr size_t kCSRSparseMatrixToSparseTensorOutputsNum = 3;
constexpr size_t kInputIndex0 = 0;
constexpr size_t kInputIndex1 = 1;
constexpr size_t kInputIndex2 = 2;
constexpr size_t kInputIndex3 = 3;
constexpr size_t kInputIndex4 = 4;
constexpr size_t kOutputIndex0 = 0;
constexpr size_t kOutputIndex1 = 1;
constexpr size_t kOutputIndex2 = 2;
constexpr int64_t kInitPrevBatch = -1;
constexpr char kKernelName[] = "CSRSparseMatrixToSparseTensor";

#define ADD_KERNEL(t1, t2, t3, t4, t5, t6, t7, t8) \
  KernelAttr()                                     \
    .AddInputAttr(kNumberType##t1)                 \
    .AddInputAttr(kNumberType##t2)                 \
    .AddInputAttr(kNumberType##t3)                 \
    .AddInputAttr(kNumberType##t4)                 \
    .AddInputAttr(kNumberType##t5)                 \
    .AddOutputAttr(kNumberType##t6)                \
    .AddOutputAttr(kNumberType##t7)                \
    .AddOutputAttr(kNumberType##t8)
}  // namespace

bool CSRSparseMatrixToSparseTensorCpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                                     const std::vector<KernelTensorPtr> &inputs,
                                                     const std::vector<KernelTensorPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kCSRSparseMatrixToSparseTensorInputsNum, kKernelName);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kCSRSparseMatrixToSparseTensorOutputsNum, kKernelName);
  indice_type_ = inputs[kInputIndex0]->GetDtype();
  value_type_ = inputs[kInputIndex4]->GetDtype();
  kernel_name_ = base_operator->GetPrim()->name();
  return true;
}

int CSRSparseMatrixToSparseTensorCpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                                      const std::vector<KernelTensorPtr> &inputs,
                                                      const std::vector<KernelTensorPtr> &outputs,
                                                      const std::map<uint32_t, tensor::TensorPtr> &) {
  auto ret = KernelMod::Resize(base_operator, inputs, outputs);
  if (ret != KRET_OK) {
    return ret;
  }

  kernel_name_ = base_operator->GetPrim()->name();
  auto x_indices_shape = inputs[kInputIndex4]->GetShapeVector();
  auto dense_shape_shape = inputs[kInputIndex0]->GetShapeVector();
  total_nnz_ = static_cast<size_t>(x_indices_shape[kZero]);
  rank_ = static_cast<size_t>(dense_shape_shape[kZero]);
  if (rank_ != kRankWithoutBatch && rank_ != kRankWithBatch) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the input dense_shape should "
                  << "have rank 2 or 3, but got " << rank_ << ".";
    return KRET_RESIZE_FAILED;
  }
  return KRET_OK;
}

bool CSRSparseMatrixToSparseTensorCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                                       const std::vector<kernel::AddressPtr> &,
                                                       const std::vector<kernel::AddressPtr> &outputs) {
  switch (indice_type_) {
    case kNumberTypeInt32:
      switch (value_type_) {
        case kNumberTypeFloat32:
          LaunchKernel<int32_t, float>(inputs, outputs);
          break;
        case kNumberTypeFloat64:
          LaunchKernel<int32_t, double>(inputs, outputs);
          break;
        case kNumberTypeComplex64:
          LaunchKernel<int32_t, complex64>(inputs, outputs);
          break;
        case kNumberTypeComplex128:
          LaunchKernel<int32_t, complex128>(inputs, outputs);
          break;
        default:
          MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', dtype of values should be "
                            << "float32, float64, complex64 or complex128, but got "
                            << TypeIdToType(value_type_)->ToString() << ".";
      }
      break;
    case kNumberTypeInt64:
      switch (value_type_) {
        case kNumberTypeFloat32:
          LaunchKernel<int64_t, float>(inputs, outputs);
          break;
        case kNumberTypeFloat64:
          LaunchKernel<int64_t, double>(inputs, outputs);
          break;
        case kNumberTypeComplex64:
          LaunchKernel<int64_t, complex64>(inputs, outputs);
          break;
        case kNumberTypeComplex128:
          LaunchKernel<int64_t, complex128>(inputs, outputs);
          break;
        default:
          MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', dtype of values should be "
                            << "float32, float64, complex64 or complex128, but got "
                            << TypeIdToType(value_type_)->ToString() << ".";
      }
      break;
    default:
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', dtype of indices should be int32 or int64, "
                        << "but got " << TypeIdToType(indice_type_)->ToString() << ".";
  }
  return true;
}

template <typename indiceT, typename valueT>
void CSRSparseMatrixToSparseTensorCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                                             const std::vector<kernel::AddressPtr> &outputs) {
  indiceT *x_dense_shape_ptr = static_cast<indiceT *>(inputs[kInputIndex0]->addr);
  indiceT *x_batch_pointers_ptr = static_cast<indiceT *>(inputs[kInputIndex1]->addr);
  indiceT *x_row_pointers_ptr = static_cast<indiceT *>(inputs[kInputIndex2]->addr);
  indiceT *x_col_indices_ptr = static_cast<indiceT *>(inputs[kInputIndex3]->addr);
  valueT *x_values_ptr = static_cast<valueT *>(inputs[kInputIndex4]->addr);
  batch_size_ = (rank_ == kRankWithoutBatch) ? kOne : static_cast<size_t>(x_dense_shape_ptr[kZero]);
  const size_t shift = (rank_ == kRankWithoutBatch) ? kZero : kOne;
  num_rows_ = static_cast<size_t>(*(static_cast<indiceT *>(inputs[kInputIndex0]->addr) + shift));
  indiceT *indices_ptr = static_cast<indiceT *>(outputs[kOutputIndex0]->addr);
  valueT *values_ptr = static_cast<valueT *>(outputs[kOutputIndex1]->addr);
  indiceT *dense_shape_ptr = static_cast<indiceT *>(outputs[kOutputIndex2]->addr);
  for (size_t i = kZero; i < rank_; i++) {
    dense_shape_ptr[i] = x_dense_shape_ptr[i];
  }
  for (size_t i = kZero; i < total_nnz_; i++) {
    values_ptr[i] = x_values_ptr[i];
  }
  for (int64_t batch_idx = static_cast<int64_t>(kZero); batch_idx < SizeToLong(batch_size_); ++batch_idx) {
    const int64_t batch_offset = x_batch_pointers_ptr[batch_idx];
    for (int row_idx = kZero; row_idx < SizeToLong(num_rows_); ++row_idx) {
      const int64_t row_offset = static_cast<int64_t>(batch_idx * (num_rows_ + kOne) + row_idx);
      const int64_t col_begin = static_cast<int64_t>(x_row_pointers_ptr[row_offset]);
      const int64_t col_end = static_cast<int64_t>(x_row_pointers_ptr[row_offset + kOne]);
      for (int64_t i = col_begin; i < col_end; ++i) {
        const int64_t col_idx = static_cast<int64_t>(x_col_indices_ptr[batch_offset + i]);
        const int64_t indices_offset = static_cast<int64_t>(rank_ * (batch_offset + i));
        if (rank_ == kRankWithoutBatch) {
          indices_ptr[indices_offset] = static_cast<indiceT>(row_idx);
          indices_ptr[indices_offset + kOne] = static_cast<indiceT>(col_idx);
        } else {
          indices_ptr[indices_offset] = static_cast<indiceT>(batch_idx);
          indices_ptr[indices_offset + kOne] = static_cast<indiceT>(row_idx);
          indices_ptr[indices_offset + kTwo] = static_cast<indiceT>(col_idx);
        }
      }
    }
  }
}

std::vector<KernelAttr> CSRSparseMatrixToSparseTensorCpuKernelMod::GetOpSupport() {
  static std::vector<KernelAttr> kernel_attr_list = {
    ADD_KERNEL(Int32, Int32, Int32, Int32, Float32, Int32, Float32, Int32),
    ADD_KERNEL(Int32, Int32, Int32, Int32, Float64, Int32, Float64, Int32),
    ADD_KERNEL(Int32, Int32, Int32, Int32, Complex64, Int32, Complex64, Int32),
    ADD_KERNEL(Int32, Int32, Int32, Int32, Complex128, Int32, Complex128, Int32),
    ADD_KERNEL(Int64, Int64, Int64, Int64, Float32, Int64, Float32, Int64),
    ADD_KERNEL(Int64, Int64, Int64, Int64, Float64, Int64, Float64, Int64),
    ADD_KERNEL(Int64, Int64, Int64, Int64, Complex64, Int64, Complex64, Int64),
    ADD_KERNEL(Int64, Int64, Int64, Int64, Complex128, Int64, Complex128, Int64)};

  return kernel_attr_list;
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, CSRSparseMatrixToSparseTensor, CSRSparseMatrixToSparseTensorCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
