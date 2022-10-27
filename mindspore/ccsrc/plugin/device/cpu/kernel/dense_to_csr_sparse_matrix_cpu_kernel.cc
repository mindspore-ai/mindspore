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

#include "plugin/device/cpu/kernel/dense_to_csr_sparse_matrix_cpu_kernel.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kZero = 0;
constexpr size_t kOne = 1;
constexpr size_t kTwo = 2;
constexpr size_t kDefaultRank = 2;
constexpr size_t kInputIndex0 = 0;
constexpr size_t kInputIndex1 = 1;
constexpr size_t kOutputIndex0 = 0;
constexpr size_t kOutputIndex1 = 1;
constexpr size_t kOutputIndex2 = 2;
constexpr size_t kOutputIndex3 = 3;
constexpr size_t kOutputIndex4 = 4;
constexpr size_t kDenseToCSRSparseMatrixInputsNum = 2;
constexpr size_t kDenseToCSRSparseMatrixOutputsNum = 5;
constexpr int64_t kInitPrevBatch = -1;
}  // namespace

bool DenseToCSRSparseMatrixCpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                              const std::vector<KernelTensorPtr> &inputs,
                                              const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->GetPrim()->name();
  indices_type_ = inputs[kInputIndex1]->GetDtype();
  values_type_ = inputs[kInputIndex0]->GetDtype();
  return true;
}

int DenseToCSRSparseMatrixCpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                               const std::vector<KernelTensorPtr> &inputs,
                                               const std::vector<KernelTensorPtr> &outputs,
                                               const std::map<uint32_t, tensor::TensorPtr> &) {
  auto ret = KernelMod::Resize(base_operator, inputs, outputs);
  if (ret != KRET_OK) {
    return ret;
  }
  auto dense_shape = inputs[kInputIndex0]->GetShapeVector();
  auto indices_shape = inputs[kInputIndex1]->GetShapeVector();
  rank_ = dense_shape.size();
  total_nnz_ = indices_shape[kZero];
  batch_size_ = (rank_ == kDefaultRank) ? kOne : dense_shape[kZero];
  num_rows_ = (rank_ == kDefaultRank) ? dense_shape[kZero] : dense_shape[kOne];
  num_cols_ = (rank_ == kDefaultRank) ? dense_shape[kOne] : dense_shape[kTwo];
  return KRET_OK;
}

bool DenseToCSRSparseMatrixCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                                const std::vector<kernel::AddressPtr> &,
                                                const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kDenseToCSRSparseMatrixInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kDenseToCSRSparseMatrixOutputsNum, kernel_name_);
  switch (indices_type_) {
    case kNumberTypeInt32:
      switch (values_type_) {
        case kNumberTypeFloat32:
          LaunchKernel<int32_t, float>(inputs, outputs);
          break;
        case kNumberTypeFloat64:
          LaunchKernel<int32_t, double>(inputs, outputs);
          break;
        case kNumberTypeComplex64:
          LaunchKernel<int32_t, std::complex<float>>(inputs, outputs);
          break;
        case kNumberTypeComplex128:
          LaunchKernel<int32_t, std::complex<double>>(inputs, outputs);
          break;
        default:
          MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', dtype of values should be "
                            << "float32, float64, complex64 or complex128, but got "
                            << TypeIdToType(values_type_)->ToString() << ".";
      }
      break;
    case kNumberTypeInt64:
      switch (values_type_) {
        case kNumberTypeFloat32:
          LaunchKernel<int64_t, float>(inputs, outputs);
          break;
        case kNumberTypeFloat64:
          LaunchKernel<int64_t, double>(inputs, outputs);
          break;
        case kNumberTypeComplex64:
          LaunchKernel<int64_t, std::complex<float>>(inputs, outputs);
          break;
        case kNumberTypeComplex128:
          LaunchKernel<int64_t, std::complex<double>>(inputs, outputs);
          break;
        default:
          MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', dtype of values should be "
                            << "float32, float64, complex64 or complex128, but got "
                            << TypeIdToType(values_type_)->ToString() << ".";
      }
      break;
    default:
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', dtype of indices should be int32 or int64, "
                        << "but got " << TypeIdToType(indices_type_)->ToString() << ".";
  }
  return true;
}

template <typename indiceT, typename valueT>
void DenseToCSRSparseMatrixCpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                                      const std::vector<AddressPtr> &outputs) const {
  auto dense_input_ptr = reinterpret_cast<valueT *>(inputs[kInputIndex0]->addr);
  auto indices_ptr = reinterpret_cast<indiceT *>(inputs[kInputIndex1]->addr);
  auto y_dense_shape_ptr = reinterpret_cast<indiceT *>(outputs[kOutputIndex0]->addr);
  auto y_batch_pointers_ptr = reinterpret_cast<indiceT *>(outputs[kOutputIndex1]->addr);
  auto y_row_pointers_ptr = reinterpret_cast<indiceT *>(outputs[kOutputIndex2]->addr);
  auto y_col_indices_ptr = reinterpret_cast<indiceT *>(outputs[kOutputIndex3]->addr);
  auto y_values_ptr = reinterpret_cast<valueT *>(outputs[kOutputIndex4]->addr);
  if (rank_ == kDefaultRank) {
    y_dense_shape_ptr[kZero] = indiceT(num_rows_);
    y_dense_shape_ptr[kOne] = indiceT(num_cols_);
  } else {
    y_dense_shape_ptr[kZero] = indiceT(batch_size_);
    y_dense_shape_ptr[kOne] = indiceT(num_rows_);
    y_dense_shape_ptr[kTwo] = indiceT(num_cols_);
  }
  for (size_t i = kZero; i < total_nnz_; i++) {
    if (rank_ == kDefaultRank) {
      auto cur_idx = indices_ptr[i * rank_] * indiceT(num_cols_) + indices_ptr[i * rank_ + kOne];
      y_values_ptr[i] = dense_input_ptr[LongToSize(cur_idx)];
    } else {
      auto cur_idx = indices_ptr[i * rank_] * indiceT(num_rows_) * indiceT(num_cols_) +
                     indices_ptr[i * rank_ + kOne] * indiceT(num_cols_) + indices_ptr[i * rank_ + kTwo];
      y_values_ptr[i] = dense_input_ptr[LongToSize(cur_idx)];
    }
  }
  for (size_t i = kZero; i < batch_size_ * (num_rows_ + kOne); i++) {
    y_row_pointers_ptr[i] = indiceT(kZero);
  }
  int64_t prev_batch = kInitPrevBatch;
  if (rank_ == kDefaultRank) {
    y_batch_pointers_ptr[kZero] = indiceT(kZero);
    ++prev_batch;
    for (size_t i = kZero; i < total_nnz_; ++i) {
      ++y_row_pointers_ptr[LongToSize(indices_ptr[i * rank_]) + kOne];
      y_col_indices_ptr[i] = indices_ptr[i * rank_ + kOne];
    }
  } else {
    for (size_t i = kZero; i < total_nnz_; ++i) {
      size_t cur_batch = LongToSize(indices_ptr[i * rank_]);
      ++y_row_pointers_ptr[cur_batch * (num_rows_ + kOne) + LongToSize(indices_ptr[i * rank_ + kOne]) + kOne];
      y_col_indices_ptr[i] = indices_ptr[i * rank_ + kTwo];
      while (prev_batch < SizeToLong(cur_batch)) {
        y_batch_pointers_ptr[prev_batch + SizeToLong(kOne)] = indiceT(i);
        ++prev_batch;
      }
    }
  }
  while (prev_batch < SizeToLong(batch_size_)) {
    y_batch_pointers_ptr[LongToSize(prev_batch + SizeToLong(kOne))] = indiceT(total_nnz_);
    ++prev_batch;
  }
  for (size_t batch_idx = 0; batch_idx < batch_size_; ++batch_idx) {
    auto *row_ptr_batch = y_row_pointers_ptr + batch_idx * (num_rows_ + kOne);
    (void)std::partial_sum(row_ptr_batch, row_ptr_batch + num_rows_ + kOne, row_ptr_batch);
  }
}

std::vector<KernelAttr> DenseToCSRSparseMatrixCpuKernelMod::GetOpSupport() {
  static const std::vector<KernelAttr> support_list = {KernelAttr()
                                                         .AddInputAttr(kNumberTypeFloat32)
                                                         .AddInputAttr(kNumberTypeInt32)
                                                         .AddOutputAttr(kNumberTypeInt32)
                                                         .AddOutputAttr(kNumberTypeInt32)
                                                         .AddOutputAttr(kNumberTypeInt32)
                                                         .AddOutputAttr(kNumberTypeInt32)
                                                         .AddOutputAttr(kNumberTypeFloat32),
                                                       KernelAttr()
                                                         .AddInputAttr(kNumberTypeFloat64)
                                                         .AddInputAttr(kNumberTypeInt32)
                                                         .AddOutputAttr(kNumberTypeInt32)
                                                         .AddOutputAttr(kNumberTypeInt32)
                                                         .AddOutputAttr(kNumberTypeInt32)
                                                         .AddOutputAttr(kNumberTypeInt32)
                                                         .AddOutputAttr(kNumberTypeFloat64),
                                                       KernelAttr()
                                                         .AddInputAttr(kNumberTypeComplex64)
                                                         .AddInputAttr(kNumberTypeInt32)
                                                         .AddOutputAttr(kNumberTypeInt32)
                                                         .AddOutputAttr(kNumberTypeInt32)
                                                         .AddOutputAttr(kNumberTypeInt32)
                                                         .AddOutputAttr(kNumberTypeInt32)
                                                         .AddOutputAttr(kNumberTypeComplex64),
                                                       KernelAttr()
                                                         .AddInputAttr(kNumberTypeComplex128)
                                                         .AddInputAttr(kNumberTypeInt32)
                                                         .AddOutputAttr(kNumberTypeInt32)
                                                         .AddOutputAttr(kNumberTypeInt32)
                                                         .AddOutputAttr(kNumberTypeInt32)
                                                         .AddOutputAttr(kNumberTypeInt32)
                                                         .AddOutputAttr(kNumberTypeComplex128),
                                                       KernelAttr()
                                                         .AddInputAttr(kNumberTypeFloat32)
                                                         .AddInputAttr(kNumberTypeInt64)
                                                         .AddOutputAttr(kNumberTypeInt64)
                                                         .AddOutputAttr(kNumberTypeInt64)
                                                         .AddOutputAttr(kNumberTypeInt64)
                                                         .AddOutputAttr(kNumberTypeInt64)
                                                         .AddOutputAttr(kNumberTypeFloat32),
                                                       KernelAttr()
                                                         .AddInputAttr(kNumberTypeFloat64)
                                                         .AddInputAttr(kNumberTypeInt64)
                                                         .AddOutputAttr(kNumberTypeInt64)
                                                         .AddOutputAttr(kNumberTypeInt64)
                                                         .AddOutputAttr(kNumberTypeInt64)
                                                         .AddOutputAttr(kNumberTypeInt64)
                                                         .AddOutputAttr(kNumberTypeFloat64),
                                                       KernelAttr()
                                                         .AddInputAttr(kNumberTypeComplex64)
                                                         .AddInputAttr(kNumberTypeInt64)
                                                         .AddOutputAttr(kNumberTypeInt64)
                                                         .AddOutputAttr(kNumberTypeInt64)
                                                         .AddOutputAttr(kNumberTypeInt64)
                                                         .AddOutputAttr(kNumberTypeInt64)
                                                         .AddOutputAttr(kNumberTypeComplex64),
                                                       KernelAttr()
                                                         .AddInputAttr(kNumberTypeComplex128)
                                                         .AddInputAttr(kNumberTypeInt64)
                                                         .AddOutputAttr(kNumberTypeInt64)
                                                         .AddOutputAttr(kNumberTypeInt64)
                                                         .AddOutputAttr(kNumberTypeInt64)
                                                         .AddOutputAttr(kNumberTypeInt64)
                                                         .AddOutputAttr(kNumberTypeComplex128)};
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, DenseToCSRSparseMatrix, DenseToCSRSparseMatrixCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
