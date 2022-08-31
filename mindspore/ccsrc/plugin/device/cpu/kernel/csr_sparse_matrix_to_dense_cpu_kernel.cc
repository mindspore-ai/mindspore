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

#include "plugin/device/cpu/kernel/csr_sparse_matrix_to_dense_cpu_kernel.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kZero = 0;
constexpr size_t kOne = 1;
constexpr size_t kDefaultRank = 2;
constexpr size_t kInputIndex0 = 0;
constexpr size_t kInputIndex1 = 1;
constexpr size_t kInputIndex2 = 2;
constexpr size_t kInputIndex3 = 3;
constexpr size_t kInputIndex4 = 4;
constexpr size_t kOutputIndex = 0;
constexpr size_t kCSRSparseMatrixToDenseInputsNum = 5;
constexpr size_t kCSRSparseMatrixToDenseOutputsNum = 1;
}  // namespace

void CSRSparseMatrixToDenseCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  node_wpt_ = kernel_node;
  indices_type = AnfAlgo::GetInputDeviceDataType(kernel_node, kInputIndex0);
  values_type = AnfAlgo::GetInputDeviceDataType(kernel_node, kInputIndex4);
  rank_ = static_cast<size_t>(common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, kInputIndex0)[kZero]);
  batch_size_ =
    static_cast<size_t>(common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, kInputIndex1)[kZero]) - kOne;
}

bool CSRSparseMatrixToDenseCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                                const std::vector<kernel::AddressPtr> &,
                                                const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kCSRSparseMatrixToDenseInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kCSRSparseMatrixToDenseOutputsNum, kernel_name_);
  switch (indices_type) {
    case kNumberTypeInt32:
      switch (values_type) {
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
                            << TypeIdToType(values_type)->ToString();
      }
      break;
    case kNumberTypeInt64:
      switch (values_type) {
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
                            << TypeIdToType(values_type)->ToString();
      }
      break;
    default:
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', dtype of indices should be int32 or int64, "
                        << "but got " << TypeIdToType(indices_type)->ToString();
  }
  auto node_ = node_wpt_.lock();
  if (!node_) {
    MS_LOG(EXCEPTION) << "node_wpt_ is expired.";
  }
  std::vector<TypeId> y_dtype = {values_type};
  std::vector<int64_t> y_dims;
  if (rank_ == kDefaultRank) {
    y_dims = {SizeToLong(num_rows_), SizeToLong(num_cols_)};
  } else {
    y_dims = {SizeToLong(batch_size_), SizeToLong(num_rows_), SizeToLong(num_cols_)};
  }
  (void)common::AnfAlgo::SetOutputInferTypeAndShape(y_dtype, {y_dims}, node_.get());
  return true;
}

template <typename indiceT, typename valueT>
void CSRSparseMatrixToDenseCpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                                      const std::vector<AddressPtr> &outputs) {
  const size_t shift = (rank_ == kDefaultRank) ? kZero : kOne;
  num_rows_ = static_cast<size_t>(*(static_cast<indiceT *>(inputs[kInputIndex0]->addr) + shift));
  num_cols_ = static_cast<size_t>(*(static_cast<indiceT *>(inputs[kInputIndex0]->addr) + shift + kOne));
  indiceT *batch_ptrs = static_cast<indiceT *>(inputs[kInputIndex1]->addr);
  indiceT *row_ptrs = static_cast<indiceT *>(inputs[kInputIndex2]->addr);
  indiceT *col_ind = static_cast<indiceT *>(inputs[kInputIndex3]->addr);
  valueT *values = static_cast<valueT *>(inputs[kInputIndex4]->addr);
  valueT *y_data = static_cast<valueT *>(outputs[kOutputIndex]->addr);
  for (size_t batch_idx = kZero; batch_idx < batch_size_; batch_idx++) {
    const size_t dense_offset = batch_idx * num_rows_ * num_cols_;
    for (size_t i = kZero; i < num_rows_ * num_cols_; ++i) {
      y_data[dense_offset + i] = valueT(kZero);
    }
    const size_t csr_batch_offset = static_cast<size_t>(batch_ptrs[batch_idx]);
    for (size_t row_idx = kZero; row_idx < num_rows_; ++row_idx) {
      const size_t row_offset = batch_idx * (num_rows_ + kOne) + row_idx;
      const size_t col_begin = static_cast<size_t>(row_ptrs[row_offset]);
      const size_t col_end = static_cast<size_t>(row_ptrs[row_offset + kOne]);
      for (size_t i = col_begin; i < col_end; ++i) {
        const size_t col_idx = static_cast<size_t>(col_ind[csr_batch_offset + i]);
        y_data[dense_offset + (row_idx * num_cols_) + col_idx] = values[csr_batch_offset + i];
      }
    }
  }
}

std::vector<KernelAttr> CSRSparseMatrixToDenseCpuKernelMod::GetOpSupport() {
  static std::vector<KernelAttr> support_list = {KernelAttr()
                                                   .AddInputAttr(kNumberTypeInt32)
                                                   .AddInputAttr(kNumberTypeInt32)
                                                   .AddInputAttr(kNumberTypeInt32)
                                                   .AddInputAttr(kNumberTypeInt32)
                                                   .AddInputAttr(kNumberTypeFloat32)
                                                   .AddOutputAttr(kNumberTypeFloat32),
                                                 KernelAttr()
                                                   .AddInputAttr(kNumberTypeInt32)
                                                   .AddInputAttr(kNumberTypeInt32)
                                                   .AddInputAttr(kNumberTypeInt32)
                                                   .AddInputAttr(kNumberTypeInt32)
                                                   .AddInputAttr(kNumberTypeFloat64)
                                                   .AddOutputAttr(kNumberTypeFloat64),
                                                 KernelAttr()
                                                   .AddInputAttr(kNumberTypeInt32)
                                                   .AddInputAttr(kNumberTypeInt32)
                                                   .AddInputAttr(kNumberTypeInt32)
                                                   .AddInputAttr(kNumberTypeInt32)
                                                   .AddInputAttr(kNumberTypeComplex64)
                                                   .AddOutputAttr(kNumberTypeComplex64),
                                                 KernelAttr()
                                                   .AddInputAttr(kNumberTypeInt32)
                                                   .AddInputAttr(kNumberTypeInt32)
                                                   .AddInputAttr(kNumberTypeInt32)
                                                   .AddInputAttr(kNumberTypeInt32)
                                                   .AddInputAttr(kNumberTypeComplex128)
                                                   .AddOutputAttr(kNumberTypeComplex128),
                                                 KernelAttr()
                                                   .AddInputAttr(kNumberTypeInt64)
                                                   .AddInputAttr(kNumberTypeInt64)
                                                   .AddInputAttr(kNumberTypeInt64)
                                                   .AddInputAttr(kNumberTypeInt64)
                                                   .AddInputAttr(kNumberTypeFloat32)
                                                   .AddOutputAttr(kNumberTypeFloat32),
                                                 KernelAttr()
                                                   .AddInputAttr(kNumberTypeInt64)
                                                   .AddInputAttr(kNumberTypeInt64)
                                                   .AddInputAttr(kNumberTypeInt64)
                                                   .AddInputAttr(kNumberTypeInt64)
                                                   .AddInputAttr(kNumberTypeFloat64)
                                                   .AddOutputAttr(kNumberTypeFloat64),
                                                 KernelAttr()
                                                   .AddInputAttr(kNumberTypeInt64)
                                                   .AddInputAttr(kNumberTypeInt64)
                                                   .AddInputAttr(kNumberTypeInt64)
                                                   .AddInputAttr(kNumberTypeInt64)
                                                   .AddInputAttr(kNumberTypeComplex64)
                                                   .AddOutputAttr(kNumberTypeComplex64),
                                                 KernelAttr()
                                                   .AddInputAttr(kNumberTypeInt64)
                                                   .AddInputAttr(kNumberTypeInt64)
                                                   .AddInputAttr(kNumberTypeInt64)
                                                   .AddInputAttr(kNumberTypeInt64)
                                                   .AddInputAttr(kNumberTypeComplex128)
                                                   .AddOutputAttr(kNumberTypeComplex128)};
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, CSRSparseMatrixToDense, CSRSparseMatrixToDenseCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
