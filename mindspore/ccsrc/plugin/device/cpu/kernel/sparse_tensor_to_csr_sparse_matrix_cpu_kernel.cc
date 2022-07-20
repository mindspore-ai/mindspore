/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/kernel/sparse_tensor_to_csr_sparse_matrix_cpu_kernel.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr int64_t kRankWithoutBatch = 2;
constexpr int64_t kRankWithBatch = 3;
constexpr int64_t kZero = 0;
constexpr int64_t kOne = 1;
constexpr int64_t kTwo = 2;
constexpr int64_t kSparseTensorToCSRSparseMatrixInputsNum = 3;
constexpr int64_t kSparseTensorToCSRSparseMatrixOutputsNum = 5;
constexpr int64_t kInputIndex0 = 0;
constexpr int64_t kInputIndex1 = 1;
constexpr int64_t kInputIndex2 = 2;
constexpr int64_t kOutputIndex0 = 0;
constexpr int64_t kOutputIndex1 = 1;
constexpr int64_t kOutputIndex2 = 2;
constexpr int64_t kOutputIndex3 = 3;
constexpr int64_t kOutputIndex4 = 4;
constexpr int64_t kInitPrevBatch = -1;
constexpr char kKernelName[] = "SparseTensorToCSRSparseMatrix";

#define ADD_KERNEL(t1, t2, t3, t4, t5, t6, t7, t8) \
  KernelAttr()                                     \
    .AddInputAttr(kNumberType##t1)                 \
    .AddInputAttr(kNumberType##t2)                 \
    .AddInputAttr(kNumberType##t3)                 \
    .AddOutputAttr(kNumberType##t4)                \
    .AddOutputAttr(kNumberType##t5)                \
    .AddOutputAttr(kNumberType##t6)                \
    .AddOutputAttr(kNumberType##t7)                \
    .AddOutputAttr(kNumberType##t8)
}  // namespace

void SparseTensorToCSRSparseMatrixCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  indice_type_ = AnfAlgo::GetInputDeviceDataType(kernel_node, kInputIndex0);
  value_type_ = AnfAlgo::GetInputDeviceDataType(kernel_node, kInputIndex1);
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  node_wpt_ = kernel_node;
  auto x_indices_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, kZero);
  total_nnz_ = x_indices_shape[0];
  auto input_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, kInputIndex2);
  rank_ = input_shape[0];
  int64_t input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
  CHECK_KERNEL_INPUTS_NUM(input_num, kSparseTensorToCSRSparseMatrixInputsNum, kKernelName);
  int64_t output_num = common::AnfAlgo::GetOutputTensorNum(kernel_node);
  CHECK_KERNEL_OUTPUTS_NUM(output_num, kSparseTensorToCSRSparseMatrixOutputsNum, kKernelName);
  if (rank_ != kRankWithoutBatch && rank_ != kRankWithBatch) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the input x_dense_shape should "
                      << "have rank 2 or 3, but got " << rank_ << ".";
  }
}

bool SparseTensorToCSRSparseMatrixCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs,
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
  auto node_ = node_wpt_.lock();
  if (!node_) {
    MS_LOG(EXCEPTION) << "node_wpt_ is expired.";
  }
  int64_t output_nm = common::AnfAlgo::GetOutputTensorNum(node_);
  std::vector<TypeId> dtypes(output_nm);
  for (int64_t i = 0; i < output_nm; i++) {
    dtypes[i] = AnfAlgo::GetOutputDeviceDataType(node_, i);
  }
  std::vector<int64_t> dense_shape_dims{rank_};
  std::vector<int64_t> batch_pointers_dims{batch_size_ + kOne};
  std::vector<int64_t> row_pointers_dims{batch_size_ * (num_rows_ + kOne)};
  std::vector<int64_t> col_indices_dims{total_nnz_};
  std::vector<int64_t> values_dims{total_nnz_};
  std::vector<std::vector<int64_t>> shapes{dense_shape_dims, batch_pointers_dims, row_pointers_dims, col_indices_dims,
                                           values_dims};
  common::AnfAlgo::SetOutputInferTypeAndShape(dtypes, shapes, node_.get());
  return true;
}

template <typename indiceT, typename valueT>
void SparseTensorToCSRSparseMatrixCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                                             const std::vector<kernel::AddressPtr> &outputs) {
  const int64_t shift = (rank_ == kRankWithoutBatch) ? kZero : kOne;
  num_rows_ = *(static_cast<indiceT *>(inputs[kInputIndex2]->addr) + shift);
  indiceT *x_indices = static_cast<indiceT *>(inputs[kInputIndex0]->addr);
  valueT *x_values = static_cast<valueT *>(inputs[kInputIndex1]->addr);
  indiceT *x_dense_shape = static_cast<indiceT *>(inputs[kInputIndex2]->addr);
  batch_size_ = (rank_ == kRankWithoutBatch) ? kOne : x_dense_shape[kZero];
  indiceT *y_dense_shape_addr = static_cast<indiceT *>(outputs[kOutputIndex0]->addr);
  indiceT *y_batch_pointers_addr = static_cast<indiceT *>(outputs[kOutputIndex1]->addr);
  indiceT *y_row_pointers_addr = static_cast<indiceT *>(outputs[kOutputIndex2]->addr);
  indiceT *y_col_indices_addr = static_cast<indiceT *>(outputs[kOutputIndex3]->addr);
  valueT *y_values_addr = static_cast<valueT *>(outputs[kOutputIndex4]->addr);

  for (int64_t i = kZero; i < rank_; i++) {
    y_dense_shape_addr[i] = x_dense_shape[i];
  }

  for (int64_t i = kZero; i < total_nnz_; i++) {
    y_values_addr[i] = x_values[i];
  }

  for (int64_t i = kZero; i < batch_size_ * (num_rows_ + 1); i++) {
    y_row_pointers_addr[i] = indiceT(kZero);
  }

  int64_t prev_batch = kInitPrevBatch;
  if (rank_ == kRankWithoutBatch) {
    y_batch_pointers_addr[kZero] = indiceT(kZero);
    ++prev_batch;
    for (int64_t i = kZero; i < total_nnz_; ++i) {
      y_row_pointers_addr[x_indices[i * rank_] + kOne] += indiceT(kOne);
      y_col_indices_addr[i] = x_indices[i * rank_ + kOne];
    }
  } else {
    for (int64_t i = kZero; i < total_nnz_; ++i) {
      int64_t cur_batch = x_indices[i * rank_];
      y_row_pointers_addr[cur_batch * (num_rows_ + kOne) + x_indices[i * rank_ + kOne] + kOne] += kOne;
      y_col_indices_addr[i] = x_indices[i * rank_ + kTwo];
      while (prev_batch < SizeToLong(cur_batch)) {
        y_batch_pointers_addr[prev_batch + kOne] = indiceT(i);
        ++prev_batch;
      }
    }
  }
  while (prev_batch < SizeToLong(batch_size_)) {
    y_batch_pointers_addr[prev_batch + kOne] = total_nnz_;
    ++prev_batch;
  }
  for (int64_t batch_idx = 0; batch_idx < batch_size_; ++batch_idx) {
    auto *row_ptr_batch = y_row_pointers_addr + batch_idx * (num_rows_ + kOne);
    std::partial_sum(row_ptr_batch, row_ptr_batch + num_rows_ + kOne, row_ptr_batch);
  }
}
std::vector<KernelAttr> SparseTensorToCSRSparseMatrixCpuKernelMod::GetOpSupport() {
  static std::vector<KernelAttr> kernel_attr_list = {
    ADD_KERNEL(Int32, Float32, Int32, Int32, Int32, Int32, Int32, Float32),
    ADD_KERNEL(Int32, Float64, Int32, Int32, Int32, Int32, Int32, Float64),
    ADD_KERNEL(Int32, Complex64, Int32, Int32, Int32, Int32, Int32, Complex64),
    ADD_KERNEL(Int32, Complex128, Int32, Int32, Int32, Int32, Int32, Complex128),
    ADD_KERNEL(Int64, Float32, Int64, Int64, Int64, Int64, Int64, Float32),
    ADD_KERNEL(Int64, Float64, Int64, Int64, Int64, Int64, Int64, Float64),
    ADD_KERNEL(Int64, Complex64, Int64, Int64, Int64, Int64, Int64, Complex64),
    ADD_KERNEL(Int64, Complex128, Int64, Int64, Int64, Int64, Int64, Complex128)};
  return kernel_attr_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, SparseTensorToCSRSparseMatrix, SparseTensorToCSRSparseMatrixCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
