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

#include "ops/sparse_matrix_transpose.h"
#include "plugin/device/cpu/kernel/sparse_matrix_transpose_cpu_kernel.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kTwo = 2;
constexpr size_t kInputsNum = 5;
constexpr size_t kOutputsNum = 5;
constexpr size_t kInputIndex0 = 0;
constexpr size_t kInputIndex1 = 1;
constexpr size_t kInputIndex2 = 2;
constexpr size_t kInputIndex3 = 3;
constexpr size_t kInputIndex4 = 4;
constexpr size_t kOutputIndex0 = 0;
constexpr size_t kOutputIndex1 = 1;
constexpr size_t kOutputIndex2 = 2;
constexpr size_t kOutputIndex3 = 3;
constexpr size_t kOutputIndex4 = 4;
constexpr size_t kDenseShape0 = 0;
constexpr size_t kDenseShape1 = 1;
constexpr size_t kDenseShape2 = 2;
constexpr size_t kRankWithOutBatch = 2;
constexpr size_t kRankWithBatch = 3;
KernelAttr AddKernel(const TypeId &ms_type1, const TypeId &ms_type2, const TypeId &ms_type3, const TypeId &ms_type4,
                     const TypeId &ms_type5, const TypeId &ms_type6, const TypeId &ms_type7, const TypeId &ms_type8,
                     const TypeId &ms_type9, const TypeId &ms_type10) {
  auto kernel = KernelAttr()
                  .AddInputAttr(ms_type1)
                  .AddInputAttr(ms_type2)
                  .AddInputAttr(ms_type3)
                  .AddInputAttr(ms_type4)
                  .AddInputAttr(ms_type5)
                  .AddOutputAttr(ms_type6)
                  .AddOutputAttr(ms_type7)
                  .AddOutputAttr(ms_type8)
                  .AddOutputAttr(ms_type9)
                  .AddOutputAttr(ms_type10);
  return kernel;
}

#define ADD_KERNEL(t1, t2, t3, t4, t5, t6, t7, t8, t9, t10)                                                       \
  AddKernel(kNumberType##t1, kNumberType##t2, kNumberType##t3, kNumberType##t4, kNumberType##t5, kNumberType##t6, \
            kNumberType##t7, kNumberType##t8, kNumberType##t9, kNumberType##t10)

#define SPARSE_MATRIX_TRANSPOSE_COMPUTE_CASE(DTYPE, VTYPEONE, VTYPETWO, inputs, outputs) \
  case DTYPE:                                                                            \
    LaunchKernel<VTYPEONE, VTYPETWO>(inputs, outputs);                                   \
    break;

#define SPARSE_MATRIX_TRANSPOSE_COMPUTE_COMPLEX_CASE(DTYPE, VTYPEONE, VTYPETWO, inputs, outputs) \
  case DTYPE:                                                                                    \
    LaunchcomplexKernel<VTYPEONE, VTYPETWO>(inputs, outputs);                                    \
    break;

#define BATCH_CHECK(batch, batch_pointers, kernel_name_)                                                   \
  do {                                                                                                     \
    if (batch + 1 != batch_pointers) {                                                                     \
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the input of batch pionters shape should equals" \
                        << "dense shape[0] + 1 to match the CSR form input when input has batch.";         \
    }                                                                                                      \
  } while (0);
}  // namespace

bool SparseMatrixTransposeCpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                             const std::vector<KernelTensorPtr> &inputs,
                                             const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kOutputsNum, kernel_name_);
  auto kernel_ptr = std::dynamic_pointer_cast<ops::SparseMatrixTranspose>(base_operator);
  MS_ERROR_IF_NULL_W_RET_VAL(kernel_ptr, false);
  conjugate = kernel_ptr->get_conjugate();
  indiceT_ = inputs.at(kInputIndex0)->GetDtype();
  valueT_ = inputs.at(kInputIndex4)->GetDtype();
  return true;
}

int SparseMatrixTransposeCpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                              const std::vector<KernelTensorPtr> &inputs,
                                              const std::vector<KernelTensorPtr> &outputs,
                                              const std::map<uint32_t, tensor::TensorPtr> &) {
  if (auto ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  auto input_dense_shape = inputs.at(kIndex0)->GetShapeVector();
  rank_x_ = static_cast<size_t>(input_dense_shape[0]);
  if (rank_x_ != kRankWithOutBatch && rank_x_ != kRankWithBatch) {
    MS_LOG(EXCEPTION) << "For SparseMatrixTranspose,the rank must be 2 or 3, but got" << rank_x_ << "!";
  }
  std::vector<int64_t> x_batch_pointers_shape = inputs.at(kInputIndex1)->GetShapeVector();
  std::vector<int64_t> x_row_pointers_shape = inputs.at(kInputIndex2)->GetShapeVector();
  std::vector<int64_t> x_col_indices_shape = inputs.at(kInputIndex3)->GetShapeVector();
  std::vector<int64_t> x_value_shape = inputs.at(kInputIndex4)->GetShapeVector();
  x_value_size_ = static_cast<size_t>(x_value_shape[0]);
  x_batch_pointers_size_ = static_cast<size_t>(x_batch_pointers_shape[0]);
  x_col_indice_size_ = static_cast<size_t>(x_col_indices_shape[0]);
  x_row_pointer_size_ = static_cast<size_t>(x_row_pointers_shape[0]);
  if (x_col_indice_size_ != x_value_size_) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the input of col indice shape should equals "
                      << "values shape to match the CSR form input.";
  }
  return KRET_OK;
}

bool SparseMatrixTransposeCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                               const std::vector<kernel::AddressPtr> &,
                                               const std::vector<kernel::AddressPtr> &outputs) {
  switch (indiceT_) {
    case kNumberTypeInt32:
      switch (valueT_) {
        SPARSE_MATRIX_TRANSPOSE_COMPUTE_CASE(kNumberTypeInt8, int32_t, int8_t, inputs, outputs)
        SPARSE_MATRIX_TRANSPOSE_COMPUTE_CASE(kNumberTypeUInt8, int32_t, uint8_t, inputs, outputs)
        SPARSE_MATRIX_TRANSPOSE_COMPUTE_CASE(kNumberTypeInt16, int32_t, int16_t, inputs, outputs)
        SPARSE_MATRIX_TRANSPOSE_COMPUTE_CASE(kNumberTypeUInt16, int32_t, uint16_t, inputs, outputs)
        SPARSE_MATRIX_TRANSPOSE_COMPUTE_CASE(kNumberTypeInt32, int32_t, int32_t, inputs, outputs)
        SPARSE_MATRIX_TRANSPOSE_COMPUTE_CASE(kNumberTypeUInt32, int32_t, uint32_t, inputs, outputs)
        SPARSE_MATRIX_TRANSPOSE_COMPUTE_CASE(kNumberTypeInt64, int32_t, int64_t, inputs, outputs)
        SPARSE_MATRIX_TRANSPOSE_COMPUTE_CASE(kNumberTypeUInt64, int32_t, uint64_t, inputs, outputs)
        SPARSE_MATRIX_TRANSPOSE_COMPUTE_CASE(kNumberTypeFloat16, int32_t, float16, inputs, outputs)
        SPARSE_MATRIX_TRANSPOSE_COMPUTE_CASE(kNumberTypeFloat32, int32_t, float, inputs, outputs)
        SPARSE_MATRIX_TRANSPOSE_COMPUTE_CASE(kNumberTypeFloat64, int32_t, double, inputs, outputs)
        SPARSE_MATRIX_TRANSPOSE_COMPUTE_COMPLEX_CASE(kNumberTypeComplex64, int32_t, complex64, inputs, outputs)
        SPARSE_MATRIX_TRANSPOSE_COMPUTE_COMPLEX_CASE(kNumberTypeComplex128, int32_t, complex128, inputs, outputs)
        default:
          MS_LOG(EXCEPTION) << "data type of values is not included.";
      }
      break;
    case kNumberTypeInt64:
      switch (valueT_) {
        SPARSE_MATRIX_TRANSPOSE_COMPUTE_CASE(kNumberTypeInt8, int64_t, int8_t, inputs, outputs)
        SPARSE_MATRIX_TRANSPOSE_COMPUTE_CASE(kNumberTypeUInt8, int64_t, uint8_t, inputs, outputs)
        SPARSE_MATRIX_TRANSPOSE_COMPUTE_CASE(kNumberTypeInt16, int64_t, int16_t, inputs, outputs)
        SPARSE_MATRIX_TRANSPOSE_COMPUTE_CASE(kNumberTypeUInt16, int64_t, uint16_t, inputs, outputs)
        SPARSE_MATRIX_TRANSPOSE_COMPUTE_CASE(kNumberTypeInt32, int64_t, int32_t, inputs, outputs)
        SPARSE_MATRIX_TRANSPOSE_COMPUTE_CASE(kNumberTypeUInt32, int64_t, uint32_t, inputs, outputs)
        SPARSE_MATRIX_TRANSPOSE_COMPUTE_CASE(kNumberTypeInt64, int64_t, int64_t, inputs, outputs)
        SPARSE_MATRIX_TRANSPOSE_COMPUTE_CASE(kNumberTypeUInt64, int64_t, uint64_t, inputs, outputs)
        SPARSE_MATRIX_TRANSPOSE_COMPUTE_CASE(kNumberTypeFloat16, int64_t, float16, inputs, outputs)
        SPARSE_MATRIX_TRANSPOSE_COMPUTE_CASE(kNumberTypeFloat32, int64_t, float, inputs, outputs)
        SPARSE_MATRIX_TRANSPOSE_COMPUTE_CASE(kNumberTypeFloat64, int64_t, double, inputs, outputs)
        SPARSE_MATRIX_TRANSPOSE_COMPUTE_COMPLEX_CASE(kNumberTypeComplex64, int64_t, complex64, inputs, outputs)
        SPARSE_MATRIX_TRANSPOSE_COMPUTE_COMPLEX_CASE(kNumberTypeComplex128, int64_t, complex128, inputs, outputs)
        default:
          MS_LOG(EXCEPTION) << "data type of values is not included.";
      }
      break;
    default:
      MS_LOG(EXCEPTION) << "The data type of dense_shape, batch_pointers, "
                        << "row_pointers, col_indices is not int32 or int64.";
  }
  return true;
}

template <typename indiceT, typename valueT>
void SparseMatrixTransposeCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                                     const std::vector<kernel::AddressPtr> &outputs) {
  indiceT *x_dense_shape = static_cast<indiceT *>(inputs[kInputIndex0]->addr);
  indiceT *x_batch_pointers = static_cast<indiceT *>(inputs[kInputIndex1]->addr);
  indiceT *x_row_pointers = static_cast<indiceT *>(inputs[kInputIndex2]->addr);
  indiceT *x_col_indices = static_cast<indiceT *>(inputs[kInputIndex3]->addr);
  valueT *x_values = static_cast<valueT *>(inputs[kInputIndex4]->addr);
  indiceT *y_dense_shape_addr = static_cast<indiceT *>(outputs[kOutputIndex0]->addr);
  indiceT *y_batch_pointers_addr = static_cast<indiceT *>(outputs[kOutputIndex1]->addr);
  indiceT *y_row_pointers_addr = static_cast<indiceT *>(outputs[kOutputIndex2]->addr);
  indiceT *y_col_indices_addr = static_cast<indiceT *>(outputs[kOutputIndex3]->addr);
  valueT *y_values_addr = static_cast<valueT *>(outputs[kOutputIndex4]->addr);
  size_t batch_pointers = x_batch_pointers_size_;
  if (rank_x_ == kRankWithBatch) {
    y_dense_shape_addr[kDenseShape0] = x_dense_shape[kDenseShape0];
    y_dense_shape_addr[kDenseShape1] = x_dense_shape[kDenseShape2];
    y_dense_shape_addr[kDenseShape2] = x_dense_shape[kDenseShape1];
    size_t batch = static_cast<size_t>(x_dense_shape[kDenseShape0]);
    BATCH_CHECK(batch, batch_pointers, kernel_name_)
  } else {
    y_dense_shape_addr[kDenseShape0] = x_dense_shape[kDenseShape1];
    y_dense_shape_addr[kDenseShape1] = x_dense_shape[kDenseShape0];
    if (batch_pointers != kTwo) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the input of batch pionters shape should equals"
                        << "2 to match the CSR form input when input has no batch.";
    }
  }
  for (size_t i = 0; i < batch_pointers; ++i) {
    y_batch_pointers_addr[i] = x_batch_pointers[i];
  }
  size_t num_rows = static_cast<size_t>(x_dense_shape[rank_x_ - kTwo]);
  size_t num_cols = static_cast<size_t>(x_dense_shape[rank_x_ - 1]);
  size_t num_batch = batch_pointers - static_cast<size_t>(1);
  std::vector<indiceT> y_part_row_pointers(num_cols + 1);
  std::vector<indiceT> part_row_pointers(num_rows + 1);
  if (x_row_pointer_size_ != num_batch * (num_rows + 1)) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the input of row pionters shape should equals"
                      << " batch*(rows + 1 ) to match the CSR form input when input has no batch.";
  }
  for (size_t j = 0; j < num_batch; ++j) {
    size_t n = static_cast<size_t>(x_batch_pointers[j + 1] - x_batch_pointers[j]);
    std::vector<valueT> part_values(n);
    std::vector<indiceT> part_col_indices(n);
    std::vector<indiceT> y_part_col_indices(n);
    std::vector<valueT> y_part_values(n);
    for (size_t i = 0; i < num_cols + 1; ++i) {
      y_part_row_pointers[i] = static_cast<indiceT>(0);
    }
    for (size_t k = 0; k < num_rows + 1; ++k) {
      part_row_pointers[k] = x_row_pointers[(num_rows + 1) * j + k];
    }
    for (size_t k = 0; k < n; ++k) {
      part_values[k] = x_values[static_cast<size_t>(x_batch_pointers[j]) + k];
      part_col_indices[k] = x_col_indices[static_cast<size_t>(x_batch_pointers[j]) + k];
    }
    for (size_t i = 0; i < n; ++i) {
      y_part_row_pointers[static_cast<size_t>(part_col_indices[i]) + 1] += static_cast<indiceT>(1);
    }
    for (size_t i = 1; i < num_cols + 1; ++i) {
      y_part_row_pointers[i] += y_part_row_pointers[i - 1];
    }
    for (size_t k = 0; k < num_cols + 1; ++k) {
      y_row_pointers_addr[(num_cols + 1) * j + k] = y_part_row_pointers[k];
    }
    std::vector<size_t> current_col_count(num_cols);
    for (size_t row_idx = 0; row_idx < num_rows; ++row_idx) {
      const size_t row_begin = static_cast<size_t>(part_row_pointers[row_idx]);
      const size_t row_end = static_cast<size_t>(part_row_pointers[row_idx + 1]);
      for (size_t i = row_begin; i < row_end; ++i) {
        const size_t col_idx = static_cast<size_t>(part_col_indices[i]);
        const size_t offset = static_cast<size_t>(y_part_row_pointers[col_idx]) + current_col_count[col_idx];
        y_part_col_indices[offset] = static_cast<indiceT>(row_idx);
        y_part_values[offset] = part_values[i];
        current_col_count[col_idx] += static_cast<size_t>(1);
      }
    }
    for (size_t k = 0; k < n; ++k) {
      y_values_addr[static_cast<size_t>(x_batch_pointers[j]) + k] = y_part_values[k];
      y_col_indices_addr[static_cast<size_t>(x_batch_pointers[j]) + k] = y_part_col_indices[k];
    }
  }
}

template <typename indiceT, typename valueT>
void SparseMatrixTransposeCpuKernelMod::LaunchcomplexKernel(const std::vector<kernel::AddressPtr> &inputs,
                                                            const std::vector<kernel::AddressPtr> &outputs) {
  indiceT *x_dense_shape = static_cast<indiceT *>(inputs[kInputIndex0]->addr);
  indiceT *x_batch_pointers = static_cast<indiceT *>(inputs[kInputIndex1]->addr);
  indiceT *x_row_pointers = static_cast<indiceT *>(inputs[kInputIndex2]->addr);
  indiceT *x_col_indices = static_cast<indiceT *>(inputs[kInputIndex3]->addr);
  valueT *x_values = static_cast<valueT *>(inputs[kInputIndex4]->addr);
  indiceT *y_dense_shape_addr = static_cast<indiceT *>(outputs[kOutputIndex0]->addr);
  indiceT *y_batch_pointers_addr = static_cast<indiceT *>(outputs[kOutputIndex1]->addr);
  indiceT *y_row_pointers_addr = static_cast<indiceT *>(outputs[kOutputIndex2]->addr);
  indiceT *y_col_indices_addr = static_cast<indiceT *>(outputs[kOutputIndex3]->addr);
  valueT *y_values_addr = static_cast<valueT *>(outputs[kOutputIndex4]->addr);
  size_t batch_pointers = x_batch_pointers_size_;
  if (rank_x_ == kRankWithBatch) {
    y_dense_shape_addr[kDenseShape0] = x_dense_shape[kDenseShape0];
    y_dense_shape_addr[kDenseShape1] = x_dense_shape[kDenseShape2];
    y_dense_shape_addr[kDenseShape2] = x_dense_shape[kDenseShape1];
    size_t batch = static_cast<size_t>(x_dense_shape[kDenseShape0]);
    BATCH_CHECK(batch, batch_pointers, kernel_name_)
  } else {
    y_dense_shape_addr[kDenseShape0] = x_dense_shape[kDenseShape1];
    y_dense_shape_addr[kDenseShape1] = x_dense_shape[kDenseShape0];
    if (batch_pointers != kTwo) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the input of batch pionters shape should equals"
                        << "2 to match the CSR form input when input has no batch.";
    }
  }
  for (size_t i = 0; i < batch_pointers; ++i) {
    y_batch_pointers_addr[i] = x_batch_pointers[i];
  }
  size_t num_rows = static_cast<size_t>(x_dense_shape[rank_x_ - kTwo]);
  size_t num_cols = static_cast<size_t>(x_dense_shape[rank_x_ - 1]);
  size_t num_batch = batch_pointers - static_cast<size_t>(1);
  std::vector<indiceT> y_part_row_pointers(num_cols + 1);
  std::vector<indiceT> part_row_pointers(num_rows + 1);
  if (x_row_pointer_size_ != num_batch * (num_rows + 1)) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the input of row pionters shape should equals"
                      << " batch*(rows + 1 ) to match the CSR form input when input has no batch.";
  }
  for (size_t j = 0; j < num_batch; ++j) {
    size_t n = static_cast<size_t>(x_batch_pointers[j + 1] - x_batch_pointers[j]);
    std::vector<valueT> part_values(n);
    std::vector<indiceT> part_col_indices(n);
    std::vector<indiceT> y_part_col_indices(n);
    std::vector<valueT> y_part_values(n);
    for (size_t i = 0; i < num_cols + 1; ++i) {
      y_part_row_pointers[i] = static_cast<indiceT>(0);
    }
    for (size_t k = 0; k < num_rows + 1; ++k) {
      part_row_pointers[k] = x_row_pointers[(num_rows + 1) * j + k];
    }
    for (size_t k = 0; k < n; ++k) {
      part_values[k] = x_values[static_cast<size_t>(x_batch_pointers[j]) + k];
      part_col_indices[k] = x_col_indices[static_cast<size_t>(x_batch_pointers[j]) + k];
    }
    for (size_t i = 0; i < n; ++i) {
      y_part_row_pointers[static_cast<size_t>(part_col_indices[i]) + 1] += static_cast<indiceT>(1);
    }
    for (size_t i = 1; i < num_cols + 1; ++i) {
      y_part_row_pointers[i] += y_part_row_pointers[i - 1];
    }
    for (size_t k = 0; k < num_cols + 1; ++k) {
      y_row_pointers_addr[(num_cols + 1) * j + k] = y_part_row_pointers[k];
    }
    std::vector<size_t> current_col_count(num_cols);
    for (size_t row_idx = 0; row_idx < num_rows; ++row_idx) {
      const size_t row_begin = static_cast<size_t>(part_row_pointers[row_idx]);
      const size_t row_end = static_cast<size_t>(part_row_pointers[row_idx + 1]);
      for (size_t i = row_begin; i < row_end; ++i) {
        const size_t col_idx = static_cast<size_t>(part_col_indices[i]);
        const size_t offset = static_cast<size_t>(y_part_row_pointers[col_idx]) + current_col_count[col_idx];
        y_part_col_indices[offset] = static_cast<indiceT>(row_idx);
        y_part_values[offset] = part_values[i];
        current_col_count[col_idx] += static_cast<size_t>(1);
      }
    }
    for (size_t k = 0; k < n; ++k) {
      y_values_addr[static_cast<size_t>(x_batch_pointers[j]) + k] = y_part_values[k];
      y_col_indices_addr[static_cast<size_t>(x_batch_pointers[j]) + k] = y_part_col_indices[k];
    }
  }
  if (conjugate == true) {
    for (size_t i = 0; i < x_value_size_; ++i) {
      y_values_addr[i] = std::conj(y_values_addr[i]);
    }
  }
}

std::vector<KernelAttr> SparseMatrixTransposeCpuKernelMod::GetOpSupport() {
  static std::vector<KernelAttr> kernel_attr_list = {
    ADD_KERNEL(Int32, Int32, Int32, Int32, Int8, Int32, Int32, Int32, Int32, Int8),
    ADD_KERNEL(Int32, Int32, Int32, Int32, UInt8, Int32, Int32, Int32, Int32, UInt8),
    ADD_KERNEL(Int32, Int32, Int32, Int32, Int16, Int32, Int32, Int32, Int32, Int16),
    ADD_KERNEL(Int32, Int32, Int32, Int32, UInt16, Int32, Int32, Int32, Int32, UInt16),
    ADD_KERNEL(Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32),
    ADD_KERNEL(Int32, Int32, Int32, Int32, Int64, Int32, Int32, Int32, Int32, Int64),
    ADD_KERNEL(Int32, Int32, Int32, Int32, UInt32, Int32, Int32, Int32, Int32, UInt32),
    ADD_KERNEL(Int32, Int32, Int32, Int32, UInt64, Int32, Int32, Int32, Int32, UInt64),
    ADD_KERNEL(Int32, Int32, Int32, Int32, Float16, Int32, Int32, Int32, Int32, Float16),
    ADD_KERNEL(Int32, Int32, Int32, Int32, Float32, Int32, Int32, Int32, Int32, Float32),
    ADD_KERNEL(Int32, Int32, Int32, Int32, Float64, Int32, Int32, Int32, Int32, Float64),
    ADD_KERNEL(Int32, Int32, Int32, Int32, Complex64, Int32, Int32, Int32, Int32, Complex64),
    ADD_KERNEL(Int32, Int32, Int32, Int32, Complex128, Int32, Int32, Int32, Int32, Complex128),
    ADD_KERNEL(Int64, Int64, Int64, Int64, Int8, Int64, Int64, Int64, Int64, Int8),
    ADD_KERNEL(Int64, Int64, Int64, Int64, UInt8, Int64, Int64, Int64, Int64, UInt8),
    ADD_KERNEL(Int64, Int64, Int64, Int64, Int16, Int64, Int64, Int64, Int64, Int16),
    ADD_KERNEL(Int64, Int64, Int64, Int64, UInt16, Int64, Int64, Int64, Int64, UInt16),
    ADD_KERNEL(Int64, Int64, Int64, Int64, Int32, Int64, Int64, Int64, Int64, Int32),
    ADD_KERNEL(Int64, Int64, Int64, Int64, Int64, Int64, Int64, Int64, Int64, Int64),
    ADD_KERNEL(Int64, Int64, Int64, Int64, UInt32, Int64, Int64, Int64, Int64, UInt32),
    ADD_KERNEL(Int64, Int64, Int64, Int64, UInt64, Int64, Int64, Int64, Int64, UInt64),
    ADD_KERNEL(Int64, Int64, Int64, Int64, Float16, Int64, Int64, Int64, Int64, Float16),
    ADD_KERNEL(Int64, Int64, Int64, Int64, Float32, Int64, Int64, Int64, Int64, Float32),
    ADD_KERNEL(Int64, Int64, Int64, Int64, Float64, Int64, Int64, Int64, Int64, Float64),
    ADD_KERNEL(Int64, Int64, Int64, Int64, Complex64, Int64, Int64, Int64, Int64, Complex64),
    ADD_KERNEL(Int64, Int64, Int64, Int64, Complex128, Int64, Int64, Int64, Int64, Complex128)};

  return kernel_attr_list;
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, SparseMatrixTranspose, SparseMatrixTransposeCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
