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
#include <utility>
#include "mindspore/core/ops/op_name.h"
#include "plugin/device/cpu/kernel/eigen/eigen_common_utils.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "plugin/device/cpu/kernel/eigen/sparse_matrix_mat_mul_cpu_kernel.h"
#include "utils/ms_utils.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kInputsNum = 6;
constexpr size_t kOutputsNum = 1;
constexpr size_t kOutputIndex0 = 0;
constexpr char kTransposeX1[] = "transpose_x1";
constexpr char kTransposeX2[] = "transpose_x2";
constexpr char kAdjointX1[] = "adjoint_x1";
constexpr char kAdjointX2[] = "adjoint_x2";
constexpr char kTransposeOutput[] = "transpose_output";
constexpr char kConjugateOutput[] = "conjugate_output";

#define ADD_KERNEL(indiceT, valueT)     \
  KernelAttr()                          \
    .AddInputAttr(kNumberType##indiceT) \
    .AddInputAttr(kNumberType##indiceT) \
    .AddInputAttr(kNumberType##indiceT) \
    .AddInputAttr(kNumberType##indiceT) \
    .AddInputAttr(kNumberType##valueT)  \
    .AddInputAttr(kNumberType##valueT)  \
    .AddOutputAttr(kNumberType##valueT)
}  // namespace
bool SparseMatrixMatMulCpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                          const std::vector<KernelTensorPtr> &inputs,
                                          const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();

  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kOutputsNum, kernel_name_);

  PrimitivePtr primitive = base_operator->GetPrim();
  MS_ERROR_IF_NULL(primitive);

  auto transpose_x1_ptr = primitive->GetAttr(kTransposeX1);
  MS_EXCEPTION_IF_NULL(transpose_x1_ptr);
  transpose_x1_ = GetValue<bool>(transpose_x1_ptr);

  auto transpose_x2_ptr = primitive->GetAttr(kTransposeX2);
  MS_EXCEPTION_IF_NULL(transpose_x2_ptr);
  transpose_x2_ = GetValue<bool>(transpose_x2_ptr);

  auto adjoint_x1_ptr = primitive->GetAttr(kAdjointX1);
  MS_EXCEPTION_IF_NULL(adjoint_x1_ptr);
  adjoint_x1_ = GetValue<bool>(adjoint_x1_ptr);

  auto adjoint_x2_ptr = primitive->GetAttr(kAdjointX2);
  MS_EXCEPTION_IF_NULL(adjoint_x2_ptr);
  adjoint_x2_ = GetValue<bool>(adjoint_x2_ptr);

  auto transpose_output_ptr = primitive->GetAttr(kTransposeOutput);
  MS_EXCEPTION_IF_NULL(transpose_output_ptr);
  transpose_output_ = GetValue<bool>(transpose_output_ptr);

  auto conjugate_output_ptr = primitive->GetAttr(kConjugateOutput);
  MS_EXCEPTION_IF_NULL(conjugate_output_ptr);
  conjugate_output_ = GetValue<bool>(conjugate_output_ptr);

  indice_type_ = inputs.at(kIndex0)->GetDtype();
  value_type_ = inputs.at(kIndex4)->GetDtype();

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(EXCEPTION) << "SparseMatrixMatMul does not support this kernel data type: " << kernel_attr;
  }
  kernel_func_ = func_list_[index].second;

  return true;
}

int SparseMatrixMatMulCpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                           const std::vector<KernelTensorPtr> &inputs,
                                           const std::vector<KernelTensorPtr> &outputs,
                                           const std::map<uint32_t, tensor::TensorPtr> &) {
  if (auto ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
    return ret;
  }

  auto input_shape1 = inputs.at(kIndex0)->GetShapeVector();
  rank_ = input_shape1[0];

  input_shape2_ = Convert2SizeT(inputs.at(kIndex5)->GetShapeVector());

  auto x1_batch_pointer_shape = inputs.at(kIndex1)->GetShapeVector();
  batch_size_ = static_cast<size_t>(x1_batch_pointer_shape[0]) - 1;

  return KRET_OK;
}

template <typename indiceT, typename valueT>
bool SparseMatrixMatMulCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                                  const std::vector<AddressPtr> &,
                                                  const std::vector<kernel::AddressPtr> &outputs) {
  CheckMatMul<indiceT>(inputs);
  using Matrix = Eigen::Matrix<valueT, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  indiceT batch_size = batch_size_;
  std::vector<Matrix> results(batch_size);
  int shift = (rank_ == 2) ? 0 : 1;

  indiceT row_x1 = *(static_cast<indiceT *>(inputs[kIndex0]->addr) + shift);
  indiceT col_x1 = *(static_cast<indiceT *>(inputs[kIndex0]->addr) + shift + 1);
  indiceT *batch_pointers_x1 = static_cast<indiceT *>(inputs[kIndex1]->addr);
  indiceT *row_pointers_x1 = static_cast<indiceT *>(inputs[kIndex2]->addr);
  indiceT *col_indices_x1 = static_cast<indiceT *>(inputs[kIndex3]->addr);
  valueT *value_x1 = static_cast<valueT *>(inputs[kIndex4]->addr);

  std::vector<size_t> shape_x2 = input_shape2_;
  const int row_dim = (rank_ == 2) ? 0 : 1;
  indiceT row_x2 = shape_x2[row_dim];
  indiceT col_x2 = shape_x2[row_dim + 1];
  valueT *value_x2 = static_cast<valueT *>(inputs[kIndex5]->addr);

  bool transpose_x1 = transpose_x1_;
  bool transpose_x2 = transpose_x2_;
  bool adjoint_x1 = adjoint_x1_;
  bool adjoint_x2 = adjoint_x2_;
  bool transpose_output = transpose_output_;
  bool conjugate_output = conjugate_output_;

  for (int64_t i = 0; i < batch_size; i++) {
    int64_t nnz_x1 = batch_pointers_x1[i + 1] - batch_pointers_x1[i];
    indiceT *row_pointers_x1_batch_i = row_pointers_x1 + (row_x1 + 1) * i;
    indiceT *col_indices_x1_batch_i = col_indices_x1 + batch_pointers_x1[i];
    valueT *value_x1_batch_i = value_x1 + batch_pointers_x1[i];
    auto x1_sparse_matrix =
      CreateEigenSparseMatrix<indiceT, valueT>(row_x1, col_x1, nnz_x1, row_pointers_x1_batch_i, col_indices_x1_batch_i,
                                               value_x1_batch_i, transpose_x1, adjoint_x1);

    Eigen::Map<Matrix> x2_dense_matrix(value_x2 + col_x2 * row_x2 * i, row_x2, col_x2);
    Matrix temp;
    if (transpose_x2) {
      temp = x1_sparse_matrix * x2_dense_matrix.transpose();
    } else if (adjoint_x2) {
      temp = x1_sparse_matrix * x2_dense_matrix.adjoint();
    } else {
      temp = x1_sparse_matrix * x2_dense_matrix;
    }

    if (transpose_output) {
      results[i] = temp.transpose();
    } else {
      results[i] = temp;
    }

    if (conjugate_output) {
      results[i] = results[i].conjugate();
    }
  }

  // computer result_row_pointers|result_col_indices|result_values data
  indiceT row_output, col_output;
  row_output = results[0].rows();
  col_output = results[0].cols();
  for (int i = 0; i < batch_size; i++) {
    valueT *output_values_data = static_cast<valueT *>(outputs[kOutputIndex0]->addr);
    std::copy(results[i].data(), results[i].data() + row_output * col_output,
              output_values_data + i * row_output * col_output);
  }
  return true;
}

template <typename T>
bool SparseMatrixMatMulCpuKernelMod::CheckMatMul(const std::vector<AddressPtr> &inputs) {
  const int row_dim = (rank_ == 2) ? 0 : 1;
  T *shape_x1 = static_cast<T *>(inputs[kIndex0]->addr);
  std::vector<size_t> shape_x2 = input_shape2_;

  T x1_col = (transpose_x1_ || adjoint_x1_) ? shape_x1[row_dim] : shape_x1[row_dim + 1];
  T x2_row = (transpose_x2_ || adjoint_x2_) ? shape_x2[row_dim + 1] : shape_x2[row_dim];
  if (x1_col != x2_row) {
    MS_EXCEPTION(ValueError) << "For SparseMatrixMatMul, x1's col and x2's row must be the same, but got x1_col = "
                             << x1_col << ", and x2_row = " << x2_row << ".";
  }
  return true;
}

template <typename indiceT, typename valueT>
using SparseMatrixPtr = Eigen::Ref<const Eigen::SparseMatrix<valueT, Eigen::RowMajor, indiceT>>;

template <typename indiceT, typename valueT>
SparseMatrixPtr<indiceT, valueT> SparseMatrixMatMulCpuKernelMod::CreateEigenSparseMatrix(
  indiceT rows, indiceT cols, int64_t nnz, indiceT *row_pointers, indiceT *col_indices, valueT *values, bool transpose,
  bool adjoint) {
  Eigen::Map<const Eigen::SparseMatrix<valueT, Eigen::RowMajor, indiceT>> sparse_matrix(rows, cols, nnz, row_pointers,
                                                                                        col_indices, values);
  // The transpose/adjoint expressions are not actually evaluated until
  // necessary. Hence we don't create copies or modify the input matrix
  // inplace.
  if (transpose) {
    return sparse_matrix.transpose();
  }
  if (adjoint) {
    return sparse_matrix.adjoint();
  }
  return sparse_matrix;
}

std::vector<std::pair<KernelAttr, SparseMatrixMatMulCpuKernelMod::SparseMatrixMatMulFunc>>
  SparseMatrixMatMulCpuKernelMod::func_list_ = {
    {ADD_KERNEL(Int32, Float32), &SparseMatrixMatMulCpuKernelMod::LaunchKernel<int32_t, float>},
    {ADD_KERNEL(Int32, Float64), &SparseMatrixMatMulCpuKernelMod::LaunchKernel<int32_t, double>},
    {ADD_KERNEL(Int32, Complex64), &SparseMatrixMatMulCpuKernelMod::LaunchKernel<int32_t, std::complex<float_t>>},
    {ADD_KERNEL(Int32, Complex128), &SparseMatrixMatMulCpuKernelMod::LaunchKernel<int32_t, std::complex<double_t>>},
    {ADD_KERNEL(Int64, Float32), &SparseMatrixMatMulCpuKernelMod::LaunchKernel<int64_t, float>},
    {ADD_KERNEL(Int64, Float64), &SparseMatrixMatMulCpuKernelMod::LaunchKernel<int64_t, double>},
    {ADD_KERNEL(Int64, Complex64), &SparseMatrixMatMulCpuKernelMod::LaunchKernel<int64_t, std::complex<float_t>>},
    {ADD_KERNEL(Int64, Complex128), &SparseMatrixMatMulCpuKernelMod::LaunchKernel<int64_t, std::complex<double_t>>}};

std::vector<KernelAttr> SparseMatrixMatMulCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, SparseMatrixMatMulFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, SparseMatrixMatMul, SparseMatrixMatMulCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
