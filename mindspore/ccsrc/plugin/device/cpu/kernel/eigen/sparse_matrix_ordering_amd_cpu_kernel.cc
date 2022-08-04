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
#include "plugin/device/cpu/kernel/eigen/sparse_matrix_ordering_amd_cpu_kernel.h"
#include <cmath>
#include <ctime>
#include <random>
#include "Eigen/Core"
#include "Eigen/SparseCholesky"
#include "Eigen/SparseCore"
#include "Eigen/OrderingMethods"
#include "unsupported/Eigen/CXX11/Tensor"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "kernel/common_utils.h"
#include "mindspore/core/ops/sparse_matrix_ordering_amd.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kInputsNum = 5;
constexpr size_t kOutputsNum = 1;
constexpr size_t kInputIndex0 = 0;
constexpr size_t kInputIndex1 = 1;
constexpr size_t kInputIndex2 = 2;
constexpr size_t kInputIndex3 = 3;
constexpr size_t kInputIndex4 = 4;
constexpr size_t kOutputIndex0 = 0;
constexpr size_t kRankNum = 2;

template <typename T>
struct TTypes {
  typedef Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor, Eigen::DenseIndex>> UnalignedVec;
};

using SparseMatrix = Eigen::SparseMatrix<int32_t, Eigen::RowMajor>;

using IndicesMap = Eigen::Map<Eigen::Matrix<int32_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
}  // namespace

bool SparseMatrixOrderingAMDCpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                               const std::vector<KernelTensorPtr> &inputs,
                                               const std::vector<KernelTensorPtr> &outputs) {
  auto kernel_ptr = std::dynamic_pointer_cast<ops::SparseMatrixOrderingAMD>(base_operator);
  kernel_name_ = kernel_ptr->name();
  x_dense_shape_shape_ = inputs[kIndex0]->GetShapeVector();
  x_batch_pointers_shape_ = inputs[kIndex1]->GetShapeVector();
  x_row_pointers_shape_ = inputs[kIndex2]->GetShapeVector();
  x_col_indices_shape_ = inputs[kIndex3]->GetShapeVector();
  x_values_shape_ = inputs[kIndex4]->GetShapeVector();
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kOutputsNum, kernel_name_);

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' does not support this kernel data type: " << kernel_attr << ".";
    return false;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int SparseMatrixOrderingAMDCpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                                const std::vector<KernelTensorPtr> &inputs,
                                                const std::vector<KernelTensorPtr> &outputs,
                                                const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  if (auto ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
    return ret;
  }

  x_dense_shape_shape_ = inputs[kIndex0]->GetShapeVector();
  x_batch_pointers_shape_ = inputs[kIndex1]->GetShapeVector();
  x_row_pointers_shape_ = inputs[kIndex2]->GetShapeVector();
  x_col_indices_shape_ = inputs[kIndex3]->GetShapeVector();
  return KRET_OK;
}

bool SparseMatrixOrderingAMDCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                                       const std::vector<kernel::AddressPtr> &outputs) {
  auto x_dense_shape_ptr = reinterpret_cast<int64_t *>(inputs[kIndex0]->addr);
  auto x_batch_pointers_ptr = reinterpret_cast<int32_t *>(inputs[kIndex1]->addr);
  auto x_row_pointers_ptr = reinterpret_cast<int32_t *>(inputs[kIndex2]->addr);
  auto x_col_pointers_ptr = reinterpret_cast<int32_t *>(inputs[kIndex3]->addr);
  auto y_ptr = reinterpret_cast<int32_t *>(outputs[kIndex0]->addr);

  const int64_t rank = x_dense_shape_shape_[0];
  const int64_t num_rows = x_dense_shape_ptr[(rank == 2) ? 0 : 1];
  const int64_t num_cols = x_dense_shape_ptr[(rank == 2) ? 1 : 2];
  const int64_t num_batch = x_batch_pointers_shape_[0] - 1;

  if (rank == kRankNum) {
    if (x_batch_pointers_shape_[0] != kRankNum) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the input of batch pionters shape should equals "
                        << "2 to match the CSR form input when input has no batch.";
    }
  }
  if (num_rows != num_cols) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', sparse matrix must be square, but got [ " << num_rows
                      << " != " << num_cols << " ].";
  }
  if (x_row_pointers_shape_[0] != num_batch * (num_rows + 1)) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the input of row pionters shape should equals"
                      << " batch * (rows + 1) to match the CSR form input when input has no batch, but got [ "
                      << x_row_pointers_shape_[0] << " != " << num_batch * (num_rows + 1) << " ].";
  }

  for (int64_t batch_index = 0; batch_index < num_batch; ++batch_index) {
    // Define an Eigen SparseMatrix Map to operate on the
    // CSRSparseMatrix component without copying the data.
    // The values doesn't matter for computing the ordering, hence we
    // reuse the column pointers as dummy values.
    int64_t rows = x_dense_shape_ptr[(rank == 2) ? 0 : 1];
    int64_t offset_rows = batch_index * (rows + 1);
    int32_t offset_cols = x_batch_pointers_ptr[batch_index];
    int32_t nnz_in_batch = x_batch_pointers_ptr[batch_index + 1] - x_batch_pointers_ptr[batch_index];
    Eigen::Map<const SparseMatrix> sparse_matrix(
      num_rows, num_rows, nnz_in_batch,
      TTypes<int32_t>::UnalignedVec(x_row_pointers_ptr + offset_rows, rows + 1).data(),
      TTypes<int32_t>::UnalignedVec(x_col_pointers_ptr + offset_cols, nnz_in_batch).data(),
      TTypes<int32_t>::UnalignedVec(x_col_pointers_ptr + offset_cols, nnz_in_batch).data());
    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, int32_t> permutation_matrix;
    // Compute the AMD ordering.
    Eigen::AMDOrdering<int32_t> amd_ordering;
    amd_ordering(sparse_matrix.template selfadjointView<Eigen::Lower>(), permutation_matrix);
    // Define an Eigen Map over the allocated output Tensor so that it
    // can be mutated in place.
    IndicesMap permutation_map(y_ptr + batch_index * num_rows, num_rows, 1);
    permutation_map = permutation_matrix.indices();
  }

  return true;
}

std::vector<std::pair<KernelAttr, SparseMatrixOrderingAMDCpuKernelMod::SparseMatrixOrderingAMDLaunchFunc>>
  SparseMatrixOrderingAMDCpuKernelMod::func_list_ = {{
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeInt32),
     &SparseMatrixOrderingAMDCpuKernelMod::LaunchKernel},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeFloat64)
       .AddOutputAttr(kNumberTypeInt32),
     &SparseMatrixOrderingAMDCpuKernelMod::LaunchKernel},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeComplex64)
       .AddOutputAttr(kNumberTypeInt32),
     &SparseMatrixOrderingAMDCpuKernelMod::LaunchKernel},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeComplex128)
       .AddOutputAttr(kNumberTypeInt32),
     &SparseMatrixOrderingAMDCpuKernelMod::LaunchKernel},
  }};

std::vector<KernelAttr> SparseMatrixOrderingAMDCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(
    func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
    [](const std::pair<KernelAttr, SparseMatrixOrderingAMDCpuKernelMod::SparseMatrixOrderingAMDLaunchFunc> &pair) {
      return pair.first;
    });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, SparseMatrixOrderingAMD, SparseMatrixOrderingAMDCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
