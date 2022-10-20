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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_EIGEN_SPARSE_MATRIX_MAT_MUL_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_EIGEN_SPARSE_MATRIX_MAT_MUL_CPU_KERNEL_H_

#include <vector>
#include <utility>
#include "Eigen/Core"
#include "Eigen/SparseCore"
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
class SparseMatrixMatMulCpuKernelMod : public DeprecatedNativeCpuKernelMod {
 public:
  SparseMatrixMatMulCpuKernelMod() = default;
  ~SparseMatrixMatMulCpuKernelMod() override = default;

  void InitKernel(const CNodePtr &kernel_node) override;
  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override {
    return kernel_func_(this, inputs, workspace, outputs);
  }

 protected:
  std::vector<KernelAttr> GetOpSupport() override;

 private:
  template <typename indiceT, typename valueT>
  bool LaunchKernel(const std::vector<kernel::AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                    const std::vector<kernel::AddressPtr> &outputs);

  using SparseMatrixMatMulFunc =
    std::function<bool(SparseMatrixMatMulCpuKernelMod *, const std::vector<kernel::AddressPtr> &,
                       const std::vector<kernel::AddressPtr> &, const std::vector<kernel::AddressPtr> &)>;
  static std::vector<std::pair<KernelAttr, SparseMatrixMatMulFunc>> func_list_;
  SparseMatrixMatMulFunc kernel_func_;

  template <typename T>
  bool CheckMatMul(const std::vector<AddressPtr> &inputs);

  // create eigen sparsematrix with eigen::map
  template <typename indiceT, typename valueT>
  Eigen::Ref<const Eigen::SparseMatrix<valueT, Eigen::RowMajor, indiceT>> CreateEigenSparseMatrix(
    indiceT rows, indiceT cols, int64_t nnz, indiceT *row_pointers, indiceT *col_indices, valueT *values,
    bool transpose, bool adjoint);

  size_t batch_size_{0};
  TypeId indice_type_{kTypeUnknown};
  TypeId value_type_{kTypeUnknown};
  bool transpose_x1_{false};
  bool transpose_x2_{false};
  bool adjoint_x1_{false};
  bool adjoint_x2_{false};
  bool transpose_output_{false};
  bool conjugate_output_{false};
  size_t rank_{0};
  size_t shift_;
  std::vector<size_t> input_shape2_;
  CNodeWeakPtr node_wpt_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_EIGEN_SPARSE_MATRIX_MAT_MUL_CPU_KERNEL_H_
