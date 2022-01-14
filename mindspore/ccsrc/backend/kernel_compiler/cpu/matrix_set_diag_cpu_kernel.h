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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_MATRIX_SET_DIAG_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_MATRIX_SET_DIAG_KERNEL_H_

#include <vector>
#include <memory>
#include <string>
#include <algorithm>
#include <utility>
#include "backend/kernel_compiler/cpu/cpu_kernel.h"
#include "backend/kernel_compiler/cpu/cpu_kernel_factory.h"
#include "backend/kernel_compiler/common_utils.h"
namespace mindspore {
namespace kernel {
class MatrixSetDiagCpuKernelMod : public NativeCpuKernelMod {
 public:
  MatrixSetDiagCpuKernelMod() = default;
  ~MatrixSetDiagCpuKernelMod() override = default;
  void InitKernel(const CNodePtr &kernel_node) override;
  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspaces,
              const std::vector<AddressPtr> &outputs) override;

 private:
  template <typename T>
  void LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspaces,
                    const std::vector<AddressPtr> &outputs);
  int lower_diag_index_{0};
  int upper_diag_index_{0};
  int inner_rows_{0};
  int inner_cols_{0};
  int num_diags_{0};
  int expected_num_diags_{0};
  int max_diag_len_{0};
  int outer_batch_{1};
  bool is_single_diag_{true};
  std::vector<size_t> input_shape_;
  // <super_matrix_diag_align, sub_matrix_diag_align>
  std::pair<MatrixDiag::Alignment, MatrixDiag::Alignment> alignment_{MatrixDiag::RIGHT, MatrixDiag::LEFT};
  TypeId data_type_{0};
};

MS_REG_CPU_KERNEL(MatrixSetDiag,
                  KernelAttr()
                    .AddInputAttr(kNumberTypeInt32)
                    .AddInputAttr(kNumberTypeInt32)
                    .AddInputAttr(kNumberTypeInt32)
                    .AddOutputAttr(kNumberTypeInt32),
                  MatrixSetDiagCpuKernelMod)

MS_REG_CPU_KERNEL(MatrixSetDiag,
                  KernelAttr()
                    .AddInputAttr(kNumberTypeFloat16)
                    .AddInputAttr(kNumberTypeFloat16)
                    .AddInputAttr(kNumberTypeInt32)
                    .AddOutputAttr(kNumberTypeFloat16),
                  MatrixSetDiagCpuKernelMod)

MS_REG_CPU_KERNEL(MatrixSetDiag,
                  KernelAttr()
                    .AddInputAttr(kNumberTypeFloat32)
                    .AddInputAttr(kNumberTypeFloat32)
                    .AddInputAttr(kNumberTypeInt32)
                    .AddOutputAttr(kNumberTypeFloat32),
                  MatrixSetDiagCpuKernelMod)

MS_REG_CPU_KERNEL(MatrixSetDiag,
                  KernelAttr()
                    .AddInputAttr(kNumberTypeFloat64)
                    .AddInputAttr(kNumberTypeFloat64)
                    .AddInputAttr(kNumberTypeInt32)
                    .AddOutputAttr(kNumberTypeFloat64),
                  MatrixSetDiagCpuKernelMod)
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_MATRIX_SET_DIAG_KERNEL_H_
