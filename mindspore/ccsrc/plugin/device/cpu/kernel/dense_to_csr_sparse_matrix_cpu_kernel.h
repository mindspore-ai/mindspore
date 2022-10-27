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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_DENSE_TO_CSR_SPARSE_MATRIX_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_DENSE_TO_CSR_SPARSE_MATRIX_KERNEL_H_

#include <complex>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
class DenseToCSRSparseMatrixCpuKernelMod : public NativeCpuKernelMod {
 public:
  DenseToCSRSparseMatrixCpuKernelMod() = default;
  ~DenseToCSRSparseMatrixCpuKernelMod() override = default;

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

  std::vector<KernelAttr> GetOpSupport() override;

 private:
  template <typename valueT, typename indiceT>
  void LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs) const;
  size_t rank_{0};
  size_t batch_size_{0};
  size_t num_rows_{0};
  size_t num_cols_{0};
  size_t total_nnz_{0};
  TypeId values_type_{kTypeUnknown};
  TypeId indices_type_{kTypeUnknown};
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_DENSE_TO_CSR_SPARSE_MATRIX_KERNEL_H_
