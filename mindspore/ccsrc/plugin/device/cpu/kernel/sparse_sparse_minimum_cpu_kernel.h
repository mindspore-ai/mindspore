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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_SPARSE_SPARSE_MINIMUM_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_SPARSE_SPARSE_MINIMUM_CPU_KERNEL_H_

#include <algorithm>
#include <complex>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <unordered_map>
#include <vector>

#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
class SparseSparseMinimumCpuKernelMod : public NativeCpuKernelMod {
 public:
  SparseSparseMinimumCpuKernelMod() = default;
  ~SparseSparseMinimumCpuKernelMod() override = default;
  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

 protected:
  void SyncData() override;
  std::vector<KernelTensorPtr> GetOutputs() override { return outputs_; }
  std::vector<KernelAttr> GetOpSupport() override;

 private:
  template <typename T>
  void LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs);

  std::vector<KernelTensorPtr> outputs_{};
  TypeId dtype_{kTypeUnknown};
  TypeId itype_{kTypeUnknown};
  int64_t indice_size_;
  int64_t value_size_;
  int64_t shape_size_;
  int64_t x1_nnz_;
  int64_t x2_nnz_;
  int64_t num_dims_;
  int64_t y_nnz_;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_SPARSE_SPARSE_MINIMUM_CPU_KERNEL_H_
