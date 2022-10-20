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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_EIGEN_SPARSE_SPARSE_MAXIMUM_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_EIGEN_SPARSE_SPARSE_MAXIMUM_CPU_KERNEL_H_
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
class SparseSparseMaximumCpuKernelMod : public NativeCpuKernelMod {
 public:
  SparseSparseMaximumCpuKernelMod() = default;
  ~SparseSparseMaximumCpuKernelMod() override = default;

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

 protected:
  std::vector<KernelAttr> GetOpSupport() override;
  void SyncData() override;
  std::vector<KernelTensorPtr> GetOutputs() override { return outputs_; }

 private:
  template <typename T>
  bool LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs);
  void CheckInputShape(const std::vector<KernelTensorPtr> &inputs, const int64_t a_nnz, const int64_t b_nnz,
                       const int64_t num_dims);

  std::vector<KernelTensorPtr> outputs_;
  TypeId dtype_{kTypeUnknown};
  TypeId itype_{kTypeUnknown};
  int64_t indice_size_;
  int64_t value_size_;
  int64_t shape_size_;
  int64_t a_nnz_;
  int64_t b_nnz_;
  int64_t num_dims_;
  int64_t sum_nnz_;
  int64_t a_values_shape0_;
  int64_t b_values_shape0_;
  int64_t a_shapes_shape0_;
  int64_t b_shapes_shape0_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_EIGEN_SPARSE_SPARSE_MAXIMUM_CPU_KERNEL_H_
