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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_SPARSE_CROSS_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_SPARSE_CROSS_CPU_KERNEL_H_

#include <vector>
#include <map>
#include <memory>
#include <string>
#include <complex>
#include <utility>
#include <functional>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"
#include "kernel/common_utils.h"

namespace mindspore {
namespace kernel {
class SparseCrossCpuKernelMod : public NativeCpuKernelMod, public MatchKernelHelper<SparseCrossCpuKernelMod> {
 public:
  SparseCrossCpuKernelMod() = default;
  ~SparseCrossCpuKernelMod() override = default;

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override {
    kernel_func_ = this->GetFuncList()[0].second;
    return kernel_func_(this, inputs, workspace, outputs);
  }

  int Resize(
    const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
    const std::vector<KernelTensorPtr> &outputs,
    const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost = std::map<uint32_t, tensor::TensorPtr>()) override;
  const std::vector<std::pair<KernelAttr, KernelRunFunc>> &GetFuncList() const override;

  std::vector<KernelAttr> GetOpSupport() override { return OpSupport(); }
  void SyncOutputShape();

 private:
  template <bool HASHED_OUTPUT, typename T, typename S>
  bool SparseCrossCann(const std::vector<std::vector<int64_t>> &indices_list_in,
                       const std::vector<std::vector<T>> &values_list_in,
                       const std::vector<std::vector<int64_t>> &shapes_list_in,
                       const std::vector<std::vector<S>> &dense_list_in,
                       const std::vector<kernel::AddressPtr> &outputs) const;

  template <typename T1, typename T2>
  bool LaunchKernel(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &workspace,
                    const std::vector<kernel::AddressPtr> &outputs);

  int64_t num_buckets_;
  uint64_t hash_key_;
  bool hash_out_;
  std::vector<std::vector<int64_t>> _indices_out_;
  std::vector<int64_t> _values_out_;
  TypeId values_type_{kTypeUnknown};
  TypeId dense_type_{kTypeUnknown};
  std::vector<TypeId> types_;
  int64_t N_;
  int64_t indices_row_{0};
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_SPARSE_CROSS_CPU_KERNEL_H_
