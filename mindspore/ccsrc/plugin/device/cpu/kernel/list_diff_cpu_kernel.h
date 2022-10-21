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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_LIST_DIFF_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_LIST_DIFF_CPU_KERNEL_H_

#include <memory>
#include <vector>
#include <map>
#include <string>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"
#include "mindspore/core/ops/list_diff.h"

namespace mindspore {
namespace kernel {
class ListDiffCPUKernelMod : public NativeCpuKernelMod {
 public:
  ListDiffCPUKernelMod() = default;
  ~ListDiffCPUKernelMod() override = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  int Resize(
    const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
    const std::vector<KernelTensorPtr> &outputs,
    const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost = std::map<uint32_t, tensor::TensorPtr>()) override;

 protected:
  void SyncData() override;

  std::vector<KernelTensorPtr> GetOutputs() override { return outputs_; }

  std::vector<KernelAttr> GetOpSupport() override;

 private:
  void ResetResource() noexcept;
  template <typename T, typename Tidx>
  bool LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs);

  std::vector<KernelTensorPtr> outputs_ = {};
  size_t data_size_;
  size_t index_size_;
  int64_t x_size_;
  int64_t y_size_;
  int64_t out_size_;
  TypeId out_type_;
  TypeId idx_type_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_LIST_DIFF_CPU_KERNEL_H_
