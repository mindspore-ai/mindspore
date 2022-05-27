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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_CTCLOSS_V2_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_CTCLOSS_V2_CPU_KERNEL_H_

#include <memory>
#include <vector>
#include <map>
#include <utility>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
class CTCLossV2CpuKernelMod : public NativeCpuKernelMod, public MatchKernelHelper<CTCLossV2CpuKernelMod> {
 public:
  CTCLossV2CpuKernelMod() = default;
  ~CTCLossV2CpuKernelMod() override = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override {
    return kernel_func_(this, inputs, workspace, outputs);
  }

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &) override;

  const std::vector<std::pair<KernelAttr, KernelRunFunc>> &GetFuncList() const override;

 protected:
  std::vector<KernelAttr> GetOpSupport() override { return OpSupport(); }

  std::vector<KernelTensorPtr> GetOutputs() override { return outputs_; };

 private:
  template <typename S>
  inline S GetBlankPaddedTarget(const S *target, int i) const {
    if (i % 2 == 0) {
      return static_cast<S>(blank_);
    } else {
      return target[i / 2];
    }
  }

  template <typename S>
  std::vector<S> IndexProcessing(const S *input_lengths, const S *target_lengths);

  template <typename T, typename S>
  T DoReduce(T *neg_log_likelihood, const S *target_lengths) const;

  enum ReductionType { None, Mean, Sum };

  // Variables for the operator itself
  int64_t blank_{0};
  ReductionType reduction_{None};
  // Stands for T
  int64_t time_series_{0};
  // Stands for N
  int64_t batch_{0};
  // Stands for number of classes
  int64_t num_classes_{0};
  std::vector<int64_t> target_shape_;
  // If targets are un-padded, all targets are concatenated within 1-d
  bool padded_targets{false};

  // Dealing with dynamic shapes
  std::vector<KernelTensorPtr> outputs_{};
  bool dyamic_shape_{false};

  // Dealing with multiple types
  template <typename T, typename S>
  bool LaunchKernel(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &workspace,
                    const std::vector<kernel::AddressPtr> &outputs);
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_DROPOUT_ND_CPU_KERNEL_H_
