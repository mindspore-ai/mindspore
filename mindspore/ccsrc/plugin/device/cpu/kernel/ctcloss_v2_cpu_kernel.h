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
#include <map>
#include <utility>
#include <vector>
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

 private:
  struct SoftParam {
    int64_t input_length;
    int64_t target_length;
    int64_t offset;
    int64_t batch;
  };
  template <typename target_t>
  inline int64_t GetBlankPaddedTarget(const target_t *target, int64_t offset, int64_t idx) {
    constexpr int64_t interval = 2;
    if (idx % interval == 0) {
      return blank_;
    } else {
      return target[offset + (idx / interval)];
    }
  }
  template <typename S, typename T>
  void LossCompute(S *log_probs_p, S *log_alpha_p, T *tar_p, SoftParam params);
  template <typename T>
  bool IndexProcessing(T *in_len_p, T *tar_len_p, std::vector<int64_t> *target_offsets);
  // Variables for the operator itself
  int64_t blank_{0};
  // Stands for T
  int64_t time_series_{0};
  // Stands for N
  int64_t batch_sizes_{0};
  // Stands for C
  int64_t num_labels_{0};
  // Stands for S
  bool zero_infinity_{false};
  int64_t max_target_length_{0};
  template <typename T, typename S>
  bool LaunchKernel(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &workspace,
                    const std::vector<kernel::AddressPtr> &outputs);
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_DROPOUT_ND_CPU_KERNEL_H_
