/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_CTCLOSS_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_CTCLOSS_CPU_KERNEL_H_
#include <memory>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <limits>
#include "backend/kernel_compiler/cpu/cpu_kernel.h"
#include "backend/kernel_compiler/cpu/cpu_kernel_factory.h"

namespace mindspore {
namespace kernel {
class CTCLossCPUKernel : public CPUKernel {
 public:
  CTCLossCPUKernel() = default;
  ~CTCLossCPUKernel() override = default;

  void InitKernel(const CNodePtr &kernel_node) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

  void GenLableWithBlank(const uint32_t *seq_len, const std::vector<std::vector<uint32_t>> &batch_label,
                         std::vector<std::vector<uint32_t>> *label_with_blank);

  template <typename T>
  void CalculateFwdVar(const std::vector<uint32_t> &label_with_blank, const std::vector<std::vector<T>> &y,
                       std::vector<std::vector<T>> *log_alpha_b);
  template <typename T>
  void CalculateBwdVar(const std::vector<uint32_t> &label_with_blank, const std::vector<std::vector<T>> &y,
                       std::vector<std::vector<T>> *log_beta_b);
  template <typename T>
  void CalculateGrad(const std::vector<uint32_t> &label_with_blank, const std::vector<std::vector<T>> &y,
                     const std::vector<std::vector<T>> &log_alpha_b, const std::vector<std::vector<T>> &log_beta_b,
                     const T log_pzx, std::vector<std::vector<T>> *dy);

  template <typename T>
  void LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs);

 private:
  void CheckParam(const CNodePtr &kernel_node);
  std::vector<size_t> probs_shape_;
  std::vector<size_t> indice_dims_;
  std::vector<size_t> labels_dims_;
  size_t num_class_;
  size_t max_time_;
  size_t batch_size_;
  uint32_t blank_index_;
  TypeId dtype_{kTypeUnknown};
  bool preprocess_collapse_repeated_;
  bool ctc_merge_repeated_;
  bool ignore_longer_outputs_than_inputs_;
};

MS_REG_CPU_KERNEL(CTCLoss,
                  KernelAttr()
                    .AddInputAttr(kNumberTypeFloat16)
                    .AddInputAttr(kNumberTypeInt64)
                    .AddInputAttr(kNumberTypeInt32)
                    .AddInputAttr(kNumberTypeInt32)
                    .AddOutputAttr(kNumberTypeFloat16)
                    .AddOutputAttr(kNumberTypeFloat16),
                  CTCLossCPUKernel);

MS_REG_CPU_KERNEL(CTCLoss,
                  KernelAttr()
                    .AddInputAttr(kNumberTypeFloat32)
                    .AddInputAttr(kNumberTypeInt64)
                    .AddInputAttr(kNumberTypeInt32)
                    .AddInputAttr(kNumberTypeInt32)
                    .AddOutputAttr(kNumberTypeFloat32)
                    .AddOutputAttr(kNumberTypeFloat32),
                  CTCLossCPUKernel);
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_CTCLOSS_CPU_KERNEL_H_
