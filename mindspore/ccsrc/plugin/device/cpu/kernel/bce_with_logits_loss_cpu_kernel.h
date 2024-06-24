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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_NN_BCE_WITH_LOGITS_LOSS_KERNEL_H
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_NN_BCE_WITH_LOGITS_LOSS_KERNEL_H

#include <vector>
#include <utility>
#include <map>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
class BCEWithLogitsLossCpuKernelMod : public NativeCpuKernelMod,
                                      public MatchKernelHelper<BCEWithLogitsLossCpuKernelMod> {
 public:
  BCEWithLogitsLossCpuKernelMod() = default;
  ~BCEWithLogitsLossCpuKernelMod() override = default;

  using bce_ptr = void *;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs) override {
    return kernel_func_(this, inputs, workspace, outputs);
  }

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  const std::vector<std::pair<KernelAttr, KernelRunFunc>> &GetFuncList() const override;

  void RunTask(int task_id);

  std::vector<KernelAttr> GetOpSupport() override { return OpSupport(); }

 private:
  bool LaunchKernel(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                    const std::vector<KernelTensor *> &outputs);
  bool is_broadcast_{false};
  bool is_reduction_{false};
  size_t input_size_{1};
  size_t thread_num_{1};
  bce_ptr logits_{nullptr};
  bce_ptr label_{nullptr};
  bce_ptr weight_{nullptr};
  bce_ptr post_weight_{nullptr};
  bce_ptr reduction_output_{nullptr};
  bce_ptr output_{nullptr};
  ShapeVector input_logits_shape_;
  ShapeVector input_label_shape_;
  ShapeVector input_weight_shape_;
  ShapeVector input_post_weight_shape_;
  size_t weight_workspace_index_ = 0;
  size_t pos_weight_workspace_index_ = 0;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_NN_BCE_WITH_LOGITS_LOSS_KERNEL_H
