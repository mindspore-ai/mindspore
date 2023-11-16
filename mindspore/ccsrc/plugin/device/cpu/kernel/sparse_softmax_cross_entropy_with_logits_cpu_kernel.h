/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_SPARSE_SOFTMAX_CROSS_ENTROPY_WITH_LOGITS_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_SPARSE_SOFTMAX_CROSS_ENTROPY_WITH_LOGITS_CPU_KERNEL_H_

#include <vector>
#include <memory>
#include <map>
#include <utility>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"
#include "ops/sparse_softmax_cross_entropy_with_logits.h"

namespace mindspore {
namespace kernel {
class SparseSoftmaxCrossEntropyWithLogitsCpuKernelMod : public NativeCpuKernelMod {
 public:
  SparseSoftmaxCrossEntropyWithLogitsCpuKernelMod() = default;
  ~SparseSoftmaxCrossEntropyWithLogitsCpuKernelMod() override = default;

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs) override {
    return kernel_func_(this, inputs, workspace, outputs);
  }

 protected:
  std::vector<KernelAttr> GetOpSupport() override;

 private:
  void SoftmaxFp32(const float *logits, float *losses);

  template <typename T, typename S>
  void ForwardPostExecute(const S *labels, const T *losses, T *output) const;

  template <typename T, typename S>
  void GradPostExecute(const S *labels, const T *losses, T *output);

  template <typename T, typename S>
  bool LaunchKernel(const std::vector<kernel::KernelTensor *> &inputs,
                    const std::vector<kernel::KernelTensor *> &workspace,
                    const std::vector<kernel::KernelTensor *> &outputs);
  using SparseSoftmaxCrossEntropyWithLogitsFunc =
    std::function<bool(SparseSoftmaxCrossEntropyWithLogitsCpuKernelMod *, const std::vector<kernel::KernelTensor *> &,
                       const std::vector<kernel::KernelTensor *> &, const std::vector<kernel::KernelTensor *> &)>;
  static std::vector<std::pair<KernelAttr, SparseSoftmaxCrossEntropyWithLogitsFunc>> func_list_;
  SparseSoftmaxCrossEntropyWithLogitsFunc kernel_func_;

  bool is_grad_{false};
  size_t class_num_{0};
  size_t batch_size_{0};
  ParallelSearchInfo grad_parallel_search_info_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_SPARSE_SOFTMAX_CROSS_ENTROPY_WITH_LOGITS_CPU_KERNEL_H_
