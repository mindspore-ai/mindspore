/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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
#include "backend/kernel_compiler/cpu/mkldnn/mkl_cpu_kernel.h"

namespace mindspore {
namespace kernel {
class SparseSoftmaxCrossEntropyWithLogitsCPUKernel : public MKLCPUKernel {
 public:
  SparseSoftmaxCrossEntropyWithLogitsCPUKernel() = default;
  ~SparseSoftmaxCrossEntropyWithLogitsCPUKernel() override = default;

  void InitKernel(const CNodePtr &kernel_node) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

 private:
  void InitInputOutputSize(const CNodePtr &kernel_node) override;
  void ForwardPostExecute(const int *labels, const float *losses, float *output) const;
  void GradPostExecute(const int *labels, const float *losses, float *output) const;
  bool is_grad_{false};
  size_t class_num_{0};
  size_t batch_size_{0};
};

MS_REG_CPU_KERNEL(
  SparseSoftmaxCrossEntropyWithLogits,
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat32),
  SparseSoftmaxCrossEntropyWithLogitsCPUKernel);
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_SPARSE_SOFTMAX_CROSS_ENTROPY_WITH_LOGITS_CPU_KERNEL_H_
