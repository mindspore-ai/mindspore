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
#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_BASE_PARTIAL_FUSION_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_BASE_PARTIAL_FUSION_H_

#include <vector>
#include "src/tensor.h"
#include "src/executor/kernel_exec.h"

// this file is going to be removed when move create actor before schedule.
namespace mindspore::kernel {
class PartialFusionKernel : public LiteKernel {
 public:
  PartialFusionKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                      const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : LiteKernel(parameter, inputs, outputs, ctx) {}
  ~PartialFusionKernel() override = default;
  int Prepare() override;
  int ReSize() override;
  int Run() override;
  void set_subgraph_kernels(const std::vector<KernelExec *> &subgraph_kernels) { subgraph_kernels_ = subgraph_kernels; }
  std::vector<KernelExec *> subgraph_kernels() const { return subgraph_kernels_; }
  // One partial corresponds to a subgraph at offline stage, after graph schedule, a subgraph may be split into many
  // graphs, so use a vector.
  std::vector<KernelExec *> subgraph_kernels_{};
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_BASE_PARTIAL_FUSION_H_
