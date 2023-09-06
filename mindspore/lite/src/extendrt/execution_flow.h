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

#ifndef MINDSPORE_LITE_SRC_EXTENDRT_EXECUTION_FLOW_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_EXECUTION_FLOW_H_

#include <string>
#include <vector>
#include <memory>

#include "infer/execution_flow.h"
#include "src/executor/sub_graph_kernel.h"

namespace mindspore::infer {
class ExecutionFlow : public abstract::ExecutionFlow {
 public:
  ExecutionFlow() = default;
  ~ExecutionFlow() override;

  std::vector<InferKernel *> GetKernels() override { return kernels_; }

  void SetKernels(const std::vector<InferKernel *> &kernels) override { kernels_ = kernels; }

  std::vector<InferTensor *> GetInputs() override { return inputs_; }

  void SetInputs(const std::vector<InferTensor *> &inputs) override { inputs_ = inputs; }

  std::vector<InferTensor *> GetOutputs() override { return outputs_; }

  void SetOutputs(const std::vector<InferTensor *> &outputs) override { outputs_ = outputs; }

  InferContextPtr GetContext() override { return context_; }

  void SetContext(InferContextPtr context) override { context_ = context; }

  const abstract::KernelCallBack &GetKernelBeforeCallBack() override { return before_; }

  void SetKernelBeforeCallBack(const abstract::KernelCallBack &callback) override { before_ = callback; }

  const abstract::KernelCallBack &GetKernelAfterCallBack() override { return after_; }

  void SetKernelAfterCallBack(const abstract::KernelCallBack &callback) override { after_ = callback; }

  kernel::SubGraphKernel *ConstructFusionKernel() override;

  std::vector<InferTensor *> GetTensors() { return tensors_; }

  void SetTensors(const std::vector<InferTensor *> &tensors) { tensors_ = tensors; }

  std::string Dump() const;

  mindspore::kernel::SubGraphType GetSubGraphType(abstract::Kernel *kernel);

 private:
  std::vector<InferKernel *> kernels_;
  std::vector<InferTensor *> inputs_;
  std::vector<InferTensor *> outputs_;
  std::vector<InferTensor *> tensors_;
  InferContextPtr context_;
  abstract::KernelCallBack before_;
  abstract::KernelCallBack after_;
};
using ExecutionFlowPtr = std::shared_ptr<ExecutionFlow>;
}  // namespace mindspore::infer

#endif  // MINDSPORE_LITE_SRC_EXTENDRT_EXECUTION_FLOW_H_
