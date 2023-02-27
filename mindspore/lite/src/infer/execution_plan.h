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
#ifndef MINDSPORE_LITE_INFER_EXECUTION_PLAN_H_
#define MINDSPORE_LITE_INFER_EXECUTION_PLAN_H_

#include <vector>
#include <memory>

#include "ir/func_graph.h"
#include "infer/execution_flow.h"
#include "infer/context.h"

namespace mindspore::infer::abstract {
class ExecutionPlan : public std::enable_shared_from_this<ExecutionPlan> {
 public:
  virtual ~ExecutionPlan() = default;

  /// \brief Get list of execution flow in execution plan.
  ///
  /// \return vector of ExecutionFlow.
  virtual std::vector<std::shared_ptr<ExecutionFlow>> GetExecutionFLows() = 0;

  /// \brief Set Execution Flows for the execution plan.
  ///
  /// \param[in] execution_flows, the list of execution flows need run
  ///
  /// \return void.
  virtual void SetExecutionFlows(std::vector<std::shared_ptr<ExecutionFlow>> execution_flows) = 0;

  /// \brief Add a Execution Flow at end of the execution plan.
  ///
  /// \param[in] execution_flow, the execution flow need to add
  ///
  /// \return void.
  virtual void AddExecutionFlow(std::shared_ptr<ExecutionFlow> execution_flow) = 0;

  /// \brief Get FuncGraph of Model need to run.
  ///
  /// \return FuncGraph pointer.
  virtual FuncGraphPtr GetFuncGraph() = 0;

  /// \brief Set FuncGraph for the execution plan.
  ///
  /// \param[in] func_graph, the graph need to run
  ///
  /// \return void.
  virtual void SetFuncGraph(FuncGraphPtr func_graph) = 0;

  /// \brief Get list of inputs for the model.
  ///
  /// \return vector of Tensor.
  virtual std::vector<Tensor *> GetInputs() = 0;

  /// \brief Set input tensors need to run.
  ///
  /// \param[in] inputs, list of input tensor
  ///
  /// \return void.
  virtual void SetInputs(const std::vector<Tensor *> &inputs) = 0;

  /// \brief Get list of outputs for the model.
  ///
  /// \return vector of Tensor.
  virtual std::vector<Tensor *> GetOutputs() = 0;

  /// \brief Set output tensors need to run.
  ///
  /// \param[in] inputs, list of output tensor
  ///
  /// \return void.
  virtual void SetOutputs(const std::vector<Tensor *> &outputs) = 0;

  /// \brief Get context of execution plan.
  ///
  /// \return Context of execution plan.
  virtual std::shared_ptr<Context> GetContext() = 0;

  /// \brief Set context to run.
  ///
  /// \param[in] context, context
  ///
  /// \return void.
  virtual void SetContext(std::shared_ptr<Context> context) = 0;

  /// \brief Get callback before kernel execution.
  ///
  /// \return KernelCallBack pointer.
  virtual const KernelCallBack &GetKernelBeforeCallBack() = 0;

  /// \brief Set callback before kernel execution.
  ///
  /// \param[in] callback, callback function pointer
  ///
  /// \return void.
  virtual void SetKernelBeforeCallBack(const KernelCallBack &callback) = 0;

  /// \brief Get callback after kernel execution.
  ///
  /// \return KernelCallBack pointer.
  virtual const KernelCallBack &GetKernelAfterCallBack() = 0;

  /// \brief Set callback after kernel execution.
  ///
  /// \param[in] callback, callback function pointer
  ///
  /// \return void.
  virtual void SetKernelAfterCallBack(const KernelCallBack &callback) = 0;

  /// \brief Convert Execution Plan to Kernel List
  ///
  /// \return Kernel List
  virtual std::vector<Kernel *> ToKernelList() = 0;
};
using ExecutionPlanPtr = std::shared_ptr<ExecutionPlan>;
}  // namespace mindspore::infer::abstract

#endif  // MINDSPORE_LITE_INFER_EXECUTION_PLAN_H_
