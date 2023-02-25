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
#ifndef MINDSPORE_LITE_INFER_GRAPH__RUNTIME_H_
#define MINDSPORE_LITE_INFER_GRAPH__RUNTIME_H_

#include <vector>
#include <memory>

#include "include/api/status.h"
#include "infer/executor.h"
#include "infer/execution_plan.h"
#include "infer/kernel_callback.h"

namespace mindspore::infer::abstract {
class GraphRuntime : public std::enable_shared_from_this<GraphRuntime> {
 public:
  virtual ~GraphRuntime() = default;

  /// \brief Prepare Execution According to ExecutionPlan.
  ///
  /// \param[in] execution_plan Abstract Execution Plan for execute.
  ///
  /// \return Status.
  virtual Status Prepare(std::shared_ptr<ExecutionPlan> execution_plan) = 0;

  /// \brief Execute According to ExecutionPlan.
  ///
  /// \return Status.
  virtual Status Execute() = 0;

  /// \brief Execute According to ExecutionPlan.
  ///
  /// \param[in] inputs, inputs tensors for compute
  /// \param[in] outputs, outputs tensors for compute
  ///
  /// \return Status.
  virtual Status Execute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                         KernelCallBack before = nullptr, KernelCallBack after = nullptr) = 0;

  /// \brief Get list of inputs for the model.
  ///
  /// \return vector of Tensor.
  virtual std::vector<Tensor *> GetInputs() = 0;

  /// \brief Get list of outputs for the model.
  ///
  /// \return vector of Tensor.
  virtual std::vector<Tensor *> GetOutputs() = 0;
};
}  // namespace mindspore::infer::abstract

#endif  // MINDSPORE_LITE_INFER_GRAPH__RUNTIME_H_
