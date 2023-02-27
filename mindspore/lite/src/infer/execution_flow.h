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
#ifndef MINDSPORE_LITE_INFER_EXECUTION_FLOW_H_
#define MINDSPORE_LITE_INFER_EXECUTION_FLOW_H_

#include <vector>
#include <memory>

#include "infer/context.h"
#include "infer/kernel.h"
#include "infer/kernel_callback.h"

namespace mindspore::infer::abstract {
class ExecutionFlow : public std::enable_shared_from_this<ExecutionFlow> {
 public:
  virtual ~ExecutionFlow() = default;

  /// \brief Get list of kernel need to run.
  ///
  /// \return vector of Kernel.
  virtual std::vector<Kernel *> GetKernels() = 0;

  /// \brief Set list of kernel need to run.
  ///
  /// \param[in] kernels, list of kernels
  ///
  /// \return void.
  virtual void SetKernels(const std::vector<Kernel *> &kernels) = 0;

  /// \brief Get list of inputs for the execution flow.
  ///
  /// \return vector of Tensor.
  virtual std::vector<Tensor *> GetInputs() = 0;

  /// \brief Set input tensors need to run.
  ///
  /// \param[in] inputs, list of input tensor
  ///
  /// \return void.
  virtual void SetInputs(const std::vector<Tensor *> &inputs) = 0;

  /// \brief Get list of outputs for the execution flow.
  ///
  /// \return vector of Tensor.
  virtual std::vector<Tensor *> GetOutputs() = 0;

  /// \brief Set output tensors need to run.
  ///
  /// \param[in] inputs, list of output tensor
  ///
  /// \return void.
  virtual void SetOutputs(const std::vector<Tensor *> &outputs) = 0;

  /// \brief Get context for the execution flow.
  ///
  /// \return Context pointer.
  virtual std::shared_ptr<Context> GetContext() = 0;

  /// \brief Set context of execution run
  ///
  /// \param[in] context, context for running
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

  /// \brief Construct flow into one fusion Kernel, eg. SubGraphKernel.
  ///
  /// \return Kernel pointer.
  virtual Kernel *ConstructFusionKernel() = 0;
};
using ExecutionFlowPtr = std::shared_ptr<ExecutionFlow>;
}  // namespace mindspore::infer::abstract

#endif  // MINDSPORE_LITE_INFER_EXECUTION_FLOW_H_
