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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_CXX_API_KERNEL_EXECUTOR_KERNEL_EXECUTOR_H_
#define MINDSPORE_LITE_SRC_RUNTIME_CXX_API_KERNEL_EXECUTOR_KERNEL_EXECUTOR_H_

#include <vector>
#include <memory>
#include "include/api/types.h"
#include "include/api/status.h"
#include "include/api/context.h"
#include "ops/base_operator.h"
#include "ops/custom.h"

namespace mindspore {
class KernelExecutorImpl;

class MS_API KernelExecutor {
 public:
  KernelExecutor() = default;
  ~KernelExecutor() = default;

  /// \brief Build a single operator so that it can run on a device.
  ///
  /// \param[in] op Define an operator pointer.
  ///               Notices:The `Init` function of BaseOperator may throw a `std::runtime_error` when the parameters
  ///                       passed into `Init` function is wrong. If don't want to break down the main process, please
  ///                       catch `std::runtime_error` to ensure execute without crash.
  /// \param[in] ms_context Define the context used to store options during execution.
  /// \param[in] inputs A vector where single operator inputs are arranged in sequence.
  ///
  /// \return Status.
  Status Build(const std::shared_ptr<ops::BaseOperator> &op, const std::vector<MSTensor> &inputs,
               const std::shared_ptr<Context> &ms_context);

  /// \brief Build a single operator so that it can run on a device.
  ///
  /// \param[in] op Define an Custom operator.
  ///
  /// \param[in] ms_context Define the context used to store options during execution.
  /// \param[in] inputs A vector where single operator inputs are arranged in sequence.
  /// \param[in] output_num Custom operator outputs size.
  ///
  /// \return Status.
  Status Build(const std::shared_ptr<ops::Custom> &op, const std::vector<MSTensor> &inputs,
               const std::shared_ptr<Context> &ms_context, const int output_num);

  /// \brief ReSize KernelExecutor. Change the shape and data type of inputs.
  ///        Notice: Conv2D can't update weight and bias by this method.
  ///
  /// \param[in] inputs A vector where single operator inputs are arranged in sequence.
  ///
  /// \return Status.
  Status ReSize(const std::vector<MSTensor> &inputs);

  /// \brief Execute KernelExecutor.
  ///
  /// \param[in] inputs A vector where single operator inputs are arranged in sequence.
  ///                   Notices: inputs Tensor should be freed by user.
  /// \param[in] outputs Which is a pointer to a vector.The outputs are filled in the container in sequence.
  ///                    Notices: outputs Tensor will be freed by ~KernelExecutorImpl(), user needn't free it.
  ///
  /// \return Status.
  Status Execute(const std::vector<MSTensor> &inputs, std::vector<MSTensor> *outputs);

 private:
  std::shared_ptr<KernelExecutorImpl> impl_ = nullptr;
};
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_RUNTIME_CXX_API_KERNEL_EXECUTOR_KERNEL_EXECUTOR_H_
