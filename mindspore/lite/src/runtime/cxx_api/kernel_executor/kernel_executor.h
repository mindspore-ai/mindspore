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

namespace mindspore {
class KernelExecutorImpl;

class MS_API KernelExecutor {
 public:
  KernelExecutor() = default;
  ~KernelExecutor() = default;

  /// \brief Build a single operator so that it can run on a device.
  ///
  /// \param[in] op Define an operator pointer.
  /// \param[in] ms_context Define the context used to store options during execution.
  /// \param[in] inputs A vector where single operator inputs are arranged in sequence.
  /// \param[in] outputs A vector where single operator outputs are arranged in sequence.
  ///
  /// \return Status.
  Status Build(const std::shared_ptr<ops::BaseOperator> &op, const std::vector<MSTensor> &inputs,
               const std::vector<MSTensor> &outputs, const std::shared_ptr<Context> &ms_context);

  /// \brief ReSize KernelExecutor.
  ///
  /// \param[in] inputs A vector where single operator inputs are arranged in sequence.
  /// \param[in] outputs A vector where single operator outputs are arranged in sequence.
  ///
  /// \return Status.
  Status ReSize(const std::vector<MSTensor> &inputs, const std::vector<MSTensor> &outputs);

  /// \brief set outputs infer shape info.
  ///
  /// \param[in] outputs A vector where single operator outputs are arranged in sequence.
  ///
  /// \return Status.
  Status Infer(std::vector<MSTensor> *outputs);

  /// \brief ReSize KernelExecutor.
  ///
  /// \param[in] inputs A vector where single operator inputs are arranged in sequence.
  /// \param[in] outputs A vector where single operator outputs are arranged in sequence.
  ///
  /// \return Status.
  Status Execute(const std::vector<MSTensor> &inputs, const std::vector<MSTensor> &outputs);

 private:
  std::shared_ptr<KernelExecutorImpl> impl_ = nullptr;
};
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_RUNTIME_CXX_API_KERNEL_EXECUTOR_KERNEL_EXECUTOR_H_
