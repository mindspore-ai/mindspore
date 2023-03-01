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
#ifndef MINDSPORE_LITE_INFER_EXECUTOR_H_
#define MINDSPORE_LITE_INFER_EXECUTOR_H_

#include <string>
#include <memory>
#include <vector>

#include "include/api/status.h"
#include "infer/tensor.h"

namespace mindspore::infer::abstract {
class Executor : public std::enable_shared_from_this<Executor> {
 public:
  virtual ~Executor() = default;

  /// \brief The Name of the Executor.
  ///
  /// \return String name of executor.
  virtual const std::string &Name() = 0;

  /// \brief Prepare Execution According to ExecutionFlow.
  ///
  /// \param[in] execution_flow Abstract Execution Plan for execute.
  ///
  /// \return Status.
  virtual Status Prepare() = 0;

  /// \brief Execute According to ExecutionFlow.
  ///
  /// \return Status.
  virtual Status Execute() = 0;

  /// \brief Resize Executor Kernels.
  ///
  /// \param[in] inputs, inputs need resize
  /// \param[in] dims, target shapes for resize inputs
  ///
  /// \return Status.
  virtual int Resize(const std::vector<Tensor *> &inputs, const std::vector<std::vector<int64_t>> &dims) = 0;
};
}  // namespace mindspore::infer::abstract

#endif  // MINDSPORE_LITE_INFER_EXECUTOR_H_
