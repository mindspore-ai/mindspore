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

#ifndef MINDSPORE_LITE_INCLUDE_REGISTRY_PASS_BASE_H_
#define MINDSPORE_LITE_INCLUDE_REGISTRY_PASS_BASE_H_

#include <memory>
#include <string>
#include "include/lite_utils.h"
#include "api/ir/func_graph.h"

namespace mindspore {
namespace registry {
/// \brief PassBase defined a base class, which provides an interface for user to operate FuncGraph.
class MS_API PassBase {
 public:
  /// \brief Constructor
  ///
  /// \param[in] name Define pass name, which should be unique with each other.
  explicit PassBase(const std::string &name = "PassBase") : name_(name) {}

  /// \brief Destructor
  virtual ~PassBase() = default;

  /// \brief An interface for user to operate FuncGraph.
  ///
  /// \param[in] func_graph Define the struct of the model.
  ///
  /// \return Boolean value to represent whether the operation is successful or not.
  virtual bool Execute(const api::FuncGraphPtr &func_graph) = 0;

 private:
  const std::string name_;
};

/// \brief PassBasePtr defined a shared_ptr type.
using PassBasePtr = std::shared_ptr<PassBase>;
}  // namespace registry
}  // namespace mindspore
#endif  // MINDSPORE_LITE_INCLUDE_REGISTRY_PASS_BASE_H_
