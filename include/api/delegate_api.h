/**
 * Copyright 2022 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef MINDSPORE_INCLUDE_API_DELEGATE_API_H
#define MINDSPORE_INCLUDE_API_DELEGATE_API_H

#include <map>
#include <vector>
#include <memory>
#include "include/api/status.h"
#include "include/api/types.h"
namespace mindspore {
class AbstractDelegate {
 public:
  AbstractDelegate() = default;
  AbstractDelegate(const std::vector<mindspore::MSTensor> &inputs, const std::vector<mindspore::MSTensor> &outputs)
      : inputs_(inputs), outputs_(outputs) {}
  virtual ~AbstractDelegate() = default;
  /// \brief Get the input tensors of DelegateModel.
  ///
  /// \return The input tensor vector of DelegateModel.
  const std::vector<mindspore::MSTensor> &inputs() { return this->inputs_; }

  /// \brief Get the output tensors of DelegateModel.
  ///
  /// \return The ioutput tensor vector of DelegateModel.
  const std::vector<mindspore::MSTensor> &outputs() { return this->outputs_; }

 protected:
  std::vector<mindspore::MSTensor> inputs_;
  std::vector<mindspore::MSTensor> outputs_;
};

template <typename Graph, typename Node, typename Kernel>
class IDelegate : public AbstractDelegate {
 public:
  IDelegate() = default;
  IDelegate(const std::vector<mindspore::MSTensor> &inputs, const std::vector<mindspore::MSTensor> &outputs)
      : AbstractDelegate(inputs, outputs) {}
  virtual ~IDelegate() = default;

  /// \brief Replace the nodes in model with delegate nodes, delegate will create kernels by its delegate nodes.
  ///
  /// \param[in] graph The graph to be built.
  virtual void ReplaceNodes(const std::shared_ptr<Graph> &graph) = 0;

  /// \brief Check if this node is belong to this delegate.
  ///
  /// \param[in] node The node need to be checked.
  ///
  /// \return True if the node is belong to this delegate, otherwise return false.
  virtual bool IsDelegateNode(const std::shared_ptr<Node> &node) = 0;

  /// \brief Create a delegate kernel if the node is a delegate node.
  ///
  /// \param[in] node Define the delegate model to be built.
  ///
  /// \return The delegate kernel, if the node is not a delegate node, return nullptr.
  virtual std::shared_ptr<Kernel> CreateKernel(const std::shared_ptr<Node> &node) = 0;
};
}  // namespace mindspore
#endif  // MINDSPORE_INCLUDE_API_DELEGATE_API_H
