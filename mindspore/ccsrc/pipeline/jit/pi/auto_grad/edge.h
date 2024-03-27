/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_PI_JIT_EDGE_H_
#define MINDSPORE_PI_JIT_EDGE_H_

#include <memory>
#include <vector>

namespace mindspore {
namespace pijit {
namespace grad {
class FunctionNode;
using FunctionNodePtr = std::shared_ptr<FunctionNode>;

/// \brief Edge is a class, which represent a called function in the previous/next step.
class Edge {
 public:
  /// \brief The constructor of Edge.
  ///
  /// \param[in] node The called function.
  ///
  /// \return The instance of Edge.
  explicit Edge(const FunctionNodePtr &node, size_t index) : node_(node), index_(index) {}

  /// \brief Destructor.
  virtual ~Edge() = default;

  /// \brief Get the backward function node.
  ///
  /// \return The backward function node.
  const FunctionNodePtr &GetNode() const { return node_; }

  /// \brief Set the backward function node.
  ///
  /// \param[in] node The backward function node.
  void SetNode(const FunctionNodePtr &node) { node_ = node; }

  /// \brief Get the index of the tensor as input.
  ///
  /// \return The index of the tensor as input.
  size_t GetIndex() const { return index_; }

 private:
  /// \brief The backward function node.
  FunctionNodePtr node_;
  /// \brief The index.
  size_t index_;
};

using EdgePtr = std::shared_ptr<Edge>;
using EdgePtrList = std::vector<EdgePtr>;
}  // namespace grad
}  // namespace pijit
}  // namespace mindspore
#endif  // MINDSPORE_PI_JIT_EDGE_H_
