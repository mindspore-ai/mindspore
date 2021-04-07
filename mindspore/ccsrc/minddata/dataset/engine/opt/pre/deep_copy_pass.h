/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef DATASET_ENGINE_OPT_PRE_DEEP_COPY_PASS_H_
#define DATASET_ENGINE_OPT_PRE_DEEP_COPY_PASS_H_

#include <memory>
#include <vector>
#include "minddata/dataset/engine/opt/pass.h"

namespace mindspore {
namespace dataset {

/// \class DeepCopyPass
/// \brief This pass clones a new copy of IR tree. A new copy is used in the compilation to avoid any modification to
///    the IR tree associated with the user code.
class DeepCopyPass : public IRNodePass {
 public:
  /// \brief Constructor
  DeepCopyPass();

  /// \brief Destructor
  ~DeepCopyPass() = default;

  /// \brief Clone a new copy of the node
  /// \param[in] node The node being visited
  /// \param[in, out] *modified indicates whether the node has been visited
  /// \return Status code
  Status Visit(std::shared_ptr<DatasetNode> node, bool *const modified) override;

  /// \brief Reset parent after walking its sub tree.
  /// \param[in] node The node being visited
  /// \param[in, out] *modified indicates whether the node has been visited
  /// \return Status code
  Status VisitAfter(std::shared_ptr<DatasetNode> node, bool *const modified) override;

  /// \brief Getter method to retrieve the root node
  /// \return the root node of the new cloned tree
  std::shared_ptr<RootNode> Root() { return root_; }

 private:
  std::shared_ptr<RootNode> root_;
  DatasetNode *parent_;
};
}  // namespace dataset
}  // namespace mindspore

#endif  // DATASET_ENGINE_OPT_PRE_DEEP_COPY_PASS_H_
