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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_OPT_PRE_NODE_REMOVAL_PASS_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_OPT_PRE_NODE_REMOVAL_PASS_H_

#include <memory>
#include <vector>
#include "minddata/dataset/engine/opt/pass.h"

namespace mindspore {
namespace dataset {

class DatasetOp;

/// \class RemovalPass removal_pass.h
/// \brief This is a tree pass that will remove nodes.  It uses removal_nodes to first identify which
///     nodes should be removed, and then removes them.
class NodeRemovalPass : public IRTreePass {
  /// \class RemovalNodes
  /// \brief This is a NodePass whose job is to identify which nodes should be removed.
  ///     It works in conjunction with the removal_pass.
  class RemovalNodes : public IRNodePass {
   public:
    /// \brief Constructor
    /// \param[in] removal_pass Raw pointer back to controlling tree pass
    RemovalNodes();

    /// \brief Destructor
    ~RemovalNodes() = default;

    /// \brief Perform RepeatNode removal check
    /// \param[in] node The node being visited
    /// \param[in, out] modified Indicator if the node was changed at all
    /// \return Status The status code returned
    Status Visit(std::shared_ptr<RepeatNode> node, bool *const modified) override;

    /// \brief Perform SkipNode removal check
    /// \param[in] node The node being visited
    /// \param[in, out] modified Indicator if the node was changed at all
    /// \return Status The status code returned
    Status Visit(std::shared_ptr<SkipNode> node, bool *const modified) override;

    /// \brief Perform TakeNode removal check
    /// \param[in] node The node being visited
    /// \param[in, out] modified Indicator if the node was changed at all
    /// \return Status The status code returned
    Status Visit(std::shared_ptr<TakeNode> node, bool *const modified) override;

    /// \brief Getter
    /// \return All the nodes to be removed
    std::vector<std::shared_ptr<DatasetNode>> nodes_to_remove() { return nodes_to_remove_; }

   private:
    std::vector<std::shared_ptr<DatasetNode>> nodes_to_remove_;
  };

 public:
  /// \brief Constructor
  NodeRemovalPass();

  /// \brief Destructor
  ~NodeRemovalPass() = default;

  /// \brief Runs a removal_nodes pass first to find out which nodes to remove, then removes them.
  /// \param[in, out] tree The tree to operate on.
  /// \param[in, out] Indicate of the tree was modified.
  /// \return Status The status code returned
  Status RunOnTree(std::shared_ptr<DatasetNode> root_ir, bool *const modified) override;
};
}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_OPT_PRE_NODE_REMOVAL_PASS_H_
