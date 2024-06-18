/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_OPT_PRE_SKIP_PUSHDOWN_PASS_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_OPT_PRE_SKIP_PUSHDOWN_PASS_H_

#include <memory>
#include <utility>
#include <vector>
#include "minddata/dataset/engine/opt/pass.h"

namespace mindspore {
namespace dataset {
class BatchNode;
class DatasetNode;
class DatasetOp;
class MappableSourceNode;
class MapNode;
#ifndef ENABLE_ANDROID
class MindDataNode;
#endif
class NonMappableSourceNode;
class ProjectNode;
class RenameNode;
class RootNode;
class SkipNode;

/// \class SkipPushdownPass skip_pushdown_pass.h
/// \brief This is a tree pass that will push down a skip node.  It uses SkipNodes to first identify if we have a skip
/// node, and then based on the node types we observe in the tree, decide where to place the skip node (or use a
/// SequentialSampler for MappableSource nodes).
class SkipPushdownPass : public IRTreePass {
  /// \class SkipNodes
  /// \brief This is a NodePass whose job is to handle different nodes accordingly.
  ///     It works in conjunction with the SkipPushdownPass.
  class SkipNodes : public IRNodePass {
   public:
    /// \brief Constructor
    SkipNodes();

    /// \brief Destructor
    ~SkipNodes() override = default;

    /// \brief Perform skip node pushdown initiation check on a SkipNode
    /// \param[in] node The node being visited
    /// \param[in, out] modified Indicator if the node was changed at all
    /// \return Status The status code returned
    Status Visit(std::shared_ptr<SkipNode> node, bool *const modified) override;

    /// \brief Perform skip node pushdown completion check on a SkipNode
    /// \param[in] node The node being visited
    /// \param[in, out] modified Indicator if the node was changed at all
    /// \return Status The status code returned
    Status VisitAfter(std::shared_ptr<SkipNode> node, bool *const modified) override;

    /// \brief Perform skip node pushdown check on a BatchNode
    /// \param[in] node The node being visited
    /// \param[in, out] modified Indicator if the node was changed at all
    /// \return Status The status code returned
    Status Visit(std::shared_ptr<BatchNode> node, bool *const modified) override;

    /// \brief Perform skip node pushdown check on a ProjectNode
    /// \param[in] node The node being visited
    /// \param[in, out] modified Indicator if the node was changed at all
    /// \return Status The status code returned
    Status Visit(std::shared_ptr<ProjectNode> node, bool *const modified) override;

    /// \brief Perform skip node pushdown check on a RenameNode
    /// \param[in] node The node being visited
    /// \param[in, out] modified Indicator if the node was changed at all
    /// \return Status The status code returned
    Status Visit(std::shared_ptr<RenameNode> node, bool *const modified) override;

    /// \brief Perform skip node pushdown check on a MappableSourceNode
    /// \param[in] node The node being visited
    /// \param[in, out] modified Indicator if the node was changed at all
    /// \return Status The status code returned
    Status Visit(std::shared_ptr<MappableSourceNode> node, bool *const modified) override;

    /// \brief Perform skip node pushdown check on a MapNode
    /// \param[in] node The node being visited
    /// \param[in, out] modified Indicator if the node was changed at all
    /// \return Status The status code returned
    Status Visit(std::shared_ptr<MapNode> node, bool *const modified) override;

    /// \brief Perform skip node pushdown check on a NonMappableSourceNode
    /// \param[in] node The node being visited
    /// \param[in, out] modified Indicator if the node was changed at all
    /// \return Status The status code returned
    Status Visit(std::shared_ptr<NonMappableSourceNode> node, bool *const modified) override;

#ifdef ENABLE_PYTHON
    /// \brief Perform skip node pushdown check on a GeneratorNode
    /// \param[in] node The node being visited
    /// \param[in, out] modified Indicator if the node was changed at all
    /// \return Status The status code returned
    Status Visit(std::shared_ptr<GeneratorNode> node, bool *const modified) override;
#endif

    /// \brief Perform skip node pushdown check on a DatasetNode
    /// \param[in] node The node being visited
    /// \param[in, out] modified Indicator if the node was changed at all
    /// \return Status The status code returned
    Status Visit(std::shared_ptr<DatasetNode> node, bool *const modified) override;

    /// \brief Perform skip node pushdown check on a RootNode
    /// \param[in] node The node being visited
    /// \param[in, out] modified Indicator if the node was changed at all
    /// \return Status The status code returned
    Status Visit(std::shared_ptr<RootNode> node, bool *const modified) override;

    /// \brief Perform skip node pushdown completion check on a DatasetNode
    /// \param[in] node The node being visited
    /// \param[in, out] modified Indicator if the node was changed at all
    /// \return Status The status code returned
    Status VisitAfter(std::shared_ptr<DatasetNode> node, bool *const modified) override { return Status::OK(); };

    /// \brief Getter
    /// \return All the nodes where a skip node needs to be inserted above (and the skip count).
    const std::vector<std::pair<std::shared_ptr<DatasetNode>, int64_t>> &insert_skip_above() const {
      return insert_skip_above_;
    }

    /// \brief Getter
    /// \return All the nodes to be removed
    const std::vector<std::shared_ptr<DatasetNode>> &nodes_to_remove() const { return nodes_to_remove_; }

   private:
    template <class T>
    Status InsertSkipNode(std::shared_ptr<T> node) {
      CHECK_FAIL_RETURN_UNEXPECTED(skip_count_ >= 0, "The skip size cannot be negative.");
      if (skip_count_ == 0) {
        return Status::OK();
      }  // no active skip node above. normal flow

      // insert a skip node above
      (void)insert_skip_above_.emplace_back(node, skip_count_);
      skip_count_ = 0;
      return Status::OK();
    }

    std::vector<std::pair<std::shared_ptr<DatasetNode>, int64_t>> insert_skip_above_;
    std::vector<std::shared_ptr<DatasetNode>> nodes_to_remove_;
    int64_t skip_count_;
    int64_t skip_steps_;
  };

 public:
  /// \brief Constructor
  SkipPushdownPass();

  /// \brief Destructor
  ~SkipPushdownPass() override = default;

  /// \brief Runs a skip_pushdown pass to push down the skip node found in the tree (for Reset scenario).
  /// \param[in, out] tree The tree to operate on.
  /// \param[in, out] Indicate of the tree was modified.
  /// \return Status The status code returned
  Status RunOnTree(std::shared_ptr<DatasetNode> root_ir, bool *const modified) override;
};
}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_OPT_PRE_SKIP_PUSHDOWN_PASS_H_
