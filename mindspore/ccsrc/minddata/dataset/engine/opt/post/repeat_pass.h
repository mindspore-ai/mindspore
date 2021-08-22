/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_OPT_PASS_POST_REPEAT_PASS_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_OPT_PASS_POST_REPEAT_PASS_

#include <memory>
#include <stack>
#include <utility>
#include "minddata/dataset/engine/opt/pass.h"

namespace mindspore {
namespace dataset {

/// \class RepeatPass
/// \brief This is a post pass that calculate the number of repeats the pipeline needs to fetch the data.
class RepeatPass : public IRNodePass {
 public:
  using op_stack = std::stack<std::shared_ptr<DatasetNode>>;

  /// \brief Constructor
  RepeatPass();

  /// \brief Destructor
  ~RepeatPass() = default;

  /// \brief Identifies the subtree below this node as being in a repeated path of the tree.
  /// \param[in] node The node being visited
  /// \param[in,out] modified Indicator if the node was changed at all
  /// \return Status The status code returned
  Status Visit(std::shared_ptr<RepeatNode> node, bool *const modified) override;

  /// \brief Identifies the subtree below this node as being in a repeated path of the tree.
  /// \param[in] node The node being visited
  /// \param[in,out] modified Indicator if the node was changed at all
  /// \return Status The status code returned
  Status Visit(std::shared_ptr<EpochCtrlNode> node, bool *const modified) override;

#ifndef ENABLE_ANDROID
  /// \brief Identifies the subtree below this node as being in a cache merge path
  /// \param[in] node The node being visited
  /// \param[in,out] modified Indicator if the node was changed at all
  /// \return Status The status code returned
  Status Visit(std::shared_ptr<CacheMergeNode> node, bool *const modified) override;

  /// \brief Identifies the subtree below this node as being cached
  /// \param[in] node The node being visited
  /// \param[in,out] modified Indicator if the node was changed at all
  /// \return Status The status code returned
  Status Visit(std::shared_ptr<CacheNode> node, bool *const modified) override;
#endif

  /// \brief Hooks up any identified eoe nodes under this repeat.
  /// \param[in] node The node being visited
  /// \param[in,out] modified Indicator if the node was changed at all
  /// \return Status The status code returned
  Status VisitAfter(std::shared_ptr<RepeatNode> node, bool *const modified) override;

  /// \brief Hooks up any identified eoe nodes under this repeat.
  /// \param[in] node The node being visited
  /// \param[in,out] modified Indicator if the node was changed at all
  /// \return Status The status code returned
  Status VisitAfter(std::shared_ptr<EpochCtrlNode> node, bool *const modified) override;

#ifndef ENABLE_ANDROID
  /// \brief CacheNode removes previous leaf ops and replaces them with itself
  /// \param[in] node The node being visited
  /// \param[in,out] modified Indicator if the node was changed at all
  /// \return Status The status code returned
  Status VisitAfter(std::shared_ptr<CacheNode> node, bool *const modified) override;

  /// \brief Turns off the tracking for operations under merge op
  /// \param[in] node The node being visited
  /// \param[in,out] modified Indicator if the node was changed at all
  /// \return Status The status code returned
  Status VisitAfter(std::shared_ptr<CacheMergeNode> node, bool *const modified) override;

  /// \brief Saves the lookup up in case it needs to be referenced by a repeat
  /// \param[in] node The node being visited
  /// \param[in,out] modified Indicator if the node was changed at all
  /// \return Status The status code returned
  Status VisitAfter(std::shared_ptr<CacheLookupNode> node, bool *const modified) override;
#endif

  /// \brief Sets the epoch count for TransferNode
  /// \param[in] node The node being visited
  /// \param[in,out] modified Indicator if the node was changed at all
  /// \return Status The status code returned
  Status VisitAfter(std::shared_ptr<TransferNode> node, bool *const modified) override;

  /// \brief All operators have a flag that might be set related to the repeat and any leaf nodes need to be set up
  ///     for use with a controlling repeat above it.
  /// \param[in] node The node being visited
  /// \param[in,out] modified Indicator if the node was changed at all
  /// \return Status The status code returned
  Status VisitAfter(std::shared_ptr<DatasetNode> node, bool *const modified) override;

 private:
  /// \brief Adds an operator to the cached stack save area
  /// \param node - The dataset node to add to cached stack
  /// \return Status The status code returned
  void AddToCachedNodeStack(std::shared_ptr<DatasetNode> node);

  /// \brief Pops an operator from the cached stack save area
  /// \return shared_ptr to the popped dataset node
  std::shared_ptr<DatasetNode> PopFromCachedNodeStack();

  bool is_merge_;                              // T/F if we are processing under a cache merge node
  bool is_cached_;                             // T/F is we are processing under a cache node
  int32_t num_repeats_;                        // A multiplier to the total number of repeats
  int32_t num_epochs_;                         // To save the total number of epochs
  op_stack cached_node_stacks_;                // A save area for operators under a cache node
  std::shared_ptr<DatasetNode> cache_lookup_;  // A save area for a cache lookup node
};
}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_OPT_PASS_POST_REPEAT_PASS_
