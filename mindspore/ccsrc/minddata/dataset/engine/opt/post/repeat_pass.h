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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_OPT_PASS_POST_REPEAT_PASS_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_OPT_PASS_POST_REPEAT_PASS_

#include <memory>
#include <stack>
#include <utility>
#include "minddata/dataset/engine/opt/pass.h"

namespace mindspore {
namespace dataset {

/// \class RepeatPass repeat_pass.h
/// \brief This is a NodePass who's job is to perform setup actions for RepeatOps. A RepeatOp needs to have references
///     to the eoe-producing (typically leaf) nodes underneath it.
class RepeatPass : public NodePass {
 public:
  using op_stack = std::stack<std::shared_ptr<DatasetOp>>;

  /// \brief Constructor
  RepeatPass();

  /// \brief Destructor
  ~RepeatPass() = default;

  /// \brief Identifies the subtree below this node as being in a repeated path of the tree.
  /// \param[in] node The node being visited
  /// \param[inout] modified Indicator if the node was changed at all
  /// \return Status The status code returned
  Status PreRunOnNode(std::shared_ptr<RepeatOp> node, bool *const modified) override;

  /// \brief Identifies the subtree below this node as being in a repeated path of the tree.
  /// \param[in] node The node being visited
  /// \param[inout] modified Indicator if the node was changed at all
  /// \return Status The status code returned
  Status PreRunOnNode(std::shared_ptr<EpochCtrlOp> node, bool *const modified) override;

  /// \brief Identifies the subtree below this node as being in a cache merge path
  /// \param[in] node The node being visited
  /// \param[inout] modified Indicator if the node was changed at all
  /// \return Status The status code returned
  Status PreRunOnNode(std::shared_ptr<CacheMergeOp> node, bool *const modified) override;

  /// \brief Identifies the subtree below this node as being cached
  /// \param[in] node The node being visited
  /// \param[inout] modified Indicator if the node was changed at all
  /// \return Status The status code returned
  Status PreRunOnNode(std::shared_ptr<CacheOp> node, bool *const modified) override;

  /// \brief Hooks up any identified eoe nodes under this repeat.
  /// \param[in] node The node being visited
  /// \param[inout] modified Indicator if the node was changed at all
  /// \return Status The status code returned
  Status RunOnNode(std::shared_ptr<RepeatOp> node, bool *const modified) override;

  /// \brief Hooks up any identified eoe nodes under this repeat.
  /// \param[in] node The node being visited
  /// \param[inout] modified Indicator if the node was changed at all
  /// \return Status The status code returned
  Status RunOnNode(std::shared_ptr<EpochCtrlOp> node, bool *const modified) override;

  /// \brief CacheOp removes previous leaf ops and replaces them with itself
  /// \param[in] node The node being visited
  /// \param[inout] modified Indicator if the node was changed at all
  /// \return Status The status code returned
  Status RunOnNode(std::shared_ptr<CacheOp> node, bool *const modified) override;

  /// \brief Turns of the tracking for operations under merge op
  /// \param[in] node The node being visited
  /// \param[inout] modified Indicator if the node was changed at all
  /// \return Status The status code returned
  Status RunOnNode(std::shared_ptr<CacheMergeOp> node, bool *const modified) override;

  /// \brief Saves the lookup up in case it needs to be referenced by a repeat
  /// \param[in] node The node being visited
  /// \param[inout] modified Indicator if the node was changed at all
  /// \return Status The status code returned
  Status RunOnNode(std::shared_ptr<CacheLookupOp> node, bool *const modified) override;

  /// \brief Set the epoch count for DeviceQueue
  /// \param[in] node The node being visited
  /// \param[inout] modified Indicator if the node was changed at all
  /// \return Status The status code returned
  Status RunOnNode(std::shared_ptr<DeviceQueueOp> node, bool *const modified) override;

  /// \brief Special case for GeneratorOp
  /// \param[in] node The node being visited
  /// \param[inout] modified Indicator if the node was changed at all
  /// \return Status The status code returned
  Status RunOnNode(std::shared_ptr<GeneratorOp> node, bool *const modified) override;

  /// \brief All operators have a flag that might be set related to the repeat and any leaf nodes need to be set up
  ///     for use with a controlling repeat above it.
  /// \param[in] node The node being visited
  /// \param[inout] modified Indicator if the node was changed at all
  /// \return Status The status code returned
  Status RunOnNode(std::shared_ptr<DatasetOp> node, bool *const modified) override;

 private:
  /// \brief Adds an operator to the cached operator stack save area
  /// \param op - The dataset op to work add to cached stack
  /// \return Status The status code returned
  void AddToCachedOpStack(std::shared_ptr<DatasetOp> dataset_op);

  /// \brief Pops an operator from the cached operator stack save area
  /// \return shared_ptr to the popped operator
  std::shared_ptr<DatasetOp> PopFromCachedOpStack();

  bool is_merge_;                            // T/F if we are processing under a cache merge op
  bool is_cached_;                           // T/F is we are processing under a cache op
  int32_t num_repeats_;                      // A multiplier to the total number of repeats
  int32_t num_epochs_;                       // To save the total number of epochs
  op_stack cached_op_stacks_;                // A save area for ops under a cache op
  std::shared_ptr<DatasetOp> cache_lookup_;  // A save area for a cache lookup op
};
}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_OPT_PASS_POST_REPEAT_PASS_
