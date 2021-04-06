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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_OPT_POST_GENERATOR_NODE_PASS_H
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_OPT_POST_GENERATOR_NODE_PASS_H

#include <memory>
#include <utility>
#include <vector>
#include "minddata/dataset/engine/opt/pass.h"

namespace mindspore {
namespace dataset {

/// \class GeneratorNodePass repeat_pass.h
/// \brief This is a NodePass who's job is to perform setup actions for RepeatOps. A RepeatOp needs to have references
///     to the eoe-producing (typically leaf) nodes underneath it.
class GeneratorNodePass : public IRNodePass {
 public:
  /// \brief Constructor
  GeneratorNodePass();

  /// \brief Destructor
  ~GeneratorNodePass() = default;

  /// \brief Record the starting point to collect the Generator node
  /// \param[in] node The node being visited
  /// \param[in, out] modified Indicator if the node was changed at all
  /// \return Status The status code returned
  Status Visit(std::shared_ptr<RepeatNode> node, bool *const modified) override;

  /// \brief Record the starting point to collect the Generator node
  /// \param[in] node The node being visited
  /// \param[in, out] modified Indicator if the node was changed at all
  /// \return Status The status code returned
  Status Visit(std::shared_ptr<EpochCtrlNode> node, bool *const modified) override;

  /// \brief Add the Generator node to the set
  /// \param[in] node The node being visited
  /// \param[in, out] modified Indicator if the node was changed at all
  /// \return Status The status code returned
  Status Visit(std::shared_ptr<GeneratorNode> node, bool *const modified) override;

  /// \brief Add the Generator node(s) from the set to this Repeat node for run-time processing
  /// \param[in] node The node being visited
  /// \param[in, out] modified Indicator if the node was changed at all
  /// \return Status The status code returned
  Status VisitAfter(std::shared_ptr<RepeatNode> node, bool *const modified) override;

  /// \brief Add the Generator node(s) from the set to this EpochCtrl node for run-time processing
  /// \param[in] node The node being visited
  /// \param[in, out] modified Indicator if the node was changed at all
  /// \return Status The status code returned
  Status VisitAfter(std::shared_ptr<EpochCtrlNode> node, bool *const modified) override;

 private:
  std::vector<std::shared_ptr<RepeatNode>> repeat_ancestors_;
};
}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_OPT_POST_GENERATOR_NODE_PASS_H
