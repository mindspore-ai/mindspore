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

#ifndef DATASET_ENGINE_OPT_PASS_PRE_REMOVAL_PASS_H_
#define DATASET_ENGINE_OPT_PASS_PRE_REMOVAL_PASS_H_

#include <memory>
#include <vector>
#include "dataset/engine/opt/pass.h"

namespace mindspore {
namespace dataset {

class DatasetOp;

/// \class RemovalPass removal_pass.h
/// \brief This is a tree pass that will remove nodes.  It uses removal_nodes to first identify which
///     nodes should be removed, and then removes them.
class RemovalPass : public TreePass {
 public:
  /// \brief Constructor
  RemovalPass();

  /// \brief Destructor
  ~RemovalPass() = default;

  /// \brief Runs a removal_nodes pass first to find out which nodes to remove, then removes them.
  /// \param[inout] tree The tree to operate on.
  /// \param[inout] Indicate of the tree was modified.
  /// \return Status The error code return
  Status RunOnTree(ExecutionTree *tree, bool *modified) override;

  /// \brief Adds an operator to the list of operators to be removed
  /// \param[in] dataset_op The operator to add to the removal list
  void AddToRemovalList(std::shared_ptr<DatasetOp> dataset_op);

 private:
  std::vector<std::shared_ptr<DatasetOp>> removal_nodes_;
};
}  // namespace dataset
}  // namespace mindspore

#endif  // DATASET_ENGINE_OPT_PASS_PRE_REMOVAL_PASS_H_
