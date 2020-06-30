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

#ifndef DATASET_ENGINE_OPT_PASS_PRE_REMOVAL_NODES_H_
#define DATASET_ENGINE_OPT_PASS_PRE_REMOVAL_NODES_H_

#include <memory>
#include "dataset/engine/opt/pass.h"

namespace mindspore {
namespace dataset {

class RemovalPass;

/// \class RemovalNodes removal_nodes.h
/// \brief This is a NodePass who's job is to identify which nodes should be removed.
///     It works in conjunction with the removal_pass.
class RemovalNodes : public NodePass {
 public:
  /// \brief Constructor
  /// \param[in] removal_pass Raw pointer back to controlling tree pass
  explicit RemovalNodes(RemovalPass *removal_pass);

  /// \brief Perform ShuffleOp removal check
  /// \param[in] node The node being visited
  /// \param[inout] modified Indicator if the node was changed at all
  /// \return Status The error code return
  Status RunOnNode(std::shared_ptr<ShuffleOp> node, bool *modified) override;

 private:
  bool is_caching_;
  RemovalPass *removal_pass_;  // Back pointer to the owning removal pass
};

}  // namespace dataset
}  // namespace mindspore

#endif  // DATASET_ENGINE_OPT_PASS_PRE_REMOVAL_NODES_
