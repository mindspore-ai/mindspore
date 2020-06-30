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

#include <memory>
#include "dataset/engine/opt/pre/removal_nodes.h"
#include "dataset/engine/opt/pre/removal_pass.h"
#include "dataset/engine/datasetops/shuffle_op.h"

namespace mindspore {
namespace dataset {

RemovalNodes::RemovalNodes(RemovalPass *removal_pass) : removal_pass_(removal_pass), is_caching_(false) {}

// Perform ShuffleOp removal check.
Status RemovalNodes::RunOnNode(std::shared_ptr<ShuffleOp> node, bool *modified) {
  *modified = false;
  // If we are in a cache descendant tree, then this shuffle op needs to be removed
  if (is_caching_) {
    MS_LOG(DEBUG) << "ShuffleOp identified for removal (CacheOp is in ascendant tree)";
    if (removal_pass_) {
      removal_pass_->AddToRemovalList(std::static_pointer_cast<DatasetOp>(node));
    } else {
      return Status(StatusCode::kUnexpectedError, __LINE__, __FILE__, "Back reference to removal pass is missing!");
    }
  }
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
