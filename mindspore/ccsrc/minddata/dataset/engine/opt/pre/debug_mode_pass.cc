/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "minddata/dataset/engine/opt/pre/debug_mode_pass.h"

#include <string>
#include "minddata/dataset/engine/ir/datasetops/map_node.h"
#include "minddata/dataset/engine/ir/datasetops/root_node.h"
#include "minddata/dataset/include/dataset/datasets.h"

namespace mindspore {
namespace dataset {
bool DebugModePass::RemoveCacheAndOffload(std::shared_ptr<DatasetNode> node) {
  // remove DatasetNode cache
  bool ret = false;
  if (node->IsCached()) {
    MS_LOG(WARNING) << node->Name() << " with cache found in the debug mode. Dropping the cache."
                    << " If performance is a concern, then disable debug mode.";
    node->SetDatasetCache(nullptr);
    ret = true;
  }
  if (node->IsDescendantOfCache()) {
    node->setDescendantOfCache(false);
    ret = true;
  }
  if (GlobalContext::config_manager()->get_auto_offload()) {
    MS_LOG(WARNING) << "Both debug mode and auto offload are enabled. Disabling auto offload."
                       "If performance is a concern, then disable debug mode and re-enable auto offload.";
    GlobalContext::config_manager()->set_auto_offload(false);
  }
  return ret;
}

Status DebugModePass::Visit(std::shared_ptr<MapNode> node, bool *const modified) {
  *modified = RemoveCacheAndOffload(node);
  if (node->GetOffload() == ManualOffloadMode::kEnabled) {
    MS_LOG(WARNING) << "Map operation with offload found in the debug mode. Ignoring offload."
                       "If performance is a concern, then disable debug mode.";
    node->SetOffload(ManualOffloadMode::kDisabled);
    *modified = true;
  }
  return Status::OK();
}

Status DebugModePass::Visit(std::shared_ptr<DatasetNode> node, bool *const modified) {
  *modified = RemoveCacheAndOffload(node);
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
