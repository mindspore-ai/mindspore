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
bool DebugModePass::DebugPass::RemoveCache(std::shared_ptr<DatasetNode> node) const {
  // remove DatasetNode cache
  bool ret = false;
  if (node->IsCached()) {
    MS_LOG(WARNING) << node->Name() << " with cache found in the debug mode. Dropping the cache."
                    << " If performance is a concern, then disable debug mode.";
    (void)node->SetDatasetCache(nullptr);
    ret = true;
  }
  if (node->IsDescendantOfCache()) {
    node->setDescendantOfCache(false);
    ret = true;
  }
  return ret;
}

Status DebugModePass::DebugPass::Visit(std::shared_ptr<MapNode> node, bool *const modified) {
  *modified = RemoveCache(node);
  if (node->GetOffload() == ManualOffloadMode::kEnabled) {
    MS_LOG(WARNING) << "Map operation with offload found in the debug mode. Ignoring offload."
                       " If performance is a concern, then disable debug mode.";
    node->SetOffload(ManualOffloadMode::kDisabled);
    *modified = true;
  }

  return Status::OK();
}

Status DebugModePass::DebugPass::Visit(std::shared_ptr<DatasetNode> node, bool *const modified) {
  *modified = RemoveCache(node);
  return Status::OK();
}

Status DebugModePass::RunOnTree(std::shared_ptr<DatasetNode> root_ir, bool *const modified) {
  RETURN_UNEXPECTED_IF_NULL(root_ir);
  RETURN_UNEXPECTED_IF_NULL(modified);

  // The debug_pass can make updates to the DebugModePass object.
  DebugPass debug_pass = DebugPass();
  RETURN_IF_NOT_OK(debug_pass.Run(root_ir, modified));

  // Debug mode requires the deterministic result. Set seed if users have not done so.
  uint32_t seed = GlobalContext::config_manager()->seed();
  if (seed == std::mt19937::default_seed) {
    int8_t kSeedValue = 1;
    MS_LOG(WARNING) << "Debug mode is enabled. Set seed to ensure deterministic results. Seed value: "
                    << std::to_string(kSeedValue) << ".";
    GlobalContext::config_manager()->set_seed(kSeedValue);
  }
  if (GlobalContext::config_manager()->get_auto_offload()) {
    MS_LOG(WARNING) << "Both debug mode and auto offload are enabled. Ignoring auto offload."
                       " If performance is a concern, then disable debug mode.";
  }
  if ((GlobalContext::config_manager()->error_samples_mode() == ErrorSamplesMode::kReplace) ||
      (GlobalContext::config_manager()->error_samples_mode() == ErrorSamplesMode::kSkip)) {
    MS_LOG(WARNING) << "Both debug mode and error samples mode are enabled. Ignoring error samples mode setting.";
  }
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
