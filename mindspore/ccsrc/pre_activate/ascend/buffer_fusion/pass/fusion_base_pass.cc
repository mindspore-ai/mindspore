/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#include "pre_activate/ascend/buffer_fusion/pass/fusion_base_pass.h"
#include <unordered_set>
#include <memory>
#include "debug/anf_ir_dump.h"
#include "utils/context/ms_context.h"
#include "pre_activate/common/fusion_id_allocator.h"

namespace mindspore {
namespace opt {
void FusionBasePass::SetRecordFusionId(const std::unordered_set<AnfNodePtr> &record) {
  auto id = fusion_id_allocator->AllocateFusionId();
  for (auto node : record) {
    fusion_id_allocator->SetFusionId(node, id);
  }
}
bool FusionBasePass::Run(const FuncGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  auto kernel_graph = graph->cast<std::shared_ptr<session::KernelGraph>>();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  return MatchUBFusionPattern(*kernel_graph);
}
}  // namespace opt
}  // namespace mindspore
