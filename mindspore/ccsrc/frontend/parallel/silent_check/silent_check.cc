/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#include "frontend/parallel/silent_check/silent_check.h"
#include "ir/graph_utils.h"
#include "ir/func_graph.h"
#include "ops/other_ops.h"

namespace mindspore {
namespace parallel {
void SilentCheck::GetLossScale() {
  MS_EXCEPTION_IF_NULL(root_);
  auto parameters = root_->parameters();
  for (const auto &param : parameters) {
    auto param_ptr = param->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(param_ptr);
    const auto &name = param_ptr->name();
    if (name == kScale_Sense) {
      loss_scale_ = param;
    }
  }
}

void SilentCheck::ModifySilentCheckOps() {
  MS_EXCEPTION_IF_NULL(root_);
  auto ret = root_->get_return();
  MS_EXCEPTION_IF_NULL(ret);
  MS_EXCEPTION_IF_NULL(mng_);
  const auto &all_nodes = DeepScopedGraphSearch(ret);
  for (const auto &node : all_nodes) {
    if (node && !IsPrimitiveCNode(node, prim::kPrimMirrorSilentCheck)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    if (loss_scale_ != nullptr) {
      mng_->SetEdge(cnode, LOSS_SCALE_INDEX, loss_scale_);
    }
  }
}
}  // namespace parallel
}  // namespace mindspore
