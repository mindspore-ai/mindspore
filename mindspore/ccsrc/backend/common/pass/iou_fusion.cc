/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "backend/common/pass/iou_fusion.h"

#include <memory>

#include "mindspore/core/ops/image_ops.h"
#include "include/common/utils/anfalgo.h"

namespace mindspore {
namespace opt {
const BaseRef IOUFusion::DefinePattern() const {
  VarPtr Xs = std::make_shared<SeqVar>();
  return VectorRef({prim::kPrimIOU, Xs});
}

const AnfNodePtr IOUFusion::Process(const FuncGraphPtr &graph, const AnfNodePtr &node, const EquivPtr &) const {
  constexpr auto kEpsName = "eps";

  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);

  const auto &prim = common::AnfAlgo::GetCNodePrimitive(node);
  MS_EXCEPTION_IF_NULL(prim);

  if (prim->GetAttr(kEpsName) == nullptr) {
    constexpr float kDefaultValue = 1.0;
    prim->AddAttr(kEpsName, MakeValue(kDefaultValue));
  }

  return node;
}
}  // namespace opt
}  // namespace mindspore
