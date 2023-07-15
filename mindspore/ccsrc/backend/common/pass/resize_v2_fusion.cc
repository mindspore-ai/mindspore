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

#include "backend/common/pass/resize_v2_fusion.h"

#include <memory>
#include <string>

#include "mindspore/core/ops/image_ops.h"
#include "include/common/utils/anfalgo.h"

namespace mindspore {
namespace opt {
const BaseRef ResizeV2Fusion::DefinePattern() const {
  VarPtr Xs = std::make_shared<SeqVar>();
  return VectorRef({prim::kPrimResizeV2, Xs});
}

const AnfNodePtr ResizeV2Fusion::Process(const FuncGraphPtr &graph, const AnfNodePtr &node, const EquivPtr &) const {
  constexpr auto kCoordinateTransformationMode = "coordinate_transformation_mode";
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);

  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);

  auto mode_str = common::AnfAlgo::GetNodeAttr<std::string>(cnode, kCoordinateTransformationMode);

  // Because of the restriction of AICPU framework, use "half_pixel" will cause conflict.
  // So, use equivalent value "pytorch_half_pixel" instead.
  if (mode_str == "half_pixel") {
    common::AnfAlgo::SetNodeAttr(kCoordinateTransformationMode, MakeValue("pytorch_half_pixel"), cnode);
  }

  return cnode;
}
}  // namespace opt
}  // namespace mindspore
