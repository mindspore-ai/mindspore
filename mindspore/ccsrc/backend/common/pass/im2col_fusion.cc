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

#include "backend/common/pass/im2col_fusion.h"

#include <vector>
#include <memory>
#include <string>

#include "utils/ms_context.h"
#include "mindspore/core/ops/image_ops.h"
#include "include/common/utils/anfalgo.h"

namespace mindspore {
namespace opt {
const BaseRef Im2ColFusion::DefinePattern() const {
  VarPtr Xs = std::make_shared<SeqVar>();
  return VectorRef({prim::kPrimIm2Col, Xs});
}

const AnfNodePtr Im2ColFusion::Process(const FuncGraphPtr &graph, const AnfNodePtr &node, const EquivPtr &) const {
  constexpr auto kPads = "pads";

  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);

  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);

  auto pads = common::AnfAlgo::GetNodeAttr<std::vector<int64_t>>(cnode, kPads);

  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  // padding pads to 4 for tbe
  if (context_ptr->get_param<std::string>(MS_CTX_DEVICE_TARGET) == kAscendDevice) {
    std::vector<int64_t> padding_pads{pads.front(), pads.front(), pads.back(), pads.back()};
    common::AnfAlgo::SetNodeAttr(kPads, MakeValue(padding_pads), cnode);
  }

  return cnode;
}
}  // namespace opt
}  // namespace mindspore
