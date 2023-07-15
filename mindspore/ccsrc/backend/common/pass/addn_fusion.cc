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

#include "backend/common/pass/addn_fusion.h"

#include <memory>

#include "mindspore/core/ops/array_ops.h"
#include "include/common/utils/anfalgo.h"

namespace mindspore {
namespace opt {
const BaseRef AddNFusion::DefinePattern() const {
  VarPtr Xs = std::make_shared<SeqVar>();
  return VectorRef({prim::kPrimAddN, Xs});
}

const AnfNodePtr AddNFusion::Process(const FuncGraphPtr &graph, const AnfNodePtr &node, const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);

  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);

  int64_t n = 0;

  for (size_t idx = kIndex1; idx < cnode->inputs().size(); ++idx) {
    auto input_node = cnode->input(idx);
    if (input_node->isa<Parameter>() || input_node->isa<ValueNode>()) {
      n += 1;
    } else if (input_node->isa<CNode>()) {
      size_t tuple_num = AnfUtils::GetOutputTensorNum(input_node);
      n += tuple_num;
    } else {
      MS_LOG(ERROR) << cnode->fullname_with_scope() << "has a unsupported input " << input_node->DebugString();
    }
  }

  common::AnfAlgo::SetNodeAttr("n", MakeValue(n), cnode);
  common::AnfAlgo::SetNodeAttr("N", MakeValue(n), cnode);

  return cnode;
}
}  // namespace opt
}  // namespace mindspore
