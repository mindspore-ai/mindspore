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
#include "plugin/device/ascend/optimizer/ge/squeeze_axis_ge.h"
#include <memory>
#include <string>
#include <vector>
#include "include/backend/optimizer/helper.h"
#include "include/common/utils/anfalgo.h"
#include "mindspore/core/ops/array_ops.h"
#include "utils/anf_utils.h"

namespace mindspore {
namespace opt {
const BaseRef SqueezeAxisGe::DefinePattern() const {
  VarPtr Xs = std::make_shared<SeqVar>();
  return VectorRef({prim::kPrimSqueeze, Xs});
}

const AnfNodePtr SqueezeAxisGe::Process(const FuncGraphPtr &graph, const AnfNodePtr &node, const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  auto squeeze_cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(squeeze_cnode);
  auto prim = common::AnfAlgo::GetCNodePrimitive(squeeze_cnode);
  MS_EXCEPTION_IF_NULL(prim);
  auto axis_value = prim->GetAttr(kAttrAxis);
  MS_EXCEPTION_IF_NULL(axis_value);
  MS_EXCEPTION_IF_CHECK_FAIL(axis_value->isa<ValueSequence>(),
                             "Squeeze node axis attr error, squeeze node: " + squeeze_cnode->DebugString() +
                               ", axis value: " + axis_value->ToString());
  auto &value_sequence = axis_value->cast<ValueSequencePtr>()->value();
  auto shape_vec = common::AnfAlgo::GetOutputInferShape(squeeze_cnode->input(1), 0);
  const auto dim = shape_vec.size();
  std::vector<int64_t> axis;
  if (value_sequence.empty()) {
    for (size_t i = 0; i < dim; ++i) {
      if (shape_vec[i] != 1) {
        continue;
      }
      (void)axis.emplace_back(i);
    }
    prim->set_attr(kAttrAxis, MakeValue(axis));
    return node;
  }

  for (const auto &value : value_sequence) {
    auto axis_data = AnfUtils::GetIntValue(value);
    auto real_idx = (axis_data < 0) ? axis_data + SizeToLong(dim) : axis_data;
    (void)axis.emplace_back(real_idx);
  }
  prim->set_attr(kAttrAxis, MakeValue(axis));
  return node;
}
}  // namespace opt
}  // namespace mindspore
