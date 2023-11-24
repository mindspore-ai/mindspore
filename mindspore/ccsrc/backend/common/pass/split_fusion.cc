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

#include "backend/common/pass/split_fusion.h"

#include <memory>
#include <vector>

#include "mindspore/core/ops/array_ops.h"
#include "mindspore/core/ops/op_utils.h"
#include "include/common/utils/anfalgo.h"

namespace mindspore {
namespace opt {
const BaseRef SplitFusion::DefinePattern() const {
  VarPtr Xs = std::make_shared<SeqVar>();
  return VectorRef({prim::kPrimSplit, Xs});
}

const AnfNodePtr SplitFusion::Process(const FuncGraphPtr &graph, const AnfNodePtr &node, const EquivPtr &) const {
  constexpr auto kAxis = "axis";
  constexpr auto kOutputNum = "output_num";
  constexpr auto kSizeSplits = "size_splits";

  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);

  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);

  size_t axis_idx = ops::GetInputIndexByName(common::AnfAlgo::GetCNodeName(cnode), kAxis);
  size_t out_num_idx = ops::GetInputIndexByName(common::AnfAlgo::GetCNodeName(cnode), kOutputNum);

  if (axis_idx == SIZE_MAX || out_num_idx == SIZE_MAX) {
    MS_LOG(ERROR) << "For '" << cnode->fullname_with_scope() << "', can't not find input of axis and output_num.";
    return cnode;
  }
  auto axis_node = common::AnfAlgo::GetInputNode(cnode, axis_idx);
  auto out_num_node = common::AnfAlgo::GetInputNode(cnode, out_num_idx);
  if (!utils::isa<ValueNodePtr>(axis_node) || !utils::isa<ValueNodePtr>(out_num_node)) {
    return cnode;
  }
  auto axis_v = ops::GetScalarValue<int64_t>(axis_node->cast<ValueNodePtr>()->value());
  auto out_num_v = ops::GetScalarValue<int64_t>(out_num_node->cast<ValueNodePtr>()->value());
  if (!out_num_v.has_value() || !axis_v.has_value()) {
    return cnode;
  }

  int64_t axis = axis_v.value();
  int64_t output_num_value = out_num_v.value();
  auto x_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(cnode, kIndex0);
  if (IsDynamicRank(x_shape)) {
    MS_LOG(ERROR) << "For '" << cnode->fullname_with_scope()
                  << "', can't add attr size_splits while shape of input[0] is dynamic rank.";
    return cnode;
  }

  auto rank = SizeToLong(x_shape.size());
  auto pos = axis < 0 ? LongToSize(axis + rank) : LongToSize(axis);

  std::vector<int64_t> size_splits;
  for (int64_t i = 0; i < output_num_value; ++i) {
    size_splits.push_back(x_shape[pos] / output_num_value);
  }

  common::AnfAlgo::SetNodeAttr(kSizeSplits, MakeValue(size_splits), cnode);

  return cnode;
}
}  // namespace opt
}  // namespace mindspore
