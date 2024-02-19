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
#include "plugin/device/ascend/optimizer/ge/reciprocal_fusion.h"
#include <vector>
#include <memory>
#include "mindspore/core/ops/array_ops.h"
#include "include/common/utils/anfalgo.h"
#include "mindspore/core/ops/op_utils.h"

namespace mindspore {
namespace opt {
const BaseRef ReciprocalFusion::DefinePattern() const {
  VarPtr input = std::make_shared<SeqVar>();
  return VectorRef({prim::kPrimReciprocal, input});
}

const AnfNodePtr ReciprocalFusion::Process(const FuncGraphPtr &graph, const AnfNodePtr &node, const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);

  auto input_type_id = common::AnfAlgo::GetPrevNodeOutputInferDataType(node, kIndex0);
  static const std::vector<TypeId> int_or_bool = {kNumberTypeUInt8,  kNumberTypeUInt16, kNumberTypeUInt32,
                                                  kNumberTypeUInt64, kNumberTypeInt8,   kNumberTypeInt16,
                                                  kNumberTypeInt32,  kNumberTypeInt64,  kNumberTypeBool};
  bool is_int_or_bool = std::any_of(int_or_bool.begin(), int_or_bool.end(),
                                    [&input_type_id](const TypeId &type_id) { return input_type_id == type_id; });
  if (is_int_or_bool) {
    auto shape = common::AnfAlgo::GetOutputInferShape(node, kIndex0);
    auto cast_node =
      graph->NewCNode({NewValueNode(std::make_shared<Primitive>(prim::kPrimCast->name())), cnode->input(kIndex1)});
    common::AnfAlgo::SetOutputInferTypeAndShape({kNumberTypeFloat32}, {shape}, cast_node.get());
    common::AnfAlgo::SetNodeAttr(kAttrDstType, TypeIdToType(kNumberTypeFloat32), cast_node);
    std::vector<AnfNodePtr> reciprocal_inputs = {
      NewValueNode(std::make_shared<Primitive>(prim::kPrimReciprocal->name())), cast_node};
    auto new_reciprocal = graph->NewCNode(reciprocal_inputs);
    common::AnfAlgo::SetOutputInferTypeAndShape({kNumberTypeFloat32}, {shape}, new_reciprocal.get());
    return new_reciprocal;
  } else {
    return node;
  }
}
}  // namespace opt
}  // namespace mindspore
