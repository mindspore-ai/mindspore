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
#include "plugin/device/ascend/optimizer/ge/add_cast_to_node.h"
#include <vector>
#include <memory>
#include "mindspore/core/ops/array_ops.h"
#include "include/common/utils/anfalgo.h"
#include "mindspore/core/ops/op_utils.h"

namespace mindspore {
namespace opt {
bool DataTypeValid(const AnfNodePtr &node, const std::vector<TypeId> &type_list) {
  auto input_type_id = common::AnfAlgo::GetPrevNodeOutputInferDataType(node, kIndex0);
  bool res = std::any_of(type_list.begin(), type_list.end(),
                         [&input_type_id](const TypeId &type_id) { return input_type_id == type_id; });
  return res;
}

const AnfNodePtr AddCastToNode::Process(const FuncGraphPtr &graph, const AnfNodePtr &node, const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);

  auto op_name = common::AnfAlgo::GetCNodeName(node);
  if (prim_type_map.find(op_name) == prim_type_map.end()) {
    return node;
  }

  auto type_list = prim_type_map.at(op_name).first;
  auto output_type = prim_type_map.at(op_name).second;
  bool is_int_or_bool = DataTypeValid(node, type_list);
  if (is_int_or_bool) {
    auto shape = common::AnfAlgo::GetOutputInferShape(node, kIndex0);
    auto cast_node =
      graph->NewCNode({NewValueNode(std::make_shared<Primitive>(prim::kPrimCast->name())), cnode->input(kIndex1)});
    common::AnfAlgo::SetOutputInferTypeAndShape({output_type}, {shape}, cast_node.get());
    common::AnfAlgo::SetNodeAttr(kAttrDstType, TypeIdToType(output_type), cast_node);
    std::vector<AnfNodePtr> cast_inputs = {NewValueNode(std::make_shared<Primitive>(op_name)), cast_node};
    auto new_node = graph->NewCNode(cast_inputs);
    common::AnfAlgo::SetOutputInferTypeAndShape({output_type}, {shape}, new_node.get());
    return new_node;
  } else {
    return node;
  }
}
}  // namespace opt
}  // namespace mindspore
