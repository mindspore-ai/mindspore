/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/ascend/optimizer/ge/convert_resize_nearest_neighbor_x_dtype.h"

#include <vector>
#include <memory>

#include "backend/common/session/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"

namespace mindspore {
namespace opt {
namespace {
bool NeedConvert(const BaseRef &ref) {
  if (utils::isa<AnfNodePtr>(ref)) {
    AnfNodePtr node = utils::cast<AnfNodePtr>(ref);
    MS_EXCEPTION_IF_NULL(node);
    if (IsPrimitive(node, prim::kPrimResizeNearestNeighborV2) ||
        IsPrimitive(node, prim::kPrimResizeNearestNeighborV2Grad)) {
      return true;
    }
  }
  return false;
}
}  // namespace
const BaseRef ConvertResizeNearestNeighborXDtype::DefinePattern() const {
  VarPtr convert = std::make_shared<CondVar>(NeedConvert);
  VarPtr inputs = std::make_shared<SeqVar>();
  return VectorRef({convert, inputs});
}

// Convert ResizeNearestNeighborX tuple input from int64 to int32.
const AnfNodePtr ConvertResizeNearestNeighborXDtype::Process(const FuncGraphPtr &, const AnfNodePtr &node,
                                                             const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto input_node = cnode->input(kIndex2);
  MS_EXCEPTION_IF_NULL(input_node);
  auto value_ptr = input_node->cast<ValueNodePtr>();
  MS_EXCEPTION_IF_NULL(value_ptr);
  auto input_value = value_ptr->value();
  MS_EXCEPTION_IF_NULL(input_value);
  if (input_value->isa<ValueTuple>()) {
    auto value_tuple = input_value->cast<ValueTuplePtr>();
    MS_EXCEPTION_IF_NULL(value_tuple);
    std::vector<ValuePtr> values{};
    for (const auto &elem : value_tuple->value()) {
      if (elem->isa<Int64Imm>()) {
        (void)values.emplace_back(MakeValue(static_cast<int32_t>(GetValue<int64_t>(elem))));
      } else if (elem->isa<Int32Imm>()) {
        (void)values.emplace_back(MakeValue(GetValue<int32_t>(elem)));
      } else {
        MS_LOG(EXCEPTION) << "Convert int64 to int32 failed for wrong value type. node: " << node->DebugString()
                          << ", value type: " << elem->type_name();
      }
    }
    auto new_tuple = std::make_shared<ValueTuple>(values);
    auto new_value_node = std::make_shared<ValueNode>(new_tuple);
    new_value_node->set_abstract(new_tuple->ToAbstract());
    cnode->set_input(kIndex2, new_value_node);
  } else {
    MS_LOG(EXCEPTION) << "Node: " << node->DebugString() << " second input is not tuple type.";
  }
  return node;
}
}  // namespace opt
}  // namespace mindspore
