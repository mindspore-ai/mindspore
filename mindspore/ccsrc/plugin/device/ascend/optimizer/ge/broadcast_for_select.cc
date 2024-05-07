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
#include "plugin/device/ascend/optimizer/ge/broadcast_for_select.h"
#include <vector>
#include <memory>
#include <algorithm>
#include "mindspore/core/ops/array_ops.h"
#include "include/common/utils/anfalgo.h"
#include "include/backend/anf_runtime_algorithm.h"

namespace mindspore {
namespace opt {
namespace {
ShapeVector GetSelectInputShape(const AnfNodePtr &input) {
  MS_EXCEPTION_IF_NULL(input);
  auto input_base_shape = input->Shape();
  MS_EXCEPTION_IF_NULL(input_base_shape);
  auto input_shape = input_base_shape->cast<abstract::ShapePtr>();
  MS_EXCEPTION_IF_NULL(input_shape);
  return input_shape->shape();
}

ShapeVector CalcBroadcastShape(AnfNodePtr cond, AnfNodePtr x, AnfNodePtr y) {
  auto cond_shape = GetSelectInputShape(cond);
  auto x_shape = GetSelectInputShape(x);
  auto y_shape = GetSelectInputShape(y);
  auto cond_size = cond_shape.size();
  auto x_size = x_shape.size();
  auto y_size = y_shape.size();
  ShapeVector broadcast_shape =
    cond_size > x_size ? cond_size > y_size ? cond_shape : y_shape : x_size > y_size ? x_shape : y_shape;
  auto n = broadcast_shape.size();
  for (size_t i = n; i > 0; --i) {
    auto cond_i = cond_size < i ? 1 : cond_shape[cond_size - i];
    auto x_i = x_size < i ? 1 : x_shape[x_size - i];
    auto y_i = y_size < i ? 1 : y_shape[y_size - i];
    auto broadcost_i = std::max(cond_i, std::max(x_i, y_i));
    if (cond_i != broadcost_i && cond_i != 1) {
      MS_EXCEPTION(ValueError) << "For select, condition input can not broadcast at index " << i;
    }
    if (x_i != broadcost_i && x_i != 1) {
      MS_EXCEPTION(ValueError) << "For select, x input can not broadcast at index " << i;
    }
    if (y_i != broadcost_i && y_i != 1) {
      MS_EXCEPTION(ValueError) << "For select, y input can not broadcast at index " << i;
    }
    broadcast_shape[n - i] = broadcost_i;
  }
  return broadcast_shape;
}

AnfNodePtr AddBroadCastToNode(const FuncGraphPtr &func_graph, const AnfNodePtr &input_node,
                              const std::vector<int64_t> &broad_shape) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(input_node);
  auto input_shape = GetSelectInputShape(input_node);
  if (input_shape == broad_shape) {
    return input_node;
  }

  auto input_type = common::AnfAlgo::GetOutputInferDataType(input_node, 0);
  auto shape_node = opt::CreateValueNodeWithKernelInfo(func_graph, MakeValue(broad_shape));

  std::vector<AnfNodePtr> broadcastto_inputs = {
    NewValueNode(std::make_shared<Primitive>(prim::kPrimBroadcastTo->name())), input_node, shape_node};
  CNodePtr broadcastto_node = NewCNode(broadcastto_inputs, func_graph);
  MS_EXCEPTION_IF_NULL(broadcastto_node);
  broadcastto_node->set_scope(input_node->scope());
  broadcastto_node->set_abstract(input_node->abstract());
  common::AnfAlgo::SetOutputInferTypeAndShape({input_type}, {broad_shape}, broadcastto_node.get());
  return broadcastto_node->cast<AnfNodePtr>();
}

CNodePtr AddSelectNode(const FuncGraphPtr &func_graph, const AnfNodePtr &cond_node, const AnfNodePtr &x_node,
                       const AnfNodePtr &y_node, const CNodePtr &select_node, const std::vector<int64_t> &broad_shape) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(cond_node);
  MS_EXCEPTION_IF_NULL(x_node);
  MS_EXCEPTION_IF_NULL(y_node);
  MS_EXCEPTION_IF_NULL(select_node);
  auto input_type = common::AnfAlgo::GetOutputInferDataType(select_node, 0);

  std::vector<AnfNodePtr> select_inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimSelect->name())),
                                           cond_node, x_node, y_node};
  CNodePtr out_node = NewCNode(select_inputs, func_graph);
  MS_EXCEPTION_IF_NULL(out_node);
  out_node->set_scope(select_node->scope());
  out_node->set_abstract(select_node->abstract());
  common::AnfAlgo::SetOutputInferTypeAndShape({input_type}, {broad_shape}, out_node.get());
  return out_node;
}
}  // namespace

const BaseRef BroadCastForSelect::DefinePattern() const {
  VarPtr inputs = std::make_shared<SeqVar>();
  return VectorRef({prim::kPrimSelect, inputs});
}

const AnfNodePtr BroadCastForSelect::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                             const EquivPtr &) const {
  // Select(...) ===> inputs -> CalcBroadcastShape -> BroadCastTo -> Select(...)
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  auto select_node = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(select_node);
  // get broadcast shape
  auto cond = select_node->input(kIndex1);
  auto x = select_node->input(kIndex2);
  auto y = select_node->input(kIndex3);
  auto output_shape = CalcBroadcastShape(cond, x, y);
  // do BroadCast
  auto new_cond = AddBroadCastToNode(graph, cond, output_shape);
  auto new_x = AddBroadCastToNode(graph, x, output_shape);
  auto new_y = AddBroadCastToNode(graph, y, output_shape);
  auto out_node = AddSelectNode(graph, new_cond, new_x, new_y, select_node, output_shape);
  return out_node;
}
}  // namespace opt
}  // namespace mindspore
