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
#include "backend/common/pass/flatten_concat_fission.h"
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <utility>
#include <functional>
#include <memory>
#include "include/common/utils/anfalgo.h"

namespace mindspore {
namespace opt {
namespace {
int64_t GetElemNum(const AnfNodePtr &input_node) {
  MS_EXCEPTION_IF_NULL(input_node);
  int64_t elem_num = 1;
  auto base_shape_ptr = input_node->Shape();
  MS_EXCEPTION_IF_NULL(base_shape_ptr);
  auto input_shape_ptr = base_shape_ptr->cast<abstract::ShapePtr>();
  if (input_shape_ptr != nullptr) {
    auto input_shape = input_shape_ptr->shape();
    elem_num = std::accumulate(input_shape.begin(), input_shape.end(), 1, std::multiplies<int64_t>());
  }
  return elem_num;
}

CNodePtr NewFlattenNode(const FuncGraphPtr &func_graph, const AnfNodePtr &input_node, int64_t elem_num) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(input_node);
  std::vector<AnfNodePtr> flatten_inputs = {NewValueNode(prim::kPrimFlatten), input_node};
  auto flatten_node = NewCNode(flatten_inputs, func_graph);

  ShapeVector shape{elem_num};
  auto input_type_id = common::AnfAlgo::GetOutputInferDataType(input_node, 0);
  auto abstract = std::make_shared<abstract::AbstractTensor>(TypeIdToType(input_type_id), shape);
  MS_EXCEPTION_IF_NULL(flatten_node);
  flatten_node->set_abstract(abstract);

  return flatten_node;
}

CNodePtr NewConcatNode(const FuncGraphPtr &func_graph, const std::pair<std::vector<AnfNodePtr>, int64_t> &node_info,
                       TypeId type_id) {
  MS_EXCEPTION_IF_NULL(func_graph);
  auto concat_node = NewCNode(node_info.first, func_graph);

  ShapeVector shape{node_info.second};
  auto abstract = std::make_shared<abstract::AbstractTensor>(TypeIdToType(type_id), shape);
  MS_EXCEPTION_IF_NULL(concat_node);
  concat_node->set_abstract(abstract);

  common::AnfAlgo::SetNodeAttr(kAttrAxis, MakeValue<int64_t>(0), concat_node);
  return concat_node;
}

CNodePtr NewMakeTupleNode(const FuncGraphPtr &func_graph, std::vector<AnfNodePtr> *concat_nodes) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(concat_nodes);
  std::vector<AbstractBasePtr> abstract_list;
  for (size_t i = 0; i < concat_nodes->size(); ++i) {
    auto concat_node = (*concat_nodes)[i];
    MS_EXCEPTION_IF_NULL(concat_node);
    abstract_list.emplace_back(concat_node->abstract());
  }
  (void)concat_nodes->insert(concat_nodes->begin(), NewValueNode(prim::kPrimMakeTuple));
  auto make_tuple_node = NewCNode(*concat_nodes, func_graph);

  auto abstract_tuple = std::make_shared<abstract::AbstractTuple>(abstract_list);
  MS_EXCEPTION_IF_NULL(make_tuple_node);
  make_tuple_node->set_abstract(abstract_tuple);

  return make_tuple_node;
}
}  // namespace

const BaseRef FlattenConcatFission::DefinePattern() const {
  VarPtr Xs = std::make_shared<SeqVar>();
  return VectorRef({prim::kPrimFlattenConcat, Xs});
}

const AnfNodePtr FlattenConcatFission::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                               const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  std::unordered_map<TypeId, std::pair<std::vector<AnfNodePtr>, int64_t>> type_id_to_node_info;
  std::vector<TypeId> output_type_id_order;
  for (size_t i = 1; i < cnode->inputs().size(); ++i) {
    auto input_node = cnode->input(i);
    MS_EXCEPTION_IF_NULL(input_node);
    int64_t elem_num = GetElemNum(input_node);
    auto flatten_node = NewFlattenNode(func_graph, input_node, elem_num);
    MS_EXCEPTION_IF_NULL(flatten_node);
    flatten_node->set_scope(cnode->scope());

    auto input_type_id = common::AnfAlgo::GetOutputInferDataType(input_node, 0);
    auto iter = type_id_to_node_info.find(input_type_id);
    if (iter == type_id_to_node_info.end()) {
      std::vector<AnfNodePtr> concat_inputs = {NewValueNode(prim::kPrimConcat), flatten_node};
      type_id_to_node_info[input_type_id] = std::make_pair(concat_inputs, elem_num);
      (void)output_type_id_order.emplace_back(input_type_id);
    } else {
      (void)iter->second.first.emplace_back(flatten_node);
      iter->second.second += elem_num;
    }
  }

  std::vector<AnfNodePtr> concat_nodes;
  for (size_t i = 0; i < output_type_id_order.size(); ++i) {
    auto type_id = output_type_id_order[i];
    auto &node_info = type_id_to_node_info[type_id];
    auto concat_node = NewConcatNode(func_graph, node_info, type_id);
    MS_EXCEPTION_IF_NULL(concat_node);
    concat_node->set_scope(cnode->scope());

    (void)concat_nodes.emplace_back(concat_node);
  }

  auto make_tuple_node = NewMakeTupleNode(func_graph, &concat_nodes);
  MS_EXCEPTION_IF_NULL(make_tuple_node);
  make_tuple_node->set_scope(cnode->scope());
  return make_tuple_node;
}
}  // namespace opt
}  // namespace mindspore
