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
#include <string>
#include "include/common/utils/anfalgo.h"
#include "include/backend/anf_runtime_algorithm.h"

namespace mindspore {
namespace opt {
namespace {
constexpr auto kAttrFusionSize = "fusion_size";

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

  size_t input_num = node_info.first.size() - 1;
  common::AnfAlgo::SetNodeAttr(kAttrAxis, MakeValue<int64_t>(0), concat_node);
  common::AnfAlgo::SetNodeAttr(kAttrInputNums, MakeValue<int64_t>(UlongToLong(input_num)), concat_node);
  std::vector<int64_t> dyn_input_size{UlongToLong(input_num)};
  common::AnfAlgo::SetNodeAttr(kAttrDynInputSizes, MakeValue(dyn_input_size), concat_node);
  return concat_node;
}

CNodePtr NewMakeTupleNode(const FuncGraphPtr &func_graph, std::vector<AnfNodePtr> *concat_nodes) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(concat_nodes);
  std::vector<AbstractBasePtr> abstract_list;
  for (size_t i = 0; i < concat_nodes->size(); ++i) {
    auto concat_node = (*concat_nodes)[i];
    MS_EXCEPTION_IF_NULL(concat_node);
    (void)abstract_list.emplace_back(concat_node->abstract());
  }
  (void)concat_nodes->insert(concat_nodes->cbegin(), NewValueNode(prim::kPrimMakeTuple));
  auto make_tuple_node = NewCNode(*concat_nodes, func_graph);

  auto abstract_tuple = std::make_shared<abstract::AbstractTuple>(abstract_list);
  MS_EXCEPTION_IF_NULL(make_tuple_node);
  make_tuple_node->set_abstract(abstract_tuple);

  return make_tuple_node;
}

size_t GetFusionSize(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto prim = GetCNodePrimitive(node);
  MS_EXCEPTION_IF_NULL(prim);
  if (prim->HasAttr(kAttrFusionSize)) {
    auto block_size_int64 = GetValue<int64_t>(prim->GetAttr(kAttrFusionSize));
    if (block_size_int64 > 0) {
      return LongToSize(block_size_int64);
    }
  }
  return 0;
}

void ExpandFlattenConcatTupleInput(const FuncGraphPtr &graph, const CNodePtr &cnode_ptr) {
  MS_EXCEPTION_IF_NULL(cnode_ptr);
  MS_EXCEPTION_IF_NULL(graph);
  if (!common::AnfAlgo::CheckPrimitiveType(cnode_ptr, prim::kPrimFlattenConcat)) {
    return;
  }
  std::vector<AnfNodePtr> plant_inputs;
  std::vector<int64_t> dyn_input_sizes;
  plant_inputs.push_back(common::AnfAlgo::GetCNodePrimitiveNode(cnode_ptr));
  size_t input_num = cnode_ptr->inputs().size() - 1;
  for (size_t i = 0; i < input_num; ++i) {
    auto input_node = common::AnfAlgo::GetInputNode(cnode_ptr, i);
    MS_EXCEPTION_IF_NULL(input_node);
    bool output_is_tuple = common::AnfAlgo::IsTupleOutput(input_node);
    if (output_is_tuple) {
      auto dyn_input_size = SplitTupleInputs(graph, input_node, &plant_inputs);
      if (dyn_input_size == 0) {
        dyn_input_sizes.push_back(-1);
        plant_inputs.push_back(input_node);
      } else {
        (void)dyn_input_sizes.emplace_back(dyn_input_size);
      }
    } else {
      dyn_input_sizes.push_back(-1);
      plant_inputs.push_back(input_node);
    }
  }
  // Expand the inputs and replace the original inputs.
  cnode_ptr->set_inputs(plant_inputs);
}
}  // namespace

std::vector<std::string> FlattenConcatFission::MustExistPrimitiveName() const {
  std::vector<std::string> ret;
  ret.emplace_back(prim::kPrimFlattenConcat->name());
  return ret;
}

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

  // After ConvertTupleInputToDynamicInput pass is removed, we need to expand tuple inputs for FlattenConcat op here.
  ExpandFlattenConcatTupleInput(func_graph, cnode);

  std::unordered_map<TypeId, std::pair<std::vector<AnfNodePtr>, int64_t>> type_id_to_node_info;
  std::vector<TypeId> output_type_id_order;
  size_t block_size = GetFusionSize(node);

  std::vector<AnfNodePtr> concat_nodes;
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
      std::vector<AnfNodePtr> concat_inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimConcat->name())),
                                               flatten_node};
      type_id_to_node_info[input_type_id] = std::make_pair(concat_inputs, elem_num);
      (void)output_type_id_order.emplace_back(input_type_id);
    } else {
      if (block_size > 0 &&
          LongToSize(iter->second.second + elem_num) * abstract::TypeIdSize(input_type_id) > block_size) {
        auto concat_node = NewConcatNode(func_graph, iter->second, input_type_id);
        MS_EXCEPTION_IF_NULL(concat_node);
        concat_node->set_scope(cnode->scope());
        (void)concat_nodes.emplace_back(concat_node);
        iter->second.second = elem_num;
        iter->second.first = {NewValueNode(std::make_shared<Primitive>(prim::kPrimConcat->name())), flatten_node};
      } else {
        (void)iter->second.first.emplace_back(flatten_node);
        iter->second.second += elem_num;
      }
    }
  }

  for (auto const &type_id : output_type_id_order) {
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
