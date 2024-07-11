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

#include "plugin/device/ascend/optimizer/ir_fusion/shape_reshape_fusion.h"
#include <set>
#include <queue>
#include <memory>
#include <vector>
#include "mindspore/core/ops/reshape_ext.h"
#include "mindspore/core/ops/scalar_graph_holder.h"
#include "mindspore/core/ops/array_ops.h"
#include "mindspore/core/utils/ms_context.h"
#include "include/common/utils/anfalgo.h"
#include "mindspore/ccsrc/include/common/utils/utils.h"
#include "plugin/device/ascend/optimizer/get_value_helper.h"

namespace mindspore {
namespace opt {
namespace {
std::set<PrimitivePtr> fusion_pattern_nodes = {
  prim::kPrimShape,     prim::kPrimReshape,       prim::kPrimTupleGetItem,   prim::kPrimRealTupleGetItem,
  prim::kPrimMakeTuple, prim::kPrimRealMakeTuple, prim::kPrimScalarAdd,      prim::kPrimScalarSub,
  prim::kPrimScalarMul, prim::kPrimScalarDiv,     prim::kPrimScalarFloorDiv,
};

bool GetScalarValueFromNode(const ValueNodePtr &v_node, int64_t *v) {
  // For the ShapeReshapeFusion, the ValueNode should be int64 scalar.
  auto value = v_node->value();
  if (value->isa<mindspore::Int64Imm>()) {
    *v = GetValue<int64_t>(value);
    return true;
  }
  if (value->isa<mindspore::Int32Imm>()) {
    *v = IntToSize(GetValue<int32_t>(value));
    return true;
  }
  return false;
}

bool CheckNodeInSubGraph(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (node->isa<ValueNode>()) {
    int64_t v;
    return GetScalarValueFromNode(node->cast<ValueNodePtr>(), &v);
  }
  if (node->isa<CNode>()) {
    auto cnode = node->cast<CNodePtr>();
    auto primitive = GetValueNode<PrimitivePtr>(cnode->input(0));
    return std::any_of(fusion_pattern_nodes.begin(), fusion_pattern_nodes.end(),
                       [=](const PrimitivePtr &prim) { return IsPrimitiveEquals(primitive, prim); });
  }
  return false;
}

void FindFusionPatternNodes(const CNodePtr &reshape_node, std::vector<AnfNodePtr> *fusion_nodes) {
  // Get the Fusion nodes. The node should be in the whitelists or ValueNode.
  auto shape = reshape_node->input(kIndex2);
  if (!IsPrimitiveCNode(shape, prim::kPrimMakeTuple) && !IsPrimitiveCNode(shape, prim::kPrimRealMakeTuple)) {
    MS_LOG(DEBUG) << "For ShapeReshapeFusion, the second input of Reshape should be scalar tuple.";
    return;
  }
  std::queue<AnfNodePtr> node_queue;
  std::set<AnfNodePtr> visited;

  auto make_tuple_node = shape->cast<CNodePtr>();
  node_queue.push(make_tuple_node);
  visited.insert(make_tuple_node);

  while (!node_queue.empty()) {
    auto cur_node = node_queue.front();
    node_queue.pop();
    if (!CheckNodeInSubGraph(cur_node)) {
      MS_LOG(DEBUG) << "For ShapeReshapeFusion, the valid node should be in the whitelist or scalar ValueNode.";
      continue;
    }
    if (std::find(fusion_nodes->begin(), fusion_nodes->end(), cur_node) == fusion_nodes->end()) {
      fusion_nodes->push_back(cur_node);
    }
    // The Shape Node should be the starting of the subgraph.
    if (IsPrimitiveCNode(cur_node, prim::kPrimShape)) {
      continue;
    }

    if (cur_node->isa<CNode>()) {
      auto cur_cnode = cur_node->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(cur_cnode);
      for (size_t i = 1; i < cur_cnode->size(); ++i) {
        auto input = cur_cnode->inputs().at(i);
        if (visited.find(input) == visited.end()) {
          node_queue.push(input);
          visited.insert(input);
        }
      }
    }
  }
  fusion_nodes->push_back(reshape_node);
}

void TopoSortNodes(std::vector<AnfNodePtr> *fusion_nodes, std::vector<AnfNodePtr> *input_shape_nodes) {
  auto nodes = *fusion_nodes;
  fusion_nodes->clear();

  std::vector<int> in_degree(nodes.size(), 0);
  for (size_t i = 0; i < nodes.size(); ++i) {
    if (nodes[i]->isa<CNode>()) {
      auto cnode = nodes[i]->cast<CNodePtr>();
      for (size_t j = 1; j < cnode->size(); ++j) {
        auto in = cnode->inputs().at(j);
        if (find(nodes.begin(), nodes.end(), in) != nodes.end()) {
          in_degree[i]++;
        }
      }
    }
  }

  std::queue<AnfNodePtr> node_queue;  // The node with zero in_degree.
  for (size_t i = 0; i < in_degree.size(); ++i) {
    if (in_degree[i] == 0) {
      node_queue.push(nodes[i]);
      if (IsPrimitiveCNode(nodes[i], prim::kPrimShape)) {
        input_shape_nodes->push_back(nodes[i]);
      } else if (!nodes[i]->isa<ValueNode>()) {
        MS_LOG(DEBUG)
          << "The starting nodes in ShapeReshapeFusion pattern subgraph should be Shape CNode or scalar ValueNode.";
        return;
      }
    }
  }

  while (!node_queue.empty()) {
    auto cur_node = node_queue.front();
    fusion_nodes->push_back(cur_node);
    node_queue.pop();
    for (size_t i = 0; i < nodes.size(); ++i) {
      if (nodes[i]->isa<CNode>()) {
        auto cnode = nodes[i]->cast<CNodePtr>();
        if (find(cnode->inputs().begin() + 1, cnode->inputs().end(), cur_node) != cnode->inputs().end()) {
          in_degree[i]--;
          if (in_degree[i] == 0) {
            node_queue.push(nodes[i]);
          }
        }
      }
    }
  }

  if (fusion_nodes->size() != nodes.size()) {
    MS_LOG(DEBUG) << "The Nodes in ShapeReshapeFusion pattern subgraph can't be topologically sorted.";
    fusion_nodes->clear();
  }
}

std::shared_ptr<ops::ScalarGraphHolder> CreateScalarGraph(const CNodePtr &reshape_node) {
  // Get the subgraph nodes, match the fusion pattern.
  std::vector<AnfNodePtr> fusion_nodes;
  FindFusionPatternNodes(reshape_node, &fusion_nodes);
  if (fusion_nodes.empty()) {
    MS_LOG(DEBUG) << "The fusion_nodes in ShapeReshapeFusion pattern is empty.";
    return nullptr;
  }

  // 1. Topo Sort the fusion nodes.
  // 2. Get the input nodes for subgraph. The input nodes should be Shape or ValueNode. Only return input Shape node.
  std::vector<AnfNodePtr> input_shape_nodes;
  TopoSortNodes(&fusion_nodes, &input_shape_nodes);
  if (fusion_nodes.empty()) {
    MS_LOG(DEBUG) << "Topo sort fusion_nodes failed.";
    return nullptr;
  }

  auto scalar_graph_holder = std::make_shared<ops::ScalarGraphHolder>();
  MS_EXCEPTION_IF_NULL(scalar_graph_holder);
  auto ret = scalar_graph_holder->Init(fusion_nodes);
  if (!ret) {
    MS_LOG(DEBUG) << "Init ScalarGraph failed.";
    return nullptr;
  }
  scalar_graph_holder->SetInputShapeNodes(input_shape_nodes);
  return scalar_graph_holder;
}
}  // namespace

const BaseRef ShapeReshapeFusion::DefinePattern() const {
  VarPtr shape_tuple = std::make_shared<SeqVar>();
  VectorRef make_tuple = VectorRef({prim::kPrimRealMakeTuple, shape_tuple});
  return VectorRef({std::make_shared<Primitive>("Reshape"), reshape_input_, make_tuple});
}

const AnfNodePtr ShapeReshapeFusion::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                             const EquivPtr &equiv) const {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (!ms_context->IsEnableInferBoost()) {
    return nullptr;
  }

  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(equiv);

  auto cnode = node->cast<CNodePtr>();  // reshape cnode
  MS_EXCEPTION_IF_NULL(cnode);
  auto scalar_graph_holder = CreateScalarGraph(cnode);
  if (scalar_graph_holder == nullptr) {
    MS_LOG(INFO) << "There is no change in ShapeReshapeFusion.";
    return node;
  }

  // create new primitive ReshapeExt
  auto prim = std::make_shared<Primitive>("ReshapeExt");
  std::vector<AnfNodePtr> inputs = {NewValueNode(prim), utils::cast<AnfNodePtr>((*equiv)[reshape_input_])};
  size_t index = kIndex2;
  std::vector<size_t> shape_index;
  std::vector<bool> only_depend_shape = {false};
  for (const auto &shape_node : scalar_graph_holder->GetInputShapeNodes()) {
    if (!IsPrimitiveCNode(shape_node, prim::kPrimShape)) {
      MS_LOG(INFO) << "The subgraph input nodes is not Shape. There is no change in ShapeReshapeFusion.";
      return node;
    }
    inputs.push_back(shape_node->cast<CNodePtr>()->inputs().at(kIndex1));
    shape_index.push_back(index);
    index++;
    only_depend_shape.push_back(true);
  }
  scalar_graph_holder->SetShapeIndex(shape_index);
  prim->AddAttr("graph", MakeValue(scalar_graph_holder));

  auto new_node = NewCNode(inputs, func_graph);
  MS_EXCEPTION_IF_NULL(new_node);
  new_node->set_abstract(node->abstract());
  common::AnfAlgo::SetNodeAttr(kAttrOnlyDependShape, MakeValue(only_depend_shape), new_node);
  return new_node;
}

const BaseRef ShapeReshapeFusion2::DefinePattern() const {
  VarPtr shape_tuple = std::make_shared<SeqVar>();
  VectorRef make_tuple = VectorRef({prim::kPrimMakeTuple, shape_tuple});
  return VectorRef({std::make_shared<Primitive>("Reshape"), reshape_input_, make_tuple});
}

const BaseRef ShapeReshapeDirectFusion::DefinePattern() const {
  auto shape_node = VectorRef({prim::kPrimShape, std::make_shared<Var>()});
  auto reshape_input = std::make_shared<Var>();
  return VectorRef({std::make_shared<Primitive>("Reshape"), reshape_input, shape_node});
}

const AnfNodePtr ShapeReshapeDirectFusion::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                   const EquivPtr &equiv) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(equiv);

  auto cnode = node->cast<CNodePtr>();  // reshape cnode
  MS_EXCEPTION_IF_NULL(cnode);

  auto input_node = cnode->input(kIndex1);
  MS_EXCEPTION_IF_NULL(input_node);

  auto shape_node = cnode->input(kIndex2);
  MS_EXCEPTION_IF_NULL(shape_node);
  auto shape_cnode = shape_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(shape_cnode);
  auto shape_input = shape_cnode->input(kIndex1);

  // create new primitive ReshapeExt
  auto prim = std::make_shared<Primitive>("ReshapeExt");
  std::vector<AnfNodePtr> inputs = {NewValueNode(prim), input_node, shape_input};

  auto new_node = NewCNode(inputs, func_graph);
  MS_EXCEPTION_IF_NULL(new_node);
  new_node->set_abstract(node->abstract());
  common::AnfAlgo::SetNodeAttr(kAttrOnlyDependShape, MakeValue(std::vector<bool>{false, true}), new_node);
  return new_node;
}
}  // namespace opt
}  // namespace mindspore
