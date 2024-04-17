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

#include "ops/scalar_graph_holder.h"
#include "ops/auto_generate/gen_ops_primitive.h"
#include "ops/sequence_ops.h"

namespace mindspore {
namespace ops {
namespace {
std::map<PrimitivePtr, ScalarOpType> node_type_map = {
  {prim::kPrimShape, ScalarOpType::SHAPE},
  {prim::kPrimTupleGetItem, ScalarOpType::TUPLE_GET_ITEM},
  {prim::kPrimRealTupleGetItem, ScalarOpType::TUPLE_GET_ITEM},
  {prim::kPrimScalarAdd, ScalarOpType::SCALAR_ADD},
  {prim::kPrimScalarSub, ScalarOpType::SCALAR_SUB},
  {prim::kPrimScalarMul, ScalarOpType::SCALAR_MUL},
  {prim::kPrimScalarDiv, ScalarOpType::SCALAR_DIV},
  {prim::kPrimScalarFloorDiv, ScalarOpType::SCALAR_FLOOR_DIV},
  {prim::kPrimMakeTuple, ScalarOpType::MAKE_TUPLE},
  {prim::kPrimRealMakeTuple, ScalarOpType::MAKE_TUPLE},
  {prim::kPrimReshape, ScalarOpType::RESHAPE},
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

std::shared_ptr<ScalarNode> CreateScalarNodeFromCNode(const CNodePtr &cnode,
                                                      const std::vector<AnfNodePtr> &graph_nodes) {
  auto prim = GetCNodePrimitive(cnode);
  MS_EXCEPTION_IF_NULL(prim);
  std::vector<int64_t> input_index;
  for (size_t i = 1; i < cnode->size(); ++i) {
    auto itr = find(graph_nodes.begin(), graph_nodes.end(), cnode->inputs().at(i));
    if (itr != graph_nodes.end()) {
      input_index.push_back(itr - graph_nodes.begin());
    } else {
      input_index.push_back(-1);
    }
  }
  ScalarOpType op_type = ScalarOpType::VALUE;
  auto itr = node_type_map.begin();
  for (; itr != node_type_map.end(); ++itr) {
    if (IsPrimitiveEquals(prim, itr->first)) {
      op_type = itr->second;
      break;
    }
  }
  if (itr == node_type_map.end()) {
    MS_LOG(DEBUG) << "Can't find type in node_type_map for " << cnode->fullname_with_scope();
    return nullptr;
  }
  return std::make_shared<ScalarNode>(op_type, input_index, std::vector<int64_t>{});
}
}  // namespace

bool ScalarGraphHolder::Init(const std::vector<AnfNodePtr> &graph_nodes) {
  for (auto &n : graph_nodes) {
    if (n->isa<ValueNode>()) {
      int64_t v;
      if (!GetScalarValueFromNode(n->cast<ValueNodePtr>(), &v)) {
        MS_LOG(ERROR) << "Create int64_t scala value from ValueNode failed.";
        return false;
      }
      auto node = std::make_shared<ScalarNode>(ScalarOpType::VALUE, std::vector<int64_t>{}, std::vector<int64_t>{v});
      nodes_.push_back(node);
    } else if (n->isa<CNode>()) {
      auto cnode = n->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(cnode);
      auto node = CreateScalarNodeFromCNode(cnode, graph_nodes);
      if (node == nullptr) {
        MS_LOG(ERROR) << "Create ScalarNode from cnode failed.";
        return false;
      }
      nodes_.push_back(node);
    } else {
      MS_LOG_EXCEPTION
        << "The node in ScalarGraph should be ValueNode or CNode. Please check the ShapeReshapeFusion pass.";
    }
  }
  return true;
}
}  // namespace ops
}  // namespace mindspore
