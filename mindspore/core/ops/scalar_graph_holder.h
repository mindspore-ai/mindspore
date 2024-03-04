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

#ifndef MINDSPORE_CORE_OPS_SCALAR_GRAPH_HOLDER_H_
#define MINDSPORE_CORE_OPS_SCALAR_GRAPH_HOLDER_H_
#include <utility>
#include <vector>
#include <memory>
#include <map>
#include "ir/anf.h"
#include "ops/auto_generate/gen_ops_primitive.h"
#include "ops/sequence_ops.h"

namespace mindspore {
namespace ops {
enum class OpType {
  VALUE = 0,
  SHAPE = 1,
  RESHAPE = 2,
  TUPLE_GET_ITEM = 3,
  MAKE_TUPLE = 4,
  SCALAR_ADD = 5,
  SCALAR_SUB = 6,
  SCALAR_MUL = 7,
  SCALAR_DIV = 8,
  SCALAR_FLOOR_DIV = 9,
};

struct ScalarNode {
  OpType type_;
  std::vector<int64_t> in_index_;  // -1 mean the input is not in ScalarGraph
  std::vector<int64_t> value_;
  ScalarNode(const OpType &type, const std::vector<int64_t> &index, const std::vector<int64_t> &value)
      : type_(type), in_index_(std::move(index)), value_(value) {}
};

namespace {
std::map<PrimitivePtr, OpType> node_type_map = {
  {prim::kPrimShape, OpType::SHAPE},
  {prim::kPrimTupleGetItem, OpType::TUPLE_GET_ITEM},
  {prim::kPrimRealTupleGetItem, OpType::TUPLE_GET_ITEM},
  {prim::kPrimScalarAdd, OpType::SCALAR_ADD},
  {prim::kPrimScalarSub, OpType::SCALAR_SUB},
  {prim::kPrimScalarMul, OpType::SCALAR_MUL},
  {prim::kPrimScalarDiv, OpType::SCALAR_DIV},
  {prim::kPrimScalarFloorDiv, OpType::SCALAR_FLOOR_DIV},
  {prim::kPrimMakeTuple, OpType::MAKE_TUPLE},
  {prim::kPrimRealMakeTuple, OpType::MAKE_TUPLE},
  {prim::kPrimReshape, OpType::RESHAPE},
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
  OpType op_type;
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

class ScalarGraphHolder : public Value {
 public:
  ScalarGraphHolder() = default;

  ~ScalarGraphHolder() override = default;

  bool Init(const std::vector<AnfNodePtr> &graph_nodes) {
    for (auto &n : graph_nodes) {
      if (n->isa<ValueNode>()) {
        int64_t v;
        if (!GetScalarValueFromNode(n->cast<ValueNodePtr>(), &v)) {
          MS_LOG(ERROR) << "Create int64_t scala value from ValueNode failed.";
          return false;
        }
        auto node = std::make_shared<ScalarNode>(OpType::VALUE, std::vector<int64_t>{}, std::vector<int64_t>{v});
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

  MS_DECLARE_PARENT(ScalarGraphHolder, Value);

  bool operator==(const Value &rhs) const override {
    if (rhs.isa<ScalarGraphHolder>()) {
      auto other_holder = dynamic_cast<const ScalarGraphHolder &>(rhs);
      auto nodes = other_holder.nodes_;
      if (nodes.size() != this->nodes_.size()) {
        return false;
      }
      for (size_t i = 0; i < nodes.size(); ++i) {
        if (nodes.at(i) != this->nodes_.at(i)) {
          return false;
        }
      }
    } else {
      return false;
    }
    return true;
  }

  size_t GetNodeSize() { return nodes_.size(); }

  const std::shared_ptr<ScalarNode> &GetScalarNode(const size_t &index) { return nodes_[index]; }

  void SetScalarValue(const size_t &index, const std::vector<int64_t> &v) { nodes_[index]->value_ = v; }

  const std::vector<int64_t> &GetScalarValue(const size_t &index) { return nodes_[index]->value_; }

  void SetInputShapeNodes(const std::vector<AnfNodePtr> &nodes) { input_shape_nodes_ = nodes; }

  const std::vector<AnfNodePtr> &GetInputShapeNodes() { return input_shape_nodes_; }

  void SetShapeIndex(const std::vector<size_t> &index) { input_shape_index_ = index; }

  const std::vector<size_t> &GetShapeIndex() { return input_shape_index_; }

 private:
  std::vector<std::shared_ptr<ScalarNode>> nodes_;
  std::vector<AnfNodePtr> input_shape_nodes_;
  std::vector<size_t> input_shape_index_;
};

using ScalarGraphHolderPtr = std::shared_ptr<ScalarGraphHolder>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_SCALAR_GRAPH_HOLDER_H_
