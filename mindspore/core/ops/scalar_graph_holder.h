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

namespace mindspore {
namespace ops {
enum class ScalarOpType {
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
  ScalarOpType type_;
  std::vector<int64_t> in_index_;  // -1 mean the input is not in ScalarGraph
  std::vector<int64_t> value_;
  ScalarNode(const ScalarOpType &type, const std::vector<int64_t> &index, const std::vector<int64_t> &value)
      : type_(type), in_index_(std::move(index)), value_(value) {}
};

class MS_CORE_API ScalarGraphHolder : public Value {
 public:
  ScalarGraphHolder() = default;

  ~ScalarGraphHolder() override = default;

  bool Init(const std::vector<AnfNodePtr> &graph_nodes);

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
