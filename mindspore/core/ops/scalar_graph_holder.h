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
#define USE_DEPRECATED_API
#include <vector>
#include <memory>
#include <map>
#include "ir/anf.h"

namespace mindspore {
namespace ops {
class ScalarGraphHolder : public Value {
 public:
  explicit ScalarGraphHolder(const std::vector<AnfNodePtr> &nodes) {  // topo sort AnfNode
    nodes_.resize(nodes.size());
    nodes_ = nodes;
  }

  ~ScalarGraphHolder() override = default;

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

  AnfNodePtr GetAnfNode(const size_t &index) { return nodes_[index]; }

  void SetScalarValue(const AnfNodePtr &node, const std::vector<int64_t> &v) {
    if (value_.find(node) != value_.end()) {
      value_[node] = v;
    } else {
      value_.insert({node, v});
    }
  }

  std::vector<int64_t> GetScalarValue(const AnfNodePtr &node) { return value_[node]; }

  void SetInputShapeNodes(const std::vector<AnfNodePtr> &nodes) { input_shape_nodes_ = nodes; }

  const std::vector<AnfNodePtr> &GetInputShapeNodes() { return input_shape_nodes_; }

  void SetShapeIndex(const AnfNodePtr &node, const size_t &index) {
    if (input_shape_index_.find(node) != input_shape_index_.end()) {
      input_shape_index_[node] = index;
    } else {
      input_shape_index_.insert({node, index});
    }
  }

  size_t GetShapeIndex(const AnfNodePtr &node) { return input_shape_index_[node]; }

 private:
  std::vector<AnfNodePtr> nodes_;
  std::map<AnfNodePtr, std::vector<int64_t>> value_;

  std::vector<AnfNodePtr> input_shape_nodes_;
  std::map<AnfNodePtr, size_t> input_shape_index_;
};

using ScalarGraphHolderPtr = std::shared_ptr<ScalarGraphHolder>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_SCALAR_GRAPH_HOLDER_H_
