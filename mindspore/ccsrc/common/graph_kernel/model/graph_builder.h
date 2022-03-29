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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_MODEL_GRAPH_BUILDER_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_MODEL_GRAPH_BUILDER_H_

#include <vector>
#include <memory>
#include <string>
#include "common/graph_kernel/model/lite_graph.h"

namespace mindspore::graphkernel::inner {
class GraphBuilder : public LiteGraph::GraphBuilderBase {
 public:
  explicit GraphBuilder(const std::string &name = "") : GraphBuilderBase(name) {}
  ~GraphBuilder() = default;
  NodePtr Add(const NodePtr &lhs, const NodePtr &rhs) const { return Emit("Add", {lhs, rhs}); }
  NodePtr Sub(const NodePtr &lhs, const NodePtr &rhs) const { return Emit("Sub", {lhs, rhs}); }
  NodePtr Mul(const NodePtr &lhs, const NodePtr &rhs) const { return Emit("Mul", {lhs, rhs}); }
  NodePtr Div(const NodePtr &lhs, const NodePtr &rhs) const { return Emit("RealDiv", {lhs, rhs}); }
  NodePtr Greater(const NodePtr &lhs, const NodePtr &rhs) const { return Emit("Greater", {lhs, rhs}); }
  NodePtr GreaterEqual(const NodePtr &lhs, const NodePtr &rhs) const { return Emit("GreaterEqual", {lhs, rhs}); }
  NodePtr LessEqual(const NodePtr &lhs, const NodePtr &rhs) const { return Emit("LessEqual", {lhs, rhs}); }
  NodePtr Equal(const NodePtr &lhs, const NodePtr &rhs) const { return Emit("Equal", {lhs, rhs}); }
  NodePtr Assign(const NodePtr &lhs, const NodePtr &rhs) const { return Emit("Assign", {lhs, rhs}); }

  NodePtr Neg(const NodePtr &input) const { return Emit("Neg", {input}); }
  NodePtr Exp(const NodePtr &input) const { return Emit("Exp", {input}); }
  NodePtr Abs(const NodePtr &input) const { return Emit("Abs", {input}); }
  NodePtr Log(const NodePtr &input) const { return Emit("Log", {input}); }
  NodePtr Sqrt(const NodePtr &input) const { return Emit("Sqrt", {input}); }
  NodePtr Tanh(const NodePtr &input) const { return Emit("Tanh", {input}); }

  NodePtr Cast(const NodePtr &input, const TypeId &type_id) const {
    return Emit("Cast", {input}, {{"dst_type", TypeIdToType(type_id)}});
  }
  NodePtr Reshape(const NodePtr &input, const ShapeVector &shape) const;
  NodePtr BroadcastTo(const NodePtr &input, const ShapeVector &shape) const;

  NodePtr ReduceSum(const NodePtr &input, const std::vector<int64_t> &axis, const bool &keep_dims = false) const;
  NodePtr ReduceMax(const NodePtr &input, const std::vector<int64_t> &axis, const bool &keep_dims = false) const;

  template <typename T>
  NodePtr Const(T input, const TypeId &type_id) const {
    tensor::TensorPtr const_input;
    switch (type_id) {
      case kNumberTypeBool:
        const_input = std::make_shared<tensor::Tensor>(static_cast<bool>(input), TypeIdToType(type_id));
        break;
      case kNumberTypeInt ... kNumberTypeInt64:
        const_input = std::make_shared<tensor::Tensor>(static_cast<int64_t>(input), TypeIdToType(type_id));
        break;
      case kNumberTypeUInt ... kNumberTypeUInt64:
        const_input = std::make_shared<tensor::Tensor>(static_cast<uint64_t>(input), TypeIdToType(type_id));
        break;
      case kNumberTypeFloat ... kNumberTypeFloat64:
        const_input = std::make_shared<tensor::Tensor>(static_cast<double>(input), TypeIdToType(type_id));
        break;
      default:
        MS_LOG(EXCEPTION) << "The input data type should be int, uint, float or bool, But Get :"
                          << TypeIdToString(type_id);
        break;
    }
    return Value(const_input);
  }
};
}  // namespace mindspore::graphkernel::inner
#endif
