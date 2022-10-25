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

#ifndef MINDSPORE_CORE_EXPANDER_EMITTER_H_
#define MINDSPORE_CORE_EXPANDER_EMITTER_H_
#include <memory>
#include <vector>
#include <string>
#include "ir/func_graph.h"
#include "ops/core_ops.h"
#include "common/graph_kernel/bprop/expander/node.h"
#include "common/graph_kernel/bprop/expander/infer.h"

namespace mindspore {
namespace expander {
class Emitter {
 public:
  Emitter(const FuncGraphPtr &func_graph, const ExpanderInferPtr &infer) : func_graph_(func_graph), infer_(infer) {
    MS_EXCEPTION_IF_NULL(infer);
  }

  /// \brief Emit a primitive CNode
  NodePtr Emit(const std::string &op_name, const NodePtrList &inputs, const DAttr &attrs = {}) const;

  /// \brief Emit a ValueNode
  NodePtr EmitValue(const ValuePtr &value) const;

  NodePtr MakeTuple(const NodePtrList &inputs) const { return Emit(prim::kMakeTuple, inputs); }
  NodePtr TupleGetItem(const NodePtr &input, size_t i) const {
    return Emit(prim::kTupleGetItem, {input, Value(static_cast<int64_t>(i))});
  }

  NodePtr Cast(const NodePtr &node, const TypePtr &type) const { return Emit("Cast", {node, EmitValue(type)}); }
  NodePtr Cast(const NodePtr &node, TypeId type_id) const { return Cast(node, TypeIdToType(type_id)); }

  NodePtr Reshape(const NodePtr &node, const ShapeVector &shape) const {
    return Emit(prim::kReshape, {node, Tensor(shape)});
  }

  NodePtr Add(const NodePtr &lhs, const NodePtr &rhs) const { return Emit(prim::kAdd, {lhs, rhs}); }
  NodePtr Sub(const NodePtr &lhs, const NodePtr &rhs) const { return Emit(prim::kSub, {lhs, rhs}); }
  NodePtr Mul(const NodePtr &lhs, const NodePtr &rhs) const { return Emit(prim::kMul, {lhs, rhs}); }
  NodePtr RealDiv(const NodePtr &lhs, const NodePtr &rhs) const { return Emit(prim::kRealDiv, {lhs, rhs}); }
  NodePtr MatMul(const NodePtr &a, const NodePtr &b, bool transpose_a = false, bool transpose_b = false) const;

  NodePtr ZerosLike(const NodePtr &node) const;

  /// \brief Emit a value node
  template <typename T>
  NodePtr Value(const T &value) const {
    return EmitValue(MakeValue(value));
  }

  /// \brief Emit a Tensor node.
  template <typename T>
  NodePtr Tensor(T data, TypePtr type_ptr = nullptr) const {
    auto tensor_ptr = std::make_shared<tensor::Tensor>(data, type_ptr);
    return EmitValue(tensor_ptr);
  }

  /// \brief Emit a tensor node.
  NodePtr Tensor(TypeId data_type, const ShapeVector &shape, void *data, TypeId src_data_type) const {
    auto tensor_ptr = std::make_shared<tensor::Tensor>(data_type, shape, data, src_data_type);
    return EmitValue(tensor_ptr);
  }

 protected:
  NodePtr NewNode(const AnfNodePtr &anfnode) const { return std::make_shared<Node>(anfnode, this); }

  FuncGraphPtr func_graph_;
  ExpanderInferPtr infer_{nullptr};
};
using EmitterPtr = std::shared_ptr<Emitter>;

NodePtr operator+(const NodePtr &lhs, const NodePtr &rhs);
NodePtr operator-(const NodePtr &lhs, const NodePtr &rhs);
NodePtr operator*(const NodePtr &lhs, const NodePtr &rhs);
NodePtr operator/(const NodePtr &lhs, const NodePtr &rhs);
}  // namespace expander
}  // namespace mindspore
#endif  // MINDSPORE_CORE_EXPANDER_EMITTER_H_
