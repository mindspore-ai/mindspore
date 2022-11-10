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

#include "expander/emitter.h"

#include <algorithm>
#include <functional>
#include <unordered_set>
#include <utility>
#include "ops/primitive_c.h"
#include "utils/anf_utils.h"

namespace mindspore {
namespace expander {
NodePtr Emitter::Emit(const std::string &op_name, const NodePtrList &inputs, const DAttr &attrs) const {
  const auto &op_primc_fns = ops::OpPrimCRegister::GetInstance().GetPrimCMap();
  const auto iter = op_primc_fns.find(op_name);
  PrimitivePtr primc = nullptr;
  if (iter == op_primc_fns.end()) {
    primc = std::make_shared<ops::PrimitiveC>(op_name);
    primc->SetAttrs(attrs);
  } else {
    primc = iter->second();
    if (!attrs.empty()) {
      for (auto &[k, v] : attrs) {
        primc->set_attr(k, v);
      }
    }
  }
  AnfNodePtrList cnode_inputs = {NewValueNode(primc)};
  cnode_inputs.reserve(inputs.size() + 1);
  (void)std::transform(inputs.cbegin(), inputs.cend(), std::back_inserter(cnode_inputs), [](const NodePtr &no) {
    MS_EXCEPTION_IF_NULL(no);
    return no->get();
  });
  auto cnode = func_graph_->NewCNode(cnode_inputs);
  if (scope_ != nullptr) {
    cnode->set_scope(scope_);
  }
  auto node = NewNode(cnode->cast<AnfNodePtr>());
  infer_->Infer(node);
  return node;
}

NodePtr Emitter::EmitValue(const ValuePtr &value) const {
  auto node = NewNode(NewValueNode(value));
  infer_->Infer(node);
  return node;
}

NodePtr Emitter::Exp(const NodePtr &x) const {
  return Emit(kExpOpName, {x},
              {{"base", MakeValue<float>(-1.0)}, {"scale", MakeValue<float>(1.0)}, {"shift", MakeValue<float>(0.0)}});
}

NodePtr Emitter::Log(const NodePtr &x) const {
  return Emit(kLogOpName, {x},
              {{"base", MakeValue<float>(-1.0)},
               {"scale", MakeValue<float>(1.0)},
               {"shift", MakeValue<float>(0.0)},
               {"cust_aicpu", MakeValue(kLogOpName)}});
}

NodePtr Emitter::Cast(const NodePtr &node, const TypePtr &type) const {
  // do not emit a node when the dst type is the same as src type
  if (node->dtype()->type_id() == type->type_id()) {
    return node;
  }
  return Emit("Cast", {node, EmitValue(type)});
}

NodePtr Emitter::Reshape(const NodePtr &node, const ShapeVector &shape) const {
  MS_EXCEPTION_IF_NULL(node);
  auto node_shape = node->shape();
  if (shape.size() != node_shape.size()) {
    return Emit(prim::kReshape, {node, Value(shape)});
  }
  for (size_t i = 0; i < shape.size(); ++i) {
    if (shape[i] != node_shape[i] && shape[i] != -1) {
      return Emit(prim::kReshape, {node, Value(shape)});
    }
  }
  return node;
}

NodePtr Emitter::MatMul(const NodePtr &a, const NodePtr &b, bool transpose_a, bool transpose_b) const {
  return UnifyDtypeAndEmit(prim::kPrimMatMul->name(), a, b,
                           {{"transpose_x1", MakeValue(transpose_a)},
                            {"transpose_x2", MakeValue(transpose_b)},
                            {"transpose_a", MakeValue(transpose_a)},
                            {"transpose_b", MakeValue(transpose_b)}});
}

NodePtr Emitter::BatchMatMul(const NodePtr &a, const NodePtr &b, bool transpose_a, bool transpose_b) const {
  return UnifyDtypeAndEmit(prim::kPrimBatchMatMul->name(), a, b,
                           {{"transpose_x1", MakeValue(transpose_a)},
                            {"transpose_x2", MakeValue(transpose_b)},
                            {"transpose_a", MakeValue(transpose_a)},
                            {"transpose_b", MakeValue(transpose_b)}});
}

NodePtr Emitter::Transpose(const NodePtr &node, const ShapeVector &perm) const {
  // perm like [0, 1, 2, 3] does not need transpose.
  auto n = SizeToLong(perm.size());
  for (size_t i = 0; i < perm.size(); ++i) {
    // perm value may be negative, e.g. [0, -3, 2, 3] is equal to [0, 1, 2, 3]
    auto perm_i = perm[i] < 0 ? (perm[i] + n) : perm[i];
    if (perm_i != static_cast<int64_t>(i)) {
      return Emit(kTransposeOpName, {node, Value(perm)});
    }
  }
  return node;
}

NodePtr Emitter::ZerosLike(const NodePtr &node) const {
  if (node->isa<ValueNode>()) {
    if (node->dtype()->type_id() == kMetaTypeNone) {
      return Emit(prim::kZerosLike, {Tensor(0)});
    }
    auto value_node = node->get<ValueNodePtr>();
    MS_EXCEPTION_IF_NULL(value_node);
    auto v = value_node->value();
    MS_EXCEPTION_IF_NULL(v);
    if (v->isa<ValueSequence>()) {
      auto sh = GetValue<std::vector<int64_t>>(v);
      return Emit(prim::kZerosLike, {Tensor(sh)});
    } else if (v->isa<Scalar>() || v->isa<Type>()) {
      return Emit(prim::kZerosLike, {Tensor(0, v->type())});
    } else if (v->isa<Monad>()) {
      return Emit(prim::kZerosLike, {Tensor(0)});
    }
  }
  return Emit(prim::kZerosLike, {node});
}

NodePtr Emitter::Fill(double value, const ShapeVector &shape, TypeId data_type) const {
  size_t data_num = LongToSize(std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int64_t>()));
  std::vector<double> data(data_num, value);
  return Tensor(data_type, shape, &data[0], TypeId::kNumberTypeFloat64);
}

NodePtr Emitter::Fill(int64_t value, const ShapeVector &shape, TypeId data_type) const {
  size_t data_num = LongToSize(std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int64_t>()));
  std::vector<int64_t> data(data_num, value);
  return Tensor(data_type, shape, &data[0], TypeId::kNumberTypeInt64);
}

std::pair<bool, ShapeVector> Emitter::NeedReduce(const ShapeVector &shape, const std::vector<int64_t> &axis,
                                                 bool keep_dim) const {
  if (shape.empty()) {
    return std::make_pair(false, shape);
  }
  auto rank = SizeToLong(shape.size());
  auto real_axis = axis;
  if (real_axis.empty()) {
    // all reduce
    for (int64_t i = 0; i < rank; ++i) {
      real_axis.push_back(i);
    }
  }
  std::unordered_set<size_t> uniq_axis;
  for (size_t i = 0; i < real_axis.size(); ++i) {
    if (real_axis[i] < -rank || real_axis[i] >= rank) {
      MS_EXCEPTION(ValueError) << "Reduce axis[" << i << "] is " << real_axis[i] << ", which is out of range [-" << rank
                               << ", " << rank << ") for shape: " << shape;
    }
    auto axis_i = real_axis[i] < 0 ? real_axis[i] + rank : real_axis[i];
    (void)uniq_axis.insert(LongToSize(axis_i));
  }
  // Calc reduce output shape
  ShapeVector out_shape;
  bool need_reduce = false;
  for (size_t i = 0; i < shape.size(); ++i) {
    if (uniq_axis.find(i) == uniq_axis.end()) {
      // not reduce axis
      out_shape.push_back(shape[i]);
    } else {
      // reduce axis
      if (shape[i] != 1) {
        need_reduce = true;
      }
      if (keep_dim) {
        out_shape.push_back(1);
      }
    }
  }
  return std::make_pair(need_reduce, out_shape);
}

NodePtr Emitter::ReduceSum(const NodePtr &x, const ShapeVector &axis, bool keep_dims) const {
  MS_EXCEPTION_IF_NULL(x);
  auto need_reduce = NeedReduce(x->shape(), axis, keep_dims);
  if (!need_reduce.first) {
    return Reshape(x, need_reduce.second);
  }
  return Emit(prim::kPrimReduceSum->name(), {x, Value<ShapeVector>(axis)}, {{"keep_dims", MakeValue(keep_dims)}});
}

std::tuple<NodePtr, NodePtr> Emitter::UnifyDtype2(const NodePtr &lhs, const NodePtr &rhs) const {
  auto it1 = type_map_.find(lhs->dtype()->type_id());
  auto it2 = type_map_.find(rhs->dtype()->type_id());
  if (it1 == type_map_.end() || it2 == type_map_.end() || it1->second == it2->second) {
    return {lhs, rhs};
  }
  if (it1->second < it2->second) {
    return {this->Cast(lhs, rhs->dtype()), rhs};
  }
  return {lhs, this->Cast(rhs, lhs->dtype())};
}

NodePtr operator+(const NodePtr &lhs, const NodePtr &rhs) { return lhs->emitter()->Add(lhs, rhs); }
NodePtr operator-(const NodePtr &lhs, const NodePtr &rhs) { return lhs->emitter()->Sub(lhs, rhs); }
NodePtr operator*(const NodePtr &lhs, const NodePtr &rhs) { return lhs->emitter()->Mul(lhs, rhs); }
NodePtr operator/(const NodePtr &lhs, const NodePtr &rhs) { return lhs->emitter()->RealDiv(lhs, rhs); }
NodePtr operator-(const NodePtr &node) { return node->emitter()->Neg(node); }
}  // namespace expander
}  // namespace mindspore
