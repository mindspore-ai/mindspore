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

#include "common/graph_kernel/bprop/expander/emitter.h"

#include <algorithm>
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
  auto node = NewNode(cnode->cast<AnfNodePtr>());
  infer_->Infer(node);
  return node;
}

NodePtr Emitter::EmitValue(const ValuePtr &value) const {
  auto node = NewNode(NewValueNode(value));
  infer_->Infer(node);
  return node;
}

NodePtr Emitter::MatMul(const NodePtr &a, const NodePtr &b, bool transpose_a, bool transpose_b) const {
  return Emit(prim::kPrimMatMul->name(), {a, b},
              {{"transpose_a", MakeValue(transpose_a)}, {"transpose_b", MakeValue(transpose_b)}});
}

NodePtr Emitter::BatchMatMul(const NodePtr &a, const NodePtr &b, bool transpose_a, bool transpose_b) const {
  return Emit(prim::kPrimBatchMatMul->name(), {a, b},
              {{"transpose_a", MakeValue(transpose_a)}, {"transpose_b", MakeValue(transpose_b)}});
}

NodePtr Emitter::ZerosLike(const NodePtr &node) const {
  if (node->isa<ValueNode>()) {
    auto value_node = node->get<ValueNodePtr>();
    MS_EXCEPTION_IF_NULL(value_node);
    auto v = value_node->value();
    MS_EXCEPTION_IF_NULL(v);
    if (v->isa<ValueSequence>()) {
      auto sh = GetValue<std::vector<int64_t>>(v);
      return Emit(prim::kZerosLike, {Tensor(sh)});
    } else if (v->isa<Scalar>() || v->isa<Type>()) {
      return Emit(prim::kZerosLike, {Tensor(0, v->type())});
    }
  }
  return Emit(prim::kZerosLike, {node});
}

NodePtr Emitter::ReduceSum(const NodePtr &x, const ShapeVector &axis, bool keep_dims) const {
  return Emit(prim::kPrimReduceSum->name(), {x, Value<ShapeVector>(axis)}, {{"keep_dims", MakeValue(keep_dims)}});
}

NodePtr operator+(const NodePtr &lhs, const NodePtr &rhs) { return lhs->emitter()->Add(lhs, rhs); }
NodePtr operator-(const NodePtr &lhs, const NodePtr &rhs) { return lhs->emitter()->Sub(lhs, rhs); }
NodePtr operator*(const NodePtr &lhs, const NodePtr &rhs) { return lhs->emitter()->Mul(lhs, rhs); }
NodePtr operator/(const NodePtr &lhs, const NodePtr &rhs) { return lhs->emitter()->RealDiv(lhs, rhs); }
NodePtr operator-(const NodePtr &node) { return node->emitter()->Neg(node); }
}  // namespace expander
}  // namespace mindspore
