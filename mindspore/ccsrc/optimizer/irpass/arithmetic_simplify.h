/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_OPTIMIZER_IRPASS_ARITHMETIC_SIMPLIFY_H_
#define MINDSPORE_CCSRC_OPTIMIZER_IRPASS_ARITHMETIC_SIMPLIFY_H_

#include <vector>
#include <memory>
#include <algorithm>

#include "optimizer/optimizer.h"
#include "optimizer/irpass.h"
#include "optimizer/irpass/prim_eliminate.h"
#include "ir/visitor.h"
#include "operator/ops.h"

namespace mindspore {
namespace opt {
namespace irpass {
// {prim::kPrimScalarMul, 0, X}, {prim::kPrimScalarMul, X, 0}
// {prim::kPrimScalarMul, 1, X}, {prim::kPrimScalarMul, X, 1}
class MultiplyByZeroOrOne : public AnfVisitor {
 public:
  MultiplyByZeroOrOne() : zero_(MakeValue(0)), one_(MakeValue(1)) {}
  ~MultiplyByZeroOrOne() override = default;

  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    Reset();
    AnfVisitor::Match(prim::kPrimScalarMul)(node);

    if (is_zero_) {
      return NewValueNode(zero_);
    }
    if (is_one_) {
      return x_;
    }
    return nullptr;
  }

  void Visit(const AnfNodePtr &node) override {
    if (is_one_ || node->isa<CNode>()) {
      x_ = node;
      return;
    }

    AnfVisitor::Visit(node);
    if (!is_one_) {
      x_ = node;
    }
  }

  void Visit(const ValueNodePtr &vnode) override {
    auto value = vnode->value();
    if (*value == *zero_) {
      is_zero_ = true;
    } else if (*value == *one_) {
      is_one_ = true;
    }
  }

  void Reset() {
    x_ = nullptr;
    is_one_ = false;
    is_zero_ = false;
  }

 private:
  bool is_zero_{false}, is_one_{false};
  ValuePtr zero_, one_;
  AnfNodePtr x_{nullptr};
};

// Support class used for checking if all values of a Tensor are equal `check_value_`
// Supported data types: double, float/float32, int/int32
class CheckTensorConstant {
 public:
  explicit CheckTensorConstant(int _check_value = 0) : check_value_(_check_value) {}
  ~CheckTensorConstant() = default;
  bool IsTensorConstant(const ValuePtr &value) {
    if (!value->isa<tensor::Tensor>()) {
      return false;
    }
    auto tensor_ptr = dyn_cast<tensor::Tensor>(value);
    TypeId tensor_type = tensor_ptr->Dtype()->type_id();
    if ((tensor_type == TypeId::kNumberTypeFloat32) || (tensor_type == TypeId::kNumberTypeFloat)) {
      float *data2 = reinterpret_cast<float *>(tensor_ptr->data_c());
      for (int i = 0; i < tensor_ptr->DataSize(); i++) {
        if (fabs(data2[i] - check_value_) > FLT_EPSILON) {
          return false;
        }
      }
      return true;
    } else if (tensor_type == TypeId::kNumberTypeFloat64) {
      double *data2 = reinterpret_cast<double *>(tensor_ptr->data_c());
      for (int i = 0; i < tensor_ptr->DataSize(); i++) {
        if (fabs(data2[i] - check_value_) > DBL_EPSILON) {
          return false;
        }
      }
      return true;
    } else if ((tensor_type == TypeId::kNumberTypeInt32) || (tensor_type == TypeId::kNumberTypeInt)) {
      int *data2 = reinterpret_cast<int *>(tensor_ptr->data_c());
      for (int i = 0; i < tensor_ptr->DataSize(); i++) {
        if (data2[i] != check_value_) {
          return false;
        }
      }
      return true;
    }
    // Un-support Data Types
    return false;
  }

  bool IsTensorScalarConstant(const ValuePtr &value) {
    if (!value->isa<tensor::Tensor>()) {
      return false;
    }
    auto tensor_ptr = dyn_cast<tensor::Tensor>(value);
    if ((tensor_ptr->DataSize() > 1) || (tensor_ptr->DataDim() > 0)) {
      return false;
    }
    return IsTensorConstant(value);
  }

 private:
  int check_value_;
};

class TensorMultiplyBase : public AnfVisitor {
 protected:
  void *GetPointerToTensorData(const AnfNodePtr &node, bool writable = false) {
    if (!node->isa<ValueNode>()) {
      return nullptr;
    }

    auto value = node->cast<ValueNodePtr>()->value();

    if (!value->isa<tensor::Tensor>()) {
      return nullptr;
    }

    tensor::TensorPtr tensor_ptr = dyn_cast<tensor::Tensor>(value);
    return tensor_ptr->data_c(writable);
  }

  // Make a new tensor (when possible) with the same shape as of `node`
  // If x is nullptr then fill new tensor will "0"
  // If x is a tensor with empty shape then fill new tensor with the single value of x
  // If x is a tensor with same shape as `node` then return x as result
  AnfNodePtr NewTensorFilledWithData(const AnfNodePtr &node, const AnfNodePtr &x = nullptr) {
    if ((node->abstract() == nullptr) || !node->abstract()->isa<abstract::AbstractTensor>()) {
      return nullptr;
    }

    auto tensor_abstract = node->abstract()->cast<abstract::AbstractTensorPtr>();
    TypePtr tensor_type_ptr = tensor_abstract->element()->BuildType();
    std::vector<int> tensor_shape = tensor_abstract->shape()->shape();

    auto new_tensor_ptr = std::make_shared<tensor::Tensor>(tensor_type_ptr->type_id(), tensor_shape);
    size_t mem_size = GetTypeByte(tensor_type_ptr) * IntToSize(new_tensor_ptr->ElementsNum());
    char *data = reinterpret_cast<char *>(new_tensor_ptr->data_c(true));

    if (x == nullptr) {
      std::memset(data, 0, mem_size);
      auto new_vnode = NewValueNode(new_tensor_ptr);
      new_vnode->set_abstract(new_tensor_ptr->ToAbstract());
      return new_vnode;
    }
    // x is not nullptr
    if (x->isa<CNode>()) {
      if ((x->abstract() == nullptr) || !x->abstract()->isa<abstract::AbstractTensor>()) {
        return nullptr;
      }
      auto x_abstract = x->abstract()->cast<abstract::AbstractTensorPtr>();
      std::vector<int> x_shape = x_abstract->shape()->shape();

      if (x_shape != tensor_shape) {
        return nullptr;
      }
      return x;
    }

    if (!x->isa<ValueNode>()) {
      return nullptr;
    }
    auto x_value = x->cast<ValueNodePtr>()->value();
    if (!x_value->isa<tensor::Tensor>()) {
      return nullptr;
    }

    auto x_tensor_ptr = dyn_cast<tensor::Tensor>(x_value);

    if ((x_tensor_ptr->DataSize() > 1) && (x_tensor_ptr->DataSize() != new_tensor_ptr->DataSize())) {
      return nullptr;
    }
    char *source_data = reinterpret_cast<char *>(GetPointerToTensorData(x));
    if (x_tensor_ptr->DataSize() == 1) {
      for (int i = 0; i < new_tensor_ptr->ElementsNum(); i++) {
        memcpy(source_data, data + i * GetTypeByte(tensor_type_ptr), GetTypeByte(tensor_type_ptr));
      }
    } else {
      memcpy(source_data, data, mem_size);
    }
    auto new_vnode = NewValueNode(new_tensor_ptr);
    new_vnode->set_abstract(new_tensor_ptr->ToAbstract());
    return new_vnode;
  }

  AnfNodePtr x_{nullptr};
};

// {prim::kPrimMul, 0, X}, {prim::kPrimMul, X, 0}
class TensorMultiplyByZero : public TensorMultiplyBase {
 public:
  TensorMultiplyByZero() : zero_(MakeValue(0)) {}
  ~TensorMultiplyByZero() override = default;
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    Reset();
    AnfVisitor::Match(prim::kPrimMul)(node);

    if (is_zero_) {
      if (x_->func_graph() != node->func_graph()) {
        return nullptr;
      }
      return NewTensorFilledWithData(node);
    }
    return nullptr;
  }

  void Visit(const AnfNodePtr &node) override {
    if (is_zero_) {
      x_ = node;
      return;
    }

    if (IsParam(node)) {
      x_ = node;
      return;
    }

    if (IsCNode(node)) {
      CNodePtr cnode = node->cast<CNodePtr>();
      if (IsPrimitive(cnode->input(0), prim::kPrimZerosLike)) {
        is_zero_ = true;
        return;
      }
      x_ = node;
      return;
    }
    auto value = node->cast<ValueNodePtr>()->value();
    if (CheckTensorConstant(0).IsTensorConstant(value)) {
      is_zero_ = true;
      return;
    }
    x_ = node;
  }

  void Visit(const ValueNodePtr &vnode) override {
    auto value = vnode->value();
    if (CheckTensorConstant(0).IsTensorConstant(value)) {
      is_zero_ = true;
      return;
    }
    x_ = vnode;
  }
  void Reset() {
    x_ = nullptr;
    is_zero_ = false;
  }

 private:
  bool is_zero_{false};
  ValuePtr zero_;
};

// {prim::kPrimMul, 1, X}, {prim::kPrimMul, X, 1}
class TensorMultiplyByOne : public TensorMultiplyBase {
 public:
  TensorMultiplyByOne() {}
  ~TensorMultiplyByOne() override = default;
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    Reset();
    AnfVisitor::Match(prim::kPrimMul)(node);

    if (is_one_) {
      return NewTensorFilledWithData(node, x_);
    }
    return nullptr;
  }

  void Visit(const AnfNodePtr &node) override {
    if (is_one_) {
      x_ = node;
      return;
    }

    if (IsParam(node) || IsCNode(node)) {
      x_ = node;
      return;
    }

    auto value = node->cast<ValueNodePtr>()->value();
    if (CheckTensorConstant(1).IsTensorConstant(value)) {
      is_one_ = true;
      return;
    }
    x_ = node;
  }

  void Visit(const ValueNodePtr &vnode) override {
    auto value = vnode->value();
    if (CheckTensorConstant(1).IsTensorConstant(value)) {
      is_one_ = true;
      return;
    }
    x_ = vnode;
  }
  void Reset() {
    x_ = nullptr;
    is_one_ = false;
  }

 private:
  bool is_one_{false};
};

// {prim::kPrimScalarAdd, X, 0}
// {prim::kPrimScalarAdd, 0, X}
class AddByZero : public AnfVisitor {
 public:
  AddByZero() : zero_(MakeValue(0)) {}
  ~AddByZero() override = default;

  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    Reset();
    AnfVisitor::Match(prim::kPrimScalarAdd)(node);

    if (is_zero_) {
      return x_;
    }
    return nullptr;
  }

  void Visit(const AnfNodePtr &node) override {
    if (node->isa<ValueNode>() &&
        ((*GetValueNode(node) == *zero_) || CheckTensorConstant(0).IsTensorScalarConstant(GetValueNode(node)))) {
      is_zero_ = true;
      return;
    }

    x_ = node;
  }

  void Reset() {
    x_ = nullptr;
    is_zero_ = false;
  }

 private:
  bool is_zero_{false};
  ValuePtr zero_;
  AnfNodePtr x_{nullptr};
};

// {prim::kPrimTensorAdd, {kPrimZerosLike, Y}, X},
// {prim::kPrimTensorAdd, X, {kPrimZerosLike, Y}}
class TensorAddByZero : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    Reset();
    AnfVisitor::Match(prim::kPrimTensorAdd)(node);

    if (is_zero_) {
      return x_;
    }
    return nullptr;
  }

  void Visit(const AnfNodePtr &node) override {
    if (IsPrimitive(node, prim::kPrimZerosLike)) {
      is_zero_ = true;
      return;
    }
    if (node->isa<ValueNode>() && CheckTensorConstant(0).IsTensorScalarConstant(GetValueNode(node))) {
      is_zero_ = true;
      return;
    }

    x_ = node;
  }

  void Visit(const ValueNodePtr &vnode) override {
    auto value = vnode->value();
    if (CheckTensorConstant(0).IsTensorConstant(value)) {
      is_zero_ = true;
      return;
    }
  }

  void Reset() {
    x_ = nullptr;
    is_zero_ = false;
  }

 private:
  bool is_zero_{false};
  AnfNodePtr x_{nullptr};
};

// {PrimMomentum, {kPrimZerosLike, X}, Y, Z, Xs}  -> {prim::kPrimMakeTuple, Z, Y}
class OptUpdateZeroTensor : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    if (!IsPrimitiveCNode(node, prim::kPrimMomentum) || node->func_graph() == nullptr) {
      return nullptr;
    }

    // {PrimMomentum, {...}, Y, Z, Xs}
    auto &inputs = node->cast<CNodePtr>()->inputs();
    if (inputs.size() < 4 || !IsPrimitiveCNode(inputs[1], prim::kPrimZerosLike)) {
      return nullptr;
    }
    auto y = inputs[2];
    auto z = inputs[3];

    // {kPrimZerosLike, X}
    if (inputs[1]->cast<CNodePtr>()->size() != 2) {
      return nullptr;
    }

    // {prim::kPrimMakeTuple, Z, Y}
    return node->func_graph()->NewCNode({NewValueNode(prim::kPrimMakeTuple), z, y});
  }
};

// {prim::kPrimMul, Tensor1, {orim::kPrimMul, Tensor2, {...}}} ->
// {prim::kPrimMul, {...}, {prim::kPrimMul, Tensor1, Tensor2}}
class ConstantDuplicateMul : public AnfVisitor {
 public:
  // Support function to multiply two constant tensors: partially support broadcasting shapes
  template <typename T>
  void Multiply(void *in_data_1, int in_data_1_size, void *in_data_2, int in_data_2_size, void **out_data,
                int out_data_size) {
    T *data_1 = reinterpret_cast<T *>(in_data_1);
    T *data_2 = reinterpret_cast<T *>(in_data_2);
    T *data_out = new T[out_data_size];

    if (in_data_1_size == 1) {
      for (int i = 0; i < out_data_size; i++) {
        data_out[i] = data_1[0];
      }
    } else {
      for (int i = 0; i < out_data_size; i++) {
        data_out[i] = data_1[i];
      }
    }
    if (in_data_2_size == 1) {
      for (int i = 0; i < out_data_size; i++) {
        data_out[i] *= data_2[0];
      }
    } else {
      for (int i = 0; i < out_data_size; i++) {
        data_out[i] *= data_2[i];
      }
    }
    *out_data = reinterpret_cast<void *>(data_out);
    return;
  }

  AnfNodePtr MulConstantTensors(const AnfNodePtr &vnode_1, const AnfNodePtr &vnode_2, const AnfNodePtr &node_3) {
    if (!vnode_1->isa<ValueNode>() || !vnode_2->isa<ValueNode>() || (vnode_1->abstract() == nullptr) ||
        (vnode_2->abstract() == nullptr) || (node_3->abstract() == nullptr)) {
      return nullptr;
    }

    auto value_1 = GetValueNode(vnode_1);
    auto value_2 = GetValueNode(vnode_2);

    if (!value_1->isa<tensor::Tensor>() || !value_2->isa<tensor::Tensor>()) {
      return nullptr;
    }

    auto tensor_ptr_1 = dyn_cast<tensor::Tensor>(value_1);
    auto tensor_ptr_2 = dyn_cast<tensor::Tensor>(value_2);

    auto tensor_1_abstract = vnode_1->abstract()->cast<abstract::AbstractTensorPtr>();
    auto tensor_2_abstract = vnode_1->abstract()->cast<abstract::AbstractTensorPtr>();
    auto tensor_3_abstract = node_3->abstract()->cast<abstract::AbstractTensorPtr>();

    TypePtr tensor_1_type_ptr = tensor_1_abstract->element()->BuildType();
    TypePtr tensor_2_type_ptr = tensor_2_abstract->element()->BuildType();
    TypePtr tensor_3_type_ptr = tensor_3_abstract->element()->BuildType();

    if ((tensor_1_type_ptr->type_id() != tensor_3_type_ptr->type_id()) ||
        (tensor_2_type_ptr->type_id() != tensor_3_type_ptr->type_id())) {
      return nullptr;
    }

    std::vector<int> tensor_out_shape = tensor_3_abstract->shape()->shape();

    int data_out_size = 1;
    for (auto it : tensor_out_shape) {
      data_out_size *= it;
    }
    if ((tensor_ptr_1->DataSize() > 1) && (tensor_ptr_1->DataSize() != data_out_size)) {
      return nullptr;
    }
    if ((tensor_ptr_2->DataSize() > 1) && (tensor_ptr_2->DataSize() != data_out_size)) {
      return nullptr;
    }

    void *data_out;

    if ((tensor_3_type_ptr->type_id() == TypeId::kNumberTypeFloat32) ||
        (tensor_3_type_ptr->type_id() == TypeId::kNumberTypeFloat)) {
      Multiply<float>(tensor_ptr_1->data_c(), tensor_ptr_1->DataSize(), tensor_ptr_2->data_c(),
                      tensor_ptr_2->DataSize(), &data_out, data_out_size);
    } else {
      if (tensor_3_type_ptr->type_id() == TypeId::kNumberTypeFloat64) {
        Multiply<double>(tensor_ptr_1->data_c(), tensor_ptr_1->DataSize(), tensor_ptr_2->data_c(),
                         tensor_ptr_2->DataSize(), &data_out, data_out_size);
      } else {
        if ((tensor_3_type_ptr->type_id() == TypeId::kNumberTypeInt32) ||
            (tensor_3_type_ptr->type_id() == TypeId::kNumberTypeInt)) {
          Multiply<int>(tensor_ptr_1->data_c(), tensor_ptr_1->DataSize(), tensor_ptr_2->data_c(),
                        tensor_ptr_2->DataSize(), &data_out, data_out_size);
        } else {
          // Un-support data types
          return nullptr;
        }
      }
    }

    auto new_tensor_ptr = std::make_shared<tensor::Tensor>(tensor_3_type_ptr->type_id(), tensor_out_shape);
    size_t mem_size = GetTypeByte(tensor_3_type_ptr) * IntToSize(new_tensor_ptr->ElementsNum());
    char *data = reinterpret_cast<char *>(new_tensor_ptr->data_c(true));
    memcpy(data, data_out, mem_size);

    auto new_vnode = NewValueNode(new_tensor_ptr);
    new_vnode->set_abstract(new_tensor_ptr->ToAbstract());
    return new_vnode;
  }

  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    Reset();
    // {prim::kPrimMul, Tensor1, {...}}
    AnfVisitor::Match(prim::kPrimMul, {IsNode, IsNode})(node);
    if (vnode_ == nullptr || c_p_node_ == nullptr) {
      return nullptr;
    }

    if (!IsCNode(c_p_node_)) {
      return nullptr;
    }

    auto tensor1 = vnode_;
    auto mul = c_p_node_->cast<CNodePtr>();

    Reset();
    // {prim::kPrimMul, Tensor2, {...}}
    AnfVisitor::Match(prim::kPrimMul, {IsNode, IsNode})(mul);
    if (vnode_ == nullptr || c_p_node_ == nullptr) {
      return nullptr;
    }
    auto tensor2 = vnode_;
    auto c_p_node = c_p_node_;

    auto PrimMul = GetValueNode<PrimitivePtr>(mul->input(0));
    auto fg = node->func_graph();

    auto new_mul_tensor = MulConstantTensors(tensor1, tensor2, c_p_node);
    if (new_mul_tensor == nullptr) {
      auto ttmul = NewCNode({NewValueNode(PrimMul), tensor1, tensor2}, fg);
      return NewCNode({NewValueNode(PrimMul), c_p_node, ttmul}, fg);
    }
    return NewCNode({NewValueNode(PrimMul), c_p_node, new_mul_tensor}, fg);
  }

  void Visit(const AnfNodePtr &node) override {
    if (IsValueNode<tensor::Tensor>(node)) {
      vnode_ = node;
    }

    if (IsCNode(node) || IsParam(node)) {
      c_p_node_ = node;
    }
  }

  void Reset() {
    vnode_ = nullptr;
    c_p_node_ = nullptr;
  }

 private:
  AnfNodePtr vnode_;
  AnfNodePtr c_p_node_;
};

class PowerOneEliminate : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    if (!IsPrimitiveCNode(node, prim::kPrimPow) || node->func_graph() == nullptr) {
      return nullptr;
    }

    auto &inputs = node->cast<CNodePtr>()->inputs();
    if (!IsValueNode<Scalar>(inputs[2])) {
      return nullptr;
    }
    auto scalar = GetValueNode<ScalarPtr>(inputs[2]);
    if (scalar->isa<FloatImm>() && GetValue<float>(scalar) == 1.0) {
      return inputs[1];
    } else if (scalar->isa<IntergerImm>() && GetValue<int>(scalar) == 1) {
      return inputs[1];
    }
    return nullptr;
  }
};

// grad = AllReduce(grad) / worker_number
// grad = grad + weight * decy
// ->
// grad = grad + weight * decy
// grad = AllReduce(grad) / worker_number

// {prim::kPrimAddN, {prim::kPrimMakeTuple, {prim::kPrimMul, {prim::kPrimAllReduce, X}, Y}, Z}} ->
// {prim::kPrimMul, {prim::kPrimAllReduce, {prim::kPrimAddN,{prim::kPrimMakeTuple, Z, X}}}, Y}
class AdjustAllReduceMulAdd : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    Reset();
    // {prim::kPrimAddN, Zs}
    if (!IsPrimitiveCNode(node, prim::kPrimAddN)) {
      return nullptr;
    }
    auto addn = node->cast<CNodePtr>();
    if (addn->size() != 2) {
      return nullptr;
    }
    AnfVisitor::Match(prim::kPrimMakeTuple, {IsNode, IsNode})(addn->input(1));
    if (x_ == nullptr || y_ == nullptr || z_ == nullptr || all_reduce_fg_ == nullptr) {
      return nullptr;
    }
    auto addn_maketuple = addn->input(1);

    auto fg = all_reduce_fg_;
    // addn inputs cross the graph, make the inputs same as allreduce node.
    if (z_->isa<CNode>() && fg != z_->func_graph()) {
      auto cnode_z = z_->cast<CNodePtr>();
      z_ = NewCNode(cnode_z->inputs(), fg);
    }

    auto addn_op_node = addn->input(0);
    auto make_tuple_op_node = addn->input(1)->cast<CNodePtr>()->input(0);

    AnfNodePtr tuple = NewCNode({make_tuple_op_node, z_, x_}, fg);
    AnfNodePtr add = NewCNode({addn_op_node, tuple}, fg);
    AnfNodePtr all_reduce = NewCNode({all_reduce_, add}, fg);
    AnfNodePtr mul = NewCNode({mul_, all_reduce, y_}, fg);
    ProcessDependEdge(fg, addn_maketuple, all_reduce);
    return mul;
  }
  void ProcessDependEdge(const FuncGraphPtr &fg, const AnfNodePtr &addn_maketuple, const AnfNodePtr &new_node) {
    // If has dynamic loss scale.
    auto &users_map = fg->manager()->node_users();
    auto it = users_map.find(mul_cnode_);
    if (it != users_map.end()) {
      auto users = it->second;
      for (auto &user_pair : users) {
        auto node = user_pair.first;
        if (node != addn_maketuple) {
          if (IsPrimitiveCNode(node, prim::kPrimMakeTuple)) {
            fg->manager()->SetEdge(node, user_pair.second, new_node);
          }
        }
      }
    }
  }
  void Visit(const AnfNodePtr &node) override {
    if (level_ == 0) {
      level_ = 1;
      is_reduce_match_ = false;
      // {prim::kPrimMul, {prim::kPrimAllReduce, X}, Y}
      AnfVisitor::Match(prim::kPrimMul)(node);
      level_ = 0;
      if (is_reduce_match_) {
        mul_ = node->cast<CNodePtr>()->input(0);
        mul_cnode_ = node->cast<CNodePtr>();
        y_ = tmp_;
      } else {
        z_ = node;
      }
    }

    if (level_ == 1) {
      // {prim::kPrimAllReduce, X}
      if (IsPrimitiveCNode(node, prim::kPrimAllReduce)) {
        auto cnode = node->cast<CNodePtr>();
        if (cnode->size() > 1) {
          all_reduce_ = cnode->input(0);
          x_ = cnode->input(1);
          is_reduce_match_ = true;
          all_reduce_fg_ = cnode->func_graph();
        }
      } else {
        tmp_ = node;
      }
    }
  }

  void Reset() {
    level_ = 0;
    is_reduce_match_ = false;
    x_ = nullptr;
    y_ = nullptr;
    z_ = nullptr;
    tmp_ = nullptr;
    all_reduce_fg_ = nullptr;
  }

 private:
  int level_{0};
  bool is_reduce_match_{false};
  AnfNodePtr x_{nullptr}, y_{nullptr}, z_{nullptr}, tmp_{nullptr};
  AnfNodePtr all_reduce_{nullptr}, mul_{nullptr}, mul_cnode_{nullptr};
  FuncGraphPtr all_reduce_fg_{nullptr};
};

class ArithmeticSimplify {
 public:
  ArithmeticSimplify()
      : multiply_by_zero_or_one_(),
        tensor_multiply_by_one_(),
        add_by_zero_(),
        tensor_add_by_zero_(),
        identity_(prim::kPrimIdentity),
        opt_update_zero_tensor_(),
        constant_duplicate_mul_(),
        power_one_() {
    eliminaters_.emplace_back(multiply_by_zero_or_one_);
    eliminaters_.emplace_back(tensor_multiply_by_one_);
    eliminaters_.emplace_back(add_by_zero_);
    eliminaters_.emplace_back(tensor_add_by_zero_);
    eliminaters_.emplace_back(identity_);
    eliminaters_.emplace_back(opt_update_zero_tensor_);
    eliminaters_.emplace_back(constant_duplicate_mul_);
    eliminaters_.emplace_back(power_one_);
  }
  ~ArithmeticSimplify() = default;

  AnfNodePtr operator()(const OptimizerPtr &optimizer, const AnfNodePtr &node) {
    AnfNodePtr new_node;
    for (auto &eliminater : eliminaters_) {
      new_node = eliminater(optimizer, node);
      if (new_node != nullptr) {
        return new_node;
      }
    }
    return nullptr;
  }

 private:
  MultiplyByZeroOrOne multiply_by_zero_or_one_;
  TensorMultiplyByOne tensor_multiply_by_one_;
  AddByZero add_by_zero_;
  TensorAddByZero tensor_add_by_zero_;
  PrimEliminater identity_;
  OptUpdateZeroTensor opt_update_zero_tensor_;
  ConstantDuplicateMul constant_duplicate_mul_;
  PowerOneEliminate power_one_;
  std::vector<TransformFuncType> eliminaters_{};
};

// Arithmetic Simplifications should be done after step_parallel.
// eg: Mul(0, weight) where weight is a parameter will be simplified to a constant tensor
// with shape(weight), but after step_parallel, shape of weight may be changed, so the
// shape of the constant tensor should also be changed. So this pass is seperated from
// ArithmeticSimplify and deferred until step_parallel.
class ArithmeticSimplify2 {
 public:
  ArithmeticSimplify2() : tensor_multiply_by_zero_() { eliminaters_.emplace_back(tensor_multiply_by_zero_); }
  ~ArithmeticSimplify2() = default;

  AnfNodePtr operator()(const OptimizerPtr &optimizer, const AnfNodePtr &node) {
    AnfNodePtr new_node;
    for (auto &eliminater : eliminaters_) {
      new_node = eliminater(optimizer, node);
      if (new_node != nullptr) {
        return new_node;
      }
    }
    return nullptr;
  }

 private:
  TensorMultiplyByZero tensor_multiply_by_zero_;
  std::vector<TransformFuncType> eliminaters_{};
};
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_OPTIMIZER_IRPASS_ARITHMETIC_SIMPLIFY_H_
