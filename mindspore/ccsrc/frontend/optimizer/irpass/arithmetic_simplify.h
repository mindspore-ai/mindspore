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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_ARITHMETIC_SIMPLIFY_H_
#define MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_ARITHMETIC_SIMPLIFY_H_

#include <algorithm>
#include <memory>
#include <vector>

#include "ir/optimizer_caller.h"
#include "ir/visitor.h"
#include "frontend/operator/ops.h"
#include "frontend/optimizer/irpass.h"
#include "frontend/optimizer/irpass/prim_eliminate.h"
#include "frontend/optimizer/optimizer.h"

namespace mindspore {
namespace opt {
namespace irpass {
// {prim::kPrimScalarMul, 0, X}, {prim::kPrimScalarMul, X, 0}
// {prim::kPrimScalarMul, 1, X}, {prim::kPrimScalarMul, X, 1}
class MultiplyByZeroOrOne : public AnfVisitor {
 public:
  MultiplyByZeroOrOne() : zero_(MakeValue(0)), one_(MakeValue(1)) {}
  ~MultiplyByZeroOrOne() override = default;

  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override;

  void Visit(const AnfNodePtr &node) override;
  void Visit(const ValueNodePtr &vnode) override;
  void Reset();

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

  bool IsTensorConstant(const ValuePtr &value);
  bool IsTensorScalarConstant(const ValuePtr &value);

 private:
  int check_value_;
};

class TensorMultiplyBase : public AnfVisitor {
 protected:
  void *GetPointerToTensorData(const AnfNodePtr &node, bool writable = false);

  // Make a new tensor (when possible) with the same shape as of `node`
  // If x is nullptr then fill new tensor will "0"
  // If x is a tensor with empty shape then fill new tensor with the single value of x
  // If x is a tensor with same shape as `node` then return x as result
  AnfNodePtr NewTensorFilledWithData(const AnfNodePtr &node, const AnfNodePtr &x = nullptr);

  AnfNodePtr x_{nullptr};
};

// {prim::kPrimMul, 0, X}, {prim::kPrimMul, X, 0}
class TensorMultiplyByZero : public TensorMultiplyBase {
 public:
  TensorMultiplyByZero() : zero_(MakeValue(0)) {}
  ~TensorMultiplyByZero() override = default;

  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override;

  void Visit(const AnfNodePtr &node) override;
  void Visit(const ValueNodePtr &vnode) override;
  void Reset();

 private:
  bool is_zero_{false};
  ValuePtr zero_;
};

// {prim::kPrimMul, 1, X}, {prim::kPrimMul, X, 1}
class TensorMultiplyByOne : public TensorMultiplyBase {
 public:
  TensorMultiplyByOne() {}
  ~TensorMultiplyByOne() override = default;
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override;

  void Visit(const AnfNodePtr &node) override;
  void Visit(const ValueNodePtr &vnode) override;
  void Reset();

 private:
  bool is_one_{false};
};

// {prim::kPrimScalarAdd, X, 0}
// {prim::kPrimScalarAdd, 0, X}
class AddByZero : public AnfVisitor {
 public:
  AddByZero() : zero_(MakeValue(0)) {}
  ~AddByZero() override = default;

  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override;

  void Visit(const AnfNodePtr &node) override;
  void Reset();

 private:
  bool is_zero_{false};
  ValuePtr zero_;
  AnfNodePtr x_{nullptr};
};

// {prim::kPrimTensorAdd, {kPrimZerosLike, Y}, X},
// {prim::kPrimTensorAdd, X, {kPrimZerosLike, Y}}
class TensorAddByZero : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override;

  void Visit(const AnfNodePtr &node) override;
  void Visit(const ValueNodePtr &vnode) override;
  void Reset();

 private:
  bool is_zero_{false};
  AnfNodePtr x_{nullptr};
};

// {PrimMomentum, {kPrimZerosLike, X}, Y, Z, Xs}  -> {prim::kPrimMakeTuple, Z, Y}
class OptUpdateZeroTensor : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override;
};

// {prim::kPrimMul, Tensor1, {orim::kPrimMul, Tensor2, {...}}} ->
// {prim::kPrimMul, {...}, {prim::kPrimMul, Tensor1, Tensor2}}
class ConstantDuplicateMul : public AnfVisitor {
 public:
  // Support function to multiply two constant tensors: partially support broadcasting shapes
  template <typename T>
  void Multiply(void *in_data_1, int in_data_1_size, void *in_data_2, int in_data_2_size, void **out_data,
                int out_data_size);

  AnfNodePtr MulConstantTensors(const AnfNodePtr &vnode_1, const AnfNodePtr &vnode_2, const AnfNodePtr &node_3);

  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override;

  void Visit(const AnfNodePtr &node) override;
  void Reset();

 private:
  AnfNodePtr vnode_;
  AnfNodePtr c_p_node_;
};

class PowerOneEliminate : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override;
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
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override;

  void ProcessDependEdge(const FuncGraphPtr &fg, const AnfNodePtr &addn_maketuple, const AnfNodePtr &new_node);
  void Visit(const AnfNodePtr &node) override;
  void Reset();

 private:
  int level_{0};
  bool is_reduce_match_{false};
  AnfNodePtr x_{nullptr}, y_{nullptr}, z_{nullptr}, tmp_{nullptr};
  AnfNodePtr all_reduce_{nullptr}, mul_{nullptr}, mul_cnode_{nullptr};
  FuncGraphPtr all_reduce_fg_{nullptr};
};

class ArithmeticSimplify : public OptimizerCaller {
 public:
  ArithmeticSimplify()
      : multiply_by_zero_or_one_(std::make_shared<MultiplyByZeroOrOne>()),
        tensor_multiply_by_one_(std::make_shared<TensorMultiplyByOne>()),
        add_by_zero_(std::make_shared<AddByZero>()),
        tensor_add_by_zero_(std::make_shared<TensorAddByZero>()),
        identity_(std::make_shared<PrimEliminater>(prim::kPrimIdentity)),
        opt_update_zero_tensor_(std::make_shared<OptUpdateZeroTensor>()),
        constant_duplicate_mul_(std::make_shared<ConstantDuplicateMul>()),
        power_one_(std::make_shared<PowerOneEliminate>()) {
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

  AnfNodePtr operator()(const OptimizerPtr &optimizer, const AnfNodePtr &node) override;

 private:
  OptimizerCallerPtr multiply_by_zero_or_one_;
  OptimizerCallerPtr tensor_multiply_by_one_;
  OptimizerCallerPtr add_by_zero_;
  OptimizerCallerPtr tensor_add_by_zero_;
  OptimizerCallerPtr identity_;
  OptimizerCallerPtr opt_update_zero_tensor_;
  OptimizerCallerPtr constant_duplicate_mul_;
  OptimizerCallerPtr power_one_;

  std::vector<OptimizerCallerPtr> eliminaters_{};
};

// Arithmetic Simplifications should be done after step_parallel.
// eg: Mul(0, weight) where weight is a parameter will be simplified to a constant tensor
// with shape(weight), but after step_parallel, shape of weight may be changed, so the
// shape of the constant tensor should also be changed. So this pass is seperated from
// ArithmeticSimplify and deferred until step_parallel.
class ArithmeticSimplify2 : public OptimizerCaller {
 public:
  ArithmeticSimplify2() : tensor_multiply_by_zero_(std::make_shared<TensorMultiplyByZero>()) {
    eliminaters_.emplace_back(tensor_multiply_by_zero_);
  }
  ~ArithmeticSimplify2() = default;

  AnfNodePtr operator()(const OptimizerPtr &optimizer, const AnfNodePtr &node) override;

 private:
  OptimizerCallerPtr tensor_multiply_by_zero_;
  std::vector<OptimizerCallerPtr> eliminaters_{};
};
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_ARITHMETIC_SIMPLIFY_H_
