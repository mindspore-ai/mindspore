/*
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPERATOR_META_DSL_COMMON_META_IMPL_H_
#define MINDSPORE_CCSRC_FRONTEND_OPERATOR_META_DSL_COMMON_META_IMPL_H_

#include <string>
#include <map>
#include <set>
#include <stack>
#include <vector>
#include <memory>
#include <utility>
#include <algorithm>
#include <unordered_map>
#include "ir/manager.h"
#include "ops/op_def.h"
#include "frontend/operator/meta_dsl/common/utils.h"
#include "frontend/operator/meta_dsl/common/meta_func_builder.h"
#include "primitive/structure_ops.h"

namespace mindspore::prim {
using NodePtr = AnfNodePtr;
using NodePtrList = AnfNodePtrList;
using BlockFunc = std::function<void()>;
using CheckFunc = std::function<void(const PrimitivePtr &, const std::vector<AbstractBasePtr> &)>;

class MetaImpl : public MetaFuncGraph {
 public:
  explicit MetaImpl(const std::string &name) : MetaFuncGraph(name + "MetaImpl"), name_(name) {}
  ~MetaImpl() override = default;
  MS_DECLARE_PARENT(MetaImpl, MetaFuncGraph)
  void set_prim(const PrimitivePtr &prim);
  void set_manager(const FuncGraphManagerPtr &manager);
  PrimitivePtr prim() const;
  FuncGraphPtr GenerateFuncGraph(const AbstractBasePtrList &input_args) override;
  virtual void GenerateFunction() = 0;

 protected:
  /// \brief Get Primitive.
  ///
  /// \note Example: Prim(Add), Prim(Mul)
  ///
  /// \param[in] name The name of Primitive.
  ///
  /// \return Primitive instance.
#define Prim(name) kPrim##name

  /// \brief Create a node with value.
  ///
  /// \note Example: Value(0), Value(1.0), Value(true), Value("valid"), Value<int32_t>(100), Value(kNone)
  ///
  /// \param[in] value Supports int, float, bool, char*, and other types allowed by MakeValue.
  ///
  /// \return ValueNode.
  template <typename S, typename U = typename ImmTraits<S>::type::element_type>
  inline ValueNodePtr Value(S value) {
    if (std::is_same_v<S, int>) {
      // int defaults to Int64.
      return NewValueNode(std::make_shared<Int64Imm>(static_cast<int64_t>(value)));
    }
    return NewValueNode(std::make_shared<U>(value));
  }
  template <typename T, typename U = typename std::enable_if<is_vector<T>::value, typename T::value_type>::type>
  ValueNodePtr Value(const T &vec) {
    std::vector<ValuePtr> list;
    (void)std::transform(vec.begin(), vec.end(), std::back_inserter(list), [](U ele) { return MakeValue(ele); });
    return NewValueNode(std::make_shared<ValueTuple>(list));
  }
  inline ValueNodePtr Value(const ValuePtr &value) { return NewValueNode(value); }
  inline ValueNodePtr Value(const std::vector<ValuePtr> &v) { return NewValueNode(std::make_shared<ValueTuple>(v)); }
  inline ValueNodePtr Value(std::initializer_list<ValuePtr> v) { return NewValueNode(std::make_shared<ValueTuple>(v)); }

  /// \brief Create a call node, whose first input is usually a Primitive.
  ///
  /// \note Example: Call(Prim(Add), x, y), Call(Prim(Rank), x)
  ///
  /// \param[in] prim A primitive.
  /// \param[in] args Nodes as inputs.
  ///
  /// \return CNode.
  template <typename... TArgs>
  inline NodePtr Call(const PrimitivePtr &prim, const TArgs &...args) {
    NodePtr prim_node = nullptr;
    if (ops::IsPrimitiveFunction(prim->name())) {
      prim_node = NewValueNode(std::make_shared<prim::DoTransPrimitiveFunction>(prim));
    } else {
      prim_node = NewValueNode(prim);
    }
    NodePtrList nodes = {prim_node, args...};
    return NewNode(nodes);
  }
  template <typename... TArgs>
  inline NodePtr Call(const TArgs &...args) {
    NodePtrList nodes = {args...};
    return NewNode(nodes);
  }

  /// \brief Set output.
  ///
  /// \note Example: Return(out)
  ///
  /// \param[in] output The output of graph.
  void Return(const NodePtr &output);

  /// \brief if-else expression.
  ///
  /// \note Example:
  ///         # python                       // cpp
  ///         if condition:                  auto true_case = [&]() { Return(x); };
  ///           return x         -->         auto false_case = [&]() { Return(y); };
  ///         return y                       auto out = If(condition, true_case, false_case)
  ///
  /// \param[in] condition The condition of if-else expression.
  /// \param[in] true_branch True branch.
  /// \param[in] false_branch False branch.
  ///
  /// \return The result node of if-else expression.
  NodePtr If(const NodePtr &condition, const BlockFunc &true_branch, const BlockFunc &false_branch);

  /// \brief if-elif-else expression. If({{cond1, br1}, {cond2, br2}, ...}, else_br)
  ///
  /// \param[in] if_branchs If conditions and corresponding branches.
  /// \param[in] else_branch Else branch.
  ///
  /// \return The result node of if-elif-else expression.
  NodePtr If(const std::vector<std::pair<NodePtr, BlockFunc>> &if_branches, const BlockFunc &else_branch);

  /// \brief for-loop.
  ///
  /// \note Example:
  ///         # python                                         // cpp
  ///         result = ...                                   auto loop_func = [&](const NodePtr &index,
  ///         for index, item in enumerate(sequence):   -->                       const NodePtr &item,
  ///           result = loop_func(index, item, result)                           const NodePtr &result) { ... };
  ///                                                        result = For(loop_func, sequence, result, lower, upper);
  ///
  /// \param[in] loop_func The loop function, take 3 argument and return value has the same type with result argument.
  /// \param[in] sequence Object that needs to be iterated.
  /// \param[in] result The result of for-loop.
  /// \param[in] lower The start index of loop.
  /// \param[in] upper The end index of loop.
  ///
  /// \return The result node of for-loop expression.
  NodePtr For(const std::function<void(const NodePtr &, const NodePtr &, const NodePtr &)> &loop_func,
              const NodePtr &sequence, const NodePtr &result, const NodePtr &lower = nullptr,
              const NodePtr &upper = nullptr);

  /// \brief Refer to `mindspore.ops.ForiLoop` for more details.
  ///
  /// \note Example:
  ///         # python                                      // cpp
  ///         for i in range(lower, upper):                 auto loop_func =
  ///           init_val = loop_func(i, init_val)    -->      [&](const NodePtr &index, const NodePtr &res) { ... };
  ///         return init_val                               auto out = ForiLoop(cond_func, loop_func, init_val);
  ///
  /// \param[in] lower The start index of loop.
  /// \param[in] upper The end index of loop.
  /// \param[in] loop_func The loop function, takes two arguments.
  /// \param[in] init_val The init value. Supports Tensor, number, str, bool, list, tuple, dict.
  ///
  /// \return The result node of for-loop expression.
  NodePtr ForiLoop(const NodePtr &lower, const NodePtr &upper,
                   const std::function<void(const NodePtr &, const NodePtr &)> &loop_func, const NodePtr &init_val);

  /// \brief while-loop. Refer to `mindspore.ops.WhileLoop` for more details.
  ///
  /// \note Example:
  ///         # python                                       // cpp
  ///         while(cond_func(init_val)):                    auto cond_func = [&](const NodePtr &x) { ... };
  ///           init_val = loop_func(init_val)      -->      auto loop_func = [&](const NodePtr &x) { ... };
  ///         return init_val                                auto out = While(cond_func, loop_func, init_val);
  ///
  /// \param[in] cond_func The condition function.
  /// \param[in] loop_func The loop function, take one argument and return value has the same type with input argument.
  /// \param[in] init_val The initial value. Supports Tensor, number, str, bool, list, tuple, dict.
  ///
  /// \return The result node of while-loop expression.
  NodePtr While(const std::function<void(const NodePtr &)> &cond_func,
                const std::function<void(const NodePtr &)> &loop_func, const NodePtr &init_val);

  /// \brief Scan a function over an array while the processing of the current element depends on the execution result
  ///        of the previous element. Refer to `mindspore.ops.Scan` for more details.
  ///
  /// \note Example:
  ///         # python                                  // cpp
  ///         if xs is None:                            auto loop_func = [&](const NodePtr &x, const NodePtr &elem) {
  ///           xs = [None] * length           -->        ...
  ///         carry = init                              };
  ///         ys = []                                   auto [carry, ys] = Scan(loop_func, init, xs, length);
  ///         for x in xs:
  ///           carry, y = loop_func(carry, x)
  ///           ys.append(y)
  ///         return carry, ys
  ///
  /// \param[in] loop_func The loop function.
  /// \param[in] init An initial loop carry value. Supports Tensor, number, str, bool, list, tuple, dict.
  /// \param[in] xs The value over which to scan.
  /// \param[in] length Optional. The size of xs.
  ///
  /// \return The result node of scan.
  NodePtr Scan(const std::function<void(const NodePtr &, const NodePtr &)> &loop_func, const NodePtr &init,
               const NodePtr &xs, const NodePtr &length = NewValueNode(kNone));

  /// \brief Create a new tuple, such as (x, y).
  ///
  /// \note Example: Tuple(x, y)
  ///
  /// \param[in] args Input nodes.
  ///
  /// \return Node with MakeTuple.
  template <typename... TArgs>
  inline NodePtr Tuple(const TArgs &...args) {
    return Call(NewValueNode(prim::kPrimMakeTuple), args...);
  }

  NodePtr MakeTuple(const std::vector<NodePtr> &nodes);

  /// \brief Create a new list, such as [0, 1, 2].
  ///
  /// \note Example: List(NewValue(0), NewValue(1), NewValue(2))
  ///
  /// \param[in] args Input nodes.
  ///
  /// \return Node with MakeList.
  template <typename... TArgs>
  inline NodePtr List(const TArgs &...args) {
    return Call(NewValueNode(prim::kPrimMakeList), args...);
  }

  /// \brief Create a new tuple from list, such as [0, 1, 2] -> (0, 1, 2).
  ///
  /// \note Example: list = List(NewValue(0), NewValue(1), NewValue(2)), ListToTuple(list)
  ///
  /// \param[in] node Input node.
  ///
  /// \return Node with ListToTuple.
  NodePtr ListToTuple(const NodePtr &node);

  /// \brief len of list or tuple
  ///
  /// \note Example: SequenceLen(node)
  ///
  /// \param[in] node Input node.
  ///
  /// \return Node with SequenceLen.
  NodePtr SequenceLen(const NodePtr &node);

  /// \brief OnesLike(x)
  ///
  /// \note Example: OnesLike(x)
  ///
  /// \param[in] x Input node.
  ///
  /// \return Output node of OnesLike.
  NodePtr OnesLike(const NodePtr &x);

  /// \brief ZerosLike(x)
  ///
  /// \note Example: ZerosLike(x)
  ///
  /// \param[in] x Input node.
  ///
  /// \return Output node of ZerosLike.
  NodePtr ZerosLike(const NodePtr &x);

  /// \brief x == y
  ///
  /// \note Example: Equal(x, y)
  ///
  /// \param[in] x Input node.
  /// \param[in] y Input node.
  ///
  /// \return Output node of Equal.
  NodePtr Equal(const NodePtr &x, const NodePtr &y);

  /// \brief x != y
  ///
  /// \note Example: NotEqual(x, y)
  ///
  /// \param[in] x Input node.
  /// \param[in] y Input node.
  ///
  /// \return Output node of NotEqual.
  NodePtr NotEqual(const NodePtr &x, const NodePtr &y);

  /// \brief x > y
  ///
  /// \note Example: Greater(x, y)
  ///
  /// \param[in] x Input node.
  /// \param[in] y Input node.
  ///
  /// \return Output node of Greater.
  NodePtr Greater(const NodePtr &x, const NodePtr &y);

  /// \brief x < y
  ///
  /// \note Example: Less(x, y)
  ///
  /// \param[in] x Input node.
  /// \param[in] y Input node.
  ///
  /// \return Output node of Less.
  NodePtr Less(const NodePtr &x, const NodePtr &y);

  /// \brief x >= y
  ///
  /// \note Example: GreaterEqual(x, y)
  ///
  /// \param[in] x Input node.
  /// \param[in] y Input node.
  ///
  /// \return Output node of GreaterEqual.
  NodePtr GreaterEqual(const NodePtr &x, const NodePtr &y);

  /// \brief x <= y
  ///
  /// \note Example: LessEqual(x, y)
  ///
  /// \param[in] x Input node.
  /// \param[in] y Input node.
  ///
  /// \return Output node of LessEqual.
  NodePtr LessEqual(const NodePtr &x, const NodePtr &y);

  /// \brief x[y]
  ///
  /// \note Example: GetItem(x, y)
  ///
  /// \param[in] x The object used with getitem.
  /// \param[in] y Index.
  ///
  /// \return Output node of GetItem.
  NodePtr GetItem(const NodePtr &x, const NodePtr &y);

  /// \brief x[y] = z
  ///
  /// \note Example: SetItem(x, y, z)
  ///
  /// \param[in] x The object used with setitem.
  /// \param[in] y Index.
  /// \param[in] z New element.
  ///
  /// \return Output node of SetItem.
  NodePtr SetItem(const NodePtr &x, const NodePtr &y, const NodePtr &z);

  /// \brief x is None
  ///
  /// \note Example: IsNone(x)
  ///
  /// \param[in] node Input node.
  ///
  /// \return Output node, used to determine whether node is None type.
  NodePtr IsNone(const NodePtr &node);

  /// \brief x is not None
  ///
  /// \note Example: IsNotNone(x)
  ///
  /// \param[in] node Input node.
  ///
  /// \return Output node, used to determine whether node is None type.
  NodePtr IsNotNone(const NodePtr &node);

  /// \brief x and y
  ///
  /// \note Example: And(x, y)
  ///
  /// \param[in] x Input node.
  /// \param[in] y Input node.
  ///
  /// \return Output node.
  NodePtr And(const NodePtr &x, const NodePtr &y);

  /// \brief all(iterable) such as all([1, 2, 3, None])
  ///
  /// \note Example: All(x)
  ///
  /// \param[in] iterable Input node.
  ///
  /// \return Output node of all operation.
  NodePtr All(const NodePtr &iterable);

  /// \brief any(iterable) such as any([1, 2, 3, None])
  ///
  /// \note Example: Any(x)
  ///
  /// \param[in] iterable Input node.
  ///
  /// \return Output node of any operation.
  NodePtr Any(const NodePtr &iterable);

  /// \brief x or y
  ///
  /// \note Example: Or(x, y)
  ///
  /// \param[in] x Input node.
  /// \param[in] y Input node.
  ///
  /// \return Output node.
  NodePtr Or(const NodePtr &x, const NodePtr &y);

  /// \brief len(x)
  ///
  /// \note Example: Len(x)
  ///
  /// \param[in] x Input node.
  ///
  /// \return Output node.
  NodePtr Len(const NodePtr &x);

  /// \brief x + y
  ///
  /// \note Example: ScalarAdd(x, y)
  ///
  /// \param[in] x Input node.
  /// \param[in] y Input node.
  ///
  /// \return Output node.
  NodePtr ScalarAdd(const NodePtr &x, const NodePtr &y);

  /// \brief x - y
  ///
  /// \note Example: ScalarSub(x, y)
  ///
  /// \param[in] x Input node.
  /// \param[in] y Input node.
  ///
  /// \return Output node.
  NodePtr ScalarSub(const NodePtr &x, const NodePtr &y);

  /// \brief x * y
  ///
  /// \note Example: ScalarMul(x, y)
  ///
  /// \param[in] x Input node.
  /// \param[in] y Input node.
  ///
  /// \return Output node.
  NodePtr ScalarMul(const NodePtr &x, const NodePtr &y);

  /// \brief x / y
  ///
  /// \note Example: ScalarDiv(x, y)
  ///
  /// \param[in] x Input node.
  /// \param[in] y Input node.
  ///
  /// \return Output node.
  NodePtr ScalarDiv(const NodePtr &x, const NodePtr &y);

  /// \brief x // y
  ///
  /// \note Example: ScalarFloorDiv(x, y)
  ///
  /// \param[in] x Input node.
  /// \param[in] y Input node.
  ///
  /// \return Output node.
  NodePtr ScalarFloorDiv(const NodePtr &x, const NodePtr &y);

  /// \brief x % y
  ///
  /// \note Example: ScalarMod(x, y)
  ///
  /// \param[in] x Input node.
  /// \param[in] y Input node.
  ///
  /// \return Output node.
  NodePtr ScalarMod(const NodePtr &x, const NodePtr &y);

  /// \brief x ** y
  ///
  /// \note Example: ScalarPow(x, y)
  ///
  /// \param[in] x Input node.
  /// \param[in] y Input node.
  ///
  /// \return Output node.
  NodePtr ScalarPow(const NodePtr &x, const NodePtr &y);

  /// \brief x's shape
  ///
  /// \note Example: Shape(x)
  ///
  /// \param[in] x Input node.
  ///
  /// \return Output node.
  NodePtr Shape(const NodePtr &x);

  /// \brief x's rank
  ///
  /// \note Example: Rank(x)
  ///
  /// \param[in] x Input node.
  ///
  /// \return Output node.
  NodePtr Rank(const NodePtr &x);

  /// \brief x's dtype_id
  ///
  /// \note Example: DTypeId(x)
  ///
  /// \param[in] x Input node.
  ///
  /// \return Output node.
  NodePtr DTypeId(const NodePtr &x);

  /// \brief reshape x
  ///
  /// \note Example: Reshape(x, shape)
  ///
  /// \param[in] x Input node.
  /// \param[in] shape Input node.
  ///
  /// \return Output node.
  NodePtr Reshape(const NodePtr &x, const NodePtr &shape);

  /// \brief not x
  ///
  /// \note Example: Not(x)
  ///
  /// \param[in] x Input node.
  ///
  /// \return Output node.
  NodePtr Not(const NodePtr &x);

  /// \brief Raise exception such as ValueError and TypeError.
  ///
  /// \note Example: Raise("ValueError", "Not supported yet")
  ///
  /// \param[in] exception_type Exception type
  /// \param[in] exception_msg Exception log message.
  ///
  /// \return Node with prim::kPrimRaise.
  NodePtr Raise(const std::string &exception_type, const std::string &exception_msg);

  /// \brief isinstance(x, int), isinstance(x, (int, Tensor))
  ///
  /// \note Example: IsInstance(x, TypeId::kNumberTypeInt),
  ///                IsInstance(x, {TypeId::kNumberTypeInt, kObjectTypeTensorType})
  ///
  /// \param[in] x Input node.
  /// \param[in] type Type to be compared.
  ///
  /// \return Node with prim::kPrimIsInstance.
  NodePtr IsInstance(const NodePtr &x, const TypeId &type);
  NodePtr IsInstance(const NodePtr &x, const std::vector<TypeId> &types);

  // Tools for implementing macro definitions, and they are basically not used during development.
  NodePtr NewParam(const std::string &name, int index = -1);
  NodePtr IfCond(const NodePtr &condition, const BlockFunc &true_branch, const BlockFunc &false_branch,
                 const NodePtrList &args);
  NodePtr IfBranchesInner(const std::vector<std::pair<NodePtr, BlockFunc>> &if_branches, const BlockFunc &else_branch,
                          size_t index);

 private:
  void BeginFunc(const std::string &func_name = "anonymous");
  FuncGraphPtr EndFunc();
  NodePtr NewNode(const NodePtrList &nodes);
  void CheckInputs(const AbstractBasePtrList &input_args) const;
  FuncGraphPtr BuildSubFunction(const std::string &func_name, const BlockFunc &sub_func);
  void DefineCustomBprop(const FuncGraphPtr &graph);
  void ConvertTypeIdToType(NodePtrList *nodes);
  void DumpIRForMetaDsl(const FuncGraphPtr &graph) const;
  NodePtr ImplAllAny(const NodePtr &input, bool is_all);

  PrimitivePtr prim_{nullptr};
  std::string name_;
  FuncGraphPtr bprop_graph_{nullptr};
  FuncGraphManagerPtr manager_{nullptr};
  std::stack<MetaFuncBuilderPtr> func_builder_stack_;
  AbstractBasePtrList input_args_;
};
using MetaImplPtr = std::shared_ptr<MetaImpl>;
using CreateFunc = std::function<std::shared_ptr<MetaImpl>()>;

class RegMetaImplFactory {
 public:
  static RegMetaImplFactory &GetInstance();
  bool IsMetaImpl(const std::string &name);
  MetaImplPtr CreateMetaImpl(const std::string &name);
  void AddMetaImpl(const std::string &name, const CreateFunc &creator);

  void RegBprop(const PrimitivePtr &prim, const CreateFunc &creator);
  FuncGraphPtr GetBprop(const PrimitivePtr &prim);

  void RegCheckFunc(const std::string &name, const CheckFunc &check_func);
  CheckFunc GetCheckFunc(const std::string &prim_name);

  class RegHelper {
   public:
    explicit RegHelper(const std::string &name, const CreateFunc &creator, const CheckFunc &check_func = nullptr) {
      RegMetaImplFactory::GetInstance().AddMetaImpl(name, creator);
      if (check_func != nullptr) {
        RegMetaImplFactory::GetInstance().RegCheckFunc(name, check_func);
      }
    }
    ~RegHelper() = default;
  };

  class RegBpropHelper {
   public:
    explicit RegBpropHelper(const PrimitivePtr &prim, const CreateFunc &creator) {
      RegMetaImplFactory::GetInstance().RegBprop(prim, creator);
    }
    ~RegBpropHelper() = default;
  };

 private:
  std::map<std::string, CreateFunc> registry_;
  std::map<std::string, CreateFunc> bprop_map_;
  std::map<std::string, CheckFunc> check_func_map_;
};
}  // namespace mindspore::prim
#endif  // MINDSPORE_CCSRC_FRONTEND_OPERATOR_META_DSL_COMMON_META_IMPL_H_
