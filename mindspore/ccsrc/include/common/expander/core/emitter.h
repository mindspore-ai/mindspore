/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_COMMON_EXPANDER_CORE_EMITTER_H_
#define MINDSPORE_CCSRC_COMMON_EXPANDER_CORE_EMITTER_H_
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>
#include "include/common/expander/core/infer.h"
#include "include/common/expander/core/node.h"
#include "include/common/utils/utils.h"
#include "ir/func_graph.h"
#include "ir/functor.h"
#include "ops/array_op_name.h"
#include "ops/comparison_op_name.h"
#include "ops/framework_op_name.h"
#include "ops/arithmetic_op_name.h"
#include "ops/math_ops.h"
#include "ops/sequence_ops.h"
#include "ops/shape_calc.h"

namespace mindspore {
namespace expander {
using ShapeValidFunc = std::function<bool(size_t, const ShapeVector &)>;

class COMMON_EXPORT Emitter {
 public:
  Emitter(const FuncGraphPtr &func_graph, const ExpanderInferPtr &infer, const ScopePtr &scope = nullptr)
      : func_graph_(func_graph), infer_(infer), scope_(scope) {
    MS_EXCEPTION_IF_NULL(func_graph);
    MS_EXCEPTION_IF_NULL(infer);
  }
  virtual ~Emitter() = default;

  /// \brief Emit a primitive CNode
  NodePtr Emit(const std::string &op_name, const NodePtrList &inputs, const DAttr &attrs = {});

  /// \brief Emit a ValueNode
  NodePtr EmitValue(const ValuePtr &value);

  NodePtr MakeTuple(const NodePtrList &inputs) { return EmitOp(prim::kPrimMakeTuple, inputs); }
  NodePtr MakeList(const NodePtrList &inputs) { return EmitOp(prim::kPrimMakeList, inputs); }
  NodePtr TupleGetItem(const NodePtr &input, size_t i) {
    return Emit(mindspore::kTupleGetItemOpName, {input, Value(static_cast<int64_t>(i))});
  }
  NodePtr TupleGetItem(const NodePtr &input, const NodePtr &i) { return Emit(kTupleGetItemOpName, {input, i}); }
  NodePtr Len(const NodePtr &input) { return Emit(kSequenceLenOpName, {input}); }
  NodePtr ScalarAdd(const NodePtr &lhs, const NodePtr &rhs) { return Emit(kScalarAddOpName, {lhs, rhs}); }
  NodePtr ScalarSub(const NodePtr &lhs, const NodePtr &rhs) { return Emit(kScalarSubOpName, {lhs, rhs}); }
  NodePtr ScalarMul(const NodePtr &lhs, const NodePtr &rhs) { return Emit(kScalarMulOpName, {lhs, rhs}); }
  NodePtr ScalarDiv(const NodePtr &lhs, const NodePtr &rhs) { return Emit(kScalarDivOpName, {lhs, rhs}); }
  NodePtr ScalarFloordiv(const NodePtr &lhs, const NodePtr &rhs) { return Emit(kScalarFloordivOpName, {lhs, rhs}); }
  NodePtr ScalarNeg(const NodePtr &node) { return Emit(kScalarUsubOpName, {node}); }

  NodePtr Cast(const NodePtr &node, const TypePtr &type);
  NodePtr Cast(const NodePtr &node, TypeId type_id) { return Cast(node, TypeIdToType(type_id)); }

  NodePtr Reshape(const NodePtr &node, const NodePtr &shape);
  NodePtr Reshape(const NodePtr &node, const ShapeVector &shape) { return Reshape(node, Value(shape)); }
  NodePtr ExpandDims(const NodePtr &node, int64_t axis) { return Emit(kExpandDimsOpName, {node, Value(axis)}); }
  NodePtr Abs(const NodePtr &node) { return Emit(mindspore::kAbsOpName, {node}); }
  NodePtr Neg(const NodePtr &node) { return Emit(mindspore::kNegOpName, {node}); }
  NodePtr Reciprocal(const NodePtr &node) { return Emit(mindspore::kReciprocalOpName, {node}); }
  NodePtr Square(const NodePtr &node) { return Emit(mindspore::kSquareOpName, {node}); }
  NodePtr Sign(const NodePtr &node) { return Emit(prim::kPrimSign->name(), {node}); }
  NodePtr Exp(const NodePtr &x);
  NodePtr Log(const NodePtr &x);
  NodePtr Transpose(const NodePtr &node, const NodePtr &perm);
  NodePtr Transpose(const NodePtr &node, const ShapeVector &perm) { return Transpose(node, Value(perm)); }
  NodePtr Tile(const NodePtr &node, const NodePtr &multiples);
  NodePtr Tile(const NodePtr &node, const ShapeVector &multiples) { return Tile(node, Value(multiples)); }
  NodePtr Concat(const NodePtrList &inputs, int64_t axis) {
    return Emit(kConcatOpName, {MakeTuple(inputs)}, {{kAttrAxis, MakeValue(axis)}});
  }

  NodePtr Add(const NodePtr &lhs, const NodePtr &rhs) { return UnifyDtypeAndEmit(mindspore::kAddOpName, lhs, rhs); }
  NodePtr Sub(const NodePtr &lhs, const NodePtr &rhs) { return UnifyDtypeAndEmit(mindspore::kSubOpName, lhs, rhs); }
  NodePtr Mul(const NodePtr &lhs, const NodePtr &rhs) { return UnifyDtypeAndEmit(mindspore::kMulOpName, lhs, rhs); }
  NodePtr Div(const NodePtr &lhs, const NodePtr &rhs) { return UnifyDtypeAndEmit(kDivOpName, lhs, rhs); }
  NodePtr RealDiv(const NodePtr &lhs, const NodePtr &rhs) {
    return UnifyDtypeAndEmit(mindspore::kRealDivOpName, lhs, rhs);
  }
  NodePtr Mod(const NodePtr &lhs, const NodePtr &rhs) { return UnifyDtypeAndEmit("Mod", lhs, rhs); }
  NodePtr Pow(const NodePtr &lhs, const NodePtr &rhs) { return UnifyDtypeAndEmit(kPowOpName, lhs, rhs); }
  NodePtr MatMul(const NodePtr &a, const NodePtr &b, bool transpose_a = false, bool transpose_b = false);
  NodePtr BatchMatMul(const NodePtr &a, const NodePtr &b, bool transpose_a = false, bool transpose_b = false);
  NodePtr Maximum(const NodePtr &lhs, const NodePtr &rhs) { return UnifyDtypeAndEmit(kMaximumOpName, lhs, rhs); }
  NodePtr Minimum(const NodePtr &lhs, const NodePtr &rhs) { return UnifyDtypeAndEmit(kMinimumOpName, lhs, rhs); }
  NodePtr FloorDiv(const NodePtr &lhs, const NodePtr &rhs) { return UnifyDtypeAndEmit("FloorDiv", lhs, rhs); }
  NodePtr FloorMod(const NodePtr &lhs, const NodePtr &rhs) { return UnifyDtypeAndEmit("FloorMod", lhs, rhs); }
  NodePtr DivNoNan(const NodePtr &lhs, const NodePtr &rhs) { return UnifyDtypeAndEmit("DivNoNan", lhs, rhs); }
  NodePtr MulNoNan(const NodePtr &lhs, const NodePtr &rhs) { return UnifyDtypeAndEmit("MulNoNan", lhs, rhs); }
  NodePtr Xdivy(const NodePtr &lhs, const NodePtr &rhs) { return UnifyDtypeAndEmit("Xdivy", lhs, rhs); }
  NodePtr Xlogy(const NodePtr &lhs, const NodePtr &rhs) { return UnifyDtypeAndEmit("Xlogy", lhs, rhs); }

  NodePtr Select(const NodePtr &cond, const NodePtr &lhs, const NodePtr &rhs) {
    auto [a, b] = UnifyDtype2(lhs, rhs);
    return Emit(kSelectOpName, {cond, a, b});
  }
  NodePtr Less(const NodePtr &lhs, const NodePtr &rhs, const TypePtr &dst_type = nullptr) {
    return CmpOpWithCast(kLessOpName, lhs, rhs, dst_type);
  }
  NodePtr LessEqual(const NodePtr &lhs, const NodePtr &rhs, const TypePtr &dst_type = nullptr) {
    return CmpOpWithCast(kLessEqualOpName, lhs, rhs, dst_type);
  }
  NodePtr Greater(const NodePtr &lhs, const NodePtr &rhs, const TypePtr &dst_type = nullptr) {
    return CmpOpWithCast(kGreaterOpName, lhs, rhs, dst_type);
  }
  NodePtr GreaterEqual(const NodePtr &lhs, const NodePtr &rhs, const TypePtr &dst_type = nullptr) {
    return CmpOpWithCast(kGreaterEqualOpName, lhs, rhs, dst_type);
  }
  NodePtr Equal(const NodePtr &lhs, const NodePtr &rhs, const TypePtr &dst_type = nullptr) {
    return CmpOpWithCast(kEqualOpName, lhs, rhs, dst_type);
  }
  NodePtr NotEqual(const NodePtr &lhs, const NodePtr &rhs, const TypePtr &dst_type = nullptr) {
    return CmpOpWithCast("NotEqual", lhs, rhs, dst_type);
  }
  NodePtr LogicalAnd(const NodePtr &lhs, const NodePtr &rhs) { return Emit("LogicalAnd", {lhs, rhs}); }
  NodePtr LogicalOr(const NodePtr &lhs, const NodePtr &rhs) { return Emit("LogicalOr", {lhs, rhs}); }
  NodePtr LogicalNot(const NodePtr &x) { return Emit("LogicalNot", {x}); }

  NodePtr OnesLike(const NodePtr &x) { return Emit("OnesLike", {x}); }
  NodePtr UnsortedSegmentSum(const NodePtr &x, const NodePtr &segment_ids, const NodePtr &num_segments) {
    return Emit("UnsortedSegmentSum", {x, segment_ids, num_segments});
  }
  NodePtr GatherNd(const NodePtr &input_x, const NodePtr &indices) { return Emit("GatherNd", {input_x, indices}); }
  NodePtr ScatterNd(const NodePtr &indices, const NodePtr &update, const NodePtr &shape) {
    return Emit("ScatterNd", {indices, update, shape});
  }
  NodePtr Stack(const NodePtr &x, const ValuePtr &axis) { return Emit("Stack", {x}, {{"axis", axis}}); }
  NodePtr Stack(const NodePtrList &x, int64_t axis) { return Stack(MakeTuple(x), MakeValue(axis)); }
  NodePtr TensorScatterUpdate(const NodePtr &input_x, const NodePtr &indices, const NodePtr &updates) {
    return Emit("TensorScatterUpdate", {input_x, indices, updates});
  }
  NodePtr Slice(const NodePtr &x, const NodePtr &begin, const NodePtr &size) { return Emit("Slice", {x, begin, size}); }
  NodePtr Squeeze(const NodePtr &x, const ValuePtr &axis) { return Emit("Squeeze", {x}, {{"axis", axis}}); }
  NodePtr Sqrt(const NodePtr &x) { return Emit("Sqrt", {x}); }
  NodePtr MatrixSetDiagV3(const NodePtr &x, const NodePtr &diagonal, const NodePtr &k, const ValuePtr &align) {
    const auto diag_max_length = 200000000;
    return Emit("MatrixSetDiagV3", {x, diagonal, k},
                {{"max_length", MakeValue<int64_t>(diag_max_length)}, {"align", align}});
  }
  NodePtr MatrixDiagPartV3(const NodePtr &x, const NodePtr &diagonal, const NodePtr &k, const ValuePtr &align) {
    const auto diag_max_length = 200000000;
    return Emit("MatrixDiagPartV3", {x, diagonal, k},
                {{"max_length", MakeValue<int64_t>(diag_max_length)}, {"align", align}});
  }
  NodePtr LinSpace(const NodePtr &start, const NodePtr &stop, const NodePtr &num) {
    return Emit("LinSpace", {start, stop, num});
  }

  // complex
  NodePtr Conj(const NodePtr &input) {
    TypeId type_id = input->dtype()->type_id();
    if (type_id == kNumberTypeComplex64 || type_id == kNumberTypeComplex128) {
      return Emit("Conj", {input});
    }
    return input;
  }
  NodePtr Complex(const NodePtr &real, const NodePtr &imag) { return Emit("Complex", {real, imag}); }
  NodePtr Real(const NodePtr &x) { return Emit(kRealOpName, {x}); }
  NodePtr Imag(const NodePtr &x) { return Emit(kImagOpName, {x}); }

  NodePtr CumProd(const NodePtr &x, const NodePtr &axis, const ValuePtr &exclusive, const ValuePtr &reverse) {
    return Emit("CumProd", {x, axis}, {{"exclusive", exclusive}, {"reverse", reverse}});
  }
  NodePtr CumProd(const NodePtr &x, const NodePtr &axis, const bool &exclusive, const bool &reverse) {
    return CumProd(x, axis, MakeValue(exclusive), MakeValue(reverse));
  }
  NodePtr CumSum(const NodePtr &x, const NodePtr &axis, const ValuePtr &exclusive, const ValuePtr &reverse) {
    return Emit("CumSum", {x, axis}, {{"exclusive", exclusive}, {"reverse", reverse}});
  }
  NodePtr CumSum(const NodePtr &x, const NodePtr &axis, const bool &exclusive, const bool &reverse) {
    return CumSum(x, axis, MakeValue(exclusive), MakeValue(reverse));
  }
  NodePtr CSR2COO(const NodePtr &indptr, const NodePtr &nnz) { return Emit("CSR2COO", {indptr, nnz}); }

  std::pair<bool, ShapeVector> NeedReduce(const ShapeVector &shape, const std::vector<int64_t> &axis, bool keep_dim,
                                          bool skip_mode = false) const;
  std::pair<bool, NodePtr> NeedReduce(const NodePtr &shape, const NodePtr &axis, bool keep_dim, bool skip_mode = false);
  NodePtr ReduceSum(const NodePtr &x, const NodePtr &axis, bool keep_dims = false, bool skip_mode = false);
  NodePtr ReduceSum(const NodePtr &x, const ShapeVector &axis = {}, bool keep_dims = false);
  NodePtr BroadcastTo(const NodePtr &x, const NodePtr &y);

  NodePtr ZerosLike(const NodePtr &node);
  NodePtr Depend(const NodePtr &value, const NodePtr &expr) {
    return Emit("Depend", {value, expr}, {{"side_effect_propagate", MakeValue(1)}});
  }
  NodePtr Fill(double value, const ShapeVector &shape, TypeId data_type);
  NodePtr Fill(int64_t value, const ShapeVector &shape, TypeId data_type);
  template <typename T>
  NodePtr Fill(const T &value, const NodePtr &shape, TypeId data_type) {
    MS_EXCEPTION_IF_NULL(shape);
    if (shape->isa<ValueNode>()) {
      auto value_node = shape->get<ValueNodePtr>();
      MS_EXCEPTION_IF_NULL(value_node);
      auto v = value_node->value();
      MS_EXCEPTION_IF_NULL(v);
      return Fill(value, GetValue<ShapeVector>(v), data_type);
    }
    auto value_tensor = Cast(Tensor(value), data_type);
    return Emit("DynamicBroadcastTo", {value_tensor, shape});
  }

  NodePtr Shape(const NodePtr &node, bool tensor = false) {
    auto shape = node->shape();
    if (tensor) {
      return IsDynamic(shape) ? Emit("TensorShape", {node}) : Tensor(shape);
    } else {
      return IsDynamic(shape) ? Emit("Shape", {node}) : Value<ShapeVector>(shape);
    }
  }

  NodePtr Gather(const NodePtr &params, const NodePtr &indices, int64_t axis, int64_t batch_dims = 0);
  NodePtr Gather(const NodePtr &params, const NodePtr &indices, const NodePtr &axis, int64_t batch_dims = 0);
  NodePtr GatherD(const NodePtr &x, const NodePtr &dim, const NodePtr &index) {
    return Emit("GatherD", {x, dim, index});
  }

  /// \brief Emit a value node
  template <typename T>
  NodePtr Value(const T &value) {
    return EmitValue(MakeValue(value));
  }

  /// \brief Emit a Tensor node.
  template <typename T>
  NodePtr Tensor(T data, TypePtr type_ptr = nullptr) {
    auto tensor_ptr = std::make_shared<tensor::Tensor>(data, type_ptr);
    return EmitValue(tensor_ptr);
  }

  /// \brief Emit a tensor node.
  NodePtr Tensor(TypeId data_type, const ShapeVector &shape, void *data, TypeId src_data_type) {
    auto tensor_ptr = std::make_shared<tensor::Tensor>(data_type, shape, data, src_data_type);
    return EmitValue(tensor_ptr);
  }

  /// \brief get the ExpanderInferPtr
  const ExpanderInferPtr &infer() const { return infer_; }

  /// \brief Shape calculation. This interface is used to unify the code between static-shape and dynamic-shape
  /// situation, the output type is depend on types of inputs.
  ///
  /// \param[in] functor The ShapeCalcFunctor object.
  /// \param[in] inputs The input tensors.
  /// \param[in] value_depend If index i exists in 'value_depend', the value of inputs[i] is sent to 'functor'.
  ///                         otherwise the shape of inputs[i] is sent.
  /// \param[in] valid_func The function to check whether the index and input shape is valid.
  /// \return NodePtrList, the outputs shape list. When inputs are all static-shape tensors, shape vectors are returned.
  /// otherwise CNode tensors are returned.
  NodePtrList ShapeCalc(const ShapeCalcFunctorPtr &functor, const NodePtrList &inputs,
                        const std::vector<int64_t> &value_depend = {}, const ShapeValidFunc &valid_func = nullptr);

  using BlockFunc = std::function<NodePtrList(Emitter *)>;
  /// \brief Generate a conditional block.
  ///
  /// \param[in] cond condition node, it should be a tensor of Bool.
  /// \param[in] true_case  the true branch.
  /// \param[in] false_case the false branch.
  /// \return node of tuple or single value, which is depends on the output list of two branches.
  /// \note The overloaded operators (like a+b) should not be used for captured variables in the true_case/false_case
  /// functions, use the function argument `Emitter` instead, like `emitter->Add(a, b)`. The output list of two branches
  /// should match the join rules of control flow.
  NodePtr Conditional(const NodePtr &cond, const BlockFunc &true_case, const BlockFunc &false_case);

  /// \brief Generate a while-loop block.
  ///
  /// \param[in] cond condition node, it should be a tensor of Bool.
  /// \param[in] body  the loop body.
  /// \param[in] init_list the initial variables that would be modified in body.
  /// \return node of tuple or single value, which is depends on the init_list.
  /// \note The overloaded operators (like `a+b`) should not be used for captured variables in the body function, use
  /// the function argument `Emitter` instead, like `emitter->Add(a, b)`. The length and node order of the output list
  /// of the body function should match init_list.
  NodePtr While(const NodePtr &cond, const BlockFunc &body, const NodePtrList &init_list);

 protected:
  virtual NodePtr EmitOp(const PrimitivePtr &prim, const NodePtrList &inputs);
  NodePtr NewNode(const AnfNodePtr &anfnode) { return std::make_shared<Node>(anfnode, this); }
  NodePtr CmpOpWithCast(const std::string &op, const NodePtr &lhs, const NodePtr &rhs, const TypePtr &dst_type) {
    auto node = UnifyDtypeAndEmit(op, lhs, rhs);
    return dst_type == nullptr ? node : Cast(node, dst_type);
  }
  std::tuple<NodePtr, NodePtr> UnifyDtype2(const NodePtr &lhs, const NodePtr &rhs);
  NodePtr UnifyDtypeAndEmit(const std::string &op, const NodePtr &a, const NodePtr &b, const DAttr &attrs = {}) {
    auto [lhs, rhs] = UnifyDtype2(a, b);
    return Emit(op, {lhs, rhs}, attrs);
  }

  class CtrlFlowBlock;

  FuncGraphPtr func_graph_;
  ExpanderInferPtr infer_{nullptr};
  ScopePtr scope_{nullptr};
  inline static const std::vector<size_t> type_vector_ = [] {
    std::vector<size_t> type_vector(kSparseTypeEnd + 1);
    type_vector[kNumberTypeBool] = 1;
    type_vector[kNumberTypeInt8] = 2;
    type_vector[kNumberTypeUInt8] = 3;
    type_vector[kNumberTypeInt16] = 4;
    type_vector[kNumberTypeInt32] = 5;
    type_vector[kNumberTypeInt64] = 6;
    type_vector[kNumberTypeFloat16] = 7;
    type_vector[kNumberTypeFloat32] = 8;
    type_vector[kNumberTypeFloat64] = 9;
    return type_vector;
  }();
  static HashMap<std::string, ops::OpPrimCDefineFunc> &primc_func_cache() {
    static HashMap<std::string, ops::OpPrimCDefineFunc> cache{};
    return cache;
  }
};
using EmitterPtr = std::shared_ptr<Emitter>;

COMMON_EXPORT NodePtr operator+(const NodePtr &lhs, const NodePtr &rhs);
COMMON_EXPORT NodePtr operator-(const NodePtr &lhs, const NodePtr &rhs);
COMMON_EXPORT NodePtr operator*(const NodePtr &lhs, const NodePtr &rhs);
COMMON_EXPORT NodePtr operator/(const NodePtr &lhs, const NodePtr &rhs);
COMMON_EXPORT NodePtr operator-(const NodePtr &node);

class PureShapeCalc : public ShapeCalcFunctor {
 public:
  using CalcFunc = ShapeArray (*)(const ShapeArray &);
  using InferFunc = std::vector<int64_t> (*)(const ShapeArray &, const HashSet<size_t> &);
  explicit PureShapeCalc(const std::string &name) : ShapeCalcFunctor(name) {
    FunctorRegistry::Instance().Register(name, [this]() { return shared_from_base<Functor>(); });
  }
  PureShapeCalc(const PureShapeCalc &) = delete;
  PureShapeCalc(PureShapeCalc &&) = delete;
  PureShapeCalc &operator=(const PureShapeCalc &) = delete;
  PureShapeCalc &operator=(PureShapeCalc &&) = delete;
  ~PureShapeCalc() override = default;
  MS_DECLARE_PARENT(PureShapeCalc, ShapeCalcFunctor)

  ValuePtr ToValue() const override { return nullptr; }
  void FromValue(const ValuePtr &) override {}
  ShapeArray Calc(const ShapeArray &inputs) const override {
    MS_EXCEPTION_IF_CHECK_FAIL(calc_func_ != nullptr, "The calc_func of " + name() + " is nullptr");
    return calc_func_(inputs);
  }
  std::vector<int64_t> Infer(const ShapeArray &inputs, const HashSet<size_t> &unknown_inputs) const override {
    MS_EXCEPTION_IF_CHECK_FAIL(infer_func_ != nullptr, "The infer_func of " + name() + " is nullptr");
    return infer_func_(inputs, unknown_inputs);
  }

  PureShapeCalc &SetCalc(const CalcFunc &calc_func) {
    calc_func_ = calc_func;
    return *this;
  }
  std::shared_ptr<PureShapeCalc> SetInfer(const InferFunc &infer_func) {
    infer_func_ = infer_func;
    return shared_from_base<PureShapeCalc>();
  }

  CalcFunc calc_func_{nullptr};
  InferFunc infer_func_{nullptr};
};

#define DEF_PURE_SHAPE_CALC(name) \
  static const std::shared_ptr<PureShapeCalc> name = (*(std::make_shared<PureShapeCalc>("ShapeCalc_" #name)))

}  // namespace expander
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_COMMON_EXPANDER_CORE_EMITTER_H_
