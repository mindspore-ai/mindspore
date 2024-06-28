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
#include "ops/auto_generate/gen_ops_name.h"

namespace mindspore {
namespace expander {
using ShapeValidFunc = std::function<bool(size_t, const ShapeVector &)>;

class COMMON_EXPORT Emitter {
 public:
  explicit Emitter(const ExpanderInferPtr &infer, const ScopePtr &scope = nullptr) : infer_(infer), scope_(scope) {}
  virtual ~Emitter() = default;

  /// \brief Emit a primitive CNode
  NodePtr Emit(const std::string &op_name, const NodePtrList &inputs, const DAttr &attrs = {});
  PrimitivePtr NewPrimitive(const std::string &name, const DAttr &attrs = {});

  /// \brief Emit a ValueNode
  virtual NodePtr EmitValue(const ValuePtr &value);

  NodePtr NewIrNode(const AnfNodePtr &anfnode) { return std::make_shared<IrNode>(anfnode, this); }
  FuncNodePtr NewFuncNode(const ValuePtr &value, const abstract::AbstractBasePtr &abs, InputType input_type) {
    return std::make_shared<FuncNode>(value, abs, input_type, this);
  }
  virtual NodePtr MakeTuple(const NodePtrList &inputs) { return EmitOp(prim::kPrimMakeTuple, inputs); }
  virtual NodePtr MakeList(const NodePtrList &inputs) { return EmitOp(prim::kPrimMakeList, inputs); }
  virtual NodePtr TupleGetItem(const NodePtr &input, size_t i) {
    return Emit(mindspore::kTupleGetItemOpName, {input, Value(static_cast<int64_t>(i))});
  }
  virtual NodePtr TupleGetItem(const NodePtr &input, const NodePtr &i) { return Emit(kTupleGetItemOpName, {input, i}); }
  NodePtr Len(const NodePtr &input) { return Emit(kSequenceLenOpName, {input}); }
  NodePtr ScalarAdd(const NodePtr &lhs, const NodePtr &rhs) { return Emit(ops::kNameScalarAdd, {lhs, rhs}); }
  NodePtr ScalarSub(const NodePtr &lhs, const NodePtr &rhs) { return Emit(ops::kNameScalarSub, {lhs, rhs}); }
  NodePtr ScalarMul(const NodePtr &lhs, const NodePtr &rhs) { return Emit(ops::kNameScalarMul, {lhs, rhs}); }
  NodePtr ScalarDiv(const NodePtr &lhs, const NodePtr &rhs) { return Emit(ops::kNameScalarDiv, {lhs, rhs}); }
  NodePtr ScalarFloorDiv(const NodePtr &lhs, const NodePtr &rhs) { return Emit(ops::kNameScalarFloorDiv, {lhs, rhs}); }
  NodePtr ScalarNeg(const NodePtr &node) { return Emit(ops::kNameScalarUsub, {node}); }
  virtual NodePtr Cast(const NodePtr &node, const TypePtr &type);
  NodePtr Cast(const NodePtr &node, TypeId type_id) { return Cast(node, TypeIdToType(type_id)); }

  virtual NodePtr Reshape(const NodePtr &node, const NodePtr &shape);
  NodePtr Reshape(const NodePtr &node, const ShapeVector &shape) { return Reshape(node, Value(shape)); }
  NodePtr ExpandDims(const NodePtr &node, int64_t axis) { return Emit(kExpandDimsOpName, {node, Value(axis)}); }
  virtual NodePtr Exp(const NodePtr &x);
  NodePtr Log(const NodePtr &x);
  virtual NodePtr Transpose(const NodePtr &node, const NodePtr &perm);
  NodePtr Transpose(const NodePtr &node, const ShapeVector &perm) { return Transpose(node, Value(perm)); }
  virtual NodePtr Tile(const NodePtr &node, const NodePtr &dims);
  NodePtr Tile(const NodePtr &node, const ShapeVector &dims) { return Tile(node, Value(dims)); }
  virtual NodePtr Concat(const NodePtr &input, const NodePtr &axis) { return Emit(kConcatOpName, {input, axis}); }
  NodePtr Concat(const NodePtr &input, int64_t axis) { return Concat(input, Value(axis)); }
  NodePtr Concat(const NodePtrList &inputs, int64_t axis) {
    return Emit(kConcatOpName, {MakeTuple(inputs), Value(axis)});
  }
  virtual NodePtr Add(const NodePtr &lhs, const NodePtr &rhs) {
    return UnifyDtypeAndEmit(mindspore::kAddOpName, lhs, rhs);
  }
  virtual NodePtr Sub(const NodePtr &lhs, const NodePtr &rhs) {
    return UnifyDtypeAndEmit(mindspore::kSubOpName, lhs, rhs);
  }
  virtual NodePtr Mul(const NodePtr &lhs, const NodePtr &rhs) {
    return UnifyDtypeAndEmit(mindspore::kMulOpName, lhs, rhs);
  }
  virtual NodePtr Div(const NodePtr &lhs, const NodePtr &rhs) { return UnifyDtypeAndEmit(kDivOpName, lhs, rhs); }
  NodePtr RealDiv(const NodePtr &lhs, const NodePtr &rhs) {
    return UnifyDtypeAndEmit(mindspore::kRealDivOpName, lhs, rhs);
  }
  NodePtr Mod(const NodePtr &lhs, const NodePtr &rhs) { return UnifyDtypeAndEmit("Mod", lhs, rhs); }
  virtual NodePtr Pow(const NodePtr &lhs, const NodePtr &rhs) { return UnifyDtypeAndEmit(kPowOpName, lhs, rhs); }
  virtual NodePtr MatMul(const NodePtr &a, const NodePtr &b, bool transpose_a = false, bool transpose_b = false);
  virtual NodePtr MatMulExt(const NodePtr &a, const NodePtr &b);
  virtual NodePtr BatchMatMul(const NodePtr &a, const NodePtr &b, bool transpose_a = false, bool transpose_b = false);
  virtual NodePtr Maximum(const NodePtr &lhs, const NodePtr &rhs) {
    return UnifyDtypeAndEmit(kMaximumOpName, lhs, rhs);
  }
  virtual NodePtr Minimum(const NodePtr &lhs, const NodePtr &rhs) {
    return UnifyDtypeAndEmit(kMinimumOpName, lhs, rhs);
  }
  NodePtr FloorDiv(const NodePtr &lhs, const NodePtr &rhs) { return UnifyDtypeAndEmit("FloorDiv", lhs, rhs); }
  NodePtr FloorMod(const NodePtr &lhs, const NodePtr &rhs) { return UnifyDtypeAndEmit("FloorMod", lhs, rhs); }
  NodePtr DivNoNan(const NodePtr &lhs, const NodePtr &rhs) { return UnifyDtypeAndEmit("DivNoNan", lhs, rhs); }
  NodePtr MulNoNan(const NodePtr &lhs, const NodePtr &rhs) { return UnifyDtypeAndEmit("MulNoNan", lhs, rhs); }
  NodePtr Xdivy(const NodePtr &lhs, const NodePtr &rhs) { return UnifyDtypeAndEmit("Xdivy", lhs, rhs); }
  NodePtr Xlogy(const NodePtr &lhs, const NodePtr &rhs) { return UnifyDtypeAndEmit("Xlogy", lhs, rhs); }

  virtual NodePtr Select(const NodePtr &cond, const NodePtr &lhs, const NodePtr &rhs) {
    auto [a, b] = UnifyDtype2(lhs, rhs);
    return Emit(kSelectOpName, {cond, a, b});
  }
  virtual NodePtr Less(const NodePtr &lhs, const NodePtr &rhs, const TypePtr &dst_type) {
    return CmpOpWithCast(kLessOpName, lhs, rhs, dst_type);
  }
  NodePtr Less(const NodePtr &lhs, const NodePtr &rhs) { return Less(lhs, rhs, nullptr); }
  virtual NodePtr LessEqual(const NodePtr &lhs, const NodePtr &rhs, const TypePtr &dst_type) {
    return CmpOpWithCast(kLessEqualOpName, lhs, rhs, dst_type);
  }
  NodePtr LessEqual(const NodePtr &lhs, const NodePtr &rhs) { return LessEqual(lhs, rhs, nullptr); }
  virtual NodePtr Greater(const NodePtr &lhs, const NodePtr &rhs, const TypePtr &dst_type) {
    return CmpOpWithCast(kGreaterOpName, lhs, rhs, dst_type);
  }
  NodePtr Greater(const NodePtr &lhs, const NodePtr &rhs) { return Greater(lhs, rhs, nullptr); }
  virtual NodePtr GreaterEqual(const NodePtr &lhs, const NodePtr &rhs, const TypePtr &dst_type) {
    return CmpOpWithCast(kGreaterEqualOpName, lhs, rhs, dst_type);
  }
  NodePtr GreaterEqual(const NodePtr &lhs, const NodePtr &rhs) { return GreaterEqual(lhs, rhs, nullptr); }
  virtual NodePtr Equal(const NodePtr &lhs, const NodePtr &rhs, const TypePtr &dst_type) {
    auto abs = lhs->abstract();
    MS_EXCEPTION_IF_NULL(abs);
    if (abs->isa<abstract::AbstractTensor>()) {
      return CmpOpWithCast(kEqualOpName, lhs, rhs, dst_type);
    } else if (abs->isa<abstract::AbstractScalar>()) {
      return ScalarEq(lhs, rhs, dst_type);
    }
    MS_LOG(EXCEPTION) << "'Equal' only support [Tensor] or [Scalar] input, but got: " << abs->ToString();
  }
  NodePtr Equal(const NodePtr &lhs, const NodePtr &rhs) { return Equal(lhs, rhs, nullptr); }
  virtual NodePtr ScalarEq(const NodePtr &lhs, const NodePtr &rhs, const TypePtr &dst_type) {
    auto node = UnifyDtypeAndEmit("ScalarEq", lhs, rhs);
    return dst_type == nullptr ? node : Cast(node, dst_type);
  }
  virtual NodePtr NotEqual(const NodePtr &lhs, const NodePtr &rhs, const TypePtr &dst_type) {
    return CmpOpWithCast("NotEqual", lhs, rhs, dst_type);
  }
  NodePtr NotEqual(const NodePtr &lhs, const NodePtr &rhs) { return NotEqual(lhs, rhs, nullptr); }
  NodePtr BoolNot(const NodePtr &node);

  NodePtr OnesLike(const NodePtr &x) { return Emit("OnesLike", {x}); }
  NodePtr UnsortedSegmentSum(const NodePtr &x, const NodePtr &segment_ids, const NodePtr &num_segments) {
    return Emit("UnsortedSegmentSum", {x, segment_ids, num_segments});
  }
  NodePtr GatherNd(const NodePtr &input_x, const NodePtr &indices) { return Emit("GatherNd", {input_x, indices}); }
  NodePtr ScatterNd(const NodePtr &indices, const NodePtr &update, const NodePtr &shape) {
    return Emit("ScatterNd", {indices, update, shape});
  }
  virtual NodePtr Stack(const NodePtr &x, const ValuePtr &axis) { return Emit("Stack", {x}, {{"axis", axis}}); }
  virtual NodePtr Stack(const NodePtrList &x, int64_t axis) { return Stack(MakeTuple(x), MakeValue(axis)); }
  NodePtr TensorScatterUpdate(const NodePtr &input_x, const NodePtr &indices, const NodePtr &updates) {
    return Emit("TensorScatterUpdate", {input_x, indices, updates});
  }
  NodePtr Slice(const NodePtr &x, const NodePtr &begin, const NodePtr &size) { return Emit("Slice", {x, begin, size}); }
  NodePtr Squeeze(const NodePtr &x, const ValuePtr &axis) { return Emit("Squeeze", {x}, {{"axis", axis}}); }

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

  NodePtr CumProd(const NodePtr &x, const NodePtr &axis, const NodePtr &exclusive, const NodePtr &reverse) {
    return Emit("CumProd", {x, axis, exclusive, reverse});
  }
  NodePtr CumProd(const NodePtr &x, const NodePtr &axis, const bool &exclusive, const bool &reverse) {
    return CumProd(x, axis, Value(exclusive), Value(reverse));
  }
  NodePtr CumSum(const NodePtr &x, const NodePtr &axis, const NodePtr &exclusive, const NodePtr &reverse) {
    return Emit("CumSum", {x, axis, exclusive, reverse});
  }
  NodePtr CumSum(const NodePtr &x, const NodePtr &axis, const bool &exclusive, const bool &reverse) {
    return CumSum(x, axis, Value(exclusive), Value(reverse));
  }
  NodePtr CSR2COO(const NodePtr &indptr, const NodePtr &nnz) { return Emit("CSR2COO", {indptr, nnz}); }
  NodePtr ScalarToTensor(const NodePtr &node);
  NodePtr ScalarToTensor(const NodePtr &node, const TypePtr &dtype);
  std::pair<bool, ShapeVector> NeedReduce(const ShapeVector &shape, const std::vector<int64_t> &axis, bool keep_dim,
                                          bool skip_mode = false) const;
  std::pair<bool, NodePtr> NeedReduce(const NodePtr &shape, const NodePtr &axis, bool keep_dim, bool skip_mode = false);
  NodePtr ReduceSum(const NodePtr &x, const NodePtr &axis, bool keep_dims = false, bool skip_mode = false);
  NodePtr ReduceSum(const NodePtr &x, const ShapeVector &axis = {}, bool keep_dims = false);
  NodePtr SumExt(const NodePtr &input, const NodePtr &axis, const NodePtr &keep_dims);
  virtual NodePtr BroadcastTo(const NodePtr &x, const NodePtr &y);

  NodePtr ZerosLike(const NodePtr &node);
  virtual NodePtr Depend(const NodePtr &value, const NodePtr &expr) {
    return Emit("Depend", {value, expr}, {{"side_effect_propagate", MakeValue(1)}});
  }
  NodePtr Fill(double value, const ShapeVector &shape, TypeId data_type);
  NodePtr Fill(int64_t value, const ShapeVector &shape, TypeId data_type);
  template <typename T>
  NodePtr Fill(const T &value, const NodePtr &shape, TypeId data_type) {
    MS_EXCEPTION_IF_NULL(shape);
    if (shape->input_type() == InputType::kConstant) {
      auto v = shape->BuildValue();
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
  virtual NodePtr BatchNormGrad(const NodePtrList &inputs, bool is_scale_or_bias_grad) {
    return Emit("BatchNormGrad", inputs);
  }
  virtual NodePtr SparseSoftmaxCrossEntropyWithLogits(const NodePtrList &inputs, const DAttr &attrs, const NodePtr &out,
                                                      const NodePtr &dout, bool is_graph_mode);
  // By comparing x with itself, test whether x is NaN
  inline NodePtr IsNanFunc(const NodePtr &x) { return NotEqual(x, x); }

  NodePtr Zeros(const NodePtr &x) {
    auto x_shape = x->shape();
    if (!x_shape.empty() && !IsDynamicRank(x_shape)) {
      // There are currently some problems under 0d that need to be fixed later.
      return Emit("Zeros", {Shape(x), Value<int64_t>(x->dtype()->type_id())});
    }
    return ZerosLike(x);
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
  /// \param[in] functor The ShapeCalcBaseFunctor object.
  /// \param[in] inputs The input tensors.
  /// \param[in] value_depend If index i exists in 'value_depend', the value of inputs[i] is sent to 'functor'.
  ///                         otherwise the shape of inputs[i] is sent.
  /// \param[in] valid_func The function to check whether the index and input shape is valid.
  /// \return NodePtrList, the outputs shape list. When inputs are all static-shape tensors, shape vectors are returned.
  /// otherwise CNode tensors are returned.
  virtual NodePtrList ShapeCalc(const ShapeCalcBaseFunctorPtr &functor, const NodePtrList &inputs,
                                const std::vector<int64_t> &value_depend, const ShapeValidFunc &valid_func);
  virtual NodePtrList ShapeCalc(const ShapeCalcBaseFunctorPtr &functor, const NodePtrList &inputs,
                                const std::vector<int64_t> &value_depend) {
    return ShapeCalc(functor, inputs, value_depend, nullptr);
  }
  virtual NodePtrList ShapeCalc(const ShapeCalcBaseFunctorPtr &functor, const NodePtrList &inputs) {
    return ShapeCalc(functor, inputs, {}, nullptr);
  }
  /// \brief Emit a TensorToTuple node.
  NodePtr TensorToTuple(const NodePtr &node);

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
  virtual NodePtr Conditional(const NodePtr &cond, const BlockFunc &true_case, const BlockFunc &false_case);

  /// \brief Generate a while-loop block.
  ///
  /// \param[in] cond condition node, it should be a tensor of Bool.
  /// \param[in] body  the loop body.
  /// \param[in] init_list the initial variables that would be modified in body.
  /// \return node of tuple or single value, which is depends on the init_list.
  /// \note The overloaded operators (like `a+b`) should not be used for captured variables in the body function, use
  /// the function argument `Emitter` instead, like `emitter->Add(a, b)`. The length and node order of the output list
  /// of the body function should match init_list.
  virtual NodePtr While(const NodePtr &cond, const BlockFunc &body, const NodePtrList &init_list);

  virtual NodePtr Abs(const NodePtr &input) { return Emit("Abs", {input}); }
  virtual NodePtr AdamW(const NodePtr &var, const NodePtr &m, const NodePtr &v, const NodePtr &max_v,
                        const NodePtr &gradient, const NodePtr &step, const NodePtr &lr, const NodePtr &beta1,
                        const NodePtr &beta2, const NodePtr &decay, const NodePtr &eps, const NodePtr &amsgrad,
                        const NodePtr &maximize) {
    return Emit("AdamW", {var, m, v, max_v, gradient, step, lr, beta1, beta2, decay, eps, amsgrad, maximize});
  }
  virtual NodePtr AddExt(const NodePtr &input, const NodePtr &other, const NodePtr &alpha) {
    return Emit("AddExt", {input, other, alpha});
  }
  virtual NodePtr AddLayerNormV2(const NodePtr &x1, const NodePtr &x2, const NodePtr &gamma, const NodePtr &beta,
                                 const NodePtr &epsilon, const NodePtr &additionalOut) {
    return Emit("AddLayerNormV2", {x1, x2, gamma, beta, epsilon, additionalOut});
  }
  virtual NodePtr Addmm(const NodePtr &input, const NodePtr &mat1, const NodePtr &mat2, const NodePtr &beta,
                        const NodePtr &alpha) {
    return Emit("Addmm", {input, mat1, mat2, beta, alpha});
  }
  virtual NodePtr Arange(const NodePtr &start, const NodePtr &end, const NodePtr &step, const NodePtr &dtype) {
    return Emit("Arange", {start, end, step, dtype});
  }
  virtual NodePtr ArgMaxExt(const NodePtr &input, const NodePtr &dim, const NodePtr &keepdim) {
    return Emit("ArgMaxExt", {input, dim, keepdim});
  }
  virtual NodePtr ArgMaxWithValue(const NodePtr &input, const NodePtr &axis, const NodePtr &keep_dims) {
    return Emit("ArgMaxWithValue", {input, axis, keep_dims});
  }
  virtual NodePtr ArgMinWithValue(const NodePtr &input, const NodePtr &axis, const NodePtr &keep_dims) {
    return Emit("ArgMinWithValue", {input, axis, keep_dims});
  }
  virtual NodePtr Atan2Ext(const NodePtr &input, const NodePtr &other) { return Emit("Atan2Ext", {input, other}); }
  virtual NodePtr AvgPool2DGrad(const NodePtr &grad, const NodePtr &image, const NodePtr &kernel_size,
                                const NodePtr &stride, const NodePtr &padding, const NodePtr &ceil_mode,
                                const NodePtr &count_include_pad, const NodePtr &divisor_override) {
    return Emit("AvgPool2DGrad",
                {grad, image, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override});
  }
  virtual NodePtr AvgPool2D(const NodePtr &input, const NodePtr &kernel_size, const NodePtr &stride,
                            const NodePtr &padding, const NodePtr &ceil_mode, const NodePtr &count_include_pad,
                            const NodePtr &divisor_override) {
    return Emit("AvgPool2D", {input, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override});
  }
  virtual NodePtr BatchMatMul(const NodePtr &x, const NodePtr &y, const NodePtr &transpose_a,
                              const NodePtr &transpose_b) {
    return Emit("BatchMatMul", {x, y, transpose_a, transpose_b});
  }
  virtual NodePtr BatchNormExt(const NodePtr &input, const NodePtr &weight, const NodePtr &bias,
                               const NodePtr &running_mean, const NodePtr &runnning_var, const NodePtr &training,
                               const NodePtr &momentum, const NodePtr &epsilon) {
    return Emit("BatchNormExt", {input, weight, bias, running_mean, runnning_var, training, momentum, epsilon});
  }
  virtual NodePtr BatchNormGradExt(const NodePtr &dout, const NodePtr &input, const NodePtr &weight,
                                   const NodePtr &running_mean, const NodePtr &running_var, const NodePtr &saved_mean,
                                   const NodePtr &saved_rstd, const NodePtr &training, const NodePtr &eps) {
    return Emit("BatchNormGradExt",
                {dout, input, weight, running_mean, running_var, saved_mean, saved_rstd, training, eps});
  }
  virtual NodePtr BinaryCrossEntropyGrad(const NodePtr &input, const NodePtr &target, const NodePtr &grad_output,
                                         const NodePtr &weight, const NodePtr &reduction) {
    return Emit("BinaryCrossEntropyGrad", {input, target, grad_output, weight, reduction});
  }
  virtual NodePtr BinaryCrossEntropy(const NodePtr &input, const NodePtr &target, const NodePtr &weight,
                                     const NodePtr &reduction) {
    return Emit("BinaryCrossEntropy", {input, target, weight, reduction});
  }
  virtual NodePtr BinaryCrossEntropyWithLogitsBackward(const NodePtr &grad_output, const NodePtr &input,
                                                       const NodePtr &target, const NodePtr &weight,
                                                       const NodePtr &posWeight, const NodePtr &reduction) {
    return Emit("BinaryCrossEntropyWithLogitsBackward", {grad_output, input, target, weight, posWeight, reduction});
  }
  virtual NodePtr BCEWithLogitsLoss(const NodePtr &input, const NodePtr &target, const NodePtr &weight,
                                    const NodePtr &posWeight, const NodePtr &reduction) {
    return Emit("BCEWithLogitsLoss", {input, target, weight, posWeight, reduction});
  }
  virtual NodePtr BatchMatMulExt(const NodePtr &input, const NodePtr &mat2) {
    return Emit("BatchMatMulExt", {input, mat2});
  }
  virtual NodePtr Cast(const NodePtr &input_x, const NodePtr &dtype) { return Emit("Cast", {input_x, dtype}); }
  virtual NodePtr Ceil(const NodePtr &input) { return Emit("Ceil", {input}); }
  virtual NodePtr Chunk(const NodePtr &input, const NodePtr &chunks, const NodePtr &dim) {
    return Emit("Chunk", {input, chunks, dim});
  }
  virtual NodePtr ClampScalar(const NodePtr &input, const NodePtr &min, const NodePtr &max) {
    return Emit("ClampScalar", {input, min, max});
  }
  virtual NodePtr ClampTensor(const NodePtr &input, const NodePtr &min, const NodePtr &max) {
    return Emit("ClampTensor", {input, min, max});
  }
  virtual NodePtr Col2ImExt(const NodePtr &input, const NodePtr &output_size, const NodePtr &kernel_size,
                            const NodePtr &dilation, const NodePtr &padding, const NodePtr &stride) {
    return Emit("Col2ImExt", {input, output_size, kernel_size, dilation, padding, stride});
  }
  virtual NodePtr Col2ImGrad(const NodePtr &input, const NodePtr &kernel_size, const NodePtr &dilation,
                             const NodePtr &padding, const NodePtr &stride) {
    return Emit("Col2ImGrad", {input, kernel_size, dilation, padding, stride});
  }
  virtual NodePtr ConstantPadND(const NodePtr &input, const NodePtr &padding, const NodePtr &value) {
    return Emit("ConstantPadND", {input, padding, value});
  }
  virtual NodePtr Contiguous(const NodePtr &input) { return Emit("Contiguous", {input}); }
  virtual NodePtr ConvolutionGrad(const NodePtr &dout, const NodePtr &input, const NodePtr &weight, const NodePtr &bias,
                                  const NodePtr &stride, const NodePtr &padding, const NodePtr &dilation,
                                  const NodePtr &transposed, const NodePtr &output_padding, const NodePtr &groups,
                                  const NodePtr &output_mask) {
    return Emit("ConvolutionGrad", {dout, input, weight, bias, stride, padding, dilation, transposed, output_padding,
                                    groups, output_mask});
  }
  virtual NodePtr Convolution(const NodePtr &input, const NodePtr &weight, const NodePtr &bias, const NodePtr &stride,
                              const NodePtr &padding, const NodePtr &dilation, const NodePtr &transposed,
                              const NodePtr &output_padding, const NodePtr &groups) {
    return Emit("Convolution", {input, weight, bias, stride, padding, dilation, transposed, output_padding, groups});
  }
  virtual NodePtr Copy(const NodePtr &input) { return Emit("Copy", {input}); }
  virtual NodePtr Cos(const NodePtr &input) { return Emit("Cos", {input}); }
  virtual NodePtr CumsumExt(const NodePtr &input, const NodePtr &dim, const NodePtr &dtype) {
    return Emit("CumsumExt", {input, dim, dtype});
  }
  virtual NodePtr Dense(const NodePtr &input, const NodePtr &weight, const NodePtr &bias) {
    return Emit("Dense", {input, weight, bias});
  }
  virtual NodePtr DivMod(const NodePtr &x, const NodePtr &y, const NodePtr &rounding_mode) {
    return Emit("DivMod", {x, y, rounding_mode});
  }
  virtual NodePtr Dot(const NodePtr &input, const NodePtr &other) { return Emit("Dot", {input, other}); }
  virtual NodePtr DropoutDoMaskExt(const NodePtr &input, const NodePtr &mask, const NodePtr &p) {
    return Emit("DropoutDoMaskExt", {input, mask, p});
  }
  virtual NodePtr DropoutExt(const NodePtr &input, const NodePtr &p, const NodePtr &seed, const NodePtr &offset) {
    return Emit("DropoutExt", {input, p, seed, offset});
  }
  virtual NodePtr DropoutGenMaskExt(const NodePtr &shape, const NodePtr &p, const NodePtr &seed, const NodePtr &offset,
                                    const NodePtr &dtype) {
    return Emit("DropoutGenMaskExt", {shape, p, seed, offset, dtype});
  }
  virtual NodePtr DropoutGradExt(const NodePtr &input, const NodePtr &mask, const NodePtr &p) {
    return Emit("DropoutGradExt", {input, mask, p});
  }
  virtual NodePtr EluExt(const NodePtr &input, const NodePtr &alpha) { return Emit("EluExt", {input, alpha}); }
  virtual NodePtr EluGradExt(const NodePtr &dout, const NodePtr &x, const NodePtr &alpha) {
    return Emit("EluGradExt", {dout, x, alpha});
  }
  virtual NodePtr EmbeddingDenseBackward(const NodePtr &grad, const NodePtr &indices, const NodePtr &num_weights,
                                         const NodePtr &padding_idx, const NodePtr &scale_grad_by_freq) {
    return Emit("EmbeddingDenseBackward", {grad, indices, num_weights, padding_idx, scale_grad_by_freq});
  }
  virtual NodePtr Embedding(const NodePtr &input, const NodePtr &weight, const NodePtr &padding_idx,
                            const NodePtr &max_norm, const NodePtr &norm_type, const NodePtr &scale_grad_by_freq) {
    return Emit("Embedding", {input, weight, padding_idx, max_norm, norm_type, scale_grad_by_freq});
  }
  virtual NodePtr Erf(const NodePtr &input) { return Emit("Erf", {input}); }
  virtual NodePtr Erfinv(const NodePtr &input) { return Emit("Erfinv", {input}); }
  virtual NodePtr Eye(const NodePtr &n, const NodePtr &m, const NodePtr &dtype) { return Emit("Eye", {n, m, dtype}); }
  virtual NodePtr FFNExt(const NodePtr &x, const NodePtr &weight1, const NodePtr &weight2, const NodePtr &expertTokens,
                         const NodePtr &bias1, const NodePtr &bias2, const NodePtr &scale, const NodePtr &offset,
                         const NodePtr &deqScale1, const NodePtr &deqScale2, const NodePtr &antiquant_scale1,
                         const NodePtr &antiquant_scale2, const NodePtr &antiquant_offset1,
                         const NodePtr &antiquant_offset2, const NodePtr &activation, const NodePtr &inner_precise) {
    return Emit("FFNExt",
                {x, weight1, weight2, expertTokens, bias1, bias2, scale, offset, deqScale1, deqScale2, antiquant_scale1,
                 antiquant_scale2, antiquant_offset1, antiquant_offset2, activation, inner_precise});
  }
  virtual NodePtr FillScalar(const NodePtr &size, const NodePtr &fill_value, const NodePtr &dtype) {
    return Emit("FillScalar", {size, fill_value, dtype});
  }
  virtual NodePtr FillTensor(const NodePtr &size, const NodePtr &fill_value, const NodePtr &dtype) {
    return Emit("FillTensor", {size, fill_value, dtype});
  }
  virtual NodePtr FlashAttentionScoreGrad(
    const NodePtr &query, const NodePtr &key, const NodePtr &value, const NodePtr &dy, const NodePtr &pse_shift,
    const NodePtr &drop_mask, const NodePtr &padding_mask, const NodePtr &atten_mask, const NodePtr &softmax_max,
    const NodePtr &softmax_sum, const NodePtr &softmax_in, const NodePtr &attention_in, const NodePtr &prefix,
    const NodePtr &actual_seq_qlen, const NodePtr &actual_seq_kvlen, const NodePtr &head_num, const NodePtr &keep_prob,
    const NodePtr &scale_value, const NodePtr &pre_tokens, const NodePtr &next_tokens, const NodePtr &inner_precise,
    const NodePtr &input_layout, const NodePtr &sparse_mode) {
    return Emit(
      "FlashAttentionScoreGrad",
      {query,       key,         value,      dy,           pse_shift,     drop_mask,       padding_mask,     atten_mask,
       softmax_max, softmax_sum, softmax_in, attention_in, prefix,        actual_seq_qlen, actual_seq_kvlen, head_num,
       keep_prob,   scale_value, pre_tokens, next_tokens,  inner_precise, input_layout,    sparse_mode});
  }
  virtual NodePtr FlashAttentionScore(const NodePtr &query, const NodePtr &key, const NodePtr &value,
                                      const NodePtr &real_shift, const NodePtr &drop_mask, const NodePtr &padding_mask,
                                      const NodePtr &attn_mask, const NodePtr &prefix, const NodePtr &actual_seq_qlen,
                                      const NodePtr &actual_seq_kvlen, const NodePtr &head_num,
                                      const NodePtr &keep_prob, const NodePtr &scale_value, const NodePtr &pre_tokens,
                                      const NodePtr &next_tokens, const NodePtr &inner_precise,
                                      const NodePtr &input_layout, const NodePtr &sparse_mode) {
    return Emit("FlashAttentionScore", {query, key, value, real_shift, drop_mask, padding_mask, attn_mask, prefix,
                                        actual_seq_qlen, actual_seq_kvlen, head_num, keep_prob, scale_value, pre_tokens,
                                        next_tokens, inner_precise, input_layout, sparse_mode});
  }
  virtual NodePtr FlattenExt(const NodePtr &input, const NodePtr &start_dim, const NodePtr &end_dim) {
    return Emit("FlattenExt", {input, start_dim, end_dim});
  }
  virtual NodePtr Floor(const NodePtr &input) { return Emit("Floor", {input}); }
  virtual NodePtr GatherDGradV2(const NodePtr &x, const NodePtr &dim, const NodePtr &index, const NodePtr &dout) {
    return Emit("GatherDGradV2", {x, dim, index, dout});
  }
  virtual NodePtr GatherD(const NodePtr &x, const NodePtr &dim, const NodePtr &index) {
    return Emit("GatherD", {x, dim, index});
  }
  virtual NodePtr GeLUGrad(const NodePtr &dy, const NodePtr &x, const NodePtr &y) {
    return Emit("GeLUGrad", {dy, x, y});
  }
  virtual NodePtr GeLU(const NodePtr &input) { return Emit("GeLU", {input}); }
  virtual NodePtr Generator(const NodePtr &cmd, const NodePtr &inputs) { return Emit("Generator", {cmd, inputs}); }
  virtual NodePtr GridSampler2DGrad(const NodePtr &grad, const NodePtr &input_x, const NodePtr &grid,
                                    const NodePtr &interpolation_mode, const NodePtr &padding_mode,
                                    const NodePtr &align_corners) {
    return Emit("GridSampler2DGrad", {grad, input_x, grid, interpolation_mode, padding_mode, align_corners});
  }
  virtual NodePtr GridSampler2D(const NodePtr &input_x, const NodePtr &grid, const NodePtr &interpolation_mode,
                                const NodePtr &padding_mode, const NodePtr &align_corners) {
    return Emit("GridSampler2D", {input_x, grid, interpolation_mode, padding_mode, align_corners});
  }
  virtual NodePtr GridSampler3DGrad(const NodePtr &grad, const NodePtr &input_x, const NodePtr &grid,
                                    const NodePtr &interpolation_mode, const NodePtr &padding_mode,
                                    const NodePtr &align_corners) {
    return Emit("GridSampler3DGrad", {grad, input_x, grid, interpolation_mode, padding_mode, align_corners});
  }
  virtual NodePtr GridSampler3D(const NodePtr &input_x, const NodePtr &grid, const NodePtr &interpolation_mode,
                                const NodePtr &padding_mode, const NodePtr &align_corners) {
    return Emit("GridSampler3D", {input_x, grid, interpolation_mode, padding_mode, align_corners});
  }
  virtual NodePtr GroupNormGrad(const NodePtr &dy, const NodePtr &x, const NodePtr &mean, const NodePtr &rstd,
                                const NodePtr &gamma_opt, const NodePtr &num_groups, const NodePtr &dx_is_require,
                                const NodePtr &dgamma_is_require, const NodePtr &dbeta_is_require) {
    return Emit("GroupNormGrad",
                {dy, x, mean, rstd, gamma_opt, num_groups, dx_is_require, dgamma_is_require, dbeta_is_require});
  }
  virtual NodePtr GroupNorm(const NodePtr &input, const NodePtr &num_groups, const NodePtr &weight, const NodePtr &bias,
                            const NodePtr &eps) {
    return Emit("GroupNorm", {input, num_groups, weight, bias, eps});
  }
  virtual NodePtr Im2ColExt(const NodePtr &input, const NodePtr &kernel_size, const NodePtr &dilation,
                            const NodePtr &padding, const NodePtr &stride) {
    return Emit("Im2ColExt", {input, kernel_size, dilation, padding, stride});
  }
  virtual NodePtr IndexAddExt(const NodePtr &input, const NodePtr &index, const NodePtr &source, const NodePtr &axis,
                              const NodePtr &alpha) {
    return Emit("IndexAddExt", {input, index, source, axis, alpha});
  }
  virtual NodePtr IndexSelect(const NodePtr &input, const NodePtr &dim, const NodePtr &index) {
    return Emit("IndexSelect", {input, dim, index});
  }
  virtual NodePtr IsClose(const NodePtr &input, const NodePtr &other, const NodePtr &rtol, const NodePtr &atol,
                          const NodePtr &equal_nan) {
    return Emit("IsClose", {input, other, rtol, atol, equal_nan});
  }
  virtual NodePtr IsFinite(const NodePtr &x) { return Emit("IsFinite", {x}); }
  virtual NodePtr LayerNormExt(const NodePtr &input, const NodePtr &normalized_shape, const NodePtr &weight,
                               const NodePtr &bias, const NodePtr &eps) {
    return Emit("LayerNormExt", {input, normalized_shape, weight, bias, eps});
  }
  virtual NodePtr LayerNormGradExt(const NodePtr &dy, const NodePtr &x, const NodePtr &normalized_shape,
                                   const NodePtr &mean, const NodePtr &variance, const NodePtr &gamma,
                                   const NodePtr &beta) {
    return Emit("LayerNormGradExt", {dy, x, normalized_shape, mean, variance, gamma, beta});
  }
  virtual NodePtr LeakyReLUExt(const NodePtr &input, const NodePtr &negative_slope) {
    return Emit("LeakyReLUExt", {input, negative_slope});
  }
  virtual NodePtr LeakyReLUGradExt(const NodePtr &dy, const NodePtr &input, const NodePtr &negative_slope,
                                   const NodePtr &is_result) {
    return Emit("LeakyReLUGradExt", {dy, input, negative_slope, is_result});
  }
  virtual NodePtr LinSpaceExt(const NodePtr &start, const NodePtr &end, const NodePtr &steps, const NodePtr &dtype) {
    return Emit("LinSpaceExt", {start, end, steps, dtype});
  }
  virtual NodePtr LogicalAnd(const NodePtr &x, const NodePtr &y) { return Emit("LogicalAnd", {x, y}); }
  virtual NodePtr LogicalNot(const NodePtr &input) { return Emit("LogicalNot", {input}); }
  virtual NodePtr LogicalOr(const NodePtr &x, const NodePtr &y) { return Emit("LogicalOr", {x, y}); }
  virtual NodePtr MaskedFill(const NodePtr &input_x, const NodePtr &mask, const NodePtr &value) {
    return Emit("MaskedFill", {input_x, mask, value});
  }
  virtual NodePtr MatrixInverseExt(const NodePtr &input) { return Emit("MatrixInverseExt", {input}); }
  virtual NodePtr Max(const NodePtr &input) { return Emit("Max", {input}); }
  virtual NodePtr MaxPoolGradWithIndices(const NodePtr &x, const NodePtr &grad, const NodePtr &argmax,
                                         const NodePtr &kernel_size, const NodePtr &strides, const NodePtr &pads,
                                         const NodePtr &dilation, const NodePtr &ceil_mode,
                                         const NodePtr &argmax_type) {
    return Emit("MaxPoolGradWithIndices",
                {x, grad, argmax, kernel_size, strides, pads, dilation, ceil_mode, argmax_type});
  }
  virtual NodePtr MaxPoolGradWithMask(const NodePtr &x, const NodePtr &grad, const NodePtr &mask,
                                      const NodePtr &kernel_size, const NodePtr &strides, const NodePtr &pads,
                                      const NodePtr &dilation, const NodePtr &ceil_mode, const NodePtr &argmax_type) {
    return Emit("MaxPoolGradWithMask", {x, grad, mask, kernel_size, strides, pads, dilation, ceil_mode, argmax_type});
  }
  virtual NodePtr MaxPoolWithIndices(const NodePtr &x, const NodePtr &kernel_size, const NodePtr &strides,
                                     const NodePtr &pads, const NodePtr &dilation, const NodePtr &ceil_mode,
                                     const NodePtr &argmax_type) {
    return Emit("MaxPoolWithIndices", {x, kernel_size, strides, pads, dilation, ceil_mode, argmax_type});
  }
  virtual NodePtr MaxPoolWithMask(const NodePtr &x, const NodePtr &kernel_size, const NodePtr &strides,
                                  const NodePtr &pads, const NodePtr &dilation, const NodePtr &ceil_mode,
                                  const NodePtr &argmax_type) {
    return Emit("MaxPoolWithMask", {x, kernel_size, strides, pads, dilation, ceil_mode, argmax_type});
  }
  virtual NodePtr MeanExt(const NodePtr &input, const NodePtr &axis, const NodePtr &keep_dims, const NodePtr &dtype) {
    return Emit("MeanExt", {input, axis, keep_dims, dtype});
  }
  virtual NodePtr Min(const NodePtr &input) { return Emit("Min", {input}); }
  virtual NodePtr Mv(const NodePtr &input, const NodePtr &vec) { return Emit("Mv", {input, vec}); }
  virtual NodePtr Neg(const NodePtr &input) { return Emit("Neg", {input}); }
  virtual NodePtr NonZeroExt(const NodePtr &input) { return Emit("NonZeroExt", {input}); }
  virtual NodePtr NonZero(const NodePtr &input) { return Emit("NonZero", {input}); }
  virtual NodePtr Norm(const NodePtr &input_x, const NodePtr &ord, const NodePtr &dim, const NodePtr &keepdim,
                       const NodePtr &dtype) {
    return Emit("Norm", {input_x, ord, dim, keepdim, dtype});
  }
  virtual NodePtr NormalFloatFloat(const NodePtr &mean, const NodePtr &std, const NodePtr &size, const NodePtr &seed,
                                   const NodePtr &offset) {
    return Emit("NormalFloatFloat", {mean, std, size, seed, offset});
  }
  virtual NodePtr NormalFloatTensor(const NodePtr &mean, const NodePtr &std, const NodePtr &seed,
                                    const NodePtr &offset) {
    return Emit("NormalFloatTensor", {mean, std, seed, offset});
  }
  virtual NodePtr NormalTensorFloat(const NodePtr &mean, const NodePtr &std, const NodePtr &seed,
                                    const NodePtr &offset) {
    return Emit("NormalTensorFloat", {mean, std, seed, offset});
  }
  virtual NodePtr NormalTensorTensor(const NodePtr &mean, const NodePtr &std, const NodePtr &seed,
                                     const NodePtr &offset) {
    return Emit("NormalTensorTensor", {mean, std, seed, offset});
  }
  virtual NodePtr OneHotExt(const NodePtr &tensor, const NodePtr &num_classes, const NodePtr &on_value,
                            const NodePtr &off_value, const NodePtr &axis) {
    return Emit("OneHotExt", {tensor, num_classes, on_value, off_value, axis});
  }
  virtual NodePtr OnesLikeExt(const NodePtr &input, const NodePtr &dtype) {
    return Emit("OnesLikeExt", {input, dtype});
  }
  virtual NodePtr Ones(const NodePtr &shape, const NodePtr &dtype) { return Emit("Ones", {shape, dtype}); }
  virtual NodePtr ProdExt(const NodePtr &input, const NodePtr &axis, const NodePtr &keep_dims, const NodePtr &dtype) {
    return Emit("ProdExt", {input, axis, keep_dims, dtype});
  }
  virtual NodePtr RandExt(const NodePtr &shape, const NodePtr &seed, const NodePtr &offset, const NodePtr &dtype) {
    return Emit("RandExt", {shape, seed, offset, dtype});
  }
  virtual NodePtr RandLikeExt(const NodePtr &tensor, const NodePtr &seed, const NodePtr &offset, const NodePtr &dtype) {
    return Emit("RandLikeExt", {tensor, seed, offset, dtype});
  }
  virtual NodePtr Reciprocal(const NodePtr &x) { return Emit("Reciprocal", {x}); }
  virtual NodePtr ReduceAll(const NodePtr &input, const NodePtr &axis, const NodePtr &keep_dims) {
    return Emit("ReduceAll", {input, axis, keep_dims});
  }
  virtual NodePtr ReduceAny(const NodePtr &x, const NodePtr &axis, const NodePtr &keep_dims) {
    return Emit("ReduceAny", {x, axis, keep_dims});
  }
  virtual NodePtr ReflectionPad1DGrad(const NodePtr &grad_output, const NodePtr &input, const NodePtr &padding) {
    return Emit("ReflectionPad1DGrad", {grad_output, input, padding});
  }
  virtual NodePtr ReflectionPad1D(const NodePtr &input, const NodePtr &padding) {
    return Emit("ReflectionPad1D", {input, padding});
  }
  virtual NodePtr ReflectionPad2DGrad(const NodePtr &grad_output, const NodePtr &input, const NodePtr &padding) {
    return Emit("ReflectionPad2DGrad", {grad_output, input, padding});
  }
  virtual NodePtr ReflectionPad2D(const NodePtr &input, const NodePtr &padding) {
    return Emit("ReflectionPad2D", {input, padding});
  }
  virtual NodePtr ReflectionPad3DGrad(const NodePtr &grad_output, const NodePtr &input, const NodePtr &padding) {
    return Emit("ReflectionPad3DGrad", {grad_output, input, padding});
  }
  virtual NodePtr ReflectionPad3D(const NodePtr &input, const NodePtr &padding) {
    return Emit("ReflectionPad3D", {input, padding});
  }
  virtual NodePtr ReluGrad(const NodePtr &y_backprop, const NodePtr &x) { return Emit("ReluGrad", {y_backprop, x}); }
  virtual NodePtr ReLU(const NodePtr &input) { return Emit("ReLU", {input}); }
  virtual NodePtr RepeatInterleaveGrad(const NodePtr &input, const NodePtr &repeats, const NodePtr &dim) {
    return Emit("RepeatInterleaveGrad", {input, repeats, dim});
  }
  virtual NodePtr RepeatInterleaveInt(const NodePtr &input, const NodePtr &repeats, const NodePtr &dim,
                                      const NodePtr &output_size) {
    return Emit("RepeatInterleaveInt", {input, repeats, dim, output_size});
  }
  virtual NodePtr RepeatInterleaveTensor(const NodePtr &input, const NodePtr &repeats, const NodePtr &dim,
                                         const NodePtr &output_size) {
    return Emit("RepeatInterleaveTensor", {input, repeats, dim, output_size});
  }
  virtual NodePtr ReplicationPad1DGrad(const NodePtr &grad_output, const NodePtr &input, const NodePtr &padding) {
    return Emit("ReplicationPad1DGrad", {grad_output, input, padding});
  }
  virtual NodePtr ReplicationPad1D(const NodePtr &input, const NodePtr &padding) {
    return Emit("ReplicationPad1D", {input, padding});
  }
  virtual NodePtr ReplicationPad2DGrad(const NodePtr &grad_output, const NodePtr &input, const NodePtr &padding) {
    return Emit("ReplicationPad2DGrad", {grad_output, input, padding});
  }
  virtual NodePtr ReplicationPad2D(const NodePtr &input, const NodePtr &padding) {
    return Emit("ReplicationPad2D", {input, padding});
  }
  virtual NodePtr ReplicationPad3DGrad(const NodePtr &grad_output, const NodePtr &input, const NodePtr &padding) {
    return Emit("ReplicationPad3DGrad", {grad_output, input, padding});
  }
  virtual NodePtr ReplicationPad3D(const NodePtr &input, const NodePtr &padding) {
    return Emit("ReplicationPad3D", {input, padding});
  }
  virtual NodePtr ReverseV2(const NodePtr &input, const NodePtr &axis) { return Emit("ReverseV2", {input, axis}); }
  virtual NodePtr RmsNormGrad(const NodePtr &dy, const NodePtr &x, const NodePtr &rstd, const NodePtr &gamma) {
    return Emit("RmsNormGrad", {dy, x, rstd, gamma});
  }
  virtual NodePtr RmsNorm(const NodePtr &x, const NodePtr &gamma, const NodePtr &epsilon) {
    return Emit("RmsNorm", {x, gamma, epsilon});
  }
  virtual NodePtr Rsqrt(const NodePtr &input) { return Emit("Rsqrt", {input}); }
  virtual NodePtr ScatterAddExt(const NodePtr &input, const NodePtr &dim, const NodePtr &index, const NodePtr &src) {
    return Emit("ScatterAddExt", {input, dim, index, src});
  }
  virtual NodePtr Scatter(const NodePtr &input, const NodePtr &dim, const NodePtr &index, const NodePtr &src,
                          const NodePtr &reduce) {
    return Emit("Scatter", {input, dim, index, src, reduce});
  }
  virtual NodePtr SearchSorted(const NodePtr &sorted_sequence, const NodePtr &values, const NodePtr &sorter,
                               const NodePtr &dtype, const NodePtr &right) {
    return Emit("SearchSorted", {sorted_sequence, values, sorter, dtype, right});
  }
  virtual NodePtr SigmoidGrad(const NodePtr &y, const NodePtr &dy) { return Emit("SigmoidGrad", {y, dy}); }
  virtual NodePtr Sigmoid(const NodePtr &input) { return Emit("Sigmoid", {input}); }
  virtual NodePtr Sign(const NodePtr &input) { return Emit("Sign", {input}); }
  virtual NodePtr SiLUGrad(const NodePtr &dout, const NodePtr &x) { return Emit("SiLUGrad", {dout, x}); }
  virtual NodePtr SiLU(const NodePtr &input) { return Emit("SiLU", {input}); }
  virtual NodePtr Sin(const NodePtr &input) { return Emit("Sin", {input}); }
  virtual NodePtr SliceExt(const NodePtr &input, const NodePtr &dim, const NodePtr &start, const NodePtr &end,
                           const NodePtr &step) {
    return Emit("SliceExt", {input, dim, start, end, step});
  }
  virtual NodePtr SoftmaxBackward(const NodePtr &dout, const NodePtr &out, const NodePtr &dim) {
    return Emit("SoftmaxBackward", {dout, out, dim});
  }
  virtual NodePtr Softmax(const NodePtr &input, const NodePtr &axis) { return Emit("Softmax", {input, axis}); }
  virtual NodePtr SoftplusExt(const NodePtr &input, const NodePtr &beta, const NodePtr &threshold) {
    return Emit("SoftplusExt", {input, beta, threshold});
  }
  virtual NodePtr SoftplusGradExt(const NodePtr &dout, const NodePtr &x, const NodePtr &beta,
                                  const NodePtr &threshold) {
    return Emit("SoftplusGradExt", {dout, x, beta, threshold});
  }
  virtual NodePtr SortExt(const NodePtr &input, const NodePtr &dim, const NodePtr &descending, const NodePtr &stable) {
    return Emit("SortExt", {input, dim, descending, stable});
  }
  virtual NodePtr SplitTensor(const NodePtr &input_x, const NodePtr &split_int, const NodePtr &axis) {
    return Emit("SplitTensor", {input_x, split_int, axis});
  }
  virtual NodePtr SplitWithSize(const NodePtr &input_x, const NodePtr &split_sections, const NodePtr &axis) {
    return Emit("SplitWithSize", {input_x, split_sections, axis});
  }
  virtual NodePtr Sqrt(const NodePtr &x) { return Emit("Sqrt", {x}); }
  virtual NodePtr Square(const NodePtr &input) { return Emit("Square", {input}); }
  virtual NodePtr StackExt(const NodePtr &tensors, const NodePtr &dim) { return Emit("StackExt", {tensors, dim}); }
  virtual NodePtr SubExt(const NodePtr &input, const NodePtr &other, const NodePtr &alpha) {
    return Emit("SubExt", {input, other, alpha});
  }
  virtual NodePtr SumExt(const NodePtr &input, const NodePtr &dim, const NodePtr &keepdim, const NodePtr &dtype) {
    return Emit("SumExt", {input, dim, keepdim, dtype});
  }
  virtual NodePtr TanhGrad(const NodePtr &y, const NodePtr &dy) { return Emit("TanhGrad", {y, dy}); }
  virtual NodePtr Tanh(const NodePtr &input) { return Emit("Tanh", {input}); }
  virtual NodePtr TopkExt(const NodePtr &input, const NodePtr &k, const NodePtr &dim, const NodePtr &largest,
                          const NodePtr &sorted) {
    return Emit("TopkExt", {input, k, dim, largest, sorted});
  }
  virtual NodePtr Triu(const NodePtr &input, const NodePtr &diagonal) { return Emit("Triu", {input, diagonal}); }
  virtual NodePtr UniformExt(const NodePtr &tensor, const NodePtr &a, const NodePtr &b, const NodePtr &seed,
                             const NodePtr &offset) {
    return Emit("UniformExt", {tensor, a, b, seed, offset});
  }
  virtual NodePtr Unique2(const NodePtr &input, const NodePtr &sorted, const NodePtr &return_inverse,
                          const NodePtr &return_counts) {
    return Emit("Unique2", {input, sorted, return_inverse, return_counts});
  }
  virtual NodePtr UniqueDim(const NodePtr &input, const NodePtr &sorted, const NodePtr &return_inverse,
                            const NodePtr &dim) {
    return Emit("UniqueDim", {input, sorted, return_inverse, dim});
  }
  virtual NodePtr UpsampleBilinear2DGrad(const NodePtr &dy, const NodePtr &input_size, const NodePtr &output_size,
                                         const NodePtr &scales, const NodePtr &align_corners) {
    return Emit("UpsampleBilinear2DGrad", {dy, input_size, output_size, scales, align_corners});
  }
  virtual NodePtr UpsampleBilinear2D(const NodePtr &x, const NodePtr &output_size, const NodePtr &scales,
                                     const NodePtr &align_corners) {
    return Emit("UpsampleBilinear2D", {x, output_size, scales, align_corners});
  }
  virtual NodePtr UpsampleLinear1DGrad(const NodePtr &dy, const NodePtr &input_size, const NodePtr &output_size,
                                       const NodePtr &scales, const NodePtr &align_corners) {
    return Emit("UpsampleLinear1DGrad", {dy, input_size, output_size, scales, align_corners});
  }
  virtual NodePtr UpsampleLinear1D(const NodePtr &x, const NodePtr &output_size, const NodePtr &scales,
                                   const NodePtr &align_corners) {
    return Emit("UpsampleLinear1D", {x, output_size, scales, align_corners});
  }
  virtual NodePtr UpsampleNearest1DGrad(const NodePtr &dy, const NodePtr &input_size, const NodePtr &output_size,
                                        const NodePtr &scales) {
    return Emit("UpsampleNearest1DGrad", {dy, input_size, output_size, scales});
  }
  virtual NodePtr UpsampleNearest1D(const NodePtr &x, const NodePtr &output_size, const NodePtr &scales) {
    return Emit("UpsampleNearest1D", {x, output_size, scales});
  }
  virtual NodePtr UpsampleNearest2DGrad(const NodePtr &dy, const NodePtr &input_size, const NodePtr &output_size,
                                        const NodePtr &scales) {
    return Emit("UpsampleNearest2DGrad", {dy, input_size, output_size, scales});
  }
  virtual NodePtr UpsampleNearest2D(const NodePtr &x, const NodePtr &output_size, const NodePtr &scales) {
    return Emit("UpsampleNearest2D", {x, output_size, scales});
  }
  virtual NodePtr UpsampleNearest3DGrad(const NodePtr &dy, const NodePtr &input_size, const NodePtr &output_size,
                                        const NodePtr &scales) {
    return Emit("UpsampleNearest3DGrad", {dy, input_size, output_size, scales});
  }
  virtual NodePtr UpsampleNearest3D(const NodePtr &x, const NodePtr &output_size, const NodePtr &scales) {
    return Emit("UpsampleNearest3D", {x, output_size, scales});
  }
  virtual NodePtr UpsampleTrilinear3DGrad(const NodePtr &dy, const NodePtr &input_size, const NodePtr &output_size,
                                          const NodePtr &scales, const NodePtr &align_corners) {
    return Emit("UpsampleTrilinear3DGrad", {dy, input_size, output_size, scales, align_corners});
  }
  virtual NodePtr UpsampleTrilinear3D(const NodePtr &x, const NodePtr &output_size, const NodePtr &scales,
                                      const NodePtr &align_corners) {
    return Emit("UpsampleTrilinear3D", {x, output_size, scales, align_corners});
  }
  virtual NodePtr ZerosLikeExt(const NodePtr &input, const NodePtr &dtype) {
    return Emit("ZerosLikeExt", {input, dtype});
  }
  virtual NodePtr Zeros(const NodePtr &size, const NodePtr &dtype) { return Emit("Zeros", {size, dtype}); }
  virtual NodePtr DynamicQuantExt(const NodePtr &x, const NodePtr &smooth_scales) {
    return Emit("DynamicQuantExt", {x, smooth_scales});
  }
  virtual NodePtr GroupedMatmul(const NodePtr &x, const NodePtr &weight, const NodePtr &bias, const NodePtr &scale,
                                const NodePtr &offset, const NodePtr &antiquant_scale, const NodePtr &antiquant_offset,
                                const NodePtr &group_list, const NodePtr &split_item, const NodePtr &group_type) {
    return Emit("GroupedMatmul", {x, weight, bias, scale, offset, antiquant_scale, antiquant_offset, group_list,
                                  split_item, group_type});
  }
  virtual NodePtr MoeFinalizeRouting(const NodePtr &expanded_x, const NodePtr &x1, const NodePtr &x2,
                                     const NodePtr &bias, const NodePtr &scales, const NodePtr &expanded_row_idx,
                                     const NodePtr &expanded_expert_idx) {
    return Emit("MoeFinalizeRouting", {expanded_x, x1, x2, bias, scales, expanded_row_idx, expanded_expert_idx});
  }
  virtual NodePtr QuantBatchMatmul(const NodePtr &x1, const NodePtr &x2, const NodePtr &scale, const NodePtr &offset,
                                   const NodePtr &bias, const NodePtr &transpose_x1, const NodePtr &transpose_x2,
                                   const NodePtr &dtype) {
    return Emit("QuantBatchMatmul", {x1, x2, scale, offset, bias, transpose_x1, transpose_x2, dtype});
  }
  virtual NodePtr QuantV2(const NodePtr &x, const NodePtr &scale, const NodePtr &offset, const NodePtr &sqrt_mode,
                          const NodePtr &rounding_mode, const NodePtr &dst_type) {
    return Emit("QuantV2", {x, scale, offset, sqrt_mode, rounding_mode, dst_type});
  }
  virtual NodePtr WeightQuantBatchMatmul(const NodePtr &x, const NodePtr &weight, const NodePtr &antiquant_scale,
                                         const NodePtr &antiquant_offset, const NodePtr &quant_scale,
                                         const NodePtr &quant_offset, const NodePtr &bias, const NodePtr &transpose_x,
                                         const NodePtr &transpose_weight, const NodePtr &antiquant_group_size) {
    return Emit("WeightQuantBatchMatmul", {x, weight, antiquant_scale, antiquant_offset, quant_scale, quant_offset,
                                           bias, transpose_x, transpose_weight, antiquant_group_size});
  }

 protected:
  virtual NodePtr EmitOp(const PrimitivePtr &prim, const NodePtrList &inputs);
  NodePtr CmpOpWithCast(const std::string &op, const NodePtr &lhs, const NodePtr &rhs, const TypePtr &dst_type) {
    auto node = UnifyDtypeAndEmit(op, lhs, rhs);
    return dst_type == nullptr ? node : Cast(node, dst_type);
  }
  std::tuple<NodePtr, NodePtr> UnifyDtype2(const NodePtr &lhs, const NodePtr &rhs);
  NodePtr UnifyDtypeAndEmit(const std::string &op, const NodePtr &a, const NodePtr &b, const DAttr &attrs = {}) {
    auto [lhs, rhs] = UnifyDtype2(a, b);
    return Emit(op, {lhs, rhs}, attrs);
  }

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

class COMMON_EXPORT CtrlFlowBlock {
 public:
  using BlockFunc = std::function<NodePtrList(Emitter *)>;
  using EmitterCreator = std::function<EmitterPtr(const FuncGraphPtr &, const ExpanderInferPtr &)>;
  CtrlFlowBlock(Emitter *emitter, const FuncGraphPtr &func_graph, const EmitterCreator &ec = nullptr)
      : emitter_(emitter), func_graph_(func_graph), emitter_creator_(ec) {
    MS_EXCEPTION_IF_NULL(emitter);
    MS_EXCEPTION_IF_NULL(func_graph);
  }
  ~CtrlFlowBlock() = default;
  NodePtr IfThenElse(const NodePtr &cond, const BlockFunc &true_case, const BlockFunc &false_case);

  NodePtr While(const NodePtr &cond, const BlockFunc &while_body_func, const NodePtrList &init_list);

 protected:
  EmitterPtr CreateInnerEmitter(const FuncGraphPtr &fg, const ExpanderInferPtr &infer) const;
  NodePtr BuildSubgraph(const BlockFunc &func);
  NodePtrList BuildSubgraphOfPartial(const BlockFunc &func);

  Emitter *emitter_;
  FuncGraphPtr func_graph_;
  EmitterCreator emitter_creator_;
  size_t output_num_{0};
  abstract::AbstractBasePtr out_abstract_{nullptr};

  class CppInferWithPartial : public CppInfer {
   public:
    void Infer(const NodePtr &node) override;
  };
};

class COMMON_EXPORT IrEmitter : public Emitter {
 public:
  IrEmitter(const FuncGraphPtr &func_graph, const ExpanderInferPtr &infer, const ScopePtr &scope = nullptr)
      : Emitter(infer, scope), func_graph_(func_graph) {
    MS_EXCEPTION_IF_NULL(func_graph);
    MS_EXCEPTION_IF_NULL(infer);
  }
  NodePtr EmitValue(const ValuePtr &value) override;
  FuncGraphPtr func_graph() { return func_graph_; }

 protected:
  NodePtr EmitOp(const PrimitivePtr &prim, const NodePtrList &inputs) override;
  FuncGraphPtr func_graph_;
};

class PureShapeCalc : public ShapeCalcBaseFunctor {
 public:
  // CalcFunc/InferFunc/CalcWithTupleFunc/InferWithTupleFunc are defined as pure function pointer other than a
  // std::function, meaning that they should be a lambda function without any capture.
  using CalcFunc = ShapeArray (*)(const ShapeArray &);
  using InferFunc = std::vector<int64_t> (*)(const ShapeArray &, const HashSet<size_t> &);
  using CalcWithTupleFunc = ShapeArray (*)(const ShapeArray &, const ElemPosIdx &);
  using InferWithTupleFunc = InferOutputInfo (*)(const ShapeArray &, const HashSet<size_t> &, const ElemPosIdx &);

  explicit PureShapeCalc(const std::string &name) : ShapeCalcBaseFunctor(name) {
    FunctorRegistry::Instance().Register(name, [this]() { return shared_from_base<Functor>(); });
  }

  PureShapeCalc(const PureShapeCalc &) = delete;
  PureShapeCalc(PureShapeCalc &&) = delete;
  PureShapeCalc &operator=(const PureShapeCalc &) = delete;
  PureShapeCalc &operator=(PureShapeCalc &&) = delete;
  ~PureShapeCalc() override = default;
  MS_DECLARE_PARENT(PureShapeCalc, ShapeCalcBaseFunctor)

  ValuePtr ToValue() const override { return nullptr; }
  void FromValue(const ValuePtr &) override {}

  ShapeArray Calc(const ShapeArray &inputs, const ElemPosIdx &pos_idx) const override {
    ShapeArray calc_res;
    if (calc_func_ != nullptr) {
      calc_res = calc_func_(inputs);
    } else if (cal_with_tuple_func_ != nullptr) {
      calc_res = cal_with_tuple_func_(inputs, pos_idx);
    } else {
      MS_LOG(EXCEPTION) << "The calc_func of " << name() << " is nullptr";
    }

    return calc_res;
  }

  InferOutputInfo Infer(const ShapeArray &inputs, const HashSet<size_t> &unknown_inputs,
                        const ElemPosIdx &pos_idx) const override {
    InferOutputInfo infer_res;
    if (infer_func_ != nullptr) {
      auto output_shapes = infer_func_(inputs, unknown_inputs);
      infer_res = std::make_pair(output_shapes, false);
    } else if (infer_with_tuple_func_ != nullptr) {
      infer_res = infer_with_tuple_func_(inputs, unknown_inputs, pos_idx);
    } else {
      MS_LOG(EXCEPTION) << "The infer_func of " << name() << " is nullptr";
    }

    return infer_res;
  }

  PureShapeCalc &SetCalc(const CalcFunc &calc_func) {
    calc_func_ = calc_func;
    return *this;
  }

  std::shared_ptr<PureShapeCalc> SetInfer(const InferFunc &infer_func) {
    infer_func_ = infer_func;
    if (calc_func_ == nullptr || cal_with_tuple_func_ != nullptr) {
      MS_LOG(EXCEPTION) << "The Calc Function and Infer Function should all not support tuple!";
    }
    return shared_from_base<PureShapeCalc>();
  }

  PureShapeCalc &SetCalc(const CalcWithTupleFunc &calc_func) {
    cal_with_tuple_func_ = calc_func;
    return *this;
  }

  std::shared_ptr<PureShapeCalc> SetInfer(const InferWithTupleFunc &infer_func) {
    infer_with_tuple_func_ = infer_func;
    if (cal_with_tuple_func_ == nullptr || calc_func_ != nullptr) {
      MS_LOG(EXCEPTION) << "The Calc Function and Infer Function should all support tuple!";
    }
    return shared_from_base<PureShapeCalc>();
  }

 private:
  CalcFunc calc_func_{nullptr};
  InferFunc infer_func_{nullptr};
  CalcWithTupleFunc cal_with_tuple_func_{nullptr};
  InferWithTupleFunc infer_with_tuple_func_{nullptr};
};

#define DEF_PURE_SHAPE_CALC(name) \
  static const std::shared_ptr<PureShapeCalc> name = (*(std::make_shared<PureShapeCalc>("ShapeCalc_" #name)))

}  // namespace expander
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_COMMON_EXPANDER_CORE_EMITTER_H_
