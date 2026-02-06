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

#ifndef MINDSPORE_CCSRC_UTILS_EXPANDER_EMITTER_H_
#define MINDSPORE_CCSRC_UTILS_EXPANDER_EMITTER_H_
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_set>
#include <unordered_map>
#include <functional>
#include <utility>
#include <vector>
#include <complex>
#include "mindspore/ccsrc/include/utils/expander/infer.h"
#include "mindspore/ccsrc/include/utils/expander/node.h"
#include "ir/func_graph.h"
#include "ir/functor.h"
#include "ir/tensor_new.h"
#include "primitive/array_op_name.h"
#include "primitive/comparison_op_name.h"
#include "primitive/framework_op_name.h"
#include "primitive/arithmetic_op_name.h"
#include "primitive/math_ops.h"
#include "primitive/sequence_ops.h"
#include "infer/shape_calc.h"

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
  NodePtr ScalarAdd(const NodePtr &lhs, const NodePtr &rhs);
  NodePtr ScalarSub(const NodePtr &lhs, const NodePtr &rhs);
  NodePtr ScalarMul(const NodePtr &lhs, const NodePtr &rhs);
  NodePtr ScalarDiv(const NodePtr &lhs, const NodePtr &rhs);
  NodePtr ScalarFloorDiv(const NodePtr &lhs, const NodePtr &rhs);
  NodePtr ScalarNeg(const NodePtr &node);
  virtual NodePtr Cast(const NodePtr &node, const TypePtr &type);
  NodePtr Cast(const NodePtr &node, TypeId type_id) { return Cast(node, TypeIdToType(type_id)); }

  virtual NodePtr Reshape(const NodePtr &node, const NodePtr &shape);
  NodePtr Reshape(const NodePtr &node, const ShapeVector &shape) { return Reshape(node, Value(shape)); }
  NodePtr ExpandDims(const NodePtr &node, int64_t axis) { return Emit(kExpandDimsOpName, {node, Value(axis)}); }
  virtual NodePtr Exp(const NodePtr &x);
  NodePtr Log(const NodePtr &x);
  virtual NodePtr Transpose(const NodePtr &node, const NodePtr &perm);
  virtual NodePtr Transpose(const NodePtr &node, int64_t dim0, int64_t dim1);
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
  NodePtr DivMod(const NodePtr &lhs, const NodePtr &rhs, int64_t rounding_mode) {
    auto [a, b] = UnifyDtype(lhs, rhs);
    return Emit(kDivModOpName, {a, b, Value(rounding_mode)});
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
  virtual NodePtr FloorDiv(const NodePtr &lhs, const NodePtr &rhs) { return UnifyDtypeAndEmit("FloorDiv", lhs, rhs); }
  NodePtr FloorMod(const NodePtr &lhs, const NodePtr &rhs) { return UnifyDtypeAndEmit("FloorMod", lhs, rhs); }
  NodePtr DivNoNan(const NodePtr &lhs, const NodePtr &rhs) { return UnifyDtypeAndEmit("DivNoNan", lhs, rhs); }
  NodePtr MulNoNan(const NodePtr &lhs, const NodePtr &rhs) { return UnifyDtypeAndEmit("MulNoNan", lhs, rhs); }
  NodePtr Xdivy(const NodePtr &lhs, const NodePtr &rhs) { return UnifyDtypeAndEmit("Xdivy", lhs, rhs); }
  NodePtr Xlogy(const NodePtr &lhs, const NodePtr &rhs) { return UnifyDtypeAndEmit("Xlogy", lhs, rhs); }
  virtual NodePtr Select(const NodePtr &cond, const NodePtr &lhs, const NodePtr &rhs) {
    auto [a, b] = UnifyDtype(lhs, rhs);
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
  NodePtr Squeeze(const NodePtr &x, const ValuePtr &axis) { return Emit("Squeeze", {x, EmitValue(axis)}); }

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

  virtual NodePtr Shape(const NodePtr &node, bool tensor = false) {
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

  virtual NodePtr InplaceCopy(const NodePtr &variable, const NodePtr &value, bool non_blocking = false) {
    return Emit("InplaceCopy", {variable, value, Value<bool>(non_blocking)},
                {{GRAPH_FLAG_SIDE_EFFECT_MEM, MakeValue(true)}});
  }
  /// \brief Emit a value node
  template <typename T>
  NodePtr Value(const T &value) {
    return EmitValue(MakeValue(value));
  }

  /// \brief Emit a Tensor node.
  template <typename T>
  NodePtr Tensor(T data, TypePtr type_ptr = nullptr) {
    auto tensor_ptr = tensor::from_scalar(data, type_ptr);
    return EmitValue(tensor_ptr);
  }

  /// \brief Emit a Tensor node.
  template <typename T>
  NodePtr Tensor(std::vector<T> data, TypePtr type_ptr = nullptr) {
    auto tensor_ptr = tensor::from_vector(data, type_ptr);
    return EmitValue(tensor_ptr);
  }

  /// \brief Emit a tensor node.
  NodePtr Tensor(TypeId data_type, const ShapeVector &shape, void *data, TypeId src_data_type) {
    auto tensor_ptr = tensor::from_buffer(data_type, shape, data, src_data_type);
    return EmitValue(tensor_ptr);
  }

  virtual void MarkSharedGradTensor(const NodePtr &lhs, const NodePtr &rhs) {}

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

  virtual NodePtr Ones(const NodePtr &shape, const NodePtr &dtype) { return Emit("Ones", {shape, dtype}); }
  virtual NodePtr LerpScalar(const NodePtr &input, const NodePtr &end, const NodePtr &weight) {
    return Emit("LerpScalar", {input, end, weight});
  }
  virtual NodePtr Atanh(const NodePtr &input) { return Emit("Atanh", {input}); }
  virtual NodePtr ClampScalar(const NodePtr &input, const NodePtr &min, const NodePtr &max) {
    return Emit("ClampScalar", {input, min, max});
  }
  virtual NodePtr InplaceRandom(const NodePtr &input, const NodePtr &from_, const NodePtr &to, const NodePtr &seed,
                                const NodePtr &offset) {
    return Emit("InplaceRandom", {input, from_, to, seed, offset});
  }
  virtual NodePtr ClampTensor(const NodePtr &input, const NodePtr &min, const NodePtr &max) {
    return Emit("ClampTensor", {input, min, max});
  }
  virtual NodePtr Kthvalue(const NodePtr &input, const NodePtr &k, const NodePtr &dim, const NodePtr &keepdim) {
    return Emit("Kthvalue", {input, k, dim, keepdim});
  }
  virtual NodePtr CumsumExt(const NodePtr &input, const NodePtr &dim, const NodePtr &dtype) {
    return Emit("CumsumExt", {input, dim, dtype});
  }
  virtual NodePtr SplitTensor(const NodePtr &input, const NodePtr &split_size, const NodePtr &dim) {
    return Emit("SplitTensor", {input, split_size, dim});
  }
  virtual NodePtr InplaceUniform(const NodePtr &input, const NodePtr &from_, const NodePtr &to, const NodePtr &seed,
                                 const NodePtr &offset) {
    return Emit("InplaceUniform", {input, from_, to, seed, offset});
  }
  virtual NodePtr RotaryPositionEmbeddingGrad(const NodePtr &dy, const NodePtr &cos, const NodePtr &sin,
                                              const NodePtr &dx, const NodePtr &mode) {
    return Emit("RotaryPositionEmbeddingGrad", {dy, cos, sin, dx, mode});
  }
  virtual NodePtr KLDiv(const NodePtr &input, const NodePtr &target, const NodePtr &reduction,
                        const NodePtr &log_target) {
    return Emit("KLDiv", {input, target, reduction, log_target});
  }
  virtual NodePtr OnesLikeExt(const NodePtr &input, const NodePtr &dtype) {
    return Emit("OnesLikeExt", {input, dtype});
  }
  virtual NodePtr Embedding(const NodePtr &input, const NodePtr &weight, const NodePtr &padding_idx,
                            const NodePtr &max_norm, const NodePtr &norm_type, const NodePtr &scale_grad_by_freq) {
    return Emit("Embedding", {input, weight, padding_idx, max_norm, norm_type, scale_grad_by_freq});
  }
  virtual NodePtr SoftplusExt(const NodePtr &input, const NodePtr &beta, const NodePtr &threshold) {
    return Emit("SoftplusExt", {input, beta, threshold});
  }
  virtual NodePtr ViewAs(const NodePtr &input, const NodePtr &other) { return Emit("ViewAs", {input, other}); }
  virtual NodePtr Cosh(const NodePtr &input) { return Emit("Cosh", {input}); }
  virtual NodePtr GroupNorm(const NodePtr &input, const NodePtr &num_groups, const NodePtr &weight, const NodePtr &bias,
                            const NodePtr &eps) {
    return Emit("GroupNorm", {input, num_groups, weight, bias, eps});
  }
  virtual NodePtr InnerIndex(const NodePtr &input, const NodePtr &indices) {
    return Emit("InnerIndex", {input, indices});
  }
  virtual NodePtr InplaceIndexPut(const NodePtr &input, const NodePtr &indices, const NodePtr &values,
                                  const NodePtr &accumulate) {
    return Emit("InplaceIndexPut", {input, indices, values, accumulate});
  }
  virtual NodePtr AddRmsNorm(const NodePtr &x1, const NodePtr &x2, const NodePtr &gamma, const NodePtr &epsilon) {
    return Emit("AddRmsNorm", {x1, x2, gamma, epsilon});
  }
  virtual NodePtr ReplicationPad3DGrad(const NodePtr &grad_output, const NodePtr &input, const NodePtr &padding) {
    return Emit("ReplicationPad3DGrad", {grad_output, input, padding});
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
  virtual NodePtr BitwiseNot(const NodePtr &input) { return Emit("BitwiseNot", {input}); }
  virtual NodePtr ConvolutionStr(const NodePtr &input, const NodePtr &weight, const NodePtr &bias,
                                 const NodePtr &stride, const NodePtr &padding, const NodePtr &dilation,
                                 const NodePtr &transposed, const NodePtr &output_padding, const NodePtr &groups) {
    return Emit("ConvolutionStr", {input, weight, bias, stride, padding, dilation, transposed, output_padding, groups});
  }
  virtual NodePtr LogSoftmax(const NodePtr &logits, const NodePtr &axis) { return Emit("LogSoftmax", {logits, axis}); }
  virtual NodePtr RemainderScalarTensor(const NodePtr &input, const NodePtr &other) {
    return Emit("RemainderScalarTensor", {input, other});
  }
  virtual NodePtr Addmm(const NodePtr &input, const NodePtr &mat1, const NodePtr &mat2, const NodePtr &beta,
                        const NodePtr &alpha) {
    return Emit("Addmm", {input, mat1, mat2, beta, alpha});
  }
  virtual NodePtr GridSampler3DGrad(const NodePtr &grad, const NodePtr &input_x, const NodePtr &grid,
                                    const NodePtr &interpolation_mode, const NodePtr &padding_mode,
                                    const NodePtr &align_corners, const NodePtr &output_mask) {
    return Emit("GridSampler3DGrad",
                {grad, input_x, grid, interpolation_mode, padding_mode, align_corners, output_mask});
  }
  virtual NodePtr MoeDistributeDispatch(const NodePtr &x, const NodePtr &expert_ids, const NodePtr &ep_world_size,
                                        const NodePtr &ep_rank_id, const NodePtr &moe_expert_num,
                                        const NodePtr &expert_scales, const NodePtr &scales,
                                        const NodePtr &x_active_mask, const NodePtr &group_ep, const NodePtr &group_tp,
                                        const NodePtr &tp_world_size, const NodePtr &tp_rank_id,
                                        const NodePtr &expert_shard_type, const NodePtr &shared_expert_num,
                                        const NodePtr &shared_expert_rank_num, const NodePtr &quant_mode,
                                        const NodePtr &global_bs, const NodePtr &expert_token_nums_type) {
    return Emit("MoeDistributeDispatch",
                {x, expert_ids, ep_world_size, ep_rank_id, moe_expert_num, expert_scales, scales, x_active_mask,
                 group_ep, group_tp, tp_world_size, tp_rank_id, expert_shard_type, shared_expert_num,
                 shared_expert_rank_num, quant_mode, global_bs, expert_token_nums_type});
  }
  virtual NodePtr Polar(const NodePtr &abs, const NodePtr &angle) { return Emit("Polar", {abs, angle}); }
  virtual NodePtr Sqrt(const NodePtr &x) { return Emit("Sqrt", {x}); }
  virtual NodePtr TraceExt(const NodePtr &input) { return Emit("TraceExt", {input}); }
  virtual NodePtr Unique2(const NodePtr &input, const NodePtr &sorted, const NodePtr &return_inverse,
                          const NodePtr &return_counts) {
    return Emit("Unique2", {input, sorted, return_inverse, return_counts});
  }
  virtual NodePtr LogSigmoidGrad(const NodePtr &dy, const NodePtr &input, const NodePtr &buffer) {
    return Emit("LogSigmoidGrad", {dy, input, buffer});
  }
  virtual NodePtr BatchMatMulExt(const NodePtr &input, const NodePtr &mat2) {
    return Emit("BatchMatMulExt", {input, mat2});
  }
  virtual NodePtr RepeatInterleaveGrad(const NodePtr &input, const NodePtr &repeats, const NodePtr &dim) {
    return Emit("RepeatInterleaveGrad", {input, repeats, dim});
  }
  virtual NodePtr MoeTokenUnpermuteGrad(const NodePtr &permuted_tokens, const NodePtr &unpermuted_tokens_grad,
                                        const NodePtr &sorted_indices, const NodePtr &probs, const NodePtr &padded_mode,
                                        const NodePtr &restore_shape) {
    return Emit("MoeTokenUnpermuteGrad",
                {permuted_tokens, unpermuted_tokens_grad, sorted_indices, probs, padded_mode, restore_shape});
  }
  virtual NodePtr FillTensor(const NodePtr &size, const NodePtr &fill_value, const NodePtr &dtype) {
    return Emit("FillTensor", {size, fill_value, dtype});
  }
  virtual NodePtr AvgPool2DGrad(const NodePtr &grad, const NodePtr &image, const NodePtr &kernel_size,
                                const NodePtr &stride, const NodePtr &padding, const NodePtr &ceil_mode,
                                const NodePtr &count_include_pad, const NodePtr &divisor_override) {
    return Emit("AvgPool2DGrad",
                {grad, image, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override});
  }
  virtual NodePtr BitwiseXorTensor(const NodePtr &input, const NodePtr &other) {
    return Emit("BitwiseXorTensor", {input, other});
  }
  virtual NodePtr ReplicationPad2D(const NodePtr &input, const NodePtr &padding) {
    return Emit("ReplicationPad2D", {input, padding});
  }
  virtual NodePtr SigmoidGrad(const NodePtr &y, const NodePtr &dy) { return Emit("SigmoidGrad", {y, dy}); }
  virtual NodePtr AvgPool3DGradExt(const NodePtr &grad, const NodePtr &input, const NodePtr &kernel_size,
                                   const NodePtr &stride, const NodePtr &padding, const NodePtr &ceil_mode,
                                   const NodePtr &count_include_pad, const NodePtr &divisor_override) {
    return Emit("AvgPool3DGradExt",
                {grad, input, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override});
  }
  virtual NodePtr RandExt(const NodePtr &shape, const NodePtr &seed, const NodePtr &offset, const NodePtr &dtype,
                          const NodePtr &device) {
    return Emit("RandExt", {shape, seed, offset, dtype, device});
  }
  virtual NodePtr GreaterEqualScalar(const NodePtr &input, const NodePtr &other) {
    return Emit("GreaterEqualScalar", {input, other});
  }
  virtual NodePtr HSigmoidGrad(const NodePtr &grads, const NodePtr &input_x) {
    return Emit("HSigmoidGrad", {grads, input_x});
  }
  virtual NodePtr Swiglu(const NodePtr &input, const NodePtr &dim) { return Emit("Swiglu", {input, dim}); }
  virtual NodePtr SplitWithSizeView(const NodePtr &input, const NodePtr &split_size, const NodePtr &dim) {
    return Emit("SplitWithSizeView", {input, split_size, dim});
  }
  virtual NodePtr Squeeze(const NodePtr &input, const NodePtr &axis) { return Emit("Squeeze", {input, axis}); }
  virtual NodePtr UpsampleNearest2D(const NodePtr &x, const NodePtr &output_size, const NodePtr &scales) {
    return Emit("UpsampleNearest2D", {x, output_size, scales});
  }
  virtual NodePtr Sin(const NodePtr &input) { return Emit("Sin", {input}); }
  virtual NodePtr TopkExt(const NodePtr &input, const NodePtr &k, const NodePtr &dim, const NodePtr &largest,
                          const NodePtr &sorted) {
    return Emit("TopkExt", {input, k, dim, largest, sorted});
  }
  virtual NodePtr BinaryCrossEntropyGrad(const NodePtr &input, const NodePtr &target, const NodePtr &grad_output,
                                         const NodePtr &weight, const NodePtr &reduction) {
    return Emit("BinaryCrossEntropyGrad", {input, target, grad_output, weight, reduction});
  }
  virtual NodePtr SwigluGrad(const NodePtr &grad_output, const NodePtr &input, const NodePtr &dim) {
    return Emit("SwigluGrad", {grad_output, input, dim});
  }
  virtual NodePtr InplaceScatterValue(const NodePtr &input, const NodePtr &dim, const NodePtr &index,
                                      const NodePtr &value) {
    return Emit("InplaceScatterValue", {input, dim, index, value});
  }
  virtual NodePtr InplaceReLU(const NodePtr &input) { return Emit("InplaceReLU", {input}); }
  virtual NodePtr SiLU(const NodePtr &input) { return Emit("SiLU", {input}); }
  virtual NodePtr AddLayerNormGrad(const NodePtr &dy, const NodePtr &x1, const NodePtr &x2, const NodePtr &rstd,
                                   const NodePtr &mean, const NodePtr &gamma, const NodePtr &dsumOptional) {
    return Emit("AddLayerNormGrad", {dy, x1, x2, rstd, mean, gamma, dsumOptional});
  }
  virtual NodePtr HShrink(const NodePtr &input, const NodePtr &lambd) { return Emit("HShrink", {input, lambd}); }
  virtual NodePtr Take(const NodePtr &input, const NodePtr &index) { return Emit("Take", {input, index}); }
  virtual NodePtr Std(const NodePtr &input, const NodePtr &dim, const NodePtr &correction, const NodePtr &keepdim) {
    return Emit("Std", {input, dim, correction, keepdim});
  }
  virtual NodePtr InplaceErfinv(const NodePtr &input) { return Emit("InplaceErfinv", {input}); }
  virtual NodePtr ToDevice(const NodePtr &input, const NodePtr &device, const NodePtr &dtype,
                           const NodePtr &non_blocking, const NodePtr &copy) {
    return Emit("ToDevice", {input, device, dtype, non_blocking, copy});
  }
  virtual NodePtr FmodTensor(const NodePtr &input, const NodePtr &other) { return Emit("FmodTensor", {input, other}); }
  virtual NodePtr MaskedFill(const NodePtr &input_x, const NodePtr &mask, const NodePtr &value) {
    return Emit("MaskedFill", {input_x, mask, value});
  }
  virtual NodePtr InplaceTanh(const NodePtr &input) { return Emit("InplaceTanh", {input}); }
  virtual NodePtr Expm1(const NodePtr &input) { return Emit("Expm1", {input}); }
  virtual NodePtr InplaceMaskedScatter(const NodePtr &input, const NodePtr &mask, const NodePtr &source) {
    return Emit("InplaceMaskedScatter", {input, mask, source});
  }
  virtual NodePtr Neg(const NodePtr &input) { return Emit("Neg", {input}); }
  virtual NodePtr InplaceBernoulliTensor(const NodePtr &input, const NodePtr &p, const NodePtr &seed,
                                         const NodePtr &offset) {
    return Emit("InplaceBernoulliTensor", {input, p, seed, offset});
  }
  virtual NodePtr DiagonalView(const NodePtr &input, const NodePtr &offset, const NodePtr &dim1, const NodePtr &dim2) {
    return Emit("DiagonalView", {input, offset, dim1, dim2});
  }
  virtual NodePtr FillScalar(const NodePtr &size, const NodePtr &fill_value, const NodePtr &dtype) {
    return Emit("FillScalar", {size, fill_value, dtype});
  }
  virtual NodePtr AdaptiveMaxPool1D(const NodePtr &input, const NodePtr &output_size) {
    return Emit("AdaptiveMaxPool1D", {input, output_size});
  }
  virtual NodePtr LinalgQr(const NodePtr &A, const NodePtr &mode) { return Emit("LinalgQr", {A, mode}); }
  virtual NodePtr ArgMinWithValue(const NodePtr &input, const NodePtr &axis, const NodePtr &keep_dims) {
    return Emit("ArgMinWithValue", {input, axis, keep_dims});
  }
  virtual NodePtr L1LossBackwardExt(const NodePtr &grad_output, const NodePtr &input, const NodePtr &target,
                                    const NodePtr &reduction) {
    return Emit("L1LossBackwardExt", {grad_output, input, target, reduction});
  }
  virtual NodePtr ReflectionPad2D(const NodePtr &input, const NodePtr &padding) {
    return Emit("ReflectionPad2D", {input, padding});
  }
  virtual NodePtr LogicalXor(const NodePtr &input, const NodePtr &other) { return Emit("LogicalXor", {input, other}); }
  virtual NodePtr Cummax(const NodePtr &input, const NodePtr &axis) { return Emit("Cummax", {input, axis}); }
  virtual NodePtr AdaptiveAvgPool2DExt(const NodePtr &input, const NodePtr &output_size) {
    return Emit("AdaptiveAvgPool2DExt", {input, output_size});
  }
  virtual NodePtr GatherDGradV2(const NodePtr &x, const NodePtr &dim, const NodePtr &index, const NodePtr &dout) {
    return Emit("GatherDGradV2", {x, dim, index, dout});
  }
  virtual NodePtr SmoothL1Loss(const NodePtr &prediction, const NodePtr &target, const NodePtr &beta,
                               const NodePtr &reduction) {
    return Emit("SmoothL1Loss", {prediction, target, beta, reduction});
  }
  virtual NodePtr CumminExt(const NodePtr &input, const NodePtr &dim) { return Emit("CumminExt", {input, dim}); }
  virtual NodePtr BCEWithLogitsLoss(const NodePtr &input, const NodePtr &target, const NodePtr &weight,
                                    const NodePtr &posWeight, const NodePtr &reduction) {
    return Emit("BCEWithLogitsLoss", {input, target, weight, posWeight, reduction});
  }
  virtual NodePtr BroadcastToView(const NodePtr &input, const NodePtr &shape) {
    return Emit("BroadcastToView", {input, shape});
  }
  virtual NodePtr RandLikeExt(const NodePtr &tensor, const NodePtr &seed, const NodePtr &offset, const NodePtr &dtype,
                              const NodePtr &device) {
    return Emit("RandLikeExt", {tensor, seed, offset, dtype, device});
  }
  virtual NodePtr InplaceExp(const NodePtr &input) { return Emit("InplaceExp", {input}); }
  virtual NodePtr BitwiseAndTensor(const NodePtr &input, const NodePtr &other) {
    return Emit("BitwiseAndTensor", {input, other});
  }
  virtual NodePtr UpsampleNearest3DGrad(const NodePtr &dy, const NodePtr &input_size, const NodePtr &output_size,
                                        const NodePtr &scales) {
    return Emit("UpsampleNearest3DGrad", {dy, input_size, output_size, scales});
  }
  virtual NodePtr MultiScaleDeformableAttn(const NodePtr &value, const NodePtr &shape, const NodePtr &offset,
                                           const NodePtr &locations, const NodePtr &weight) {
    return Emit("MultiScaleDeformableAttn", {value, shape, offset, locations, weight});
  }
  virtual NodePtr LogicalOr(const NodePtr &x, const NodePtr &y) { return Emit("LogicalOr", {x, y}); }
  virtual NodePtr MaxPoolWithMask(const NodePtr &x, const NodePtr &kernel_size, const NodePtr &strides,
                                  const NodePtr &pads, const NodePtr &dilation, const NodePtr &ceil_mode,
                                  const NodePtr &argmax_type) {
    return Emit("MaxPoolWithMask", {x, kernel_size, strides, pads, dilation, ceil_mode, argmax_type});
  }
  virtual NodePtr InplaceFloorDivides(const NodePtr &input, const NodePtr &other) {
    return Emit("InplaceFloorDivides", {input, other});
  }
  virtual NodePtr ScatterAddExt(const NodePtr &input, const NodePtr &dim, const NodePtr &index, const NodePtr &src) {
    return Emit("ScatterAddExt", {input, dim, index, src});
  }
  virtual NodePtr ReflectionPad3D(const NodePtr &input, const NodePtr &padding) {
    return Emit("ReflectionPad3D", {input, padding});
  }
  virtual NodePtr HSwishGrad(const NodePtr &y_grad, const NodePtr &x) { return Emit("HSwishGrad", {y_grad, x}); }
  virtual NodePtr FlattenExt(const NodePtr &input, const NodePtr &start_dim, const NodePtr &end_dim) {
    return Emit("FlattenExt", {input, start_dim, end_dim});
  }
  virtual NodePtr Square(const NodePtr &input) { return Emit("Square", {input}); }
  virtual NodePtr Addbmm(const NodePtr &input, const NodePtr &batch1, const NodePtr &batch2, const NodePtr &beta,
                         const NodePtr &alpha) {
    return Emit("Addbmm", {input, batch1, batch2, beta, alpha});
  }
  virtual NodePtr Arange(const NodePtr &start, const NodePtr &end, const NodePtr &step, const NodePtr &dtype) {
    return Emit("Arange", {start, end, step, dtype});
  }
  virtual NodePtr InplaceIndexFillTensor(const NodePtr &input, const NodePtr &dim, const NodePtr &index,
                                         const NodePtr &value) {
    return Emit("InplaceIndexFillTensor", {input, dim, index, value});
  }
  virtual NodePtr Round(const NodePtr &input, const NodePtr &decimals) { return Emit("Round", {input, decimals}); }
  virtual NodePtr SliceExtView(const NodePtr &input, const NodePtr &dim, const NodePtr &start, const NodePtr &end,
                               const NodePtr &step) {
    return Emit("SliceExtView", {input, dim, start, end, step});
  }
  virtual NodePtr ArgMinExt(const NodePtr &input, const NodePtr &dim, const NodePtr &keepdim) {
    return Emit("ArgMinExt", {input, dim, keepdim});
  }
  virtual NodePtr ReplicationPad1DGrad(const NodePtr &grad_output, const NodePtr &input, const NodePtr &padding) {
    return Emit("ReplicationPad1DGrad", {grad_output, input, padding});
  }
  virtual NodePtr MaskedSelectGrad(const NodePtr &input, const NodePtr &mask, const NodePtr &grad) {
    return Emit("MaskedSelectGrad", {input, mask, grad});
  }
  virtual NodePtr SubExt(const NodePtr &input, const NodePtr &other, const NodePtr &alpha) {
    return Emit("SubExt", {input, other, alpha});
  }
  virtual NodePtr InnerMoeTokenUnpermute(const NodePtr &permuted_tokens, const NodePtr &sorted_indices,
                                         const NodePtr &probs, const NodePtr &padded_mode,
                                         const NodePtr &restore_shape) {
    return Emit("InnerMoeTokenUnpermute", {permuted_tokens, sorted_indices, probs, padded_mode, restore_shape});
  }
  virtual NodePtr SelectExtView(const NodePtr &input, const NodePtr &dim, const NodePtr &index) {
    return Emit("SelectExtView", {input, dim, index});
  }
  virtual NodePtr InplaceMaskedFillTensor(const NodePtr &input, const NodePtr &mask, const NodePtr &value) {
    return Emit("InplaceMaskedFillTensor", {input, mask, value});
  }
  virtual NodePtr InplaceDivMod(const NodePtr &input, const NodePtr &other, const NodePtr &rounding_mode) {
    return Emit("InplaceDivMod", {input, other, rounding_mode});
  }
  virtual NodePtr NormalFloatTensor(const NodePtr &mean, const NodePtr &std, const NodePtr &seed,
                                    const NodePtr &offset) {
    return Emit("NormalFloatTensor", {mean, std, seed, offset});
  }
  virtual NodePtr SplitTensorView(const NodePtr &input, const NodePtr &split_size, const NodePtr &dim) {
    return Emit("SplitTensorView", {input, split_size, dim});
  }
  virtual NodePtr ReflectionPad2DGrad(const NodePtr &grad_output, const NodePtr &input, const NodePtr &padding) {
    return Emit("ReflectionPad2DGrad", {grad_output, input, padding});
  }
  virtual NodePtr Sign(const NodePtr &input) { return Emit("Sign", {input}); }
  virtual NodePtr Narrow(const NodePtr &input, const NodePtr &dim, const NodePtr &start, const NodePtr &length) {
    return Emit("Narrow", {input, dim, start, length});
  }
  virtual NodePtr GridSampler3D(const NodePtr &input_x, const NodePtr &grid, const NodePtr &interpolation_mode,
                                const NodePtr &padding_mode, const NodePtr &align_corners) {
    return Emit("GridSampler3D", {input_x, grid, interpolation_mode, padding_mode, align_corners});
  }
  virtual NodePtr AddLayerNormV2(const NodePtr &x1, const NodePtr &x2, const NodePtr &gamma, const NodePtr &beta,
                                 const NodePtr &epsilon, const NodePtr &additionalOut) {
    return Emit("AddLayerNormV2", {x1, x2, gamma, beta, epsilon, additionalOut});
  }
  virtual NodePtr IsInf(const NodePtr &input) { return Emit("IsInf", {input}); }
  virtual NodePtr InplaceIndexAddExt(const NodePtr &input, const NodePtr &dim, const NodePtr &index,
                                     const NodePtr &source, const NodePtr &alpha) {
    return Emit("InplaceIndexAddExt", {input, dim, index, source, alpha});
  }
  virtual NodePtr BatchNormGradExt(const NodePtr &dout, const NodePtr &input, const NodePtr &weight,
                                   const NodePtr &running_mean, const NodePtr &running_var, const NodePtr &saved_mean,
                                   const NodePtr &saved_rstd, const NodePtr &training, const NodePtr &eps,
                                   const NodePtr &output_mask) {
    return Emit("BatchNormGradExt",
                {dout, input, weight, running_mean, running_var, saved_mean, saved_rstd, training, eps, output_mask});
  }
  virtual NodePtr DivMod(const NodePtr &input, const NodePtr &other, const NodePtr &rounding_mode) {
    return Emit("DivMod", {input, other, rounding_mode});
  }
  virtual NodePtr Slice(const NodePtr &input, const NodePtr &begin, const NodePtr &size) {
    return Emit("Slice", {input, begin, size});
  }
  virtual NodePtr RandIntLike(const NodePtr &input, const NodePtr &low, const NodePtr &high, const NodePtr &seed,
                              const NodePtr &offset, const NodePtr &dtype, const NodePtr &device) {
    return Emit("RandIntLike", {input, low, high, seed, offset, dtype, device});
  }
  virtual NodePtr AsStrided(const NodePtr &input, const NodePtr &size, const NodePtr &stride,
                            const NodePtr &storage_offset) {
    return Emit("AsStrided", {input, size, stride, storage_offset});
  }
  virtual NodePtr MaxPoolGradWithIndices(const NodePtr &x, const NodePtr &grad, const NodePtr &argmax,
                                         const NodePtr &kernel_size, const NodePtr &strides, const NodePtr &pads,
                                         const NodePtr &dilation, const NodePtr &ceil_mode,
                                         const NodePtr &argmax_type) {
    return Emit("MaxPoolGradWithIndices",
                {x, grad, argmax, kernel_size, strides, pads, dilation, ceil_mode, argmax_type});
  }
  virtual NodePtr RemainderTensorScalar(const NodePtr &input, const NodePtr &other) {
    return Emit("RemainderTensorScalar", {input, other});
  }
  virtual NodePtr FmodScalar(const NodePtr &input, const NodePtr &other) { return Emit("FmodScalar", {input, other}); }
  virtual NodePtr Randn(const NodePtr &shape, const NodePtr &seed, const NodePtr &offset, const NodePtr &dtype,
                        const NodePtr &device) {
    return Emit("Randn", {shape, seed, offset, dtype, device});
  }
  virtual NodePtr BitwiseXorScalar(const NodePtr &input, const NodePtr &other) {
    return Emit("BitwiseXorScalar", {input, other});
  }
  virtual NodePtr UpsampleTrilinear3D(const NodePtr &x, const NodePtr &output_size, const NodePtr &scales,
                                      const NodePtr &align_corners) {
    return Emit("UpsampleTrilinear3D", {x, output_size, scales, align_corners});
  }
  virtual NodePtr ArgMaxWithValue(const NodePtr &input, const NodePtr &axis, const NodePtr &keep_dims) {
    return Emit("ArgMaxWithValue", {input, axis, keep_dims});
  }
  virtual NodePtr InplaceFloor(const NodePtr &input) { return Emit("InplaceFloor", {input}); }
  virtual NodePtr UnstackExtView(const NodePtr &input, const NodePtr &dim) {
    return Emit("UnstackExtView", {input, dim});
  }
  virtual NodePtr InplaceFloorDivide(const NodePtr &input, const NodePtr &other) {
    return Emit("InplaceFloorDivide", {input, other});
  }
  virtual NodePtr InplaceSubExt(const NodePtr &input, const NodePtr &other, const NodePtr &alpha) {
    return Emit("InplaceSubExt", {input, other, alpha});
  }
  virtual NodePtr GeLU(const NodePtr &input) { return Emit("GeLU", {input}); }
  virtual NodePtr ReplicationPad1D(const NodePtr &input, const NodePtr &padding) {
    return Emit("ReplicationPad1D", {input, padding});
  }
  virtual NodePtr InplaceCopy(const NodePtr &input, const NodePtr &src, const NodePtr &non_blocking) {
    return Emit("InplaceCopy", {input, src, non_blocking});
  }
  virtual NodePtr Baddbmm(const NodePtr &input, const NodePtr &batch1, const NodePtr &batch2, const NodePtr &beta,
                          const NodePtr &alpha) {
    return Emit("Baddbmm", {input, batch1, batch2, beta, alpha});
  }
  virtual NodePtr ExpandDims(const NodePtr &input_x, const NodePtr &axis) {
    return Emit("ExpandDims", {input_x, axis});
  }
  virtual NodePtr LeakyReLUExt(const NodePtr &input, const NodePtr &negative_slope) {
    return Emit("LeakyReLUExt", {input, negative_slope});
  }
  virtual NodePtr UniqueDim(const NodePtr &input, const NodePtr &sorted, const NodePtr &return_inverse,
                            const NodePtr &dim) {
    return Emit("UniqueDim", {input, sorted, return_inverse, dim});
  }
  virtual NodePtr HistcExt(const NodePtr &input, const NodePtr &bins, const NodePtr &min, const NodePtr &max) {
    return Emit("HistcExt", {input, bins, min, max});
  }
  virtual NodePtr IncreFlashAttention(const NodePtr &query, const NodePtr &key, const NodePtr &value,
                                      const NodePtr &attn_mask, const NodePtr &actual_seq_lengths,
                                      const NodePtr &pse_shift, const NodePtr &dequant_scale1,
                                      const NodePtr &quant_scale1, const NodePtr &dequant_scale2,
                                      const NodePtr &quant_scale2, const NodePtr &quant_offset2,
                                      const NodePtr &antiquant_scale, const NodePtr &antiquant_offset,
                                      const NodePtr &block_table, const NodePtr &kv_padding_size,
                                      const NodePtr &num_heads, const NodePtr &input_layout, const NodePtr &scale_value,
                                      const NodePtr &num_key_value_heads, const NodePtr &block_size,
                                      const NodePtr &inner_precise) {
    return Emit("IncreFlashAttention", {query,
                                        key,
                                        value,
                                        attn_mask,
                                        actual_seq_lengths,
                                        pse_shift,
                                        dequant_scale1,
                                        quant_scale1,
                                        dequant_scale2,
                                        quant_scale2,
                                        quant_offset2,
                                        antiquant_scale,
                                        antiquant_offset,
                                        block_table,
                                        kv_padding_size,
                                        num_heads,
                                        input_layout,
                                        scale_value,
                                        num_key_value_heads,
                                        block_size,
                                        inner_precise});
  }
  virtual NodePtr Log10(const NodePtr &input) { return Emit("Log10", {input}); }
  virtual NodePtr EmbeddingDenseBackward(const NodePtr &grad, const NodePtr &indices, const NodePtr &num_weights,
                                         const NodePtr &padding_idx, const NodePtr &scale_grad_by_freq) {
    return Emit("EmbeddingDenseBackward", {grad, indices, num_weights, padding_idx, scale_grad_by_freq});
  }
  virtual NodePtr Mm(const NodePtr &input, const NodePtr &mat2) { return Emit("Mm", {input, mat2}); }
  virtual NodePtr Col2ImExt(const NodePtr &input, const NodePtr &output_size, const NodePtr &kernel_size,
                            const NodePtr &dilation, const NodePtr &padding, const NodePtr &stride) {
    return Emit("Col2ImExt", {input, output_size, kernel_size, dilation, padding, stride});
  }
  virtual NodePtr GeLUGrad(const NodePtr &dy, const NodePtr &x, const NodePtr &y) {
    return Emit("GeLUGrad", {dy, x, y});
  }
  virtual NodePtr OneHotExt(const NodePtr &tensor, const NodePtr &num_classes, const NodePtr &on_value,
                            const NodePtr &off_value, const NodePtr &axis) {
    return Emit("OneHotExt", {tensor, num_classes, on_value, off_value, axis});
  }
  virtual NodePtr SiLUGrad(const NodePtr &dout, const NodePtr &x) { return Emit("SiLUGrad", {dout, x}); }
  virtual NodePtr ConvolutionStrGrad(const NodePtr &dout, const NodePtr &input, const NodePtr &weight,
                                     const NodePtr &bias, const NodePtr &stride, const NodePtr &padding,
                                     const NodePtr &dilation, const NodePtr &transposed, const NodePtr &output_padding,
                                     const NodePtr &groups, const NodePtr &output_mask) {
    return Emit("ConvolutionStrGrad", {dout, input, weight, bias, stride, padding, dilation, transposed, output_padding,
                                       groups, output_mask});
  }
  virtual NodePtr InplaceDivMods(const NodePtr &input, const NodePtr &other, const NodePtr &rounding_mode) {
    return Emit("InplaceDivMods", {input, other, rounding_mode});
  }
  virtual NodePtr SortExt(const NodePtr &input, const NodePtr &dim, const NodePtr &descending, const NodePtr &stable) {
    return Emit("SortExt", {input, dim, descending, stable});
  }
  virtual NodePtr Generator(const NodePtr &cmd, const NodePtr &inputs) { return Emit("Generator", {cmd, inputs}); }
  virtual NodePtr LinSpaceExt(const NodePtr &start, const NodePtr &end, const NodePtr &steps, const NodePtr &dtype) {
    return Emit("LinSpaceExt", {start, end, steps, dtype});
  }
  virtual NodePtr InnerUnique(const NodePtr &input, const NodePtr &sorted, const NodePtr &return_inverse) {
    return Emit("InnerUnique", {input, sorted, return_inverse});
  }
  virtual NodePtr AddcdivExt(const NodePtr &input, const NodePtr &tensor1, const NodePtr &tensor2,
                             const NodePtr &value) {
    return Emit("AddcdivExt", {input, tensor1, tensor2, value});
  }
  virtual NodePtr LogAddExp2(const NodePtr &input, const NodePtr &other) { return Emit("LogAddExp2", {input, other}); }
  virtual NodePtr ThresholdGrad(const NodePtr &grad_output, const NodePtr &input, const NodePtr &threshold) {
    return Emit("ThresholdGrad", {grad_output, input, threshold});
  }
  virtual NodePtr LogSoftmaxExt(const NodePtr &input, const NodePtr &dim, const NodePtr &dtype) {
    return Emit("LogSoftmaxExt", {input, dim, dtype});
  }
  virtual NodePtr PowScalarTensor(const NodePtr &input, const NodePtr &exponent) {
    return Emit("PowScalarTensor", {input, exponent});
  }
  virtual NodePtr AvgPool3DExt(const NodePtr &input, const NodePtr &kernel_size, const NodePtr &stride,
                               const NodePtr &padding, const NodePtr &ceil_mode, const NodePtr &count_include_pad,
                               const NodePtr &divisor_override) {
    return Emit("AvgPool3DExt", {input, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override});
  }
  virtual NodePtr InplaceFillDiagonal(const NodePtr &input, const NodePtr &fill_value, const NodePtr &wrap) {
    return Emit("InplaceFillDiagonal", {input, fill_value, wrap});
  }
  virtual NodePtr Col2ImGrad(const NodePtr &input, const NodePtr &kernel_size, const NodePtr &dilation,
                             const NodePtr &padding, const NodePtr &stride) {
    return Emit("Col2ImGrad", {input, kernel_size, dilation, padding, stride});
  }
  virtual NodePtr AllGatherMatmul(const NodePtr &input, const NodePtr &x2, const NodePtr &group,
                                  const NodePtr &world_size, const NodePtr &bias, const NodePtr &gather_index,
                                  const NodePtr &gather_output, const NodePtr &comm_turn, const NodePtr &trans_input,
                                  const NodePtr &trans_x2) {
    return Emit("AllGatherMatmul",
                {input, x2, group, world_size, bias, gather_index, gather_output, comm_turn, trans_input, trans_x2});
  }
  virtual NodePtr MaxUnpool2DExt(const NodePtr &input, const NodePtr &indices, const NodePtr &kernel_size,
                                 const NodePtr &stride, const NodePtr &padding, const NodePtr &output_size) {
    return Emit("MaxUnpool2DExt", {input, indices, kernel_size, stride, padding, output_size});
  }
  virtual NodePtr InplaceGroupedMatmulAdd(const NodePtr &x, const NodePtr &weight, const NodePtr &group_list,
                                          const NodePtr &out) {
    return Emit("InplaceGroupedMatmulAdd", {x, weight, group_list, out});
  }
  virtual NodePtr MaxPoolWithIndices(const NodePtr &x, const NodePtr &kernel_size, const NodePtr &strides,
                                     const NodePtr &pads, const NodePtr &dilation, const NodePtr &ceil_mode,
                                     const NodePtr &argmax_type) {
    return Emit("MaxPoolWithIndices", {x, kernel_size, strides, pads, dilation, ceil_mode, argmax_type});
  }
  virtual NodePtr SoftmaxBackward(const NodePtr &dout, const NodePtr &out, const NodePtr &dim) {
    return Emit("SoftmaxBackward", {dout, out, dim});
  }
  virtual NodePtr MatrixInverseExt(const NodePtr &input) { return Emit("MatrixInverseExt", {input}); }
  virtual NodePtr Tanh(const NodePtr &input) { return Emit("Tanh", {input}); }
  virtual NodePtr DropoutGradExt(const NodePtr &input, const NodePtr &mask, const NodePtr &p) {
    return Emit("DropoutGradExt", {input, mask, p});
  }
  virtual NodePtr InnerNonZero(const NodePtr &input) { return Emit("InnerNonZero", {input}); }
  virtual NodePtr AllFinite(const NodePtr &tensors) { return Emit("AllFinite", {tensors}); }
  virtual NodePtr ReshapeAndCache(const NodePtr &key, const NodePtr &value, const NodePtr &key_cache,
                                  const NodePtr &value_cache, const NodePtr &slot_mapping) {
    return Emit("ReshapeAndCache", {key, value, key_cache, value_cache, slot_mapping});
  }
  virtual NodePtr InplaceClampScalar(const NodePtr &input, const NodePtr &min, const NodePtr &max) {
    return Emit("InplaceClampScalar", {input, min, max});
  }
  virtual NodePtr NewOnes(const NodePtr &input, const NodePtr &size, const NodePtr &dtype) {
    return Emit("NewOnes", {input, size, dtype});
  }
  virtual NodePtr Dot(const NodePtr &input, const NodePtr &other) { return Emit("Dot", {input, other}); }
  virtual NodePtr InplaceAddExt(const NodePtr &input, const NodePtr &other, const NodePtr &alpha) {
    return Emit("InplaceAddExt", {input, other, alpha});
  }
  virtual NodePtr XLogYScalarOther(const NodePtr &input, const NodePtr &other) {
    return Emit("XLogYScalarOther", {input, other});
  }
  virtual NodePtr AvgPool1D(const NodePtr &input, const NodePtr &kernel_size, const NodePtr &stride,
                            const NodePtr &padding, const NodePtr &ceil_mode, const NodePtr &count_include_pad) {
    return Emit("AvgPool1D", {input, kernel_size, stride, padding, ceil_mode, count_include_pad});
  }
  virtual NodePtr RotaryPositionEmbedding(const NodePtr &x, const NodePtr &cos, const NodePtr &sin,
                                          const NodePtr &mode) {
    return Emit("RotaryPositionEmbedding", {x, cos, sin, mode});
  }
  virtual NodePtr RmsNorm(const NodePtr &x, const NodePtr &gamma, const NodePtr &epsilon) {
    return Emit("RmsNorm", {x, gamma, epsilon});
  }
  virtual NodePtr InplaceZero(const NodePtr &input) { return Emit("InplaceZero", {input}); }
  virtual NodePtr ExpandDimsView(const NodePtr &input, const NodePtr &dim) {
    return Emit("ExpandDimsView", {input, dim});
  }
  virtual NodePtr Outer(const NodePtr &input, const NodePtr &vec2) { return Emit("Outer", {input, vec2}); }
  virtual NodePtr InplaceLog(const NodePtr &input) { return Emit("InplaceLog", {input}); }
  virtual NodePtr ToOther(const NodePtr &input, const NodePtr &other, const NodePtr &non_blocking,
                          const NodePtr &copy) {
    return Emit("ToOther", {input, other, non_blocking, copy});
  }
  virtual NodePtr InplaceAddmm(const NodePtr &input, const NodePtr &mat1, const NodePtr &mat2, const NodePtr &beta,
                               const NodePtr &alpha) {
    return Emit("InplaceAddmm", {input, mat1, mat2, beta, alpha});
  }
  virtual NodePtr InplaceThreshold(const NodePtr &input, const NodePtr &threshold, const NodePtr &value) {
    return Emit("InplaceThreshold", {input, threshold, value});
  }
  virtual NodePtr IsClose(const NodePtr &input, const NodePtr &other, const NodePtr &rtol, const NodePtr &atol,
                          const NodePtr &equal_nan) {
    return Emit("IsClose", {input, other, rtol, atol, equal_nan});
  }
  virtual NodePtr GridSampler2DGrad(const NodePtr &grad, const NodePtr &input_x, const NodePtr &grid,
                                    const NodePtr &interpolation_mode, const NodePtr &padding_mode,
                                    const NodePtr &align_corners, const NodePtr &output_mask) {
    return Emit("GridSampler2DGrad",
                {grad, input_x, grid, interpolation_mode, padding_mode, align_corners, output_mask});
  }
  virtual NodePtr ReflectionPad1D(const NodePtr &input, const NodePtr &padding) {
    return Emit("ReflectionPad1D", {input, padding});
  }
  virtual NodePtr InplaceIndexCopy(const NodePtr &input, const NodePtr &dim, const NodePtr &index,
                                   const NodePtr &tensor) {
    return Emit("InplaceIndexCopy", {input, dim, index, tensor});
  }
  virtual NodePtr InplaceStopGradient(const NodePtr &input) { return Emit("InplaceStopGradient", {input}); }
  virtual NodePtr BernoulliExt(const NodePtr &input, const NodePtr &seed, const NodePtr &offset) {
    return Emit("BernoulliExt", {input, seed, offset});
  }
  virtual NodePtr InplaceDiv(const NodePtr &input, const NodePtr &other) { return Emit("InplaceDiv", {input, other}); }
  virtual NodePtr Log1p(const NodePtr &input) { return Emit("Log1p", {input}); }
  virtual NodePtr SubScalar(const NodePtr &input, const NodePtr &other, const NodePtr &alpha) {
    return Emit("SubScalar", {input, other, alpha});
  }
  virtual NodePtr Addmv(const NodePtr &input, const NodePtr &mat, const NodePtr &vec, const NodePtr &beta,
                        const NodePtr &alpha) {
    return Emit("Addmv", {input, mat, vec, beta, alpha});
  }
  virtual NodePtr SearchSorted(const NodePtr &sorted_sequence, const NodePtr &values, const NodePtr &sorter,
                               const NodePtr &dtype, const NodePtr &right) {
    return Emit("SearchSorted", {sorted_sequence, values, sorter, dtype, right});
  }
  virtual NodePtr UpsampleBicubic2D(const NodePtr &x, const NodePtr &output_size, const NodePtr &scales,
                                    const NodePtr &align_corners) {
    return Emit("UpsampleBicubic2D", {x, output_size, scales, align_corners});
  }
  virtual NodePtr GatherD(const NodePtr &x, const NodePtr &dim, const NodePtr &index) {
    return Emit("GatherD", {x, dim, index});
  }
  virtual NodePtr Scatter(const NodePtr &input, const NodePtr &dim, const NodePtr &index, const NodePtr &src,
                          const NodePtr &reduce) {
    return Emit("Scatter", {input, dim, index, src, reduce});
  }
  virtual NodePtr AcoshExt(const NodePtr &input) { return Emit("AcoshExt", {input}); }
  virtual NodePtr Convolution(const NodePtr &input, const NodePtr &weight, const NodePtr &bias, const NodePtr &stride,
                              const NodePtr &padding, const NodePtr &dilation, const NodePtr &transposed,
                              const NodePtr &output_padding, const NodePtr &groups) {
    return Emit("Convolution", {input, weight, bias, stride, padding, dilation, transposed, output_padding, groups});
  }
  virtual NodePtr Chunk(const NodePtr &input, const NodePtr &chunks, const NodePtr &dim) {
    return Emit("Chunk", {input, chunks, dim});
  }
  virtual NodePtr Clone(const NodePtr &input) { return Emit("Clone", {input}); }
  virtual NodePtr ReLU(const NodePtr &input) { return Emit("ReLU", {input}); }
  virtual NodePtr VarMean(const NodePtr &input, const NodePtr &dim, const NodePtr &correction, const NodePtr &keepdim) {
    return Emit("VarMean", {input, dim, correction, keepdim});
  }
  virtual NodePtr InplaceFillScalar(const NodePtr &input, const NodePtr &value) {
    return Emit("InplaceFillScalar", {input, value});
  }
  virtual NodePtr MultinomialExt(const NodePtr &input, const NodePtr &num_samples, const NodePtr &replacement,
                                 const NodePtr &seed, const NodePtr &offset) {
    return Emit("MultinomialExt", {input, num_samples, replacement, seed, offset});
  }
  virtual NodePtr MishGradExt(const NodePtr &dout, const NodePtr &x) { return Emit("MishGradExt", {dout, x}); }
  virtual NodePtr ReduceMax(const NodePtr &x, const NodePtr &axis, const NodePtr &keep_dims) {
    return Emit("ReduceMax", {x, axis, keep_dims});
  }
  virtual NodePtr ArgSort(const NodePtr &input, const NodePtr &dim, const NodePtr &descending, const NodePtr &stable) {
    return Emit("ArgSort", {input, dim, descending, stable});
  }
  virtual NodePtr GeluGradExt(const NodePtr &grad, const NodePtr &input, const NodePtr &approximate) {
    return Emit("GeluGradExt", {grad, input, approximate});
  }
  virtual NodePtr BinaryCrossEntropyWithLogitsBackward(const NodePtr &grad_output, const NodePtr &input,
                                                       const NodePtr &target, const NodePtr &weight,
                                                       const NodePtr &posWeight, const NodePtr &reduction) {
    return Emit("BinaryCrossEntropyWithLogitsBackward", {grad_output, input, target, weight, posWeight, reduction});
  }
  virtual NodePtr LinalgVectorNorm(const NodePtr &x, const NodePtr &ord, const NodePtr &dim, const NodePtr &keepdim,
                                   const NodePtr &dtype) {
    return Emit("LinalgVectorNorm", {x, ord, dim, keepdim, dtype});
  }
  virtual NodePtr Norm(const NodePtr &input, const NodePtr &p, const NodePtr &dim, const NodePtr &keepdim,
                       const NodePtr &dtype) {
    return Emit("Norm", {input, p, dim, keepdim, dtype});
  }
  virtual NodePtr BatchNormElemtGrad(const NodePtr &dout, const NodePtr &input, const NodePtr &mean,
                                     const NodePtr &invstd, const NodePtr &weight, const NodePtr &sumd_dy,
                                     const NodePtr &sum_dy_xmu, const NodePtr &count) {
    return Emit("BatchNormElemtGrad", {dout, input, mean, invstd, weight, sumd_dy, sum_dy_xmu, count});
  }
  virtual NodePtr RepeatInterleaveTensor(const NodePtr &input, const NodePtr &repeats, const NodePtr &dim,
                                         const NodePtr &output_size) {
    return Emit("RepeatInterleaveTensor", {input, repeats, dim, output_size});
  }
  virtual NodePtr TrilExt(const NodePtr &input, const NodePtr &diagonal) { return Emit("TrilExt", {input, diagonal}); }
  virtual NodePtr PReLUGrad(const NodePtr &dy, const NodePtr &x, const NodePtr &weight) {
    return Emit("PReLUGrad", {dy, x, weight});
  }
  virtual NodePtr InplaceScatterSrcReduce(const NodePtr &input, const NodePtr &dim, const NodePtr &index,
                                          const NodePtr &src, const NodePtr &reduce) {
    return Emit("InplaceScatterSrcReduce", {input, dim, index, src, reduce});
  }
  virtual NodePtr AdaptiveAvgPool3DGradExt(const NodePtr &input_grad, const NodePtr &input) {
    return Emit("AdaptiveAvgPool3DGradExt", {input_grad, input});
  }
  virtual NodePtr BitwiseOrScalar(const NodePtr &input, const NodePtr &other) {
    return Emit("BitwiseOrScalar", {input, other});
  }
  virtual NodePtr InplaceNormal(const NodePtr &input, const NodePtr &mean, const NodePtr &std, const NodePtr &seed,
                                const NodePtr &offset) {
    return Emit("InplaceNormal", {input, mean, std, seed, offset});
  }
  virtual NodePtr CountNonZero(const NodePtr &input, const NodePtr &dim) { return Emit("CountNonZero", {input, dim}); }
  virtual NodePtr EqualExt(const NodePtr &input, const NodePtr &other) { return Emit("EqualExt", {input, other}); }
  virtual NodePtr StdMean(const NodePtr &input, const NodePtr &dim, const NodePtr &correction, const NodePtr &keepdim) {
    return Emit("StdMean", {input, dim, correction, keepdim});
  }
  virtual NodePtr BatchNormReduceGrad(const NodePtr &dout, const NodePtr &input, const NodePtr &mean,
                                      const NodePtr &invstd, const NodePtr &weight, const NodePtr &input_g,
                                      const NodePtr &weight_g, const NodePtr &bias_g) {
    return Emit("BatchNormReduceGrad", {dout, input, mean, invstd, weight, input_g, weight_g, bias_g});
  }
  virtual NodePtr GroupNormGrad(const NodePtr &dy, const NodePtr &x, const NodePtr &mean, const NodePtr &rstd,
                                const NodePtr &gamma_opt, const NodePtr &num_groups, const NodePtr &dx_is_require,
                                const NodePtr &dgamma_is_require, const NodePtr &dbeta_is_require) {
    return Emit("GroupNormGrad",
                {dy, x, mean, rstd, gamma_opt, num_groups, dx_is_require, dgamma_is_require, dbeta_is_require});
  }
  virtual NodePtr TanhGrad(const NodePtr &y, const NodePtr &dy) { return Emit("TanhGrad", {y, dy}); }
  virtual NodePtr MaskedScatter(const NodePtr &input, const NodePtr &mask, const NodePtr &source) {
    return Emit("MaskedScatter", {input, mask, source});
  }
  virtual NodePtr BitwiseOrTensor(const NodePtr &input, const NodePtr &other) {
    return Emit("BitwiseOrTensor", {input, other});
  }
  virtual NodePtr NLLLoss2d(const NodePtr &input, const NodePtr &target, const NodePtr &weight,
                            const NodePtr &reduction, const NodePtr &ignore_index) {
    return Emit("NLLLoss2d", {input, target, weight, reduction, ignore_index});
  }
  virtual NodePtr BatchNormElemt(const NodePtr &input, const NodePtr &weight, const NodePtr &bias, const NodePtr &mean,
                                 const NodePtr &invstd, const NodePtr &eps) {
    return Emit("BatchNormElemt", {input, weight, bias, mean, invstd, eps});
  }
  virtual NodePtr Hardtanh(const NodePtr &input, const NodePtr &min_val, const NodePtr &max_val) {
    return Emit("Hardtanh", {input, min_val, max_val});
  }
  virtual NodePtr Exp2(const NodePtr &input) { return Emit("Exp2", {input}); }
  virtual NodePtr Cos(const NodePtr &input) { return Emit("Cos", {input}); }
  virtual NodePtr SmoothL1LossGrad(const NodePtr &prediction, const NodePtr &target, const NodePtr &dout,
                                   const NodePtr &beta, const NodePtr &reduction) {
    return Emit("SmoothL1LossGrad", {prediction, target, dout, beta, reduction});
  }
  virtual NodePtr MishExt(const NodePtr &input) { return Emit("MishExt", {input}); }
  virtual NodePtr TransposeView(const NodePtr &input, const NodePtr &input_perm) {
    return Emit("TransposeView", {input, input_perm});
  }
  virtual NodePtr AddExt(const NodePtr &input, const NodePtr &other, const NodePtr &alpha) {
    return Emit("AddExt", {input, other, alpha});
  }
  virtual NodePtr TransposeExtView(const NodePtr &input, const NodePtr &dim0, const NodePtr &dim1) {
    return Emit("TransposeExtView", {input, dim0, dim1});
  }
  virtual NodePtr ZerosLikeExt(const NodePtr &input, const NodePtr &dtype) {
    return Emit("ZerosLikeExt", {input, dtype});
  }
  virtual NodePtr NewZeros(const NodePtr &input, const NodePtr &size, const NodePtr &dtype) {
    return Emit("NewZeros", {input, size, dtype});
  }
  virtual NodePtr Roll(const NodePtr &input, const NodePtr &shifts, const NodePtr &dims) {
    return Emit("Roll", {input, shifts, dims});
  }
  virtual NodePtr InplaceClampTensor(const NodePtr &input, const NodePtr &min, const NodePtr &max) {
    return Emit("InplaceClampTensor", {input, min, max});
  }
  virtual NodePtr ExpandAs(const NodePtr &input, const NodePtr &other) { return Emit("ExpandAs", {input, other}); }
  virtual NodePtr Conv1DExt(const NodePtr &input, const NodePtr &weight, const NodePtr &bias, const NodePtr &stride,
                            const NodePtr &padding, const NodePtr &dilation, const NodePtr &groups) {
    return Emit("Conv1DExt", {input, weight, bias, stride, padding, dilation, groups});
  }
  virtual NodePtr ReflectionPad3DGrad(const NodePtr &grad_output, const NodePtr &input, const NodePtr &padding) {
    return Emit("ReflectionPad3DGrad", {grad_output, input, padding});
  }
  virtual NodePtr AvgPool2D(const NodePtr &input, const NodePtr &kernel_size, const NodePtr &stride,
                            const NodePtr &padding, const NodePtr &ceil_mode, const NodePtr &count_include_pad,
                            const NodePtr &divisor_override) {
    return Emit("AvgPool2D", {input, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override});
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
  virtual NodePtr BatchNormGatherStatsWithCounts(const NodePtr &input, const NodePtr &mean, const NodePtr &invstd,
                                                 const NodePtr &running_mean, const NodePtr &running_var,
                                                 const NodePtr &momentum, const NodePtr &eps, const NodePtr &counts) {
    return Emit("BatchNormGatherStatsWithCounts",
                {input, mean, invstd, running_mean, running_var, momentum, eps, counts});
  }
  virtual NodePtr AtanExt(const NodePtr &input) { return Emit("AtanExt", {input}); }
  virtual NodePtr Log2(const NodePtr &input) { return Emit("Log2", {input}); }
  virtual NodePtr RandpermExt(const NodePtr &n, const NodePtr &seed, const NodePtr &offset, const NodePtr &dtype) {
    return Emit("RandpermExt", {n, seed, offset, dtype});
  }
  virtual NodePtr LogAddExp(const NodePtr &input, const NodePtr &other) { return Emit("LogAddExp", {input, other}); }
  virtual NodePtr LogSigmoid(const NodePtr &input) { return Emit("LogSigmoid", {input}); }
  virtual NodePtr XLogYScalarSelf(const NodePtr &input, const NodePtr &other) {
    return Emit("XLogYScalarSelf", {input, other});
  }
  virtual NodePtr TriangularSolve(const NodePtr &b, const NodePtr &A, const NodePtr &upper, const NodePtr &transpose,
                                  const NodePtr &unitriangular) {
    return Emit("TriangularSolve", {b, A, upper, transpose, unitriangular});
  }
  virtual NodePtr SpeedFusionAttention(const NodePtr &query, const NodePtr &key, const NodePtr &value,
                                       const NodePtr &head_num, const NodePtr &input_layout, const NodePtr &seed,
                                       const NodePtr &offset, const NodePtr &pse, const NodePtr &padding_mask,
                                       const NodePtr &atten_mask, const NodePtr &scale, const NodePtr &keep_prob,
                                       const NodePtr &pre_tokens, const NodePtr &next_tokens,
                                       const NodePtr &inner_precise, const NodePtr &prefix,
                                       const NodePtr &actual_seq_qlen, const NodePtr &actual_seq_kvlen,
                                       const NodePtr &sparse_mode, const NodePtr &gen_mask_parallel,
                                       const NodePtr &sync, const NodePtr &pse_type, const NodePtr &q_start_idx,
                                       const NodePtr &kv_start_idx) {
    return Emit("SpeedFusionAttention", {query,
                                         key,
                                         value,
                                         head_num,
                                         input_layout,
                                         seed,
                                         offset,
                                         pse,
                                         padding_mask,
                                         atten_mask,
                                         scale,
                                         keep_prob,
                                         pre_tokens,
                                         next_tokens,
                                         inner_precise,
                                         prefix,
                                         actual_seq_qlen,
                                         actual_seq_kvlen,
                                         sparse_mode,
                                         gen_mask_parallel,
                                         sync,
                                         pse_type,
                                         q_start_idx,
                                         kv_start_idx});
  }
  virtual NodePtr GluGrad(const NodePtr &grads, const NodePtr &x, const NodePtr &axis) {
    return Emit("GluGrad", {grads, x, axis});
  }
  virtual NodePtr IsNegInf(const NodePtr &input) { return Emit("IsNegInf", {input}); }
  virtual NodePtr DropoutGenMaskExt(const NodePtr &shape, const NodePtr &p, const NodePtr &seed, const NodePtr &offset,
                                    const NodePtr &dtype) {
    return Emit("DropoutGenMaskExt", {shape, p, seed, offset, dtype});
  }
  virtual NodePtr HShrinkGrad(const NodePtr &gradients, const NodePtr &features, const NodePtr &lambd) {
    return Emit("HShrinkGrad", {gradients, features, lambd});
  }
  virtual NodePtr EmptyLike(const NodePtr &input, const NodePtr &dtype, const NodePtr &device,
                            const NodePtr &pin_memory) {
    return Emit("EmptyLike", {input, dtype, device, pin_memory});
  }
  virtual NodePtr MeanExt(const NodePtr &input, const NodePtr &dim, const NodePtr &keepdim, const NodePtr &dtype) {
    return Emit("MeanExt", {input, dim, keepdim, dtype});
  }
  virtual NodePtr InplaceScatterAdd(const NodePtr &input, const NodePtr &dim, const NodePtr &index,
                                    const NodePtr &src) {
    return Emit("InplaceScatterAdd", {input, dim, index, src});
  }
  virtual NodePtr InplaceMul(const NodePtr &input, const NodePtr &other) { return Emit("InplaceMul", {input, other}); }
  virtual NodePtr LayerNormExt(const NodePtr &input, const NodePtr &normalized_shape, const NodePtr &weight,
                               const NodePtr &bias, const NodePtr &eps) {
    return Emit("LayerNormExt", {input, normalized_shape, weight, bias, eps});
  }
  virtual NodePtr LogicalAnd(const NodePtr &x, const NodePtr &y) { return Emit("LogicalAnd", {x, y}); }
  virtual NodePtr Divs(const NodePtr &input, const NodePtr &other) { return Emit("Divs", {input, other}); }
  virtual NodePtr InnerInplaceIndexPut(const NodePtr &input, const NodePtr &indices, const NodePtr &values,
                                       const NodePtr &accumulate) {
    return Emit("InnerInplaceIndexPut", {input, indices, values, accumulate});
  }
  virtual NodePtr InplaceIndexFillScalar(const NodePtr &input, const NodePtr &dim, const NodePtr &index,
                                         const NodePtr &value) {
    return Emit("InplaceIndexFillScalar", {input, dim, index, value});
  }
  virtual NodePtr NormalTensorFloat(const NodePtr &mean, const NodePtr &std, const NodePtr &seed,
                                    const NodePtr &offset) {
    return Emit("NormalTensorFloat", {mean, std, seed, offset});
  }
  virtual NodePtr AdaptiveAvgPool2DGradExt(const NodePtr &grad_output, const NodePtr &x) {
    return Emit("AdaptiveAvgPool2DGradExt", {grad_output, x});
  }
  virtual NodePtr ProdExt(const NodePtr &input, const NodePtr &dim, const NodePtr &keepdim, const NodePtr &dtype) {
    return Emit("ProdExt", {input, dim, keepdim, dtype});
  }
  virtual NodePtr Softmax(const NodePtr &input, const NodePtr &axis) { return Emit("Softmax", {input, axis}); }
  virtual NodePtr InplaceElu(const NodePtr &input, const NodePtr &alpha) { return Emit("InplaceElu", {input, alpha}); }
  virtual NodePtr NeScalar(const NodePtr &input, const NodePtr &other) { return Emit("NeScalar", {input, other}); }
  virtual NodePtr Conv2DExt(const NodePtr &input, const NodePtr &weight, const NodePtr &bias, const NodePtr &stride,
                            const NodePtr &padding, const NodePtr &dilation, const NodePtr &groups) {
    return Emit("Conv2DExt", {input, weight, bias, stride, padding, dilation, groups});
  }
  virtual NodePtr RandnLike(const NodePtr &input, const NodePtr &seed, const NodePtr &offset, const NodePtr &dtype,
                            const NodePtr &device) {
    return Emit("RandnLike", {input, seed, offset, dtype, device});
  }
  virtual NodePtr Conv3DPadding(const NodePtr &input, const NodePtr &weight, const NodePtr &bias, const NodePtr &stride,
                                const NodePtr &padding, const NodePtr &dilation, const NodePtr &groups) {
    return Emit("Conv3DPadding", {input, weight, bias, stride, padding, dilation, groups});
  }
  virtual NodePtr Ceil(const NodePtr &input) { return Emit("Ceil", {input}); }
  virtual NodePtr EluGradExt(const NodePtr &dout, const NodePtr &x_or_out, const NodePtr &alpha,
                             const NodePtr &is_result) {
    return Emit("EluGradExt", {dout, x_or_out, alpha, is_result});
  }
  virtual NodePtr TypeAs(const NodePtr &input, const NodePtr &other) { return Emit("TypeAs", {input, other}); }
  virtual NodePtr BatchNormStats(const NodePtr &input, const NodePtr &eps) {
    return Emit("BatchNormStats", {input, eps});
  }
  virtual NodePtr MaxDim(const NodePtr &input, const NodePtr &dim, const NodePtr &keepdim) {
    return Emit("MaxDim", {input, dim, keepdim});
  }
  virtual NodePtr FFNExt(const NodePtr &x, const NodePtr &weight1, const NodePtr &weight2, const NodePtr &expertTokens,
                         const NodePtr &bias1, const NodePtr &bias2, const NodePtr &scale, const NodePtr &offset,
                         const NodePtr &deqScale1, const NodePtr &deqScale2, const NodePtr &antiquant_scale1,
                         const NodePtr &antiquant_scale2, const NodePtr &antiquant_offset1,
                         const NodePtr &antiquant_offset2, const NodePtr &activation, const NodePtr &inner_precise) {
    return Emit("FFNExt",
                {x, weight1, weight2, expertTokens, bias1, bias2, scale, offset, deqScale1, deqScale2, antiquant_scale1,
                 antiquant_scale2, antiquant_offset1, antiquant_offset2, activation, inner_precise});
  }
  virtual NodePtr ConvolutionGrad(const NodePtr &dout, const NodePtr &input, const NodePtr &weight, const NodePtr &bias,
                                  const NodePtr &stride, const NodePtr &padding, const NodePtr &dilation,
                                  const NodePtr &transposed, const NodePtr &output_padding, const NodePtr &groups,
                                  const NodePtr &output_mask) {
    return Emit("ConvolutionGrad", {dout, input, weight, bias, stride, padding, dilation, transposed, output_padding,
                                    groups, output_mask});
  }
  virtual NodePtr MSELossExt(const NodePtr &input, const NodePtr &target, const NodePtr &reduction) {
    return Emit("MSELossExt", {input, target, reduction});
  }
  virtual NodePtr NLLLoss2dGrad(const NodePtr &loss_grad, const NodePtr &input, const NodePtr &target,
                                const NodePtr &weight, const NodePtr &reduction, const NodePtr &ignore_index,
                                const NodePtr &total_weight) {
    return Emit("NLLLoss2dGrad", {loss_grad, input, target, weight, reduction, ignore_index, total_weight});
  }
  virtual NodePtr ReflectionPad1DGrad(const NodePtr &grad_output, const NodePtr &input, const NodePtr &padding) {
    return Emit("ReflectionPad1DGrad", {grad_output, input, padding});
  }
  virtual NodePtr AdaptiveAvgPool3DExt(const NodePtr &input, const NodePtr &output_size) {
    return Emit("AdaptiveAvgPool3DExt", {input, output_size});
  }
  virtual NodePtr PromptFlashAttention(
    const NodePtr &query, const NodePtr &key, const NodePtr &value, const NodePtr &attn_mask,
    const NodePtr &actual_seq_lengths, const NodePtr &actual_seq_lengths_kv, const NodePtr &pse_shift,
    const NodePtr &deq_scale1, const NodePtr &quant_scale1, const NodePtr &deq_scale2, const NodePtr &quant_scale2,
    const NodePtr &quant_offset2, const NodePtr &num_heads, const NodePtr &scale_value, const NodePtr &pre_tokens,
    const NodePtr &next_tokens, const NodePtr &input_layout, const NodePtr &num_key_value_heads,
    const NodePtr &sparse_mode, const NodePtr &inner_precise) {
    return Emit("PromptFlashAttention",
                {query,       key,          value,        attn_mask,   actual_seq_lengths, actual_seq_lengths_kv,
                 pse_shift,   deq_scale1,   quant_scale1, deq_scale2,  quant_scale2,       quant_offset2,
                 num_heads,   scale_value,  pre_tokens,   next_tokens, input_layout,       num_key_value_heads,
                 sparse_mode, inner_precise});
  }
  virtual NodePtr MinDim(const NodePtr &input, const NodePtr &dim, const NodePtr &keepdim) {
    return Emit("MinDim", {input, dim, keepdim});
  }
  virtual NodePtr MatmulReduceScatter(const NodePtr &input, const NodePtr &x2, const NodePtr &group,
                                      const NodePtr &world_size, const NodePtr &reduce_op, const NodePtr &bias,
                                      const NodePtr &comm_turn, const NodePtr &trans_input, const NodePtr &trans_x2) {
    return Emit("MatmulReduceScatter",
                {input, x2, group, world_size, reduce_op, bias, comm_turn, trans_input, trans_x2});
  }
  virtual NodePtr SpeedFusionAttentionGrad(
    const NodePtr &query, const NodePtr &key, const NodePtr &value, const NodePtr &dy, const NodePtr &head_num,
    const NodePtr &input_layout, const NodePtr &pse, const NodePtr &padding_mask, const NodePtr &atten_mask,
    const NodePtr &softmax_max, const NodePtr &softmax_sum, const NodePtr &softmax_in, const NodePtr &attention_in,
    const NodePtr &scale_value, const NodePtr &keep_prob, const NodePtr &pre_tokens, const NodePtr &next_tokens,
    const NodePtr &inner_precise, const NodePtr &seed, const NodePtr &offset, const NodePtr &numels,
    const NodePtr &prefix, const NodePtr &actual_seq_qlen, const NodePtr &actual_seq_kvlen, const NodePtr &sparse_mode,
    const NodePtr &gen_mask_parallel, const NodePtr &sync, const NodePtr &pse_type, const NodePtr &q_start_idx,
    const NodePtr &kv_start_idx) {
    return Emit("SpeedFusionAttentionGrad", {query,
                                             key,
                                             value,
                                             dy,
                                             head_num,
                                             input_layout,
                                             pse,
                                             padding_mask,
                                             atten_mask,
                                             softmax_max,
                                             softmax_sum,
                                             softmax_in,
                                             attention_in,
                                             scale_value,
                                             keep_prob,
                                             pre_tokens,
                                             next_tokens,
                                             inner_precise,
                                             seed,
                                             offset,
                                             numels,
                                             prefix,
                                             actual_seq_qlen,
                                             actual_seq_kvlen,
                                             sparse_mode,
                                             gen_mask_parallel,
                                             sync,
                                             pse_type,
                                             q_start_idx,
                                             kv_start_idx});
  }
  virtual NodePtr MaskedFillScalar(const NodePtr &input, const NodePtr &mask, const NodePtr &value) {
    return Emit("MaskedFillScalar", {input, mask, value});
  }
  virtual NodePtr Atan2Ext(const NodePtr &input, const NodePtr &other) { return Emit("Atan2Ext", {input, other}); }
  virtual NodePtr DequantSwigluQuant(const NodePtr &x, const NodePtr &weight_scale, const NodePtr &activation_scale,
                                     const NodePtr &bias, const NodePtr &quant_scale, const NodePtr &quant_offset,
                                     const NodePtr &group_index, const NodePtr &activate_left,
                                     const NodePtr &quant_mode) {
    return Emit("DequantSwigluQuant", {x, weight_scale, activation_scale, bias, quant_scale, quant_offset, group_index,
                                       activate_left, quant_mode});
  }
  virtual NodePtr InplaceSiLU(const NodePtr &input) { return Emit("InplaceSiLU", {input}); }
  virtual NodePtr Var(const NodePtr &input, const NodePtr &dim, const NodePtr &correction, const NodePtr &keepdim) {
    return Emit("Var", {input, dim, correction, keepdim});
  }
  virtual NodePtr Mv(const NodePtr &input, const NodePtr &vec) { return Emit("Mv", {input, vec}); }
  virtual NodePtr AdamW(const NodePtr &var, const NodePtr &m, const NodePtr &v, const NodePtr &max_v,
                        const NodePtr &gradient, const NodePtr &step, const NodePtr &lr, const NodePtr &beta1,
                        const NodePtr &beta2, const NodePtr &decay, const NodePtr &eps, const NodePtr &amsgrad,
                        const NodePtr &maximize) {
    return Emit("AdamW", {var, m, v, max_v, gradient, step, lr, beta1, beta2, decay, eps, amsgrad, maximize});
  }
  virtual NodePtr InplaceMatmulAdd(const NodePtr &x, const NodePtr &weight, const NodePtr &C) {
    return Emit("InplaceMatmulAdd", {x, weight, C});
  }
  virtual NodePtr BincountExt(const NodePtr &input, const NodePtr &weights, const NodePtr &minlength) {
    return Emit("BincountExt", {input, weights, minlength});
  }
  virtual NodePtr SeluGrad(const NodePtr &gradient, const NodePtr &result) {
    return Emit("SeluGrad", {gradient, result});
  }
  virtual NodePtr StackExt(const NodePtr &tensors, const NodePtr &dim) { return Emit("StackExt", {tensors, dim}); }
  virtual NodePtr NormalTensorTensor(const NodePtr &mean, const NodePtr &std, const NodePtr &seed,
                                     const NodePtr &offset) {
    return Emit("NormalTensorTensor", {mean, std, seed, offset});
  }
  virtual NodePtr ReduceAll(const NodePtr &input, const NodePtr &axis, const NodePtr &keep_dims) {
    return Emit("ReduceAll", {input, axis, keep_dims});
  }
  virtual NodePtr DropoutDoMaskExt(const NodePtr &input, const NodePtr &mask, const NodePtr &p) {
    return Emit("DropoutDoMaskExt", {input, mask, p});
  }
  virtual NodePtr UpsampleBicubic2DGrad(const NodePtr &dy, const NodePtr &input_size, const NodePtr &output_size,
                                        const NodePtr &scales, const NodePtr &align_corners) {
    return Emit("UpsampleBicubic2DGrad", {dy, input_size, output_size, scales, align_corners});
  }
  virtual NodePtr AddcmulExt(const NodePtr &input, const NodePtr &tensor1, const NodePtr &tensor2,
                             const NodePtr &value) {
    return Emit("AddcmulExt", {input, tensor1, tensor2, value});
  }
  virtual NodePtr InplaceScatterValueReduce(const NodePtr &input, const NodePtr &dim, const NodePtr &index,
                                            const NodePtr &value, const NodePtr &reduce) {
    return Emit("InplaceScatterValueReduce", {input, dim, index, value, reduce});
  }
  virtual NodePtr Gcd(const NodePtr &input, const NodePtr &other) { return Emit("Gcd", {input, other}); }
  virtual NodePtr Eye(const NodePtr &n, const NodePtr &m, const NodePtr &dtype) { return Emit("Eye", {n, m, dtype}); }
  virtual NodePtr NanToNum(const NodePtr &input, const NodePtr &nan, const NodePtr &posinf, const NodePtr &neginf) {
    return Emit("NanToNum", {input, nan, posinf, neginf});
  }
  virtual NodePtr GeluExt(const NodePtr &input, const NodePtr &approximate) {
    return Emit("GeluExt", {input, approximate});
  }
  virtual NodePtr Repeat(const NodePtr &input, const NodePtr &repeats) { return Emit("Repeat", {input, repeats}); }
  virtual NodePtr Conv2DPadding(const NodePtr &input, const NodePtr &weight, const NodePtr &bias, const NodePtr &stride,
                                const NodePtr &padding, const NodePtr &dilation, const NodePtr &groups) {
    return Emit("Conv2DPadding", {input, weight, bias, stride, padding, dilation, groups});
  }
  virtual NodePtr InplaceSubScalar(const NodePtr &input, const NodePtr &other, const NodePtr &alpha) {
    return Emit("InplaceSubScalar", {input, other, alpha});
  }
  virtual NodePtr Copy(const NodePtr &input) { return Emit("Copy", {input}); }
  virtual NodePtr Zeros(const NodePtr &size, const NodePtr &dtype) { return Emit("Zeros", {size, dtype}); }
  virtual NodePtr Muls(const NodePtr &input, const NodePtr &other) { return Emit("Muls", {input, other}); }
  virtual NodePtr NLLLossGrad(const NodePtr &logits, const NodePtr &loss_grad, const NodePtr &labels,
                              const NodePtr &weight, const NodePtr &total_weight, const NodePtr &reduction,
                              const NodePtr &ignore_index) {
    return Emit("NLLLossGrad", {logits, loss_grad, labels, weight, total_weight, reduction, ignore_index});
  }
  virtual NodePtr AdaptiveAvgPool1D(const NodePtr &input, const NodePtr &output_size) {
    return Emit("AdaptiveAvgPool1D", {input, output_size});
  }
  virtual NodePtr Index(const NodePtr &input, const NodePtr &indices) { return Emit("Index", {input, indices}); }
  virtual NodePtr HardtanhGrad(const NodePtr &dout, const NodePtr &input, const NodePtr &min_val,
                               const NodePtr &max_val) {
    return Emit("HardtanhGrad", {dout, input, min_val, max_val});
  }
  virtual NodePtr RepeatInterleaveInt(const NodePtr &input, const NodePtr &repeats, const NodePtr &dim,
                                      const NodePtr &output_size) {
    return Emit("RepeatInterleaveInt", {input, repeats, dim, output_size});
  }
  virtual NodePtr Conv3DExt(const NodePtr &input, const NodePtr &weight, const NodePtr &bias, const NodePtr &stride,
                            const NodePtr &padding, const NodePtr &dilation, const NodePtr &groups) {
    return Emit("Conv3DExt", {input, weight, bias, stride, padding, dilation, groups});
  }
  virtual NodePtr Sigmoid(const NodePtr &input) { return Emit("Sigmoid", {input}); }
  virtual NodePtr Threshold(const NodePtr &input, const NodePtr &threshold, const NodePtr &value) {
    return Emit("Threshold", {input, threshold, value});
  }
  virtual NodePtr NormalFloatFloat(const NodePtr &mean, const NodePtr &std, const NodePtr &size, const NodePtr &seed,
                                   const NodePtr &offset) {
    return Emit("NormalFloatFloat", {mean, std, size, seed, offset});
  }
  virtual NodePtr BatchNormExt(const NodePtr &input, const NodePtr &weight, const NodePtr &bias,
                               const NodePtr &running_mean, const NodePtr &runnning_var, const NodePtr &training,
                               const NodePtr &momentum, const NodePtr &epsilon) {
    return Emit("BatchNormExt", {input, weight, bias, running_mean, runnning_var, training, momentum, epsilon});
  }
  virtual NodePtr AsinExt(const NodePtr &input) { return Emit("AsinExt", {input}); }
  virtual NodePtr Cast(const NodePtr &input, const NodePtr &dtype) { return Emit("Cast", {input, dtype}); }
  virtual NodePtr LayerNormGradExt(const NodePtr &dy, const NodePtr &x, const NodePtr &normalized_shape,
                                   const NodePtr &mean, const NodePtr &variance, const NodePtr &gamma,
                                   const NodePtr &beta, const NodePtr &output_mask) {
    return Emit("LayerNormGradExt", {dy, x, normalized_shape, mean, variance, gamma, beta, output_mask});
  }
  virtual NodePtr KLDivGrad(const NodePtr &grad_output, const NodePtr &input, const NodePtr &target,
                            const NodePtr &reduction, const NodePtr &log_target) {
    return Emit("KLDivGrad", {grad_output, input, target, reduction, log_target});
  }
  virtual NodePtr ApplyRotaryPosEmb(const NodePtr &query, const NodePtr &key, const NodePtr &cos, const NodePtr &sin,
                                    const NodePtr &position_ids, const NodePtr &cos_format) {
    return Emit("ApplyRotaryPosEmb", {query, key, cos, sin, position_ids, cos_format});
  }
  virtual NodePtr BatchMatMul(const NodePtr &x, const NodePtr &y, const NodePtr &transpose_a,
                              const NodePtr &transpose_b) {
    return Emit("BatchMatMul", {x, y, transpose_a, transpose_b});
  }
  virtual NodePtr HSigmoid(const NodePtr &input) { return Emit("HSigmoid", {input}); }
  virtual NodePtr NonZero(const NodePtr &input) { return Emit("NonZero", {input}); }
  virtual NodePtr Meshgrid(const NodePtr &inputs, const NodePtr &indexing) {
    return Emit("Meshgrid", {inputs, indexing});
  }
  virtual NodePtr Erfinv(const NodePtr &input) { return Emit("Erfinv", {input}); }
  virtual NodePtr MaxPoolGradWithMask(const NodePtr &x, const NodePtr &grad, const NodePtr &mask,
                                      const NodePtr &kernel_size, const NodePtr &strides, const NodePtr &pads,
                                      const NodePtr &dilation, const NodePtr &ceil_mode, const NodePtr &argmax_type) {
    return Emit("MaxPoolGradWithMask", {x, grad, mask, kernel_size, strides, pads, dilation, ceil_mode, argmax_type});
  }
  virtual NodePtr UniformExt(const NodePtr &tensor, const NodePtr &a, const NodePtr &b, const NodePtr &seed,
                             const NodePtr &offset) {
    return Emit("UniformExt", {tensor, a, b, seed, offset});
  }
  virtual NodePtr GridSampler2D(const NodePtr &input_x, const NodePtr &grid, const NodePtr &interpolation_mode,
                                const NodePtr &padding_mode, const NodePtr &align_corners) {
    return Emit("GridSampler2D", {input_x, grid, interpolation_mode, padding_mode, align_corners});
  }
  virtual NodePtr RemainderTensorTensor(const NodePtr &input, const NodePtr &other) {
    return Emit("RemainderTensorTensor", {input, other});
  }
  virtual NodePtr Dense(const NodePtr &input, const NodePtr &weight, const NodePtr &bias) {
    return Emit("Dense", {input, weight, bias});
  }
  virtual NodePtr SeLUExt(const NodePtr &input) { return Emit("SeLUExt", {input}); }
  virtual NodePtr AsinhExt(const NodePtr &input) { return Emit("AsinhExt", {input}); }
  virtual NodePtr AcosExt(const NodePtr &input) { return Emit("AcosExt", {input}); }
  virtual NodePtr SoftMarginLoss(const NodePtr &input, const NodePtr &target, const NodePtr &reduction) {
    return Emit("SoftMarginLoss", {input, target, reduction});
  }
  virtual NodePtr ChunkView(const NodePtr &input, const NodePtr &chunks, const NodePtr &dim) {
    return Emit("ChunkView", {input, chunks, dim});
  }
  virtual NodePtr InplaceMuls(const NodePtr &input, const NodePtr &other) {
    return Emit("InplaceMuls", {input, other});
  }
  virtual NodePtr HSwish(const NodePtr &input) { return Emit("HSwish", {input}); }
  virtual NodePtr TExt(const NodePtr &input) { return Emit("TExt", {input}); }
  virtual NodePtr UpsampleBilinear2D(const NodePtr &x, const NodePtr &output_size, const NodePtr &scales,
                                     const NodePtr &align_corners) {
    return Emit("UpsampleBilinear2D", {x, output_size, scales, align_corners});
  }
  virtual NodePtr Cross(const NodePtr &input, const NodePtr &other, const NodePtr &dim) {
    return Emit("Cross", {input, other, dim});
  }
  virtual NodePtr SumExt(const NodePtr &input, const NodePtr &dim, const NodePtr &keepdim, const NodePtr &dtype) {
    return Emit("SumExt", {input, dim, keepdim, dtype});
  }
  virtual NodePtr InplacePut(const NodePtr &input, const NodePtr &index, const NodePtr &source,
                             const NodePtr &accumulate) {
    return Emit("InplacePut", {input, index, source, accumulate});
  }
  virtual NodePtr SliceExt(const NodePtr &input, const NodePtr &dim, const NodePtr &start, const NodePtr &end,
                           const NodePtr &step) {
    return Emit("SliceExt", {input, dim, start, end, step});
  }
  virtual NodePtr ScatterValue(const NodePtr &input, const NodePtr &dim, const NodePtr &index, const NodePtr &src,
                               const NodePtr &reduce) {
    return Emit("ScatterValue", {input, dim, index, src, reduce});
  }
  virtual NodePtr ReverseV2(const NodePtr &input, const NodePtr &axis) { return Emit("ReverseV2", {input, axis}); }
  virtual NodePtr UpsampleNearest2DGrad(const NodePtr &dy, const NodePtr &input_size, const NodePtr &output_size,
                                        const NodePtr &scales) {
    return Emit("UpsampleNearest2DGrad", {dy, input_size, output_size, scales});
  }
  virtual NodePtr Nansum(const NodePtr &input, const NodePtr &dim, const NodePtr &keepdim, const NodePtr &dtype) {
    return Emit("Nansum", {input, dim, keepdim, dtype});
  }
  virtual NodePtr UpsampleNearest1DGrad(const NodePtr &dy, const NodePtr &input_size, const NodePtr &output_size,
                                        const NodePtr &scales) {
    return Emit("UpsampleNearest1DGrad", {dy, input_size, output_size, scales});
  }
  virtual NodePtr MoeTokenPermuteGrad(const NodePtr &permuted_tokens_grad, const NodePtr &sorted_indices,
                                      const NodePtr &num_topk, const NodePtr &padded_mode) {
    return Emit("MoeTokenPermuteGrad", {permuted_tokens_grad, sorted_indices, num_topk, padded_mode});
  }
  virtual NodePtr DivMods(const NodePtr &input, const NodePtr &other, const NodePtr &rounding_mode) {
    return Emit("DivMods", {input, other, rounding_mode});
  }
  virtual NodePtr Trunc(const NodePtr &input) { return Emit("Trunc", {input}); }
  virtual NodePtr MedianDim(const NodePtr &input, const NodePtr &dim, const NodePtr &keepdim) {
    return Emit("MedianDim", {input, dim, keepdim});
  }
  virtual NodePtr Max(const NodePtr &input) { return Emit("Max", {input}); }
  virtual NodePtr MedianExt(const NodePtr &input) { return Emit("MedianExt", {input}); }
  virtual NodePtr Erfc(const NodePtr &input) { return Emit("Erfc", {input}); }
  virtual NodePtr GLU(const NodePtr &x, const NodePtr &axis) { return Emit("GLU", {x, axis}); }
  virtual NodePtr Reciprocal(const NodePtr &input) { return Emit("Reciprocal", {input}); }
  virtual NodePtr SoftShrink(const NodePtr &input, const NodePtr &lambd) { return Emit("SoftShrink", {input, lambd}); }
  virtual NodePtr InplaceRemainderTensorScalar(const NodePtr &input, const NodePtr &other) {
    return Emit("InplaceRemainderTensorScalar", {input, other});
  }
  virtual NodePtr Contiguous(const NodePtr &input) { return Emit("Contiguous", {input}); }
  virtual NodePtr ToDtype(const NodePtr &input, const NodePtr &dtype, const NodePtr &non_blocking,
                          const NodePtr &copy) {
    return Emit("ToDtype", {input, dtype, non_blocking, copy});
  }
  virtual NodePtr SplitWithSize(const NodePtr &input, const NodePtr &split_size, const NodePtr &dim) {
    return Emit("SplitWithSize", {input, split_size, dim});
  }
  virtual NodePtr MoeTokenPermute(const NodePtr &tokens, const NodePtr &indices, const NodePtr &num_out_tokens,
                                  const NodePtr &padded_mode) {
    return Emit("MoeTokenPermute", {tokens, indices, num_out_tokens, padded_mode});
  }
  virtual NodePtr AdaptiveMaxPool2D(const NodePtr &input, const NodePtr &output_size) {
    return Emit("AdaptiveMaxPool2D", {input, output_size});
  }
  virtual NodePtr ReplicationPad3D(const NodePtr &input, const NodePtr &padding) {
    return Emit("ReplicationPad3D", {input, padding});
  }
  virtual NodePtr SilentCheckV3(const NodePtr &val, const NodePtr &max, const NodePtr &avg, const NodePtr &input_grad,
                                const NodePtr &step, const NodePtr &c_thresh_l1, const NodePtr &c_thresh_l2,
                                const NodePtr &beta1, const NodePtr &npu_asd_detect) {
    return Emit("SilentCheckV3", {val, max, avg, input_grad, step, c_thresh_l1, c_thresh_l2, beta1, npu_asd_detect});
  }
  virtual NodePtr BinaryCrossEntropy(const NodePtr &input, const NodePtr &target, const NodePtr &weight,
                                     const NodePtr &reduction) {
    return Emit("BinaryCrossEntropy", {input, target, weight, reduction});
  }
  virtual NodePtr L1LossExt(const NodePtr &input, const NodePtr &target, const NodePtr &reduction) {
    return Emit("L1LossExt", {input, target, reduction});
  }
  virtual NodePtr Min(const NodePtr &input) { return Emit("Min", {input}); }
  virtual NodePtr InplaceBernoulliScalar(const NodePtr &input, const NodePtr &p, const NodePtr &seed,
                                         const NodePtr &offset) {
    return Emit("InplaceBernoulliScalar", {input, p, seed, offset});
  }
  virtual NodePtr FloorDivScalar(const NodePtr &input, const NodePtr &other) {
    return Emit("FloorDivScalar", {input, other});
  }
  virtual NodePtr FullLike(const NodePtr &input, const NodePtr &fill_value, const NodePtr &dtype) {
    return Emit("FullLike", {input, fill_value, dtype});
  }
  virtual NodePtr Empty(const NodePtr &size, const NodePtr &dtype, const NodePtr &device, const NodePtr &pin_memory) {
    return Emit("Empty", {size, dtype, device, pin_memory});
  }
  virtual NodePtr MultiScaleDeformableAttnGrad(const NodePtr &value, const NodePtr &shape, const NodePtr &offset,
                                               const NodePtr &locations_trans, const NodePtr &weight,
                                               const NodePtr &grad_output) {
    return Emit("MultiScaleDeformableAttnGrad", {value, shape, offset, locations_trans, weight, grad_output});
  }
  virtual NodePtr LogSoftmaxGrad(const NodePtr &logits, const NodePtr &grad, const NodePtr &axis) {
    return Emit("LogSoftmaxGrad", {logits, grad, axis});
  }
  virtual NodePtr RandInt(const NodePtr &low, const NodePtr &high, const NodePtr &shape, const NodePtr &seed,
                          const NodePtr &offset, const NodePtr &dtype, const NodePtr &device) {
    return Emit("RandInt", {low, high, shape, seed, offset, dtype, device});
  }
  virtual NodePtr Frac(const NodePtr &input) { return Emit("Frac", {input}); }
  virtual NodePtr ArgMaxExt(const NodePtr &input, const NodePtr &dim, const NodePtr &keepdim) {
    return Emit("ArgMaxExt", {input, dim, keepdim});
  }
  virtual NodePtr UniqueConsecutive(const NodePtr &input, const NodePtr &return_inverse, const NodePtr &return_counts,
                                    const NodePtr &dim) {
    return Emit("UniqueConsecutive", {input, return_inverse, return_counts, dim});
  }
  virtual NodePtr ReduceAny(const NodePtr &x, const NodePtr &axis, const NodePtr &keep_dims) {
    return Emit("ReduceAny", {x, axis, keep_dims});
  }
  virtual NodePtr UpsampleLinear1DGrad(const NodePtr &dy, const NodePtr &input_size, const NodePtr &output_size,
                                       const NodePtr &scales, const NodePtr &align_corners) {
    return Emit("UpsampleLinear1DGrad", {dy, input_size, output_size, scales, align_corners});
  }
  virtual NodePtr InplaceHardtanh(const NodePtr &input, const NodePtr &min_val, const NodePtr &max_val) {
    return Emit("InplaceHardtanh", {input, min_val, max_val});
  }
  virtual NodePtr IndexFillScalar(const NodePtr &input, const NodePtr &dim, const NodePtr &index,
                                  const NodePtr &value) {
    return Emit("IndexFillScalar", {input, dim, index, value});
  }
  virtual NodePtr PagedAttention(const NodePtr &query, const NodePtr &key_cache, const NodePtr &value_cache,
                                 const NodePtr &block_tables, const NodePtr &context_lens,
                                 const NodePtr &antiquant_scale, const NodePtr &antiquant_offset,
                                 const NodePtr &attn_mask, const NodePtr &q_seq_lens, const NodePtr &alibi_mask,
                                 const NodePtr &head_num, const NodePtr &scale_value, const NodePtr &kv_head_num,
                                 const NodePtr &kv_cache_quant_mode, const NodePtr &mask_mode,
                                 const NodePtr &mla_v_dim) {
    return Emit("PagedAttention", {query, key_cache, value_cache, block_tables, context_lens, antiquant_scale,
                                   antiquant_offset, attn_mask, q_seq_lens, alibi_mask, head_num, scale_value,
                                   kv_head_num, kv_cache_quant_mode, mask_mode, mla_v_dim});
  }
  virtual NodePtr PowTensorScalar(const NodePtr &input, const NodePtr &exponent) {
    return Emit("PowTensorScalar", {input, exponent});
  }
  virtual NodePtr NonZeroExt(const NodePtr &input) { return Emit("NonZeroExt", {input}); }
  virtual NodePtr SoftMarginLossGrad(const NodePtr &predict, const NodePtr &label, const NodePtr &dout,
                                     const NodePtr &reduction) {
    return Emit("SoftMarginLossGrad", {predict, label, dout, reduction});
  }
  virtual NodePtr SelectV2(const NodePtr &condition, const NodePtr &input, const NodePtr &other) {
    return Emit("SelectV2", {condition, input, other});
  }
  virtual NodePtr ReluGrad(const NodePtr &y_backprop, const NodePtr &x) { return Emit("ReluGrad", {y_backprop, x}); }
  virtual NodePtr EluExt(const NodePtr &input, const NodePtr &alpha) { return Emit("EluExt", {input, alpha}); }
  virtual NodePtr IndexSelect(const NodePtr &input, const NodePtr &dim, const NodePtr &index) {
    return Emit("IndexSelect", {input, dim, index});
  }
  virtual NodePtr Split(const NodePtr &input_x, const NodePtr &axis, const NodePtr &output_num) {
    return Emit("Split", {input_x, axis, output_num});
  }
  virtual NodePtr IndexAddExt(const NodePtr &input, const NodePtr &dim, const NodePtr &index, const NodePtr &source,
                              const NodePtr &alpha) {
    return Emit("IndexAddExt", {input, dim, index, source, alpha});
  }
  virtual NodePtr DropoutExt(const NodePtr &input, const NodePtr &p, const NodePtr &seed, const NodePtr &offset) {
    return Emit("DropoutExt", {input, p, seed, offset});
  }
  virtual NodePtr SoftplusGradExt(const NodePtr &dout, const NodePtr &x, const NodePtr &beta,
                                  const NodePtr &threshold) {
    return Emit("SoftplusGradExt", {dout, x, beta, threshold});
  }
  virtual NodePtr IsFinite(const NodePtr &input) { return Emit("IsFinite", {input}); }
  virtual NodePtr Abs(const NodePtr &input) { return Emit("Abs", {input}); }
  virtual NodePtr NLLLoss(const NodePtr &logits, const NodePtr &labels, const NodePtr &weight, const NodePtr &reduction,
                          const NodePtr &ignore_index) {
    return Emit("NLLLoss", {logits, labels, weight, reduction, ignore_index});
  }
  virtual NodePtr UpsampleTrilinear3DGrad(const NodePtr &dy, const NodePtr &input_size, const NodePtr &output_size,
                                          const NodePtr &scales, const NodePtr &align_corners) {
    return Emit("UpsampleTrilinear3DGrad", {dy, input_size, output_size, scales, align_corners});
  }
  virtual NodePtr RmsNormGrad(const NodePtr &dy, const NodePtr &x, const NodePtr &rstd, const NodePtr &gamma) {
    return Emit("RmsNormGrad", {dy, x, rstd, gamma});
  }
  virtual NodePtr LeakyReLUGradExt(const NodePtr &dy, const NodePtr &input, const NodePtr &negative_slope,
                                   const NodePtr &is_result) {
    return Emit("LeakyReLUGradExt", {dy, input, negative_slope, is_result});
  }
  virtual NodePtr LogSumExp(const NodePtr &input, const NodePtr &dim, const NodePtr &keepdim) {
    return Emit("LogSumExp", {input, dim, keepdim});
  }
  virtual NodePtr Erf(const NodePtr &input) { return Emit("Erf", {input}); }
  virtual NodePtr SilentCheckV2(const NodePtr &val, const NodePtr &input_grad, const NodePtr &sfda, const NodePtr &step,
                                const NodePtr &c_min_steps, const NodePtr &c_thresh_l1, const NodePtr &c_coeff_l1,
                                const NodePtr &c_thresh_l2, const NodePtr &c_coeff_l2, const NodePtr &npu_asd_detect) {
    return Emit("SilentCheckV2", {val, input_grad, sfda, step, c_min_steps, c_thresh_l1, c_coeff_l1, c_thresh_l2,
                                  c_coeff_l2, npu_asd_detect});
  }
  virtual NodePtr InplaceScatterSrc(const NodePtr &input, const NodePtr &dim, const NodePtr &index,
                                    const NodePtr &src) {
    return Emit("InplaceScatterSrc", {input, dim, index, src});
  }
  virtual NodePtr BitwiseAndScalar(const NodePtr &input, const NodePtr &other) {
    return Emit("BitwiseAndScalar", {input, other});
  }
  virtual NodePtr MSELossGradExt(const NodePtr &dout, const NodePtr &x, const NodePtr &target,
                                 const NodePtr &reduction) {
    return Emit("MSELossGradExt", {dout, x, target, reduction});
  }
  virtual NodePtr UpsampleLinear1D(const NodePtr &x, const NodePtr &output_size, const NodePtr &scales,
                                   const NodePtr &align_corners) {
    return Emit("UpsampleLinear1D", {x, output_size, scales, align_corners});
  }
  virtual NodePtr ReduceMin(const NodePtr &x, const NodePtr &axis, const NodePtr &keep_dims) {
    return Emit("ReduceMin", {x, axis, keep_dims});
  }
  virtual NodePtr LogicalNot(const NodePtr &input) { return Emit("LogicalNot", {input}); }
  virtual NodePtr SoftShrinkGrad(const NodePtr &input_grad, const NodePtr &input_x, const NodePtr &lambd) {
    return Emit("SoftShrinkGrad", {input_grad, input_x, lambd});
  }
  virtual NodePtr CrossEntropyLossGrad(const NodePtr &grad_loss, const NodePtr &log_prob, const NodePtr &target,
                                       const NodePtr &weight, const NodePtr &grad_zloss, const NodePtr &lse_for_zloss,
                                       const NodePtr &reduction, const NodePtr &ignore_index,
                                       const NodePtr &label_smoothing, const NodePtr &lse_square_scale_for_zloss) {
    return Emit("CrossEntropyLossGrad", {grad_loss, log_prob, target, weight, grad_zloss, lse_for_zloss, reduction,
                                         ignore_index, label_smoothing, lse_square_scale_for_zloss});
  }
  virtual NodePtr MatMul(const NodePtr &input, const NodePtr &mat2, const NodePtr &transpose_a,
                         const NodePtr &transpose_b) {
    return Emit("MatMul", {input, mat2, transpose_a, transpose_b});
  }
  virtual NodePtr Triu(const NodePtr &input, const NodePtr &diagonal) { return Emit("Triu", {input, diagonal}); }
  virtual NodePtr Lerp(const NodePtr &input, const NodePtr &end, const NodePtr &weight) {
    return Emit("Lerp", {input, end, weight});
  }
  virtual NodePtr ReplicationPad2DGrad(const NodePtr &grad_output, const NodePtr &input, const NodePtr &padding) {
    return Emit("ReplicationPad2DGrad", {grad_output, input, padding});
  }
  virtual NodePtr InplaceDivs(const NodePtr &input, const NodePtr &other) {
    return Emit("InplaceDivs", {input, other});
  }
  virtual NodePtr Im2ColExt(const NodePtr &input, const NodePtr &kernel_size, const NodePtr &dilation,
                            const NodePtr &padding, const NodePtr &stride) {
    return Emit("Im2ColExt", {input, kernel_size, dilation, padding, stride});
  }
  virtual NodePtr DiagExt(const NodePtr &input, const NodePtr &diagonal) { return Emit("DiagExt", {input, diagonal}); }
  virtual NodePtr InplaceFillTensor(const NodePtr &input, const NodePtr &value) {
    return Emit("InplaceFillTensor", {input, value});
  }
  virtual NodePtr NewFull(const NodePtr &input, const NodePtr &size, const NodePtr &fill_value, const NodePtr &dtype) {
    return Emit("NewFull", {input, size, fill_value, dtype});
  }
  virtual NodePtr PReLU(const NodePtr &input, const NodePtr &weight) { return Emit("PReLU", {input, weight}); }
  virtual NodePtr IndexFillTensor(const NodePtr &input, const NodePtr &dim, const NodePtr &index,
                                  const NodePtr &value) {
    return Emit("IndexFillTensor", {input, dim, index, value});
  }
  virtual NodePtr ConvTranspose2D(const NodePtr &input, const NodePtr &weight, const NodePtr &bias,
                                  const NodePtr &stride, const NodePtr &padding, const NodePtr &output_padding,
                                  const NodePtr &groups, const NodePtr &dilation) {
    return Emit("ConvTranspose2D", {input, weight, bias, stride, padding, output_padding, groups, dilation});
  }
  virtual NodePtr InplaceRemainderTensorTensor(const NodePtr &input, const NodePtr &other) {
    return Emit("InplaceRemainderTensorTensor", {input, other});
  }
  virtual NodePtr Sinc(const NodePtr &input) { return Emit("Sinc", {input}); }
  virtual NodePtr InplaceAddsExt(const NodePtr &input, const NodePtr &other, const NodePtr &alpha) {
    return Emit("InplaceAddsExt", {input, other, alpha});
  }
  virtual NodePtr Tan(const NodePtr &input) { return Emit("Tan", {input}); }
  virtual NodePtr UpsampleNearest1D(const NodePtr &x, const NodePtr &output_size, const NodePtr &scales) {
    return Emit("UpsampleNearest1D", {x, output_size, scales});
  }
  virtual NodePtr MoeDistributeCombine(
    const NodePtr &expand_x, const NodePtr &expert_ids, const NodePtr &expand_idx, const NodePtr &ep_send_counts,
    const NodePtr &expert_scales, const NodePtr &ep_world_size, const NodePtr &ep_rank_id,
    const NodePtr &moe_expert_num, const NodePtr &tp_send_counts, const NodePtr &x_active_mask,
    const NodePtr &activate_scale, const NodePtr &weight_scale, const NodePtr &group_list, const NodePtr &expand_scales,
    const NodePtr &group_ep, const NodePtr &group_tp, const NodePtr &tp_world_size, const NodePtr &tp_rank_id,
    const NodePtr &expert_shard_type, const NodePtr &shared_expert_num, const NodePtr &shared_export_rank_num,
    const NodePtr &global_bs, const NodePtr &out_dtype, const NodePtr &common_quant_mode,
    const NodePtr &group_list_type) {
    return Emit("MoeDistributeCombine", {expand_x,          expert_ids,        expand_idx,
                                         ep_send_counts,    expert_scales,     ep_world_size,
                                         ep_rank_id,        moe_expert_num,    tp_send_counts,
                                         x_active_mask,     activate_scale,    weight_scale,
                                         group_list,        expand_scales,     group_ep,
                                         group_tp,          tp_world_size,     tp_rank_id,
                                         expert_shard_type, shared_expert_num, shared_export_rank_num,
                                         global_bs,         out_dtype,         common_quant_mode,
                                         group_list_type});
  }
  virtual NodePtr ConstantPadND(const NodePtr &input, const NodePtr &padding, const NodePtr &value) {
    return Emit("ConstantPadND", {input, padding, value});
  }
  virtual NodePtr UpsampleNearest3D(const NodePtr &x, const NodePtr &output_size, const NodePtr &scales) {
    return Emit("UpsampleNearest3D", {x, output_size, scales});
  }
  virtual NodePtr Rsqrt(const NodePtr &input) { return Emit("Rsqrt", {input}); }
  virtual NodePtr RingAttentionUpdate(const NodePtr &prev_attn_out, const NodePtr &prev_softmax_max,
                                      const NodePtr &prev_softmax_sum, const NodePtr &cur_attn_out,
                                      const NodePtr &cur_softmax_max, const NodePtr &cur_softmax_sum,
                                      const NodePtr &actual_seq_qlen, const NodePtr &layout) {
    return Emit("RingAttentionUpdate", {prev_attn_out, prev_softmax_max, prev_softmax_sum, cur_attn_out,
                                        cur_softmax_max, cur_softmax_sum, actual_seq_qlen, layout});
  }
  virtual NodePtr InplaceMaskedFillScalar(const NodePtr &input, const NodePtr &mask, const NodePtr &value) {
    return Emit("InplaceMaskedFillScalar", {input, mask, value});
  }
  virtual NodePtr NewEmpty(const NodePtr &input, const NodePtr &size, const NodePtr &dtype, const NodePtr &device) {
    return Emit("NewEmpty", {input, size, dtype, device});
  }
  virtual NodePtr CrossEntropyLoss(const NodePtr &input, const NodePtr &target, const NodePtr &weight,
                                   const NodePtr &reduction, const NodePtr &ignore_index,
                                   const NodePtr &label_smoothing, const NodePtr &lse_square_scale_for_zloss,
                                   const NodePtr &return_zloss) {
    return Emit("CrossEntropyLoss", {input, target, weight, reduction, ignore_index, label_smoothing,
                                     lse_square_scale_for_zloss, return_zloss});
  }
  virtual NodePtr AddScalar(const NodePtr &input, const NodePtr &other, const NodePtr &alpha) {
    return Emit("AddScalar", {input, other, alpha});
  }
  virtual NodePtr UpsampleBilinear2DGrad(const NodePtr &dy, const NodePtr &input_size, const NodePtr &output_size,
                                         const NodePtr &scales, const NodePtr &align_corners) {
    return Emit("UpsampleBilinear2DGrad", {dy, input_size, output_size, scales, align_corners});
  }
  virtual NodePtr Floor(const NodePtr &input) { return Emit("Floor", {input}); }
  virtual NodePtr Mla(const NodePtr &query, const NodePtr &q_rope, const NodePtr &kv_cache, const NodePtr &k_rope,
                      const NodePtr &block_tables, const NodePtr &attn_mask, const NodePtr &deq_scale_qk,
                      const NodePtr &deq_scale_pv, const NodePtr &q_seq_lens, const NodePtr &context_lens,
                      const NodePtr &head_num, const NodePtr &scale_value, const NodePtr &kv_head_num,
                      const NodePtr &mask_mode, const NodePtr &is_ring) {
    return Emit("Mla", {query, q_rope, kv_cache, k_rope, block_tables, attn_mask, deq_scale_qk, deq_scale_pv,
                        q_seq_lens, context_lens, head_num, scale_value, kv_head_num, mask_mode, is_ring});
  }
  virtual NodePtr MaskedSelect(const NodePtr &input, const NodePtr &mask) {
    return Emit("MaskedSelect", {input, mask});
  }
  virtual NodePtr NarrowView(const NodePtr &input, const NodePtr &dim, const NodePtr &start, const NodePtr &length) {
    return Emit("NarrowView", {input, dim, start, length});
  }
  virtual NodePtr Sinh(const NodePtr &input) { return Emit("Sinh", {input}); }
  virtual NodePtr Conv1DPadding(const NodePtr &input, const NodePtr &weight, const NodePtr &bias, const NodePtr &stride,
                                const NodePtr &padding, const NodePtr &dilation, const NodePtr &groups) {
    return Emit("Conv1DPadding", {input, weight, bias, stride, padding, dilation, groups});
  }
  virtual NodePtr QuantMatmul(const NodePtr &x1, const NodePtr &x2, const NodePtr &scale, const NodePtr &offset,
                              const NodePtr &pertoken_scale, const NodePtr &bias, const NodePtr &output_dtype,
                              const NodePtr &x1_dtype, const NodePtr &x2_dtype, const NodePtr &pertoken_scale_dtype,
                              const NodePtr &scale_dtype, const NodePtr &group_sizes) {
    return Emit("QuantMatmul", {x1, x2, scale, offset, pertoken_scale, bias, output_dtype, x1_dtype, x2_dtype,
                                pertoken_scale_dtype, scale_dtype, group_sizes});
  }
  virtual NodePtr AddRmsNormQuantV2(const NodePtr &x1, const NodePtr &x2, const NodePtr &gamma, const NodePtr &scale,
                                    const NodePtr &offset, const NodePtr &epsilon) {
    return Emit("AddRmsNormQuantV2", {x1, x2, gamma, scale, offset, epsilon});
  }
  virtual NodePtr MoeInitRouting(const NodePtr &x, const NodePtr &row_idx, const NodePtr &expert_idx,
                                 const NodePtr &active_num) {
    return Emit("MoeInitRouting", {x, row_idx, expert_idx, active_num});
  }
  virtual NodePtr FusedInferAttentionScore(
    const NodePtr &query, const NodePtr &key, const NodePtr &value, const NodePtr &pse_shift, const NodePtr &attn_mask,
    const NodePtr &actual_seq_lengths, const NodePtr &actual_seq_lengths_kv, const NodePtr &dequant_scale1,
    const NodePtr &quant_scale1, const NodePtr &dequant_scale2, const NodePtr &quant_scale2,
    const NodePtr &quant_offset2, const NodePtr &antiquant_scale, const NodePtr &antiquant_offset,
    const NodePtr &block_table, const NodePtr &query_padding_size, const NodePtr &kv_padding_size,
    const NodePtr &key_antiquant_scale, const NodePtr &key_antiquant_offset, const NodePtr &value_antiquant_scale,
    const NodePtr &value_antiquant_offset, const NodePtr &key_shared_prefix, const NodePtr &value_shared_prefix,
    const NodePtr &actual_shared_prefix_len, const NodePtr &num_heads, const NodePtr &scale_value,
    const NodePtr &pre_tokens, const NodePtr &next_tokens, const NodePtr &input_layout,
    const NodePtr &num_key_value_heads, const NodePtr &sparse_mode, const NodePtr &inner_precise,
    const NodePtr &block_size, const NodePtr &antiquant_mode, const NodePtr &softmax_lse_flag,
    const NodePtr &key_antiquant_mode, const NodePtr &value_antiquant_mode) {
    return Emit("FusedInferAttentionScore", {query,
                                             key,
                                             value,
                                             pse_shift,
                                             attn_mask,
                                             actual_seq_lengths,
                                             actual_seq_lengths_kv,
                                             dequant_scale1,
                                             quant_scale1,
                                             dequant_scale2,
                                             quant_scale2,
                                             quant_offset2,
                                             antiquant_scale,
                                             antiquant_offset,
                                             block_table,
                                             query_padding_size,
                                             kv_padding_size,
                                             key_antiquant_scale,
                                             key_antiquant_offset,
                                             value_antiquant_scale,
                                             value_antiquant_offset,
                                             key_shared_prefix,
                                             value_shared_prefix,
                                             actual_shared_prefix_len,
                                             num_heads,
                                             scale_value,
                                             pre_tokens,
                                             next_tokens,
                                             input_layout,
                                             num_key_value_heads,
                                             sparse_mode,
                                             inner_precise,
                                             block_size,
                                             antiquant_mode,
                                             softmax_lse_flag,
                                             key_antiquant_mode,
                                             value_antiquant_mode});
  }
  virtual NodePtr GroupedMatmulV4(const NodePtr &x, const NodePtr &weight, const NodePtr &bias, const NodePtr &scale,
                                  const NodePtr &offset, const NodePtr &antiquant_scale,
                                  const NodePtr &antiquant_offset, const NodePtr &pre_token_scale,
                                  const NodePtr &group_list, const NodePtr &activation_input,
                                  const NodePtr &activation_quant_scale, const NodePtr &activation_quant_offset,
                                  const NodePtr &split_item, const NodePtr &group_type, const NodePtr &group_list_type,
                                  const NodePtr &act_type, const NodePtr &output_dtype) {
    return Emit("GroupedMatmulV4", {x, weight, bias, scale, offset, antiquant_scale, antiquant_offset, pre_token_scale,
                                    group_list, activation_input, activation_quant_scale, activation_quant_offset,
                                    split_item, group_type, group_list_type, act_type, output_dtype});
  }
  virtual NodePtr QuantBatchMatmul(const NodePtr &x1, const NodePtr &x2, const NodePtr &scale, const NodePtr &offset,
                                   const NodePtr &bias, const NodePtr &pertokenScaleOptional,
                                   const NodePtr &transpose_x1, const NodePtr &transpose_x2, const NodePtr &dtype) {
    return Emit("QuantBatchMatmul",
                {x1, x2, scale, offset, bias, pertokenScaleOptional, transpose_x1, transpose_x2, dtype});
  }
  virtual NodePtr MoeComputeExpertTokens(const NodePtr &sorted_experts, const NodePtr &num_expert) {
    return Emit("MoeComputeExpertTokens", {sorted_experts, num_expert});
  }
  virtual NodePtr GroupedMatmul(const NodePtr &x, const NodePtr &weight, const NodePtr &bias, const NodePtr &scale,
                                const NodePtr &offset, const NodePtr &antiquant_scale, const NodePtr &antiquant_offset,
                                const NodePtr &group_list, const NodePtr &split_item, const NodePtr &group_type,
                                const NodePtr &transpose_a, const NodePtr &transpose_b) {
    return Emit("GroupedMatmul", {x, weight, bias, scale, offset, antiquant_scale, antiquant_offset, group_list,
                                  split_item, group_type, transpose_a, transpose_b});
  }
  virtual NodePtr WeightQuantBatchMatmul(const NodePtr &x, const NodePtr &weight, const NodePtr &antiquant_scale,
                                         const NodePtr &antiquant_offset, const NodePtr &quant_scale,
                                         const NodePtr &quant_offset, const NodePtr &bias, const NodePtr &transpose_x,
                                         const NodePtr &transpose_weight, const NodePtr &antiquant_group_size) {
    return Emit("WeightQuantBatchMatmul", {x, weight, antiquant_scale, antiquant_offset, quant_scale, quant_offset,
                                           bias, transpose_x, transpose_weight, antiquant_group_size});
  }
  virtual NodePtr MatmulAllReduceAddRmsNorm(const NodePtr &x1, const NodePtr &x2, const NodePtr &bias,
                                            const NodePtr &residual, const NodePtr &gamma, const NodePtr &epsilon,
                                            const NodePtr &group, const NodePtr &reduce_op, const NodePtr &comm_turn,
                                            const NodePtr &stream_mode) {
    return Emit("MatmulAllReduceAddRmsNorm",
                {x1, x2, bias, residual, gamma, epsilon, group, reduce_op, comm_turn, stream_mode});
  }
  virtual NodePtr GroupedMatmulV2(const NodePtr &x, const NodePtr &weight, const NodePtr &bias, const NodePtr &scale,
                                  const NodePtr &offset, const NodePtr &antiquant_scale,
                                  const NodePtr &antiquant_offset, const NodePtr &group_list, const NodePtr &split_item,
                                  const NodePtr &group_type) {
    return Emit("GroupedMatmulV2", {x, weight, bias, scale, offset, antiquant_scale, antiquant_offset, group_list,
                                    split_item, group_type});
  }
  virtual NodePtr QuantV2(const NodePtr &x, const NodePtr &scale, const NodePtr &offset, const NodePtr &sqrt_mode,
                          const NodePtr &rounding_mode, const NodePtr &dst_type) {
    return Emit("QuantV2", {x, scale, offset, sqrt_mode, rounding_mode, dst_type});
  }
  virtual NodePtr MoeInitRoutingV2(const NodePtr &x, const NodePtr &expert_idx, const NodePtr &active_num,
                                   const NodePtr &expert_capacity, const NodePtr &expert_num,
                                   const NodePtr &drop_pad_mode, const NodePtr &expert_tokens_count_or_cumsum_flag,
                                   const NodePtr &expert_tokens_before_capacity_flag) {
    return Emit("MoeInitRoutingV2", {x, expert_idx, active_num, expert_capacity, expert_num, drop_pad_mode,
                                     expert_tokens_count_or_cumsum_flag, expert_tokens_before_capacity_flag});
  }
  virtual NodePtr MoeGatingTopKSoftmax(const NodePtr &x, const NodePtr &finished, const NodePtr &k) {
    return Emit("MoeGatingTopKSoftmax", {x, finished, k});
  }
  virtual NodePtr MoeFinalizeRouting(const NodePtr &expanded_x, const NodePtr &x1, const NodePtr &x2,
                                     const NodePtr &bias, const NodePtr &scales, const NodePtr &expanded_row_idx,
                                     const NodePtr &expanded_expert_idx) {
    return Emit("MoeFinalizeRouting", {expanded_x, x1, x2, bias, scales, expanded_row_idx, expanded_expert_idx});
  }
  virtual NodePtr MoeInitRoutingQuantV2(const NodePtr &x, const NodePtr &expert_idx, const NodePtr &active_num,
                                        const NodePtr &expert_capacity, const NodePtr &expert_num,
                                        const NodePtr &drop_pad_mode, const NodePtr &expert_tokens_count_or_cumsum_flag,
                                        const NodePtr &expert_tokens_before_capacity_flag, const NodePtr &quant_mode,
                                        const NodePtr &scale, const NodePtr &offset) {
    return Emit("MoeInitRoutingQuantV2",
                {x, expert_idx, active_num, expert_capacity, expert_num, drop_pad_mode,
                 expert_tokens_count_or_cumsum_flag, expert_tokens_before_capacity_flag, quant_mode, scale, offset});
  }
  virtual NodePtr DynamicQuantExt(const NodePtr &x, const NodePtr &smooth_scales) {
    return Emit("DynamicQuantExt", {x, smooth_scales});
  }
  virtual NodePtr KVCacheScatterUpdate(const NodePtr &var, const NodePtr &indices, const NodePtr &updates,
                                       const NodePtr &axis, const NodePtr &reduce) {
    return Emit("KVCacheScatterUpdate", {var, indices, updates, axis, reduce});
  }
  virtual NodePtr FuncDropoutExt(const NodePtr &input, const NodePtr &p, const NodePtr &training,
                                 const NodePtr &inplace, const NodePtr &seed, const NodePtr &offset) {
    return Emit("FuncDropoutExt", {input, p, training, inplace, seed, offset});
  }
  virtual NodePtr GmmBackward(const NodePtr &grad, const NodePtr &x, const NodePtr &weight, const NodePtr &group_list,
                              const NodePtr &group_list_type) {
    return Emit("GmmBackward", {grad, x, weight, group_list, group_list_type});
  }
  virtual NodePtr PixelShuffle(const NodePtr &input, const NodePtr &upscale_factor) {
    return Emit("PixelShuffle", {input, upscale_factor});
  }
  virtual NodePtr GmmV2BackwardFusion(const NodePtr &grad, const NodePtr &weight, const NodePtr &group_list,
                                      const NodePtr &group_list_type) {
    return Emit("GmmV2BackwardFusion", {grad, weight, group_list, group_list_type});
  }
  virtual NodePtr AnyExt(const NodePtr &input, const NodePtr &dim, const NodePtr &keepdim) {
    return Emit("AnyExt", {input, dim, keepdim});
  }
  virtual NodePtr GmmV2Backward(const NodePtr &grad, const NodePtr &x, const NodePtr &weight, const NodePtr &group_list,
                                const NodePtr &group_list_type) {
    return Emit("GmmV2Backward", {grad, x, weight, group_list, group_list_type});
  }
  virtual NodePtr FuncMaxPool2D(const NodePtr &input, const NodePtr &kernel_size, const NodePtr &stride,
                                const NodePtr &padding, const NodePtr &dilation, const NodePtr &ceil_mode,
                                const NodePtr &return_indices) {
    return Emit("FuncMaxPool2D", {input, kernel_size, stride, padding, dilation, ceil_mode, return_indices});
  }
  virtual NodePtr MoeTokenUnpermute(const NodePtr &permuted_tokens, const NodePtr &sorted_indices, const NodePtr &probs,
                                    const NodePtr &padded_mode, const NodePtr &restore_shape) {
    return Emit("MoeTokenUnpermute", {permuted_tokens, sorted_indices, probs, padded_mode, restore_shape});
  }
  virtual NodePtr Any(const NodePtr &input) { return Emit("Any", {input}); }
  virtual NodePtr InplaceExponential(const NodePtr &input, const NodePtr &lambd, const NodePtr &seed,
                                     const NodePtr &offset) {
    return Emit("InplaceExponential", {input, lambd, seed, offset});
  }
  virtual NodePtr Dropout2dExt(const NodePtr &input, const NodePtr &p, const NodePtr &training, const NodePtr &inplace,
                               const NodePtr &seed, const NodePtr &offset) {
    return Emit("Dropout2dExt", {input, p, training, inplace, seed, offset});
  }
  virtual NodePtr GmmBackwardFusion(const NodePtr &grad, const NodePtr &weight, const NodePtr &group_list,
                                    const NodePtr &group_list_type) {
    return Emit("GmmBackwardFusion", {grad, weight, group_list, group_list_type});
  }
  virtual NodePtr Gmm(const NodePtr &x, const NodePtr &weight, const NodePtr &bias, const NodePtr &group_list,
                      const NodePtr &group_type, const NodePtr &group_list_type) {
    return Emit("Gmm", {x, weight, bias, group_list, group_type, group_list_type});
  }
  virtual NodePtr EinsumExt(const NodePtr &equation, const NodePtr &operands) {
    return Emit("EinsumExt", {equation, operands});
  }
  virtual NodePtr GmmV2(const NodePtr &x, const NodePtr &weight, const NodePtr &bias, const NodePtr &group_list,
                        const NodePtr &group_type, const NodePtr &group_list_type) {
    return Emit("GmmV2", {x, weight, bias, group_list, group_type, group_list_type});
  }

  virtual NodePtr InnerCommAllReduce(const NodePtr &grad, const NodePtr &op, const NodePtr &group) {
    return Emit("InnerCommAllReduce", {grad, op, group});
  }

  virtual NodePtr InnerCommAllGather(const NodePtr &grad, const NodePtr &rank_size, const NodePtr &group) {
    return Emit("InnerCommAllGather", {grad, rank_size, group});
  }

  virtual NodePtr InnerCommReduceScatter(const NodePtr &grad, const NodePtr &rank_size, const NodePtr &type,
                                         const NodePtr &group) {
    return Emit("InnerCommReduceScatter", {grad, rank_size, type, group});
  }

  virtual NodePtr InnerCommIsend(const NodePtr &grad, const NodePtr &rank_size, const NodePtr &group,
                                 const NodePtr &tag) {
    return Emit("InnerCommIsend", {grad, rank_size, group, tag});
  }

  virtual NodePtr InnerCommIrecv(const NodePtr &tag, const NodePtr &rank_size, const NodePtr &shape,
                                 const NodePtr &group, const NodePtr &type) {
    return Emit("InnerCommIrecv", {tag, rank_size, shape, group, type});
  }

  virtual NodePtr InnerCommAllToAllV(const NodePtr &grad, const NodePtr &group, const NodePtr &send_numel_list,
                                     const NodePtr &recv_numel_list, const NodePtr &rank_size,
                                     const NodePtr &split_sizes_empty) {
    return Emit("InnerCommAllToAllV", {grad, group, send_numel_list, recv_numel_list, rank_size, split_sizes_empty});
  }

  template <typename Target, typename T>
  NodePtr ConvertAndEmit(const T &value) {
    return EmitValue(MakeValue<Target>(static_cast<Target>(value)));
  }

  template <typename T>
  NodePtr CreateNode(const T &v, const TypeId &type_id) {
    switch (type_id) {
      case TypeId::kNumberTypeFloat16:
        return ConvertAndEmit<float16>(v);
      case TypeId::kNumberTypeBFloat16:
        return ConvertAndEmit<bfloat16>(v);
      case TypeId::kNumberTypeFloat32:
      case TypeId::kNumberTypeFloat:
        return ConvertAndEmit<float>(v);
      case TypeId::kNumberTypeFloat64:
      case TypeId::kNumberTypeDouble:
        return ConvertAndEmit<double>(v);
      case TypeId::kNumberTypeUInt8:
        return ConvertAndEmit<uint8_t>(v);
      case TypeId::kNumberTypeUInt16:
        return ConvertAndEmit<uint16_t>(v);
      case TypeId::kNumberTypeUInt32:
        return ConvertAndEmit<uint32_t>(v);
      case TypeId::kNumberTypeUInt64:
        return ConvertAndEmit<uint64_t>(v);
      case TypeId::kNumberTypeInt8:
        return ConvertAndEmit<int8_t>(v);
      case TypeId::kNumberTypeInt16:
        return ConvertAndEmit<int16_t>(v);
      case TypeId::kNumberTypeInt:
      case TypeId::kNumberTypeInt32:
        return ConvertAndEmit<int32_t>(v);
      case TypeId::kNumberTypeInt64:
        return ConvertAndEmit<int64_t>(v);
      case TypeId::kNumberTypeComplex64:
      case TypeId::kNumberTypeComplex:
        return ConvertAndEmit<std::complex<float>>(v);
      case TypeId::kNumberTypeComplex128:
        return ConvertAndEmit<std::complex<double>>(v);
      default:
        MS_LOG(INFO) << "Unknown Tensor type: " << type_id << ". Using default type";
        return EmitValue(MakeValue(v));
    }
  }

  /// \brief Emit a value node by type
  template <typename T>
  NodePtr ValueByType(const T &value, TypePtr data_type) {
    MS_EXCEPTION_IF_NULL(data_type);
    return CreateNode(value, data_type->type_id());
  }

 protected:
  virtual NodePtr EmitOp(const PrimitivePtr &prim, const NodePtrList &inputs);
  NodePtr CmpOpWithCast(const std::string &op, const NodePtr &lhs, const NodePtr &rhs, const TypePtr &dst_type) {
    auto node = UnifyDtypeAndEmit(op, lhs, rhs);
    return dst_type == nullptr ? node : Cast(node, dst_type);
  }
  /// \brief Convert two tensors to the same dtype
  std::tuple<NodePtr, NodePtr> UnifyDtype(const NodePtr &lhs, const NodePtr &rhs);
  NodePtr UnifyDtypeAndEmit(const std::string &op, const NodePtr &a, const NodePtr &b, const DAttr &attrs = {}) {
    auto [lhs, rhs] = UnifyDtype(a, b);
    return Emit(op, {lhs, rhs}, attrs);
  }

  ExpanderInferPtr infer_{nullptr};
  ScopePtr scope_{nullptr};
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
#endif  // MINDSPORE_CCSRC_UTILS_EXPANDER_EMITTER_H_
