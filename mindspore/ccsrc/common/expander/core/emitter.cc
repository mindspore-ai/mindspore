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

#include "include/common/expander/core/emitter.h"

#include <algorithm>
#include <functional>
#include <unordered_set>
#include <utility>
#include "include/common/utils/utils.h"
#include "ir/anf.h"
#include "ops/sequence_ops.h"
#include "ops/math_ops.h"
#include "ops/array_ops.h"
#include "ops/framework_ops.h"
#include "include/common/utils/convert_utils.h"
#include "ir/functor.h"
#include "ops/primitive_c.h"
#include "utils/anf_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/ms_context.h"
#include "ops/op_def.h"
#include "ir/primitive.h"

namespace mindspore {
namespace expander {
namespace {
std::pair<bool, std::vector<int64_t>> GetIntList(const NodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  ValuePtr value_ptr = node->BuildValue();
  if (value_ptr != nullptr) {
    if ((value_ptr->isa<ValueSequence>() && !value_ptr->ContainsValueAny()) || value_ptr->isa<Scalar>()) {
      return std::make_pair(true, CheckAndConvertUtils::CheckIntOrTupleInt("value", value_ptr, "GetIntList"));
    }
    if (value_ptr->isa<tensor::Tensor>()) {
      auto tensor = value_ptr->cast<tensor::TensorPtr>();
      MS_EXCEPTION_IF_NULL(tensor);
      // In pynative mode, need data sync before get tensor value, otherwise the tensor value may be undefined.
      tensor->data_sync();
      return std::make_pair(true, CheckAndConvertUtils::CheckTensorIntValue("value", value_ptr, "GetIntList"));
    }
  }
  return std::make_pair(false, std::vector<int64_t>{});
}

ValuePtr CreateZeroScalar(const TypePtr &type) {
  auto tensor = std::make_shared<tensor::Tensor>(0, type);
  return CreateValueFromTensor(tensor);
}

ShapeVector CalReshapeRealDstShape(const ShapeVector &x_shape, const ShapeVector &dst_shape) {
  if (!IsDynamicShape(dst_shape)) {
    return dst_shape;
  }

  if (IsDynamicRank(dst_shape) || IsDynamic(x_shape)) {
    MS_LOG(EXCEPTION) << "The source shape(" << x_shape << ") or target shape(" << dst_shape
                      << ") is invalid for Reshape const infer!";
  }

  ShapeVector res_shape(dst_shape.begin(), dst_shape.end());
  if (std::count(dst_shape.begin(), dst_shape.end(), abstract::Shape::kShapeDimAny) != 1) {
    MS_LOG(EXCEPTION) << "The target shape can only have one -1 for Reshape, bug got " << dst_shape;
  }

  auto total_size = std::accumulate(x_shape.cbegin(), x_shape.cend(), 1, std::multiplies<int64_t>());
  size_t target_idx = 0;
  int64_t dst_size = 1;
  for (size_t i = 0; i < dst_shape.size(); ++i) {
    if (dst_shape[i] == abstract::Shape::kShapeDimAny) {
      target_idx = i;
      continue;
    }
    dst_size *= dst_shape[i];
  }
  MS_EXCEPTION_IF_CHECK_FAIL(dst_size != 0, "Cannot divide zeros!");
  res_shape[target_idx] = total_size / dst_size;
  return res_shape;
}
}  // namespace

NodePtr Emitter::Emit(const std::string &op_name, const NodePtrList &inputs, const DAttr &attrs) {
  auto prim = NewPrimitive(op_name, attrs);
  return EmitOp(prim, inputs);
}

NodePtr Emitter::EmitOp(const PrimitivePtr &prim, const NodePtrList &inputs) {
  MS_EXCEPTION(NotImplementedError) << "Base Emitter not implemented EmitOp() method";
}

PrimitivePtr Emitter::NewPrimitive(const std::string &op_name, const DAttr &attrs) {
  PrimitivePtr prim = nullptr;
  if (mindspore::ops::IsPrimitiveFunction(op_name)) {
    prim = std::make_shared<Primitive>(op_name);
  } else {
    auto &func = Emitter::primc_func_cache()[op_name];
    if (func == nullptr) {
      const auto &op_primc_fns = ops::OpPrimCRegister::GetInstance().GetPrimCMap();
      const auto iter = op_primc_fns.find(op_name);
      prim = iter == op_primc_fns.end() ? std::make_shared<ops::PrimitiveC>(op_name) : (func = iter->second)();
    } else {
      prim = func();
    }
  }
  MS_EXCEPTION_IF_NULL(prim);
  if (!attrs.empty()) {
    (void)prim->SetAttrs(attrs);
  }
  return prim;
}

NodePtr Emitter::EmitValue(const ValuePtr &value) {
  MS_EXCEPTION(NotImplementedError) << "Base Emitter not implemented EmitValue() method";
}

NodePtr Emitter::Exp(const NodePtr &x) {
  return Emit(kExpOpName, {x},
              {{"base", MakeValue<float>(-1.0)}, {"scale", MakeValue<float>(1.0)}, {"shift", MakeValue<float>(0.0)}});
}

NodePtr Emitter::Log(const NodePtr &x) {
  return Emit(kLogOpName, {x},
              {{"base", MakeValue<pyfloat>(-1.0)},
               {"scale", MakeValue<pyfloat>(1.0)},
               {"shift", MakeValue<pyfloat>(0.0)},
               {"cust_aicpu", MakeValue(kLogOpName)}});
}

NodePtr Emitter::Cast(const NodePtr &node, const TypePtr &type) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(type);
  // do not emit a node when the dst type is the same as src type
  if (node->dtype()->type_id() == type->type_id()) {
    return node;
  }
  return Emit("Cast", {node, Value(static_cast<int64_t>(type->type_id()))});
}

NodePtr Emitter::Reshape(const NodePtr &node, const NodePtr &shape) {
  MS_EXCEPTION_IF_NULL(node);
  auto [success, dst_shape] = GetIntList(shape);
  if (!success) {
    auto tuple_shape = TensorToTuple(shape);
    return Emit(kReshapeOpName, {node, tuple_shape});
  }

  if (node->input_type() == InputType::kConstant) {
    // If node and shape is both known, return node itself or a new tensor with target shape.
    auto value = node->BuildValue();
    MS_EXCEPTION_IF_NULL(value);
    auto tensor = value->cast<tensor::TensorPtr>();
    if (tensor != nullptr && tensor->data().const_data() != nullptr) {
      const auto &tensor_shape = tensor->shape_c();
      auto update_shape = CalReshapeRealDstShape(tensor_shape, dst_shape);
      if (tensor_shape == update_shape) {
        return node;
      }
      auto type_id = tensor->data_type();
      return this->Tensor(type_id, update_shape, tensor->data_c(), type_id);
    }
  }

  auto node_shape = node->shape();
  if (IsDynamicRank(node_shape)) {
    return Emit(kReshapeOpName, {node, shape});
  }
  if (dst_shape.size() != node_shape.size()) {
    return Emit(kReshapeOpName, {node, shape});
  }
  for (size_t i = 0; i < dst_shape.size(); ++i) {
    if (dst_shape[i] != node_shape[i] && dst_shape[i] != -1) {
      return Emit(kReshapeOpName, {node, shape});
    }
  }
  return node;
}

NodePtr Emitter::MatMul(const NodePtr &a, const NodePtr &b, bool transpose_a, bool transpose_b) {
  return Emit(prim::kPrimMatMul->name(), {a, b, Value(transpose_a), Value(transpose_b)});
}

NodePtr Emitter::BatchMatMul(const NodePtr &a, const NodePtr &b, bool transpose_a, bool transpose_b) {
  return Emit(prim::kPrimBatchMatMul->name(), {a, b, Value(transpose_a), Value(transpose_b)});
}

NodePtr Emitter::MatMulExt(const NodePtr &a, const NodePtr &b) {
  return UnifyDtypeAndEmit(prim::kPrimMatMulExt->name(), a, b, {});
}

NodePtr Emitter::Transpose(const NodePtr &node, const NodePtr &perm) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(perm);
  auto [success, perm_list] = GetIntList(perm);
  if (!success) {
    auto tuple_perm = TensorToTuple(perm);
    return Emit(kTransposeOpName, {node, tuple_perm});
  }
  // perm like [0, 1, 2, 3] does not need transpose.
  auto n = SizeToLong(perm_list.size());
  for (size_t i = 0; i < perm_list.size(); ++i) {
    // perm value may be negative, e.g. [0, -3, 2, 3] is equal to [0, 1, 2, 3]
    auto perm_i = perm_list[i] < 0 ? (perm_list[i] + n) : perm_list[i];
    if (perm_i != static_cast<int64_t>(i)) {
      return Emit(kTransposeOpName, {node, perm});
    }
  }
  return node;
}

NodePtr Emitter::Tile(const NodePtr &node, const NodePtr &dims) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(dims);
  auto [success, multiples_list] = GetIntList(dims);
  if (!success) {
    auto tuple_multiples = TensorToTuple(dims);
    return Emit(kTileOpName, {node, tuple_multiples});
  }
  bool is_all_one = std::all_of(multiples_list.begin(), multiples_list.end(), [](int64_t shp) { return shp == 1; });
  if (is_all_one && node->shape().size() >= multiples_list.size()) {
    return node;
  }
  return Emit(kTileOpName, {node, dims});
}

NodePtr Emitter::BroadcastTo(const NodePtr &x, const NodePtr &y) {
  if (IsDynamic(x->shape()) || IsDynamic(y->shape())) {
    return Emit("BroadcastTo", {x, Shape(y)});
  }

  return x->shape() == y->shape() ? x : Emit("BroadcastTo", {x, Shape(y)});
}

NodePtr Emitter::ZerosLike(const NodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (node->input_type() == InputType::kConstant) {
    if (node->dtype()->type_id() == kMetaTypeNone) {
      return Tensor(0);
    }
    auto v = node->BuildValue();
    MS_EXCEPTION_IF_NULL(v);
    if (v->isa<ValueSequence>()) {
      return Emit(kSequenceZerosLikeOpName, {node});
    } else if (v->isa<Scalar>()) {
      return EmitValue(CreateZeroScalar(v->type()));
    } else if (v->isa<Type>()) {
      return Tensor(0, v->type());
    } else if (v->isa<Monad>()) {
      return Tensor(0);
    }
  }

  auto abs = node->abstract();
  MS_EXCEPTION_IF_NULL(abs);

  if (abs->isa<abstract::AbstractTensor>()) {
    return Emit(kZerosLikeOpName, {node});
  } else if (abs->isa<abstract::AbstractMonad>() || abs->isa<abstract::AbstractType>() ||
             abs->isa<abstract::AbstractNone>()) {
    return node;
  } else if (abs->isa<abstract::AbstractSequence>()) {
    auto sequence_abs = abs->cast<abstract::AbstractSequencePtr>();
    if (!sequence_abs->dynamic_len() && sequence_abs->empty()) {
      return node;
    }
    return Emit(kSequenceZerosLikeOpName, {node});
  } else if (abs->isa<abstract::AbstractScalar>()) {
    auto value = CreateZeroScalar(abs->BuildType());
    return EmitValue(value);
  }

  MS_LOG(EXCEPTION) << "Cannot emit ZerosLike for " << node->ToString() << " with abstract " << abs;
}

NodePtr Emitter::Fill(double value, const ShapeVector &shape, TypeId data_type) {
  size_t data_num = LongToSize(std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int64_t>()));
  std::vector<double> data(data_num, value);
  return Tensor(data_type, shape, &data[0], TypeId::kNumberTypeFloat64);
}

NodePtr Emitter::Fill(int64_t value, const ShapeVector &shape, TypeId data_type) {
  size_t data_num = LongToSize(std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int64_t>()));
  std::vector<int64_t> data(data_num, value);
  return Tensor(data_type, shape, &data[0], TypeId::kNumberTypeInt64);
}

NodePtr Emitter::ScalarToTensor(const NodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto value = node->BuildValue();
  MS_EXCEPTION_IF_NULL(value);
  auto scalar = value->cast<ScalarPtr>();
  MS_EXCEPTION_IF_NULL(scalar);
  auto tensor = mindspore::ScalarToTensor(scalar);
  return EmitValue(tensor);
}

NodePtr Emitter::ScalarToTensor(const NodePtr &node, const TypePtr &dtype) {
  MS_EXCEPTION_IF_NULL(node);
  auto value = node->BuildValue();
  MS_EXCEPTION_IF_NULL(value);
  auto scalar = value->cast<ScalarPtr>();
  if (scalar == nullptr) {
    return Emit("ScalarToTensor", {node, Value(static_cast<int64_t>(dtype->type_id()))});
  }
  MS_EXCEPTION_IF_NULL(scalar);
  auto tensor = mindspore::ScalarToTensor(scalar);
  return EmitValue(tensor);
}

NodePtr Emitter::BoolNot(const NodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto abs = node->abstract();
  MS_EXCEPTION_IF_NULL(abs);
  NodePtr new_node{nullptr};
  if (abs->isa<abstract::AbstractScalar>()) {
    auto value_ptr = node->BuildValue();
    MS_EXCEPTION_IF_NULL(value_ptr);
    if (!(value_ptr->isa<ValueAny>() || value_ptr->isa<None>())) {
      auto value = GetValue<bool>(value_ptr);
      new_node = Value(static_cast<bool>(!value));
    } else {
      new_node = Emit("BoolNot", {node});
    }
  } else {
    MS_LOG(EXCEPTION) << "BooNot only support scalar, but got " << abs->ToString();
  }
  return new_node;
}

std::pair<bool, ShapeVector> Emitter::NeedReduce(const ShapeVector &shape, const std::vector<int64_t> &axis,
                                                 bool keep_dim, bool skip_mode) const {
  if (IsDynamic(shape)) {
    return std::make_pair(true, shape);
  }
  if (shape.empty() || (skip_mode && axis.empty())) {
    return std::make_pair(false, shape);
  }
  auto rank = SizeToLong(shape.size());
  std::vector<bool> axis_map;
  if (axis.empty()) {
    axis_map = std::vector<bool>(shape.size(), true);
  } else {
    axis_map = std::vector<bool>(shape.size(), false);
    for (size_t i = 0; i < axis.size(); ++i) {
      if (axis[i] < -rank || axis[i] >= rank) {
        MS_EXCEPTION(ValueError) << "Reduce axis[" << i << "] is " << axis[i] << ", which is out of range [-" << rank
                                 << ", " << rank << ") for shape: " << shape;
      }
      auto axis_i = axis[i] < 0 ? axis[i] + rank : axis[i];
      axis_map[LongToSize(axis_i)] = true;
    }
  }
  // Calc reduce output shape
  ShapeVector out_shape;
  bool need_reduce = false;
  for (size_t i = 0; i < shape.size(); ++i) {
    if (!axis_map[i]) {
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

std::pair<bool, NodePtr> Emitter::NeedReduce(const NodePtr &shape, const NodePtr &axis, bool keep_dim, bool skip_mode) {
  auto [axis_success, axis_value] = GetIntList(axis);
  if (axis_success && skip_mode && axis_value.empty()) {
    return std::make_pair(false, shape);
  }
  auto [shape_success, shape_value] = GetIntList(shape);
  if (shape_success && axis_success) {
    auto [need_reduce, shape_vec] = NeedReduce(shape_value, axis_value, keep_dim, skip_mode);
    return std::make_pair(need_reduce, Value(shape_vec));
  }

  auto v = Value(ShapeVector{});
  return std::make_pair(true, v);
}

NodePtr Emitter::ReduceSum(const NodePtr &x, const NodePtr &axis, bool keep_dims, bool skip_mode) {
  MS_EXCEPTION_IF_NULL(x);
  MS_EXCEPTION_IF_NULL(axis);
  auto need_reduce = NeedReduce(Shape(x), axis, keep_dims, skip_mode);
  if (!need_reduce.first) {
    return Reshape(x, need_reduce.second);
  }
  auto tuple_axis = TensorToTuple(axis);
  return Emit(prim::kPrimReduceSum->name(), {x, tuple_axis, Value(keep_dims), Value(skip_mode)});
}

NodePtr Emitter::ReduceSum(const NodePtr &x, const ShapeVector &axis, bool keep_dims) {
  MS_EXCEPTION_IF_NULL(x);
  auto real_axis = axis;
#ifdef WITH_BACKEND
  const auto &shape = x->shape();
  if (real_axis.empty()) {
    if (IsDynamicRank(shape)) {
      MS_LOG(DEBUG) << "For ReduceSum, it may wrong with a empty axis for dynamic rank case.";
    } else {
      for (int64_t i = 0; i < SizeToLong(shape.size()); i++) {
        real_axis.push_back(i);
      }
    }
  }
#endif
  return ReduceSum(x, Value<ShapeVector>(real_axis), keep_dims, false);
}

NodePtr Emitter::SumExt(const NodePtr &input, const NodePtr &axis, const NodePtr &keep_dims) {
  MS_EXCEPTION_IF_NULL(input);
  MS_EXCEPTION_IF_NULL(axis);
  MS_EXCEPTION_IF_NULL(keep_dims);
  auto input_dtype_id = input->dtype()->type_id();
  auto dtype = Value(static_cast<int64_t>(input_dtype_id));
  MS_EXCEPTION_IF_NULL(dtype);

  return SumExt(input, axis, keep_dims, dtype);
}

NodePtr Emitter::Gather(const NodePtr &params, const NodePtr &indices, const NodePtr &axis, int64_t batch_dims) {
  MS_EXCEPTION_IF_NULL(params);
  MS_EXCEPTION_IF_NULL(indices);
  MS_EXCEPTION_IF_NULL(axis);
  return Emit(kGatherOpName, {params, indices, axis, Value(batch_dims)});
}
NodePtr Emitter::Gather(const NodePtr &params, const NodePtr &indices, int64_t axis, int64_t batch_dims) {
  return Gather(params, indices, Value(axis), batch_dims);
}

std::tuple<bool, ShapeArray, std::vector<std::vector<size_t>>> GetConstInputs(
  const NodePtrList &inputs, const std::vector<bool> &only_depend_shape, const ShapeValidFunc &valid_func) {
  bool all_const = true;
  ShapeArray const_args;
  std::vector<std::vector<size_t>> pos_idx;

  for (size_t i = 0; i < inputs.size(); ++i) {
    MS_EXCEPTION_IF_NULL(inputs[i]);

    if (!only_depend_shape[i]) {
      // input[i]'s value is used
      auto [success, vec] = GetIntList(inputs[i]);
      if (!success) {
        all_const = false;
        break;
      }
      pos_idx.push_back({const_args.size()});
      const_args.push_back(vec);
    } else {
      // input[i]'s shape is used
      auto abs = inputs[i]->abstract();
      MS_EXCEPTION_IF_NULL(abs);

      if (auto sequence_abs = abs->cast<abstract::AbstractSequencePtr>(); sequence_abs != nullptr) {
        auto begin_idx = const_args.size();
        auto is_const = ops::TryGetShapeArg(sequence_abs, &const_args, &pos_idx);

        if (is_const) {
          for (size_t j = begin_idx; j < const_args.size(); ++j) {
            is_const = valid_func ? valid_func(j, const_args[j]) : !IsDynamic(const_args[j]);
            if (!is_const) {
              break;
            }
          }
        }

        if (!is_const) {
          all_const = false;
          break;
        }
      } else {
        auto input_shape = inputs[i]->shape();
        auto input_valid = valid_func ? valid_func(i, input_shape) : !IsDynamic(input_shape);
        if (!input_valid) {
          all_const = false;
          break;
        }
        pos_idx.push_back({const_args.size()});
        const_args.push_back(input_shape);
      }
    }
  }

  return std::make_tuple(all_const, const_args, pos_idx);
}

NodePtrList Emitter::ShapeCalc(const ShapeCalcBaseFunctorPtr &functor, const NodePtrList &inputs,
                               const std::vector<int64_t> &value_depend, const ShapeValidFunc &valid_func) {
  std::vector<bool> only_depend_shape(inputs.size(), true);
  for (auto idx : value_depend) {
    only_depend_shape[LongToSize(idx)] = false;
  }

  bool all_const;
  ShapeArray const_args;
  std::vector<std::vector<size_t>> pos_idx;
  // Try to get all const input shapes or values, and call the shape calc function when success.
  std::tie(all_const, const_args, pos_idx) = GetConstInputs(inputs, only_depend_shape, valid_func);
  NodePtrList res;
  // all inputs are static-shape tensors,
  if (all_const) {
    auto out = functor->Calc(const_args, pos_idx);
    res.reserve(out.size());
    (void)std::transform(out.begin(), out.end(), std::back_inserter(res),
                         [this](const ShapeVector &sh) { return Value(sh); });
    return res;
  }

  auto out = Emit(kShapeCalcOpName, inputs,
                  {{kAttrFunctor, functor},
                   {kAttrOnlyDependShape, MakeValue(only_depend_shape)},
                   {kAttrInputIsDynamicShape, MakeValue(true)}});
  MS_EXCEPTION_IF_NULL(out);
  auto abs = out->abstract();
  MS_EXCEPTION_IF_NULL(abs);
  auto tuple_abs = abs->cast<abstract::AbstractTuplePtr>();
  MS_EXCEPTION_IF_NULL(tuple_abs);
  if (!tuple_abs->dynamic_len() && tuple_abs->size() != 0 && tuple_abs->elements()[0]->isa<abstract::AbstractTuple>()) {
    res.reserve(tuple_abs->size());
    for (size_t i = 0; i < tuple_abs->size(); ++i) {
      res.push_back(TupleGetItem(out, i));
    }
  } else {
    res.push_back(out);
  }
  return res;
}

NodePtr Emitter::TensorToTuple(const NodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto abs = node->abstract();
  MS_EXCEPTION_IF_NULL(abs);
  if (abs->isa<abstract::AbstractTensor>()) {
    auto [success, value] = GetIntList(node);
    if (success) {
      return EmitValue(MakeValue(value));
    }
    return Emit(kTensorToTupleOpName, {node});
  }
  if (!abs->isa<abstract::AbstractTuple>()) {
    MS_LOG(INTERNAL_EXCEPTION) << "A Tensor or tuple is expected, but got " << abs->BuildType()->ToString() << ".";
  }
  return node;
}

std::tuple<NodePtr, NodePtr> Emitter::UnifyDtype2(const NodePtr &lhs, const NodePtr &rhs) {
  auto it1 = type_vector_[lhs->dtype()->type_id()];
  auto it2 = type_vector_[rhs->dtype()->type_id()];
  if (!it1 || !it2 || it1 == it2) {
    return {lhs, rhs};
  }
  if (it1 < it2) {
    return {this->Cast(lhs, rhs->dtype()), rhs};
  }
  return {lhs, this->Cast(rhs, lhs->dtype())};
}

NodePtr Emitter::SparseSoftmaxCrossEntropyWithLogits(const NodePtrList &inputs, const DAttr &attrs, const NodePtr &out,
                                                     const NodePtr &dout, bool is_graph_mode) {
  auto grad = Emit("SparseSoftmaxCrossEntropyWithLogits", inputs, attrs);
  if (is_graph_mode) {
    grad = Depend(grad, out);
  }
  grad = Mul(grad, dout);
  return grad;
}

NodePtr Emitter::Conditional(const NodePtr &cond, const BlockFunc &true_case, const BlockFunc &false_case) {
  MS_EXCEPTION(NotImplementedError) << "Base Emitter not implement Conditional() method";
}

NodePtr Emitter::While(const NodePtr &cond, const BlockFunc &body, const NodePtrList &init_list) {
  MS_EXCEPTION(NotImplementedError) << "Base Emitter not implement While() method";
}

NodePtr CtrlFlowBlock::IfThenElse(const NodePtr &cond, const BlockFunc &true_case, const BlockFunc &false_case) {
  auto tb = BuildSubgraph(true_case);
  auto fb = BuildSubgraph(false_case);
  auto s = emitter_->Emit("Switch", {cond, tb, fb});

  auto cnode = func_graph_->FuncGraph::NewCNode({s->get()});
  cnode->set_abstract(out_abstract_);
  auto node = emitter_->NewIrNode(cnode->cast<AnfNodePtr>());
  return node;
}

NodePtr CtrlFlowBlock::While(const NodePtr &cond, const BlockFunc &while_body_func, const NodePtrList &init_list) {
  auto while_fg = std::make_shared<FuncGraph>();
  MS_EXCEPTION_IF_NULL(while_fg);
  auto cond_cnode = cond->get()->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cond_cnode);

  cond_cnode->set_func_graph(while_fg);
  auto while_fg_emitter = CreateInnerEmitter(while_fg, std::make_shared<CppInferWithPartial>());
  MS_EXCEPTION_IF_NULL(while_fg_emitter);
  AnfNodePtrList main_while_fg_inputs = {NewValueNode(while_fg)};
  std::map<AnfNodePtr, ParameterPtr> param_map;
  auto replace_by_param = [&main_while_fg_inputs, &param_map, &while_fg](const AnfNodePtr &inp) {
    auto &param = param_map[inp];
    if (param == nullptr) {
      param = while_fg->add_parameter();
      param->set_abstract(inp->abstract());
      (void)main_while_fg_inputs.emplace_back(inp);
    }
    return param;
  };

  auto empty_body_func = [&init_list](Emitter *) { return init_list; };
  auto empty_body_fg_with_inputs = BuildSubgraphOfPartial(empty_body_func);
  for (size_t i = 1; i < empty_body_fg_with_inputs.size(); i++) {
    auto inp = empty_body_fg_with_inputs[i]->get();
    empty_body_fg_with_inputs[i] = while_fg_emitter->NewIrNode(replace_by_param(inp));
  }
  for (size_t i = 1; i < cond_cnode->size(); i++) {
    auto inp = cond_cnode->input(i);
    MS_EXCEPTION_IF_NULL(inp);
    if (!inp->isa<ValueNode>()) {
      cond_cnode->set_input(i, replace_by_param(inp));
    }
  }

  auto body_with_inputs = BuildSubgraphOfPartial(while_body_func);
  auto body_fg = body_with_inputs[0]->get()->cast<ValueNodePtr>()->value()->cast<FuncGraphPtr>();
  for (size_t i = 1; i < body_with_inputs.size(); i++) {
    body_with_inputs[i] = while_fg_emitter->NewIrNode(replace_by_param(body_with_inputs[i]->get()));
  }
  // replace the body's output to call the outside while-fg
  AnfNodePtrList body_while_fg_inputs{NewValueNode(while_fg)};
  if (IsPrimitiveCNode(body_fg->output(), prim::kPrimMakeTuple)) {
    auto mt = body_fg->output()->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(mt);
    (void)body_while_fg_inputs.insert(body_while_fg_inputs.end(), mt->inputs().begin() + 1, mt->inputs().end());
  } else {
    body_while_fg_inputs.push_back(body_fg->output());
  }
  if (body_while_fg_inputs.size() - 1 != init_list.size()) {
    MS_LOG(EXCEPTION) << "The while body's output size should be equal to init_list.size(), but got "
                      << (body_while_fg_inputs.size() - 1) << " vs " << init_list.size();
  }
  if (body_while_fg_inputs.size() < main_while_fg_inputs.size()) {
    for (size_t i = body_while_fg_inputs.size(); i < main_while_fg_inputs.size(); i++) {
      auto inp = while_fg->parameters()[i - 1];
      auto iter = std::find_if(body_with_inputs.begin(), body_with_inputs.end(),
                               [&inp](const NodePtr &no) { return no->get() == inp; });
      if (iter != body_with_inputs.end()) {
        auto param_idx = iter - body_with_inputs.begin() - 1;
        body_while_fg_inputs.push_back(body_fg->parameters()[LongToSize(param_idx)]);
      } else {
        body_with_inputs.push_back(while_fg_emitter->NewIrNode(inp));
        auto p = body_fg->add_parameter();
        p->set_abstract(inp->abstract());
        body_while_fg_inputs.push_back(p);
      }
    }
  }
  auto body_call_fg = body_fg->NewCNode(body_while_fg_inputs);
  body_call_fg->set_abstract(out_abstract_);
  body_fg->set_output(body_call_fg);

  auto tb = while_fg_emitter->Emit("Partial", body_with_inputs);
  auto fb = while_fg_emitter->Emit("Partial", empty_body_fg_with_inputs);
  auto s = while_fg_emitter->Emit("Switch", {cond, tb, fb});
  auto cnode = while_fg->NewCNode({s->get()});
  cnode->set_abstract(out_abstract_);
  while_fg->set_output(cnode);

  auto main_cnode = func_graph_->FuncGraph::NewCNode(main_while_fg_inputs);
  main_cnode->set_abstract(out_abstract_);
  return emitter_->NewIrNode(main_cnode);
}

EmitterPtr CtrlFlowBlock::CreateInnerEmitter(const FuncGraphPtr &fg, const ExpanderInferPtr &infer) const {
  return emitter_creator_ ? emitter_creator_(fg, infer) : std::make_shared<IrEmitter>(fg, infer);
}

NodePtr CtrlFlowBlock::BuildSubgraph(const BlockFunc &func) {
  auto fg = std::make_shared<FuncGraph>();
  MS_EXCEPTION_IF_NULL(fg);
  fg->set_indirect(std::make_shared<bool>(true));
  auto e = CreateInnerEmitter(fg, emitter_->infer());
  MS_EXCEPTION_IF_NULL(e);
  auto outputs = func(e.get());
  if (outputs.empty()) {
    MS_LOG(EXCEPTION) << "The block function should not return empty list.";
  }
  if (output_num_ == 0) {
    output_num_ = outputs.size();
  } else if (output_num_ != outputs.size()) {
    MS_LOG(EXCEPTION) << "The count of outputs of each block function should be equal, but got " << output_num_
                      << " vs " << outputs.size() << ".";
  }
  NodePtr output;
  if (output_num_ > 1) {
    output = e->MakeTuple(outputs);
    SetSequenceNodeElementsUseFlags(output->get(), std::make_shared<std::vector<bool>>(output_num_, true));
  } else {
    output = outputs[0];
  }
  fg->set_output(output->get());
  if (out_abstract_ == nullptr) {
    out_abstract_ = output->abstract();
  }
  return emitter_->Value(fg);
}

NodePtrList CtrlFlowBlock::BuildSubgraphOfPartial(const BlockFunc &func) {
  auto fg = std::make_shared<FuncGraph>();
  MS_EXCEPTION_IF_NULL(fg);
  fg->set_indirect(std::make_shared<bool>(true));
  auto sub_emitter = CreateInnerEmitter(fg, emitter_->infer());
  MS_EXCEPTION_IF_NULL(sub_emitter);
  auto output = func(sub_emitter.get());
  if (output.empty()) {
    MS_LOG(EXCEPTION) << "The block function should not return empty list.";
  }
  if (output_num_ == 0) {
    output_num_ = output.size();
  } else if (output_num_ != output.size()) {
    MS_LOG(EXCEPTION) << "The count of outputs of each block function should be equal, but got " << output_num_
                      << " vs " << output.size() << ".";
  }
  fg->set_output((output_num_ > 1) ? sub_emitter->MakeTuple(output)->get() : output[0]->get());
  if (out_abstract_ == nullptr) {
    out_abstract_ = fg->output()->abstract();
  }
  if (output_num_ > 1) {
    SetSequenceNodeElementsUseFlags(fg->output(), std::make_shared<std::vector<bool>>(output_num_, true));
  }

  // replace the captured inputs to parameter
  std::function<void(const CNodePtr &)> dfs;
  std::unordered_set<AnfNodePtr> visited;
  std::map<AnfNodePtr, ParameterPtr> param_map;
  NodePtrList fg_with_inputs = {emitter_->Value(fg)};
  dfs = [&visited, &dfs, &fg, &param_map, &fg_with_inputs, this](const CNodePtr &node) {
    (void)visited.insert(node);
    for (size_t i = 0; i < node->size(); i++) {
      auto inp = node->input(i);
      if (inp->func_graph() == nullptr) {
        continue;
      }
      if (inp->func_graph() == fg) {
        if (inp->isa<CNode>() && visited.count(inp) == 0) {
          dfs(inp->cast<CNodePtr>());
        }
      } else {
        auto &param = param_map[inp];
        if (param == nullptr) {
          param = fg->add_parameter();
          param->set_abstract(inp->abstract());
          (void)fg_with_inputs.emplace_back(emitter_->NewIrNode(inp));
        }
        node->set_input(i, param);
      }
    }
  };
  dfs(fg->get_return());
  return fg_with_inputs;
}

void CtrlFlowBlock::CppInferWithPartial::Infer(const NodePtr &node) {
  if (IsPrimitiveCNode(node->get(), prim::kPrimPartial) || IsPrimitiveCNode(node->get(), prim::kPrimSwitch)) {
    return;
  }
  CppInfer::Infer(node);
}

NodePtr IrEmitter::EmitOp(const PrimitivePtr &prim, const NodePtrList &inputs) {
  AnfNodePtrList cnode_inputs = {NewValueNode(prim)};
  cnode_inputs.reserve(inputs.size() + 1);
  (void)std::transform(inputs.cbegin(), inputs.cend(), std::back_inserter(cnode_inputs), [](const NodePtr &no) {
    MS_EXCEPTION_IF_NULL(no);
    return no->get();
  });
  auto cnode = func_graph_->NewCNode(cnode_inputs);
  if (scope_ != nullptr) {
    cnode->set_scope(scope_);
  }
  auto node = NewIrNode(cnode->cast<AnfNodePtr>());
  infer_->Infer(node);
  return node;
}

NodePtr IrEmitter::EmitValue(const ValuePtr &value) {
  auto node = NewIrNode(NewValueNode(value));
  infer_->Infer(node);
  return node;
}

NodePtr operator+(const NodePtr &lhs, const NodePtr &rhs) { return lhs->emitter()->Add(lhs, rhs); }
NodePtr operator-(const NodePtr &lhs, const NodePtr &rhs) { return lhs->emitter()->Sub(lhs, rhs); }
NodePtr operator*(const NodePtr &lhs, const NodePtr &rhs) { return lhs->emitter()->Mul(lhs, rhs); }
NodePtr operator/(const NodePtr &lhs, const NodePtr &rhs) { return lhs->emitter()->RealDiv(lhs, rhs); }
NodePtr operator-(const NodePtr &node) { return node->emitter()->Neg(node); }
}  // namespace expander
}  // namespace mindspore
