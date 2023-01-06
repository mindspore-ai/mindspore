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

#include "frontend/operator/graph_bprop/ops_utils.h"
#include <algorithm>
#include <utility>
#include <map>
#include "frontend/operator/ops.h"
#include "frontend/operator/ops_front_infer_function.h"
#include "pipeline/jit/parse/resolve.h"
#include "pipeline/jit/parse/data_converter.h"
#include "utils/check_convert_utils.h"
#include "include/common/utils/utils.h"
#include "include/common/utils/convert_utils_py.h"
#include "ops/op_name.h"

namespace mindspore {
namespace graph_bprop {
namespace {
bool IsContainUndetermined(const AbstractBasePtr &arg) {
  MS_EXCEPTION_IF_NULL(arg);
  if (arg->isa<abstract::AbstractSequence>()) {
    auto seq_arg = arg->cast_ptr<abstract::AbstractSequence>();
    return std::any_of(seq_arg->elements().begin(), seq_arg->elements().end(), IsContainUndetermined);
  }

  if (arg->isa<abstract::AbstractKeywordArg>()) {
    auto kw_arg = arg->cast_ptr<abstract::AbstractKeywordArg>();
    return IsContainUndetermined(kw_arg->get_arg());
  }

  return arg->isa<abstract::AbstractUndetermined>() && arg->IsBroaden();
}

py::tuple GetParameters(const AbstractBasePtrList &args_spec_list) {
  std::size_t params_size = args_spec_list.size();
  auto params = py::tuple(params_size);
  for (size_t i = 0; i < params_size; i++) {
    const auto &arg = args_spec_list[i];
    MS_EXCEPTION_IF_NULL(arg);
    if (IsContainUndetermined(arg)) {
      MS_EXCEPTION(TypeError) << "The " << i << "th initializing input to create instance for "
                              << args_spec_list[0]->BuildValue()->ToString()
                              << " should be a constant, but got: " << arg->ToString();
    }
    ValuePtr param_value = arg->BuildValue();
    py::object param = ValueToPyData(param_value);
    params[i] = param;
  }
  return params;
}

PrimitivePtr CreatePrimInstance(const parse::ClassTypePtr &class_type, const AbstractBasePtrList &args_spec_list) {
  MS_LOG(DEBUG) << "Get class type: " << class_type->ToString() << ".";
  // Get the create instance obj's parameters, `params` may contain tuple(args, kwargs).
  py::tuple params = GetParameters(args_spec_list);
  // Create class instance.
  auto obj = parse::data_converter::CreatePythonObject(class_type->obj(), params);
  if (py::isinstance<py::none>(obj)) {
    MS_LOG(EXCEPTION) << "Create python object `" << py::str(class_type->obj())
                      << "` failed, only support to create \'Cell\', \'Primitive\' or "
                      << "user-defined Class decorated with \'jit_class\'.";
  }

  ValuePtr converted_res = nullptr;
  bool converted = parse::ConvertData(obj, &converted_res, true);
  if (!converted) {
    MS_LOG(EXCEPTION) << "Convert the python object failed";
  }
  return GetValueWithoutDoSignature(converted_res)->cast<PrimitivePtr>();
}

void GetInputsAbstractList(const std::vector<AnfNodePtr> &inputs, AbstractBasePtrList *input_abs) {
  (void)std::transform(inputs.cbegin() + 1, inputs.cend(), std::back_inserter(*input_abs), [](const AnfNodePtr &node) {
    MS_EXCEPTION_IF_NULL(node);
    const auto &abs = node->abstract();
    if (abs == nullptr) {
      MS_EXCEPTION_IF_CHECK_FAIL(node->isa<ValueNode>(),
                                 node->DebugString() + " is not a ValueNode and has not abstract.");
      return node->cast<ValueNodePtr>()->value()->ToAbstract();
    }
    return abs;
  });
}

ValuePtr Infer(const CNodePtr &cnode, bool infer_value) {
  const auto &inputs = cnode->inputs();
  if (inputs.empty()) {
    MS_LOG(EXCEPTION) << "The inputs to create new node should not be empty.";
  }
  PrimitivePtr prim;
  if (IsValueNode<Primitive>(inputs[0])) {
    prim = GetValueNode<PrimitivePtr>(inputs[0]);
  } else if (inputs[0]->isa<CNode>()) {
    auto class_type_cnode = inputs[0]->cast_ptr<CNode>();
    auto class_type = GetValueNode<parse::ClassTypePtr>(class_type_cnode->input(0));
    if (class_type == nullptr) {
      MS_LOG(EXCEPTION) << "The first input should be a ClassType node, but got "
                        << class_type_cnode->input(0)->DebugString();
    }
    AbstractBasePtrList class_type_input_abs;
    GetInputsAbstractList(class_type_cnode->inputs(), &class_type_input_abs);
    prim = CreatePrimInstance(class_type, class_type_input_abs);
    cnode->set_input(0, NewValueNode(prim));
  } else {
    MS_LOG(EXCEPTION)
      << "The first input to create new node should be a Primitive ValueNode or a ClassType CNode, but got "
      << inputs[0]->DebugString();
  }
  MS_EXCEPTION_IF_NULL(prim);
  auto eval_impl = abstract::GetFrontendPrimitiveInferImpl(prim);
  if (eval_impl.has_value()) {
    auto infer = eval_impl.value();
    MS_EXCEPTION_IF_CHECK_FAIL(infer.IsImplInferShapeAndType(), "There is no infer-abstract implement!");
    AbstractBasePtrList input_abs;
    GetInputsAbstractList(inputs, &input_abs);
    if (infer_value) {
      auto value = infer.InferValue(prim, input_abs);
      if (value != nullptr) {
        return value;
      }
    }
    auto abs = infer.InferShapeAndType(nullptr, prim, input_abs);
    cnode->set_abstract(abs);
    return nullptr;
  }
  MS_LOG(EXCEPTION) << "Could not find the infer impl for node " << cnode->DebugString();
}
}  // namespace

AnfNodePtr NewNode(const FuncGraphPtr &fg, const std::vector<AnfNodePtr> &inputs, bool need_infer, bool infer_value) {
  MS_EXCEPTION_IF_NULL(fg);
  auto new_node = fg->NewCNodeInOrder(inputs);
  if (need_infer || infer_value) {
    auto value = Infer(new_node, infer_value);
    if (value != nullptr) {
      return NewValueNode(value);
    }
  }
  return new_node;
}

AnfNodePtr Add() { return NewValueNode(prim::GetPythonOps("add", "mindspore.ops.composite.multitype_ops.add_impl")); }

AnfNodePtr Mod() { return NewValueNode(prim::GetPythonOps("mod", "mindspore.ops.composite.multitype_ops.mod_impl")); }

AnfNodePtr Mul(const FuncGraphPtr &fg) {
  return fg->NewCNodeInOrder({GetClassType("mindspore.ops.operations.math_ops", "Mul")});
}

AnfNodePtr ZerosLikeFunction(const FuncGraphPtr &fg, const AnfNodePtr &input) {
  return fg->NewCNodeInOrder(
    {NewValueNode(prim::GetPythonOps("zeros_like", "mindspore.ops.composite.multitype_ops.zeros_like_impl")), input});
}

AnfNodePtr BiasAddGrad(const string &format) {
  auto prim = NewPrimitive(prim::kPrimBiasAddGrad, {{"format", MakeValue(format)}});
  return NewValueNode(prim);
}

AnfNodePtr MatMul(const FuncGraphPtr &fg, bool transpose_a, bool transpose_b) {
  return fg->NewCNodeInOrder({GetClassType("mindspore.ops.operations.math_ops", "MatMul"), NewValueNode(transpose_a),
                              NewValueNode(transpose_b)});
}

AnfNodePtr Conj() { return NewValueNode(prim::kPrimConj); }

AnfNodePtr ReluGrad() { return NewValueNode(prim::kPrimReluGrad); }

AnfNodePtr GeLUGrad() { return NewValueNode(prim::kPrimGeLUGrad); }

AnfNodePtr MakeTuple() { return NewValueNode(prim::kPrimMakeTuple); }

AnfNodePtr TensorShape() { return NewValueNode(prim::kPrimTensorShape); }

AnfNodePtr Shape() { return NewValueNode(prim::kPrimShape); }

AnfNodePtr RowTensorGetValues() { return NewValueNode(prim::kPrimRowTensorGetValues); }

AnfNodePtr RowTensorGetIndices() { return NewValueNode(prim::kPrimRowTensorGetIndices); }

AnfNodePtr RowTensorGetDenseShape() { return NewValueNode(prim::kPrimRowTensorGetDenseShape); }

AnfNodePtr MakeRowTensor() { return NewValueNode(prim::kPrimMakeRowTensor); }

AnfNodePtr Cast(const FuncGraphPtr &fg) {
  return fg->NewCNodeInOrder({GetClassType("mindspore.ops.operations.array_ops", "Cast")});
}

AnfNodePtr ReduceProd(const FuncGraphPtr &fg) {
  return fg->NewCNodeInOrder({GetClassType("mindspore.ops.operations.math_ops", "ReduceProd")});
}

AnfNodePtr ExpandDims(const FuncGraphPtr &fg) {
  return fg->NewCNodeInOrder({GetClassType("mindspore.ops.operations.array_ops", "ExpandDims")});
}

AnfNodePtr Range(const FuncGraphPtr &fg) {
  return fg->NewCNodeInOrder({GetClassType("mindspore.ops.operations.array_ops", "Range")});
}

AnfNodePtr TensorScatterUpdate(const FuncGraphPtr &fg) {
  return fg->NewCNodeInOrder({GetClassType("mindspore.ops.operations.array_ops", "TensorScatterUpdate")});
}

AnfNodePtr InvertPermutation(const FuncGraphPtr &fg) {
  return fg->NewCNodeInOrder({GetClassType("mindspore.ops.operations.array_ops", "InvertPermutation")});
}

AnfNodePtr Transpose(const FuncGraphPtr &fg) {
  return fg->NewCNodeInOrder({GetClassType("mindspore.ops.operations.array_ops", "Transpose")});
}

AnfNodePtr ZerosLike() { return NewValueNode(prim::kPrimZerosLike); }

AnfNodePtr Neg() { return NewValueNode(prim::kPrimNeg); }

AnfNodePtr LayerNormGrad(const FuncGraphPtr &fg, const ValuePtr &begin_norm_axis, const ValuePtr &begin_params_axis) {
  return fg->NewCNodeInOrder({GetClassType("mindspore.ops.operations._grad_ops", "LayerNormGrad"),
                              NewValueNode(begin_norm_axis), NewValueNode(begin_params_axis)});
}

AnfNodePtr GetAttr(const FuncGraphPtr &fg, const AnfNodePtr &node, const std::string &attr) {
  return fg->NewCNodeInOrder({NewValueNode(prim::kPrimGetAttr), node, NewValueNode(attr)});
}

AnfNodePtr Conv2DBackpropInput(const FuncGraphPtr &fg, const PrimitivePtr &primal) {
  auto out_channel = GetAndCheckAttr(primal, ops::kOutChannel);
  auto kernel_size = GetAndCheckAttr(primal, kAttrKernelSize);
  auto pad_mode = GetPadModStr(GetAndCheckAttr(primal, kAttrPadMode));
  auto pad = GetAndCheckAttr(primal, kAttrPad);
  auto pad_list = GetAndCheckAttr(primal, "pad_list");
  auto mode = GetAndCheckAttr(primal, kAttrMode);
  auto dilation = GetAndCheckAttr(primal, kAttrDilation);
  auto stride = GetAndCheckAttr(primal, kAttrStride);
  auto group = GetAndCheckAttr(primal, kAttrGroup);
  auto format = GetAndCheckAttr(primal, kAttrFormat);
  return fg->NewCNodeInOrder({GetClassType("mindspore.ops.operations.nn_ops", "Conv2DBackpropInput"),
                              NewValueNode(out_channel), NewValueNode(kernel_size), NewValueNode(pad_mode),
                              NewValueNode(pad), NewValueNode(pad_list), NewValueNode(mode), NewValueNode(stride),
                              NewValueNode(dilation), NewValueNode(group), NewValueNode(format)});
}

AnfNodePtr Conv2DBackpropFilter(const FuncGraphPtr &fg, const PrimitivePtr &primal) {
  auto out_channel = GetAndCheckAttr(primal, ops::kOutChannel);
  auto kernel_size = GetAndCheckAttr(primal, kAttrKernelSize);
  auto pad_mode = GetPadModStr(GetAndCheckAttr(primal, kAttrPadMode));
  auto pad = GetAndCheckAttr(primal, kAttrPad);
  auto pad_list = GetAndCheckAttr(primal, "pad_list");
  auto mode = GetAndCheckAttr(primal, kAttrMode);
  auto dilation = GetAndCheckAttr(primal, kAttrDilation);
  auto stride = GetAndCheckAttr(primal, kAttrStride);
  auto group = GetAndCheckAttr(primal, kAttrGroup);
  auto format = GetAndCheckAttr(primal, kAttrFormat);
  return fg->NewCNodeInOrder({GetClassType("mindspore.ops.operations._grad_ops", "Conv2DBackpropFilter"),
                              NewValueNode(out_channel), NewValueNode(kernel_size), NewValueNode(pad_mode),
                              NewValueNode(pad), NewValueNode(pad_list), NewValueNode(mode), NewValueNode(stride),
                              NewValueNode(dilation), NewValueNode(group), NewValueNode(format)});
}

AnfNodePtr ReduceSum(const FuncGraphPtr &fg, bool keep_dims, bool skip_mode) {
  return fg->NewCNodeInOrder(
    {GetClassType("mindspore.ops.operations.math_ops", "ReduceSum"), NewValueNode(keep_dims), NewValueNode(skip_mode)});
}

AnfNodePtr Reshape(const FuncGraphPtr &fg) {
  return fg->NewCNodeInOrder({GetClassType("mindspore.ops.operations.array_ops", "Reshape")});
}

AnfNodePtr DynamicBroadcastGradientArgs() { return NewValueNode(prim::kPrimDynamicBroadcastGradientArgs); }

AnfNodePtr MaxPoolGrad(const FuncGraphPtr &fg, const PrimitivePtr &primal) {
  auto kernel_size = GetAndCheckAttr(primal, kAttrKernelSize);
  auto strides = GetAndCheckAttr(primal, kAttrStrides);
  auto pad_mode = GetPadModStr(GetAndCheckAttr(primal, kAttrPadMode), true);
  auto data_format = GetAndCheckAttr(primal, kAttrFormat);
  return fg->NewCNodeInOrder({GetClassType("mindspore.ops.operations._grad_ops", "MaxPoolGrad"),
                              NewValueNode(kernel_size), NewValueNode(strides), NewValueNode(pad_mode),
                              NewValueNode(data_format)});
}

AnfNodePtr TupleGetItem(const FuncGraphPtr &fg, const AnfNodePtr &output, int64_t idx) {
  return NewNode(fg, {NewValueNode(prim::kPrimTupleGetItem), output, NewValueNode(idx)});
}

AnfNodePtr BatchNormGrad(const FuncGraphPtr &fg, const PrimitivePtr &primal) {
  auto is_training = GetAndCheckAttr(primal, kAttrIsTraining);
  auto epsilon = GetAndCheckAttr(primal, kAttrEpsilon);
  auto data_format = GetAndCheckAttr(primal, kAttrFormat);
  return fg->NewCNodeInOrder({GetClassType("mindspore.ops.operations._grad_ops", "BatchNormGrad"),
                              NewValueNode(is_training), NewValueNode(epsilon), NewValueNode(data_format)});
}

AnfNodePtr DynSize(const FuncGraphPtr &fg, const AnfNodePtr &node, const TypePtr &dtype) {
  auto shape = NewNode(fg, {Cast(fg), NewNode(fg, {TensorShape(), node}), NewValueNode(kFloat32)});
  auto size = NewNode(fg, {Cast(fg), NewNode(fg, {ReduceProd(fg), shape}), NewValueNode(dtype)});
  return size;
}

AnfNodePtr DynInvertPermutation(const FuncGraphPtr &fg, const AnfNodePtr &perm) {
  auto indices = NewNode(fg, {ExpandDims(fg), perm, NewValueNode(static_cast<int64_t>(-1))});
  auto end = DynSize(fg, perm);
  auto end_dtype = GetAttr(fg, end, kAttrDType);
  auto cast1 = NewNode(fg, {Cast(fg), NewValueNode(0), end_dtype});
  auto cast2 = NewNode(fg, {Cast(fg), NewValueNode(1), end_dtype});
  auto updates = NewNode(fg, {Range(fg), cast1, end, cast2});
  auto output = NewNode(fg, {ZerosLike(), updates});
  auto cast3 = NewNode(fg, {Cast(fg), output, NewValueNode(kFloat32)});
  auto cast4 = NewNode(fg, {Cast(fg), updates, NewValueNode(kFloat32)});
  auto new_perm = NewNode(fg, {TensorScatterUpdate(fg), cast3, indices, cast4});
  return NewNode(fg, {Cast(fg), new_perm, NewValueNode(kInt32)});
}

AnfNodePtr ReduceSumWithCast(const FuncGraphPtr &fg, const AnfNodePtr &dx, const ShapeVector &axis) {
  auto dx_origin_dtype = GetTensorDType(dx->abstract());
  // Currently, for Ascend and GPU, the reduce_sum's input does not support int16, int32 and int64.
  if (dx_origin_dtype == kNumberTypeInt16 || dx_origin_dtype == kNumberTypeInt32 ||
      dx_origin_dtype == kNumberTypeInt64) {
    auto new_dx = NewNode(fg, {ReduceSum(fg), NewNode(fg, {Cast(fg), dx, NewValueNode(kFloat32)}), NewValueNode(axis)});
    return NewNode(fg, {Cast(fg), new_dx, NewValueNode(TypeIdToType(dx_origin_dtype))});
  }
  return NewNode(fg, {ReduceSum(fg), dx, NewValueNode(axis)});
}

AnfNodePtr DynBinopGradCommon(const FuncGraphPtr &fg, const AnfNodePtr &x, const AnfNodePtr &y, const AnfNodePtr &dx,
                              const AnfNodePtr &dy) {
  auto shape_x = NewNode(fg, {TensorShape(), x});
  auto shape_y = NewNode(fg, {TensorShape(), y});
  auto dynamic_broadcast_gradient_args = NewNode(fg, {DynamicBroadcastGradientArgs(), shape_x, shape_y});
  auto rx = TupleGetItem(fg, dynamic_broadcast_gradient_args, SizeToLong(kIndex0));
  auto ry = TupleGetItem(fg, dynamic_broadcast_gradient_args, SizeToLong(kIndex1));

  auto dx_origin_dtype = GetTensorDType(dx->abstract());
  AnfNodePtr new_dx;
  if (dx_origin_dtype == kNumberTypeInt16 || dx_origin_dtype == kNumberTypeInt32 ||
      dx_origin_dtype == kNumberTypeInt64) {
    new_dx = NewNode(fg, {Cast(fg), dx, NewValueNode(kFloat32)});
    new_dx = SumGradReduceAxis(fg, new_dx, rx);
    new_dx = NewNode(fg, {Cast(fg), new_dx, NewValueNode(TypeIdToType(dx_origin_dtype))});
  } else {
    new_dx = SumGradReduceAxis(fg, dx, rx);
  }

  auto dy_origin_dtype = GetTensorDType(dy->abstract());
  AnfNodePtr new_dy;
  if (dy_origin_dtype == kNumberTypeInt16 || dy_origin_dtype == kNumberTypeInt32 ||
      dy_origin_dtype == kNumberTypeInt64) {
    new_dy = NewNode(fg, {Cast(fg), dy, NewValueNode(kFloat32)});
    new_dy = SumGradReduceAxis(fg, new_dy, ry);
    new_dy = NewNode(fg, {Cast(fg), new_dy, NewValueNode(TypeIdToType(dy_origin_dtype))});
  } else {
    new_dy = SumGradReduceAxis(fg, dy, ry);
  }

  auto reduce_dx = NewNode(fg, {Reshape(fg), new_dx, shape_x});
  auto reduce_dy = NewNode(fg, {Reshape(fg), new_dy, shape_y});
  return NewNode(fg, {MakeTuple(), reduce_dx, reduce_dy});
}

namespace {
std::pair<abstract::AbstractTuplePtr, abstract::AbstractTuplePtr> GetBroadcastGradientArgsAbstract(
  const AnfNodePtr &node) {
  const auto &abs = node->abstract();
  MS_EXCEPTION_IF_NULL(abs);
  auto abs_tuple = abs->cast_ptr<abstract::AbstractTuple>();
  if (abs_tuple == nullptr) {
    MS_LOG(EXCEPTION) << "The abstract of node " << node->DebugString() << " should be an AbstractTuple, but got"
                      << abs->ToString();
  }
  constexpr size_t output_size = 2;
  if (abs_tuple->size() != output_size) {
    MS_LOG(EXCEPTION) << "The abstract size of abstract tuple " << abs_tuple->ToString() << " should be " << output_size
                      << ", but got " << abs_tuple->size();
  }
  auto abs0_tuple = abs_tuple->elements()[0]->cast<abstract::AbstractTuplePtr>();
  if (abs0_tuple == nullptr) {
    MS_LOG(EXCEPTION) << "The abstract 0 of abstract tuple " << abs_tuple->ToString()
                      << " should be an AbstractTuple, but got" << abs_tuple->elements()[0]->ToString();
  }
  auto abs1_tuple = abs_tuple->elements()[1]->cast<abstract::AbstractTuplePtr>();
  if (abs1_tuple == nullptr) {
    MS_LOG(EXCEPTION) << "The abstract 1 of abstract tuple " << abs_tuple->ToString()
                      << " should be an AbstractTuple, but got" << abs_tuple->elements()[1]->ToString();
  }
  return {abs0_tuple, abs1_tuple};
}

AnfNodePtr GetReduceNode(const FuncGraphPtr &fg, const AnfNodePtr &dx,
                         const abstract::AbstractTuplePtr &scalar_abs_tuple, const AnfNodePtr &shape_x_node) {
  ShapeVector scalar_list;
  const auto &elements = scalar_abs_tuple->elements();
  (void)std::transform(elements.begin(), elements.end(), std::back_inserter(scalar_list),
                       [](const AbstractBasePtr &abs) {
                         auto abs_scalar = dyn_cast_ptr<abstract::AbstractScalar>(abs);
                         if (abs_scalar == nullptr) {
                           MS_LOG(EXCEPTION) << "The abstract should be AbstractScalar, but got " << abs->ToString();
                         }
                         return GetValue<int64_t>(abs_scalar->BuildValue());
                       });
  auto new_dx = dx;
  const auto &dx_abs = dx->abstract();
  MS_EXCEPTION_IF_NULL(dx_abs);
  if (dx_abs->BuildShape()->isa<abstract::Shape>()) {
    new_dx = ReduceSumWithCast(fg, dx, scalar_list);
  }
  return NewNode(fg, {Reshape(fg), new_dx, shape_x_node});
}
}  // namespace

AnfNodePtr BinopGradCommon(const FuncGraphPtr &fg, const AnfNodePtr &x, const AnfNodePtr &y, const AnfNodePtr &dx,
                           const AnfNodePtr &dy) {
  auto reduce_dx = dx;
  auto reduce_dy = dy;
  auto x_abs = x->abstract();
  MS_EXCEPTION_IF_NULL(x_abs);
  auto y_abs = y->abstract();
  MS_EXCEPTION_IF_NULL(y_abs);
  // x or y is scalar
  if (!x_abs->BuildShape()->isa<abstract::Shape>() || !y_abs->BuildShape()->isa<abstract::Shape>()) {
    if (!x_abs->BuildShape()->isa<abstract::Shape>()) {
      reduce_dx = ReduceSumWithCast(fg, dx, ShapeVector());
    }
    if (!y_abs->BuildShape()->isa<abstract::Shape>()) {
      reduce_dy = ReduceSumWithCast(fg, dy, ShapeVector());
    }
    return NewNode(fg, {MakeTuple(), reduce_dx, reduce_dy});
  }

  auto shape_x = NewNode(fg, {Shape(), x}, true, true);
  auto shape_y = NewNode(fg, {Shape(), y}, true, true);
  if (!(IsSequenceValueUnknown(fg, shape_x) || IsSequenceValueUnknown(fg, shape_y))) {
    auto rx = BroadcastGradientArgs(fg, shape_x, shape_y);
    auto rx_abs = GetBroadcastGradientArgsAbstract(rx);
    if (!rx_abs.first->elements().empty()) {
      reduce_dx = GetReduceNode(fg, dx, rx_abs.first, shape_x);
    }
    if (!rx_abs.second->elements().empty()) {
      reduce_dy = GetReduceNode(fg, dy, rx_abs.second, shape_y);
    }
    return NewNode(fg, {MakeTuple(), reduce_dx, reduce_dy});
  }
  return DynBinopGradCommon(fg, x, y, dx, dy);
}

AnfNodePtr SumGradReduceAxis(const FuncGraphPtr &fg, const AnfNodePtr &x, const AnfNodePtr &rx, bool keep_dims) {
  return fg->NewCNodeInOrder({ReduceSum(fg, keep_dims, true), x, rx});
}

bool IsSequenceValueUnknown(const FuncGraphPtr &fg, const AnfNodePtr &shape_node) {
  auto is_shape_known_node = NewNode(fg, {NewValueNode(prim::kPrimIsShapeUnknown), shape_node}, true);
  const auto &abs = is_shape_known_node->abstract();
  if (abs == nullptr) {
    MS_LOG(EXCEPTION) << "The node " << is_shape_known_node->DebugString() << " should be inferred.";
  }
  auto abs_scalar = abs->cast_ptr<abstract::AbstractScalar>();
  if (abs_scalar == nullptr) {
    MS_LOG(EXCEPTION) << "The abstract of node " << is_shape_known_node->DebugString()
                      << " should be a AbstractScalar.";
  }
  return GetValue<bool>(abs_scalar->BuildValue());
}

AnfNodePtr BroadcastGradientArgs(const FuncGraphPtr &fg, const AnfNodePtr &x_shape_node,
                                 const AnfNodePtr &y_shape_node) {
  auto broadcast_gradient_args =
    NewNode(fg, {NewValueNode(prim::kPrimBroadcastGradientArgs), x_shape_node, y_shape_node}, true);
  return broadcast_gradient_args;
}

ValuePtr GetPadModStr(const ValuePtr &value, bool upper) {
  static std::map<int64_t, std::string> PadModToStrMap = {
    {PadMode::PAD, "pad"},
    {PadMode::SAME, "same"},
    {PadMode::VALID, "valid"},
  };
  if (value->isa<StringImm>()) {
    return value;
  }
  if (!value->isa<Int64Imm>()) {
    MS_LOG(EXCEPTION) << "The pad mode value should be an int64 number, but got " << value->ToString();
  }
  auto iter = PadModToStrMap.find(GetValue<int64_t>(value));
  if (iter == PadModToStrMap.end()) {
    MS_LOG(EXCEPTION) << "The pad mode value should be an valid number, but got " << value->ToString();
  }
  if (!upper) {
    return MakeValue(iter->second);
  }
  auto str = iter->second;
  (void)std::transform(str.begin(), str.end(), str.begin(), toupper);
  return MakeValue(str);
}

AnfNodePtr DType() { return NewValueNode(prim::kPrimDType); }
}  // namespace graph_bprop
}  // namespace mindspore
