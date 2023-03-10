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
