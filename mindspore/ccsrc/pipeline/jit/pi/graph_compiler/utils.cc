/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "pipeline/jit/pi/graph_compiler/utils.h"
#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "include/common/utils/python_adapter.h"
#include "pipeline/jit/pi/pydef.h"
#include "abstract/ops/primitive_infer_map.h"
#include "frontend/operator/ops.h"
#include "mindspore/core/ops/sparse_tensor_ops.h"
#include "mindspore/core/ops/sequence_ops.h"
#include "mindspore/core/ops/comparison_ops.h"
#include "mindspore/core/ops/array_ops.h"
#include "mindspore/core/ops/math_ops.h"
#include "mindspore/core/ops/structure_ops.h"
#include "mindspore/core/ops/arithmetic_ops.h"
#include "mindspore/core/ops/framework_ops.h"
#include "pipeline/jit/ps/parse/data_converter.h"
#include "pipeline/jit/ps/resource.h"
#include "pipeline/jit/ps/static_analysis/static_analysis.h"

namespace mindspore {
namespace pijit {
namespace {
// Arg is mutable when it is mutable or it is meta tensor and it is not const
bool IsMutableArg(const py::object &obj, const ValuePtr &value) {
  return value->isa<tensor::MetaSparseTensor>() || (value->isa<tensor::MetaTensor>() && !GraphUtils::IsConst(obj)) ||
         GraphUtils::IsMutable(obj);
}

bool IsMetaTensorTuple(const ValuePtr &value) {
  if (!value->isa<ValueTuple>()) {
    return false;
  }
  auto tuple = value->cast<ValueTuplePtr>();
  for (auto element : tuple->value()) {
    if (!element->isa<tensor::MetaTensor>() && !IsMetaTensorTuple(element)) {
      return false;
    }
  }
  return true;
}

bool EnableArgBroaden(const py::object &obj, const ValuePtr &value, bool enable_tuple_broaden) {
  return IsMutableArg(obj, value) || value->isa<tensor::MetaSparseTensor>() ||
         (value->isa<Scalar>() && MsContext::GetInstance()->get_param<bool>(MS_CTX_GRAD_FOR_SCALAR)) ||
         (enable_tuple_broaden && IsMetaTensorTuple(value));
}

void CheckAndConvertToVariableLenSequence(const py::object &obj, AbstractBasePtr abs) {
  if (!GraphUtils::IsDynamicLength(obj)) {
    return;
  }
  if (!abs->isa<abstract::AbstractSequence>()) {
    MS_EXCEPTION(TypeError) << "For mutable, when the variable_len the True, the first input should be"
                            << " list or tuple, but got: " << abs->ToString();
  }
  auto abs_seq = abs->cast<abstract::AbstractSequencePtr>();
  abs_seq->CheckAndConvertToDynamicLenSequence();
}
}  // namespace

bool GraphUtils::IsTupleCanBroaden(const py::object &obj) {
  if (!py::isinstance<py::tuple>(obj)) {
    return false;
  }
  py::tuple tuple = py::cast<py::tuple>(obj);
  for (auto item : tuple) {
    auto elem = py::cast<py::object>(item);
    if (!py::isinstance<mindspore::tensor::Tensor>(elem) && !IsTupleCanBroaden(elem)) {
      return false;
    }
  }
  return true;
}

bool GraphUtils::IsGradForScalar(const py::object &obj) {
  return MsContext::GetInstance()->get_param<bool>(MS_CTX_GRAD_FOR_SCALAR) &&
         (py::isinstance<py::int_>(obj) || py::isinstance<py::float_>(obj));
}

bool GraphUtils::IsTensor(const py::object &obj) {
  return py::isinstance<mindspore::tensor::Tensor>(obj) || py::isinstance<mindspore::tensor::CSRTensor>(obj) ||
         py::isinstance<mindspore::tensor::COOTensor>(obj) || py::isinstance<mindspore::tensor::RowTensor>(obj);
}

AbstractBasePtr GraphUtils::ArgsToAbstract(const py::object &arg, const ValuePtr &value, bool enable_tuple_broaden) {
  auto ret = abstract::ToAbstract(value, nullptr, nullptr);
  if (EnableArgBroaden(arg, value, enable_tuple_broaden)) {
    ret = AbstractBroaden(ret);
  }
  CheckAndConvertToVariableLenSequence(arg, ret);
  return ret;
}

AnfNodePtr GraphUtils::GetPrimOrMetaFuncGraph(int op_code) {
  auto ret = GetPrimitive(op_code);
  if (ret != nullptr) {
    return NewValueNode(ret);
  }
  return GetMetaFuncGraph(op_code);
}

PrimitivePtr GraphUtils::GetPrimitive(int op_code) {
  static std::map<int, PrimitivePtr> op_code_2_prim = {
    {UNARY_INVERT, prim::kPrimInvert},       {RETURN_VALUE, prim::kPrimReturn},
    {LIST_TO_TUPLE, prim::kPrimMakeTuple},   {LIST_APPEND, prim::kPrimListAppend},
    {BUILD_TUPLE, prim::kPrimMakeTuple},     {BUILD_LIST, prim::kPrimMakeList},
    {BUILD_SET, prim::kPrimMakeList},        {BUILD_MAP, prim::kPrimMakeDict},
    {BUILD_SLICE, prim::kPrimMakeSlice},     {BUILD_CONST_KEY_MAP, prim::kPrimMakeDict},
    {BUILD_STRING, prim::kPrimStringConcat}, {LOAD_ATTR, prim::kPrimGetAttr},
    {LOAD_METHOD, prim::kPrimGetAttr}};

  if (op_code_2_prim.find(op_code) == op_code_2_prim.end()) {
    return nullptr;
  }

  return op_code_2_prim.at(op_code);
}

std::string GraphUtils::OpCodeToGraphName(int op_code) {
  static std::map<int, std::string> op_code_2_graph_name = {{UNARY_NEGATIVE, "negative"},
                                                            {UNARY_NOT, "logical_not"},
                                                            {BINARY_POWER, "pow_"},
                                                            {BINARY_MULTIPLY, "mul"},
                                                            {BINARY_MODULO, "mod"},
                                                            {BINARY_ADD, "add"},
                                                            {BINARY_SUBTRACT, "sub"},
                                                            {BINARY_SUBSCR, "getitem"},
                                                            {BINARY_FLOOR_DIVIDE, "floordiv"},
                                                            {BINARY_TRUE_DIVIDE, "div"},
                                                            {INPLACE_FLOOR_DIVIDE, "floordiv"},
                                                            {INPLACE_TRUE_DIVIDE, "div"},
                                                            {INPLACE_ADD, "add"},
                                                            {INPLACE_SUBTRACT, "sub"},
                                                            {INPLACE_MULTIPLY, "mul"},
                                                            {INPLACE_MODULO, "mod"},
                                                            {BINARY_LSHIFT, "left_shift"},
                                                            {BINARY_RSHIFT, "right_shift"},
                                                            {BINARY_AND, "bitwise_and"},
                                                            {BINARY_XOR, "bitwise_xor"},
                                                            {BINARY_OR, "bitwise_or"},
                                                            {INPLACE_POWER, "pow"},
                                                            {INPLACE_LSHIFT, "left_shift"},
                                                            {INPLACE_RSHIFT, "right_shift"},
                                                            {INPLACE_AND, "bitwise_and"},
                                                            {INPLACE_XOR, "bitwise_xor"},
                                                            {INPLACE_OR, "bitwise_or"},
                                                            {DICT_MERGE, "add"},
                                                            {LIST_EXTEND, "add"}};
  auto iter = op_code_2_graph_name.find(op_code);
  if (iter == op_code_2_graph_name.end()) {
    return "";
  }
  return iter->second;
}

std::string GraphUtils::OpCompareArgToGraphName(int oparg) {
  static std::map<int, std::string> compare_arg_2_graph_name = {{Py_LT, "less"},    {Py_LE, "less_equal"},
                                                                {Py_EQ, "equal"},   {Py_NE, "not_equal"},
                                                                {Py_GT, "greater"}, {Py_GE, "greater_equal"}};
  auto iter = compare_arg_2_graph_name.find(oparg);
  if (iter == compare_arg_2_graph_name.end()) {
    return "";
  }
  return iter->second;
}

AnfNodePtr GraphUtils::GetMetaFuncGraph(int op_code) {
  // MS_EXCEPTION_IF_CHECK_FAIL(op_code_2_graph_name.find(op_code) != op_code_2_graph_name.end(),
  //                            "Not find the mutitype ops of OpCode " + std::to_string(op_code) + ".");
  const auto &graph_name = OpCodeToGraphName(op_code);
  if (graph_name != "") {
    return GetMetaFuncGraph(graph_name);
  }
  return nullptr;
}

AnfNodePtr GraphUtils::GetMetaFuncGraph(const std::string &name) {
  py::object obj = python_adapter::GetPyFn("mindspore.ops.composite.multitype_ops", name);
  return ConvertPythonObjectToAnfNode(obj);
}

AnfNodePtr GraphUtils::ConvertPythonObjectToAnfNode(const py::object &object) {
  ValuePtr value = nullptr;
  bool succ = mindspore::parse::ConvertData(object, &value, python_adapter::UseSignatureInResolve());
  if (!succ) {
    MS_LOG(EXCEPTION) << "Convert " << (std::string)py::str(object) << " To AnfNode Fail.";
  }
  return NewValueNode(value);
}

}  // namespace pijit
}  // namespace mindspore
