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

#include "pipeline/jit/graph_jit/graph_compiler/utils.h"
#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "include/common/utils/python_adapter.h"
#include "pipeline/jit/graph_jit/pydef.h"
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
namespace jit {
namespace graph {
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

std::string GenerateOperationNodeKey(int op_code, const AbstractBasePtrList &abs_list) {
  std::string key = std::to_string(op_code);
  for (const auto &abs : abs_list) {
    key += "_" + abs->type_name();
    if (abs->isa<abstract::AbstractScalar>() && abs->GetTypeTrack()->isa<String>()) {
      key += "_String";
    }
  }
  return key;
}

// Get abstract of the default value in the given parameter.
AbstractBasePtr GetDefaultValueAbstract(const ParameterPtr &param) {
  auto value = param->default_param();
  MS_EXCEPTION_IF_NULL(value);
  auto value_abs = value->ToAbstract();
  MS_EXCEPTION_IF_NULL(value_abs);
  if (value_abs->isa<abstract::AbstractMapTensor>()) {
    // Return AbstractMapTensor for map parameter.
    return value_abs;
  }
  // Make an AbstractRefTensor for the tensor value.
  auto abs_tensor = value_abs->cast<abstract::AbstractTensorPtr>();
  MS_EXCEPTION_IF_NULL(abs_tensor);
  auto ref_key = std::make_shared<RefKey>(param->name());
  return std::make_shared<abstract::AbstractRefTensor>(abs_tensor, ref_key);
}

AbstractBasePtr InferFuncGraph(const FuncGraphPtr &func_graph) {
  GraphUtils::ResolveFuncGraph(func_graph);
  abstract::AnalysisEnginePtr engine =
    std::make_shared<abstract::AnalysisEngine>(abstract::GetPrimEvaluatorConstructors(), func_graph->manager());
  abstract::AbstractBasePtrList args_abs;
  // Handle the Parameter from FV inputs.
  for (const auto &param : func_graph->parameters()) {
    auto param_node = std::static_pointer_cast<Parameter>(param);
    MS_EXCEPTION_IF_NULL(param_node);
    if (param_node->has_default()) {
      auto param_abs = GetDefaultValueAbstract(param_node);
      (void)args_abs.emplace_back(param_abs);
    } else {
      if (param->abstract() == nullptr) {
        param_node->set_default_param(kNone);
        param_node->set_abstract(kNone->ToAbstract());
      }
      (void)args_abs.emplace_back(param_node->abstract());
    }
  }
  abstract::AnalysisResult res = engine->Run(func_graph, args_abs);
  return res.eval_result->abstract();
}

abstract::AbstractBasePtr InferPrimitiveNode(const PrimitivePtr &prim, const AbstractBasePtrList &args_spec_list) {
  auto eval_impl = abstract::GetPrimitiveInferImpl(prim);
  if (eval_impl.has_value() && eval_impl.value().IsImplInferShapeAndType()) {
    // Call Cpp infer
    return eval_impl.value().InferShapeAndType(nullptr, prim, args_spec_list);
  } else if (prim->isa<PrimitivePy>()) {
    PrimitivePyPtr prim_py = prim->cast<PrimitivePyPtr>();
    if (prim_py->HasAttr("constexpr_prim")) {
      auto evaluator = std::make_shared<abstract::ConstexprEvaluator>(prim_py);
      return evaluator->EvalPrim(nullptr, args_spec_list, nullptr, nullptr)->abstract();
    } else {
      auto evaluator = std::make_shared<abstract::PythonPrimEvaluator>(prim_py);
      return evaluator->EvalPrim(nullptr, args_spec_list)->abstract();
    }
  } else {
    auto evaluator = abstract::GetPrimEvaluator(prim, nullptr);
    auto trivial_evaluator = dyn_cast_ptr<abstract::TrivialPrimEvaluator>(evaluator);
    if (trivial_evaluator != nullptr) {
      return trivial_evaluator->EvalPrim(nullptr, args_spec_list)->abstract();
    }
    // Support MakeTuple/MakeList ops in PyNative mode.
    auto transition_evaluator = dyn_cast_ptr<abstract::TransitionPrimEvaluator>(evaluator);
    if (transition_evaluator != nullptr && (transition_evaluator->isa<abstract::MakeTupleEvaluator>() ||
                                            transition_evaluator->isa<abstract::MakeListEvaluator>())) {
      return transition_evaluator->EvalPrim(nullptr, args_spec_list, nullptr, nullptr)->abstract();
    }
  }
  // manually throw an exception to avoid the critical log.
  MS_LOG(EXCEPTION) << "The infer function of the primitive [" + prim->name() + "] is not defined.";
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
    {UNARY_INVERT, prim::kPrimInvert},          {RETURN_VALUE, prim::kPrimReturn},
    {LIST_TO_TUPLE, prim::kPrimMakeTuple},      {BUILD_TUPLE, prim::kPrimMakeTuple},
    {BUILD_LIST, prim::kPrimMakeList},          {BUILD_SET, prim::kPrimMakeList},
    {BUILD_MAP, prim::kPrimMakeDict},           {BUILD_SLICE, prim::kPrimMakeSlice},
    {BUILD_CONST_KEY_MAP, prim::kPrimMakeDict}, {BUILD_STRING, prim::kPrimStringConcat}};

  if (op_code_2_prim.find(op_code) == op_code_2_prim.end()) {
    return nullptr;
  }

  return op_code_2_prim.at(op_code);
}

AnfNodePtr GraphUtils::GetMetaFuncGraph(int op_code) {
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
                                                            {INPLACE_OR, "bitwise_or"}};
  MS_EXCEPTION_IF_CHECK_FAIL(op_code_2_graph_name.find(op_code) != op_code_2_graph_name.end(),
                             "Not find the mutitype ops of OpCode " + std::to_string(op_code) + ".");
  return GetMetaFuncGraph(op_code_2_graph_name.at(op_code));
}

AnfNodePtr GraphUtils::GetMetaFuncGraph(const std::string &name) {
  py::object obj = python_adapter::GetPyFn("mindspore.ops.composite.multitype_ops", name);
  return ConvertPythonObjectToAnfNode(obj);
}

AnfNodePtr GraphUtils::GetOperationNode(int op_code, int op_arg, const AbstractBasePtrList &abs_list) {
  static mindspore::HashMap<std::string, std::vector<Any>> key_to_operation = {
    // op : 11, opname : UNARY_NEGATIVE
    {"11", {GetMetaFuncGraph("negative")}},

    // op : 12, opname : UNARY_NOT
    {"12", {GetMetaFuncGraph("logical_not")}},

    // op : 19, opname : BINARY_POWER
    {"19", {GetMetaFuncGraph("pow_")}},

    // op : 20, opname : BINARY_MULTIPLY
    {"20", {GetMetaFuncGraph("mul")}},
    {"20_AbstractScalar_AbstractScalar", {prim::kPrimScalarMul}},
    {"20_AbstractScalar_String_AbstractScalar_String", {prim::kPrimStringMul}},

    // op : 23, opname : BINARY_ADD
    {"23", {GetMetaFuncGraph("add")}},
    {"23_AbstractScalar_AbstractScalar", {prim::kPrimScalarAdd}},
    {"23_AbstractScalar_String_AbstractScalar_String", {prim::kPrimStringConcat}},

    // op : 24, opname : BINARY_SUBTRACT
    {"24", {GetMetaFuncGraph("sub")}},
    {"24_AbstractScalar_AbstractScalar", {prim::kPrimScalarSub}},

    // op : 25, opname : BINARY_SUBSCR
    {"25", {GetMetaFuncGraph("getitem")}},
    {"25_AbstractList_AbstractScalar", {prim::kPrimListGetItem}},
    {"25_AbstractTuple_AbstractScalar", {prim::kPrimTupleGetItem}},
    {"25_AbstractDictionary_AbstractScalar_String", {prim::kPrimDictGetItem}},

    // op : 26, opname : BINARY_FLOOR_DIVIDE
    {"26", {GetMetaFuncGraph("floordiv")}},

    // op : 27, opname : BINARY_TRUE_DIVIDE
    {"27", {GetMetaFuncGraph("div")}},
    {"27_AbstractScalar_AbstractScalar", {prim::kPrimScalarDiv}},

    // op : 62, opname : BINARY_LSHIFT
    {"62", {GetMetaFuncGraph("left_shift")}},
    // op : 63, opname : BINARY_RSHIFT
    {"63", {GetMetaFuncGraph("right_shift")}},
    // op : 64, opname : BINARY_AND
    {"64", {GetMetaFuncGraph("bitwise_and")}},
    // op : 65, opname : BINARY_XOR
    {"65", {GetMetaFuncGraph("bitwise_xor")}},
    // op : 66, opname : BINARY_OR
    {"66", {GetMetaFuncGraph("bitwise_or")}},

    // op : 107, opname : COMPARE_OP
    {"107",
     {GetMetaFuncGraph("less"), GetMetaFuncGraph("less_equal"), GetMetaFuncGraph("equal"),
      GetMetaFuncGraph("not_equal"), GetMetaFuncGraph("greater"), GetMetaFuncGraph("greater_equal")}},
    {"107_AbstractScalar_AbstractScalar",
     {prim::kPrimScalarLt, prim::kPrimScalarLe, prim::kPrimScalarEq, prim::kPrimScalarNe, prim::kPrimScalarGt,
      prim::kPrimScalarGe}},
    {"107_AbstractScalar_String_AbstractScalar_String",
     {prim::kPrimStringLt, prim::kPrimStringLe, prim::kPrimStringEq, prim::kPrimStringNot, prim::kPrimStringGt,
      prim::kPrimStringGe}},
  };
  auto key = GenerateOperationNodeKey(op_code, abs_list);
  auto it = key_to_operation.find(key);
  if (it == key_to_operation.end()) {
    key = std::to_string(op_code);
    it = key_to_operation.find(key);
    MS_EXCEPTION_IF_CHECK_FAIL(it != key_to_operation.end(), "The operation of OpCode " + key + " is not define.");
  }
  Any method = it->second[op_arg];
  if (method.is<PrimitivePtr>()) {
    return NewValueNode(method.cast<PrimitivePtr>());
  }
  return method.cast<AnfNodePtr>();
}

AnfNodePtr GraphUtils::ConvertPythonObjectToAnfNode(const py::object &object) {
  ValuePtr value = nullptr;
  bool succ = mindspore::parse::ConvertData(object, &value, python_adapter::UseSignatureInResolve());
  if (!succ) {
    MS_LOG(EXCEPTION) << "Convert " << (std::string)py::str(object) << " To AnfNode Fail.";
  }
  auto node = NewValueNode(value);
  node->set_abstract(ArgsToAbstract(object, value));
  return node;
}

py::object GraphUtils::CreatePythonClassInstance(const py::object &cls, const py::object &args,
                                                 const py::object &kwargs) {
  MS_EXCEPTION_IF_CHECK_FAIL(PyType_Check(cls.ptr()), "Expected Python Class Type.");
  py::object instance = py::reinterpret_steal<py::object>(PyObject_Call(cls.ptr(), args.ptr(), kwargs.ptr()));
  MS_EXCEPTION_IF_CHECK_FAIL((instance.ptr() != nullptr),
                             "Create the instance of " + py::str(cls).cast<std::string>() + " fail.");
  return instance;
}

void GraphUtils::ResolveFuncGraph(const FuncGraphPtr &func_graph) {
  FuncGraphManagerPtr manager = func_graph->manager();
  auto resource = std::make_shared<pipeline::Resource>();
  resource->set_func_graph(func_graph);
  resource->set_manager(manager);
  (void)parse::ResolveFuncGraph(func_graph, resource);
}

void GraphUtils::InferAnfNode(const AnfNodePtr &node) {
  if (node->abstract() != nullptr) {
    return;
  }
  // All valueNode should be infer in function ConvertPythonObjectToAnfNode
  MS_EXCEPTION_IF_CHECK_FAIL(node->isa<CNode>(), " Infer Node : " + node->DebugString() + " fail.");
  auto cnode = node->cast<CNodePtr>();
  AbstractBasePtrList args_spec_list;
  auto inputs = cnode->inputs();
  (void)std::transform(inputs.begin() + 1, inputs.end(), std::back_inserter(args_spec_list),
                       [](const AnfNodePtr &node) {
                         // In order for DDE (Dead Data Elimination) to work correctly,
                         // need to set the abstract of the CNode to nullptr before the special optimization is
                         // performed but the abstract of this CNode node may be used in the next graph, so it is
                         // necessary to re-infer
                         InferAnfNode(node);
                         return node->abstract();
                       });
  if (IsPrimitiveCNode(cnode)) {
    auto prim = GetCNodePrimitive(cnode);
    auto abstract = InferPrimitiveNode(prim, args_spec_list);
    MS_EXCEPTION_IF_NULL(abstract);
    cnode->set_abstract(abstract);
  } else {
    MS_EXCEPTION_IF_CHECK_FAIL((IsValueNode<FuncGraph>(inputs[0]) || IsValueNode<MetaFuncGraph>(inputs[0])),
                               "Expected FuncGraph, but got " + inputs[0]->DebugString());
    auto sub_func_graph = GetValueNode<FuncGraphPtr>(inputs[0]);
    if (IsValueNode<MetaFuncGraph>(inputs[0])) {
      sub_func_graph = GetValueNode<MetaFuncGraphPtr>(inputs[0])->GenerateFuncGraph(args_spec_list);
      cnode->set_input(0, NewValueNode(sub_func_graph));
      inputs = cnode->inputs();
    }
    auto parameters = sub_func_graph->parameters();
    for (size_t index = 0; index < args_spec_list.size(); index++) {
      parameters[index]->set_abstract(args_spec_list[index]);
    }
    parse::Parser::UpdateTopFuncGraph(sub_func_graph);
    cnode->func_graph()->manager()->AddFuncGraph(sub_func_graph);
    auto abs = InferFuncGraph(sub_func_graph);
    parameters = sub_func_graph->parameters();
    for (size_t index = inputs.size() - 1; index < parameters.size(); index++) {
      auto param = std::static_pointer_cast<Parameter>(parameters[index]);
      cnode->add_input(NewValueNode(param->default_param()));
    }
    cnode->set_abstract(abs);
  }
}
}  // namespace graph
}  // namespace jit
}  // namespace mindspore
