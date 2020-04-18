/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include "pipeline/static_analysis/prim.h"

#include <algorithm>
#include <limits>
#include <mutex>
#include <set>
#include <string>
#include <utility>

#include "operator/cc_implementations.h"
#include "operator/ops.h"
#include "operator/composite/do_signature.h"
#include "operator/prim_to_function.h"
#include "pipeline/static_analysis/utils.h"
#include "utils/symbolic.h"
#include "./common.h"
#include "pipeline/resource.h"
#include "pipeline/parse/resolve.h"
#include "ir/meta_tensor.h"
#include "utils/convert_utils.h"
#include "pipeline/parse/data_converter.h"
#include "pipeline/static_analysis/param_validator.h"
#include "common/utils.h"

namespace mindspore {
namespace abstract {
PrimitiveEvalImplMap &GetPrimitiveToEvalImplMap() {
  static PrimitiveEvalImplMap prim_eval_implement_map = {
    // Statements
    {prim::kPrimReturn, {InferImplReturn, true}},
    {prim::kPrimTypeOf, {InferImplTypeof, false}},
    {prim::kPrimHasType, {InferImplHasType, false}},
    {prim::kPrimDot, {InferImplDot, true}},
    {prim::kPrimSwitch, {InferImplSwitch, true}},
    {prim::kPrimIs_, {InferImplIs_, true}},
    {prim::kPrimIsNot, {InferImplIsNot, true}},
    {prim::kPrimInDict, {InferImplInDict, true}},
    {prim::kPrimNotInDict, {InferImplNotInDict, true}},
    // Maths
    {prim::kPrimMaximumGrad, {InferImplMinOrMaxGrad, true}},
    {prim::kPrimMinimumGrad, {InferImplMinOrMaxGrad, true}},
    // Array
    {prim::kPrimScalarToArray, {InferImplScalarToArray, true}},
    {prim::kPrimArrayToScalar, {InferImplArrayToScalar, true}},
    {prim::kPrimBroadcastShape, {InferImplBroadCastShape, true}},
    {prim::kPrimShape, {InferImplShape, true}},
    {prim::kPrimPack, {InferImplPack, true}},
    // Structure
    {prim::kPrimMakeTuple, {InferImplMakeTuple, true}},
    {prim::kPrimMakeList, {InferImplMakeList, true}},
    {prim::kPrimMakeDict, {InferImplMakeDict, true}},
    {prim::kPrimMakeSlice, {InferImplMakeSlice, true}},
    {prim::kPrimMakeKeywordArg, {InferImplMakeKwarg, true}},
    {prim::kPrimExtractKeywordArg, {InferImplExtractKwarg, true}},
    {prim::kPrimMakeRecord, {InferImplMakeRecord, false}},
    {prim::kPrimTupleGetItem, {InferImplTupleGetItem, true}},
    {prim::kPrimListGetItem, {InferImplListGetItem, true}},
    {prim::kPrimTupleSetItem, {InferImplTupleSetItem, true}},
    {prim::kPrimListSetItem, {InferImplListSetItem, true}},
    {prim::kPrimDictGetItem, {InferImplDictGetItem, true}},
    {prim::kPrimDictSetItem, {InferImplDictSetItem, true}},
    {prim::kPrimListAppend, {InferImplListAppend, true}},
    {prim::kPrimTupleLen, {InferImplTupleLen, true}},
    {prim::kPrimListLen, {InferImplListLen, true}},
    {prim::kPrimArrayLen, {InferImplArrayLen, true}},
    {prim::kPrimListMap, {InferImplListMap, false}},
    {prim::kPrimListReduce, {InferImplListReduce, false}},
    {prim::kPrimTupleReversed, {InferImplTupleReversed, false}},
    {prim::kPrimReducedShape, {InferImplReduceShape, false}},
    {prim::kPrimTupleDiv, {InferImplTupleDiv, false}},
    {prim::kPrimTupleToArray, {InferImplTuple2Array, false}},
    {prim::kPrimShapeMul, {InferImplShapeMul, false}},
    {prim::kPrimTupleEqual, {InferImplTupleEqual, false}},
    {prim::kPrimListEqual, {InferImplListEqual, false}},
    {prim::kPrimMakeRange, {InferImplMakeRange, false}},
    {prim::kPrimStopGradient, {InferImplStopGradient, false}},
    {prim::kPrimStringEqual, {InferImplStringEqual, false}},
    {prim::kPrimStringConcat, {InferImplStringConcat, false}},
    {prim::kPrimDictLen, {InferImplDictLen, false}},
    // NN
    {prim::kPrimPooling, {InferImplPooling, true}},
    {prim::kPrimPoolingGrad, {InferImplPoolingGrad, true}},
    {prim::kPrimFusedBatchNorm, {InferImplFusedBatchNorm, true}},
    {prim::kPrimFusedBatchNormGrad, {InferImplFusedBatchNormGrad, true}},
    {prim::kPrimReluGrad, {InferImplReluGrad, true}},
    {prim::kPrimConv2DBackpropInput, {InferImplConv2DBackpropInput, true}},
    {prim::kPrimConv2DBackpropFilter, {InferImplConv2DBackpropFilter, true}},
    {prim::kPrimBiasAddGrad, {InferImplBiasAddGrad, true}},
    {prim::kPrimRelu, {InferImplRelu, true}},
    {prim::kPrimZerosLikeTensor, {InferImplZerosLikeTensor, true}},
    {prim::kPrimFakeBprop, {InferImplFakeBprop, false}},
    {prim::kPrimLayerNorm, {InferImplLayerNorm, true}},
    {prim::kPrimLayerNormGrad, {InferImplLayerNormGrad, true}},
    {prim::kPrimDropoutGenMask, {InferImplDropoutGenMask, true}},
    // Others
    {prim::kPrimIdentity, {InferImplIdentity, true}},
    // Set impl to null as it will use PartialEvaluator;
    {prim::kPrimPartial, {nullptr, true}},
    {prim::kPrimJ, {InferImplJ, false}},
    {prim::kPrimEnvGetItem, {InferImplEnvGetItem, true}},
    {prim::kPrimEnvSetItem, {InferImplEnvSetItem, true}},
    {prim::kPrimEnvAdd, {InferImplEnvAdd, true}},
    {prim::kPrimMakeRefKey, {InferImplMakeRefKey, true}},
    {prim::kPrimMakeRef, {InferImplMakeRef, true}},
    {prim::kPrimGetRefKey, {InferImplGetRefKey, true}},
    {prim::kPrimGetRefValue, {InferImplGetRefValue, true}},
    {prim::kPrimGetRefOrigin, {InferImplGetRefOrigin, true}},
    {prim::kPrimStateSetItem, {InferImplStateSetItem, true}},
    {prim::kPrimDepend, {InferImplDepend, true}},
    {prim::kPrimBroadcastGradientArgs, {InferImplBroadcastGradientArgs, false}},
    {prim::kPrimControlDepend, {InferImplControlDepend, true}},
    // Debug
    {prim::kPrimScalarSummary, {InferImplScalarSummary, true}},
    {prim::kPrimImageSummary, {InferImplTensorSummary, true}},
    {prim::kPrimTensorSummary, {InferImplTensorSummary, true}},
  };
  return prim_eval_implement_map;
}

using mindspore::parse::PyObjectWrapper;

AbstractBasePtr StandardPrimEvaluator::EvalPrim(const AnalysisEnginePtr &engine, const AbstractBasePtrList &args) {
  AbstractBasePtr abs_base = eval_impl_(engine, prim_, args);
  return abs_base;
}

AbstractBasePtr DoSignatureEvaluator::Run(AnalysisEnginePtr engine, const ConfigPtrList &args_conf_list,
                                          AnfNodeConfigPtr out_conf) {
  AbstractBasePtrList args_spec_list;
  if (!prim_->isa<prim::DoSignaturePrimitive>()) {
    MS_LOG(EXCEPTION) << "Primitive should be DoSignature, but " << prim_->ToString();
  }
  if (out_conf->node() == nullptr || !out_conf->node()->isa<CNode>()) {
    MS_LOG(EXCEPTION) << "Node of out_conf should be CNode";
  }

  auto do_signature = dyn_cast<prim::DoSignaturePrimitive>(prim_);
  auto out_node = dyn_cast<CNode>(out_conf->node());
  const auto &out_node_inputs = out_node->inputs();
  if (out_node->inputs().size() == 0 || (out_node_inputs.size() - 1) != args_conf_list.size()) {
    MS_LOG(EXCEPTION) << "Op: " << do_signature->function()->ToString()
                      << " args size should equal to inputs size minus 1, but args size " << args_conf_list.size()
                      << ", inputs size " << out_node_inputs.size();
  }
  AnfNodePtrList args_inputs{out_node_inputs.begin() + 1, out_node_inputs.end()};

  (void)std::transform(args_conf_list.begin(), args_conf_list.end(), std::back_inserter(args_spec_list),
                       [](const ConfigPtr &ref) -> AbstractBasePtr { return ref->GetEvaluatedValue(); });

  ScopePtr scope = kDefaultScope;
  if (out_conf != nullptr) {
    scope = out_conf->node()->scope();
  }
  ScopeGuard scope_guard(scope);

  AnfNodePtr new_cnode = nullptr;
  if (bound_node() != nullptr) {
    TraceManager::DebugTrace(std::make_shared<TraceDoSignature>(bound_node()->debug_info()));
    new_cnode = prim::GenerateCNode(out_node->func_graph(), prim_->ToString(), do_signature->function(), args_spec_list,
                                    args_inputs);
    TraceManager::EndTrace();
  } else {
    new_cnode = prim::GenerateCNode(out_node->func_graph(), prim_->ToString(), do_signature->function(), args_spec_list,
                                    args_inputs);
  }
  AnfNodeConfigPtr fn_conf = engine->MakeConfig(new_cnode, out_conf->context());

  return engine->ForwardConfig(out_conf, fn_conf);
}

static AbstractBasePtrList GetUnpackGraphSpecArgsList(AbstractBasePtrList args_spec_list, bool need_unpack) {
  // arg[0] is the func graph to unpack, ignore it
  AbstractBasePtrList specialize_args_before_unpack(args_spec_list.begin() + 1, args_spec_list.end());
  AbstractBasePtrList graph_specialize_args;
  if (need_unpack) {
    for (size_t index = 0; index < specialize_args_before_unpack.size(); index++) {
      MS_EXCEPTION_IF_NULL(specialize_args_before_unpack[index]);
      if (specialize_args_before_unpack[index]->isa<AbstractTuple>()) {
        AbstractTuplePtr arg_tuple = specialize_args_before_unpack[index]->cast<AbstractTuplePtr>();
        std::transform(arg_tuple->elements().begin(), arg_tuple->elements().end(),
                       std::back_inserter(graph_specialize_args), [](AbstractBasePtr abs) { return abs; });
      } else if (specialize_args_before_unpack[index]->isa<AbstractDictionary>()) {
        AbstractDictionaryPtr arg_dict = specialize_args_before_unpack[index]->cast<AbstractDictionaryPtr>();
        auto dict_elems = arg_dict->elements();
        (void)std::transform(
          dict_elems.begin(), dict_elems.end(), std::back_inserter(graph_specialize_args),
          [](const AbstractAttribute &item) { return std::make_shared<AbstractKeywordArg>(item.first, item.second); });
      } else {
        MS_LOG(EXCEPTION) << "UnpackGraph require args should be tuple or dict, but got "
                          << specialize_args_before_unpack[index]->ToString();
      }
    }
  } else {
    graph_specialize_args = specialize_args_before_unpack;
  }
  return graph_specialize_args;
}

AbstractBasePtr UnpackGraphEvaluator::Run(AnalysisEnginePtr engine, const ConfigPtrList &args_conf_list,
                                          AnfNodeConfigPtr out_conf) {
  if (out_conf->node() == nullptr || !out_conf->node()->isa<CNode>()) {
    MS_LOG(EXCEPTION) << "Node of out_conf should be CNode";
  }
  if (!prim_->isa<prim::UnpackGraphPrimitive>()) {
    MS_LOG(EXCEPTION) << "Primitive should be UnpackGraphPrimitive, but got " << prim_->ToString();
  }

  auto unpack_graph = prim_->cast<prim::UnpackGraphPrimitivePtr>();
  auto out_node = out_conf->node()->cast<CNodePtr>();
  const auto &out_node_inputs = out_node->inputs();
  if (out_node->inputs().size() == 0 || (out_node_inputs.size() - 1) != args_conf_list.size()) {
    MS_LOG(EXCEPTION) << "UnpackGraphPrimitive"
                      << " args size should equal to inputs size minus 1, but args size " << args_conf_list.size()
                      << ", inputs size " << out_node_inputs.size();
  }
  AnfNodePtrList args_inputs{out_node_inputs.begin() + 1, out_node_inputs.end()};
  AbstractBasePtrList args_spec_list;
  (void)std::transform(args_conf_list.begin(), args_conf_list.end(), std::back_inserter(args_spec_list),
                       [](const ConfigPtr &ref) -> AbstractBasePtr { return ref->GetEvaluatedValue(); });
  // get the forward graph
  MS_EXCEPTION_IF_NULL(args_spec_list[0]);
  AbstractFunctionPtr fn = args_spec_list[0]->cast<AbstractFunctionPtr>();
  if (fn == nullptr) {
    MS_LOG(EXCEPTION) << "UnpackGraphPrimitive arg0 must be AbstractFunction, but " << args_spec_list[0]->ToString();
  }
  auto real_fn = fn->cast<FuncGraphAbstractClosurePtr>();
  MS_EXCEPTION_IF_NULL(real_fn);
  FuncGraphPtr forward_graph = real_fn->func_graph();
  MS_EXCEPTION_IF_NULL(forward_graph);
  AbstractBasePtrList graph_specialize_args =
    GetUnpackGraphSpecArgsList(args_spec_list, unpack_graph->need_unpack_args());

  AbstractBasePtrList graph_specialize_args_without_sens;
  (void)std::transform(graph_specialize_args.begin(),
                       graph_specialize_args.end() - (unpack_graph->with_sens_in_args() ? 1 : 0),
                       std::back_inserter(graph_specialize_args_without_sens), [](AbstractBasePtr abs) { return abs; });
  auto new_graph = forward_graph->GenerateGraph(graph_specialize_args_without_sens);
  engine->func_graph_manager()->AddFuncGraph(new_graph);
  ScopePtr scope = kDefaultScope;
  if (out_conf != nullptr) {
    scope = out_conf->node()->scope();
  }
  ScopeGuard scope_guard(scope);
  AnfNodePtr new_vnode = NewValueNode(new_graph);
  AnfNodeConfigPtr fn_conf = engine->MakeConfig(new_vnode, out_conf->context());

  return engine->ForwardConfig(out_conf, fn_conf);
}

namespace {
py::object BuildValue(const ValuePtr &value_ptr) {
  if (value_ptr == nullptr) {
    return py::none();
  } else {
    return ValuePtrToPyData(value_ptr);
  }
}
}  // end anonymous namespace

py::dict ConvertAbstractToPython(const AbstractBasePtr &abs_base) {
  MS_EXCEPTION_IF_NULL(abs_base);
  py::dict dic;
  if (abs_base->isa<AbstractTensor>()) {
    auto arg_tensor = dyn_cast<AbstractTensor>(abs_base);
    dic["shape"] = arg_tensor->shape()->shape();
    dic["dtype"] = arg_tensor->BuildType();
    dic["value"] = BuildValue(arg_tensor->BuildValue());
  } else if (abs_base->isa<AbstractScalar>() || abs_base->isa<AbstractType>() || abs_base->isa<AbstractRefKey>()) {
    std::vector<int> shape;
    dic["shape"] = shape;
    dic["dtype"] = abs_base->BuildType();
    dic["value"] = BuildValue(abs_base->BuildValue());
  } else if (abs_base->isa<AbstractTuple>()) {
    auto arg_tuple = dyn_cast<AbstractTuple>(abs_base);
    size_t len = arg_tuple->size();
    py::tuple shape_tuple(len);
    py::tuple dtype_tuple(len);

    for (size_t i = 0; i < len; i++) {
      py::dict out = ConvertAbstractToPython(arg_tuple->elements()[i]);
      shape_tuple[i] = out["shape"];
      dtype_tuple[i] = out["dtype"];
    }
    dic["shape"] = shape_tuple;
    dic["dtype"] = dtype_tuple;
    dic["value"] = BuildValue(arg_tuple->BuildValue());
  } else if (abs_base->isa<AbstractList>()) {
    auto arg_list = dyn_cast<AbstractList>(abs_base);
    size_t len = arg_list->size();
    py::list shape_list(len);
    py::list dtype_list(len);

    for (size_t i = 0; i < len; i++) {
      py::dict out = ConvertAbstractToPython(arg_list->elements()[i]);
      shape_list[i] = out["shape"];
      dtype_list[i] = out["dtype"];
    }
    dic["shape"] = shape_list;
    dic["dtype"] = dtype_list;
    dic["value"] = BuildValue(arg_list->BuildValue());
  } else if (abs_base->isa<AbstractNone>()) {
    dic["shape"] = py::none();
    dic["dtype"] = py::none();
    dic["value"] = py::none();
  } else {
    auto value = abs_base->BuildValue();
    if ((*value == *kAnyValue)) {
      auto value_desc = abs_base->value_desc();
      MS_EXCEPTION(TypeError) << "Unsupported parameter " << (value_desc.empty() ? "type" : value_desc)
                              << " for python primitive.";
    }
    MS_EXCEPTION(TypeError) << "Unsupported parameter type for python primitive, the parameter value is "
                            << value->ToString();
  }
  return dic;
}

namespace {
py::tuple PreparePyInputs(const PrimitivePyPtr &prim_py, const AbstractBasePtrList &args) {
  const AbstractBasePtrList *args_ptr;

  if (prim_py->is_tuple_input_) {
    if (args.empty()) {
      MS_LOG(EXCEPTION) << "Primitive args is empty";
    }
    if (args[0] == nullptr || !args[0]->isa<AbstractTuple>()) {
      MS_LOG(EXCEPTION) << "Custom Primitive inputs should be packed into a Tuple after converting"
                           "prim convert pass for GE.";
    }
    args_ptr = &(args[0]->cast<AbstractTuplePtr>()->elements());
  } else {
    args_ptr = &args;
  }

  py::tuple py_args(args_ptr->size());
  for (size_t i = 0; i < args_ptr->size(); i++) {
    auto arg_i = (*args_ptr)[i];
    py_args[i] = ConvertAbstractToPython(arg_i);
  }
  return py_args;
}

AbstractBasePtr PyInferRes2Abstract(const PrimitivePyPtr &prim_py, const py::dict &output) {
  // Convert to AbstractValue based on type and shape
  if (output["value"].is_none()) {
    auto out_shape = output["shape"];
    auto out_dtype = output["dtype"];
    return PyListDtype2AbstractTensor(out_shape, out_dtype);
  }
  // Convert pyobject to Value, then to AbstractValue
  ValuePtr converted_ret = nullptr;
  bool converted = parse::ConvertData(output["value"], &converted_ret);
  if (!converted) {
    MS_LOG(EXCEPTION) << "Convert data failed";
  }
  auto res_spec = FromValue(converted_ret);
  MS_EXCEPTION_IF_NULL(res_spec);
  if (res_spec->isa<AbstractTensor>()) {
    // Replace to tensor constant node in specialize
    auto res_tensor = res_spec->cast<AbstractTensorPtr>();
    res_tensor->set_value(converted_ret);
  }
  if (prim_py->IsCustomPrim()) {
    // Raise error if output_num is not match the infer result.
    int output_num = GetValue<int>(prim_py->GetAttr("output_num"));
    if (res_spec->isa<AbstractTensor>() && output_num != 1) {
      MS_LOG(EXCEPTION) << "Custom primitive " << prim_py->ToString() << " output_num " << output_num
                        << " not matches the infer result.";
    } else if (res_spec->isa<AbstractTuple>() &&
               (res_spec->cast<AbstractTuplePtr>()->size() != IntToSize(output_num))) {
      MS_LOG(EXCEPTION) << "Custom primitive " << prim_py->ToString() << " output_num " << output_num
                        << " not matches the infer result.";
    }
  }
  return res_spec;
}
}  // end anonymous namespace

AbstractBasePtr PythonPrimEvaluator::EvalPrim(const AnalysisEnginePtr &, const AbstractBasePtrList &args) {
  MS_LOG(DEBUG) << "Eval for:" << prim_py_->ToString();

  auto py_args = PreparePyInputs(prim_py_, args);

  auto pyobj = prim_py_->GetPyObj();
  if (pyobj == nullptr) {
    MS_LOG(EXCEPTION) << "[" << prim_py_->ToString() << "]: pyobj is empty";
  }
  auto infer_fuc = pyobj.attr("__infer__");

  py::dict output = infer_fuc(*py_args);
  MS_LOG(DEBUG) << "Output type is " << (std::string)py::str(output);
  auto res_spec = PyInferRes2Abstract(prim_py_, output);

  MS_LOG(DEBUG) << "Python InferTensor result spec: " << res_spec->ToString() << ".";
  return res_spec;
}

AbstractBasePtr UniformPrimEvaluator::EvalPrim(const AnalysisEnginePtr &, const AbstractBasePtrList &args) {
  // if func_desc_.retval type is super class of parameter type, then make the retval type as parameter type.
  if (nargs_ != args.size()) {
    MS_LOG(ERROR) << "UniformPrimEvaluator expect " << nargs_ << " args, but got " << args.size() << " inputs";
    return nullptr;
  }
  TypePtr ret_value_type = return_value_type_;
  ValuePtrList value_list;
  for (const auto &arg : args) {
    // Check if all arguments are scalar type.
    MS_EXCEPTION_IF_NULL(arg);
    if (arg->isa<AbstractScalar>()) {
      auto arg_scalar = dyn_cast<AbstractScalar>(arg);
      auto arg_value = arg_scalar->GetValueTrack();
      value_list.push_back(arg_value);
    } else {
      // Raise TypeError Expected Scalar.
      MS_LOG(EXCEPTION) << "Expect scalar arguments for uniform primitives.";
    }
  }
  for (const auto &item : type_map_) {
    TypePtrList selections;
    MS_EXCEPTION_IF_NULL(item.second);
    (void)std::transform(item.second->begin(), item.second->end(), std::back_inserter(selections),
                         [&args](size_t arg_idx) -> TypePtr { return args[arg_idx]->GetTypeTrack(); });
    TypePtr res = CheckTypeList(item.first, selections);
    if (*return_value_type_ == *(item.first)) {
      ret_value_type = res;
    }
  }

  ValuePtr inferred_value = RunImpl(value_list);
  if (!(*inferred_value == *kAnyValue)) {
    ret_value_type = inferred_value->type();
  }
  // for comparison primitives , return type shall have be specified to be bool.
  if (specify_out_type_ != nullptr) {
    ret_value_type = specify_out_type_;
  }

  AbstractScalarPtr abs_base = std::make_shared<AbstractScalar>(inferred_value, ret_value_type);
  return abs_base;
}

ValuePtr UniformPrimEvaluator::RunImpl(const ValuePtrList &args) const {
  if (!eval_value_) {
    return kAnyValue;
  } else {
    if (std::any_of(args.begin(), args.end(), [](const ValuePtr &arg) {
          MS_EXCEPTION_IF_NULL(arg);
          return arg->isa<AnyValue>();
        })) {
      return kAnyValue;
    }
    return impl_(args);
  }
}

// Primitive implementation
// static function start
namespace {
EvaluatorPtr InitStandardPrimEvaluator(PrimitivePtr primitive, const StandardPrimitiveEvalImpl eval_impl) {
  EvaluatorPtr prim_evaluator = std::make_shared<StandardPrimEvaluator>(primitive, eval_impl);
  return prim_evaluator;
}

EvaluatorPtr InitUniformPrimEvaluator(const PrimitivePtr &primitive, PrimitiveImpl prim_impl, bool eval_value,
                                      const TypePtr &specify_out_type) {
  FunctionPtr func = nullptr;
  (void)prim::PrimToFunction::GetInstance().GetFunction(primitive, &func);
  MS_EXCEPTION_IF_NULL(func);

  EvaluatorPtr uniform_primitive_evaluator =
    std::make_shared<UniformPrimEvaluator>(func, prim_impl, eval_value, specify_out_type);
  return uniform_primitive_evaluator;
}

const int kResolveCaseUserDefineClass = 1;
const int kResolveCaseBuildinTypeMethod = 2;
const int kResolveCaseFunction = 3;
int GetResolveCase(const TypePtr &data_type) {
  MS_EXCEPTION_IF_NULL(data_type);
  if (data_type->type_id() == kObjectTypeClass) {
    return kResolveCaseUserDefineClass;
  }

  // try method map, if not in method map, the data_type should be External type.
  if (pipeline::Resource::IsTypeInMethodMap(data_type->type_id())) {
    return kResolveCaseBuildinTypeMethod;
  }

  return kResolveCaseFunction;
}

FuncGraphPtr PyObjToGraph(const AnalysisEnginePtr &engine, const ValuePtr &method) {
  MS_EXCEPTION_IF_NULL(engine);
  MS_EXCEPTION_IF_NULL(method);
  if (!method->isa<parse::PyObjectWrapper>()) {
    MS_LOG(EXCEPTION) << "Method type error: " << method->ToString();
  }

  std::shared_ptr<PyObjectWrapper> obj = method->cast<std::shared_ptr<PyObjectWrapper>>();
  FuncGraphPtr func_graph = mindspore::parse::ConvertToFuncGraph(obj->obj());
  if (func_graph == nullptr) {
    MS_LOG(EXCEPTION) << "Parse python object: " << method->ToString() << " failed";
  }

  FuncGraphManagerPtr manager = engine->func_graph_manager();
  manager->AddFuncGraph(func_graph);
  return func_graph;
}

inline void AddToManager(const AnalysisEnginePtr &engine, const FuncGraphPtr func_graph) {
  MS_EXCEPTION_IF_NULL(engine);
  FuncGraphManagerPtr manager = engine->func_graph_manager();
  manager->AddFuncGraph(func_graph);
}

AbstractBasePtr StaticGetterInferred(const ValuePtr &value, const ConfigPtr &data_conf,
                                     const AnfNodeConfigPtr &old_conf) {
  MS_EXCEPTION_IF_NULL(old_conf);

  AbstractBasePtr abs_ptr = ToAbstract(value, AnalysisContext::DummyContext(), old_conf);
  AbstractFunctionPtr abs_func = dyn_cast<abstract::AbstractFunction>(abs_ptr);
  MS_EXCEPTION_IF_NULL(abs_func);

  // Create new cnode
  std::vector<AnfNodePtr> input = {NewValueNode(prim::kPrimPartial)};
  auto func_graph_func = dyn_cast<abstract::FuncGraphAbstractClosure>(abs_func);
  if (func_graph_func != nullptr) {
    FuncGraphPtr fg = func_graph_func->func_graph();
    input.push_back(NewValueNode(fg));
  } else {
    auto prim_func = dyn_cast<abstract::PrimitiveAbstractClosure>(abs_func);
    MS_EXCEPTION_IF_NULL(prim_func);
    PrimitivePtr prim = prim_func->prim();
    input.push_back(NewValueNode(prim));
  }

  AnfNodeConfigPtr conf = dyn_cast<abstract::AnfNodeConfig>(data_conf);
  MS_EXCEPTION_IF_NULL(conf);
  input.push_back(conf->node());
  MS_EXCEPTION_IF_NULL(old_conf);
  FuncGraphPtr func_graph = old_conf->node()->func_graph();
  CNodePtr new_cnode = func_graph->NewCNode(input);
  AnalysisEnginePtr eng = old_conf->engine();
  AnfNodeConfigPtr fn_conf = eng->MakeConfig(new_cnode, old_conf->context());
  return eng->ForwardConfig(old_conf, fn_conf);
}

AbstractBasePtr GetEvaluatedValueForNameSpaceString(const AnalysisEnginePtr &engine,
                                                    const AbstractBasePtrList &args_spec_list,
                                                    const AnfNodeConfigPtr &out_conf) {
  // args_spec_list: same as StaticGetter
  if (args_spec_list.size() < 2) {
    MS_LOG(EXCEPTION) << "Size of args_spec_list is less than 2";
  }
  MS_EXCEPTION_IF_NULL(out_conf);
  // An external type.
  MS_EXCEPTION_IF_NULL(args_spec_list[0]);
  MS_EXCEPTION_IF_NULL(args_spec_list[1]);
  MS_LOG(DEBUG) << "Args[0]: " << args_spec_list[0]->ToString();
  MS_LOG(DEBUG) << "Args[1]: " << args_spec_list[1]->ToString();
  auto data_v = args_spec_list[0]->BuildValue();
  if (!data_v->isa<parse::NameSpace>()) {
    MS_LOG(EXCEPTION) << "Data is not NameSpace : " << data_v->ToString();
  }

  auto item_v = args_spec_list[1]->BuildValue();
  if (item_v->isa<StringImm>()) {
    item_v = std::make_shared<parse::Symbol>(item_v->cast<StringImmPtr>()->value());
  }

  if (!item_v->isa<parse::Symbol>()) {
    MS_LOG(EXCEPTION) << "The value of the attribute could not be inferred: " << item_v->ToString();
  }

  // item_name to func addr from obj_map
  parse::SymbolPtr symbol = item_v->cast<parse::SymbolPtr>();
  parse::NameSpacePtr name_space = data_v->cast<parse::NameSpacePtr>();
  FuncGraphPtr func_graph = out_conf->node()->func_graph();

  auto new_node = parse::ResolveSymbol(func_graph->manager(), name_space, symbol, out_conf->node());
  if (new_node == nullptr) {
    MS_LOG(EXCEPTION) << "Resolve node failed";
  }

  AnalysisEnginePtr eng = out_conf->engine();
  AnfNodeConfigPtr fn_conf = eng->MakeConfig(new_node, out_conf->context());
  return eng->ForwardConfig(out_conf, fn_conf);
}

AbstractBasePtr GetEvaluatedValueForClassAttrOrMethod(const AnalysisEnginePtr &engine,
                                                      const AbstractBasePtrList &args_spec_list, const ValuePtr &item_v,
                                                      const ConfigPtr &data_conf, const AnfNodeConfigPtr &out_conf) {
  if (args_spec_list.empty()) {
    MS_LOG(EXCEPTION) << "args_spec_list is empty";
  }
  AbstractClassPtr cls = CheckArg<AbstractClass>("__FUNC__", args_spec_list, 0);

  // If item_v is an attribute, get abstract value from AbstractClass
  MS_EXCEPTION_IF_NULL(item_v);
  if (!item_v->isa<StringImm>()) {
    MS_LOG(EXCEPTION) << "Attribute type error";
  }
  std::string item_name = item_v->cast<StringImmPtr>()->value();
  MS_LOG(DEBUG) << "Resolve name: " << cls->tag().name();
  MS_LOG(DEBUG) << "Resolve item: " << item_name;

  AbstractBasePtr attr = cls->GetAttribute(item_name);
  if (attr != nullptr) {
    return attr;
  }

  ValuePtr method = cls->GetMethod(item_name);
  if (method->isa<AnyValue>()) {
    MS_LOG(EXCEPTION) << "Unknown field, data type: " << args_spec_list[0]->BuildType()->ToString()
                      << ", item value: " << item_v->ToString();
  }

  // Infer class method
  ValuePtr converted_v = PyObjToGraph(engine, method);
  return StaticGetterInferred(converted_v, data_conf, out_conf);
}

AbstractBasePtr GetEvaluatedValueForBuiltinTypeMethod(const AnalysisEnginePtr &engine, const ValuePtr &item_v,
                                                      const TypePtr &data_type, const ConfigPtr &data_conf,
                                                      const AnfNodeConfigPtr &out_conf) {
  MS_EXCEPTION_IF_NULL(item_v);
  MS_EXCEPTION_IF_NULL(data_type);
  // The method maybe a Primitive or Composite
  if (!item_v->isa<StringImm>()) {
    MS_LOG(EXCEPTION) << "Error item is not string";
  }

  std::string item_name = item_v->cast<StringImmPtr>()->value();
  Any method = pipeline::Resource::GetMethodPtr(data_type->type_id(), item_name);
  if (method.empty()) {
    MS_LOG(EXCEPTION) << "Object type: " << data_type->ToString() << " has no method: " << item_name;
  }

  ValuePtr converted_v = nullptr;
  if (method.is<std::string>()) {
    // composite registered in standard_method_map go to this branch
    converted_v = prim::GetPythonOps(method.cast<std::string>());
    AddToManager(engine, converted_v->cast<FuncGraphPtr>());
  } else if (method.is<PrimitivePtr>()) {
    converted_v = method.cast<PrimitivePtr>();
  } else {
    MS_LOG(EXCEPTION) << "Expect to get string or PrimitivePtr from method map, but got " << method.ToString();
  }
  return StaticGetterInferred(converted_v, data_conf, out_conf);
}

AbstractBasePtr StaticGetter(const AnalysisEnginePtr &engine, const AbstractBasePtrList &args_spec_list,
                             const ConfigPtr &data_conf, const AnfNodeConfigPtr &out_conf) {
  // Inputs: namespace and its static function; or class and its member function
  CheckArgsSize("StaticGetter", args_spec_list, 2);

  MS_EXCEPTION_IF_NULL(args_spec_list[0]);
  MS_EXCEPTION_IF_NULL(args_spec_list[1]);
  TypePtr data_type = args_spec_list[0]->BuildType();
  ValuePtr item_value = args_spec_list[1]->BuildValue();
  ScopePtr scope = kDefaultScope;
  if (out_conf != nullptr) {
    scope = out_conf->node()->scope();
  }
  ScopeGuard scope_guard(scope);
  if (item_value->isa<AnyValue>()) {
    MS_LOG(EXCEPTION) << "The value of the attribute could not be inferred: " << item_value->ToString();
  }

  int case_v = GetResolveCase(data_type);
  if (case_v == kResolveCaseUserDefineClass) {
    return GetEvaluatedValueForClassAttrOrMethod(engine, args_spec_list, item_value, data_conf, out_conf);
  } else if (case_v == kResolveCaseBuildinTypeMethod) {
    return GetEvaluatedValueForBuiltinTypeMethod(engine, item_value, data_type, data_conf, out_conf);
  } else {
    return GetEvaluatedValueForNameSpaceString(engine, args_spec_list, out_conf);
  }
}
}  // end anonymous namespace

// static variable start;
namespace {
class EmbedEvaluator : public SymbolicPrimEvaluator {
 public:
  EmbedEvaluator() : SymbolicPrimEvaluator("EmbedEvaluator") {}
  ~EmbedEvaluator() override = default;
  MS_DECLARE_PARENT(EmbedEvaluator, SymbolicPrimEvaluator);
  AbstractBasePtr EvalPrim(const ConfigPtrList &args_conf_list) override {
    // arg: free variable to be embedded
    if (args_conf_list.size() != 1) {
      MS_LOG(EXCEPTION) << "EmbedEvaluator requires 1 parameter, but got " << args_conf_list.size();
    }
    AnfNodeConfigPtr node_conf = dyn_cast<AnfNodeConfig>(args_conf_list[0]);
    MS_EXCEPTION_IF_NULL(node_conf);

    AbstractBasePtr x = node_conf->GetEvaluatedValue();
    x = SensitivityTransform(x);
    SymbolicKeyInstancePtr key = std::make_shared<SymbolicKeyInstance>(node_conf->node(), x);
    AbstractScalarPtr abs_scalar = std::make_shared<AbstractScalar>(key, std::make_shared<SymbolicKeyType>());
    return abs_scalar;
  }
};

static AnfNodePtr FindParameterNodeByString(const FuncGraphManagerPtr &manager, const std::string &name) {
  auto root_g_set = manager->roots();
  if (root_g_set.size() != 1) {
    return nullptr;
  }
  const FuncGraphPtr &root_g = root_g_set.back();

  for (auto &param_node : root_g->parameters()) {
    auto param = param_node->cast<ParameterPtr>();
    if (param && name == param->name()) {
      return param;
    }
  }
  return nullptr;
}

class RefToEmbedEvaluator : public SymbolicPrimEvaluator {
 public:
  RefToEmbedEvaluator() : SymbolicPrimEvaluator("RefToEmbedEvaluator") {}
  ~RefToEmbedEvaluator() override = default;
  MS_DECLARE_PARENT(RefToEmbedEvaluator, SymbolicPrimEvaluator);
  AbstractBasePtr EvalPrim(const ConfigPtrList &args_conf_list) override {
    if (args_conf_list.size() != 1) {
      MS_LOG(ERROR) << "Requires 1 parameter, but has: " << args_conf_list.size();
      return nullptr;
    }
    static TypePtr type = std::make_shared<SymbolicKeyType>();
    auto node_conf = dyn_cast<AnfNodeConfig>(args_conf_list[0]);
    if (node_conf == nullptr) {
      MS_LOG(ERROR) << "Conf should be AnfNodeConfig";
      return nullptr;
    }
    AbstractBasePtr abs = node_conf->GetEvaluatedValue();
    AbstractRefPtr ref_abs = abs->cast<AbstractRefPtr>();
    if (ref_abs == nullptr) {
      MS_LOG(ERROR) << "The first parameter of RefToEmbed should be Ref.";
      return nullptr;
    }
    auto key_abs = ref_abs->ref_key();
    if (key_abs == nullptr) {
      MS_LOG(ERROR) << "RefToEmbed input Ref key is nullptr.";
      return nullptr;
    }
    auto key_value = key_abs->BuildValue();
    if (key_value == nullptr) {
      MS_LOG(ERROR) << "RefToEmbed input Ref key value is nullptr.";
      return nullptr;
    }
    auto refkey = key_value->cast<RefKeyPtr>();
    if (refkey == nullptr) {
      return std::make_shared<AbstractScalar>(type);
    }

    std::string name = refkey->tag();
    const auto &manager = node_conf->node()->func_graph()->manager();
    auto node = FindParameterNodeByString(manager, name);
    if (node == nullptr) {
      MS_LOG(ERROR) << "RefToEmbed input can't find parameter \"" << name << "\" in graph.";
      return nullptr;
    }
    AbstractBasePtr x = ref_abs->ref();
    x = SensitivityTransform(x);
    std::shared_ptr<SymbolicKeyInstance> key = std::make_shared<SymbolicKeyInstance>(node, x);
    std::shared_ptr<AbstractScalar> abs_scalar = std::make_shared<AbstractScalar>(key, type);
    return abs_scalar;
  }
};

class GetAttrEvaluator : public TransitionPrimEvaluator {
 public:
  GetAttrEvaluator() : TransitionPrimEvaluator("GetAttrEvaluator") {}
  ~GetAttrEvaluator() override = default;
  MS_DECLARE_PARENT(GetAttrEvaluator, TransitionPrimEvaluator);
  AbstractBasePtr EvalPrim(const AnalysisEnginePtr &engine, const AbstractBasePtrList &args_spec_list,
                           const ConfigPtr &in_conf0, const AnfNodeConfigPtr &out_conf) override {
    // Inputs: data, item
    if (args_spec_list.size() != 2) {
      MS_LOG(EXCEPTION) << "Expected args_spec_list size = 2, but has size:" << args_spec_list.size();
    }
    AbstractBasePtr ret = nullptr;
    if (bound_node() != nullptr) {
      TraceManager::DebugTrace(std::make_shared<TraceResolve>(bound_node()->debug_info()));
      ret = StaticGetter(engine, args_spec_list, in_conf0, out_conf);
      TraceManager::EndTrace();
    } else {
      ret = StaticGetter(engine, args_spec_list, in_conf0, out_conf);
    }
    // don't lookup from cache, as different out_conf with same node but different context
    // may add different entry to anfnode_config_map, like getattr primitive;
    (*cache_)[args_spec_list] = ret;
    return ret;
  }
};

class ResolveEvaluator : public TransitionPrimEvaluator {
 public:
  ResolveEvaluator() : TransitionPrimEvaluator("ResolveEvaluator") {}
  ~ResolveEvaluator() override = default;
  MS_DECLARE_PARENT(ResolveEvaluator, TransitionPrimEvaluator);
  AbstractBasePtr EvalPrim(const AnalysisEnginePtr &engine, const AbstractBasePtrList &args_spec_list,
                           const ConfigPtr &in_conf0, const AnfNodeConfigPtr &out_conf) override {
    // Inputs: namespace, symbol
    if (args_spec_list.size() != 2) {
      MS_LOG(EXCEPTION) << "Expected args_spec_list size = 2, but has size:" << args_spec_list.size();
    }
    AbstractBasePtr ret = nullptr;
    if (bound_node() != nullptr) {
      TraceManager::DebugTrace(std::make_shared<TraceResolve>(bound_node()->debug_info()));
      ret = StaticGetter(engine, args_spec_list, in_conf0, out_conf);
      TraceManager::EndTrace();
    } else {
      ret = StaticGetter(engine, args_spec_list, in_conf0, out_conf);
    }
    return ret;
  }
};

class CreateInstanceEvaluator : public TransitionPrimEvaluator {
 public:
  CreateInstanceEvaluator() : TransitionPrimEvaluator("CreateInstanceEvaluator") {}
  ~CreateInstanceEvaluator() override = default;
  MS_DECLARE_PARENT(CreateInstanceEvaluator, TransitionPrimEvaluator);
  AbstractBasePtr EvalPrim(const AnalysisEnginePtr &engine, const AbstractBasePtrList &args_spec_list,
                           const ConfigPtr &, const AnfNodeConfigPtr &out_conf) override {
    if (args_spec_list.empty()) {
      MS_LOG(EXCEPTION) << "'args_spec_list' should not be empty";
    }

    // get the type parameter
    MS_EXCEPTION_IF_NULL(args_spec_list[0]);
    TypePtr type = args_spec_list[0]->GetTypeTrack();
    if (type->type_id() != kMetaTypeTypeType) {
      MS_LOG(EXCEPTION) << "CreateInstanceEvaluator require first parameter should be an object of TypeType, but got "
                        << type->ToString();
    }

    ValuePtr value_track = args_spec_list[0]->GetValueTrack();
    MS_EXCEPTION_IF_NULL(value_track);

    std::shared_ptr<parse::PyObjectWrapper> type_obj = dyn_cast<parse::PyObjectWrapper>(value_track);
    if (type_obj == nullptr) {
      MS_LOG(EXCEPTION) << "Cast value failed, not PyObjectWrapper:" << value_track->ToString() << ".";
    }

    if (!type_obj->isa<parse::ClassType>()) {
      MS_LOG(EXCEPTION) << "CreateInstanceEvaluator the type_obj should be an object of ClassType, but got "
                        << type_obj->ToString() << ".";
    }

    auto class_type = type_obj->obj();
    MS_LOG(DEBUG) << "Get class type is " << type_obj->ToString() << ".";

    // get the create instance obj's parameters
    pybind11::tuple params = GetParameters(args_spec_list);

    // create class instance
    auto obj = parse::data_converter::CreatePythonObject(class_type, params);
    if (py::isinstance<py::none>(obj)) {
      MS_LOG(EXCEPTION) << "Create python object failed, only support Cell and Primitive type";
    }

    // process the object
    ValuePtr converted_ret = nullptr;
    bool converted = parse::ConvertData(obj, &converted_ret, true);
    if (!converted) {
      MS_LOG(EXCEPTION) << "Convert the python object failed";
    }
    MS_EXCEPTION_IF_NULL(converted_ret);

    if (converted_ret->isa<FuncGraph>()) {
      AddToManager(engine, converted_ret->cast<FuncGraphPtr>());
    }

    AbstractBasePtr ret = ToAbstract(converted_ret, AnalysisContext::DummyContext(), out_conf);
    (*cache_)[args_spec_list] = ret;
    return ret;
  }

  pybind11::tuple GetParameters(const AbstractBasePtrList &args_spec_list) const {
    // Exclude class type by minus 1;
    std::size_t params_size = args_spec_list.size() - 1;
    auto params = py::tuple(params_size);
    if (params_size > 0) {
      for (size_t i = 0; i < params_size; i++) {
        // Only support the Scalar parameters type. Bypass class type by offset with 1.
        auto arg = args_spec_list[i + 1];
        MS_EXCEPTION_IF_NULL(arg);
        // Because the Tensor's AbstractTensor can't get value from GetValueTrack.
        ValuePtr param_value = arg->BuildValue();
        py::object param = ValuePtrToPyData(param_value);
        params[i] = param;
      }
    }
    return params;
  }
};

class PartialEvaluator : public Evaluator {
 public:
  PartialEvaluator() : Evaluator("PartialEvaluator") {}
  ~PartialEvaluator() override = default;
  AbstractBasePtr Run(AnalysisEnginePtr engine, const ConfigPtrList &args_conf_list,
                      AnfNodeConfigPtr out_conf = nullptr) override {
    if (args_conf_list.size() == 0) {
      MS_LOG(EXCEPTION) << "Args size should be greater than 0";
    }
    auto arg0_value = args_conf_list[0]->GetEvaluatedValue();
    AbstractBasePtrList args_spec_list{arg0_value};
    auto func = CheckArg<AbstractFunction>("partial", args_spec_list, 0);
    // Sometimes, node[0] in out_conf becomes phi0;
    if (func->isa<PrimitiveAbstractClosure>()) {
      auto prim_func = dyn_cast<PrimitiveAbstractClosure>(func);
      if (prim_func->prim()->isa<prim::DoSignaturePrimitive>()) {
        prim::DoSignaturePrimitivePtr do_signature_prim = dyn_cast<prim::DoSignaturePrimitive>(prim_func->prim());
        return HandleDoSignature(engine, do_signature_prim->function(), out_conf);
      }
    }
    (void)std::transform(args_conf_list.begin() + 1, args_conf_list.end(), std::back_inserter(args_spec_list),
                         [](const ConfigPtr &ref) -> AbstractBasePtr { return ref->GetEvaluatedValue(); });

    AbstractBasePtrList args(args_spec_list.begin() + 1, args_spec_list.end());

    AbstractFuncAtomPtrList partialPtrList;
    auto build_partial = [args, &partialPtrList](const AbstractFuncAtomPtr &atom_func) {
      auto new_func = std::make_shared<PartialAbstractClosure>(atom_func, args);
      partialPtrList.push_back(new_func);
    };
    func->Visit(build_partial);

    auto ret = AbstractFunction::MakeAbstractFunction(partialPtrList);
    (*cache_)[args_spec_list] = ret;
    return ret;
  }

  AbstractBasePtr Infer(AnalysisEnginePtr, const AbstractBasePtrList &) override {
    MS_LOG(EXCEPTION) << "Infer() should not be called, Run() method should be called";
  }

  AbstractBasePtr HandleDoSignature(const AnalysisEnginePtr &engine, const ValuePtr &signature_value,
                                    const AnfNodeConfigPtr &out_conf = nullptr) const {
    MS_EXCEPTION_IF_NULL(out_conf);
    MS_EXCEPTION_IF_NULL(out_conf->node());
    auto cnode = out_conf->node()->cast<CNodePtr>();
    if (cnode == nullptr) {
      MS_LOG(EXCEPTION) << "Cnode is nullptr";
    }
    std::vector<AnfNodePtr> new_nodes_inputs = cnode->inputs();
    auto new_signature_value = std::make_shared<prim::DoSignatureMetaFuncGraph>("signature", signature_value);
    new_nodes_inputs[1] = NewValueNode(new_signature_value);
    FuncGraphPtr func_graph = cnode->func_graph();

    ScopePtr scope = out_conf->node()->scope();
    ScopeGuard scope_guard(scope);

    CNodePtr new_cnode = func_graph->NewCNode(new_nodes_inputs);
    AnfNodeConfigPtr fn_conf = engine->MakeConfig(new_cnode, out_conf->context());
    return engine->ForwardConfig(out_conf, fn_conf);
  }
};

struct PrimitiveImplInferValue {
  PrimitiveImpl impl_;        // implement function of primitive
  bool eval_value_;           // whether evaluate value
  TypePtr specify_out_type_;  // whether specify return type
  bool in_white_list_;        // true if this Primitive in white list, else false.
};

using PrimitiveToImplMap = std::unordered_map<PrimitivePtr, PrimitiveImplInferValue, PrimitiveHasher, PrimitiveEqual>;
PrimitiveToImplMap &GetUniformPrimitiveToImplMap() {
  static PrimitiveToImplMap uniform_prim_implement_map = {
    {prim::kPrimScalarAdd, {prim::ScalarAdd, true, nullptr, true}},
    {prim::kPrimScalarSub, {prim::ScalarSub, true, nullptr, true}},
    {prim::kPrimScalarMul, {prim::ScalarMul, true, nullptr, true}},
    {prim::kPrimScalarDiv, {prim::ScalarDiv, true, nullptr, true}},
    {prim::kPrimScalarMod, {prim::ScalarMod, true, nullptr, true}},
    {prim::kPrimScalarPow, {prim::ScalarPow, true, nullptr, true}},
    {prim::kPrimScalarFloordiv, {prim::ScalarFloordiv, true, nullptr, true}},
    {prim::kPrimScalarUadd, {prim::ScalarUAdd, true, nullptr, true}},
    {prim::kPrimScalarUsub, {prim::ScalarUSub, true, nullptr, true}},
    {prim::kPrimScalarLog, {prim::ScalarLog, true, nullptr, true}},
    {prim::kPrimScalarEq, {prim::ScalarEq, true, std::make_shared<Bool>(), true}},
    {prim::kPrimScalarLt, {prim::ScalarLt, true, std::make_shared<Bool>(), true}},
    {prim::kPrimScalarGt, {prim::ScalarGt, true, std::make_shared<Bool>(), true}},
    {prim::kPrimScalarNe, {prim::ScalarNe, true, std::make_shared<Bool>(), true}},
    {prim::kPrimScalarLe, {prim::ScalarLe, true, std::make_shared<Bool>(), true}},
    {prim::kPrimScalarGe, {prim::ScalarGe, true, std::make_shared<Bool>(), true}},
    {prim::kPrimBoolNot, {prim::BoolNot, true, std::make_shared<Bool>(), true}},
    {prim::kPrimBoolAnd, {prim::BoolAnd, true, std::make_shared<Bool>(), true}},
    {prim::kPrimBoolEq, {prim::BoolEq, true, std::make_shared<Bool>(), true}},
    {prim::kPrimBoolOr, {prim::BoolOr, true, std::make_shared<Bool>(), true}},
  };
  return uniform_prim_implement_map;
}

PrimEvaluatorMap PrimEvaluatorConstructors = PrimEvaluatorMap();
std::mutex PrimEvaluatorConstructorMutex;

void InitPrimEvaluatorConstructors() {
  PrimEvaluatorMap &constructor = PrimEvaluatorConstructors;

  for (const auto &iter : GetPrimitiveToEvalImplMap()) {
    constructor[iter.first] = InitStandardPrimEvaluator(iter.first, iter.second.impl_);
  }

  for (const auto &iter : GetUniformPrimitiveToImplMap()) {
    constructor[iter.first] =
      InitUniformPrimEvaluator(iter.first, iter.second.impl_, iter.second.eval_value_, iter.second.specify_out_type_);
  }
  constructor[prim::kPrimEmbed] = std::make_shared<EmbedEvaluator>();
  constructor[prim::kPrimRefToEmbed] = std::make_shared<RefToEmbedEvaluator>();
  constructor[prim::kPrimGetAttr] = std::make_shared<GetAttrEvaluator>();
  constructor[prim::kPrimResolve] = std::make_shared<ResolveEvaluator>();
  constructor[prim::kPrimCreateInstance] = std::make_shared<CreateInstanceEvaluator>();
  constructor[prim::kPrimPartial] = std::make_shared<PartialEvaluator>();
}
}  // namespace

void ClearPrimEvaluatorMap() {
  PrimEvaluatorConstructors.clear();
  GetPrimitiveToEvalImplMap().clear();
  GetUniformPrimitiveToImplMap().clear();
}

bool IsInWhiteList(const PrimitivePtr primitive) {
  MS_EXCEPTION_IF_NULL(primitive);

  auto iter = GetPrimitiveToEvalImplMap().find(primitive);
  if (iter != GetPrimitiveToEvalImplMap().end()) {
    return iter->second.in_white_list_;
  }

  auto uni_iter = GetUniformPrimitiveToImplMap().find(primitive);
  if (uni_iter != GetUniformPrimitiveToImplMap().end()) {
    return uni_iter->second.in_white_list_;
  }

  return false;
}

StandardPrimitiveEvalImpl GetPrimitiveInferImpl(const PrimitivePtr &primitive) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto iter = GetPrimitiveToEvalImplMap().find(primitive);
  if (iter == GetPrimitiveToEvalImplMap().end()) {
    return nullptr;
  }
  return iter->second.impl_;
}

PrimEvaluatorMap &GetPrimEvaluatorConstructors() {
  PrimEvaluatorMap &constructor = PrimEvaluatorConstructors;
  if (!constructor.empty()) {
    return constructor;
  }
  std::lock_guard<std::mutex> initLock(PrimEvaluatorConstructorMutex);
  if (constructor.empty()) {
    InitPrimEvaluatorConstructors();
  }

  return constructor;
}

namespace {
bool IsSubtypeTuple(const AbstractBasePtr x, const TypePtr model) {
  MS_EXCEPTION_IF_NULL(x);
  MS_EXCEPTION_IF_NULL(model);
  auto x_tuple = dyn_cast<AbstractTuple>(x);
  auto model_tuple = dyn_cast<Tuple>(model);

  if (x_tuple == nullptr || model_tuple == nullptr) {
    return false;
  }

  if (model->IsGeneric()) {
    return true;
  }

  if (x_tuple->size() != model_tuple->size()) {
    return false;
  }

  for (size_t i = 0; i < x_tuple->size(); i++) {
    bool is_subtype = IsSubtype((*x_tuple)[i], (*model_tuple)[i]);
    if (!is_subtype) {
      return false;
    }
  }
  return true;
}

bool IsSubtypeArray(const AbstractBasePtr x, const TypePtr model) {
  MS_EXCEPTION_IF_NULL(x);
  MS_EXCEPTION_IF_NULL(model);
  auto x_tensor = dyn_cast<AbstractTensor>(x);
  auto model_tensor = dyn_cast<TensorType>(model);

  if (x_tensor == nullptr || model_tensor == nullptr) {
    return false;
  }

  if (model->IsGeneric()) {
    return true;
  }

  return IsSubtype(x_tensor->element(), model_tensor->element());
}

bool IsSubtypeList(const AbstractBasePtr x, const TypePtr model) {
  MS_EXCEPTION_IF_NULL(x);
  MS_EXCEPTION_IF_NULL(model);
  auto x_list = dyn_cast<AbstractList>(x);
  auto model_list = dyn_cast<List>(model);

  if (x_list == nullptr || model_list == nullptr) {
    return false;
  }

  if (model->IsGeneric()) {
    return true;
  }

  if (x_list->size() != model_list->size()) {
    return false;
  }

  bool is_subtype = true;
  for (size_t i = 0; i < x_list->size(); i++) {
    is_subtype = IsSubtype((*x_list)[i], (*model_list)[i]);
    if (!is_subtype) {
      return false;
    }
  }
  return is_subtype;
}

bool IsSubtypeClass(const AbstractBasePtr x, const TypePtr model) {
  MS_EXCEPTION_IF_NULL(x);
  MS_EXCEPTION_IF_NULL(model);
  auto x_class = dyn_cast<AbstractClass>(x);
  auto model_class = dyn_cast<Class>(model);
  if (x_class == nullptr) {
    return false;
  }
  if (model->IsGeneric()) {
    return true;
  }

  if (x_class->tag() == model_class->tag()) {
    auto m_attributes = model_class->GetAttributes();
    auto x_attributes = x_class->attributes();
    if (m_attributes.size() != x_attributes.size()) {
      return false;
    }

    for (size_t i = 0; i < m_attributes.size(); i++) {
      if (!IsSubtype(x_attributes[i].second, m_attributes[i].second)) {
        return false;
      }
    }
    return true;
  }

  return false;
}

inline bool IsSubtypeScalar(const AbstractBasePtr x, const TypePtr model) {
  MS_EXCEPTION_IF_NULL(x);
  MS_EXCEPTION_IF_NULL(model);
  if (dyn_cast<AbstractScalar>(x) == nullptr) {
    return false;
  }
  TypePtr x_type = x->GetTypeTrack();
  return IsSubType(x_type, model);
}
}  // namespace

bool IsSubtype(const AbstractBasePtr x, const TypePtr model) {
  MS_EXCEPTION_IF_NULL(x);
  MS_EXCEPTION_IF_NULL(model);
  TypeId model_typeid = model->type_id();
  switch (model_typeid) {
    case kMetaTypeObject:
      return true;
    case kObjectTypeTuple:
      return IsSubtypeTuple(x, model);
    case kObjectTypeTensorType:
      return IsSubtypeArray(x, model);
    case kObjectTypeList:
      return IsSubtypeList(x, model);
    case kObjectTypeClass:
      return IsSubtypeClass(x, model);
    default:
      if (IsSubType(model, std::make_shared<Number>())) {
        return IsSubtypeScalar(x, model);
      }
      MS_LOG(EXCEPTION) << "Invalid model type: " << model->ToString() << ".";
  }
}
}  // namespace abstract
}  // namespace mindspore
