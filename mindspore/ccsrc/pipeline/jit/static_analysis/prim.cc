/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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

#include "pipeline/jit/static_analysis/prim.h"

#include <algorithm>
#include <limits>
#include <mutex>
#include <string>
#include <utility>
#include <unordered_set>

#include "frontend/operator/cc_implementations.h"
#include "frontend/operator/ops.h"
#include "frontend/operator/composite/do_signature.h"
#include "frontend/operator/prim_to_function.h"
#include "abstract/utils.h"
#include "utils/symbolic.h"
#include "pipeline/jit/resource.h"
#include "pipeline/jit/parse/resolve.h"
#include "utils/convert_utils.h"
#include "utils/convert_utils_py.h"
#include "utils/ms_context.h"
#include "pipeline/jit/parse/data_converter.h"
#include "abstract/primitive_infer_map.h"
#include "abstract/param_validator.h"
#include "utils/ms_utils.h"
#include "utils/shape_utils.h"

namespace mindspore {
namespace abstract {
using mindspore::parse::PyObjectWrapper;

std::unordered_set<std::string> prims_to_skip_undetermined_infer{
  "MakeTuple", "make_list", "Switch", "env_setitem", "env_getitem", "Load", "UpdateState"};

EvalResultPtr DoSignatureEvaluator::Run(AnalysisEnginePtr engine, const ConfigPtrList &args_conf_list,
                                        AnfNodeConfigPtr out_conf) {
  AbstractBasePtrList args_spec_list;
  (void)std::transform(args_conf_list.begin(), args_conf_list.end(), std::back_inserter(args_spec_list),
                       [](const ConfigPtr &ref) -> AbstractBasePtr { return ref->ObtainEvalResult()->abstract(); });
  auto do_signature = prim_->cast<prim::DoSignaturePrimitivePtr>();
  auto &func = do_signature->function();
  if (func->isa<Primitive>()) {
    auto sig_prim = func->cast<PrimitivePtr>();
    if (prims_to_skip_undetermined_infer.find(sig_prim->name()) == prims_to_skip_undetermined_infer.end()) {
      auto ret_abstract = AbstractEval(args_spec_list);
      if (ret_abstract != nullptr) {
        MS_LOG(DEBUG) << "DoSignatureEvaluator eval Undetermined";
        return ret_abstract;
      }
    }
  }

  if (out_conf->node() == nullptr || !out_conf->node()->isa<CNode>()) {
    MS_LOG(EXCEPTION) << "Node of out_conf should be CNode";
  }

  auto out_node = dyn_cast<CNode>(out_conf->node());
  const auto &out_node_inputs = out_node->inputs();
  if (out_node->inputs().size() == 0 || (out_node_inputs.size() - 1) != args_conf_list.size()) {
    MS_LOG(EXCEPTION) << "Op: " << do_signature->function()->ToString()
                      << " args size should equal to inputs size minus 1, but args size " << args_conf_list.size()
                      << ", inputs size " << out_node_inputs.size();
  }
  AnfNodePtrList args_inputs{out_node_inputs.begin() + 1, out_node_inputs.end()};

  ScopePtr scope = kDefaultScope;
  if (out_conf != nullptr) {
    scope = out_conf->node()->scope();
  }
  ScopeGuard scope_guard(scope);

  AnfNodePtr new_cnode = nullptr;
  if (bound_node() != nullptr) {
    TraceGuard trace_guard(std::make_shared<TraceDoSignature>(bound_node()->debug_info()));
    new_cnode = prim::GenerateCNode(out_node->func_graph(), prim_->ToString(), do_signature->function(), args_spec_list,
                                    args_inputs);
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
        auto arg_tuple = specialize_args_before_unpack[index]->cast<AbstractTuplePtr>();
        std::transform(arg_tuple->elements().begin(), arg_tuple->elements().end(),
                       std::back_inserter(graph_specialize_args), [](AbstractBasePtr abs) { return abs; });
      } else if (specialize_args_before_unpack[index]->isa<AbstractDictionary>()) {
        auto arg_dict = specialize_args_before_unpack[index]->cast<AbstractDictionaryPtr>();
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

EvalResultPtr UnpackGraphEvaluator::Run(AnalysisEnginePtr engine, const ConfigPtrList &args_conf_list,
                                        AnfNodeConfigPtr out_conf) {
  if (out_conf->node() == nullptr || !out_conf->node()->isa<CNode>()) {
    MS_LOG(EXCEPTION) << "Node of out_conf should be CNode";
  }

  auto unpack_graph = prim_->cast<prim::UnpackGraphPrimitivePtr>();
  auto out_node = out_conf->node()->cast<CNodePtr>();
  const auto &out_node_inputs = out_node->inputs();
  if (out_node->inputs().empty() || (out_node_inputs.size() - 1) != args_conf_list.size()) {
    MS_LOG(EXCEPTION) << "UnpackGraphPrimitive"
                      << " args size should equal to inputs size minus 1, but args size " << args_conf_list.size()
                      << ", inputs size " << out_node_inputs.size();
  }
  AnfNodePtrList args_inputs{out_node_inputs.begin() + 1, out_node_inputs.end()};
  AbstractBasePtrList args_spec_list;
  (void)std::transform(args_conf_list.begin(), args_conf_list.end(), std::back_inserter(args_spec_list),
                       [](const ConfigPtr &ref) -> AbstractBasePtr { return ref->ObtainEvalResult()->abstract(); });
  // get the forward graph
  MS_EXCEPTION_IF_NULL(args_spec_list[0]);
  auto fn = args_spec_list[0]->cast<AbstractFunctionPtr>();
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
  if (unpack_graph->with_sens_in_args() && graph_specialize_args.empty()) {
    MS_EXCEPTION(ValueError) << "Grad with sens, but the sens is not provided.";
  }
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

AnfNodePtr MixedPrecisionCastHelper(const AnfNodePtr &source_node, const AbstractBasePtr &node_type,
                                    const AnfNodePtr &target_type, const FuncGraphPtr &func_graph) {
  AnfNodePtr target_node = source_node;
  if (node_type->isa<AbstractTensor>()) {
    auto x = node_type->cast<AbstractTensorPtr>();
    if (x->element()->BuildType()->isa<Float>()) {
      auto cast = prim::GetPythonOps("cast", "mindspore.ops.functional");
      MS_EXCEPTION_IF_NULL(cast);
      target_node = func_graph->NewCNodeAfter(source_node, {NewValueNode(cast), source_node, target_type});
    }
  } else if (node_type->isa<AbstractTuple>()) {
    auto x = node_type->cast<AbstractTuplePtr>();
    auto &items = x->elements();
    std::vector<AnfNodePtr> nodes;
    nodes.emplace_back(NewValueNode(prim::kPrimMakeTuple));
    int64_t idx = 0;
    for (const auto &item : items) {
      AnfNodePtr tuple_node =
        func_graph->NewCNode({NewValueNode(prim::kPrimTupleGetItem), source_node, NewValueNode(idx)});
      AnfNodePtr node = MixedPrecisionCastHelper(tuple_node, item, target_type, func_graph);
      nodes.emplace_back(node);
      ++idx;
    }
    target_node = func_graph->NewCNode(nodes);
  } else if (node_type->isa<AbstractDictionary>()) {
    auto x = node_type->cast<AbstractDictionaryPtr>();
    auto &items = x->elements();
    std::vector<AnfNodePtr> dict_key_nodes;
    std::vector<AnfNodePtr> dict_value_nodes;
    dict_key_nodes.emplace_back(NewValueNode(prim::kPrimMakeTuple));
    dict_value_nodes.emplace_back(NewValueNode(prim::kPrimMakeTuple));
    for (const auto &item : items) {
      AnfNodePtr dict_value_node =
        func_graph->NewCNode({NewValueNode(prim::kPrimDictGetItem), source_node, NewValueNode(item.first)});
      AnfNodePtr node = MixedPrecisionCastHelper(dict_value_node, item.second, target_type, func_graph);
      dict_key_nodes.emplace_back(NewValueNode(item.first));
      dict_value_nodes.emplace_back(node);
    }
    target_node = func_graph->NewCNode({NewValueNode(prim::kPrimMakeDict), func_graph->NewCNode(dict_key_nodes),
                                        func_graph->NewCNode(dict_value_nodes)});
  } else if (node_type->isa<AbstractKeywordArg>()) {
    auto x = node_type->cast<AbstractKeywordArgPtr>();
    std::string kwarg_key = x->get_key();
    AnfNodePtr kwarg_value_node =
      func_graph->NewCNode({NewValueNode(prim::kPrimExtractKeywordArg), NewValueNode(kwarg_key), source_node});
    AnfNodePtr node = MixedPrecisionCastHelper(kwarg_value_node, x->get_arg(), target_type, func_graph);
    target_node = func_graph->NewCNode({NewValueNode(prim::kPrimMakeKeywordArg), NewValueNode(kwarg_key), node});
  }
  return target_node;
}

EvalResultPtr MixedPrecisionCastEvaluator::Run(AnalysisEnginePtr engine, const ConfigPtrList &args_conf_list,
                                               AnfNodeConfigPtr out_conf) {
  AbstractBasePtrList args_spec_list;
  if (out_conf->node() == nullptr || !out_conf->node()->isa<CNode>()) {
    MS_LOG(EXCEPTION) << "Node of out_conf should be CNode";
  }
  auto out_node = out_conf->node()->cast<CNodePtr>();
  const auto &out_node_inputs = out_node->inputs();
  if (out_node->inputs().empty() || (out_node_inputs.size() - 1) != args_conf_list.size()) {
    MS_LOG(EXCEPTION) << "MixedPrecisionCast"
                      << " args size should equal to inputs size minus 1, but args size " << args_conf_list.size()
                      << ", inputs size " << out_node_inputs.size();
  }
  (void)std::transform(args_conf_list.begin(), args_conf_list.end(), std::back_inserter(args_spec_list),
                       [](const ConfigPtr &ref) -> AbstractBasePtr { return ref->ObtainEvalResult()->abstract(); });

  ScopePtr scope = kDefaultScope;
  if (out_conf != nullptr) {
    scope = out_conf->node()->scope();
  }
  ScopeGuard scope_guard(scope);

  FuncGraphPtr func_graph = out_conf->node()->func_graph();
  AnfNodePtr new_node = MixedPrecisionCastHelper(out_node_inputs[2], args_spec_list[1], out_node_inputs[1], func_graph);
  AnfNodeConfigPtr fn_conf = engine->MakeConfig(new_node, out_conf->context());

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

py::dict AbstractTupleToPython(const AbstractBasePtr &abs_base) {
  auto arg_tuple = dyn_cast<AbstractTuple>(abs_base);
  size_t len = arg_tuple->size();
  py::tuple shape_tuple(len);
  py::tuple dtype_tuple(len);
  py::tuple value_tuple(len);
  py::tuple min_value_tuple(len);
  py::tuple max_value_tuple(len);
  py::tuple min_shape_tuple(len);
  py::tuple max_shape_tuple(len);
  bool dyn_shape = false;
  bool dyn_value = false;

  for (size_t i = 0; i < len; i++) {
    auto arg = arg_tuple->elements()[i];
    py::dict out = ConvertAbstractToPython(arg);
    shape_tuple[i] = out[ATTR_SHAPE];
    dtype_tuple[i] = out[ATTR_DTYPE];
    value_tuple[i] = out[ATTR_VALUE];

    // Elements in tuple is tensor shape value.
    if (out.contains(py::str(ATTR_MIN_VALUE)) && out.contains(py::str(ATTR_MAX_VALUE))) {
      min_value_tuple[i] = out[ATTR_MIN_VALUE];
      max_value_tuple[i] = out[ATTR_MAX_VALUE];
      dyn_value = true;
    }

    // Elements in tuple is tensor, which shape is dynamic.
    if (out.contains(py::str(ATTR_MIN_SHAPE)) && out.contains(py::str(ATTR_MAX_SHAPE))) {
      min_shape_tuple[i] = out[ATTR_MIN_SHAPE];
      max_shape_tuple[i] = out[ATTR_MAX_SHAPE];
      dyn_shape = true;
    }
  }
  auto dic = py::dict();
  dic[ATTR_SHAPE] = shape_tuple;
  dic[ATTR_DTYPE] = dtype_tuple;
  if (arg_tuple->BuildValue()->isa<AnyValue>()) {
    dic[ATTR_VALUE] = py::none();
  } else {
    dic[ATTR_VALUE] = value_tuple;
  }

  if (dyn_value) {
    dic[ATTR_MIN_VALUE] = min_value_tuple;
    dic[ATTR_MAX_VALUE] = max_value_tuple;
  }
  if (dyn_shape) {
    dic[ATTR_MIN_SHAPE] = min_shape_tuple;
    dic[ATTR_MAX_SHAPE] = max_shape_tuple;
  }

  return dic;
}

py::dict AbstractListToPython(const AbstractBasePtr &abs_base) {
  auto arg_list = dyn_cast<AbstractList>(abs_base);
  size_t len = arg_list->size();
  py::list shape_list(len);
  py::list dtype_list(len);
  py::list value_list(len);
  py::list min_shape_list(len);
  py::list max_shape_list(len);
  bool dyn_shape = false;

  for (size_t i = 0; i < len; i++) {
    py::dict out = ConvertAbstractToPython(arg_list->elements()[i]);
    shape_list[i] = out[ATTR_SHAPE];
    dtype_list[i] = out[ATTR_DTYPE];
    value_list[i] = out[ATTR_VALUE];

    // Elements in list is tensor, which shape is dynamic.
    if (out.contains(py::str(ATTR_MIN_SHAPE)) && out.contains(py::str(ATTR_MAX_SHAPE))) {
      min_shape_list[i] = out[ATTR_MIN_SHAPE];
      max_shape_list[i] = out[ATTR_MAX_SHAPE];
      dyn_shape = true;
    }
  }
  auto dic = py::dict();
  dic[ATTR_SHAPE] = shape_list;
  dic[ATTR_DTYPE] = dtype_list;
  if (arg_list->BuildValue()->isa<AnyValue>()) {
    dic[ATTR_VALUE] = py::none();
  } else {
    dic[ATTR_VALUE] = value_list;
  }

  if (dyn_shape) {
    dic[ATTR_MIN_SHAPE] = min_shape_list;
    dic[ATTR_MAX_SHAPE] = max_shape_list;
  }

  return dic;
}

void ConvertAbstractTensorToPython(const AbstractBasePtr &abs_base, py::dict *dic) {
  auto arg_tensor = dyn_cast<AbstractTensor>(abs_base);
  (*dic)[ATTR_SHAPE] = arg_tensor->shape()->shape();
  if (MsContext::GetInstance()->get_param<int>(MS_CTX_EXECUTION_MODE) == kGraphMode) {
    const auto &min_shape = arg_tensor->shape()->min_shape();
    const auto &max_shape = arg_tensor->shape()->max_shape();
    if (!min_shape.empty() && !max_shape.empty()) {
      (*dic)[ATTR_MIN_SHAPE] = min_shape;
      (*dic)[ATTR_MAX_SHAPE] = max_shape;
    }
  }

  auto min_value = arg_tensor->get_min_value();
  auto max_value = arg_tensor->get_max_value();
  if (min_value != nullptr && max_value != nullptr) {
    (*dic)[ATTR_MIN_VALUE] = BuildValue(min_value);
    (*dic)[ATTR_MAX_VALUE] = BuildValue(max_value);
  }

  (*dic)[ATTR_DTYPE] = arg_tensor->BuildType();
  (*dic)[ATTR_VALUE] = BuildValue(arg_tensor->BuildValue());
}

void ConvertAbstractFunctionToPython(const AbstractBasePtr &abs_base, py::dict *dic) {
  (*dic)[ATTR_SHAPE] = py::none();
  (*dic)[ATTR_DTYPE] = abs_base->BuildType();
  (*dic)[ATTR_VALUE] = py::none();
  if (abs_base->isa<PartialAbstractClosure>()) {
    AbstractBasePtrList args = abs_base->cast<PartialAbstractClosurePtr>()->args();
    if (!args.empty()) {
      auto value = args[0]->BuildValue()->cast<parse::ClassTypePtr>();
      if (value != nullptr) {
        (*dic)[ATTR_DTYPE] = std::make_shared<TypeType>();
        (*dic)[ATTR_VALUE] = value->obj();
      }
    }
  }
}
}  // end anonymous namespace

py::dict ConvertAbstractToPython(const AbstractBasePtr &abs_base) {
  MS_EXCEPTION_IF_NULL(abs_base);
  auto dic = py::dict();
  if (abs_base->isa<AbstractTensor>()) {
    ConvertAbstractTensorToPython(abs_base, &dic);
  } else if (abs_base->isa<AbstractRowTensor>()) {
    auto arg = dyn_cast<AbstractRowTensor>(abs_base);
    dic[ATTR_SHAPE] = arg->shape()->shape();
    dic[ATTR_DTYPE] = arg->BuildType();
    dic[ATTR_VALUE] = BuildValue(arg->BuildValue());
  } else if (abs_base->isa<AbstractSparseTensor>()) {
    auto arg = dyn_cast<AbstractSparseTensor>(abs_base);
    dic[ATTR_SHAPE] = arg->shape()->shape();
    dic[ATTR_DTYPE] = arg->BuildType();
    dic[ATTR_VALUE] = BuildValue(arg->BuildValue());
  } else if (abs_base->isa<AbstractScalar>() || abs_base->isa<AbstractType>() || abs_base->isa<AbstractRefKey>()) {
    ShapeVector shape;
    dic[ATTR_SHAPE] = shape;
    dic[ATTR_DTYPE] = abs_base->BuildType();
    dic[ATTR_VALUE] = BuildValue(abs_base->BuildValue());
  } else if (abs_base->isa<AbstractSlice>()) {
    auto arg_slice = dyn_cast<AbstractSlice>(abs_base);
    ShapeVector shape;
    dic[ATTR_SHAPE] = shape;
    dic[ATTR_DTYPE] = arg_slice->BuildType();
    dic[ATTR_VALUE] = BuildValue(arg_slice->BuildValue());
  } else if (abs_base->isa<AbstractEllipsis>()) {
    dic[ATTR_SHAPE] = py::none();
    dic[ATTR_DTYPE] = py::ellipsis();
    dic[ATTR_VALUE] = py::ellipsis();
  } else if (abs_base->isa<AbstractTuple>()) {
    return AbstractTupleToPython(abs_base);
  } else if (abs_base->isa<AbstractList>()) {
    return AbstractListToPython(abs_base);
  } else if (abs_base->isa<AbstractNone>()) {
    dic[ATTR_SHAPE] = py::none();
    dic[ATTR_DTYPE] = py::none();
    dic[ATTR_VALUE] = py::none();
  } else if (abs_base->isa<AbstractFunction>()) {
    ConvertAbstractFunctionToPython(abs_base, &dic);
  } else if (abs_base->isa<AbstractUndetermined>()) {
    auto arg = dyn_cast<AbstractUndetermined>(abs_base);
    dic[ATTR_SHAPE] = py::none();
    dic[ATTR_DTYPE] = arg->BuildType();
    dic[ATTR_VALUE] = py::none();
  } else if (abs_base->isa<AbstractMonad>()) {
    dic[ATTR_SHAPE] = py::none();
    dic[ATTR_DTYPE] = abs_base->BuildType();
    dic[ATTR_VALUE] = py::none();
  } else {
    auto value = abs_base->BuildValue();
    if ((*value == *kAnyValue)) {
      auto value_desc = abs_base->value_desc();
      MS_EXCEPTION(TypeError) << "Unsupported parameter " << (value_desc.empty() ? "type" : value_desc)
                              << " for python primitive." << abs_base->ToString();
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

  // The monad parameter is defined at the end of the parameter and needs to be ignored
  std::size_t size_args = args_ptr->size() - GetAbstractMonadNum(*args_ptr);
  py::tuple py_args(size_args);
  for (size_t i = 0; i < size_args; i++) {
    auto arg_i = (*args_ptr)[i];
    py_args[i] = ConvertAbstractToPython(arg_i);
  }
  return py_args;
}

AbstractBasePtr PyInferRes2Abstract(const PrimitivePyPtr &prim_py, const py::dict &output) {
  // Convert to AbstractValue based on type and shape
  auto out_dtype = output[ATTR_DTYPE];
  if (output[ATTR_VALUE].is_none()) {
    auto out_shape = output[ATTR_SHAPE];
    return PyListDtype2AbstractTensor(out_shape, out_dtype, output);
  }
  // Convert pyobject to Value, then to AbstractValue
  ValuePtr converted_ret = nullptr;
  TypePtr dtype = py::isinstance<Type>(out_dtype) ? out_dtype.cast<TypePtr>() : nullptr;
  bool converted = parse::ConvertData(output[ATTR_VALUE], &converted_ret, false, dtype);
  if (!converted) {
    MS_LOG(EXCEPTION) << "Convert data failed";
  }
  auto res_spec = FromValue(converted_ret);
  MS_EXCEPTION_IF_NULL(res_spec);
  if (res_spec->isa<AbstractTensor>()) {
    // Replace to tensor constant node in specialize
    auto res_tensor = res_spec->cast<AbstractTensorPtr>();
    res_tensor->set_value(converted_ret);
    SetValueRange(res_tensor, output);
  }
  if (prim_py->IsCustomPrim()) {
    // Raise error if output_num is not match the infer result.
    int64_t output_num = GetValue<int64_t>(prim_py->GetAttr("output_num"));
    if (res_spec->isa<AbstractTensor>() && output_num != 1) {
      MS_LOG(EXCEPTION) << "Custom primitive " << prim_py->ToString() << " output_num " << output_num
                        << " not matches the infer result.";
    } else if (res_spec->isa<AbstractTuple>() &&
               (res_spec->cast<AbstractTuplePtr>()->size() != LongToSize(output_num))) {
      MS_LOG(EXCEPTION) << "Custom primitive " << prim_py->ToString() << " output_num " << output_num
                        << " not matches the infer result.";
    }
  }
  return res_spec;
}
}  // end anonymous namespace

EvalResultPtr StandardPrimEvaluator::EvalPyCheckPrim(const AnalysisEnginePtr &engine, const AbstractBasePtrList &args) {
  auto prim_py = dyn_cast<PrimitivePy>(prim_);
  if (prim_py == nullptr) {
    MS_LOG(EXCEPTION) << "The primitive with type 'kPrimTypePyInferCheck' should be a python primitive.";
  }

  // Call checking method '__check__' for subclass of 'PrimitiveWithCheck'
  MS_LOG(DEBUG) << "Begin input args checking for: " << prim_py->ToString();
  auto py_args = PreparePyInputs(prim_py, args);
  prim_py->RunCheck(py_args);

  prim_->BeginRecordAddAttr();
  AbstractBasePtr abs_base = eval_impl_(engine, prim_, args);
  prim_->EndRecordAddAttr();
  auto added_attrs = prim_->evaluate_added_attrs();

  if (!py::hasattr(prim_py->GetPyObj(), PY_PRIM_METHOD_INFER_VALUE)) {
    return std::make_shared<EvalResult>(abs_base, std::make_shared<AttrValueMap>(added_attrs));
  }

  // Call method 'infer_value' for primitive with this method for constant propagation
  py::tuple py_vals(py_args.size());
  for (size_t i = 0; i < py_args.size(); ++i) {
    py_vals[i] = py_args[i][ATTR_VALUE];
  }
  py::object py_ret = prim_py->RunInferValue(py_vals);
  if (py::isinstance<py::none>(py_ret)) {
    return std::make_shared<EvalResult>(abs_base, std::make_shared<AttrValueMap>(added_attrs));
  }

  // Convert pyobject to Value, then to AbstractValue
  ValuePtr converted_ret = nullptr;
  TypePtr dtype = abs_base->BuildType();
  bool converted = parse::ConvertData(py_ret, &converted_ret, false, dtype);
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
  return std::make_shared<EvalResult>(res_spec, std::make_shared<AttrValueMap>(added_attrs));
}

EvalResultPtr StandardPrimEvaluator::EvalPrim(const AnalysisEnginePtr &engine, const AbstractBasePtrList &args) {
  if (prims_to_skip_undetermined_infer.find(prim_->name()) == prims_to_skip_undetermined_infer.end()) {
    auto ret_abstract = AbstractEval(args);
    if (ret_abstract != nullptr) {
      MS_LOG(DEBUG) << "StandardPrimEvaluator eval Undetermined";
      return ret_abstract;
    }
  }

  if (prim_->prim_type() == PrimType::kPrimTypePyInferCheck) {
    return EvalPyCheckPrim(engine, args);
  }

  prim_->BeginRecordAddAttr();
  AbstractBasePtr abs_base = eval_impl_(engine, prim_, args);
  prim_->EndRecordAddAttr();
  auto added_attrs = prim_->evaluate_added_attrs();
  auto eval_result = std::make_shared<EvalResult>(abs_base, std::make_shared<AttrValueMap>(added_attrs));
  return eval_result;
}

EvalResultPtr PythonPrimEvaluator::EvalPrim(const AnalysisEnginePtr &, const AbstractBasePtrList &args) {
  auto ret_abstract = AbstractEval(args);
  if (ret_abstract != nullptr) {
    MS_LOG(DEBUG) << "PythonPrimEvaluator eval Undetermined";
    return ret_abstract;
  }
  MS_LOG(DEBUG) << "Eval for:" << prim_py_->ToString();

  const auto &iter = evaluator_cache_map_->find(args);
  if (iter != evaluator_cache_map_->end()) {
    return iter->second;
  }
  auto py_args = PreparePyInputs(prim_py_, args);
  prim_py_->BeginRecordAddAttr();
  py::dict output = prim_py_->RunInfer(py_args);
  prim_py_->EndRecordAddAttr();
  auto added_attrs = prim_py_->evaluate_added_attrs();
  MS_LOG(DEBUG) << "Output type is " << (std::string)py::str(output);
  auto res_spec = PyInferRes2Abstract(prim_py_, output);

  MS_LOG(DEBUG) << "Python InferTensor result spec: " << res_spec->ToString() << ".";
  auto infer_result = std::make_shared<EvalResult>(res_spec, std::make_shared<AttrValueMap>(added_attrs));
  (*evaluator_cache_map_)[args] = infer_result;
  return infer_result;
}

EvalResultPtr UniformPrimEvaluator::EvalPrim(const AnalysisEnginePtr &, const AbstractBasePtrList &args) {
  auto ret_abstract = AbstractEval(args);
  if (ret_abstract != nullptr) {
    MS_LOG(DEBUG) << "UniformPrimEvaluator eval Undetermined";
    return ret_abstract;
  }
  // if func_desc_.retval type is super class of parameter type, then make the retval type as parameter type.
  if (nargs_ != args.size()) {
    MS_LOG(EXCEPTION) << "UniformPrimEvaluator expect " << nargs_ << " args, but got " << args.size() << " inputs";
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

  ValuePtr evaluated_value = RunImpl(value_list);
  if (!(*evaluated_value == *kAnyValue)) {
    ret_value_type = evaluated_value->type();
  }
  // for comparison primitives , return type shall have be specified to be bool.
  if (specify_out_type_ != nullptr) {
    ret_value_type = specify_out_type_;
  }

  AbstractScalarPtr abs_base = std::make_shared<AbstractScalar>(evaluated_value, ret_value_type);
  return std::make_shared<EvalResult>(abs_base, std::make_shared<AttrValueMap>());
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

const int64_t kResolveCaseUserDefineClass = 1;
const int64_t kResolveCaseBuiltInType = 2;
const int64_t kResolveCaseFunction = 3;
int64_t GetResolveCase(const TypePtr &data_type) {
  MS_EXCEPTION_IF_NULL(data_type);
  if (data_type->type_id() == kObjectTypeClass) {
    return kResolveCaseUserDefineClass;
  }

  // try method map, if not in method map, the data_type should be External type.
  if (pipeline::Resource::IsTypeInBuiltInMap(data_type->type_id())) {
    return kResolveCaseBuiltInType;
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

enum REQUIRE_TYPE { ATTR, METHOD };

EvalResultPtr StaticGetterInferred(const ValuePtr &value, const ConfigPtr &data_conf, const AnfNodeConfigPtr &old_conf,
                                   REQUIRE_TYPE require_type = REQUIRE_TYPE::METHOD) {
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
  if (require_type == REQUIRE_TYPE::ATTR) {
    new_cnode = func_graph->NewCNode({new_cnode});
  }
  AnalysisEnginePtr eng = old_conf->engine();
  AnfNodeConfigPtr fn_conf = eng->MakeConfig(new_cnode, old_conf->context());
  return eng->ForwardConfig(old_conf, fn_conf);
}

EvalResultPtr GetEvaluatedValueForNameSpaceString(const AnalysisEnginePtr &engine,
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
  auto out_node = out_conf->node();
  FuncGraphPtr func_graph = out_node->func_graph();

  auto new_node = parse::ResolveSymbol(func_graph->manager(), name_space, symbol, out_node);
  if (new_node == nullptr) {
    MS_LOG(EXCEPTION) << "Resolve node failed";
  }

  // Replace old node with the resolved new node in order list.
  func_graph->ReplaceInOrder(out_node, new_node);

  AnalysisEnginePtr eng = out_conf->engine();
  AnfNodeConfigPtr fn_conf = eng->MakeConfig(new_node, out_conf->context());
  return eng->ForwardConfig(out_conf, fn_conf);
}

EvalResultPtr GetEvaluatedValueForClassAttrOrMethod(const AnalysisEnginePtr &engine,
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
    return std::make_shared<EvalResult>(attr, nullptr);
  }

  ValuePtr method = cls->GetMethod(item_name);
  if (method->isa<AnyValue>()) {
    MS_EXCEPTION(AttributeError) << "Unknown field, data type: " << args_spec_list[0]->BuildType()->ToString()
                                 << ", item value: " << item_v->ToString();
  }

  // Infer class method
  ValuePtr converted_v = PyObjToGraph(engine, method);
  return StaticGetterInferred(converted_v, data_conf, out_conf);
}

EvalResultPtr GetEvaluatedValueForBuiltinTypeAttrOrMethod(const AnalysisEnginePtr &engine, const ValuePtr &item_v,
                                                          const TypePtr &data_type, const ConfigPtr &data_conf,
                                                          const AnfNodeConfigPtr &out_conf) {
  MS_EXCEPTION_IF_NULL(item_v);
  MS_EXCEPTION_IF_NULL(data_type);
  // The method maybe a Primitive or Composite
  if (!item_v->isa<StringImm>()) {
    MS_LOG(EXCEPTION) << "Error item is not string";
  }

  std::string item_name = item_v->cast<StringImmPtr>()->value();
  REQUIRE_TYPE require_type = REQUIRE_TYPE::METHOD;
  Any require = pipeline::Resource::GetMethodPtr(data_type->type_id(), item_name);
  if (require.empty()) {
    require = pipeline::Resource::GetAttrPtr(data_type->type_id(), item_name);
    if (require.empty()) {
      MS_LOG(EXCEPTION) << "The object of type: " << data_type->ToString() << " has no method or attr: " << item_name;
    }
    require_type = REQUIRE_TYPE::ATTR;
  }

  ValuePtr converted_v = nullptr;
  if (require.is<std::string>()) {
    // composite registered in standard_method_map go to this branch
    converted_v = prim::GetPythonOps(require.cast<std::string>());
    if (!converted_v->isa<Primitive>()) {
      AddToManager(engine, converted_v->cast<FuncGraphPtr>());
    }
  } else if (require.is<PrimitivePtr>()) {
    converted_v = require.cast<PrimitivePtr>();
  } else {
    MS_LOG(EXCEPTION) << "Expect to get string or PrimitivePtr from attr or method map, but got " << require.ToString();
  }
  return StaticGetterInferred(converted_v, data_conf, out_conf, require_type);
}

EvalResultPtr StaticGetter(const AnalysisEnginePtr &engine, const AbstractBasePtrList &args_spec_list,
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

  int64_t case_v = GetResolveCase(data_type);
  if (case_v == kResolveCaseUserDefineClass) {
    return GetEvaluatedValueForClassAttrOrMethod(engine, args_spec_list, item_value, data_conf, out_conf);
  } else if (case_v == kResolveCaseBuiltInType) {
    return GetEvaluatedValueForBuiltinTypeAttrOrMethod(engine, item_value, data_type, data_conf, out_conf);
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
  EvalResultPtr EvalPrim(const ConfigPtrList &args_conf_list) override {
    // arg: free variable to be embedded
    if (args_conf_list.size() != 1) {
      MS_LOG(EXCEPTION) << "EmbedEvaluator requires 1 parameter, but got " << args_conf_list.size();
    }
    AnfNodeConfigPtr node_conf = dyn_cast<AnfNodeConfig>(args_conf_list[0]);
    MS_EXCEPTION_IF_NULL(node_conf);

    AbstractBasePtr x = node_conf->ObtainEvalResult()->abstract();
    x = SensitivityTransform(x);
    SymbolicKeyInstancePtr key = std::make_shared<SymbolicKeyInstance>(node_conf->node(), x);
    AbstractScalarPtr abs_scalar = std::make_shared<AbstractScalar>(key, std::make_shared<SymbolicKeyType>());
    return std::make_shared<EvalResult>(abs_scalar, std::make_shared<AttrValueMap>());
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
  EvalResultPtr EvalPrim(const ConfigPtrList &args_conf_list) override {
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
    AbstractBasePtr abs = node_conf->ObtainEvalResult()->abstract();
    AbstractRefPtr ref_abs = abs->cast<AbstractRefPtr>();
    if (ref_abs == nullptr) {
      MS_LOG(ERROR) << "The first parameter of RefToEmbed should be Ref, but " << abs->ToString();
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
      auto ret = std::make_shared<AbstractScalar>(type);
      auto ref_value = ref_abs->ref();
      MS_EXCEPTION_IF_NULL(ref_value);
      return std::make_shared<EvalResult>(ret, std::make_shared<AttrValueMap>());
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
    return std::make_shared<EvalResult>(abs_scalar, std::make_shared<AttrValueMap>());
  }
};

class GetAttrEvaluator : public TransitionPrimEvaluator {
 public:
  GetAttrEvaluator() : TransitionPrimEvaluator("GetAttrEvaluator") {}
  ~GetAttrEvaluator() override = default;
  MS_DECLARE_PARENT(GetAttrEvaluator, TransitionPrimEvaluator);
  EvalResultPtr EvalPrim(const AnalysisEnginePtr &engine, const AbstractBasePtrList &args_spec_list,
                         const ConfigPtr &in_conf0, const AnfNodeConfigPtr &out_conf) override {
    auto ret_abstract = AbstractEval(args_spec_list);
    if (ret_abstract != nullptr) {
      MS_LOG(DEBUG) << "GetAttrEvaluator eval Undetermined";
      return ret_abstract;
    }
    // Inputs: data, item
    if (args_spec_list.size() != 2) {
      MS_LOG(EXCEPTION) << "Expected args_spec_list size = 2, but has size:" << args_spec_list.size();
    }
    EvalResultPtr ret = nullptr;
    if (bound_node() != nullptr) {
      TraceGuard trace_guard(std::make_shared<TraceResolve>(bound_node()->debug_info()));
      ret = StaticGetter(engine, args_spec_list, in_conf0, out_conf);
    } else {
      ret = StaticGetter(engine, args_spec_list, in_conf0, out_conf);
    }
    // don't lookup from cache, as different out_conf with same node but different context
    // may add different entry to anfnode_config_map, like getattr primitive;
    (*evaluator_cache_map_)[args_spec_list] = ret;
    return ret;
  }
};

class ResolveEvaluator : public TransitionPrimEvaluator {
 public:
  ResolveEvaluator() : TransitionPrimEvaluator("ResolveEvaluator") {}
  ~ResolveEvaluator() override = default;
  MS_DECLARE_PARENT(ResolveEvaluator, TransitionPrimEvaluator);
  EvalResultPtr EvalPrim(const AnalysisEnginePtr &engine, const AbstractBasePtrList &args_spec_list,
                         const ConfigPtr &in_conf0, const AnfNodeConfigPtr &out_conf) override {
    // Inputs: namespace, symbol
    if (args_spec_list.size() != 2) {
      MS_LOG(EXCEPTION) << "Expected args_spec_list size = 2, but has size:" << args_spec_list.size();
    }
    EvalResultPtr ret = nullptr;
    if (bound_node() != nullptr) {
      TraceGuard trace_guard(std::make_shared<TraceResolve>(bound_node()->debug_info()));
      ret = StaticGetter(engine, args_spec_list, in_conf0, out_conf);
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
  EvalResultPtr EvalPrim(const AnalysisEnginePtr &engine, const AbstractBasePtrList &args_spec_list, const ConfigPtr &,
                         const AnfNodeConfigPtr &out_conf) override {
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
      MS_LOG(EXCEPTION) << "Create python object" << py::str(class_type)
                        << " failed, only support create Cell or Primitive object.";
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
    auto infer_result = std::make_shared<EvalResult>(ret, std::make_shared<AttrValueMap>());
    (*evaluator_cache_map_)[args_spec_list] = infer_result;
    return infer_result;
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
  EvalResultPtr Run(AnalysisEnginePtr engine, const ConfigPtrList &args_conf_list,
                    AnfNodeConfigPtr out_conf = nullptr) override {
    if (args_conf_list.size() == 0) {
      MS_LOG(EXCEPTION) << "Args size should be greater than 0";
    }

    MS_EXCEPTION_IF_NULL(out_conf);
    MS_EXCEPTION_IF_NULL(out_conf->node());
    auto arg0_value = args_conf_list[0]->ObtainEvalResult()->abstract();
    AbstractBasePtrList args_spec_list{arg0_value};
    // Func in hypermap(partial(Func, arg0), arg1, arg2) may become Poly Node.
    if (arg0_value->isa<AbstractError>()) {
      auto ret = std::make_shared<AbstractError>(arg0_value->GetValueTrack()->cast<StringImmPtr>(), out_conf->node());
      MS_LOG(DEBUG) << "AbstractError for node: " << out_conf->node()->DebugString()
                    << " as func is: " << arg0_value->ToString();
      auto eval_result = std::make_shared<EvalResult>(ret, std::make_shared<AttrValueMap>());
      (*evaluator_cache_map_)[args_spec_list] = eval_result;
      return eval_result;
    }
    auto func = CheckArg<AbstractFunction>("partial", args_spec_list, 0);
    // Sometimes, node[0] in out_conf becomes phi0;
    if (func->isa<PrimitiveAbstractClosure>()) {
      auto prim_func = dyn_cast<PrimitiveAbstractClosure>(func);
      if (prim_func->prim()->isa<prim::DoSignaturePrimitive>()) {
        prim::DoSignaturePrimitivePtr do_signature_prim = dyn_cast<prim::DoSignaturePrimitive>(prim_func->prim());
        return HandleDoSignature(engine, do_signature_prim->function(), out_conf);
      }
    }

    (void)std::transform(
      args_conf_list.begin() + 1, args_conf_list.end(), std::back_inserter(args_spec_list),
      [](const ConfigPtr &config) -> AbstractBasePtr { return config->ObtainEvalResult()->abstract(); });
    AbstractBasePtrList args(args_spec_list.begin() + 1, args_spec_list.end());

    auto cnode = out_conf->node()->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    if (cnode->size() != (args_conf_list.size() + 1)) {
      MS_LOG(EXCEPTION) << "Out_conf node: " << cnode->DebugString()
                        << ", args_conf_list: " << mindspore::ToString(args_conf_list);
    }
    AbstractFuncAtomPtrList partial_funcs_list;
    auto build_partial = [args, cnode, &partial_funcs_list](const AbstractFuncAtomPtr &atom_func) {
      auto new_func = std::make_shared<PartialAbstractClosure>(atom_func, args, cnode);
      partial_funcs_list.push_back(new_func);
    };
    func->Visit(build_partial);

    auto ret = AbstractFunction::MakeAbstractFunction(partial_funcs_list);
    auto eval_result = std::make_shared<EvalResult>(ret, std::make_shared<AttrValueMap>());
    (*evaluator_cache_map_)[args_spec_list] = eval_result;
    return eval_result;
  }

  EvalResultPtr Eval(AnalysisEnginePtr, const AbstractBasePtrList &) override {
    MS_LOG(EXCEPTION) << "Eval() should not be called, Run() method should be called";
  }

  EvalResultPtr HandleDoSignature(const AnalysisEnginePtr &engine, const ValuePtr &signature_value,
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

bool IsInWhiteList(const PrimitivePtr &primitive) {
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
