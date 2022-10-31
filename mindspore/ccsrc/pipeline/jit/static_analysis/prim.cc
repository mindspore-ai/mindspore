/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#include "ir/anf.h"
#include "ir/cell.h"
#include "utils/hash_set.h"
#include "frontend/operator/cc_implementations.h"
#include "frontend/operator/ops.h"
#include "frontend/operator/composite/do_signature.h"
#include "frontend/operator/prim_to_function.h"
#include "abstract/utils.h"
#include "utils/log_adapter.h"
#include "utils/symbolic.h"
#include "pipeline/jit/resource.h"
#include "pipeline/jit/parse/resolve.h"
#include "pipeline/jit/pipeline.h"
#include "pipeline/jit/static_analysis/static_analysis.h"
#include "pipeline/jit/debug/trace.h"
#include "include/common/utils/convert_utils.h"
#include "include/common/utils/convert_utils_py.h"
#include "utils/ms_context.h"
#include "pipeline/jit/parse/data_converter.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/param_validator.h"
#include "utils/ms_utils.h"
#include "utils/shape_utils.h"
#include "utils/parallel_node_check.h"
#include "utils/check_convert_utils.h"
#include "frontend/operator/ops_front_infer_function.h"

namespace mindspore {
namespace abstract {
namespace interpret_abstract_bool_checker {
std::pair<bool, bool> InterpretAbstractBoolChecker(const AbstractBasePtr &cond) {
  MS_EXCEPTION_IF_NULL(cond);
  bool is_interpret = false;
  bool has_true = false;
  auto value = cond->BuildValue();
  if (value->isa<parse::InterpretedObject>()) {
    is_interpret = true;
    auto interpreted_obj = value->cast_ptr<parse::InterpretedObject>();
    MS_EXCEPTION_IF_NULL(interpreted_obj);
    py::object obj = interpreted_obj->obj();
    constexpr char PYTHON_MOD_PARSE_MODULE[] = "mindspore._extends.parse";
    constexpr char PYTHON_MOD_CHECK_OBJ_BOOL[] = "check_obj_bool";
    py::module mod = python_adapter::GetPyModule(PYTHON_MOD_PARSE_MODULE);
    bool res = python_adapter::CallPyModFn(mod, PYTHON_MOD_CHECK_OBJ_BOOL, obj).cast<bool>();
    // eval("np.array(1) >= 1")                            ---> obj: array([ True])
    // eval("(np.array([1, 2]) > np.array([1, 0])).any()") ---> obj: True
    if (res) {
      has_true = true;
    }
  }
  return {is_interpret, has_true};
}

struct InterpretAbstractBoolCheckerRegister {
  InterpretAbstractBoolCheckerRegister() noexcept {
    abstract::AbstractBase::set_interpret_bool_checker(
      [](const AbstractBasePtr &cond) { return InterpretAbstractBoolChecker(cond); });
  }
  ~InterpretAbstractBoolCheckerRegister() {}
} interpret_abstract_bool_checker_register;
}  // namespace interpret_abstract_bool_checker

using mindspore::parse::PyObjectWrapper;

mindspore::HashSet<std::string> prims_to_skip_undetermined_infer{prim::kMakeTuple,  prim::kMakeList,   prim::kSwitch,
                                                                 prim::kEnvironSet, prim::kEnvironGet, prim::kLoad,
                                                                 prim::kUpdateState};

// The Python primitives who visit tuple/list elements, but not consume all elements.
// Including:
// - Consume no element. For instance, MakeTuple.
// - Consume partial elements, not all. For instance, TupleGetItem.
// Map{"primitive name", {vector<int>:"index to transparent pass, -1 means all elements"}}
mindspore::HashMap<std::string, std::vector<int>> prims_transparent_pass_sequence{
  {prim::kReturn, std::vector({0})},       {prim::kDepend, std::vector({0})},     {prim::kidentity, std::vector({0})},
  {prim::kMakeTuple, std::vector({-1})},   {prim::kMakeList, std::vector({-1})},  {prim::kListAppend, std::vector({0})},
  {prim::kTupleGetItem, std::vector({0})}, {prim::kListGetItem, std::vector({0})}};

EvalResultPtr DoSignatureEvaluator::Run(AnalysisEnginePtr engine, const ConfigPtrList &args_conf_list,
                                        const AnfNodeConfigPtr &out_conf) {
  MS_EXCEPTION_IF_NULL(engine);
  MS_EXCEPTION_IF_NULL(out_conf);
  AbstractBasePtrList args_spec_list;
  (void)std::transform(args_conf_list.begin(), args_conf_list.end(), std::back_inserter(args_spec_list),
                       [](const ConfigPtr &config) -> AbstractBasePtr {
                         MS_EXCEPTION_IF_NULL(config);
                         const auto &eval_result = config->ObtainEvalResult();
                         MS_EXCEPTION_IF_NULL(eval_result);
                         return eval_result->abstract();
                       });

  // Do undetermined infer firstly.
  auto do_signature = prim_->cast_ptr<prim::DoSignaturePrimitive>();
  MS_EXCEPTION_IF_NULL(do_signature);
  auto &func = do_signature->function();
  auto do_signature_func = dyn_cast_ptr<Primitive>(func);
  if (do_signature_func != nullptr) {
    if (prims_to_skip_undetermined_infer.find(do_signature_func->name()) == prims_to_skip_undetermined_infer.end()) {
      auto ret_abstract = EvalUndeterminedArgs(args_spec_list);
      if (ret_abstract != nullptr) {
        MS_LOG(DEBUG) << "DoSignatureEvaluator eval Undetermined for " << do_signature_func->name()
                      << ", ret_abstract: " << ret_abstract->ToString();
        return ret_abstract;
      }
    }
  }

  // Create new CNode with old CNode.
  if (out_conf->node() == nullptr || !out_conf->node()->isa<CNode>()) {
    MS_LOG(EXCEPTION) << "Node of out_conf should be CNode";
  }
  auto out_cnode = dyn_cast<CNode>(out_conf->node());
  MS_EXCEPTION_IF_NULL(out_cnode);
  const auto &out_node_inputs = out_cnode->inputs();
  if (out_cnode->inputs().size() == 0 || (out_node_inputs.size() - 1) != args_conf_list.size()) {
    MS_LOG(EXCEPTION) << "Op: " << func->ToString() << " args size should equal to inputs size minus 1, but args size "
                      << args_conf_list.size() << ", inputs size " << out_node_inputs.size();
  }
  AnfNodePtrList args_inputs{out_node_inputs.begin() + 1, out_node_inputs.end()};
  AnfNodePtr new_node = nullptr;
  ScopePtr scope = out_conf->node()->scope();
  ScopeGuard scope_guard(scope);
  if (bound_node() != nullptr) {
    TraceGuard trace_guard(std::make_shared<TraceDoSignature>(bound_node()->debug_info()));
    new_node = prim::GenerateCNode(out_cnode->func_graph(), prim_->ToString(), func, args_spec_list, args_inputs);
  } else {
    new_node = prim::GenerateCNode(out_cnode->func_graph(), prim_->ToString(), func, args_spec_list, args_inputs);
  }
  // Update new CNode info.
  auto new_cnode = dyn_cast<CNode>(new_node);
  MS_EXCEPTION_IF_NULL(new_cnode);
  new_cnode->CloneCNodeInfo(out_cnode);

  // Do forward with old config and new config.
  AnfNodeConfigPtr new_conf = engine->MakeConfig(new_node, out_conf->context(), out_conf->func_graph());
  return engine->ForwardConfig(out_conf, new_conf);
}

static AbstractBasePtrList GetUnpackGraphSpecArgsList(AbstractBasePtrList args_spec_list, bool need_unpack) {
  // arg[0] is the func graph to unpack, ignore it
  AbstractBasePtrList specialize_args_before_unpack(args_spec_list.begin() + 1, args_spec_list.end());
  AbstractBasePtrList graph_specialize_args;
  if (need_unpack) {
    for (size_t index = 0; index < specialize_args_before_unpack.size(); index++) {
      MS_EXCEPTION_IF_NULL(specialize_args_before_unpack[index]);
      if (specialize_args_before_unpack[index]->isa<AbstractTuple>()) {
        auto arg_tuple = specialize_args_before_unpack[index]->cast_ptr<AbstractTuple>();
        std::transform(arg_tuple->elements().begin(), arg_tuple->elements().end(),
                       std::back_inserter(graph_specialize_args), [](AbstractBasePtr abs) { return abs; });
      } else if (specialize_args_before_unpack[index]->isa<AbstractDictionary>()) {
        auto arg_dict = specialize_args_before_unpack[index]->cast_ptr<AbstractDictionary>();
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
                                        const AnfNodeConfigPtr &out_conf) {
  MS_EXCEPTION_IF_NULL(engine);
  MS_EXCEPTION_IF_NULL(out_conf);
  MS_EXCEPTION_IF_NULL(out_conf->node());
  if (out_conf->node() == nullptr || !out_conf->node()->isa<CNode>()) {
    MS_LOG(EXCEPTION) << "Node of out_conf should be CNode";
  }

  auto unpack_graph = prim_->cast_ptr<prim::UnpackGraphPrimitive>();
  MS_EXCEPTION_IF_NULL(unpack_graph);
  auto out_node = out_conf->node()->cast_ptr<CNode>();
  MS_EXCEPTION_IF_NULL(out_node);
  const auto &out_node_inputs = out_node->inputs();
  if (out_node->inputs().empty() || (out_node_inputs.size() - 1) != args_conf_list.size()) {
    MS_LOG(EXCEPTION) << "UnpackGraphPrimitive"
                      << " args size should equal to inputs size minus 1, but args size " << args_conf_list.size()
                      << ", inputs size " << out_node_inputs.size();
  }
  AbstractBasePtrList args_spec_list;
  (void)std::transform(args_conf_list.begin(), args_conf_list.end(), std::back_inserter(args_spec_list),
                       [](const ConfigPtr &ref) -> AbstractBasePtr {
                         MS_EXCEPTION_IF_NULL(ref);
                         const auto &eval_result = ref->ObtainEvalResult();
                         MS_EXCEPTION_IF_NULL(eval_result);
                         return eval_result->abstract();
                       });
  // get the forward graph
  if (args_spec_list.empty()) {
    MS_LOG(EXCEPTION) << "args_spec_list can't be empty.";
  }
  MS_EXCEPTION_IF_NULL(args_spec_list[0]);
  auto fn = args_spec_list[0]->cast_ptr<AbstractFunction>();
  if (fn == nullptr) {
    MS_LOG(EXCEPTION) << "UnpackGraphPrimitive arg0 must be AbstractFunction, but " << args_spec_list[0]->ToString();
  }
  auto real_fn = fn->cast_ptr<FuncGraphAbstractClosure>();
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
  AnfNodeConfigPtr fn_conf = engine->MakeConfig(new_vnode, out_conf->context(), out_conf->func_graph());

  return engine->ForwardConfig(out_conf, fn_conf);
}

AnfNodePtr MixedPrecisionCastHelper(const AnfNodePtr &source_node, const AbstractBasePtr &node_type,
                                    const AnfNodePtr &target_type, const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(node_type);
  MS_EXCEPTION_IF_NULL(func_graph);
  AnfNodePtr target_node = source_node;
  if (node_type->isa<AbstractTensor>()) {
    auto x = node_type->cast_ptr<AbstractTensor>();
    if (x->element()->BuildType()->isa<Float>()) {
      auto cast = prim::GetPythonOps("cast", "mindspore.ops.functional");
      MS_EXCEPTION_IF_NULL(cast);
      target_node = func_graph->NewCNodeAfter(source_node, {NewValueNode(cast), source_node, target_type});
    }
  } else if (node_type->isa<AbstractTuple>()) {
    auto x = node_type->cast_ptr<AbstractTuple>();
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
    auto x = node_type->cast_ptr<AbstractDictionary>();
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
    target_node =
      func_graph->NewCNode({NewValueNode(prim::kPrimMakeDict), func_graph->NewCNode(std::move(dict_key_nodes)),
                            func_graph->NewCNode(std::move(dict_value_nodes))});
  } else if (node_type->isa<AbstractKeywordArg>()) {
    auto x = node_type->cast_ptr<AbstractKeywordArg>();
    std::string kwarg_key = x->get_key();
    AnfNodePtr kwarg_value_node =
      func_graph->NewCNode({NewValueNode(prim::kPrimExtractKeywordArg), NewValueNode(kwarg_key), source_node});
    AnfNodePtr node = MixedPrecisionCastHelper(kwarg_value_node, x->get_arg(), target_type, func_graph);
    target_node = func_graph->NewCNode({NewValueNode(prim::kPrimMakeKeywordArg), NewValueNode(kwarg_key), node});
  }
  return target_node;
}

EvalResultPtr MixedPrecisionCastEvaluator::Run(AnalysisEnginePtr engine, const ConfigPtrList &args_conf_list,
                                               const AnfNodeConfigPtr &out_conf) {
  MS_EXCEPTION_IF_NULL(engine);
  AbstractBasePtrList args_spec_list;
  MS_EXCEPTION_IF_NULL(out_conf);
  if (out_conf->node() == nullptr || !out_conf->node()->isa<CNode>()) {
    MS_LOG(EXCEPTION) << "Node of out_conf should be CNode";
  }
  auto out_node = out_conf->node()->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(out_node);
  const auto &out_node_inputs = out_node->inputs();
  if (out_node->inputs().empty() || (out_node_inputs.size() - 1) != args_conf_list.size()) {
    MS_LOG(EXCEPTION) << "MixedPrecisionCast"
                      << " args size should equal to inputs size minus 1, but args size " << args_conf_list.size()
                      << ", inputs size " << out_node_inputs.size();
  }
  (void)std::transform(args_conf_list.begin(), args_conf_list.end(), std::back_inserter(args_spec_list),
                       [](const ConfigPtr &ref) -> AbstractBasePtr {
                         MS_EXCEPTION_IF_NULL(ref);
                         const auto &eval_result = ref->ObtainEvalResult();
                         MS_EXCEPTION_IF_NULL(eval_result);
                         return eval_result->abstract();
                       });

  ScopeGuard scope_guard(out_conf->node()->scope());
  TraceGuard trace_guard(std::make_shared<TraceMixedPrecision>(out_conf->node()->debug_info()));

  FuncGraphPtr func_graph = out_node->func_graph();
  constexpr size_t source_node_index = 2;
  if (out_node_inputs.size() <= source_node_index) {
    MS_LOG(EXCEPTION) << "Input size:" << out_node_inputs.size() << " should bigger than 2.";
  }

  AnfNodePtr new_node =
    MixedPrecisionCastHelper(out_node_inputs[source_node_index], args_spec_list[1], out_node_inputs[1], func_graph);
  AnfNodeConfigPtr fn_conf = engine->MakeConfig(new_node, out_conf->context(), out_conf->func_graph());

  if (new_node->isa<CNode>()) {
    auto new_cnode = new_node->cast_ptr<CNode>();
    new_cnode->CloneCNodeInfo(out_node);
  }
  return engine->ForwardConfig(out_conf, fn_conf);
}

namespace {
py::object BuildValue(const ValuePtr &value_ptr) {
  if (value_ptr == nullptr) {
    return py::none();
  } else {
    return ValueToPyData(value_ptr);
  }
}

py::object AbstractTupleValueToPython(const AbstractTuple *tuple_abs) {
  MS_EXCEPTION_IF_NULL(tuple_abs);
  auto value = tuple_abs->BuildValue();
  if (value->isa<AnyValue>()) {
    return py::none();
  }
  const auto &elements = tuple_abs->elements();
  size_t len = elements.size();
  py::tuple value_tuple(len);
  for (size_t i = 0; i < len; ++i) {
    value_tuple[i] = ConvertAbstractToPython(elements[i], true)[ATTR_VALUE];
  }
  return value_tuple;
}

tensor::TensorPtr GetShapeValue(const AbstractBasePtr &arg_element) {
  ValuePtr const_value = nullptr;
  if (arg_element->isa<abstract::AbstractTensor>()) {
    auto const_abstract_value = arg_element->cast_ptr<abstract::AbstractTensor>();
    MS_EXCEPTION_IF_NULL(const_abstract_value);
    const_value = const_abstract_value->BuildValue();
  } else if (arg_element->isa<abstract::AbstractScalar>()) {
    auto const_abstract_value = arg_element->cast_ptr<abstract::AbstractScalar>();
    MS_EXCEPTION_IF_NULL(const_abstract_value);
    const_value = const_abstract_value->BuildValue();
  } else {
    MS_LOG(EXCEPTION) << "Unsupported shape data:" << arg_element->ToString();
  }
  MS_EXCEPTION_IF_NULL(const_value);
  return const_value->cast<tensor::TensorPtr>();
}

py::dict AbstractTupleToPython(const AbstractBasePtr &abs_base, bool only_convert_value) {
  auto arg_tuple = dyn_cast_ptr<AbstractTuple>(abs_base);
  MS_EXCEPTION_IF_NULL(arg_tuple);
  auto dic = py::dict();
  if (only_convert_value) {
    dic[ATTR_VALUE] = AbstractTupleValueToPython(arg_tuple);
    return dic;
  }
  size_t len = arg_tuple->size();
  py::tuple shape_tuple(len);
  py::tuple dtype_tuple(len);
  py::tuple value_tuple(len);
  py::tuple min_value_tuple(len);
  py::tuple max_value_tuple(len);
  py::tuple min_shape_tuple(len);
  py::tuple max_shape_tuple(len);
  py::tuple shape_value_tuple(len);
  std::vector<py::dict> res;

  bool dyn_shape = false;
  bool dyn_value = false;
  bool dyn_shape_value = false;
  for (size_t i = 0; i < len; i++) {
    py::dict out = ConvertAbstractToPython(arg_tuple->elements()[i]);
    res.push_back(out);
    shape_tuple[i] = out[ATTR_SHAPE];
    dtype_tuple[i] = out[ATTR_DTYPE];
    value_tuple[i] = out[ATTR_VALUE];
    if (out.contains(py::str(ATTR_SHAPE_VALUE))) {
      shape_value_tuple[i] = out[ATTR_SHAPE_VALUE];
      dyn_shape_value = true;
    }

    // Elements in tuple is tensor shape value.
    if (out.contains(py::str(ATTR_MIN_VALUE)) && out.contains(py::str(ATTR_MAX_VALUE))) {
      min_value_tuple[i] = out[ATTR_MIN_VALUE];
      max_value_tuple[i] = out[ATTR_MAX_VALUE];
      dyn_value = true;
    } else {
      min_value_tuple[i] = out[ATTR_VALUE];
      max_value_tuple[i] = out[ATTR_VALUE];
    }

    // Elements in tuple is tensor, which shape is dynamic.
    if (out.contains(py::str(ATTR_MIN_SHAPE)) && out.contains(py::str(ATTR_MAX_SHAPE))) {
      min_shape_tuple[i] = out[ATTR_MIN_SHAPE];
      max_shape_tuple[i] = out[ATTR_MAX_SHAPE];
      dyn_shape = true;
    } else {
      min_shape_tuple[i] = out[ATTR_SHAPE];
      max_shape_tuple[i] = out[ATTR_SHAPE];
    }
  }
  dic[ATTR_SHAPE] = shape_tuple;
  dic[ATTR_DTYPE] = dtype_tuple;
  MS_EXCEPTION_IF_NULL(arg_tuple->BuildValue());
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
  if (dyn_shape_value) {
    for (size_t i = 0; i < len; i++) {
      if (!res[i].contains(py::str(ATTR_SHAPE_VALUE))) {
        auto arg_element = arg_tuple->elements()[i];
        MS_EXCEPTION_IF_NULL(arg_element);
        auto const_tensor = GetShapeValue(arg_element);
        if (const_tensor == nullptr) {
          return dic;
        }
        std::vector<int64_t> const_tensor_vector = TensorValueToVector<int64_t>(const_tensor);
        shape_value_tuple[i] = BuildValue(MakeValue(const_tensor_vector));
      }
    }
    dic[ATTR_SHAPE_VALUE] = shape_value_tuple;
  }

  return dic;
}

py::dict AbstractDictionaryToPython(const AbstractBasePtr &abs_base) {
  auto arg_dict = dyn_cast_ptr<AbstractDictionary>(abs_base);
  MS_EXCEPTION_IF_NULL(arg_dict);

  size_t len = arg_dict->size();
  const auto &arg_dict_elements = arg_dict->elements();
  py::list shape_list(len);
  py::list dtype_list(len);
  py::dict value_dict = py::dict();

  for (size_t i = 0; i < len; ++i) {
    auto cur_attr = arg_dict_elements[i];
    auto cur_key = cur_attr.first;
    auto cur_value = cur_attr.second;

    py::dict cur_value_out = ConvertAbstractToPython(cur_value);
    shape_list[i] = cur_value_out[ATTR_SHAPE];
    dtype_list[i] = cur_value_out[ATTR_DTYPE];
    value_dict[py::str(cur_key)] = cur_value_out[ATTR_VALUE];
  }

  py::dict dic = py::dict();
  dic[ATTR_SHAPE] = shape_list;
  dic[ATTR_DTYPE] = dtype_list;
  MS_EXCEPTION_IF_NULL(arg_dict->BuildValue());
  if (arg_dict->BuildValue()->isa<AnyValue>()) {
    dic[ATTR_VALUE] = py::none();
  } else {
    dic[ATTR_VALUE] = value_dict;
  }
  return dic;
}

py::object AbstractListValueToPython(const AbstractList *list_abs) {
  MS_EXCEPTION_IF_NULL(list_abs);
  auto value = list_abs->BuildValue();
  if (value->isa<AnyValue>()) {
    return py::none();
  }
  const auto &elements = list_abs->elements();
  size_t len = elements.size();
  py::list value_list(len);
  for (size_t i = 0; i < len; ++i) {
    value_list[i] = ConvertAbstractToPython(elements[i], true)[ATTR_VALUE];
  }
  return value_list;
}

py::dict AbstractListToPython(const AbstractBasePtr &abs_base, bool only_convert_value) {
  auto arg_list = dyn_cast_ptr<AbstractList>(abs_base);
  MS_EXCEPTION_IF_NULL(arg_list);
  auto dic = py::dict();
  if (only_convert_value) {
    dic[ATTR_VALUE] = AbstractListValueToPython(arg_list);
    return dic;
  }
  size_t len = arg_list->size();
  py::list shape_list(len);
  py::list dtype_list(len);
  py::list value_list(len);
  py::list min_value_list(len);
  py::list max_value_list(len);
  py::list min_shape_list(len);
  py::list max_shape_list(len);
  py::list shape_value_list(len);
  std::vector<py::dict> res;

  bool dyn_value = false;
  bool dyn_shape = false;
  bool shape_value = false;

  for (size_t i = 0; i < len; i++) {
    py::dict out = ConvertAbstractToPython(arg_list->elements()[i]);
    res.push_back(out);
    shape_list[i] = out[ATTR_SHAPE];
    dtype_list[i] = out[ATTR_DTYPE];
    value_list[i] = out[ATTR_VALUE];

    if (out.contains(py::str(ATTR_SHAPE_VALUE))) {
      shape_value_list[i] = out[ATTR_SHAPE_VALUE];
      shape_value = true;
    }
    // Elements in list is tensor, which value is dynamic.
    if (out.contains(py::str(ATTR_MIN_VALUE)) && out.contains(py::str(ATTR_MAX_VALUE))) {
      min_value_list[i] = out[ATTR_MIN_VALUE];
      max_value_list[i] = out[ATTR_MAX_VALUE];
      dyn_value = true;
    } else {
      min_value_list[i] = out[ATTR_VALUE];
      max_value_list[i] = out[ATTR_VALUE];
    }

    // Elements in list is tensor, which shape is dynamic.
    if (out.contains(py::str(ATTR_MIN_SHAPE)) && out.contains(py::str(ATTR_MAX_SHAPE))) {
      min_shape_list[i] = out[ATTR_MIN_SHAPE];
      max_shape_list[i] = out[ATTR_MAX_SHAPE];
      dyn_shape = true;
    } else {
      min_shape_list[i] = out[ATTR_SHAPE];
      max_shape_list[i] = out[ATTR_SHAPE];
    }
  }

  dic[ATTR_SHAPE] = shape_list;
  dic[ATTR_DTYPE] = dtype_list;
  MS_EXCEPTION_IF_NULL(arg_list->BuildValue());
  if (arg_list->BuildValue()->isa<AnyValue>()) {
    dic[ATTR_VALUE] = py::none();
  } else {
    dic[ATTR_VALUE] = value_list;
  }

  if (dyn_value) {
    dic[ATTR_MIN_VALUE] = min_value_list;
    dic[ATTR_MAX_VALUE] = max_value_list;
  }
  if (dyn_shape) {
    dic[ATTR_MIN_SHAPE] = min_shape_list;
    dic[ATTR_MAX_SHAPE] = max_shape_list;
  }
  if (shape_value) {
    for (size_t i = 0; i < len; i++) {
      if (!res[i].contains(py::str(ATTR_SHAPE_VALUE))) {
        auto arg_element = arg_list->elements()[i];
        MS_EXCEPTION_IF_NULL(arg_element);
        auto const_tensor = GetShapeValue(arg_element);
        if (const_tensor == nullptr) {
          return dic;
        }
        std::vector<int64_t> const_tensor_vector = TensorValueToVector<int64_t>(const_tensor);
        shape_value_list[i] = BuildValue(MakeValue(const_tensor_vector));
      }
    }
    dic[ATTR_SHAPE_VALUE] = shape_value_list;
  }
  return dic;
}

void ConvertAbstractTensorToPython(const AbstractBasePtr &abs_base, bool only_convert_value, py::dict *dic) {
  auto arg_tensor = dyn_cast_ptr<AbstractTensor>(abs_base);
  MS_EXCEPTION_IF_NULL(dic);
  MS_EXCEPTION_IF_NULL(arg_tensor);
  if (only_convert_value) {
    (*dic)[ATTR_VALUE] = BuildValue(arg_tensor->BuildValue());
    return;
  }
  MS_EXCEPTION_IF_NULL(arg_tensor->shape());
  (*dic)[ATTR_SHAPE] = arg_tensor->shape()->shape();
  const auto &min_shape = arg_tensor->shape()->min_shape();
  const auto &max_shape = arg_tensor->shape()->max_shape();
  if (!min_shape.empty() && !max_shape.empty()) {
    (*dic)[ATTR_MIN_SHAPE] = min_shape;
    (*dic)[ATTR_MAX_SHAPE] = max_shape;
  }

  auto min_value = arg_tensor->get_min_value();
  auto max_value = arg_tensor->get_max_value();
  auto shape_value = arg_tensor->get_shape_value();
  if (min_value != nullptr && max_value != nullptr) {
    (*dic)[ATTR_MIN_VALUE] = BuildValue(min_value);
    (*dic)[ATTR_MAX_VALUE] = BuildValue(max_value);
  }
  if (shape_value != nullptr) {
    (*dic)[ATTR_SHAPE_VALUE] = BuildValue(shape_value);
  }
  (*dic)[ATTR_DTYPE] = arg_tensor->BuildType();
  (*dic)[ATTR_VALUE] = BuildValue(arg_tensor->BuildValue());
}
namespace {
py::object GetPyObjForPrimitiveAbstract(const PrimitiveAbstractClosurePtr &prim_abs) {
  auto prim = prim_abs->BuildValue();
  if (prim == nullptr) {
    return py::none();
  }
  if (prim->isa<prim::DoSignaturePrimitive>()) {
    auto do_sig_prim = prim->cast_ptr<prim::DoSignaturePrimitive>();
    auto value = do_sig_prim->function();
    if (!value->isa<PrimitivePy>()) {
      return py::none();
    }
    auto prim_py = value->cast_ptr<PrimitivePy>();
    return prim_py->GetPyObj();
  }
  if (prim->isa<PrimitivePy>()) {
    auto prim_py = prim->cast_ptr<PrimitivePy>();
    return prim_py->GetPyObj();
  }
  return py::none();
}

bool IsCallInstance(const PartialAbstractClosurePtr &partial_abs) {
  auto fn = partial_abs->fn();
  if (!fn->isa<PrimitiveAbstractClosure>()) {
    return false;
  }
  auto abs_prim = fn->cast_ptr<PrimitiveAbstractClosure>();
  auto prim = abs_prim->prim();
  if (prim->name() == prim::kPrimCallInstance->name()) {
    return true;
  }
  return false;
}
}  // namespace

void ConvertAbstractFunctionToPython(const AbstractBasePtr &abs_base, py::dict *dic) {
  MS_EXCEPTION_IF_NULL(dic);
  MS_EXCEPTION_IF_NULL(abs_base);
  (*dic)[ATTR_SHAPE] = py::none();
  (*dic)[ATTR_DTYPE] = abs_base->BuildType();
  (*dic)[ATTR_VALUE] = py::none();
  if (abs_base->isa<PartialAbstractClosure>()) {
    auto partial_abs = abs_base->cast<PartialAbstractClosurePtr>();
    AbstractBasePtrList args = partial_abs->args();
    if (!args.empty()) {
      auto value = args[0]->BuildValue();
      MS_EXCEPTION_IF_NULL(value);
      if (IsCallInstance(partial_abs)) {
        auto value_obj = value->cast_ptr<parse::MsClassObject>();
        if (value_obj != nullptr) {
          (*dic)[ATTR_DTYPE] = std::make_shared<MsClassType>();
          (*dic)[ATTR_VALUE] = value_obj->obj();
          return;
        }
      }
      auto value_obj = value->cast_ptr<parse::ClassType>();
      if (value_obj != nullptr) {
        (*dic)[ATTR_DTYPE] = std::make_shared<TypeType>();
        (*dic)[ATTR_VALUE] = value_obj->obj();
      }
    }
  }
  if (abs_base->isa<PrimitiveAbstractClosure>()) {
    (*dic)[ATTR_VALUE] = GetPyObjForPrimitiveAbstract(abs_base->cast<PrimitiveAbstractClosurePtr>());
  }
}

bool CheckType(const TypePtr &expected_type, const TypePtr &x) {
  // As x and predicate both are mindspore type statically, here we only to judge whether
  // x is predicate or is a subclass of predicate.
  return IsIdentidityOrSubclass(x, expected_type);
}

// Join all types in args_type_list;
TypePtr TypeJoin(const TypePtrList &args_type_list) {
  if (args_type_list.empty()) {
    MS_LOG(EXCEPTION) << "args_type_list is empty";
  }

  TypePtr type_tmp = args_type_list[0];
  for (std::size_t i = 1; i < args_type_list.size(); i++) {
    type_tmp = abstract::TypeJoin(type_tmp, args_type_list[i]);
  }
  return type_tmp;
}

TypePtr CheckTypeList(const TypePtr &predicate, const TypePtrList &args_type_list) {
  MS_EXCEPTION_IF_NULL(predicate);
  for (const auto &arg_type : args_type_list) {
    MS_EXCEPTION_IF_NULL(arg_type);
    if (!CheckType(predicate, arg_type)) {
      MS_LOG(EXCEPTION) << "The expected is " << predicate->ToString() << ", not " << arg_type->ToString();
    }
  }
  return TypeJoin(args_type_list);
}
}  // namespace

py::dict ConvertAbstractToPython(const AbstractBasePtr &abs_base, bool only_convert_value) {
  MS_EXCEPTION_IF_NULL(abs_base);
  auto dic = py::dict();
  if (abs_base->isa<AbstractTensor>()) {
    ConvertAbstractTensorToPython(abs_base, only_convert_value, &dic);
  } else if (abs_base->isa<AbstractScalar>() || abs_base->isa<AbstractType>()) {
    ShapeVector shape;
    dic[ATTR_SHAPE] = shape;
    dic[ATTR_DTYPE] = abs_base->BuildType();
    dic[ATTR_VALUE] = BuildValue(abs_base->BuildValue());
  } else if (abs_base->isa<AbstractTuple>()) {
    return AbstractTupleToPython(abs_base, only_convert_value);
  } else if (abs_base->isa<AbstractList>()) {
    return AbstractListToPython(abs_base, only_convert_value);
  } else if (abs_base->isa<AbstractDictionary>()) {
    return AbstractDictionaryToPython(abs_base);
  } else if (abs_base->isa<AbstractSlice>()) {
    auto arg_slice = dyn_cast_ptr<AbstractSlice>(abs_base);
    ShapeVector shape;
    dic[ATTR_SHAPE] = shape;
    dic[ATTR_DTYPE] = arg_slice->BuildType();
    dic[ATTR_VALUE] = BuildValue(arg_slice->BuildValue());
  } else if (abs_base->isa<AbstractRowTensor>()) {
    auto arg = dyn_cast_ptr<AbstractRowTensor>(abs_base);
    dic[ATTR_SHAPE] = arg->shape()->shape();
    dic[ATTR_DTYPE] = arg->BuildType();
    dic[ATTR_VALUE] = BuildValue(arg->BuildValue());
  } else if (abs_base->isa<AbstractCOOTensor>()) {
    auto arg = dyn_cast_ptr<AbstractCOOTensor>(abs_base);
    AbstractBasePtrList sparse_shape = arg->shape()->elements();
    ShapeVector sparse_shape_vector;
    (void)std::transform(sparse_shape.begin(), sparse_shape.end(), std::back_inserter(sparse_shape_vector),
                         [](const AbstractBasePtr &e) -> int64_t {
                           ValuePtr value = e->cast_ptr<AbstractScalar>()->BuildValue();
                           return GetValue<int64_t>(value);
                         });
    dic[ATTR_SHAPE] = sparse_shape_vector;
    dic[ATTR_DTYPE] = arg->BuildType();
    dic[ATTR_VALUE] = BuildValue(arg->BuildValue());
  } else if (abs_base->isa<AbstractCSRTensor>()) {
    auto arg = dyn_cast_ptr<AbstractCSRTensor>(abs_base);
    AbstractBasePtrList sparse_shape = arg->shape()->elements();
    ShapeVector sparse_shape_vector;
    (void)std::transform(sparse_shape.begin(), sparse_shape.end(), std::back_inserter(sparse_shape_vector),
                         [](const AbstractBasePtr &e) -> int64_t {
                           ValuePtr value = e->cast_ptr<AbstractScalar>()->BuildValue();
                           return GetValue<int64_t>(value);
                         });
    dic[ATTR_SHAPE] = sparse_shape_vector;
    dic[ATTR_DTYPE] = arg->BuildType();
    dic[ATTR_VALUE] = BuildValue(arg->BuildValue());
  } else if (abs_base->isa<AbstractEllipsis>()) {
    dic[ATTR_SHAPE] = py::none();
    dic[ATTR_DTYPE] = py::ellipsis();
    dic[ATTR_VALUE] = py::ellipsis();
  } else if (abs_base->isa<AbstractNone>()) {
    dic[ATTR_SHAPE] = py::none();
    dic[ATTR_DTYPE] = py::none();
    dic[ATTR_VALUE] = py::none();
  } else if (abs_base->isa<AbstractFunction>()) {
    ConvertAbstractFunctionToPython(abs_base, &dic);
  } else if (abs_base->isa<AbstractUndetermined>()) {
    auto arg = dyn_cast_ptr<AbstractUndetermined>(abs_base);
    dic[ATTR_SHAPE] = py::none();
    dic[ATTR_DTYPE] = arg->BuildType();
    dic[ATTR_VALUE] = py::none();
  } else if (abs_base->isa<AbstractMonad>()) {
    dic[ATTR_SHAPE] = py::none();
    dic[ATTR_DTYPE] = abs_base->BuildType();
    dic[ATTR_VALUE] = py::none();
  } else {
    auto value = abs_base->BuildValue();
    MS_EXCEPTION_IF_NULL(value);
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
py::tuple PreparePyInputs(const PrimitivePyPtr &, const AbstractBasePtrList &args) {
  // The monad parameter is defined at the end of the parameter and needs to be ignored
  std::size_t size_args = args.size() - GetAbstractMonadNum(args);
  py::tuple py_args(size_args);
  for (size_t i = 0; i < size_args; i++) {
    auto arg_i = (args)[i];
    py_args[i] = ConvertAbstractToPython(arg_i);
  }
  return py_args;
}

void CheckCustomPrimOutputInferResult(const PrimitivePtr &prim, const AbstractBasePtr &res_spec) {
  MS_EXCEPTION_IF_NULL(prim);
  MS_EXCEPTION_IF_NULL(res_spec);
  const string kOutputNum = "output_num";
  if (prim->IsCustomPrim()) {
    // Raise error if output_num is not match the infer result.
    auto output_num_value = prim->GetAttr(kOutputNum);
    if (output_num_value == nullptr) {
      MS_LOG(DEBUG) << "The output num may no need to check";
      return;
    }
    int64_t output_num = GetValue<int64_t>(output_num_value);
    if (res_spec->isa<AbstractTensor>() && output_num != 1) {
      MS_LOG(EXCEPTION) << "Custom operator primitive[" << prim->ToString()
                        << "]'s attribute[output_num]:" << output_num << " not matches the infer result "
                        << res_spec->ToString();
    } else if (res_spec->isa<AbstractTuple>() &&
               (res_spec->cast_ptr<AbstractTuple>()->size() != LongToSize(output_num))) {
      MS_LOG(EXCEPTION) << "Custom operator primitive[" << prim->ToString()
                        << "]'s attribute[output_num]:" << output_num << " not matches the infer result "
                        << res_spec->ToString();
    }
  }
}

void SetShapeValue(const AbstractBasePtr &tensor, const py::object &output) {
  if (output.is_none()) {
    return;
  }
  if (output.contains(py::str(ATTR_MIN_VALUE)) && output.contains(py::str(ATTR_MAX_VALUE))) {
    const py::object &obj_min = output[ATTR_MIN_VALUE];
    const py::object &obj_max = output[ATTR_MAX_VALUE];
    if (obj_min.is_none() || obj_max.is_none()) {
      return;
    }
    bool converted = true;
    ValuePtr min_value = nullptr;
    ValuePtr max_value = nullptr;
    converted = parse::ConvertData(obj_min, &min_value);
    if (!converted) {
      MS_LOG(EXCEPTION) << "Convert shape min value data failed";
    }
    converted = parse::ConvertData(obj_max, &max_value);
    if (!converted) {
      MS_LOG(EXCEPTION) << "Convert shape max value data failed";
    }
    auto abs_tensor = dyn_cast_ptr<abstract::AbstractTensor>(tensor);
    MS_EXCEPTION_IF_NULL(abs_tensor);
    abs_tensor->set_value_range(min_value, max_value);
  }
  if (!output.contains(py::str(ATTR_SHAPE_VALUE))) {
    return;
  }
  const py::object &obj_shape_value = output[ATTR_SHAPE_VALUE];
  if (obj_shape_value.is_none()) {
    return;
  }
  bool converted = true;
  ValuePtr shape_value = nullptr;
  converted = parse::ConvertData(obj_shape_value, &shape_value);
  if (!converted) {
    MS_LOG(EXCEPTION) << "Convert shape value data failed";
  }
  auto abs_tensor = dyn_cast_ptr<abstract::AbstractTensor>(tensor);
  abs_tensor->set_shape_value(shape_value);
}

static bool IsMonadType(const py::object &type_obj) {
  if (py::isinstance<Type>(type_obj)) {
    auto type = type_obj.cast<Type *>();
    return type->isa<MonadType>();
  }
  return false;
}

AbstractBasePtr ToMonadAbstract(const py::object &type_obj) {
  if (py::isinstance<Type>(type_obj)) {
    auto type = type_obj.cast<Type *>();
    if (!type->isa<MonadType>()) {
      MS_LOG(EXCEPTION) << "Not a monad type object: " << py::str(type_obj);
    }
    return abstract::MakeMonadAbstract(type->cast<MonadTypePtr>());
  }
  MS_LOG(EXCEPTION) << "Not a type object: " << py::str(type_obj);
}

py::object GetPyAbsItemOfTupleOut(const py::object &output, const size_t index) {
  auto out_dict = output.cast<py::dict>();
  auto type_obj = out_dict[ATTR_DTYPE];
  auto shape_obj = out_dict[ATTR_SHAPE];
  auto out_item = py::dict();
  auto shape_tuple = shape_obj.cast<py::tuple>();
  auto typeid_tuple = type_obj.cast<py::tuple>();
  out_item[ATTR_DTYPE] = typeid_tuple[index];
  out_item[ATTR_SHAPE] = shape_tuple[index];
  if (output.contains(py::str(ATTR_MIN_SHAPE))) {
    out_item[ATTR_MIN_SHAPE] = output[ATTR_MIN_SHAPE].cast<py::tuple>()[index];
  }
  if (output.contains(py::str(ATTR_MAX_SHAPE))) {
    out_item[ATTR_MAX_SHAPE] = output[ATTR_MAX_SHAPE].cast<py::tuple>()[index];
  }
  out_item[ATTR_VALUE] = py::none();
  return out_item;
}

ShapeVector GetShapeVectorByAttr(const std::string &attr, const py::object &obj) {
  if (!obj.contains(py::str(attr))) {
    return {};
  }
  const py::object &shape = obj[py::str(attr)];
  if (shape.is_none()) {
    return {};
  }
  return shape.cast<ShapeVector>();
}

AbstractBasePtr MakePyInferRes2AbstractTensor(const py::object &shape_obj, const py::object &type_obj,
                                              const py::object &output) {
  auto ret_vec = shape_obj.cast<ShapeVector>();
  auto ret_dtype = type_obj.cast<TypePtr>();
  ShapeVector min_shape_vec;
  ShapeVector max_shape_vec;

  if (!output.is_none()) {
    min_shape_vec = GetShapeVectorByAttr(ATTR_MIN_SHAPE, output);
    max_shape_vec = GetShapeVectorByAttr(ATTR_MAX_SHAPE, output);
  }

  auto ret_shape = std::make_shared<abstract::Shape>(ret_vec, min_shape_vec, max_shape_vec);
  AbstractBasePtr tensor = MakeAbstractTensor(ret_shape, ret_dtype);

  SetShapeValue(tensor, output);
  return tensor;
}

AbstractBasePtr MakePyInferRes2Abstract(const py::object &output) {
  auto out_dict = output.cast<py::dict>();
  auto type_obj = out_dict[ATTR_DTYPE];
  auto shape_obj = out_dict[ATTR_SHAPE];
  if ((py::isinstance<py::list>(shape_obj) || py::isinstance<py::tuple>(shape_obj)) && py::isinstance<Type>(type_obj)) {
    auto ret_vec = shape_obj.cast<ShapeVector>();
    auto ret_dtype = type_obj.cast<TypePtr>();
    MS_EXCEPTION_IF_NULL(ret_dtype);
    // if the size of shape list is empty, return an scalar abstract
    if (ret_vec.empty() && (!ret_dtype->isa<TensorType>())) {
      abstract::AbstractScalarPtr abs_scalar = std::make_shared<abstract::AbstractScalar>(kAnyValue, ret_dtype);
      return abs_scalar;
    }
    return MakePyInferRes2AbstractTensor(shape_obj, type_obj, output);
  } else if (py::isinstance<py::tuple>(shape_obj) && py::isinstance<py::tuple>(type_obj)) {
    auto typeid_tuple = type_obj.cast<py::tuple>();
    AbstractBasePtrList ptr_list;
    for (size_t it = 0; it < typeid_tuple.size(); ++it) {
      auto output_it = GetPyAbsItemOfTupleOut(output, it);
      auto tensor_it = MakePyInferRes2Abstract(output_it);
      ptr_list.push_back(tensor_it);
    }
    auto tuple = std::make_shared<abstract::AbstractTuple>(ptr_list);
    return tuple;
  } else if (py::isinstance<py::list>(shape_obj) && py::isinstance<py::list>(type_obj)) {
    auto typeid_list = type_obj.cast<py::list>();
    AbstractBasePtrList ptr_list;
    for (size_t it = 0; it < typeid_list.size(); ++it) {
      auto output_it = GetPyAbsItemOfTupleOut(output, it);
      auto tensor_it = MakePyInferRes2Abstract(output_it);
      ptr_list.push_back(tensor_it);
    }
    auto list = std::make_shared<abstract::AbstractList>(ptr_list);
    return list;
  } else if (shape_obj.is_none() && type_obj.is_none()) {
    // AbstractNone indicates there is no output for this CNode node.
    auto abstract_none = std::make_shared<abstract::AbstractNone>();
    return abstract_none;
  } else if (IsMonadType(type_obj)) {
    // Return monad abstract if it is monad type.
    return ToMonadAbstract(type_obj);
  } else {
    MS_LOG(EXCEPTION) << "Python evaluator return invalid shape or type. " << py::str(type_obj);
  }
}

AbstractBasePtr PyInferRes2Abstract(const PrimitivePyPtr &prim_py, const py::dict &output) {
  // Convert to AbstractValue based on type and shape
  if (output[ATTR_VALUE].is_none()) {
    return MakePyInferRes2Abstract(output);
  }

  // Convert pyobject to Value, then to AbstractValue
  auto out_dtype = output[ATTR_DTYPE];
  TypePtr dtype = py::isinstance<Type>(out_dtype) ? out_dtype.cast<TypePtr>() : nullptr;
  ValuePtr converted_ret = nullptr;
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
    SetShapeValue(res_tensor, output);
  }
  CheckCustomPrimOutputInferResult(prim_py, res_spec);
  return res_spec;
}
}  // namespace

EvalResultPtr StandardPrimEvaluator::RunPyInferValue(const AnalysisEnginePtr &, const AbstractBasePtr &abs_base,
                                                     const AbstractBasePtrList &args) {
  auto prim_py = dyn_cast<PrimitivePy>(prim_);
  if (prim_py == nullptr) {
    MS_LOG(EXCEPTION) << "The primitive with type 'kPrimTypePyCheck' should be a python primitive.";
  }
  // Call checking method 'infer_value' for python primitive
  MS_LOG(DEBUG) << "Begin input args checking for: " << prim_py->ToString();
  auto py_args = PreparePyInputs(prim_py, args);
  py::tuple py_vals(py_args.size());
  auto added_attrs = prim_->evaluate_added_attrs();
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
    auto res_tensor = res_spec->cast_ptr<AbstractTensor>();
    res_tensor->set_value(converted_ret);
  }
  return std::make_shared<EvalResult>(res_spec, std::make_shared<AttrValueMap>(added_attrs));
}

// Apply EvalResult from cached result for a given primitive.
static inline EvalResultPtr ApplyCacheEvalResult(const PrimitivePtr &prim, const EvalResultPtr &result) {
  auto &attrs = result->attribute();
  if (attrs != nullptr) {
    prim->set_evaluate_added_attrs(*attrs);
  }
  return std::make_shared<EvalResult>(result->abstract()->Clone(), attrs);
}

EvalResultPtr StandardPrimEvaluator::EvalPyCheckPrim(const AnalysisEnginePtr &engine, const AbstractBasePtrList &args) {
  // Try to get infer result from evaluator cache.
  auto eval_result = evaluator_cache_mgr_->GetValue(args);
  if (eval_result != nullptr) {
    // Evaluator cache hit.
    return std::make_shared<EvalResult>(eval_result->abstract()->Clone(), eval_result->attribute());
  }
  // In pynative mode (engine == nullptr), it is difficult to set added_attrs to
  // python object by C++ code, so we disable global eval cache in pynative mode.
  const bool enable_global_cache = (engine != nullptr);
  if (enable_global_cache) {
    // Try to get infer result from global primitive evaluate cache.
    eval_result = eval_cache_->Get(prim_, args);
    if (eval_result != nullptr) {
      // Global primitive evaluate cache hit.
      evaluator_cache_mgr_->SetValue(args, eval_result);
      return ApplyCacheEvalResult(prim_, eval_result);
    }
  }
  // PrimitivePy is expected for EvalPyCheckPrim.
  auto prim_py = dyn_cast<PrimitivePy>(prim_);
  if (prim_py == nullptr) {
    MS_LOG(EXCEPTION) << "The primitive with type 'kPrimTypePyCheck' should be a python primitive.";
  }
  // We should copy attributes before running check and infer,
  // since they may be changed during check and infer.
  auto input_attrs = prim_py->attrs();
  prim_py->BeginRecordAddAttr();
  auto py_args = PreparePyInputs(prim_py, args);
  // Call checking method '__check__' for subclass of 'PrimitiveWithCheck'.
  prim_py->RunCheck(py_args);
  auto abs = eval_impl_.InferShapeAndType(engine, prim_py, args);
  MS_EXCEPTION_IF_NULL(abs);
  prim_py->EndRecordAddAttr();
  auto &added_attrs = prim_py->evaluate_added_attrs();
  eval_result = std::make_shared<EvalResult>(abs, std::make_shared<AttrValueMap>(added_attrs));
  if (py::hasattr(prim_py->GetPyObj(), PY_PRIM_METHOD_INFER_VALUE)) {
    // Call 'infer_value()' method if it is exsited, for constant propagation.
    eval_result = RunPyInferValue(engine, eval_result->abstract(), args);
  }
  // Save infer result to caches (evaluator cache and global cache).
  if (enable_global_cache) {
    eval_cache_->Put(prim_py, std::move(input_attrs), args, eval_result);
  }
  evaluator_cache_mgr_->SetValue(args, eval_result);
  return eval_result;
}

namespace {
void CheckSequenceArgumentForCppPrimitive(const PrimitivePtr &prim, const AbstractBasePtrList &args) {
  // To check tuple/list operations with a white list of Python primitive.
  MS_EXCEPTION_IF_NULL(prim);
  auto iter = prims_transparent_pass_sequence.find(prim->name());
  if (iter == prims_transparent_pass_sequence.end()) {
    // The primitive use all elements of each argument.
    for (size_t i = 0; i < args.size(); ++i) {
      MS_EXCEPTION_IF_NULL(args[i]);
      if (args[i]->isa<abstract::AbstractSequence>()) {
        MS_LOG(DEBUG) << "Primitive \'" << prim->name() << "\' is consuming tuple/list arguments[" << i
                      << "]: " << args[i]->ToString();
        SetSequenceElementsUseFlagsRecursively(args[i], true);
      }
    }
    return;
  }

  // It's transparent pass primitive or using partial elements primitive.
  auto index_list = iter->second;
  if (index_list.empty()) {
    MS_LOG(EXCEPTION) << "The primitive list should not be empty for " << prim->name();
  }
  // Ignore all arguments, no need checking if AbstractSequence.
  if (index_list[0] == -1) {
    return;
  }
  // Check the specific arguments index.
  for (size_t i = 0; i < args.size(); ++i) {
    if (!args[i]->isa<abstract::AbstractSequence>()) {
      continue;
    }
    if (std::find(index_list.begin(), index_list.end(), i) == index_list.end()) {
      // For current tuple/list argument, it's not a primitive of total transparent pass or partial element use.
      MS_LOG(DEBUG) << "Primitive \'" << prim->name() << "\' is consuming specific tuple/list arguments[" << i
                    << "]: " << args[i]->ToString();
      SetSequenceElementsUseFlagsRecursively(args[i], true);
    }
  }
}

void CheckSequenceArgumentForPythonPrimitive(const PrimitivePtr &prim, const AbstractBasePtrList &args) {
  MS_EXCEPTION_IF_NULL(prim);
  // Consider all primitive implemented python infer() real use the tuple/list arguments.
  for (size_t i = 0; i < args.size(); ++i) {
    if (args[i]->isa<abstract::AbstractSequence>()) {
      MS_EXCEPTION_IF_NULL(args[i]);
      MS_LOG(DEBUG) << "Primitive \'" << prim->name() << "\' is consuming tuple/list arguments[" << i
                    << "]: " << args[i]->ToString();
      SetSequenceElementsUseFlagsRecursively(args[i], true);
    }
  }
}
}  // namespace

EvalResultPtr StandardPrimEvaluator::EvalPrim(const AnalysisEnginePtr &engine, const AbstractBasePtrList &args) {
  // To check tuple/list operations with a white list of Python primitive.
  CheckSequenceArgumentForCppPrimitive(prim_, args);

  if (prims_to_skip_undetermined_infer.find(prim_->name()) == prims_to_skip_undetermined_infer.end()) {
    auto ret_abstract = EvalUndeterminedArgs(args);
    if (ret_abstract != nullptr) {
      MS_LOG(DEBUG) << "StandardPrimEvaluator eval Undetermined";
      return ret_abstract;
    }
  }
  if (prim_->prim_type() == PrimType::kPrimTypePyCheck) {
    return EvalPyCheckPrim(engine, args);
  }
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  bool need_infer_value = !eval_impl_.IsInWhileList();
  if (need_infer_value == false) {
    need_infer_value = ((context->get_param<int>(MS_CTX_EXECUTION_MODE) == kGraphMode)) &&
                       std::all_of(args.begin(), args.end(), [](const AbstractBasePtr &abs) -> bool {
                         MS_EXCEPTION_IF_NULL(abs);
                         auto value = abs->BuildValue();
                         return (value != nullptr && !value->isa<AnyValue>() && !value->isa<None>() &&
                                 !value->isa<Monad>() && !value->isa<FuncGraph>());
                       });
  }
  AbstractBasePtr abs_base = nullptr;
  ValuePtr value = nullptr;
  prim_->BeginRecordAddAttr();
  if (need_infer_value && eval_impl_.IsImplInferValue()) {
    value = eval_impl_.InferValue(prim_, args);
    if (value != nullptr) {
      abs_base = value->ToAbstract();
      prim_->EndRecordAddAttr();
      auto added_attrs = prim_->evaluate_added_attrs();
      return std::make_shared<EvalResult>(abs_base, std::make_shared<AttrValueMap>(added_attrs));
    }
  }
  abs_base = eval_impl_.InferShapeAndType(engine, prim_, args);
  MS_EXCEPTION_IF_NULL(abs_base);
  prim_->EndRecordAddAttr();
  const auto &added_attrs = prim_->evaluate_added_attrs();
  return std::make_shared<EvalResult>(abs_base, std::make_shared<AttrValueMap>(added_attrs));
}

EvalResultPtr PythonPrimEvaluator::EvalPrim(const AnalysisEnginePtr &engine, const AbstractBasePtrList &args) {
  // Consider all primitive implemented python infer() real use the tuple/list arguments.
  CheckSequenceArgumentForPythonPrimitive(prim_py_, args);

  // Ensure input arguments are evaluated.
  auto ret_abstract = EvalUndeterminedArgs(args);
  if (ret_abstract != nullptr) {
    MS_LOG(DEBUG) << "PythonPrimEvaluator eval Undetermined";
    return ret_abstract;
  }
  auto forbid_reuse = prim_py_->HasAttr(GRAPH_FLAG_FORBID_REUSE_RESULT);
  if (!forbid_reuse) {
    // Try to get infer result from evaluator cache.
    EvalResultPtr eval_result = evaluator_cache_mgr_->GetValue(args);
    if (eval_result != nullptr) {
      return std::make_shared<EvalResult>(eval_result->abstract()->Clone(), eval_result->attribute());
    }
  }
  // In pynative mode (engine == nullptr), it is difficult to set added_attrs to
  // python object by C++ code, so we disable global eval cache in pynative mode.
  const bool enable_global_cache = (engine != nullptr && !forbid_reuse);
  if (enable_global_cache) {
    // Try to get infer result from global primitive eval cache.
    EvalResultPtr eval_result = eval_cache_->Get(prim_py_, args);
    if (eval_result != nullptr) {
      // Global cache hit.
      evaluator_cache_mgr_->SetValue(args, eval_result);
      return ApplyCacheEvalResult(prim_py_, eval_result);
    }
  }
  // Cache miss, run infer. We should copy attributes before
  // running infer, since they may be changed during infer.
  auto input_attrs = prim_py_->attrs();
  auto py_args = PreparePyInputs(prim_py_, args);
  prim_py_->BeginRecordAddAttr();
  py::dict output = prim_py_->RunInfer(py_args);
  prim_py_->EndRecordAddAttr();
  const auto &added_attrs = prim_py_->evaluate_added_attrs();
  MS_LOG(DEBUG) << "Output type is " << py::str(output);
  auto res_abs = PyInferRes2Abstract(prim_py_, output);
  MS_LOG(DEBUG) << "Python InferTensor result abstract: " << res_abs->ToString();
  EvalResultPtr eval_result = std::make_shared<EvalResult>(res_abs, std::make_shared<AttrValueMap>(added_attrs));
  // Save result to global primitive eval cache.
  if (enable_global_cache) {
    eval_cache_->Put(prim_py_, std::move(input_attrs), args, eval_result);
  }
  evaluator_cache_mgr_->SetValue(args, eval_result);
  return eval_result;
}

EvalResultPtr UniformPrimEvaluator::EvalPrim(const AnalysisEnginePtr &, const AbstractBasePtrList &args) {
  auto ret_abstract = EvalUndeterminedArgs(args);
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
      auto arg_scalar = dyn_cast_ptr<AbstractScalar>(arg);
      const auto &arg_value = arg_scalar->GetValueTrack();
      value_list.push_back(arg_value);
    } else {
      // Raise TypeError Expected Scalar.
      MS_LOG(EXCEPTION) << "Expect scalar arguments for uniform primitives.";
    }
  }
  for (const auto &item : type_map_) {
    TypePtrList selections;
    (void)std::transform(item.second.begin(), item.second.end(), std::back_inserter(selections),
                         [&args](size_t arg_idx) -> TypePtr {
                           if (arg_idx >= args.size()) {
                             MS_LOG(EXCEPTION) << "Index: " << arg_idx << " out of range: " << args.size();
                           }
                           MS_EXCEPTION_IF_NULL(args[arg_idx]);
                           return args[arg_idx]->GetTypeTrack();
                         });
    TypePtr res = CheckTypeList(item.first, selections);
    MS_EXCEPTION_IF_NULL(return_value_type_);
    MS_EXCEPTION_IF_NULL(item.first);
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
EvaluatorPtr InitStandardPrimEvaluator(PrimitivePtr primitive, const StandardPrimitiveImplReg eval_impl) {
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

inline void AddToManager(const AnalysisEnginePtr &engine, const FuncGraphPtr func_graph) {
  MS_EXCEPTION_IF_NULL(engine);
  FuncGraphManagerPtr manager = engine->func_graph_manager();
  manager->AddFuncGraph(func_graph);
}

enum class REQUIRE_TYPE { ATTR, METHOD };

EvalResultPtr StaticGetterInferred(const ValuePtr &value, const ConfigPtr &data_conf, const AnfNodeConfigPtr &old_conf,
                                   REQUIRE_TYPE require_type = REQUIRE_TYPE::METHOD) {
  MS_EXCEPTION_IF_NULL(old_conf);
  AbstractBasePtr abstract = ToAbstract(value, AnalysisContext::DummyContext(), old_conf);
  // Create new cnode
  std::vector<AnfNodePtr> input = {NewValueNode(prim::kPrimPartial)};
  auto func_graph_func = dyn_cast_ptr<abstract::FuncGraphAbstractClosure>(abstract);
  if (func_graph_func != nullptr) {
    FuncGraphPtr fg = func_graph_func->func_graph();
    input.push_back(NewValueNode(fg));
  } else {
    auto prim_func = dyn_cast_ptr<abstract::PrimitiveAbstractClosure>(abstract);
    MS_EXCEPTION_IF_NULL(prim_func);
    PrimitivePtr prim = prim_func->prim();
    input.push_back(NewValueNode(prim));
  }

  auto conf = dyn_cast_ptr<abstract::AnfNodeConfig>(data_conf);
  MS_EXCEPTION_IF_NULL(conf);
  input.push_back(conf->node());
  MS_EXCEPTION_IF_NULL(old_conf);
  FuncGraphPtr func_graph = old_conf->node()->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  CNodePtr new_cnode = func_graph->NewCNode(input);
  if (require_type == REQUIRE_TYPE::ATTR) {
    new_cnode = func_graph->NewCNode({new_cnode});
  }
  AnalysisEnginePtr eng = old_conf->engine();
  AnfNodeConfigPtr fn_conf = eng->MakeConfig(new_cnode, old_conf->context(), old_conf->func_graph());
  return eng->ForwardConfig(old_conf, fn_conf);
}

EvalResultPtr GetEvaluatedValueForNameSpaceString(const AbstractBasePtrList &args_spec_list, const ValuePtr &data_value,
                                                  const AnfNodeConfigPtr &out_conf, const std::string &data) {
  constexpr size_t item_index = 1;
  auto item_args = args_spec_list[item_index];
  ValuePtr item_value = item_args->BuildValue();
  MS_EXCEPTION_IF_NULL(data_value);
  MS_EXCEPTION_IF_NULL(item_value);
  if (item_value->isa<StringImm>()) {
    item_value = std::make_shared<parse::Symbol>(item_value->cast_ptr<StringImm>()->value());
  }
  if (!item_value->isa<parse::Symbol>()) {
    MS_LOG(EXCEPTION) << "The value of the attribute could not be inferred: " << item_value->ToString();
  }

  // item_name to func addr from obj_map
  auto symbol = item_value->cast<parse::SymbolPtr>();
  auto name_space = data_value->cast<parse::NameSpacePtr>();
  MS_EXCEPTION_IF_NULL(out_conf);
  auto out_node = out_conf->node();
  FuncGraphPtr func_graph = out_node->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  auto new_node = parse::ResolveSymbol(func_graph->manager(), name_space, symbol, out_node);
  if (new_node == nullptr) {
    MS_LOG(EXCEPTION) << "Resolve node failed";
  }

  if (IsValueNode<TypeNull>(new_node)) {
    // Do not find the attribute.
    constexpr auto max_args_len = 3;
    bool has_default = (args_spec_list.size() == max_args_len);
    if (!has_default) {
      MS_EXCEPTION(AttributeError) << data << " object has no attribute " << symbol->symbol();
    }
    auto out_cnode = out_node->cast_ptr<CNode>();
    MS_EXCEPTION_IF_NULL(out_cnode);
    constexpr auto default_index = 3;
    auto default_node = out_cnode->inputs()[default_index];
    func_graph->ReplaceInOrder(out_node, default_node);
    auto eng = out_conf->engine();
    MS_EXCEPTION_IF_NULL(eng);
    auto fn_conf = eng->MakeConfig(default_node, out_conf->context(), out_conf->func_graph());
    return eng->ForwardConfig(out_conf, fn_conf);
  }
  if (pipeline::GetJitLevel() == "O0" && IsValueNode<FuncGraph>(new_node)) {
    UpdateDebugInfo(GetValueNode<FuncGraphPtr>(new_node), out_node->scope(), out_node->debug_info());
  }

  // Replace old node with the resolved new node in order list.
  func_graph->ReplaceInOrder(out_node, new_node);

  AnalysisEnginePtr eng = out_conf->engine();
  MS_EXCEPTION_IF_NULL(eng);
  AnfNodeConfigPtr fn_conf = eng->MakeConfig(new_node, out_conf->context(), out_conf->func_graph());
  return eng->ForwardConfig(out_conf, fn_conf);
}

EvalResultPtr GetEvaluatedValueForNameSpace(const AbstractBasePtrList &args_spec_list,
                                            const AnfNodeConfigPtr &out_conf) {
  // args_spec_list: same as StaticGetter
  constexpr size_t args_min_size = 2;
  if (args_spec_list.size() < args_min_size) {
    MS_LOG(EXCEPTION) << "Size of args_spec_list is less than 2";
  }
  MS_EXCEPTION_IF_NULL(out_conf);
  // An external type.
  constexpr auto data_index = 0;
  constexpr auto item_index = 1;
  auto data = args_spec_list[data_index];
  auto item = args_spec_list[item_index];
  MS_EXCEPTION_IF_NULL(data);
  MS_EXCEPTION_IF_NULL(item);
  auto data_value = data->BuildValue();
  MS_EXCEPTION_IF_NULL(data_value);
  if (!data_value->isa<parse::NameSpace>()) {
    if (data_value->isa<parse::ClassType>()) {
      auto class_val = dyn_cast_ptr<parse::ClassType>(data_value);
      const auto &class_name = class_val->name();
      auto item_value = item->BuildValue();
      MS_EXCEPTION(TypeError)
        << "Can not get attribute '" << item_value->ToString() << "' from " << class_name
        << " in graph mode. Try using jit_class to decorate the class? "
        << ".\nFor more details, please refer to "
        << "https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.jit_class.html \n";
    }
    MS_EXCEPTION(TypeError) << "Not supported to get attribute for " << data_value->ToString()
                            << "\nThe first argument should be a NameSpace, but got " << data->ToString();
  }

  auto item_value = item->BuildValue();
  MS_EXCEPTION_IF_NULL(item_value);
  auto data_type = data->BuildType();
  MS_EXCEPTION_IF_NULL(data_type);
  const auto &data_id_str = TypeIdToString(data_type->type_id());
  return GetEvaluatedValueForNameSpaceString(args_spec_list, data_value, out_conf, data_id_str);
}

EvalResultPtr GetEvaluatedValueForMsClassAttrOrMethod(const AbstractBasePtrList &args_spec_list,
                                                      const ValuePtr &data_value, const AnfNodeConfigPtr &out_conf) {
  constexpr size_t item_index = 1;
  auto item_args = args_spec_list[item_index];
  ValuePtr item_value = item_args->BuildValue();

  MS_EXCEPTION_IF_NULL(item_value);
  MS_EXCEPTION_IF_NULL(data_value);
  MS_EXCEPTION_IF_NULL(out_conf);
  // Get the name of item.
  if (!item_value->isa<StringImm>()) {
    MS_LOG(EXCEPTION) << "Expect a string, but got: " << item_value->ToString();
  }
  const auto &item_name = item_value->cast_ptr<StringImm>()->value();
  // Get ms_class object.
  if (!data_value->isa<parse::MsClassObject>()) {
    MS_LOG(EXCEPTION) << "Expect a ms_class object, but got " << data_value->ToString();
  }
  auto ms_class = data_value->cast_ptr<parse::MsClassObject>();
  MS_LOG(DEBUG) << "Resolve ms_class (" << ms_class->name() << ") with item " << item_name << ".";

  // Get the attr/method of ms_class object.
  auto out_node = out_conf->node();
  FuncGraphPtr func_graph = out_node->func_graph();
  // If the attribute is not found and the default is not set, AttributeError will be raised.
  auto new_node = parse::ResolveMsClassWithAttr(func_graph->manager(), ms_class->obj(), item_name, out_node);
  if (new_node == nullptr) {
    constexpr auto max_args_len = 3;
    bool has_default = (args_spec_list.size() == max_args_len);
    if (!has_default) {
      MS_EXCEPTION(AttributeError) << py::str(ms_class->obj()) << " object has no attribute: " << item_name << ".";
    }
    constexpr auto default_index = 3;
    auto out_cnode = out_node->cast_ptr<CNode>();
    MS_EXCEPTION_IF_NULL(out_cnode);
    new_node = out_cnode->inputs()[default_index];
  }

  func_graph->ReplaceInOrder(out_node, new_node);
  AnalysisEnginePtr eng = out_conf->engine();
  MS_EXCEPTION_IF_NULL(eng);
  AnfNodeConfigPtr fn_conf = eng->MakeConfig(new_node, out_conf->context(), out_conf->func_graph());
  return eng->ForwardConfig(out_conf, fn_conf);
}

EvalResultPtr GetEvaluatedValueForCellAttrOrMethod(const AbstractBasePtrList &args_spec_list,
                                                   const FuncGraphPtr &func_value, const AnfNodeConfigPtr &out_conf) {
  constexpr size_t item_index = 1;
  auto item_args = args_spec_list[item_index];
  ValuePtr item_value = item_args->BuildValue();

  MS_EXCEPTION_IF_NULL(item_value);
  MS_EXCEPTION_IF_NULL(func_value);
  if (!item_value->isa<StringImm>()) {
    MS_LOG(EXCEPTION) << "Expect a string, but got: " << item_value->ToString();
  }
  auto python_obj = func_value->python_obj();
  if (python_obj == nullptr) {
    return nullptr;
  }
  auto wrapper_obj = dyn_cast_ptr<parse::PyObjectWrapper>(python_obj);
  MS_EXCEPTION_IF_NULL(wrapper_obj);
  py::object real_python_obj = wrapper_obj->obj();
  const auto &py_obj_str = py::str(real_python_obj);
  MS_LOG(DEBUG) << "item_value: " << item_value->ToString() << ", func_value: " << func_value->ToString()
                << ", real_python_obj: " << py_obj_str;
  if (py::isinstance<Cell>(real_python_obj)) {
    py::module mod = python_adapter::GetPyModule(parse::PYTHON_MOD_PARSE_MODULE);
    py::object ns_obj =
      python_adapter::CallPyModFn(mod, parse::PYTHON_MOD_GET_MEMBER_NAMESPACE_SYMBOL, real_python_obj);
    auto ns = std::make_shared<parse::NameSpace>(parse::RESOLVE_NAMESPACE_NAME_CLASS_MEMBER, ns_obj);
    return GetEvaluatedValueForNameSpaceString(args_spec_list, ns, out_conf, py_obj_str);
  }
  return nullptr;
}

EvalResultPtr GetEvaluatedValueForBuiltinTypeAttrOrMethod(const AnalysisEnginePtr &engine,
                                                          const AbstractBasePtrList &args_spec_list,
                                                          const ConfigPtr &data_conf,
                                                          const AnfNodeConfigPtr &out_conf) {
  constexpr size_t data_index = 0;
  constexpr size_t item_index = 1;
  auto data_args = args_spec_list[data_index];
  auto item_args = args_spec_list[item_index];
  MS_EXCEPTION_IF_NULL(data_args);
  MS_EXCEPTION_IF_NULL(item_args);
  ValuePtr item_value = item_args->BuildValue();
  MS_EXCEPTION_IF_NULL(item_value);
  TypePtr data_type = data_args->BuildType();
  MS_EXCEPTION_IF_NULL(data_type);
  // The method maybe a Primitive or Composite
  if (!item_value->isa<StringImm>()) {
    MS_LOG(EXCEPTION) << "Expect a string, but got: " << item_value->ToString();
  }
  std::string item_name = item_value->cast_ptr<StringImm>()->value();
  REQUIRE_TYPE require_type = REQUIRE_TYPE::METHOD;
  Any require = pipeline::Resource::GetMethodPtr(data_type->type_id(), item_name);
  if (require.empty()) {
    require = pipeline::Resource::GetAttrPtr(data_type->type_id(), item_name);
    if (require.empty()) {
      constexpr auto max_args_len = 3;
      bool has_default = (args_spec_list.size() == max_args_len);
      if (!has_default) {
        MS_EXCEPTION(AttributeError) << data_type->ToString() << " object has no attribute: " << item_name;
      }
      auto out_node = out_conf->node();
      auto out_cnode = out_node->cast_ptr<CNode>();
      MS_EXCEPTION_IF_NULL(out_cnode);
      auto fg = out_cnode->func_graph();
      constexpr auto default_index = 3;
      auto default_node = out_cnode->inputs()[default_index];
      fg->ReplaceInOrder(out_node, default_node);
      auto eng = out_conf->engine();
      MS_EXCEPTION_IF_NULL(eng);
      auto fn_conf = eng->MakeConfig(default_node, out_conf->context(), out_conf->func_graph());
      return eng->ForwardConfig(out_conf, fn_conf);
    }
    require_type = REQUIRE_TYPE::ATTR;
  }

  ValuePtr converted_value = nullptr;
  if (require.is<std::string>()) {
    // composite registered in standard_method_map go to this branch
    converted_value = prim::GetPythonOps(require.cast<std::string>());
    MS_EXCEPTION_IF_NULL(converted_value);
    if (pipeline::GetJitLevel() == "O0" && converted_value->isa<FuncGraph>()) {
      UpdateDebugInfo(converted_value->cast<FuncGraphPtr>(), out_conf->node()->scope(), out_conf->node()->debug_info());
    }
    if (!converted_value->isa<Primitive>()) {
      AddToManager(engine, converted_value->cast<FuncGraphPtr>());
    }
  } else if (require.is<PrimitivePtr>()) {
    converted_value = require.cast<PrimitivePtr>();
  } else {
    MS_LOG(EXCEPTION) << "Expect to get string or PrimitivePtr from attr or method map, but got " << require.ToString();
  }
  return StaticGetterInferred(converted_value, data_conf, out_conf, require_type);
}

ValuePtr GetMsClassObject(const AbstractBasePtr &abs) {
  MS_EXCEPTION_IF_NULL(abs);
  if (!abs->isa<abstract::PartialAbstractClosure>()) {
    return nullptr;
  }
  auto partial_abs = abs->cast_ptr<abstract::PartialAbstractClosure>();
  auto fn = partial_abs->fn();
  if (!fn->isa<abstract::PrimitiveAbstractClosure>()) {
    return nullptr;
  }
  // Check if type is kObjectTypeClass.
  auto args = partial_abs->args();
  if (args.size() > 0) {
    constexpr size_t first_input_index = 0;
    auto first_arg = args[first_input_index];
    MS_EXCEPTION_IF_NULL(first_arg);
    auto type = first_arg->BuildType();
    MS_EXCEPTION_IF_NULL(type);
    if (type->type_id() == kObjectTypeClass) {
      return first_arg->BuildValue();
    }
  }
  return nullptr;
}

EvalResultPtr StaticGetter(const AnalysisEnginePtr &engine, const AbstractBasePtrList &args_spec_list,
                           const ConfigPtr &data_conf, const AnfNodeConfigPtr &out_conf) {
  // Inputs: namespace and its static function; or class and its member function

  constexpr size_t data_index = 0;
  constexpr size_t item_index = 1;
  auto data_args = args_spec_list[data_index];
  auto item_args = args_spec_list[item_index];
  MS_EXCEPTION_IF_NULL(data_args);
  MS_EXCEPTION_IF_NULL(item_args);
  MS_LOG(DEBUG) << "StaticGetter, data: " << data_args->ToString() << ", item: " << item_args->ToString();
  ValuePtr item_value = item_args->BuildValue();

  ScopePtr scope = kDefaultScope;
  if (out_conf != nullptr) {
    scope = out_conf->node()->scope();
  }
  ScopeGuard scope_guard(scope);
  MS_EXCEPTION_IF_NULL(item_value);
  if (item_value->isa<AnyValue>()) {
    MS_LOG(EXCEPTION) << "The value of the attribute could not be inferred: " << item_value->ToString();
  }

  if (data_args->isa<abstract::AbstractScalar>()) {
    ValuePtr data_value = data_args->BuildValue();
    if (data_value->isa<parse::InterpretedObject>()) {
      auto obj = ValueToPyData(data_value);
      auto type_str = python_adapter::CallPyFn(parse::PYTHON_MOD_PARSE_MODULE, parse::PYTHON_PARSE_GET_TYPE, obj);
      MS_EXCEPTION(TypeError) << "Do not support to get attribute from " << py::str(type_str) << " object "
                              << py::str(obj) << ".\nFor more details, please refer to "
                              << "https://mindspore.cn/docs/zh-CN/master/faq/network_compilation.html?highlight=do"
                              << "%20support%20get%20attribute%20from";
    }
  }

  constexpr auto max_args_size = 3;
  if (args_spec_list.size() == max_args_size) {
    constexpr size_t default_index = 2;
    auto default_args = args_spec_list[default_index];
    if (default_args->isa<abstract::AbstractScalar>()) {
      ValuePtr default_value = default_args->BuildValue();
      if (default_value->isa<parse::InterpretedObject>()) {
        auto obj = ValueToPyData(default_value);
        auto type_str = python_adapter::CallPyFn(parse::PYTHON_MOD_PARSE_MODULE, parse::PYTHON_PARSE_GET_TYPE, obj);
        MS_EXCEPTION(TypeError) << "For 'getattr', the third input 'default' can not be " << py::str(type_str)
                                << " object " << py::str(obj);
      }
    }
  }

  // Get attribute or method of class object decorated with 'jit_class'.
  auto class_value = GetMsClassObject(data_args);
  if (class_value != nullptr) {
    return GetEvaluatedValueForMsClassAttrOrMethod(args_spec_list, class_value, out_conf);
  }
  // Get attribute or method of nn.Cell object.
  auto data_func_graph = dyn_cast_ptr<FuncGraphAbstractClosure>(data_args);
  if (data_func_graph != nullptr) {
    auto res = GetEvaluatedValueForCellAttrOrMethod(args_spec_list, data_func_graph->func_graph(), out_conf);
    if (res != nullptr) {
      return res;
    }
  }
  // Try to search method map, if not found, the data_type should be External type.
  TypePtr data_type = data_args->BuildType();
  if (pipeline::Resource::IsTypeInBuiltInMap(data_type->type_id())) {
    return GetEvaluatedValueForBuiltinTypeAttrOrMethod(engine, args_spec_list, data_conf, out_conf);
  }
  return GetEvaluatedValueForNameSpace(args_spec_list, out_conf);
}
}  // namespace

EvalResultPtr MakeTupleEvaluator::EvalPrim(const AnalysisEnginePtr &, const AbstractBasePtrList &args_spec_list,
                                           const ConfigPtr &, const AnfNodeConfigPtr &out_conf) {
  std::shared_ptr<AnfNodeWeakPtrList> sequence_nodes = std::make_shared<AnfNodeWeakPtrList>();
  if (out_conf != nullptr) {  // 'out_conf' maybe nullptr in PyNative mode.
    if (args_spec_list.empty()) {
      MS_LOG(INFO) << "For MakeTuple, the inputs should not be empty. node: " << out_conf->node()->DebugString();
    }
    static const auto enable_eliminate_unused_element = (common::GetEnv("MS_DEV_ENABLE_DDE") != "0");
    if (enable_eliminate_unused_element) {
      auto flags = GetSequenceNodeElementsUseFlags(out_conf->node());
      if (flags == nullptr) {
        SetSequenceNodeElementsUseFlags(out_conf->node(), std::make_shared<std::vector<bool>>(args_spec_list.size()));
      }

      (void)sequence_nodes->emplace_back(AnfNodeWeakPtr(out_conf->node()));
    }
  }
  auto abs = std::make_shared<AbstractTuple>(args_spec_list, sequence_nodes);
  auto res = std::make_shared<EvalResult>(abs, std::make_shared<AttrValueMap>());
  evaluator_cache_mgr_->SetValue(args_spec_list, res);
  return res;
}

EvalResultPtr MakeListEvaluator::EvalPrim(const AnalysisEnginePtr &, const AbstractBasePtrList &args_spec_list,
                                          const ConfigPtr &, const AnfNodeConfigPtr &out_conf) {
  std::shared_ptr<AnfNodeWeakPtrList> sequence_nodes = std::make_shared<AnfNodeWeakPtrList>();
  if (out_conf != nullptr) {  // 'out_conf' maybe nullptr in PyNative mode.
    if (args_spec_list.empty()) {
      MS_LOG(INFO) << "For MakeList, the inputs should not be empty. node: " << out_conf->node()->DebugString();
    }
    static const auto enable_eliminate_unused_element = (common::GetEnv("MS_DEV_ENABLE_DDE") != "0");
    if (enable_eliminate_unused_element) {
      auto flags = GetSequenceNodeElementsUseFlags(out_conf->node());
      if (flags == nullptr) {
        SetSequenceNodeElementsUseFlags(out_conf->node(), std::make_shared<std::vector<bool>>(args_spec_list.size()));
      }

      (void)sequence_nodes->emplace_back(AnfNodeWeakPtr(out_conf->node()));
    }
  }
  auto abs = std::make_shared<AbstractList>(args_spec_list, sequence_nodes);
  auto res = std::make_shared<EvalResult>(abs, std::make_shared<AttrValueMap>());
  evaluator_cache_mgr_->SetValue(args_spec_list, res);
  return res;
}

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
    auto node_conf = dyn_cast_ptr<AnfNodeConfig>(args_conf_list[0]);
    MS_EXCEPTION_IF_NULL(node_conf);
    const auto &eval_result = node_conf->ObtainEvalResult();
    MS_EXCEPTION_IF_NULL(eval_result);
    AbstractBasePtr x = eval_result->abstract();
    x = SensitivityTransform(x);
    SymbolicKeyInstancePtr key = std::make_shared<SymbolicKeyInstance>(node_conf->node(), x);
    AbstractScalarPtr abs_scalar = std::make_shared<AbstractScalar>(key, std::make_shared<SymbolicKeyType>());
    return std::make_shared<EvalResult>(abs_scalar, std::make_shared<AttrValueMap>());
  }
};

static AnfNodePtr FindParameterNodeByString(const FuncGraphManagerPtr &manager, const std::string &name) {
  MS_EXCEPTION_IF_NULL(manager);
  auto root_g_set = manager->roots();
  if (root_g_set.size() != 1) {
    return nullptr;
  }
  const FuncGraphPtr &root_g = root_g_set.back();
  for (auto &param_node : root_g->parameters()) {
    auto param = param_node->cast<ParameterPtr>();
    if (param != nullptr && param->name() == name) {
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
    auto node_conf = dyn_cast_ptr<AnfNodeConfig>(args_conf_list[0]);
    if (node_conf == nullptr) {
      MS_LOG(ERROR) << "Conf should be AnfNodeConfig";
      return nullptr;
    }
    const auto &eval_result = node_conf->ObtainEvalResult();
    MS_EXCEPTION_IF_NULL(eval_result);
    AbstractBasePtr abs = eval_result->abstract();
    MS_EXCEPTION_IF_NULL(abs);
    auto ref_key_value = abstract::GetRefKeyValue(abs);
    if (ref_key_value == nullptr) {
      MS_LOG(ERROR) << "The first parameter of RefToEmbed should be Ref, but " << abs->ToString();
      return nullptr;
    }
    // Check if the input of RefEmbed is a weight parameter, if not, don't create the
    // specific SymbolicKey.
    // Notes: when different weight parameter have same type and shape passed as parameter to same funcgraph
    // which has RefToEmbed CNode, that funcgraph will not be specialized to different funcgraph, so the
    // RefToEmbed CNode in that funcgraph also should not be evaluated to specific SymbolicKey.
    // Only after that funcgrpah is inlined, the RefToEmbed CNode should be evaluated to specific SymbolicKey.
    bool ifEmbedIsWeight = false;
    if (node_conf->node() != nullptr && node_conf->node()->isa<Parameter>()) {
      auto param = node_conf->node()->cast_ptr<Parameter>();
      MS_EXCEPTION_IF_NULL(param);
      ifEmbedIsWeight = param->has_default();
    }
    auto refkey = ref_key_value->cast_ptr<StringImm>();
    if (refkey == nullptr || !ifEmbedIsWeight) {
      auto ret = std::make_shared<AbstractScalar>(type);
      return std::make_shared<EvalResult>(ret, std::make_shared<AttrValueMap>());
    }

    std::string name = refkey->value();
    MS_EXCEPTION_IF_NULL(node_conf->node());
    if (node_conf->node()->func_graph() == nullptr) {
      MS_LOG(EXCEPTION) << "Should not evaluate a ValueNode, node: " << node_conf->node()->DebugString();
    }
    const auto &manager = node_conf->node()->func_graph()->manager();
    auto node = FindParameterNodeByString(manager, name);
    if (node == nullptr) {
      MS_LOG(ERROR) << "RefToEmbed input can't find parameter \"" << name << "\" in graph.";
      return nullptr;
    }
    AbstractBasePtr x = SensitivityTransform(abs);
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
    constexpr auto args_min_size = 2;
    constexpr auto args_max_size = 3;
    constexpr auto attr_index = 1;
    auto ret_abstract = EvalUndeterminedArgs(args_spec_list);
    if (ret_abstract != nullptr) {
      MS_LOG(DEBUG) << "GetAttrEvaluator eval Undetermined";
      return ret_abstract;
    }
    // Inputs: data, item
    const auto args_size = args_spec_list.size();
    if (args_size != args_min_size && args_size != args_max_size) {
      MS_LOG(EXCEPTION) << "For Primitive GetAttr, the input size should be " << args_min_size << " or "
                        << args_max_size << ", but got size:" << args_size;
    }
    auto attr_abs = args_spec_list[attr_index];
    auto attr_abs_type = attr_abs->BuildType();
    MS_EXCEPTION_IF_NULL(attr_abs_type);
    auto type_id = attr_abs_type->type_id();
    if (type_id != TypeId::kObjectTypeString) {
      MS_EXCEPTION(TypeError) << "getattr(): attribute name must be string but got: " << TypeIdToString(type_id);
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
    evaluator_cache_mgr_->SetValue(args_spec_list, ret);
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
    constexpr auto resolve_args_size = 2;
    // Inputs: namespace, symbol
    if (args_spec_list.size() != resolve_args_size) {
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

bool IsContainUndetermined(const AbstractBasePtr &arg) {
  MS_EXCEPTION_IF_NULL(arg);
  if (arg->isa<AbstractSequence>()) {
    auto seq_arg = arg->cast_ptr<AbstractSequence>();
    return std::any_of(seq_arg->elements().begin(), seq_arg->elements().end(), IsContainUndetermined);
  }

  if (arg->isa<AbstractKeywordArg>()) {
    auto kw_arg = arg->cast_ptr<AbstractKeywordArg>();
    return IsContainUndetermined(kw_arg->get_arg());
  }

  return arg->isa<AbstractUndetermined>() && arg->IsBroaden();
}

class CreateInstanceEvaluator : public TransitionPrimEvaluator {
 public:
  CreateInstanceEvaluator() : TransitionPrimEvaluator("CreateInstanceEvaluator") {}
  ~CreateInstanceEvaluator() override = default;
  MS_DECLARE_PARENT(CreateInstanceEvaluator, TransitionPrimEvaluator);
  EvalResultPtr EvalPrim(const AnalysisEnginePtr &engine, const AbstractBasePtrList &args_spec_list, const ConfigPtr &,
                         const AnfNodeConfigPtr &out_conf) override {
    // Check the type parameter.
    if (args_spec_list.empty()) {
      MS_LOG(EXCEPTION) << "'args_spec_list' should not be empty";
    }
    constexpr size_t type_index = 0;
    auto arg_class_type = args_spec_list[type_index];
    MS_EXCEPTION_IF_NULL(arg_class_type);
    TypePtr type = arg_class_type->GetTypeTrack();
    MS_EXCEPTION_IF_NULL(type);
    if (type->type_id() != kMetaTypeTypeType && type->type_id() != kObjectTypeClass) {
      MS_LOG(EXCEPTION)
        << "CreateInstanceEvaluator require first parameter should be an object of TypeType or TypeClass, but got "
        << type->ToString();
    }

    ValuePtr value_track = arg_class_type->GetValueTrack();
    MS_EXCEPTION_IF_NULL(value_track);
    auto type_obj = dyn_cast_ptr<parse::PyObjectWrapper>(value_track);
    if (type_obj == nullptr) {
      MS_LOG(EXCEPTION) << "Cast value failed, not PyObjectWrapper:" << value_track->ToString() << ".";
    }
    if (!type_obj->isa<parse::ClassType>() && !type_obj->isa<parse::MsClassObject>()) {
      MS_LOG(EXCEPTION)
        << "CreateInstanceEvaluator the type_obj should be an object of ClassType or MsClassObject, but got "
        << type_obj->ToString() << ".";
    }
    auto class_type = type_obj->obj();
    MS_LOG(DEBUG) << "Get class type: " << type_obj->ToString() << ".";

    // Get the create instance obj's parameters, `params` may contain tuple(args, kwargs).
    py::tuple params = GetParameters(args_spec_list);
    // Create class instance.
    auto obj = parse::data_converter::CreatePythonObject(class_type, params);
    if (py::isinstance<py::none>(obj)) {
      MS_LOG(EXCEPTION) << "Create python object `" << py::str(class_type)
                        << "` failed, only support to create \'Cell\', \'Primitive\' or "
                        << "user-defined Class decorated with \'jit_class\'.";
    }

    // Process the object.
    MS_EXCEPTION_IF_NULL(out_conf->node());
    TraceGuard guard(std::make_shared<TraceResolve>(out_conf->node()->debug_info()));
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
    evaluator_cache_mgr_->SetValue(args_spec_list, infer_result);
    return infer_result;
  }

  py::tuple GetParameters(const AbstractBasePtrList &args_spec_list) const {
    if (args_spec_list.empty()) {
      MS_LOG(EXCEPTION) << "Unexpected arguments num, the min arguments num must be 1, but got 0.";
    }
    // Exclude class type by minus 1;
    std::size_t params_size = args_spec_list.size() - 1;
    auto params = py::tuple(params_size);
    for (size_t i = 0; i < params_size; i++) {
      // Only support the Scalar parameters type. Bypass class type by offset with 1.
      auto arg = args_spec_list[i + 1];
      MS_EXCEPTION_IF_NULL(arg);
      if (IsContainUndetermined(arg)) {
        MS_EXCEPTION(TypeError) << "The " << i << "th initializing input to create instance for "
                                << args_spec_list[0]->BuildValue()->ToString()
                                << " should be a constant, but got: " << arg->ToString();
      }
      // Because the Tensor's AbstractTensor can't get value from GetValueTrack.
      ValuePtr param_value = arg->BuildValue();
      py::object param = ValueToPyData(param_value);
      params[i] = param;
    }
    return params;
  }
};

class CallInstanceEvaluator : public TransitionPrimEvaluator {
 public:
  CallInstanceEvaluator() : TransitionPrimEvaluator("CallInstanceEvaluator") {}
  ~CallInstanceEvaluator() override = default;
  MS_DECLARE_PARENT(CallInstanceEvaluator, TransitionPrimEvaluator);
  EvalResultPtr EvalPrim(const AnalysisEnginePtr &engine, const AbstractBasePtrList &args_spec_list, const ConfigPtr &,
                         const AnfNodeConfigPtr &out_conf) override {
    if (args_spec_list.empty()) {
      MS_LOG(EXCEPTION) << "args_spec_list should not be empty.";
    }
    constexpr size_t cls_index = 0;
    auto arg_cls = args_spec_list[cls_index];
    MS_EXCEPTION_IF_NULL(arg_cls);
    TypePtr type = arg_cls->GetTypeTrack();
    MS_EXCEPTION_IF_NULL(type);
    if (type->type_id() != kObjectTypeClass) {
      MS_LOG(EXCEPTION) << "CallInstanceEvaluator require first parameter should be an object of TypeClass, but got "
                        << type->ToString();
    }
    ValuePtr value_track = arg_cls->GetValueTrack();
    MS_EXCEPTION_IF_NULL(value_track);
    auto ms_class = dyn_cast_ptr<parse::MsClassObject>(value_track);
    if (ms_class == nullptr) {
      MS_LOG(EXCEPTION) << "CallInstanceEvaluator only supports MsClassObject.";
    }

    // Call class instance, net(x, y) -> net.__call__(x, y)
    py::object cls_obj = ms_class->obj();
    const std::string call_func = "__call__";
    if (!py::hasattr(cls_obj, common::SafeCStr(call_func))) {
      MS_LOG(EXCEPTION) << ms_class->name() << " has no " << call_func << " function, please check the code.";
    }
    py::object call_obj = py::getattr(cls_obj, common::SafeCStr(call_func));
    FuncGraphPtr call_func_graph = parse::ConvertToFuncGraph(call_obj);
    if (call_func_graph == nullptr) {
      MS_LOG(EXCEPTION) << "Parse python object " << call_func << " failed.";
    }
    FuncGraphManagerPtr manager = engine->func_graph_manager();
    manager->AddFuncGraph(call_func_graph);

    // Replace net with net.__call__
    AnfNodePtr old_node = out_conf->node();
    MS_EXCEPTION_IF_NULL(old_node);
    auto old_cnode = dyn_cast_ptr<CNode>(old_node);
    MS_EXCEPTION_IF_NULL(old_cnode);
    std::vector<AnfNodePtr> inputs = {NewValueNode(call_func_graph)};
    for (size_t i = 1; i < old_cnode->size(); i++) {
      (void)inputs.emplace_back(old_cnode->input(i));
    }
    FuncGraphPtr func_graph = out_conf->func_graph();
    auto new_cnode = func_graph->NewCNode(inputs);
    // Continue to eval new_cnode.
    AnfNodeConfigPtr fn_conf = engine->MakeConfig(new_cnode, out_conf->context(), out_conf->func_graph());
    return engine->ForwardConfig(out_conf, fn_conf);
  }
};

class PyInterpretEvaluator : public TransitionPrimEvaluator {
 public:
  PyInterpretEvaluator() : TransitionPrimEvaluator("PyInterpretEvaluator") {}
  ~PyInterpretEvaluator() override = default;
  MS_DECLARE_PARENT(PyInterpretEvaluator, TransitionPrimEvaluator);
  EvalResultPtr EvalPrim(const AnalysisEnginePtr &, const AbstractBasePtrList &args_spec_list, const ConfigPtr &,
                         const AnfNodeConfigPtr &out_conf) override {
    if (args_spec_list.empty()) {
      MS_LOG(ERROR) << "'args_spec_list' should not be empty";
    }

    auto current_interpret_node = out_conf->node();
    MS_EXCEPTION_IF_NULL(current_interpret_node);
    MS_LOG(DEBUG) << "The current interpret node: " << current_interpret_node->DebugString();
    // Get the type parameter.
    MS_EXCEPTION_IF_NULL(args_spec_list[0]);
    ValuePtr value_track = args_spec_list[0]->GetValueTrack();
    MS_EXCEPTION_IF_NULL(value_track);

    auto script_obj = dyn_cast_ptr<parse::Script>(value_track);
    if (script_obj == nullptr) {
      MS_LOG(EXCEPTION) << "Cast value failed, not PyObjectWrapper:" << value_track->ToString() << ".";
    }

    // Make global and local parameters.
    const std::string &script = script_obj->script();
    py::tuple params = MakeParameters(args_spec_list, script);

    // Call python script string.
    MS_LOG(DEBUG) << "Call script: " << script << ", params: " << py::str(params);
    auto obj = parse::data_converter::CallPythonScript(py::str(script), params);
    if (py::isinstance<py::none>(obj)) {
      AbstractBasePtr res = std::make_shared<abstract::AbstractNone>();
      auto infer_result = std::make_shared<EvalResult>(res, nullptr);
      evaluator_cache_mgr_->SetValue(args_spec_list, infer_result);
      return infer_result;
    }

    ValuePtr converted_val = nullptr;
    // converted_val could be a InterpretedObject.
    bool converted = parse::ConvertData(obj, &converted_val, true);
    if (!converted) {
      MS_LOG(EXCEPTION) << "Convert the python object failed";
    }
    MS_EXCEPTION_IF_NULL(converted_val);
    if (converted_val->isa<tensor::Tensor>() && HasConstArgAttr(obj)) {
      MS_LOG(WARNING) << "The tensor " << converted_val->ToString()
                      << " which is not used for network input argument should not be set const.";
    }
    AbstractBasePtr res = ToAbstract(converted_val, AnalysisContext::DummyContext(), out_conf);
    auto infer_result = std::make_shared<EvalResult>(res, std::make_shared<AttrValueMap>());
    evaluator_cache_mgr_->SetValue(args_spec_list, infer_result);
    return infer_result;
  }

  void CheckInterpretInput(const AbstractDictionaryPtr &abstract_dict, const std::string &script) const {
    // Check whether this node should be interpretive executed.
    MS_EXCEPTION_IF_NULL(abstract_dict);
    const auto &elements = abstract_dict->elements();
    if (elements.empty()) {
      return;
    }
    for (const auto &element : elements) {
      const auto &name = element.first;
      const auto &local_abs = element.second;
      const auto &local_abs_val = local_abs->BuildValue();
      MS_EXCEPTION_IF_NULL(local_abs_val);
      if (local_abs_val == kAnyValue) {
        MS_EXCEPTION(ValueError) << "When using JIT Fallback to handle script '" << script
                                 << "', the inputs should be constant, but found variable '" << name
                                 << "' to be nonconstant.";
      }
      if (local_abs->isa<abstract::AbstractTensor>()) {
        MS_LOG(WARNING) << "When using JIT Fallback to handle script '" << script << "', found variable '" << name
                        << "' to be a tensor.";
      }
    }
  }

  void AddGlobalPythonFunction(const AbstractDictionaryPtr &global_dict, py::object *global_params_dict) const {
    MS_EXCEPTION_IF_NULL(global_dict);
    MS_EXCEPTION_IF_NULL(global_params_dict);
    const auto &global_dict_elements = global_dict->elements();
    for (const auto &element : global_dict_elements) {
      const auto &element_name = element.first;
      const auto &element_abs = element.second;
      if (element_abs->isa<abstract::FuncGraphAbstractClosure>()) {
        auto element_abs_fn = element_abs->cast_ptr<abstract::FuncGraphAbstractClosure>();
        auto fg = element_abs_fn->func_graph();
        MS_EXCEPTION_IF_NULL(fg);
        auto wrapper_obj = fg->python_obj();
        if (wrapper_obj != nullptr && wrapper_obj->isa<parse::PyObjectWrapper>()) {
          auto fn_py_obj = wrapper_obj->cast_ptr<parse::PyObjectWrapper>()->obj();
          (*global_params_dict)[py::str(element_name)] = fn_py_obj;
          MS_LOG(DEBUG) << "Found global python function object for " << element_name << ", add it to global dict.";
        }
      }
    }
    return;
  }

  py::tuple MakeParameters(const AbstractBasePtrList &args_spec_list, const std::string &script) const {
    constexpr int params_size = 3;
    if (params_size != args_spec_list.size()) {
      MS_LOG(EXCEPTION) << "Unexpected params_size: " << params_size
                        << ", not equal to arguments.size:" << args_spec_list.size();
    }
    // The first argument is script string, ignore it.
    auto params = py::tuple(params_size - 1);

    // Make the global parameters.
    auto global_dict = dyn_cast<AbstractDictionary>(args_spec_list[1]);  // Global parameters dict.
    MS_EXCEPTION_IF_NULL(global_dict);
    auto filtered_global_dict = FilterParameters(global_dict);
    MS_LOG(DEBUG) << "arg_1, global_dict: " << global_dict->ToString()
                  << ", filtered_global_dict: " << filtered_global_dict->ToString();
    ValuePtr global_dict_value = filtered_global_dict->BuildValue();
    py::object global_params_dict = ValueToPyData(global_dict_value);
    MS_LOG(DEBUG) << "arg_1, python global_params_dict: " << global_dict_value->ToString() << " -> "
                  << py::str(global_params_dict);

    // Add global python function to global_params_dict.
    AddGlobalPythonFunction(global_dict, &global_params_dict);
    params[0] = global_params_dict;

    // Make the local parameters.
    constexpr size_t local_index = 2;
    auto local_dict = dyn_cast<AbstractDictionary>(args_spec_list[local_index]);  // Local parameters dict.
    MS_EXCEPTION_IF_NULL(local_dict);
    auto filtered_local_dict = FilterParameters(local_dict);
    MS_LOG(DEBUG) << "arg_2, local_dict: " << local_dict->ToString()
                  << ", filtered_local_dict: " << filtered_local_dict->ToString();
    ValuePtr local_dict_value = filtered_local_dict->BuildValue();
    py::dict local_params_dict = ReCheckLocalDict(filtered_local_dict);
    MS_LOG(DEBUG) << "arg_2, python local_params_dict: " << local_dict_value->ToString() << " -> "
                  << py::str(local_params_dict);
    params[1] = local_params_dict;

    CheckInterpretInput(filtered_local_dict, script);

    return params;
  }

  py::dict ReCheckLocalDict(const AbstractDictionaryPtr &filtered_local_dict) const {
    const auto &keys_values = filtered_local_dict->elements();
    py::dict local_params_dict;
    for (auto &key_value : keys_values) {
      ValuePtr element_value = key_value.second->BuildValue();
      MS_EXCEPTION_IF_NULL(element_value);
      auto py_data = ValueToPyData(element_value);
      local_params_dict[py::str(key_value.first)] = py_data;
    }
    return local_params_dict;
  }

  AbstractDictionaryPtr FilterParameters(const AbstractDictionaryPtr &abstract_dict) const {
    MS_EXCEPTION_IF_NULL(abstract_dict);
    std::vector<AbstractAttribute> kv;
    const auto &keys_values = abstract_dict->elements();
    // Filter out the element of Function type.
    (void)std::copy_if(keys_values.cbegin(), keys_values.cend(), std::back_inserter(kv),
                       [](const AbstractAttribute &item) {
                         MS_EXCEPTION_IF_NULL(item.second);
                         return (!item.second->isa<abstract::AbstractFunction>());
                       });
    return std::make_shared<AbstractDictionary>(kv);
  }

  bool HasConstArgAttr(const py::object &obj) const {
    constexpr char const_arg_attr[] = "const_arg";
    return py::hasattr(obj, const_arg_attr) && py::cast<bool>(py::getattr(obj, const_arg_attr));
  }
};

class PartialEvaluator : public Evaluator {
 public:
  PartialEvaluator() : Evaluator("PartialEvaluator") {}
  ~PartialEvaluator() override = default;
  EvalResultPtr Run(AnalysisEnginePtr engine, const ConfigPtrList &args_conf_list,
                    const AnfNodeConfigPtr &out_conf) override {
    if (args_conf_list.size() == 0) {
      MS_LOG(EXCEPTION) << "Args size should be greater than 0";
    }

    MS_EXCEPTION_IF_NULL(out_conf);
    MS_EXCEPTION_IF_NULL(out_conf->node());
    MS_EXCEPTION_IF_NULL(args_conf_list[0]);
    const auto &arg0_eval_result = args_conf_list[0]->ObtainEvalResult();
    MS_EXCEPTION_IF_NULL(arg0_eval_result);
    auto arg0_value = arg0_eval_result->abstract();
    MS_EXCEPTION_IF_NULL(arg0_value);
    AbstractBasePtrList args_spec_list{arg0_value};
    // Func in hypermap(partial(Func, arg0), arg1, arg2) may become Poly Node.
    if (arg0_value->isa<AbstractError>()) {
      MS_EXCEPTION_IF_NULL(arg0_value->GetValueTrack());
      auto ret = std::make_shared<AbstractError>(arg0_value->GetValueTrack()->cast<ErrorValuePtr>(), out_conf->node());
      MS_LOG(DEBUG) << "AbstractError for node: " << out_conf->node()->DebugString()
                    << " as func is: " << arg0_value->ToString();
      auto eval_result = std::make_shared<EvalResult>(ret, std::make_shared<AttrValueMap>());
      evaluator_cache_mgr_->SetValue(args_spec_list, eval_result);
      return eval_result;
    }
    auto func = CheckArg<AbstractFunction>("partial", args_spec_list, 0);
    // Sometimes, node[0] in out_conf becomes phi0;
    if (func->isa<PrimitiveAbstractClosure>()) {
      auto prim_func = dyn_cast_ptr<PrimitiveAbstractClosure>(func);
      MS_EXCEPTION_IF_NULL(prim_func);
      MS_EXCEPTION_IF_NULL(prim_func->prim());
      if (prim_func->prim()->isa<prim::DoSignaturePrimitive>()) {
        auto do_signature_prim = dyn_cast_ptr<prim::DoSignaturePrimitive>(prim_func->prim());
        return HandleDoSignature(engine, do_signature_prim->function(), out_conf);
      }
    }

    (void)std::transform(args_conf_list.begin() + 1, args_conf_list.end(), std::back_inserter(args_spec_list),
                         [](const ConfigPtr &config) -> AbstractBasePtr {
                           MS_EXCEPTION_IF_NULL(config);
                           const auto &eval_result = config->ObtainEvalResult();
                           MS_EXCEPTION_IF_NULL(eval_result);
                           return eval_result->abstract();
                         });
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
    evaluator_cache_mgr_->SetValue(args_spec_list, eval_result);
    return eval_result;
  }

  EvalResultPtr Eval(AnalysisEnginePtr, const AbstractBasePtrList &, const AnfNodeConfigPtr &) override {
    MS_LOG(EXCEPTION) << "Eval() should not be called, Run() method should be called";
  }

  EvalResultPtr HandleDoSignature(const AnalysisEnginePtr &engine, const ValuePtr &signature_value,
                                  const AnfNodeConfigPtr &out_conf) const {
    MS_EXCEPTION_IF_NULL(engine);
    MS_EXCEPTION_IF_NULL(out_conf);
    MS_EXCEPTION_IF_NULL(out_conf->node());
    auto cnode = out_conf->node()->cast_ptr<CNode>();
    if (cnode == nullptr) {
      MS_LOG(EXCEPTION) << "Cnode is nullptr";
    }

    ScopeGuard scope_guard(out_conf->node()->scope());
    TraceGuard trace_guard(std::make_shared<TraceDoSignature>(out_conf->node()->debug_info()));
    std::vector<AnfNodePtr> new_nodes_inputs = cnode->inputs();
    auto new_signature_value = std::make_shared<prim::DoSignatureMetaFuncGraph>("signature", signature_value);
    new_nodes_inputs[1] = NewValueNode(new_signature_value);
    FuncGraphPtr func_graph = cnode->func_graph();
    MS_EXCEPTION_IF_NULL(func_graph);
    CNodePtr new_cnode = func_graph->NewCNode(std::move(new_nodes_inputs));
    AnfNodeConfigPtr fn_conf = engine->MakeConfig(new_cnode, out_conf->context(), out_conf->func_graph());
    return engine->ForwardConfig(out_conf, fn_conf);
  }
};

class RaiseEvaluator : public TransitionPrimEvaluator {
 public:
  RaiseEvaluator() : TransitionPrimEvaluator("RaiseEvaluator") {}
  ~RaiseEvaluator() override = default;
  MS_DECLARE_PARENT(RaiseEvaluator, TransitionPrimEvaluator);
  EvalResultPtr EvalPrim(const AnalysisEnginePtr &, const AbstractBasePtrList &args_spec_list, const ConfigPtr &,
                         const AnfNodeConfigPtr &out_conf) override {
    auto node = out_conf->node();
    MS_EXCEPTION_IF_NULL(node);
    auto cur_graph = node->func_graph();
    MS_EXCEPTION_IF_NULL(cur_graph);
    if (cur_graph->is_tensor_condition_branch()) {
      MS_LOG(EXCEPTION) << "Currently only supports raise in constant scenarios. "
                        << "Tensor type data cannot exist in the conditional statement. "
                        << "Please check your conditions which raise node is located at: "
                        << trace::GetDebugInfo(node->debug_info());
    }
    if (args_spec_list.empty()) {
      // process raise
      MS_LOG(EXCEPTION) << "No active exception to reraise.";
    }

    std::string exception_type = GetExceptionType(args_spec_list[0]);
    auto iter = exception_types_map.find(exception_type);
    if (iter == exception_types_map.end()) {
      MS_LOG(EXCEPTION) << "Unsupported exception type: " << exception_type
                        << ". Raise only support some Python standard exception types: "
                        << SupportedExceptionsToString();
    }
    ExceptionType type = iter->second;
    if (args_spec_list.size() == 1) {
      // Process raise ValueError()
      MS_EXCEPTION(type);
    }
    std::string exception_string;
    // Processed in units of nodes. Raise ValueError(xxxx)
    size_t index_begin = 2;
    auto cnode = node->cast_ptr<CNode>();
    MS_EXCEPTION_IF_NULL(cnode);
    auto inputs = cnode->inputs();
    bool need_out_symbol = inputs.size() > 3;
    if (need_out_symbol) {
      exception_string += "(";
    }
    for (size_t index = index_begin; index < inputs.size(); ++index) {
      const auto input = inputs[index];
      auto input_abs = args_spec_list[index - 1];
      MS_EXCEPTION_IF_NULL(input_abs);
      bool need_symbol = CheckNeedSymbol(input, input_abs);
      if (need_symbol) {
        exception_string += "'";
      }
      exception_string += GetExceptionString(input_abs, input, node);
      if (need_symbol) {
        exception_string += "'";
      }
      if (index != inputs.size() - 1) {
        exception_string += ", ";
      }
    }
    if (need_out_symbol) {
      exception_string += ")";
    }
    MS_EXCEPTION(type) << exception_string;
  }

 private:
  // string need add quotation marks
  bool CheckNeedSymbol(const AnfNodePtr &, const AbstractBasePtr &abs) const {
    MS_EXCEPTION_IF_NULL(abs);
    bool need_symbol = false;
    if (abs->isa<abstract::AbstractScalar>()) {
      auto scalar = abs->cast_ptr<abstract::AbstractScalar>();
      MS_EXCEPTION_IF_NULL(scalar);
      auto scalar_value = scalar->BuildValue();
      MS_EXCEPTION_IF_NULL(scalar_value);
      if (scalar_value->isa<StringImm>()) {
        need_symbol = true;
      }
    } else if (abs->isa<abstract::AbstractSequence>()) {
      auto abs_list = abs->cast_ptr<abstract::AbstractSequence>();
      MS_EXCEPTION_IF_NULL(abs_list);
      const auto &elements = abs_list->elements();
      for (auto &element : elements) {
        MS_EXCEPTION_IF_NULL(element);
        if (element->isa<abstract::AbstractScalar>()) {
          auto scalar = element->cast_ptr<abstract::AbstractScalar>();
          auto scalar_value = scalar->BuildValue();
          if (scalar_value->isa<StringImm>()) {
            need_symbol = true;
            break;
          }
        }
      }
    }
    return need_symbol;
  }
  std::string GetExceptionString(const AbstractBasePtr &arg, const AnfNodePtr &input, const AnfNodePtr &node) {
    std::string exception_str;
    MS_EXCEPTION_IF_NULL(arg);
    if (arg->isa<abstract::AbstractTensor>()) {
      MS_LOG(EXCEPTION) << "Currently only supports raise in constant scenarios. "
                        << "Tensor type data cannot exist in the raise statement. "
                        << "Please check your raise statement which is located at: "
                        << trace::GetDebugInfo(node->debug_info());
    } else if (arg->isa<abstract::AbstractTuple>()) {
      return GetTupleString(arg, input, node);
    } else if (arg->isa<abstract::AbstractList>()) {
      return GetListString(arg, input, node);
    } else {
      // Process raise ValueError
      exception_str += GetScalarStringValue(arg, node);
    }
    return exception_str;
  }

  std::string GetTupleString(const AbstractBasePtr &arg, const AnfNodePtr &input, const AnfNodePtr &node) {
    MS_EXCEPTION_IF_NULL(arg);
    std::string exception_str;
    // Process raise ValueError("str")
    auto arg_tuple = arg->cast_ptr<abstract::AbstractTuple>();
    MS_EXCEPTION_IF_NULL(arg_tuple);
    const auto &arg_tuple_elements = arg_tuple->elements();
    if (arg_tuple_elements.size() == 0) {
      MS_LOG(EXCEPTION) << "The arg_tuple_elements can't be empty.";
    }
    if (arg_tuple_elements.size() > 1) {
      exception_str += "(";
    }
    for (size_t index = 0; index < arg_tuple_elements.size(); ++index) {
      auto &element = arg_tuple_elements[index];
      exception_str += GetExceptionString(element, input, node);
      if (index != arg_tuple_elements.size() - 1 && !IsPrimitiveCNode(input, prim::kPrimMakeTuple)) {
        exception_str += ", ";
      }
    }
    if (arg_tuple_elements.size() > 1) {
      exception_str += ")";
    }
    return exception_str;
  }

  std::string GetListString(const AbstractBasePtr &arg, const AnfNodePtr &input, const AnfNodePtr &node) {
    MS_EXCEPTION_IF_NULL(arg);
    std::string exception_str;
    // Process raise ValueError("str")
    auto arg_list = arg->cast_ptr<abstract::AbstractList>();
    MS_EXCEPTION_IF_NULL(arg_list);
    const auto &arg_list_elements = arg_list->elements();
    if (arg_list_elements.size() == 0) {
      MS_LOG(EXCEPTION) << "The arg_list_elements can't be empty.";
    }
    if (arg_list_elements.size() > 1) {
      exception_str += "[";
    }
    for (size_t index = 0; index < arg_list_elements.size(); ++index) {
      auto &element = arg_list_elements[index];
      exception_str += GetExceptionString(element, input, node);
      if (index != arg_list_elements.size() - 1 && !IsPrimitiveCNode(input, prim::kPrimMakeList)) {
        exception_str += ", ";
      }
    }
    if (arg_list_elements.size() > 1) {
      exception_str += "]";
    }
    return exception_str;
  }

  std::string GetExceptionType(const AbstractBasePtr &abs) const {
    MS_EXCEPTION_IF_NULL(abs);
    std::string str;
    if (abs->isa<abstract::AbstractScalar>()) {
      auto scalar = abs->cast_ptr<abstract::AbstractScalar>();
      MS_EXCEPTION_IF_NULL(scalar);
      auto scalar_value = scalar->BuildValue();
      MS_EXCEPTION_IF_NULL(scalar_value);
      if (scalar_value->isa<StringImm>()) {
        str = GetValue<std::string>(scalar_value);
      }
      return str;
    }
    MS_LOG(EXCEPTION) << "The abstract of exception type is not scalar: " << abs->ToString();
  }

  std::string GetScalarStringValue(const AbstractBasePtr &abs, const AnfNodePtr &node) const {
    MS_EXCEPTION_IF_NULL(abs);
    MS_EXCEPTION_IF_NULL(node);
    std::string str;
    if (abs->isa<abstract::AbstractScalar>()) {
      auto scalar = abs->cast<abstract::AbstractScalarPtr>();
      auto scalar_value = scalar->BuildValue();
      auto scalar_type = scalar->BuildType();
      if (scalar_type->isa<Float>()) {
        str = std::to_string(GetValue<float>(scalar_value));
      } else {
        str = scalar_value->ToString();
      }
      return str;
    }
    MS_LOG(DEBUG) << "The abstract is not scalar: " << abs->ToString();
    MS_LOG(EXCEPTION) << "Currently only supports raise in constant scenarios. "
                      << "Tensor type data cannot exist in the raise statement. "
                      << "Please check your raise statement which is located at: "
                      << trace::GetDebugInfo(node->debug_info());
  }
};

class WithEnterEvaluator : public TransitionPrimEvaluator {
 public:
  WithEnterEvaluator() : TransitionPrimEvaluator("WithEnterEvaluator") {}
  ~WithEnterEvaluator() override = default;
  MS_DECLARE_PARENT(WithEnterEvaluator, TransitionPrimEvaluator);
  EvalResultPtr EvalPrim(const AnalysisEnginePtr &engine, const AbstractBasePtrList &args_spec_list, const ConfigPtr &,
                         const AnfNodeConfigPtr &out_conf) override {
    auto node = out_conf->node()->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(node);
    auto cur_graph = node->func_graph();
    MS_EXCEPTION_IF_NULL(cur_graph);

    if (args_spec_list.size() != 1) {
      MS_LOG(EXCEPTION) << "The enter node has wrong input." << node->debug_info();
    }

    // Check class object
    auto partial_abs = args_spec_list[0]->cast<PartialAbstractClosurePtr>();
    MS_EXCEPTION_IF_NULL(partial_abs);
    if (!IsCallInstance(partial_abs)) {
      MS_LOG(EXCEPTION) << "The enter node has wrong input." << node->debug_info();
    }

    AbstractBasePtrList args = partial_abs->args();
    py::object cls_obj;
    ValuePtr value = nullptr;
    if (!args.empty()) {
      value = args[0]->BuildValue();
      MS_EXCEPTION_IF_NULL(value);
      auto value_obj = value->cast<parse::MsClassObjectPtr>();
      if (value_obj != nullptr) {
        cls_obj = value_obj->obj();
      }
    }
    const std::string call_func = "__enter__";
    if (!py::hasattr(cls_obj, common::SafeCStr(call_func))) {
      MS_EXCEPTION_IF_NULL(value);
      auto ms_class = dyn_cast_ptr<parse::MsClassObject>(value);
      MS_LOG(EXCEPTION) << ms_class->name() << " has no " << call_func << " function, please check the code.";
    }
    py::object call_obj = py::getattr(cls_obj, common::SafeCStr(call_func));
    FuncGraphPtr call_func_graph = parse::ConvertToFuncGraph(call_obj);
    if (call_func_graph == nullptr) {
      MS_LOG(EXCEPTION) << "Parse python object " << call_func << " failed.";
    }
    FuncGraphManagerPtr manager = engine->func_graph_manager();
    manager->AddFuncGraph(call_func_graph);

    std::vector<AnfNodePtr> enter_inputs{NewValueNode(call_func_graph)};
    //  __enter__(self)
    auto call_enter_node = cur_graph->NewCNodeInOrder(enter_inputs);
    // Continue to eval call_enter_node.
    AnfNodeConfigPtr fn_conf = engine->MakeConfig(call_enter_node, out_conf->context(), out_conf->func_graph());
    return engine->ForwardConfig(out_conf, fn_conf);
  }
};

class WithExitEvaluator : public TransitionPrimEvaluator {
 public:
  WithExitEvaluator() : TransitionPrimEvaluator("WithExitEvaluator") {}
  ~WithExitEvaluator() override = default;
  MS_DECLARE_PARENT(WithExitEvaluator, TransitionPrimEvaluator);
  EvalResultPtr EvalPrim(const AnalysisEnginePtr &engine, const AbstractBasePtrList &args_spec_list, const ConfigPtr &,
                         const AnfNodeConfigPtr &out_conf) override {
    auto node = out_conf->node()->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(node);
    auto cur_graph = node->func_graph();
    MS_EXCEPTION_IF_NULL(cur_graph);

    if (args_spec_list.size() != 1) {
      MS_LOG(EXCEPTION) << "The exit node has wrong input." << node->debug_info();
    }

    // Check class object
    auto partial_abs = args_spec_list[0]->cast<PartialAbstractClosurePtr>();
    MS_EXCEPTION_IF_NULL(partial_abs);
    if (!IsCallInstance(partial_abs)) {
      MS_LOG(EXCEPTION) << "The exit node has wrong input." << node->debug_info();
    }

    AbstractBasePtrList args = partial_abs->args();
    py::object cls_obj;
    ValuePtr value = nullptr;
    if (!args.empty()) {
      value = args[0]->BuildValue();
      MS_EXCEPTION_IF_NULL(value);
      auto value_obj = value->cast<parse::MsClassObjectPtr>();
      if (value_obj != nullptr) {
        cls_obj = value_obj->obj();
      }
    }
    const std::string call_func = "__exit__";
    if (!py::hasattr(cls_obj, common::SafeCStr(call_func))) {
      MS_EXCEPTION_IF_NULL(value);
      auto ms_class = dyn_cast_ptr<parse::MsClassObject>(value);
      MS_EXCEPTION_IF_NULL(ms_class);
      MS_LOG(EXCEPTION) << ms_class->name() << " has no " << call_func << " function, please check the code.";
    }
    py::object call_obj = py::getattr(cls_obj, common::SafeCStr(call_func));
    FuncGraphPtr call_func_graph = parse::ConvertToFuncGraph(call_obj);
    if (call_func_graph == nullptr) {
      MS_LOG(EXCEPTION) << "Parse python object " << call_func << " failed.";
    }
    FuncGraphManagerPtr manager = engine->func_graph_manager();
    manager->AddFuncGraph(call_func_graph);

    std::vector<AnfNodePtr> exit_inputs{NewValueNode(call_func_graph)};
    constexpr size_t arg_size = 3;
    //  __exit__(self, type, value, trace)
    for (size_t i = 0; i < arg_size; ++i) {
      (void)exit_inputs.emplace_back(NewValueNode(kNone));
    }
    auto call_exit_node = cur_graph->NewCNodeInOrder(exit_inputs);
    // Continue to eval call_exit_node.
    AnfNodeConfigPtr fn_conf = engine->MakeConfig(call_exit_node, out_conf->context(), out_conf->func_graph());
    return engine->ForwardConfig(out_conf, fn_conf);
  }
};

class JoinedStrEvaluator : public TransitionPrimEvaluator {
 public:
  JoinedStrEvaluator() : TransitionPrimEvaluator("JoinedStrEvaluator") {}
  ~JoinedStrEvaluator() override = default;
  MS_DECLARE_PARENT(JoinedStrEvaluator, TransitionPrimEvaluator);
  EvalResultPtr EvalPrim(const AnalysisEnginePtr &, const AbstractBasePtrList &args_spec_list, const ConfigPtr &,
                         const AnfNodeConfigPtr &out_conf) override {
    auto node = out_conf->node()->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(node);
    auto cur_graph = node->func_graph();
    MS_EXCEPTION_IF_NULL(cur_graph);
    bool exist_tensor = std::any_of(args_spec_list.begin(), args_spec_list.end(), [](const AbstractBasePtr &arg) {
      auto arg_value = arg->BuildValue();
      MS_EXCEPTION_IF_NULL(arg_value);
      return arg_value->isa<AnyValue>();
    });
    AnfNodePtr new_node = nullptr;
    if (exist_tensor) {
      // This is a variable scenario, and the specific value cannot be obtained in the static analysis stage.
      auto cnode = node->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(cnode);
      auto cnode_inputs = cnode->inputs();
      std::vector<AnfNodePtr> new_inputs{NewValueNode(prim::kPrimMakeTuple)};
      (void)new_inputs.insert(new_inputs.end(), cnode_inputs.begin() + 1, cnode_inputs.end());
      new_node = cur_graph->NewCNode(new_inputs);
    } else {
      std::string ret;
      for (const auto &arg : args_spec_list) {
        auto arg_value = arg->BuildValue();
        MS_EXCEPTION_IF_NULL(arg_value);
        ret += arg_value->ToString();
      }
      new_node = NewValueNode(ret);
    }

    cur_graph->ReplaceInOrder(node, new_node);
    AnalysisEnginePtr eng = out_conf->engine();
    MS_EXCEPTION_IF_NULL(eng);
    AnfNodeConfigPtr fn_conf = eng->MakeConfig(new_node, out_conf->context(), out_conf->func_graph());
    return eng->ForwardConfig(out_conf, fn_conf);
  }
};

struct PrimitiveImplInferValue {
  PrimitiveImpl impl_;        // implement function of primitive
  bool eval_value_;           // whether evaluate value
  TypePtr specify_out_type_;  // whether specify return type
  bool in_white_list_;        // true if this Primitive in white list, else false.
};

using PrimitiveToImplMap = mindspore::HashMap<PrimitivePtr, PrimitiveImplInferValue, PrimitiveHasher, PrimitiveEqual>;
PrimitiveToImplMap &GetUniformPrimitiveToImplMap() {
  using R = PrimitiveToImplMap::mapped_type;
  static PrimitiveToImplMap uniform_prim_implement_map{
    {prim::kPrimScalarAdd, R{prim::ScalarAdd, true, nullptr, true}},
    {prim::kPrimScalarSub, R{prim::ScalarSub, true, nullptr, true}},
    {prim::kPrimScalarMul, R{prim::ScalarMul, true, nullptr, true}},
    {prim::kPrimScalarDiv, R{prim::ScalarDiv, true, nullptr, true}},
    {prim::kPrimScalarMod, R{prim::ScalarMod, true, nullptr, true}},
    {prim::kPrimScalarPow, R{prim::ScalarPow, true, nullptr, true}},
    {prim::kPrimScalarFloordiv, R{prim::ScalarFloordiv, true, nullptr, true}},
    {prim::kPrimScalarUadd, R{prim::ScalarUAdd, true, nullptr, true}},
    {prim::kPrimScalarUsub, R{prim::ScalarUSub, true, nullptr, true}},
    {prim::kPrimScalarLog, R{prim::ScalarLog, true, nullptr, true}},
    {prim::kPrimBitAnd, R{prim::BitAnd, true, nullptr, true}},
    {prim::kPrimBitOr, R{prim::BitOr, true, nullptr, true}},
    {prim::kPrimBitXor, R{prim::BitXor, true, nullptr, true}},
    {prim::kPrimBitLeftShift, R{prim::BitLeftShift, true, nullptr, true}},
    {prim::kPrimBitRightShift, R{prim::BitRightShift, true, nullptr, true}},
    {prim::kPrimScalarEq, R{prim::ScalarEq, true, std::make_shared<Bool>(), true}},
    {prim::kPrimScalarLt, R{prim::ScalarLt, true, std::make_shared<Bool>(), true}},
    {prim::kPrimScalarGt, R{prim::ScalarGt, true, std::make_shared<Bool>(), true}},
    {prim::kPrimScalarNe, R{prim::ScalarNe, true, std::make_shared<Bool>(), true}},
    {prim::kPrimScalarLe, R{prim::ScalarLe, true, std::make_shared<Bool>(), true}},
    {prim::kPrimScalarGe, R{prim::ScalarGe, true, std::make_shared<Bool>(), true}},
    {prim::kPrimBoolNot, R{prim::BoolNot, true, std::make_shared<Bool>(), true}},
    {prim::kPrimBoolAnd, R{prim::BoolAnd, true, std::make_shared<Bool>(), true}},
    {prim::kPrimBoolEq, R{prim::BoolEq, true, std::make_shared<Bool>(), true}},
    {prim::kPrimBoolOr, R{prim::BoolOr, true, std::make_shared<Bool>(), true}},
    {prim::kPrimStringConcat, R{prim::StringConcat, true, nullptr, true}},
    {prim::kPrimStringEq, R{prim::StringEq, true, std::make_shared<Bool>(), true}},
    {prim::kPrimStringLt, R{prim::StringLt, true, std::make_shared<Bool>(), true}},
    {prim::kPrimStringGt, R{prim::StringGt, true, std::make_shared<Bool>(), true}},
    {prim::kPrimStringLe, R{prim::StringLe, true, std::make_shared<Bool>(), true}},
    {prim::kPrimStringGe, R{prim::StringGe, true, std::make_shared<Bool>(), true}},
    {prim::kPrimStringNot, R{prim::StringNot, true, std::make_shared<Bool>(), true}},
    {prim::kPrimStringIn, R{prim::StringIn, true, std::make_shared<Bool>(), true}},
  };
  return uniform_prim_implement_map;
}

PrimEvaluatorMap PrimEvaluatorConstructors = PrimEvaluatorMap();
std::mutex PrimEvaluatorConstructorMutex;

void InitPrimEvaluatorConstructors() {
  PrimEvaluatorMap &constructor = PrimEvaluatorConstructors;

  for (const auto &iter : GetPrimitiveToEvalImplMap()) {
    constructor[iter.first] = InitStandardPrimEvaluator(iter.first, iter.second);
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
  constructor[prim::kPrimCallInstance] = std::make_shared<CallInstanceEvaluator>();
  constructor[prim::kPrimPartial] = std::make_shared<PartialEvaluator>();
  constructor[prim::kPrimPyInterpret] = std::make_shared<PyInterpretEvaluator>();
  constructor[prim::kPrimMakeTuple] = std::make_shared<MakeTupleEvaluator>();
  constructor[prim::kPrimMakeList] = std::make_shared<MakeListEvaluator>();
  constructor[prim::kPrimRaise] = std::make_shared<RaiseEvaluator>();
  constructor[prim::kPrimWithEnter] = std::make_shared<WithEnterEvaluator>();
  constructor[prim::kPrimWithExit] = std::make_shared<WithExitEvaluator>();
  constructor[prim::kPrimJoinedStr] = std::make_shared<JoinedStrEvaluator>();
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
    return iter->second.IsInWhileList();
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
  auto x_tuple = dyn_cast_ptr<AbstractTuple>(x);
  auto model_tuple = dyn_cast_ptr<Tuple>(model);

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
  auto x_tensor = dyn_cast_ptr<AbstractTensor>(x);
  auto model_tensor = dyn_cast_ptr<TensorType>(model);

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
  auto x_list = dyn_cast_ptr<AbstractList>(x);
  auto model_list = dyn_cast_ptr<List>(model);

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

inline bool IsSubtypeScalar(const AbstractBasePtr x, const TypePtr model) {
  MS_EXCEPTION_IF_NULL(x);
  MS_EXCEPTION_IF_NULL(model);
  if (dyn_cast_ptr<AbstractScalar>(x) == nullptr) {
    return false;
  }
  auto &x_type = x->GetTypeTrack();
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
    default:
      if (IsSubType(model, std::make_shared<Number>())) {
        return IsSubtypeScalar(x, model);
      }
      MS_LOG(EXCEPTION) << "Invalid model type: " << model->ToString() << ".";
  }
}
}  // namespace abstract
}  // namespace mindspore
