/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
 * Copyright 2019-2025 Huawei Technologies Co., Ltd
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

#include "frontend/jit/ps/static_analysis/prim.h"

#include <algorithm>
#include <limits>
#include <map>
#include <mutex>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "abstract/ops/primitive_infer_map.h"
#include "frontend/operator/cc_implementations.h"
#include "frontend/operator/composite/do_signature.h"
#include "frontend/operator/composite/unpack_call.h"
#include "frontend/operator/composite/functional_overload.h"
#include "frontend/operator/ops.h"
#include "include/frontend/operator/frontend_primitive_infer.h"
#include "frontend/parallel/ops_info/ops_utils.h"
#include "mindspore/ccsrc/frontend/operator/meta_dsl/common/meta_impl.h"
#include "include/backend/common/kernel_graph/anf_runtime_algorithm.h"
#include "include/utils/pipeline/fallback.h"
#include "include/utils/operator/primitive_utils.h"
#include "ir/anf.h"
#include "ir/cell.h"
#include "ir/dtype/tensor_type.h"
#include "primitive/array_op_name.h"
#include "primitive/comparison_ops.h"
#include "primitive/framework_ops.h"
#include "primitive/other_ops.h"
#include "primitive/sequence_ops.h"
#include "ops/infer_info/infer_info_utils.h"
#include "frontend/jit/ps/fallback.h"
#include "frontend/jit/ps/parse/data_converter.h"
#include "frontend/jit/ps/parse/parse_base.h"
#include "frontend/jit/ps/resource.h"
#include "frontend/jit/ps/static_analysis/evaluator.h"
#include "frontend/jit/ps/static_analysis/builtin_prim.h"
#include "frontend/jit/ps/static_analysis/prim_utils.h"
#include "frontend/jit/trace/trace_recorder.h"
#include "frontend/operator/ops_front_infer_function.h"
#include "include/frontend/jit/ps/static_analysis/py_infer_convert.h"
#include "utils/log_adapter.h"
#include "utils/symbolic.h"
#include "utils/trace_info.h"
#include "utils/check_convert_utils.h"
#include "primitive/auto_generate/gen_ops_primitive_b.h"
#include "primitive/auto_generate/gen_ops_primitive_c.h"
#include "primitive/auto_generate/gen_ops_primitive_d.h"
#include "primitive/auto_generate/gen_ops_primitive_e.h"
#include "primitive/auto_generate/gen_ops_primitive_f.h"
#include "primitive/auto_generate/gen_ops_primitive_g.h"
#include "primitive/auto_generate/gen_ops_primitive_i.h"
#include "primitive/auto_generate/gen_ops_primitive_l.h"
#include "primitive/auto_generate/gen_ops_primitive_m.h"
#include "primitive/auto_generate/gen_ops_primitive_p.h"
#include "primitive/auto_generate/gen_ops_primitive_r.h"
#include "primitive/auto_generate/gen_ops_primitive_s.h"
#include "primitive/auto_generate/gen_ops_primitive_t.h"
#include "primitive/auto_generate/gen_ops_primitive_w.h"
#include "ir/func_graph_flag.h"

namespace mindspore {
using ClassTypePtr = std::shared_ptr<parse::ClassType>;
namespace abstract {
using mindspore::parse::PyObjectWrapper;
constexpr auto kHasViewOutputFlag = "has_view_output";

bool NeedInfectViewOutputFlag(const AbstractBasePtrList &args) {
  for (const auto &arg : args) {
    if (arg->isa<abstract::AbstractRefTensor>()) {
      const auto ref = arg->cast<abstract::AbstractRefPtr>();
      if (ref->is_view_output()) {
        return true;
      }
    }
    auto has_view_output_flag = arg->user_data<bool>(kHasViewOutputFlag);
    if (has_view_output_flag != nullptr && *has_view_output_flag) {
      return true;
    }
  }
  return false;
}

namespace {
mindspore::HashSet<std::string> prims_to_skip_undetermined_infer{kMakeTupleOpName,  kMakeListOpName,   kSwitchOpName,
                                                                 kEnvironSetOpName, kEnvironGetOpName, kLoadOpName,
                                                                 kUpdateStateOpName};

// The Python primitives who visit tuple/list elements, but not consume all elements.
// Including:
// - Consume no element. For instance, MakeTuple.
// - Consume partial elements, not all. For instance, TupleGetItem.
// Map{"primitive name", {vector<int>:"index to transparent pass, -1 means all elements"}}
mindspore::HashMap<std::string, std::vector<int>> prims_transparent_pass_sequence{
  {kReturnOpName, std::vector({0})},       {kDependOpName, std::vector({0})},     {kidentityOpName, std::vector({0})},
  {kMakeTupleOpName, std::vector({-1})},   {kMakeListOpName, std::vector({-1})},  {kListAppendOpName, std::vector({0})},
  {kTupleGetItemOpName, std::vector({0})}, {kListGetItemOpName, std::vector({0})}};

bool CheckTypeIdAndShapeEqual(const AbstractBasePtr &left, const AbstractBasePtr &right) {
  // Regard Tensor and Parameter as the same type
  auto left_type = left->BuildType();
  MS_EXCEPTION_IF_NULL(left_type);
  auto right_type = right->BuildType();
  MS_EXCEPTION_IF_NULL(right_type);
  return left_type->type_id() == right_type->type_id() &&
         CheckAndConvertUtils::CheckAbstractShapeSame({left, right}) == 0;
}

CNodePtr GetInputsAfterUnpackCall(const CNodePtr &source_node, const AnalysisEnginePtr &engine,
                                  const AnfNodeConfigPtr &out_conf) {
  AnfNodePtrList new_inputs;
  auto fg = out_conf->node()->func_graph();
  for (size_t idx = 0; idx < source_node->size(); ++idx) {
    auto input = source_node->input(idx);
    AnfNodeConfigPtr config = engine->MakeConfig(input, out_conf->context(), out_conf->func_graph());
    MS_EXCEPTION_IF_NULL(config);
    const auto &eval_result = config->ObtainEvalResult();
    MS_EXCEPTION_IF_NULL(eval_result);
    auto input_abs = eval_result->abstract();
    if (input_abs->isa<AbstractDictionary>()) {
      const auto &dict_elems = input_abs->cast<AbstractDictionaryPtr>()->elements();
      for (const auto &elem : dict_elems) {
        auto key_node = NewValueNode(elem.first->BuildValue());
        auto value_node = fg->NewCNode({NewValueNode(prim::kPrimDictGetItem), input, key_node});
        (void)new_inputs.emplace_back(fg->NewCNode({NewValueNode(prim::kPrimMakeKeywordArg), key_node, value_node}));
      }
    } else if (input_abs->isa<AbstractTuple>()) {
      auto arg_tuple = input_abs->cast<AbstractTuplePtr>();
      for (size_t i = 0; i < arg_tuple->size(); ++i) {
        (void)new_inputs.emplace_back(
          fg->NewCNode({NewValueNode(prim::kPrimTupleGetItem), input, NewValueNode(SizeToLong(i))}));
      }
    } else if (input_abs->isa<AbstractList>()) {
      auto arg_list = input_abs->cast<AbstractListPtr>();
      for (size_t i = 0; i < arg_list->size(); ++i) {
        (void)new_inputs.emplace_back(
          fg->NewCNode({NewValueNode(prim::kPrimListGetItem), input, NewValueNode(SizeToLong(i))}));
      }
    } else {
      (void)new_inputs.emplace_back(input);
    }
  }
  return fg->NewCNodeInOrder(new_inputs);
}

AbstractBasePtr ConvertTensorToRef(const AbstractBasePtr &abs) {
  MS_EXCEPTION_IF_NULL(abs);
  if (abs->inplace_abstract() != nullptr) {
    return abs->inplace_abstract();
  }
  if (abs->isa<abstract::AbstractRefTensor>() || abs->isa<abstract::AbstractNone>()) {
    return abs;
  }
  auto tensor_abs = dyn_cast<abstract::AbstractTensor>(abs);
  MS_EXCEPTION_IF_NULL(tensor_abs);
  auto ref_abs = std::make_shared<abstract::AbstractRefTensor>(tensor_abs, std::make_shared<RefKey>("None"));
  std::stringstream ss;
  ss << ref_abs.get();
  static std::atomic<size_t> index = 0;
  // It is necessary to ensure the uniqueness of ref_key.
  ref_abs->set_ref_key_value(std::make_shared<RefKey>(ss.str() + std::to_string(index++)));
  MS_LOG(DEBUG) << "ref_abs: " << ref_abs->ToString();
  return ref_abs;
}

void AddRefKeyForSequenceArgs(const AbstractBasePtr &input_arg) {
  const auto &eles = input_arg->cast<AbstractSequencePtr>()->elements();
  AbstractBasePtrList new_eles;
  for (const auto &ele : eles) {
    AbstractBasePtr ele_ref_tensor = ele;
    if (!ele->isa<AbstractRefTensor>()) {
      ele_ref_tensor = ConvertTensorToRef(ele);
      if (ele_ref_tensor->isa<abstract::AbstractRefTensor>()) {
        ele_ref_tensor->cast<abstract::AbstractRefPtr>()->set_is_inplace(true);
        ele_ref_tensor = ele_ref_tensor->Broaden();
      }
      ele->set_inplace_abstract(ele_ref_tensor);
    } else {
      ele_ref_tensor->cast<abstract::AbstractRefPtr>()->set_is_inplace(true);
    }
    new_eles.push_back(ele_ref_tensor);
  }
  if (input_arg->isa<abstract::AbstractTuple>()) {
    auto new_seq = std::make_shared<abstract::AbstractTuple>(new_eles);
    input_arg->set_inplace_abstract(new_seq);
    return;
  }
  auto new_seq = std::make_shared<abstract::AbstractList>(new_eles);
  input_arg->set_inplace_abstract(new_seq);
}

AbstractBasePtr AddRefKeyForArgs(const AbstractBasePtr &output_abs, const AbstractBasePtrList &input_args,
                                 const std::vector<size_t> &rw_write_indexes,
                                 const std::vector<int64_t> &inplace_indexes) {
  // Convert input tensor to ref if this tensor is rw_write.
  for (const auto &index : rw_write_indexes) {
    if (input_args[index]->isa<AbstractSequence>()) {
      AddRefKeyForSequenceArgs(input_args[index]);
    } else if (!input_args[index]->isa<AbstractRefTensor>()) {
      auto ref_tensor = ConvertTensorToRef(input_args[index]);
      if (ref_tensor->isa<abstract::AbstractRefTensor>()) {
        ref_tensor->cast<abstract::AbstractRefPtr>()->set_is_inplace(true);
        ref_tensor = ref_tensor->Broaden();
      }
      input_args[index]->set_inplace_abstract(ref_tensor);
    } else {
      auto ref_tensor = input_args[index]->cast<abstract::AbstractRefPtr>();
      ref_tensor->set_is_inplace(true);
    }
  }
  if (inplace_indexes.size() == 0 || output_abs == nullptr) {
    return output_abs;
  }
  // If output is a tensor.
  if (output_abs->isa<AbstractTensor>()) {
    auto inplace_index = inplace_indexes[0];
    if (inplace_index != -1) {
      auto res_abs = input_args[inplace_index];
      MS_EXCEPTION_IF_NULL(res_abs);
      auto cur_res = res_abs->isa<AbstractRefTensor>() ? res_abs : res_abs->inplace_abstract();
      MS_EXCEPTION_IF_NULL(cur_res);
      cur_res = cur_res->Broaden();
      return cur_res;
    }
  }
  // If output is a tuple or a list of tensors.
  AbstractBasePtrList output_list;
  if (output_abs->isa<AbstractSequence>()) {
    // Get outputs after infer.
    const auto &output_args = output_abs->cast<AbstractSequencePtr>()->elements();
    if (inplace_indexes.size() > output_args.size()) {
      MS_LOG(EXCEPTION) << "The number of outputs must be greater than the inplace_indexes."
                        << " But got the number of outputs: " << output_args.size()
                        << ", the number of inplace_indexes: " << inplace_indexes.size();
    }
    for (size_t i = 0; i < inplace_indexes.size(); ++i) {
      auto inplace_index = inplace_indexes[i];
      if (inplace_index != -1) {
        auto outi_arg = input_args[inplace_index]->isa<AbstractRefTensor>()
                          ? input_args[inplace_index]
                          : input_args[inplace_index]->inplace_abstract();
        outi_arg = outi_arg->Broaden();
        (void)output_list.emplace_back(outi_arg);
      } else {
        (void)output_list.emplace_back(output_args[i]);
      }
    }
    std::copy(output_args.begin() + inplace_indexes.size(), output_args.end(), std::back_inserter(output_list));
    auto output_sequence_abs = dyn_cast_ptr<AbstractSequence>(output_abs);
    MS_EXCEPTION_IF_NULL(output_sequence_abs);
    output_sequence_abs->set_elements(output_list);
  }
  return output_abs;
}
}  // namespace

CNodePtr DoSignatureEvaluator::GenerateNewNodeBySignatures(const ValuePtr &func,
                                                           const AbstractBasePtrList &args_abs_list,
                                                           const AnalysisEnginePtr &engine,
                                                           const AnfNodeConfigPtr &out_conf) {
  if (out_conf->node() == nullptr || !out_conf->node()->isa<CNode>()) {
    MS_LOG(INTERNAL_EXCEPTION) << "Node of out_conf should be CNode";
  }
  auto out_cnode = dyn_cast<CNode>(out_conf->node());
  MS_EXCEPTION_IF_NULL(out_cnode);
  auto fg = out_cnode->func_graph();
  MS_EXCEPTION_IF_NULL(fg);
  if (out_cnode->size() == 0 || (out_cnode->size() - 1) != args_abs_list.size()) {
    MS_LOG(EXCEPTION) << "Op: " << func->ToString() << " args size should equal to inputs size minus 1, but args size "
                      << args_abs_list.size() << ", inputs size " << out_cnode->size();
  }

  // Handle primitive signatures.
  AnfNodePtrList args_inputs;
  (void)std::transform(out_cnode->weak_inputs().cbegin() + 1, out_cnode->weak_inputs().cend(),
                       std::back_inserter(args_inputs), [](const AnfNodeWeakPtr &weak_node) {
                         const auto &node = weak_node.lock();
                         MS_EXCEPTION_IF_NULL(node);
                         return node;
                       });
  auto op_inputs = prim::GetNewInputsBySignatures(fg, prim_->ToString(), func, args_abs_list, args_inputs, out_cnode);
  AnfNodePtrList new_inputs{NewValueNode(func)};
  (void)std::copy(op_inputs.begin(), op_inputs.end(), std::back_inserter(new_inputs));
  return fg->NewCNodeInOrder(new_inputs);
}

EvalResultPtr DoSignatureEvaluator::Run(AnalysisEnginePtr engine, const ConfigPtrList &args_conf_list,
                                        const AnfNodeConfigPtr &out_conf) {
  MS_EXCEPTION_IF_NULL(engine);
  MS_EXCEPTION_IF_NULL(out_conf);
  auto do_signature = prim_->cast_ptr<prim::DoSignaturePrimitive>();
  MS_EXCEPTION_IF_NULL(do_signature);
  auto &func = do_signature->function();
  MS_EXCEPTION_IF_NULL(func);

  AbstractBasePtrList args_abs_list;
  (void)std::transform(args_conf_list.begin(), args_conf_list.end(), std::back_inserter(args_abs_list),
                       [](const ConfigPtr &config) -> AbstractBasePtr {
                         MS_EXCEPTION_IF_NULL(config);
                         const auto &eval_result = config->ObtainEvalResult();
                         MS_EXCEPTION_IF_NULL(eval_result);
                         return eval_result->abstract();
                       });
  if (func->isa<Primitive>()) {
    auto do_signature_func = func->cast<PrimitivePtr>();
    if (do_signature_func->name() == kIsInstanceOpName) {
      // Handle for DDE.
      for (size_t i = 0; i < args_abs_list.size(); ++i) {
        MS_EXCEPTION_IF_NULL(args_abs_list[i]);
        if (args_abs_list[i]->isa<abstract::AbstractSequence>()) {
          MS_LOG(DEBUG) << "Primitive \'IsInstance\' is consuming tuple/list arguments[" << i
                        << "]: " << args_abs_list[i]->ToString();
          SetSequenceElementsUseFlagsRecursively(args_abs_list[i], true);
        }
      }
    }
    // Do undetermined infer firstly.
    if (prims_to_skip_undetermined_infer.find(do_signature_func->name()) == prims_to_skip_undetermined_infer.end()) {
      auto res_abstract = EvalUndeterminedArgs(args_abs_list);
      if (res_abstract != nullptr) {
        MS_LOG(DEBUG) << "DoSignatureEvaluator eval Undetermined for " << do_signature_func->name()
                      << ", res_abstract: " << res_abstract->ToString();
        return res_abstract;
      }
    }
  }

  CNodePtr new_cnode = nullptr;
  ScopePtr scope = out_conf->node()->scope();
  ScopeGuard scope_guard(scope);
  if (bound_node() != nullptr) {
    TraceGuard trace_guard(MakeTraceInfo<TraceDoSignature>(bound_node()->debug_info()));
    new_cnode = GenerateNewNodeBySignatures(func, args_abs_list, engine, out_conf);
  } else {
    new_cnode = GenerateNewNodeBySignatures(func, args_abs_list, engine, out_conf);
  }
  // Update new CNode info.
  auto out_cnode = dyn_cast<CNode>(out_conf->node());
  MS_EXCEPTION_IF_NULL(out_cnode);
  new_cnode->CloneCNodeInfo(out_cnode);

  // Do forward with old config and new config.
  AnfNodeConfigPtr new_conf = engine->MakeConfig(new_cnode, out_conf->context(), out_conf->func_graph());
  return engine->ForwardConfig(out_conf, new_conf);
}

static AbstractBasePtrList GetUnpackGraphSpecArgsList(const AbstractBasePtrList &args_abs_list, bool need_unpack) {
  if (!need_unpack) {
    // arg[0] is the func graph to unpack, ignore it
    AbstractBasePtrList specialize_args_before_unpack(args_abs_list.begin() + 1, args_abs_list.end());
    return specialize_args_before_unpack;
  }

  AbstractBasePtrList graph_specialize_args;
  // arg[0] is the func graph to unpack, ignore it
  for (size_t index = 1; index < args_abs_list.size(); index++) {
    MS_EXCEPTION_IF_NULL(args_abs_list[index]);
    if (args_abs_list[index]->isa<AbstractTuple>()) {
      const auto &arg_tuple = args_abs_list[index]->cast_ptr<AbstractTuple>();
      (void)std::transform(arg_tuple->elements().cbegin(), arg_tuple->elements().cend(),
                           std::back_inserter(graph_specialize_args), [](AbstractBasePtr abs) { return abs; });
    } else if (args_abs_list[index]->isa<AbstractDictionary>()) {
      auto arg_dict = args_abs_list[index]->cast_ptr<AbstractDictionary>();
      MS_EXCEPTION_IF_NULL(arg_dict);
      const auto &dict_elems = arg_dict->elements();
      (void)std::transform(dict_elems.cbegin(), dict_elems.cend(), std::back_inserter(graph_specialize_args),
                           [](const AbstractElementPair &item) {
                             MS_EXCEPTION_IF_NULL(item.first);
                             // Dict_elems's first element represents parameter names, which should be string type.
                             return std::make_shared<AbstractKeywordArg>(
                               GetValue<std::string>(item.first->BuildValue()), item.second);
                           });
    } else {
      MS_LOG(INTERNAL_EXCEPTION) << "UnpackGraph require args should be tuple or dict, but got "
                                 << args_abs_list[index]->ToString();
    }
  }
  return graph_specialize_args;
}

EvalResultPtr UnpackGraphEvaluator::Run(AnalysisEnginePtr engine, const ConfigPtrList &args_conf_list,
                                        const AnfNodeConfigPtr &out_conf) {
  MS_EXCEPTION_IF_NULL(engine);
  MS_EXCEPTION_IF_NULL(out_conf);
  MS_EXCEPTION_IF_NULL(out_conf->node());
  if (!out_conf->node()->isa<CNode>()) {
    MS_LOG(INTERNAL_EXCEPTION) << "Node of out_conf should be CNode";
  }
  MS_EXCEPTION_IF_NULL(prim_);
  auto unpack_graph = prim_->cast_ptr<prim::UnpackGraphPrimitive>();
  MS_EXCEPTION_IF_NULL(unpack_graph);
  auto out_cnode = out_conf->node()->cast_ptr<CNode>();
  MS_EXCEPTION_IF_NULL(out_cnode);
  if (out_cnode->empty() || (out_cnode->size() - 1) != args_conf_list.size()) {
    MS_LOG(EXCEPTION) << "UnpackGraphPrimitive"
                      << " args size should equal to inputs size minus 1, but args size " << args_conf_list.size()
                      << ", inputs size " << out_cnode->size();
  }
  AbstractBasePtrList args_abs_list;
  (void)std::transform(args_conf_list.begin(), args_conf_list.end(), std::back_inserter(args_abs_list),
                       [](const ConfigPtr &ref) -> AbstractBasePtr {
                         MS_EXCEPTION_IF_NULL(ref);
                         const auto &eval_result = ref->ObtainEvalResult();
                         MS_EXCEPTION_IF_NULL(eval_result);
                         return eval_result->abstract();
                       });
  // Get the forward graph
  if (args_abs_list.empty()) {
    MS_LOG(INTERNAL_EXCEPTION) << "args_abs_list can't be empty.";
  }
  MS_EXCEPTION_IF_NULL(args_abs_list[0]);
  auto fn = args_abs_list[0]->cast_ptr<AbstractFunction>();
  if (fn == nullptr) {
    MS_LOG(INTERNAL_EXCEPTION) << "UnpackGraphPrimitive arg0 must be AbstractFunction, but "
                               << args_abs_list[0]->ToString();
  }
  AbstractBasePtrList graph_specialize_args_without_sens;
  FuncGraphAbstractClosure *real_fn = nullptr;
  // If it's Partial closure, fetch the func graph from it.
  const auto &partial_fn_abs = fn->cast_ptr<PartialAbstractClosure>();
  if (partial_fn_abs != nullptr) {
    const auto &partial_fn = partial_fn_abs->fn();
    MS_EXCEPTION_IF_NULL(partial_fn);
    real_fn = partial_fn->cast_ptr<FuncGraphAbstractClosure>();
  } else {
    real_fn = fn->cast_ptr<FuncGraphAbstractClosure>();
  }
  MS_EXCEPTION_IF_NULL(real_fn);
  FuncGraphPtr forward_graph = real_fn->func_graph();
  MS_EXCEPTION_IF_NULL(forward_graph);
  AbstractBasePtrList graph_specialize_args =
    GetUnpackGraphSpecArgsList(args_abs_list, unpack_graph->need_unpack_args());
  if (unpack_graph->with_sens_in_args() && graph_specialize_args.empty()) {
    MS_EXCEPTION(ValueError) << "Grad with sens, but the sens is not provided.";
  }
  // If it's Partial closure, copy the arg list in advance.
  if (partial_fn_abs != nullptr) {
    (void)std::copy(partial_fn_abs->args().begin(), partial_fn_abs->args().end(),
                    std::back_inserter(graph_specialize_args_without_sens));
  }
  (void)std::transform(graph_specialize_args.begin(),
                       graph_specialize_args.end() - (unpack_graph->with_sens_in_args() ? 1 : 0),
                       std::back_inserter(graph_specialize_args_without_sens), [](AbstractBasePtr abs) { return abs; });
  MS_LOG(DEBUG) << "forward_graph: " << forward_graph->ToString()
                << ", graph_specialize_args_without_sens size: " << graph_specialize_args_without_sens.size();
  auto new_forward_graph = forward_graph->GenerateFuncGraph(graph_specialize_args_without_sens);
  MS_EXCEPTION_IF_NULL(engine->func_graph_manager());
  engine->func_graph_manager()->AddFuncGraph(new_forward_graph);
  ScopePtr scope = kDefaultScope;
  if (out_conf != nullptr) {
    scope = out_conf->node()->scope();
  }
  ScopeGuard scope_guard(scope);
  AnfNodePtr new_node = NewValueNode(new_forward_graph);
  AnfNodeConfigPtr fn_conf = engine->MakeConfig(new_node, out_conf->context(), out_conf->func_graph());
  return engine->ForwardConfig(out_conf, fn_conf);
}

AnfNodePtr MixedPrecisionCastHelper(const AnfNodePtr &source_node, const AbstractBasePtr &node_type,
                                    const AnfNodePtr &target_type, const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(node_type);
  MS_EXCEPTION_IF_NULL(func_graph);
  AnfNodePtr target_node = source_node;
  if (node_type->isa<AbstractTensor>()) {
    auto x = node_type->cast_ptr<AbstractTensor>();
    MS_EXCEPTION_IF_NULL(x->element());
    MS_EXCEPTION_IF_NULL(x->element()->BuildType());
    if (x->element()->BuildType()->isa<Float>() || x->element()->BuildType()->isa<BFloat>()) {
      auto cast = prim::GetPythonOps("_cast", "mindspore.ops.functional");
      MS_EXCEPTION_IF_NULL(cast);
      target_node = func_graph->NewCNodeAfter(source_node, {NewValueNode(cast), source_node, target_type});
    }
  } else if (node_type->isa<AbstractSequence>()) {
    auto x = node_type->cast_ptr<AbstractSequence>();
    auto &items = x->elements();
    std::vector<AnfNodePtr> nodes;
    (void)nodes.emplace_back(NewValueNode(prim::kPrimMakeTuple));
    int64_t idx = 0;
    for (const auto &item : items) {
      AnfNodePtr sequence_node = nullptr;
      if (node_type->isa<AbstractList>()) {
        sequence_node = func_graph->NewCNode({NewValueNode(prim::kPrimListGetItem), source_node, NewValueNode(idx)});
      } else {
        sequence_node = func_graph->NewCNode({NewValueNode(prim::kPrimTupleGetItem), source_node, NewValueNode(idx)});
      }
      AnfNodePtr node = MixedPrecisionCastHelper(sequence_node, item, target_type, func_graph);
      (void)nodes.emplace_back(node);
      ++idx;
    }
    target_node = func_graph->NewCNode(nodes);
  } else if (node_type->isa<AbstractDictionary>()) {
    auto x = node_type->cast_ptr<AbstractDictionary>();
    auto &items = x->elements();
    std::vector<AnfNodePtr> dict_key_nodes;
    std::vector<AnfNodePtr> dict_value_nodes;
    (void)dict_key_nodes.emplace_back(NewValueNode(prim::kPrimMakeTuple));
    (void)dict_value_nodes.emplace_back(NewValueNode(prim::kPrimMakeTuple));
    for (const auto &item : items) {
      MS_EXCEPTION_IF_NULL(item.first);
      auto key_value = item.first->BuildValue();
      MS_EXCEPTION_IF_NULL(key_value);
      AnfNodePtr dict_key_node = NewValueNode(key_value);
      AnfNodePtr dict_value_node =
        func_graph->NewCNode({NewValueNode(prim::kPrimDictGetItem), source_node, NewValueNode(key_value)});
      AnfNodePtr key_node = MixedPrecisionCastHelper(dict_key_node, item.first, target_type, func_graph);
      AnfNodePtr value_node = MixedPrecisionCastHelper(dict_value_node, item.second, target_type, func_graph);
      (void)dict_key_nodes.emplace_back(key_node);
      (void)dict_value_nodes.emplace_back(value_node);
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
  AbstractBasePtrList args_abs_list;
  MS_EXCEPTION_IF_NULL(out_conf);
  if (out_conf->node() == nullptr || !out_conf->node()->isa<CNode>()) {
    MS_LOG(INTERNAL_EXCEPTION) << "Node of out_conf should be CNode";
  }
  auto out_cnode = out_conf->node()->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(out_cnode);
  if (out_cnode->empty() || (out_cnode->size() - 1) != args_conf_list.size()) {
    MS_LOG(EXCEPTION) << "MixedPrecisionCast"
                      << " args size should equal to inputs size minus 1, but args size " << args_conf_list.size()
                      << ", inputs size " << out_cnode->size();
  }
  (void)std::transform(args_conf_list.begin(), args_conf_list.end(), std::back_inserter(args_abs_list),
                       [](const ConfigPtr &ref) -> AbstractBasePtr {
                         MS_EXCEPTION_IF_NULL(ref);
                         const auto &eval_result = ref->ObtainEvalResult();
                         MS_EXCEPTION_IF_NULL(eval_result);
                         return eval_result->abstract();
                       });

  ScopeGuard scope_guard(out_conf->node()->scope());
  TraceGuard trace_guard(MakeTraceInfo<TraceMixedPrecision>(out_conf->node()->debug_info()));

  FuncGraphPtr func_graph = out_cnode->func_graph();
  constexpr size_t source_node_index = 2;
  if (out_cnode->size() <= source_node_index) {
    MS_LOG(EXCEPTION) << "Input size: " << out_cnode->size() << " should bigger than 2.";
  }

  AnfNodePtr new_node =
    MixedPrecisionCastHelper(out_cnode->input(source_node_index), args_abs_list[1], out_cnode->input(1), func_graph);
  AnfNodeConfigPtr fn_conf = engine->MakeConfig(new_node, out_conf->context(), out_conf->func_graph());

  if (new_node->isa<CNode>()) {
    auto new_cnode = new_node->cast_ptr<CNode>();
    new_cnode->CloneCNodeInfo(out_cnode);
  }
  return engine->ForwardConfig(out_conf, fn_conf);
}

namespace {
void CheckTensorCondValid(const AbstractBasePtr &cond) {
  // Tensor condition must be one element or dynamic shape.
  auto base_shape = cond->BuildShape();
  MS_EXCEPTION_IF_NULL(base_shape);
  auto shape_ptr = base_shape->cast<ShapePtr>();
  MS_EXCEPTION_IF_NULL(shape_ptr);
  ShapeVector cond_shape = shape_ptr->shape();
  if (cond_shape.empty()) {
    return;
  }
  constexpr auto num_one = 1;
  for (size_t i = 0; i < cond_shape.size(); ++i) {
    if (cond_shape[i] != num_one && cond_shape[i] != Shape::kShapeDimAny && cond_shape[i] != Shape::kShapeRankAny) {
      MS_LOG(ERROR) << "The condition value of control flow can be a tensor with one element, "
                    << "but got tensor with shape " << base_shape->ToString();
      MS_EXCEPTION(ValueError) << "The truth value of an array with more than one element is ambiguous.";
    }
  }
}
}  // namespace

EvalResultPtr SwitchEvaluator::Run(AnalysisEnginePtr engine, const ConfigPtrList &args_conf_list,
                                   const AnfNodeConfigPtr &out_conf) {
  MS_EXCEPTION_IF_NULL(engine);
  AbstractBasePtrList args_abs_list;
  MS_EXCEPTION_IF_NULL(out_conf);
  MS_EXCEPTION_IF_NULL(out_conf->node());
  if (!out_conf->node()->isa<CNode>()) {
    MS_LOG(INTERNAL_EXCEPTION) << "Node of out_conf should be CNode";
  }
  auto out_cnode = out_conf->node()->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(out_cnode);
  if (out_cnode->empty() || (out_cnode->size() - 1) != args_conf_list.size()) {
    MS_LOG(EXCEPTION) << "For 'Switch',"
                      << " the args size should equal to inputs size minus 1, but args size " << args_conf_list.size()
                      << ", inputs size " << out_cnode->size();
  }

  // Inputs: condition, true branch, false branch
  constexpr size_t switch_input_size = 3;
  if (args_conf_list.size() != switch_input_size) {
    MS_LOG(EXCEPTION) << "Switch evaluator requires 3 parameters, while the input size is " << args_abs_list.size()
                      << ".";
  }

  auto eval_func = [](const ConfigPtr &conf) -> AbstractBasePtr {
    MS_EXCEPTION_IF_NULL(conf);
    const auto &eval_result = conf->ObtainEvalResult();
    MS_EXCEPTION_IF_NULL(eval_result);
    auto abs = eval_result->abstract();
    MS_EXCEPTION_IF_NULL(abs);
    return abs;
  };

  auto cond_abstract = eval_func(args_conf_list[0]);
  ValuePtr cond_value = cond_abstract->GetValueTrack();
  MS_EXCEPTION_IF_NULL(cond_value);
  // If the value of condition is ValueAny or the abstract of condition is AbstractTensor,
  // keeps both true and false branch.
  if (cond_value->isa<ValueAny>() || cond_abstract->isa<AbstractTensor>()) {
    if (cond_abstract->isa<AbstractTensor>()) {
      CheckTensorCondValid(cond_abstract);
    }
    auto true_branch = eval_func(args_conf_list[1]);
    // Need record two func_graph
    constexpr auto false_branch_index = 2;
    auto false_branch = eval_func(args_conf_list[false_branch_index]);
    SetVariableFlag(true_branch);
    SetVariableFlag(false_branch);
    auto res_abs = true_branch->Join(false_branch);
    auto eval_result = std::make_shared<EvalResult>(res_abs, std::make_shared<AttrValueMap>());
    return eval_result;
  }

  if (cond_value->isa<Scalar>()) {
    AbstractBasePtr res_abs = nullptr;
    if (cond_value->cast<ScalarPtr>()->IsOne()) {
      const auto &true_branch = eval_func(args_conf_list[1]);
      res_abs = true_branch;
    } else {
      constexpr auto false_branch_index = 2;
      auto false_branch = eval_func(args_conf_list[false_branch_index]);
      res_abs = false_branch;
    }
    auto eval_result = std::make_shared<EvalResult>(res_abs, std::make_shared<AttrValueMap>());
    return eval_result;
  }
  MS_LOG(EXCEPTION) << "Not support this condition value: " << cond_abstract->GetValueTrack()->ToString();
}

namespace {
// Join all types in args_type_list;
TypePtr TypeJoin(const TypePtrList &args_type_list) {
  if (args_type_list.empty()) {
    MS_LOG(INTERNAL_EXCEPTION) << "args_type_list is empty";
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
    if (!IsIdentidityOrSubclass(arg_type, predicate)) {
      MS_LOG(INTERNAL_EXCEPTION) << "The expected is " << predicate->ToString() << ", not " << arg_type->ToString();
    }
  }
  return TypeJoin(args_type_list);
}
}  // namespace

EvalResultPtr StandardPrimEvaluator::RunPyInferValue(const AnalysisEnginePtr &, const AbstractBasePtr &abs_base,
                                                     const AbstractBasePtrList &args) {
  auto prim_py = dyn_cast<PrimitivePy>(prim_);
  if (prim_py == nullptr) {
    MS_LOG(INTERNAL_EXCEPTION) << "The primitive with type 'kPrimTypePyCheck' should be a python primitive.";
  }
  // Call checking method 'infer_value' for python primitive
  MS_LOG(DEBUG) << "Begin input args checking for: " << prim_py->ToString();
  auto py_args = PreparePyInputs(args);
  py::tuple py_vals(py_args.size());
  MS_EXCEPTION_IF_NULL(prim_);
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
  MS_EXCEPTION_IF_NULL(abs_base);
  TypePtr dtype = abs_base->BuildType();
  bool converted = parse::ConvertData(py_ret, &converted_ret, false, dtype);
  if (!converted) {
    MS_LOG(INTERNAL_EXCEPTION) << "Convert data failed";
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
    MS_LOG(INTERNAL_EXCEPTION) << "The primitive with type 'kPrimTypePyCheck' should be a python primitive.";
  }
  // We should copy attributes before running check and infer,
  // since they may be changed during check and infer.
  auto input_attrs = prim_py->attrs();
  prim_py->BeginRecordAddAttr();
  auto py_args = PreparePyInputs(args);
  // Call checking method '__check__' for subclass of 'PrimitiveWithCheck'.
  prim_py->RunCheck(py_args);
  auto abs = eval_impl_.InferShapeAndType(nullptr, prim_py, args);
  MS_EXCEPTION_IF_NULL(abs);
  prim_py->EndRecordAddAttr();
  auto &added_attrs = prim_py->evaluate_added_attrs();
  eval_result = std::make_shared<EvalResult>(abs, std::make_shared<AttrValueMap>(added_attrs));
  if (py::hasattr(prim_py->GetPyObj(), PY_PRIM_METHOD_INFER_VALUE)) {
    // Call 'infer_value()' method if it is existed, for constant propagation.
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
    MS_LOG(INTERNAL_EXCEPTION) << "The primitive list should not be empty for " << prim->name();
  }
  // Ignore all arguments, no need checking if AbstractSequence.
  if (index_list[0] == -1) {
    return;
  }
  // Check the specific arguments index.
  for (size_t i = 0; i < args.size(); ++i) {
    MS_EXCEPTION_IF_NULL(args[i]);
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
    MS_EXCEPTION_IF_NULL(args[i]);
    if (args[i]->isa<abstract::AbstractSequence>()) {
      MS_EXCEPTION_IF_NULL(args[i]);
      MS_LOG(DEBUG) << "Primitive \'" << prim->name() << "\' is consuming tuple/list arguments[" << i
                    << "]: " << args[i]->ToString();
      SetSequenceElementsUseFlagsRecursively(args[i], true);
    }
  }
}
}  // namespace

PrimitiveFunctionEvaluator::PrimitiveFunctionEvaluator(const PrimitivePtr &prim_func)
    : TrivialPrimEvaluator("PrimitiveFunctionEvaluator"), prim_func_(prim_func) {
  frontend_func_impl_ = mindspore::ops::GetOpFrontendFuncImplPtr(prim_func->name());
  op_def_ = mindspore::ops::GetOpDef(prim_func->name());
}

void PrimitiveFunctionEvaluator::CheckArgsSizeAndType(const AbstractBasePtrList &abs_args) {
  auto op_args = op_def_->args_;
  // Ignore monad.
  AbstractBasePtrList real_abs_args;
  (void)std::copy_if(abs_args.cbegin(), abs_args.cend(), std::back_inserter(real_abs_args),
                     [](const AbstractBasePtr &abs) {
                       MS_EXCEPTION_IF_NULL(abs);
                       return !abs->isa<abstract::AbstractMonad>();
                     });
  // Check inputs number.
  if (op_args.size() != real_abs_args.size()) {
    MS_EXCEPTION(TypeError) << "For Operator[" << op_def_->name_ << "], the inputs number should be " << op_args.size()
                            << " but got " << real_abs_args.size() << ".";
  }

  // Check inputs type.
  for (size_t i = 0; i < op_args.size(); ++i) {
    if (HasAbstractType<AbstractUndetermined>(real_abs_args[i])) {
      continue;
    }
    if (!ValidateArgOptional(real_abs_args[i], op_args[i]) &&
        !abstract::ValidateArgsType(real_abs_args[i], op_args[i].arg_dtype_)) {
      std::vector<std::string> op_type_list;
      for (const auto &op_abs : real_abs_args) {
        (void)op_type_list.emplace_back(op_abs->BuildType()->ToString());
      }
      MS_INTERNAL_EXCEPTION(TypeError)
        << "For Operator[" << op_def_->name_ << "], " << op_args[i].arg_name_ << "'s type '"
        << real_abs_args[i]->BuildType()->ToString() << "' does not match expected type '"
        << prim::OpDTypeToString(op_args[i].arg_dtype_)
        << "'.\nThe reason may be: lack of definition of type cast, or incorrect type when creating the node.";
    }
  }
}

AbstractBasePtr UpdateViewOpsAbstract(const AbstractBasePtr &res, const AbstractBasePtrList &args) {
  MS_EXCEPTION_IF_NULL(res);
  if (!res->isa<abstract::AbstractTensor>() && !res->isa<abstract::AbstractTuple>()) {
    MS_LOG(EXCEPTION) << "The abstract of view operation is exception: " << res->ToString();
  }

  // Update the abstract of first input of view operation.
  auto arg0_tensor = dyn_cast<abstract::AbstractTensor>(args[0]);
  AbstractBasePtr new_input_arg = ConvertTensorToRef(arg0_tensor);
  if (arg0_tensor != nullptr && arg0_tensor->isa<abstract::AbstractRefTensor>()) {
    const auto ref = arg0_tensor->cast<abstract::AbstractRefPtr>();
    if (new_input_arg != nullptr && new_input_arg->isa<abstract::AbstractRefTensor>()) {
      // Keep the original ref_type.
      new_input_arg->cast<AbstractRefPtr>()->set_ref_tensor_type(ref->ref_tensor_type());
    }
  }
  if (new_input_arg != nullptr) {
    if (new_input_arg->isa<abstract::AbstractRefTensor>()) {
      // Added is_view_input ref_type.
      new_input_arg->cast<AbstractRefPtr>()->set_is_view_input(true);
    }
    args[0]->set_inplace_abstract(new_input_arg);
  }

  // Update the abstract of view operation.
  AbstractBasePtr new_res = res;
  if (res->isa<abstract::AbstractTensor>()) {
    // The output of the view operator shares the same address with the first input of the operator.
    new_res = ConvertTensorToRef(res);
    if (new_res->isa<abstract::AbstractRefTensor>()) {
      new_res->cast<abstract::AbstractRefPtr>()->set_is_view_output(true);
    }
  } else if (res->isa<abstract::AbstractTuple>()) {
    // Update the elements of output.
    AbstractBasePtrList output_list;
    const auto &res_args = res->cast<abstract::AbstractTuplePtr>()->elements();
    for (size_t i = 0; i < res_args.size(); ++i) {
      auto ele = res_args[i];
      MS_EXCEPTION_IF_NULL(ele);
      if (ele->isa<abstract::AbstractRefTensor>()) {
        (void)output_list.emplace_back(ele);
        continue;
      }
      if (!ele->isa<abstract::AbstractTensor>()) {
        MS_LOG(EXCEPTION) << "The abstract of view operation is exception: " << res->ToString();
      }
      auto ele_abs = dyn_cast<abstract::AbstractTensor>(ele);
      auto new_ele_abs = ConvertTensorToRef(ele_abs);
      if (new_ele_abs->isa<abstract::AbstractRefTensor>()) {
        new_ele_abs->cast<abstract::AbstractRefPtr>()->set_is_view_output(true);
      }
      (void)output_list.emplace_back(new_ele_abs);
      ele->set_inplace_abstract(new_ele_abs);
    }
    auto output_sequence_abs = dyn_cast_ptr<AbstractSequence>(res);
    MS_EXCEPTION_IF_NULL(output_sequence_abs);
    output_sequence_abs->set_elements(output_list);
    new_res = res;
  }
  MS_LOG(DEBUG) << "The new abstract of view operation is: " << new_res->ToString();
  return new_res;
}

AbstractBasePtr PrimitiveFunctionEvaluator::ProcessViewInplaceAbstract(const AbstractBasePtrList &args,
                                                                       const AbstractBasePtr &res) {
  if (graph_view_prim()) {
    auto ge_mode = AnfAlgo::IsBackendGe();
    if (ge_mode) {
      MS_LOG(WARNING) << "The view feature is not currently supported in GE mode. "
                      << "The code utilizes the View operator: " << prim_func_->ToString();
    } else {
      MS_LOG(DEBUG) << "View prim infer.";
      return UpdateViewOpsAbstract(res, args);
    }
  }
  const auto &rw_write_indexes = rw_write_input_indexes();
  const auto &inplace_indexes = inplace_input_indexes();
  return inplace_prim() ? AddRefKeyForArgs(res, args, rw_write_indexes, inplace_indexes) : res;
}

AbstractBasePtr PrimitiveFunctionEvaluator::CheckAndInfer(const AbstractBasePtrList &args) {
  if (op_def_ != nullptr) {
    MS_LOG(DEBUG) << "prim_func_: " << prim_func_->ToString();
    if (op_def_->func_impl_.GeneralInferRegistered()) {
      auto res = ops::DoGeneralInfer(prim_func_, args, frontend_func_impl_);
      return ProcessViewInplaceAbstract(args, res);
    } else {
      (void)op_def_->func_impl_.CheckValidation(prim_func_, args);
      if (frontend_func_impl_ != nullptr) {
        auto infer_result = frontend_func_impl_->InferAbstract(prim_func_, args);
        if (infer_result != nullptr) {
          return ProcessViewInplaceAbstract(args, infer_result);
        }
      }
      auto type = op_def_->func_impl_.InferType(prim_func_, args);
      auto shape = op_def_->func_impl_.InferShape(prim_func_, args);
      auto res = MakeAbstract(shape, type);
      return ProcessViewInplaceAbstract(args, res);
    }
  }
  MS_LOG(INTERNAL_EXCEPTION) << "Find infer function failed, primitive: " << prim_func_->ToString();
}

EvalResultPtr PrimitiveFunctionEvaluator::EvalPrim(const AnalysisEnginePtr &engine, const AbstractBasePtrList &args) {
  MS_EXCEPTION_IF_NULL(prim_func_);
  CheckArgsSizeAndType(args);
  // To check tuple/list operations with a white list of Python primitive.
  CheckSequenceArgumentForCppPrimitive(prim_func_, args);

  bool need_infer_value = std::all_of(args.begin(), args.end(), [](const AbstractBasePtr &abs) -> bool {
    MS_EXCEPTION_IF_NULL(abs);
    auto value = abs->BuildValue();
    return (value != nullptr && !value->isa<Monad>() && !value->isa<FuncGraph>());
  });

  AbstractBasePtr abs_base = nullptr;
  prim_func_->BeginRecordAddAttr();
  if (need_infer_value && frontend_func_impl_ != nullptr) {
    auto value = frontend_func_impl_->InferValue(prim_func_, args);
    if (value != nullptr && !value->ContainsValueAny()) {
      abs_base = value->ToAbstract();
      prim_func_->EndRecordAddAttr();
      auto added_attrs = prim_func_->evaluate_added_attrs();
      return std::make_shared<EvalResult>(abs_base, std::make_shared<AttrValueMap>(added_attrs));
    }
  }
  abs_base = CheckAndInfer(args);
  MS_EXCEPTION_IF_NULL(abs_base);
  bool need_infect_view_output_flag = NeedInfectViewOutputFlag(args);
  if (need_infect_view_output_flag) {
    abs_base->set_user_data<bool>(kHasViewOutputFlag, std::make_shared<bool>(true));
  }
  prim_func_->EndRecordAddAttr();
  const auto &added_attrs = prim_func_->evaluate_added_attrs();
  return std::make_shared<EvalResult>(abs_base, std::make_shared<AttrValueMap>(added_attrs));
}

EvalResultPtr StandardPrimEvaluator::EvalPrim(const AnalysisEnginePtr &engine, const AbstractBasePtrList &args) {
  // To check tuple/list operations with a white list of Python primitive.
  CheckSequenceArgumentForCppPrimitive(prim_, args);
  MS_EXCEPTION_IF_NULL(prim_);
  if (prims_to_skip_undetermined_infer.find(prim_->name()) == prims_to_skip_undetermined_infer.end()) {
    auto res_abstract = EvalUndeterminedArgs(args);
    if (res_abstract != nullptr) {
      MS_LOG(DEBUG) << "StandardPrimEvaluator eval Undetermined";
      return res_abstract;
    }
  }
  if (prim_->prim_type() == PrimType::kPrimTypePyCheck) {
    return EvalPyCheckPrim(engine, args);
  }
  bool need_infer_value = std::all_of(args.begin(), args.end(), [](const AbstractBasePtr &abs) -> bool {
    MS_EXCEPTION_IF_NULL(abs);
    auto value = abs->BuildValue();
    return (value != nullptr && !value->ContainsValueAny() && !value->isa<None>() && !value->isa<Monad>() &&
            !value->isa<FuncGraph>());
  });

  prim_->BeginRecordAddAttr();
  if (need_infer_value && eval_impl_.IsImplInferValue()) {
    auto value = eval_impl_.InferValue(prim_, args);
    if (value != nullptr) {
      auto abs_base = value->ToAbstract();
      prim_->EndRecordAddAttr();
      auto added_attrs = prim_->evaluate_added_attrs();
      return std::make_shared<EvalResult>(abs_base, std::make_shared<AttrValueMap>(added_attrs));
    }
  }
  auto output_abs = eval_impl_.InferShapeAndType(nullptr, prim_, args);
  const auto &rw_write_indexes = rw_write_input_indexes();
  const auto &inplace_indexes = inplace_input_indexes();
  auto abs_base = inplace_prim() ? AddRefKeyForArgs(output_abs, args, rw_write_indexes, inplace_indexes) : output_abs;
  MS_EXCEPTION_IF_NULL(abs_base);
  // Set output's kHasViewOutputFlag according to input args
  if (prim_->name() == kDependOpName) {
    if (NeedInfectViewOutputFlag({args[0]})) {
      abs_base->set_user_data<bool>(kHasViewOutputFlag, std::make_shared<bool>(true));
    }
  } else if (prim_->name() != kUpdateStateOpName && NeedInfectViewOutputFlag(args)) {
    abs_base->set_user_data<bool>(kHasViewOutputFlag, std::make_shared<bool>(true));
  }
  prim_->EndRecordAddAttr();
  const auto &added_attrs = prim_->evaluate_added_attrs();
  return std::make_shared<EvalResult>(abs_base, std::make_shared<AttrValueMap>(added_attrs));
}

EvalResultPtr PythonPrimEvaluator::EvalPrim(const AnalysisEnginePtr &engine, const AbstractBasePtrList &args) {
  // Consider all primitive implemented python infer() real use the tuple/list arguments.
  CheckSequenceArgumentForPythonPrimitive(prim_py_, args);
  // Ensure input arguments are evaluated.
  auto res_abstract = EvalUndeterminedArgs(args);
  if (res_abstract != nullptr) {
    MS_LOG(DEBUG) << "PythonPrimEvaluator eval Undetermined";
    return res_abstract;
  }
  MS_EXCEPTION_IF_NULL(prim_py_);
  auto forbid_reuse = prim_py_->HasAttr(GRAPH_FLAG_FORBID_REUSE_RESULT);
  if (!forbid_reuse) {
    // Try to get infer result from evaluator cache.
    EvalResultPtr eval_result = evaluator_cache_mgr_->GetValue(args);
    if (eval_result != nullptr) {
      MS_EXCEPTION_IF_NULL(eval_result->abstract());
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
  auto py_args = PreparePyInputs(args);
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
  auto res_abstract = EvalUndeterminedArgs(args);
  if (res_abstract != nullptr) {
    MS_LOG(DEBUG) << "UniformPrimEvaluator eval Undetermined";
    return res_abstract;
  }
  // if func_desc_.retval type is super class of parameter type, then make the retval type as parameter type.
  if (nargs_ != args.size()) {
    MS_LOG(INTERNAL_EXCEPTION) << "UniformPrimEvaluator expect " << nargs_ << " args, but got " << args.size()
                               << " inputs";
  }
  TypePtr res_value_type = return_value_type_;
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
      MS_LOG(INTERNAL_EXCEPTION) << "Expect scalar arguments for uniform primitives.";
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
      res_value_type = res;
    }
  }

  ValuePtr evaluated_value = RunImpl(value_list);
  MS_EXCEPTION_IF_NULL(evaluated_value);
  if (!(*evaluated_value == *kValueAny)) {
    res_value_type = evaluated_value->type();
  }
  // for comparison primitives , return type shall have be specified to be bool.
  if (specify_out_type_ != nullptr) {
    res_value_type = specify_out_type_;
  }

  AbstractScalarPtr abs_base = std::make_shared<AbstractScalar>(evaluated_value, res_value_type);
  return std::make_shared<EvalResult>(abs_base, std::make_shared<AttrValueMap>());
}

ValuePtr UniformPrimEvaluator::RunImpl(const ValuePtrList &args) const {
  if (!eval_value_) {
    return kValueAny;
  } else {
    if (std::any_of(args.begin(), args.end(), [](const ValuePtr &arg) {
          MS_EXCEPTION_IF_NULL(arg);
          return arg->ContainsValueAny();
        })) {
      return kValueAny;
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
  EvaluatorPtr uniform_primitive_evaluator =
    std::make_shared<UniformPrimEvaluator>(primitive, prim_impl, eval_value, specify_out_type);
  return uniform_primitive_evaluator;
}

inline void AddToManager(const AnalysisEnginePtr &engine, const FuncGraphPtr func_graph) {
  MS_EXCEPTION_IF_NULL(engine);
  FuncGraphManagerPtr manager = engine->func_graph_manager();
  MS_EXCEPTION_IF_NULL(manager);
  manager->AddFuncGraph(func_graph);
}

enum class REQUIRE_TYPE { ATTR, METHOD };

bool IsPyExecuteData(const AbstractBasePtr &data_abstract) {
  MS_EXCEPTION_IF_NULL(data_abstract);
  return data_abstract->isa<abstract::AbstractAny>();
}

void CheckObjAttrValid(const TypePtr &data_type, const std::string &item_name, const AbstractBasePtr &data_args) {
  MS_EXCEPTION_IF_NULL(data_type);
  MS_EXCEPTION_IF_NULL(data_args);
  // Check if the obj's attr is invalid or decoratored by @jit_forbidden_register
  std::string data_type_str = TypeIdLabel(NormalizeTypeId(data_type->type_id()));
  if (data_args->isa<AbstractRefTensor>()) {
    data_type_str = "Parameter";
  } else if (data_args->isa<AbstractNamedTuple>()) {
    data_type_str = "NamedTuple";
  }
  py::module mod1 = python_adapter::GetPyModule(parse::PYTHON_MOD_PARSE_MODULE);
  py::object obj_define = python_adapter::CallPyModFn(mod1, parse::PYTHON_MOD_GET_OBJ_DEFINED, data_type_str);
  if (py::isinstance<py::none>(obj_define)) {
    return;
  }
  py::module mod2 = python_adapter::GetPyModule(parse::PYTHON_MOD_MODULE);
  auto is_jit_forbidden_method =
    python_adapter::CallPyModFn(mod2, parse::PYTHON_MOD_IS_INVALID_METHOD, obj_define, data_type_str, item_name);
  if (py::cast<bool>(is_jit_forbidden_method) || data_args->isa<AbstractRefTensor>()) {
    MS_LOG(EXCEPTION) << "Failed to compile in GRAPH_MODE because the '" << data_type_str << "' object's method '"
                      << item_name << "' is not supported in 'construct' or function with @jit decorator. "
                      << "Try to use the '" << data_type_str << "." << item_name << "' externally "
                      << "such as initialized in the method '__init__' before assigning"
                      << ".\nFor more details, please refer to "
                      << "https://www.mindspore.cn/tutorials/zh-CN/master/compile/static_graph.html \n";
  }
}

AnfNodePtr SetTypeForGetAttr(const AnfNodePtr &getattr_node, const AbstractBasePtr &value_abs) {
  // Set setattr's abstract as getattr's abstract.
  if (value_abs != nullptr &&
      (value_abs->isa<abstract::AbstractTensor>() || value_abs->isa<abstract::AbstractScalar>())) {
    auto type = value_abs->BuildType();
    auto shape = value_abs->BuildShape();
    fallback::SetRealType<AnfNode, Type>(getattr_node, type);
    fallback::SetRealShape<AnfNode, abstract::BaseShape>(getattr_node, shape);
  }
  return getattr_node;
}

EvalResultPtr InterpretGetAttrNode(const AbstractBasePtrList &args_abs_list, const AnfNodeConfigPtr &out_conf) {
  MS_EXCEPTION_IF_NULL(out_conf);
  auto out_node = out_conf->node();
  MS_EXCEPTION_IF_NULL(out_node);
  const auto cnode = dyn_cast<CNode>(out_node);
  MS_EXCEPTION_IF_NULL(cnode);
  auto fg = cnode->func_graph();

  auto data_args = args_abs_list[0];
  MS_EXCEPTION_IF_NULL(data_args);
  // Not check if the data is from PyExecute CNode.
  // Do not check the validity of the attribute in the variable scenario.
  if (!IsPyExecuteData(data_args) && !raiseutils::HasVariableCondition(fg)) {
    TypePtr data_type = data_args->BuildType();
    MS_EXCEPTION_IF_NULL(data_type);
    auto item_args = args_abs_list[1];
    MS_EXCEPTION_IF_NULL(item_args);
    ValuePtr item_value = item_args->BuildValue();
    auto item_str = item_value->cast_ptr<StringImm>();
    MS_EXCEPTION_IF_NULL(item_str);
    std::string item_name = item_str->value();
    CheckObjAttrValid(data_type, item_name, data_args);
  }

  const auto &debug_info = trace::GetSourceCodeDebugInfo(out_node->debug_info());
  if (debug_info == nullptr || debug_info->location() == nullptr) {
    MS_LOG(WARNING) << "Location info is null, node: " << out_node->DebugString(AnfNode::DebugStringLevel::kLevel2);
    return nullptr;
  }
  const auto expr = debug_info->location()->expr_src();
  if (expr.empty()) {
    MS_LOG(WARNING) << "Location's expr is empty, node: " << out_node->DebugString(AnfNode::DebugStringLevel::kLevel2);
  }

  constexpr auto item_index = 1;
  auto item_arg = args_abs_list.at(item_index);
  MS_EXCEPTION_IF_NULL(item_arg);
  auto attr_name = GetValue<string>(item_arg->BuildValue());
  AnfNodePtr getattr_node;
  auto obj_change = cnode->user_data<bool>(fallback::kObjectAttrChange);
  if (obj_change != nullptr && *obj_change) {
    // The object is changed by setattr node, directly convert it to PyExecute node.
    getattr_node = fallback::ConvertCNodeToPyExecuteForPrim(cnode, "getattr");
    constexpr auto args_size = 3;
    if (args_abs_list.size() == args_size) {  // Has setattr node as input.
      auto getattr_cnode = getattr_node->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(getattr_cnode);
      getattr_cnode->add_input(cnode->input(args_size));
      constexpr auto value_index = 2;
      getattr_node = SetTypeForGetAttr(getattr_cnode, args_abs_list[value_index]);
    }
  } else {
    getattr_node = fallback::ConvertGetAttrNodeToPyInterpret(fg, cnode, attr_name);
  }
  MS_LOG(DEBUG) << "Convert: " << cnode->DebugString() << " -> " << getattr_node->DebugString();
  auto eng = out_conf->engine();
  MS_EXCEPTION_IF_NULL(eng);
  auto fn_conf = eng->MakeConfig(getattr_node, out_conf->context(), out_conf->func_graph());
  return eng->ForwardConfig(out_conf, fn_conf);
}

EvalResultPtr InterpretTensorSetDataNode(const AbstractBasePtrList &args_abs_list, const AnfNodeConfigPtr &out_conf) {
  MS_EXCEPTION_IF_NULL(out_conf);
  auto out_node = out_conf->node();
  MS_EXCEPTION_IF_NULL(out_node);
  const auto cnode = out_node->cast_ptr<CNode>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto fg = cnode->func_graph();
  MS_EXCEPTION_IF_NULL(fg);
  auto owner_abs = args_abs_list[0];
  MS_EXCEPTION_IF_NULL(owner_abs);
  if (!owner_abs->isa<abstract::AbstractRefTensor>() && !owner_abs->isa<abstract::AbstractTensor>()) {
    MS_LOG(DEBUG) << "Expect AbstractRefTensor or AbstractTensor, but got: " << owner_abs->ToString();
    return nullptr;
  }
  ValuePtr attr_str_value = args_abs_list[1]->BuildValue();
  MS_EXCEPTION_IF_NULL(attr_str_value);
  if (!attr_str_value->isa<StringImm>()) {
    MS_LOG(INTERNAL_EXCEPTION) << "Expect a string, but got: " << attr_str_value->ToString();
  }
  auto attr_str = attr_str_value->cast<StringImmPtr>();
  MS_EXCEPTION_IF_NULL(attr_str);
  if (attr_str->value() != "data") {
    MS_LOG(DEBUG) << "Expect set data, but got: " << attr_str->value();
    return nullptr;
  }
  constexpr size_t min_set_data_input_size = 4;
  size_t set_data_input_size = cnode->inputs().size();
  if (set_data_input_size < min_set_data_input_size) {
    MS_LOG(DEBUG) << "For Primitive SetData, got input size: " << set_data_input_size;
    return nullptr;
  }
  const auto set_data_node =
    fg->NewCNodeInOrder({NewValueNode(prim::kPrimSetData), cnode->input(kIndex1), cnode->input(kIndex3)});
  MS_LOG(DEBUG) << "set_data_node: " << set_data_node->DebugString(AnfNode::DebugStringLevel::kLevel2);
  auto eng = out_conf->engine();
  MS_EXCEPTION_IF_NULL(eng);
  auto fn_conf = eng->MakeConfig(set_data_node, out_conf->context(), out_conf->func_graph());
  return eng->ForwardConfig(out_conf, fn_conf);
}

EvalResultPtr InterpretSetAttrNode(const AbstractBasePtrList &args_abs_list, const AnfNodeConfigPtr &out_conf) {
  MS_EXCEPTION_IF_NULL(out_conf);
  auto out_node = out_conf->node();
  MS_EXCEPTION_IF_NULL(out_node);
  const auto cnode = dyn_cast<CNode>(out_node);
  MS_EXCEPTION_IF_NULL(cnode);
  auto fg = cnode->func_graph();
  MS_EXCEPTION_IF_NULL(fg);
  auto owner_abs = args_abs_list[0];
  MS_EXCEPTION_IF_NULL(owner_abs);
  if (owner_abs->isa<abstract::AbstractRefTensor>()) {
    MS_EXCEPTION(ValueError) << "Do not support to set attribute for a parameter.";
  }
  auto owner_value = owner_abs->BuildValue();
  auto owner_node = cnode->input(1);
  MS_EXCEPTION_IF_NULL(owner_value);
  MS_LOG(DEBUG) << "node: " << out_conf->node()->DebugString(AnfNode::DebugStringLevel::kLevel2)
                << ", owner_value: " << owner_value->ToString();
  if (owner_value->isa<parse::InterpretedObject>()) {
    const auto &interpreted_value = dyn_cast<parse::InterpretedObject>(owner_value);
    const auto &key = interpreted_value->name();
    owner_node = fallback::ConvertPyObjectToPyExecute(fg, key, interpreted_value->obj(), owner_node, true);
  }

  ValuePtr attr_str_value = args_abs_list[1]->BuildValue();
  MS_EXCEPTION_IF_NULL(attr_str_value);
  if (!attr_str_value->isa<StringImm>()) {
    MS_LOG(EXCEPTION) << "Expect a string, but got: " << attr_str_value->ToString();
  }
  auto attr_str = attr_str_value->cast<StringImmPtr>();
  MS_EXCEPTION_IF_NULL(attr_str);

  constexpr auto internal_setattr_owner_str = "__internal_setattr_owner__";
  constexpr auto internal_setattr_value_str = "__internal_setattr_value__";
  std::stringstream script_buffer;
  script_buffer << "__import__('mindspore').common._utils._jit_fallback_set_attr(" << internal_setattr_owner_str << ", "
                << attr_str->value() << ", " << internal_setattr_value_str << ")";
  MS_LOG(DEBUG) << "script: " << script_buffer.str();
  const auto script_setattr_str = std::make_shared<StringImm>(script_buffer.str());

  std::vector<ValuePtr> key_list;
  (void)key_list.emplace_back(std::make_shared<StringImm>(internal_setattr_owner_str));
  (void)key_list.emplace_back(attr_str);
  (void)key_list.emplace_back(std::make_shared<StringImm>(internal_setattr_value_str));
  const auto key_tuple = std::make_shared<ValueTuple>(key_list);

  std::vector<AnfNodePtr> value_list{NewValueNode(prim::kPrimMakeTuple)};
  (void)value_list.emplace_back(owner_node);
  (void)value_list.emplace_back(NewValueNode(attr_str));
  constexpr auto value_node_index = 3;
  (void)value_list.emplace_back(cnode->input(value_node_index));
  const auto value_tuple_node = fg->NewCNode(value_list);

  const auto setattr_node =
    fallback::CreatePyExecuteCNode(cnode, NewValueNode(script_setattr_str), NewValueNode(key_tuple), value_tuple_node);
  MS_LOG(DEBUG) << "setattr_node: " << setattr_node->DebugString(AnfNode::DebugStringLevel::kLevel2);

  // Save abstract for getattr.
  constexpr auto value_abs_index = 2;
  auto value_abs = args_abs_list[value_abs_index];
  if (value_abs != nullptr &&
      (value_abs->isa<abstract::AbstractTensor>() || value_abs->isa<abstract::AbstractScalar>())) {
    auto type = value_abs->BuildType();
    auto shape = value_abs->BuildShape();
    fallback::SetRealType<AnfNode, Type>(setattr_node, type);
    fallback::SetRealShape<AnfNode, abstract::BaseShape>(setattr_node, shape);
  }

  auto eng = out_conf->engine();
  MS_EXCEPTION_IF_NULL(eng);
  auto fn_conf = eng->MakeConfig(setattr_node, out_conf->context(), out_conf->func_graph());
  return eng->ForwardConfig(out_conf, fn_conf);
}

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
  MS_EXCEPTION_IF_NULL(old_conf->node());
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

void SetSideEffectFlag(const PrimitivePtr &prim, const AnfNodeConfigPtr &out_conf) {
  if (prim == nullptr) {
    return;
  }
  auto effect_info = GetPrimEffectInfo(prim);
  if (effect_info.memory || effect_info.io) {
    const auto &cnode = dyn_cast<CNode>(out_conf->node());
    MS_EXCEPTION_IF_NULL(cnode);
    MS_EXCEPTION_IF_NULL(out_conf->func_graph());
    MS_LOG(DEBUG) << "Found side-effect, cnode: " << cnode->DebugString()
                  << ", func_graph: " << out_conf->func_graph()->ToString();
    cnode->set_has_side_effect_node(true);
    out_conf->func_graph()->set_has_side_effect_node(true);
  }
}

void SetOriginObject(const AnfNodePtr &node, const AnfNodeConfigPtr &out_conf) {
  if (!node->isa<ValueNode>()) {
    return;
  }
  auto vnode = node->cast<ValueNodePtr>();
  if (vnode->has_user_data("origin_object")) {
    auto origin_object = vnode->user_data<py::object>("origin_object");
    out_conf->node()->set_user_data<py::object>("origin_object", origin_object);
  }
}

void SetSparseBpropFlag(const PrimitivePtr &prim, const AnfNodeConfigPtr &out_conf) {
  if (GetPrimitiveFlag(prim, GRAPH_FLAG_BPROP_RETURN_SPARSE)) {
    out_conf->func_graph()->set_flag(FUNC_GRAPH_FLAG_SPARSE_BPROP, true);
    EnvSetSparseResultMgr::GetInstance().Set(true);
  }
}

EvalResultPtr GetEvaluatedValueForNameSpaceString(const AbstractBasePtrList &args_abs_list, const ValuePtr &data_value,
                                                  const AnfNodeConfigPtr &out_conf, const std::string &data) {
  constexpr size_t item_index = 1;
  auto item_args = args_abs_list[item_index];
  MS_EXCEPTION_IF_NULL(item_args);
  ValuePtr item_value = item_args->BuildValue();
  MS_EXCEPTION_IF_NULL(data_value);
  MS_EXCEPTION_IF_NULL(item_value);
  if (item_value->isa<StringImm>()) {
    auto string_value = item_value->cast_ptr<StringImm>();
    MS_EXCEPTION_IF_NULL(string_value);
    item_value = std::make_shared<parse::Symbol>(string_value->value());
  }
  if (!item_value->isa<parse::Symbol>()) {
    MS_LOG(INTERNAL_EXCEPTION) << "The value of the attribute could not be inferred: " << item_value->ToString();
  }

  // item_name to func addr from obj_map
  auto symbol = item_value->cast<parse::SymbolPtr>();
  auto name_space = data_value->cast<parse::NameSpacePtr>();
  constexpr auto tensors_queue_attr = "__is_tensors_queue__";
  constexpr auto pop_attr = "pop";
  if (name_space != nullptr && py::hasattr(name_space->namespace_obj(), tensors_queue_attr) &&
      symbol->symbol() == pop_attr) {
    constexpr auto graph_pop_attr = "__graph_pop__";
    symbol = std::make_shared<parse::Symbol>(graph_pop_attr);
  }
  MS_EXCEPTION_IF_NULL(out_conf);
  auto out_node = out_conf->node();
  MS_EXCEPTION_IF_NULL(out_node);
  FuncGraphPtr func_graph = out_node->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  AnalysisEnginePtr eng = out_conf->engine();
  MS_EXCEPTION_IF_NULL(eng);
  parse::Resolver resolver(eng->top_func_graph());
  auto new_node = resolver.ResolveSymbol(func_graph->manager(), name_space, symbol, out_node);
  if (new_node == nullptr) {
    MS_LOG(INTERNAL_EXCEPTION) << "Resolve node failed";
  }

  // If exist side effects of the class method, need insert the Dependent node to retain the side effect info.
  // h = method(xx, xxx)
  // h.wait()   // wait() will be parsed into a subgraph
  // Process h.wait
  if (IsValueNode<FuncGraph>(new_node)) {
    auto out_cnode = out_node->cast_ptr<CNode>();
    MS_EXCEPTION_IF_NULL(out_cnode);
    auto handle_node = out_cnode->input(1);
    MS_EXCEPTION_IF_NULL(handle_node);
    bool need_insert_depend = false;
    if (handle_node->isa<CNode>()) {
      MS_LOG(DEBUG) << "handle_node: " << handle_node->DebugString()
                    << " side_effect:" << handle_node->cast<CNodePtr>()->has_side_effect_node();
      if (handle_node->cast<CNodePtr>()->has_side_effect_node()) {
        need_insert_depend = true;
      }
    }
    MS_LOG(DEBUG) << "need_insert_depend: " << need_insert_depend;
    if (need_insert_depend) {
      auto func = GetValueNode<FuncGraphPtr>(new_node);
      auto output = func->output();
      MS_EXCEPTION_IF_NULL(output);
      auto depend = func->NewCNodeInOrder({NewValueNode(prim::kPrimDepend), output, handle_node});
      MS_LOG(DEBUG) << "depend: " << depend->DebugString();
      depend->set_has_side_effect_node(true);
      MS_LOG(DEBUG) << "Set has_side_effect_node flag for depend: " << out_cnode->DebugString();
      func->set_output(depend);
    }
  }

  auto prim = GetPrimitiveWithoutDoSignature(new_node);
  SetSparseBpropFlag(prim, out_conf);
  SetSideEffectFlag(prim, out_conf);
  SetOriginObject(new_node, out_conf);

  if (IsValueNode<TypeNull>(new_node)) {
    // Do not find the attribute.
    constexpr auto max_args_len = 3;
    bool has_default = (args_abs_list.size() == max_args_len);
    if (!has_default) {
      MS_EXCEPTION_IF_NULL(name_space);
      MS_EXCEPTION_IF_NULL(symbol);
      MS_EXCEPTION(AttributeError) << py::str(name_space->name()) << " object has no attribute '" << symbol->symbol()
                                   << "'";
    }
    auto out_cnode = out_node->cast_ptr<CNode>();
    MS_EXCEPTION_IF_NULL(out_cnode);
    constexpr auto default_index = 3;
    auto default_node = out_cnode->input(default_index);
    auto fn_conf = eng->MakeConfig(default_node, out_conf->context(), out_conf->func_graph());
    return eng->ForwardConfig(out_conf, fn_conf);
  }

  auto new_node_to_fg = GetValueNode<FuncGraphPtr>(new_node);
  if (new_node_to_fg != nullptr) {
    bool has_recompute_scope = (out_node->scope() != nullptr &&
                                out_node->scope()->name().compare(0, strlen(kAttrRecompute), kAttrRecompute) == 0);
    if (has_recompute_scope) {
      parse::UpdateRecomputeScope(new_node_to_fg);
    }
  }

  AnfNodeConfigPtr fn_conf = eng->MakeConfig(new_node, out_conf->context(), out_conf->func_graph());
  return eng->ForwardConfig(out_conf, fn_conf);
}

EvalResultPtr GenerateFuncGraphForOverriddenMethod(AnfNodePtr node, const ValuePtr &item_value,
                                                   const AnfNodeConfigPtr &out_conf) {
  MS_EXCEPTION_IF_NULL(item_value);
  const auto &item_str = item_value->cast_ptr<StringImm>();
  FuncGraphPtr inner_fg = nullptr;
  py::object overridden_method = py::none();
  py::object value_obj = py::none();
  if (item_str == nullptr) {
    return nullptr;
  }
  const std::string &item_name = item_str->value();
  if (node->has_user_data(item_name)) {
    value_obj = *node->user_data<py::object>(item_name);
    overridden_method = value_obj.attr("__class__").attr(item_name.c_str());
  }
  bool is_getattr = node->has_user_data("__getattr__");
  if (is_getattr) {
    value_obj = *node->user_data<py::object>("__getattr__");
    try {
      overridden_method = value_obj.attr("__class__").attr("__getattr__");
    } catch (const std::exception &e) {
      MS_LOG(DEBUG) << value_obj << " has no attribute getattr.";
    }
  }
  if (py::isinstance<py::none>(overridden_method) || py::isinstance<py::none>(value_obj)) {
    return nullptr;
  }
  {
    MS_LOG_TRY_CATCH_SCOPE;
    inner_fg = parse::ParsePythonCode(overridden_method);
  }
  MS_EXCEPTION_IF_NULL(out_conf);
  auto eng = out_conf->engine();
  MS_EXCEPTION_IF_NULL(eng);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  FuncGraphPtr func_graph = node->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  const auto &interpreted_obj = std::make_shared<parse::InterpretedObject>(value_obj);
  const auto &value_node = NewValueNode(interpreted_obj);
  if (inner_fg == nullptr) {
    std::vector<AnfNodePtr> new_inputs;
    for (size_t i = 0; i < cnode->size(); i++) {
      if (i == 1) {
        new_inputs.push_back(value_node);
      } else {
        new_inputs.push_back(cnode->input(i));
      }
    }
    CNodePtr new_cnode = func_graph->NewCNode(new_inputs);
    auto fn_conf = eng->MakeConfig(new_cnode, out_conf->context(), out_conf->func_graph());
    return eng->ForwardConfig(out_conf, fn_conf);
  }
  AddToManager(eng, inner_fg);
  if (is_getattr) {
    std::vector<AnfNodePtr> new_inputs = {NewValueNode(inner_fg)};
    for (size_t i = 0; i < cnode->size(); i++) {
      if (i > 0) {
        new_inputs.push_back(cnode->input(i));
      }
    }
    CNodePtr new_cnode = func_graph->NewCNode(new_inputs);
    auto fn_conf = eng->MakeConfig(new_cnode, out_conf->context(), out_conf->func_graph());
    return eng->ForwardConfig(out_conf, fn_conf);
  }
  std::vector<AnfNodePtr> input = {NewValueNode(prim::kPrimPartial)};
  input.push_back(NewValueNode(inner_fg));
  input.push_back(value_node);
  CNodePtr new_cnode = func_graph->NewCNode(input);
  auto fn_conf = eng->MakeConfig(new_cnode, out_conf->context(), out_conf->func_graph());
  return eng->ForwardConfig(out_conf, fn_conf);
}

EvalResultPtr GetEvaluatedValueForNameSpace(const AbstractBasePtrList &args_abs_list, const AnfNodeConfigPtr &out_conf,
                                            const bool check_override = false) {
  // args_abs_list: same as StaticGetter
  constexpr size_t args_min_size = 2;
  if (args_abs_list.size() < args_min_size) {
    MS_LOG(INTERNAL_EXCEPTION) << "Size of args_abs_list is less than 2";
  }
  MS_EXCEPTION_IF_NULL(out_conf);
  // An external type.
  constexpr auto data_index = 0;
  constexpr auto item_index = 1;
  auto data = args_abs_list[data_index];
  auto item = args_abs_list[item_index];
  MS_EXCEPTION_IF_NULL(data);
  MS_EXCEPTION_IF_NULL(item);
  MS_EXCEPTION_IF_NULL(out_conf->node());
  auto data_value = data->BuildValue();
  MS_EXCEPTION_IF_NULL(data_value);
  auto data_type = data->BuildType();
  MS_EXCEPTION_IF_NULL(data_type);
  auto item_value = item->BuildValue();
  std::string data_id_str = TypeIdToString(data_type->type_id());
  if (check_override) {
    auto inner_fg_res = GenerateFuncGraphForOverriddenMethod(out_conf->node(), item_value, out_conf);
    if (inner_fg_res != nullptr) return inner_fg_res;
  }
  py::module mod = python_adapter::GetPyModule(parse::PYTHON_MOD_PARSE_MODULE);
  if (data_value->isa<parse::ClassType>()) {
    auto class_val = dyn_cast_ptr<parse::ClassType>(data_value);
    auto class_obj = class_val->obj();
    py::object ns_obj = python_adapter::CallPyModFn(mod, parse::PYTHON_MOD_GET_MEMBER_NAMESPACE_SYMBOL, class_obj);
    data_value = std::make_shared<parse::NameSpace>(parse::RESOLVE_NAMESPACE_NAME_CLASS_MEMBER, ns_obj);
    data_id_str = class_val->name();
  }
  if (data_value->isa<parse::MsClassObject>()) {
    auto class_val = dyn_cast_ptr<parse::MsClassObject>(data_value);
    auto class_obj = class_val->obj();
    py::object ns_obj = python_adapter::CallPyModFn(mod, parse::PYTHON_MOD_GET_MEMBER_NAMESPACE_SYMBOL, class_obj);
    data_value = std::make_shared<parse::NameSpace>(parse::RESOLVE_NAMESPACE_NAME_CLASS_MEMBER, ns_obj);
    data_id_str = class_val->name();
  }
  if (!data_value->isa<parse::NameSpace>()) {
    MS_EXCEPTION_IF_NULL(item_value);
    MS_LOG(DEBUG) << "Evaluate " << data_value->ToString() << " attribute: " << item_value->ToString()
                  << ".\nnode: " << out_conf->node()->DebugString() << "\n"
                  << trace::GetDebugInfoStr(out_conf->node()->debug_info());
    auto res = InterpretGetAttrNode(args_abs_list, out_conf);
    if (res == nullptr) {
      MS_EXCEPTION(AttributeError) << data_value->ToString() << " object has no attribute: " << item_value->ToString();
    }
    return res;
  }
  return GetEvaluatedValueForNameSpaceString(args_abs_list, data_value, out_conf, data_id_str);
}

EvalResultPtr GetEvaluatedValueForPrimitiveAttr(const AbstractBasePtrList &args_abs_list,
                                                const AbstractFunctionPtr &data_args) {
  MS_EXCEPTION_IF_NULL(data_args);
  if (!data_args->isa<PrimitiveAbstractClosure>()) {
    return nullptr;
  }
  auto prim_abs = dyn_cast_ptr<PrimitiveAbstractClosure>(data_args);
  const auto &prim = prim_abs->prim();
  MS_EXCEPTION_IF_NULL(prim);
  constexpr auto item_index = 1;
  auto item_arg = args_abs_list.at(item_index);
  MS_EXCEPTION_IF_NULL(item_arg);
  auto attr_name = GetValue<string>(item_arg->BuildValue());
  auto value = prim->GetAttr(attr_name);
  if (value == nullptr) {
    MS_LOG(INFO) << "The Primitive: " << prim->ToString() << " has not attr " << attr_name;
    MS_LOG(INFO) << "PrimAttr: " << prim->GetAttrsText();
    return nullptr;
  }
  return std::make_shared<EvalResult>(value->ToAbstract(), nullptr);
}

EvalResultPtr GetEvaluatedValueForFunctionalMethod(const AnalysisEnginePtr &engine, const ValuePtr &method_value,
                                                   const ConfigPtr &data_conf, const AnfNodeConfigPtr &out_conf) {
  MS_EXCEPTION_IF_NULL(method_value);
  auto method_str = method_value->cast_ptr<StringImm>();
  MS_EXCEPTION_IF_NULL(method_str);
  std::string method_name = method_str->value();
  MS_EXCEPTION_IF_NULL(out_conf);
  MS_EXCEPTION_IF_NULL(out_conf->node());
  FuncGraphPtr func_graph = out_conf->node()->func_graph();
  // Create node: {Partial, Functional(method_name), Tensor}
  const auto &functional = BuildMethodFunctional(method_name);
  auto data_node_conf = dyn_cast_ptr<abstract::AnfNodeConfig>(data_conf);
  MS_EXCEPTION_IF_NULL(data_node_conf);
  auto data_node = data_node_conf->node();
  auto new_cnode = func_graph->NewCNode({NewValueNode(prim::kPrimPartial), NewValueNode(functional), data_node});
  MS_LOG(DEBUG) << "Convert py_method '" << method_name << "' to node: " << new_cnode->DebugString();
  AnalysisEnginePtr eng = out_conf->engine();
  AnfNodeConfigPtr fn_conf = eng->MakeConfig(new_cnode, out_conf->context(), out_conf->func_graph());
  return eng->ForwardConfig(out_conf, fn_conf);
}

py::object GetOriginObj(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  py::object origin_obj;
  if (node->has_user_data("origin_object")) {
    return *node->user_data<py::object>("origin_object");
  }
  return origin_obj;
}

EvalResultPtr GetEvaluatedValueForAttrOrMethodNotInMap(const AnalysisEnginePtr &engine,
                                                       const AbstractBasePtrList &args_abs_list,
                                                       const AnfNodeConfigPtr &out_conf, const std::string &item_name,
                                                       const TypePtr &data_type) {
  constexpr auto max_args_len = 3;
  bool has_default = (args_abs_list.size() == max_args_len);
  auto out_node = out_conf->node();
  auto out_cnode = out_node->cast_ptr<CNode>();
  MS_EXCEPTION_IF_NULL(out_cnode);
  auto eng = out_conf->engine();
  MS_EXCEPTION_IF_NULL(eng);
  if (has_default) {
    constexpr auto default_index = 3;
    auto default_node = out_cnode->input(default_index);
    auto fn_conf = eng->MakeConfig(default_node, out_conf->context(), out_conf->func_graph());
    return eng->ForwardConfig(out_conf, fn_conf);
  }

  py::object value_obj = GetOriginObj(out_cnode->input(1));
  if (value_obj.ptr() != nullptr) {
    std::vector<AnfNodePtr> new_inputs;
    std::string data_type_str = TypeIdLabel(NormalizeTypeId(data_type->type_id()));
    py::module mod1 = python_adapter::GetPyModule(parse::PYTHON_MOD_PARSE_MODULE);
    py::object obj_define = python_adapter::CallPyModFn(mod1, parse::PYTHON_MOD_GET_OBJ_DEFINED, data_type_str);
    py::object check_res =
      python_adapter::CallPyModFn(mod1, parse::PYTHON_MOD_CHECK_IS_SUBCLASS, value_obj, obj_define);
    if (py::cast<bool>(check_res)) {
      for (size_t i = 0; i < out_cnode->size(); i++) {
        if (i == 1) {
          const auto &interpreted_obj = std::make_shared<parse::InterpretedObject>(value_obj);
          const auto &value_node = NewValueNode(interpreted_obj);
          new_inputs.push_back(value_node);
        } else {
          new_inputs.push_back(out_cnode->input(i));
        }
      }
      CNodePtr new_cnode = out_conf->func_graph()->NewCNode(new_inputs);
      auto fn_conf = eng->MakeConfig(new_cnode, out_conf->context(), out_conf->func_graph());
      return eng->ForwardConfig(out_conf, fn_conf);
    }
  }
  const auto allow_fallback_runtime = (fallback::GetJitSyntaxLevel() == kLax);
  if (!allow_fallback_runtime) {
    MS_EXCEPTION(AttributeError) << "In JIT strict mode, cannot get attributes " << item_name << " or the "
                                 << data_type->ToString() << " object has no attribute: '" << item_name
                                 << "'. You can use os.environ['MS_DEV_JIT_SYNTAX_LEVEL'] = '2' "
                                 << "to enable the JIT lax mode to support the current syntax.\n\n"
                                 << trace::GetDebugInfoStr(out_conf->node()->debug_info());
  }

  MS_LOG(DEBUG) << "Evaluate " << data_type->ToString() << " attribute: " << item_name
                << ".\nnode: " << out_conf->node()->DebugString(AnfNode::DebugStringLevel::kLevel3) << "\n"
                << trace::GetDebugInfoStr(out_conf->node()->debug_info());
  auto res = InterpretGetAttrNode(args_abs_list, out_conf);
  if (res == nullptr) {
    MS_EXCEPTION(AttributeError) << data_type->ToString() << " object has no attribute: " << item_name;
  }
  return res;
}

EvalResultPtr GetEvaluatedValueForBuiltinTypeAttrOrMethod(const AnalysisEnginePtr &engine,
                                                          const AbstractBasePtrList &args_abs_list,
                                                          const ConfigPtr &data_conf,
                                                          const AnfNodeConfigPtr &out_conf) {
  constexpr size_t data_index = 0;
  constexpr size_t item_index = 1;
  auto data_args = args_abs_list[data_index];
  auto item_args = args_abs_list[item_index];
  MS_EXCEPTION_IF_NULL(data_args);
  MS_EXCEPTION_IF_NULL(item_args);
  ValuePtr item_value = item_args->BuildValue();
  MS_EXCEPTION_IF_NULL(item_value);
  TypePtr data_type = data_args->BuildType();
  MS_EXCEPTION_IF_NULL(data_type);
  // Handle NameTuple: getattr(XX, item_value) -> ValueNode().
  if (data_args->isa<AbstractNamedTuple>()) {
    auto named_tuple = data_args->cast<AbstractNamedTuplePtr>();
    const auto &keys = named_tuple->key();
    for (size_t it = 0; it < keys.size(); ++it) {
      auto key_value = keys[it]->BuildValue();
      MS_EXCEPTION_IF_NULL(key_value);
      if (*item_value == *key_value) {
        auto getattr_node = NewValueNode(named_tuple->elements()[it]->BuildValue());
        auto eng = out_conf->engine();
        MS_EXCEPTION_IF_NULL(eng);
        auto fn_conf = eng->MakeConfig(getattr_node, out_conf->context(), out_conf->func_graph());
        return eng->ForwardConfig(out_conf, fn_conf);
      }
    }
  }

  // The method maybe a Primitive or Composite
  if (!item_value->isa<StringImm>()) {
    MS_LOG(EXCEPTION) << "Expect a string, but got: " << item_value->ToString();
  }
  auto item_str = item_value->cast_ptr<StringImm>();
  MS_EXCEPTION_IF_NULL(item_str);
  std::string item_name = item_str->value();
  REQUIRE_TYPE require_type = REQUIRE_TYPE::METHOD;
  Any require = pipeline::Resource::GetMethodPtr(data_type->type_id(), item_name);
  MS_EXCEPTION_IF_NULL(out_conf->node());
  if (require.empty()) {
    require = pipeline::Resource::GetAttrPtr(data_type->type_id(), item_name);
    if (require.empty()) {
      return GetEvaluatedValueForAttrOrMethodNotInMap(engine, args_abs_list, out_conf, item_name, data_type);
    }
    require_type = REQUIRE_TYPE::ATTR;
  }

  ValuePtr converted_value = nullptr;
  if (require.is<std::string>()) {
    // composite registered in standard_method_map go to this branch
    converted_value = prim::GetPythonOps(require.cast<std::string>());
    MS_EXCEPTION_IF_NULL(converted_value);

    auto converted_fg = converted_value->cast<FuncGraphPtr>();
    if (converted_fg != nullptr) {
      bool has_recompute_scope =
        (out_conf->node()->scope() != nullptr &&
         out_conf->node()->scope()->name().compare(0, strlen(kAttrRecompute), kAttrRecompute) == 0);
      if (has_recompute_scope) {
        parse::UpdateRecomputeScope(converted_fg);
      }
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

EvalResultPtr TransPropertyToFunc(const AnfNodeConfigPtr &out_conf, py::object property_net_obj,
                                  std::string item_name) {
  py::object property_func = py::none();
  try {
    property_func = property_net_obj.attr("__class__").attr(py::str(item_name));
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << property_net_obj << " has no attribute " << item_name;
  }
  py::object property_func_fget = property_func.attr(py::str("fget"));
  auto inner_fg = parse::ParsePythonCode(property_func_fget);
  auto eng = out_conf->engine();
  MS_EXCEPTION_IF_NULL(eng);
  AddToManager(eng, inner_fg);
  auto node = out_conf->node();
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  FuncGraphPtr func_graph = node->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  std::vector<AnfNodePtr> new_inputs = {NewValueNode(inner_fg)};
  new_inputs.push_back(cnode->input(1));
  CNodePtr new_cnode = func_graph->NewCNode(new_inputs);
  MS_LOG(DEBUG) << "new_cnode:" << new_cnode->DebugString();
  auto fn_conf = eng->MakeConfig(new_cnode, out_conf->context(), out_conf->func_graph());
  return eng->ForwardConfig(out_conf, fn_conf);
}

EvalResultPtr GetClassAttrFromPyObject(const py::object &cls_obj, const std::string &cls_name,
                                       const AbstractBasePtrList &args_abs_list, const AnfNodeConfigPtr &out_conf) {
  py::module mod = python_adapter::GetPyModule(parse::PYTHON_MOD_PARSE_MODULE);
  constexpr auto item_index = 1;
  auto item_arg = args_abs_list.at(item_index);
  MS_EXCEPTION_IF_NULL(item_arg);
  auto attr_name = GetValue<string>(item_arg->BuildValue());
  bool is_property =
    (python_adapter::CallPyModFn(mod, parse::PYTHON_PARSE_CHECK_ATTR_IS_PROPERTY, cls_obj, attr_name)).cast<bool>();
  if (is_property) {
    ValuePtr item_value = item_arg->BuildValue();
    MS_EXCEPTION_IF_NULL(item_value);
    const auto &item_str = item_value->cast_ptr<StringImm>();
    MS_EXCEPTION_IF_NULL(item_str);
    const std::string &item_name = item_str->value();
    return TransPropertyToFunc(out_conf, cls_obj, item_name);
  }
  py::object ns_obj = python_adapter::CallPyModFn(mod, parse::PYTHON_MOD_GET_MEMBER_NAMESPACE_SYMBOL, cls_obj);
  auto ns = std::make_shared<parse::NameSpace>(parse::RESOLVE_NAMESPACE_NAME_CLASS_MEMBER, ns_obj);
  return GetEvaluatedValueForNameSpaceString(args_abs_list, ns, out_conf, cls_name);
}

EvalResultPtr GetFuncAbstractAttr(const AbstractFunctionPtr &data_args, const AbstractBasePtrList &args_abs_list,
                                  const AnfNodeConfigPtr &out_conf) {
  if (data_args == nullptr) {
    return nullptr;
  }
  // Get attribute or method of PartialAbstractClosure, the object could be nn.Cell/ms_class object.
  auto data_partial = dyn_cast_ptr<PartialAbstractClosure>(data_args);
  if (data_partial != nullptr) {
    const auto &partial_args = data_partial->args();
    auto prim_abs = dyn_cast_ptr<PrimitiveAbstractClosure>(data_partial->fn());
    if (prim_abs != nullptr && !partial_args.empty()) {
      MS_EXCEPTION_IF_NULL(prim_abs->prim());
      const auto &prim_name = prim_abs->prim()->name();
      if (prim_name == prim::kPrimCreateInstance->name()) {
        constexpr size_t class_index = 0;
        MS_EXCEPTION_IF_NULL(partial_args[class_index]);
        auto class_val = partial_args[class_index]->BuildValue();
        MS_EXCEPTION_IF_NULL(class_val);
        auto wrapper = dyn_cast_ptr<parse::PyObjectWrapper>(class_val);
        MS_EXCEPTION_IF_NULL(wrapper);
        return GetClassAttrFromPyObject(wrapper->obj(), wrapper->name(), args_abs_list, out_conf);
      }
    }
    return nullptr;
  }
  // Get attribute or method of FuncGraphAbstractClosure, the object could be nn.Cell/ms_class object.
  const auto &cls_obj = fallback::GetPyObjForFuncGraphAbstractClosure(data_args);
  if (py::isinstance<Cell>(cls_obj) || py::hasattr(cls_obj, PYTHON_MS_CLASS)) {
    return GetClassAttrFromPyObject(cls_obj, py::str(cls_obj), args_abs_list, out_conf);
  }
  return GetEvaluatedValueForPrimitiveAttr(args_abs_list, data_args);
}

bool CheckHasOverriddenMethod(AnfNodePtr node, ValuePtr item_value) {
  const auto &item_str = item_value->cast_ptr<StringImm>();
  py::module mod = python_adapter::GetPyModule(parse::PYTHON_MOD_PARSE_MODULE);
  if (item_str == nullptr) {
    return false;
  }
  const std::string &item_name = item_str->value();
  if (node->has_user_data(item_name)) {
    auto value_obj = *node->user_data<py::object>(item_name);
    py::bool_ check = python_adapter::CallPyModFn(mod, parse::PYTHON_MOD_CHECK_ATTRS, value_obj, item_name);
    return py::cast<bool>(check);
  }
  if (node->has_user_data("__getattr__")) {
    auto value_obj = *node->user_data<py::object>("__getattr__");
    if (py::hasattr(value_obj, item_name.c_str())) {
      return false;
    }
    py::bool_ check = python_adapter::CallPyModFn(mod, parse::PYTHON_MOD_CHECK_ATTRS, value_obj, "__getattr__");
    return py::cast<bool>(check);
  }
  return false;
}

bool CheckFunctionalMethod(const TypeId &type_id, const ValuePtr &method_value) {
  // ge does not support tensor method overloading.
  auto ge_mode = AnfAlgo::IsBackendGe();
  if (ge_mode) {
    return false;
  }
  // Check if tensor.
  if (NormalizeTypeId(type_id) != kObjectTypeTensorType) {
    return false;
  }
  // Get method name.
  if (!method_value->isa<StringImm>()) {
    MS_LOG(EXCEPTION) << "Expect a string, but got: " << method_value->ToString();
  }
  auto method_name = method_value->cast_ptr<StringImm>()->value();
  return prim::IsFunctionalMethod(type_id, method_name);
}

EvalResultPtr StaticGetter(const AnalysisEnginePtr &engine, const AbstractBasePtrList &args_abs_list,
                           const ConfigPtr &data_conf, const AnfNodeConfigPtr &out_conf) {
  // Inputs: namespace and its static function; or class and its member function
  constexpr size_t data_index = 0;
  constexpr size_t item_index = 1;
  auto data_args = args_abs_list[data_index];
  auto item_args = args_abs_list[item_index];
  MS_EXCEPTION_IF_NULL(data_args);
  MS_EXCEPTION_IF_NULL(item_args);
  MS_EXCEPTION_IF_NULL(out_conf);
  MS_EXCEPTION_IF_NULL(out_conf->node());
  MS_LOG(DEBUG) << "StaticGetter, data: " << data_args->ToString() << ", item: " << item_args->ToString()
                << ", node: " << out_conf->node()->DebugString(AnfNode::DebugStringLevel::kLevel2);
  ScopePtr scope = out_conf->node()->scope();
  ScopeGuard scope_guard(scope);
  ValuePtr item_value = item_args->BuildValue();
  MS_EXCEPTION_IF_NULL(item_value);
  if (item_value->ContainsValueAny()) {
    MS_LOG(INTERNAL_EXCEPTION) << "The value of the attribute could not be inferred: " << item_value->ToString();
  }

  const auto allow_fallback_runtime = (fallback::GetJitSyntaxLevel() == kLax);
  constexpr auto max_args_size = 3;
  if (!allow_fallback_runtime && args_abs_list.size() == max_args_size) {
    constexpr size_t default_index = 2;
    auto default_args = args_abs_list[default_index];
    MS_EXCEPTION_IF_NULL(default_args);
    if (default_args->isa<abstract::AbstractScalar>()) {
      ValuePtr default_value = default_args->BuildValue();
      MS_EXCEPTION_IF_NULL(default_value);
      if (default_value->isa<parse::InterpretedObject>()) {
        auto obj = ValueToPyData(default_value);
        auto type_str = python_adapter::CallPyFn(parse::PYTHON_MOD_PARSE_MODULE, parse::PYTHON_PARSE_GET_TYPE, obj);
        MS_EXCEPTION(TypeError) << "For 'getattr', the third input 'default' can not be " << py::str(type_str)
                                << " object " << py::str(obj);
      }
    }
  }

  auto res = GetFuncAbstractAttr(data_args->cast<AbstractFunctionPtr>(), args_abs_list, out_conf);
  if (res != nullptr) {
    return res;
  }

  // Try to search method map, if not found, the data_type should be External type.
  TypePtr data_type = data_args->BuildType();
  MS_EXCEPTION_IF_NULL(data_type);
  // Check if attr is a overridden method.
  bool check_override = CheckHasOverriddenMethod(out_conf->node(), item_value);
  // Not check if the data is from PyExecute CNode, since its Tensor output is pseud.
  auto data_type_id = data_type->type_id();
  if (!IsPyExecuteData(data_args) && !check_override) {
    if (CheckFunctionalMethod(data_type_id, item_value)) {
      return GetEvaluatedValueForFunctionalMethod(engine, item_value, data_conf, out_conf);
    }
    if (pipeline::Resource::IsTypeInBuiltInMap(data_type_id)) {
      return GetEvaluatedValueForBuiltinTypeAttrOrMethod(engine, args_abs_list, data_conf, out_conf);
    }
  }
  return GetEvaluatedValueForNameSpace(args_abs_list, out_conf, check_override);
}

TypePtr GetAnnotationType(const AnfNodePtr &node, const AbstractBasePtrList &args_abs_list) {
  MS_EXCEPTION_IF_NULL(node);
  fallback::FormatedVariableTypeFunc func = [&node, &args_abs_list](const std::string &type_var_str) -> TypePtr {
    // For PyInterpret, the args[1] is global dict, and the args[2] is local dict.
    // For PyExecute, the args[1] is local dict keys, and the args[2] is local dict values.
    ValuePtr type_value = nullptr;
    const auto &keys_tuple_abs = args_abs_list[1];
    MS_EXCEPTION_IF_NULL(keys_tuple_abs);
    const auto &keys_tuple = keys_tuple_abs->BuildValue();
    const auto &keys = dyn_cast<ValueSequence>(keys_tuple);
    bool is_py_execute = (keys != nullptr);
    if (is_py_execute) {  // PyExecute.
      bool found = false;
      size_t i = 0;
      for (; i < keys->value().size(); ++i) {
        const auto &key = dyn_cast<StringImm>(keys->value()[i]);
        MS_EXCEPTION_IF_NULL(key);
        if (key->value() == type_var_str) {
          found = true;
          break;
        }
      }

      if (!found) {
        MS_LOG(INFO) << "Not valid PyExecute CNode. node: " << node->DebugString() << ", keys: " << keys->ToString()
                     << ", not found " << type_var_str;
        return nullptr;
      }
      constexpr auto values_index = 2;
      const auto &values_tuple_abs = dyn_cast<AbstractSequence>(args_abs_list[values_index]);
      MS_EXCEPTION_IF_NULL(values_tuple_abs);
      const auto &type_value_abs = values_tuple_abs->elements()[i];
      if (type_value_abs == nullptr) {
        MS_LOG(INFO) << "Not valid PyExecute CNode. node: " << node->DebugString() << ", key: " << type_var_str
                     << ", values_tuple_abs: " << values_tuple_abs->ToString();
        return nullptr;
      }
      bool only_has_real_type = !fallback::HasRealShape(type_value_abs) && fallback::HasRealType(type_value_abs);
      type_value =
        only_has_real_type ? fallback::GetRealType<AbstractBase, Type>(type_value_abs) : type_value_abs->BuildValue();
    } else {  // PyInterpret
      constexpr auto local_dict_index = 2;
      const auto &local_dict_abs = args_abs_list[local_dict_index];
      const auto &dict = dyn_cast<AbstractDictionary>(local_dict_abs);
      if (dict == nullptr || dict->elements().empty()) {
        MS_EXCEPTION_IF_NULL(local_dict_abs);
        MS_LOG(INFO) << "Not valid PyInterpret CNode. node: " << node->DebugString() << ", key: " << type_var_str
                     << ", local_dict_abs: " << local_dict_abs->ToString();
        return nullptr;
      }
      for (const auto &element : dict->elements()) {
        MS_EXCEPTION_IF_NULL(element.first);
        const auto &key = element.first->BuildValue();
        if (key == nullptr || !key->isa<StringImm>()) {
          continue;
        }
        if (key->cast<StringImmPtr>()->value() == type_var_str) {
          MS_EXCEPTION_IF_NULL(element.second);
          type_value = element.second->BuildValue();
          break;
        }
      }
    }

    if (type_value == nullptr) {
      MS_LOG(INFO) << "Not valid " << (is_py_execute ? "PyExecute" : "PyInterpret")
                   << " CNode. node: " << node->DebugString() << ", key: " << type_var_str << ", type value is null.";
      return nullptr;
    }
    const auto &py_type = ValueToPyData(type_value);

    MS_LOG(DEBUG) << "type_value: " << type_value->ToString() << ", py_type: " << py_type;
    if (!py::isinstance<py::none>(py_type)) {
      return py::cast<TypePtr>(py_type);
    }
    MS_LOG(INFO) << "Not valid " << (is_py_execute ? "PyExecute" : "PyInterpret")
                 << " CNode. node: " << node->DebugString() << ", key: " << type_var_str << ", type value is None.";
    return nullptr;
  };
  const auto &type = fallback::GetJitAnnotationTypeFromComment(node, func);
  return type;
}

TypePtr GetLocalArgsUniqueDtype(const AnfNodePtr &node, const AbstractBasePtrList &args_abs_list) {
  // If force to use ANY.
  static const auto force_any = (common::GetCompileConfig("FALLBACK_FORCE_ANY") == "1");
  if (force_any) {
    return nullptr;
  }

  TypePtr res = nullptr;
  // Check the abstract, return true if continue, otherwise return false.
  auto unique_dtype_check = [&node, &res](const AbstractBasePtr &element_value_abs) -> bool {
    MS_EXCEPTION_IF_NULL(element_value_abs);
    if (!element_value_abs->isa<abstract::AbstractTensor>()) {
      return true;
    }
    // Fetch the dtype from element_value_abs of tensor.
    auto element_abs_tensor = element_value_abs->cast_ptr<abstract::AbstractTensor>();
    MS_EXCEPTION_IF_NULL(element_abs_tensor);
    MS_EXCEPTION_IF_NULL(element_abs_tensor->element());
    const auto dtype = element_abs_tensor->element()->BuildType();
    MS_EXCEPTION_IF_NULL(dtype);
    // Check default dtype if it's AbstractAny(AbstractTensor)
    if (element_value_abs->isa<abstract::AbstractAny>() &&
        !element_value_abs->cast_ptr<abstract::AbstractAny>()->supposed_tensor_dtype()) {
      return true;
    }
    if (res == nullptr) {
      MS_EXCEPTION_IF_NULL(node);
      MS_LOG(INFO) << "Tensor dtype found, set as unique dtype: " << dtype->ToString()
                   << ", node: " << node->DebugString() << "\n\n"
                   << trace::GetDebugInfoStr(node->debug_info());
      res = dtype;
      return true;
    }
    if (res != dtype) {
      MS_EXCEPTION_IF_NULL(node);
      MS_LOG(INFO) << "More than one tensor dtype found, not set unique dtype. node: " << node->DebugString() << "\n\n"
                   << trace::GetDebugInfoStr(node->debug_info());
      return false;
    }
    return true;
  };
  constexpr auto values_index = 2;
  if (args_abs_list.size() <= values_index) {
    return nullptr;
  }
  const auto &values_tuple_abs = dyn_cast<AbstractSequence>(args_abs_list[values_index]);
  bool is_py_execute = (values_tuple_abs != nullptr);
  if (is_py_execute) {  // PyExecute CNode.
    const auto &elements_abs = values_tuple_abs->elements();
    for (const auto &element_abs : elements_abs) {
      if (!unique_dtype_check(element_abs)) {
        return nullptr;
      }
    }
  } else {  // PyInterpret CNode.
    const auto &local_dict_abs = dyn_cast<AbstractDictionary>(args_abs_list[values_index]);
    MS_EXCEPTION_IF_NULL(local_dict_abs);
    const auto &elements_abs = local_dict_abs->elements();
    for (const auto &element_abs_pair : elements_abs) {
      const auto &element_value_abs = element_abs_pair.second;
      if (!unique_dtype_check(element_value_abs)) {
        return nullptr;
      }
    }
  }

  if (res != nullptr) {
    MS_LOG(INFO) << "Apply unique dtype: " << res->ToString() << " to node: " << node->DebugString() << "\n\n"
                 << trace::GetDebugInfoStr(node->debug_info());
  }
  return res;
}

void AddLabelsToPrimitiveFunction(const PrimitivePtr &prim_func) {
  MS_EXCEPTION_IF_NULL(prim_func);
  auto prim_name = prim_func->name();
  py::module mod = py::module::import(parse::PYTHON_MOD_PRIMITIVE_OP_CREATE_INSTANCE_HELPER_MODULE);
  if (!py::hasattr(mod, parse::PYTHON_MOD_PRIMITIVE_OP_LABELS_DICT)) {
    MS_LOG(INTERNAL_EXCEPTION) << "Can not found " << parse::PYTHON_MOD_PRIMITIVE_OP_LABELS_DICT << " in "
                               << parse::PYTHON_MOD_PRIMITIVE_OP_CREATE_INSTANCE_HELPER_MODULE << ".";
  }
  py::dict op_labels = mod.attr(parse::PYTHON_MOD_PRIMITIVE_OP_LABELS_DICT);
  if (!op_labels.contains(py::str(prim_name))) {
    return;
  }
  py::dict labels = op_labels[py::str(prim_name)];
  for (const auto &p : labels) {
    auto attr_name = py::cast<std::string>(p.first);
    auto attr_obj = py::reinterpret_borrow<py::object>(p.second);
    ValuePtr converted_ret = nullptr;
    bool converted = parse::ConvertData(attr_obj, &converted_ret);
    if (!converted) {
      MS_LOG(INTERNAL_EXCEPTION) << "Call 'add_attr' to add attribute to primitive failed,"
                                 << " convert python obj to MindSpore obj failed; primitive name: " << prim_name
                                 << ", attribute name:" << attr_name << ", attribute value:" << py::str(attr_obj)
                                 << ", attribute type:"
                                 << py::cast<std::string>(attr_obj.attr("__class__").attr("__name__"));
    }
    MS_LOG(DEBUG) << "Add attr {" << attr_name << ": " << converted_ret->ToString() << "} to " << prim_name;
    (void)prim_func->AddAttr(attr_name, converted_ret);
  }
}

AnfNodePtr ConvertArgsToInputs(const PrimitivePtr &prim, const AnfNodeWeakPtrList &inputs, const FuncGraphPtr &fg,
                               const AnalysisEnginePtr &engine, const AnfNodeConfigPtr &out_conf) {
  // Append Primitive arguments to the inputs.
  auto prim_py = prim->cast<PrimitivePyPtr>();
  MS_EXCEPTION_IF_NULL(prim_py);
  auto op_def = mindspore::ops::GetOpDef(prim->name());
  MS_EXCEPTION_IF_NULL(op_def);
  // Get init args.
  const AnfNodePtrList &prim_init_arg_nodes = GetPrimitiveInitArgs(prim_py, op_def);

  // Get call args.
  AnfNodePtrList prim_call_arg_nodes;
  (void)std::transform(inputs.cbegin() + 1, inputs.cend(), std::back_inserter(prim_call_arg_nodes),
                       [](const AnfNodeWeakPtr &weak_node) {
                         const auto &node = weak_node.lock();
                         MS_EXCEPTION_IF_NULL(node);
                         return node;
                       });
  // Create new node.
  auto new_prim = std::make_shared<Primitive>(*prim);
  auto args_pair = std::make_pair(prim_init_arg_nodes, prim_call_arg_nodes);
  return CheckAndConvertPrimitiveArgs(new_prim, args_pair, engine, out_conf, true);
}

bool IsGetAttrNode(const AnfNodePtr &op_node) {
  if (IsPrimitiveCNode(op_node, prim::kPrimGetAttr) ||
      IsPrimitiveCNodeWithoutDoSignature(op_node, prim::kPrimGetAttr)) {
    return true;
  }
  constexpr size_t index_op = 0;
  constexpr size_t index_symbol = 2;
  parse::SymbolPtr symbol_node = nullptr;
  if (op_node->isa<CNode>()) {
    auto inner_op_node = op_node->cast<CNodePtr>()->input(index_op);
    if (IsPrimitiveCNode(inner_op_node, prim::kPrimResolve)) {
      auto resolve_node = inner_op_node->cast<CNodePtr>();
      symbol_node = GetValueNode<parse::SymbolPtr>(resolve_node->input(index_symbol));
    }
  }
  return symbol_node != nullptr && symbol_node->symbol() == "getattr";
}

bool IsPrimitiveAbstract(const AnfNodePtr &op_node, const AnalysisEnginePtr &engine, const AnfNodeConfigPtr &out_conf) {
  auto eval_func = [&engine, &out_conf](const AnfNodePtr &node) {
    AnfNodeConfigPtr config = engine->MakeConfig(node, out_conf->context(), out_conf->func_graph());
    MS_EXCEPTION_IF_NULL(config);
    const auto &eval_result = config->ObtainEvalResult();
    MS_EXCEPTION_IF_NULL(eval_result);
    return eval_result->abstract();
  };
  auto op_abs = eval_func(op_node);
  MS_EXCEPTION_IF_NULL(op_abs);
  return op_abs->isa<PrimitiveAbstractClosure>();
}

ValuePtr GetRefKeyValue(const AbstractBasePtr &abs) {
  auto abs_ref = abs->cast_ptr<AbstractRefTensor>();
  if (abs_ref != nullptr) {
    return abs_ref->ref_key_value();
  }
  auto abs_map_tensor = abs->cast_ptr<AbstractMapTensor>();
  if (abs_map_tensor != nullptr) {
    return abs_map_tensor->ref_key_value();
  }
  return nullptr;
}
}  // namespace

EvalResultPtr PrimitiveArgsToInputsEvaluator::EvalPrim(const AnalysisEnginePtr &engine,
                                                       const AbstractBasePtrList &args_abs_list, const ConfigPtr &,
                                                       const AnfNodeConfigPtr &out_conf) {
  // Convert primitive args to inputs.
  MS_EXCEPTION_IF_NULL(out_conf);
  auto cnode = out_conf->node()->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto fg = cnode->func_graph();
  MS_EXCEPTION_IF_NULL(fg);

  constexpr size_t index_op = 0;
  constexpr size_t index_data = 1;
  auto op_node = cnode->input(index_op);
  AnfNodePtr new_node = nullptr;
  if (IsPrimitiveCNode(op_node, prim::kPrimPartial)) {
    // The input may be a Partial node, such as {{prim::kPrimPartial, prim::kPrimRank, x}} -> {prim::kPrimRank, x}.
    AnfNodeWeakPtrList partial_inputs;
    auto op_cnode = op_node->cast<CNodePtr>();
    (void)std::copy(op_cnode->weak_inputs().begin() + index_data, op_cnode->weak_inputs().end(),
                    std::back_inserter(partial_inputs));
    (void)std::copy(cnode->weak_inputs().begin() + index_data, cnode->weak_inputs().end(),
                    std::back_inserter(partial_inputs));
    new_node = ConvertArgsToInputs(prim_, partial_inputs, fg, engine, out_conf);
  } else if (IsGetAttrNode(op_node) && !IsPrimitiveAbstract(op_node, engine, out_conf)) {
    // The input may be a GetAttr node, such as x.abs(): {{prim::kPrimGetAttr, x, abs}} -> {prim::kPrimAbs, x}
    auto op_cnode = op_node->cast<CNodePtr>();
    AnfNodeWeakPtrList getattr_inputs;
    auto new_prim = std::make_shared<Primitive>(prim_->name());
    auto new_prim_node = NewValueNode(new_prim);
    (void)getattr_inputs.emplace_back(new_prim_node);
    (void)getattr_inputs.emplace_back(op_cnode->input(index_data));
    (void)std::copy(cnode->weak_inputs().begin() + index_data, cnode->weak_inputs().end(),
                    std::back_inserter(getattr_inputs));
    new_node = ConvertArgsToInputs(prim_, getattr_inputs, fg, engine, out_conf);
  } else {
    new_node = ConvertArgsToInputs(prim_, cnode->weak_inputs(), fg, engine, out_conf);
    MS_LOG(DEBUG) << "Convert args to inputs for Operator[" << prim_->name()
                  << "], node: " << cnode->DebugString(AnfNode::DebugStringLevel::kLevel2);
  }

  new_node->set_debug_info(cnode->debug_info());
  auto new_conf = engine->MakeConfig(new_node, out_conf->context(), out_conf->func_graph());
  MS_LOG(DEBUG) << "Convert primitive args to inputs: " << prim_->ToString() << ". node: " << cnode->DebugString()
                << ", new node: " << new_node->DebugString();
  return engine->ForwardConfig(out_conf, new_conf);
}

namespace {
AnfNodePtr ConvertWeakNode(const AnfNodeWeakPtr &weak_node) {
  const auto &node = weak_node.lock();
  MS_EXCEPTION_IF_NULL(node);
  return node;
}

ValuePtr GetShardStrategy(const PrimitivePtr &prim, const std::string &name) {
  const auto &strategy = prim->GetAttr(name);
  return strategy != nullptr ? strategy : kNone;
}

AnfNodePtr HandleShardForPrimitive(const PrimitivePtr &prim, const prim::MetaImplPtr &meta_op, const FuncGraphPtr &fg) {
  ValuePtr in_strategy = nullptr;
  ValuePtr out_strategy = nullptr;
  if (prim->HasAttr(parallel::IN_STRATEGY) || prim->HasAttr(parallel::OUT_STRATEGY)) {
    in_strategy = GetShardStrategy(prim, parallel::IN_STRATEGY);
    out_strategy = GetShardStrategy(prim, parallel::OUT_STRATEGY);
  } else if (prim->HasAttr(parallel::IN_LAYOUT) || prim->HasAttr(parallel::OUT_LAYOUT)) {
    in_strategy = GetShardStrategy(prim, parallel::IN_LAYOUT);
    out_strategy = GetShardStrategy(prim, parallel::OUT_LAYOUT);
  } else {
    return NewValueNode(meta_op);
  }
  MS_EXCEPTION_IF_NULL(in_strategy);
  MS_EXCEPTION_IF_NULL(out_strategy);
  MS_LOG(DEBUG) << "Set shard for Primitive[" << prim->name() << "] with in_strategy `" << in_strategy->ToString()
                << "` and out_strategy `" << out_strategy->ToString() << "`.";
  return fg->NewCNodeInOrder({NewValueNode(prim::kPrimShard), NewValueNode(meta_op), NewValueNode(in_strategy),
                              NewValueNode(out_strategy), NewValueNode(MakeValue(kAscendDevice)),
                              NewValueNode(MakeValue(static_cast<int64_t>(0)))});
}
}  // namespace

EvalResultPtr PrimitiveToMetaEvaluator::EvalPrim(const AnalysisEnginePtr &engine,
                                                 const AbstractBasePtrList &args_abs_list, const ConfigPtr &,
                                                 const AnfNodeConfigPtr &out_conf) {
  // Convert Primitive to MetaImpl.
  MS_EXCEPTION_IF_NULL(out_conf);
  MS_EXCEPTION_IF_NULL(out_conf->node());
  auto cnode = out_conf->node()->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  ScopeGuard scope_guard(cnode->scope());
  TraceGuard trace_guard(MakeTraceInfo<TraceResolve>(cnode->debug_info()));
  const auto &fg = cnode->func_graph();
  MS_EXCEPTION_IF_NULL(fg);

  const auto &op_name = prim_->name();
  const auto &meta_op = prim::RegMetaImplFactory::GetInstance().CreateMetaImpl(op_name);
  MS_EXCEPTION_IF_NULL(meta_op);
  meta_op->set_prim(prim_);
  meta_op->set_manager(fg->manager());
  auto new_op_node = HandleShardForPrimitive(prim_, meta_op, fg);
  AnfNodePtrList op_inputs{new_op_node};
  constexpr size_t index_data = 1;
  (void)std::transform(cnode->weak_inputs().begin() + index_data, cnode->weak_inputs().end(),
                       std::back_inserter(op_inputs), ConvertWeakNode);
  AnfNodePtr new_cnode = fg->NewCNodeInOrder(op_inputs);
  new_cnode->set_debug_info(cnode->debug_info());
  auto new_conf = engine->MakeConfig(new_cnode, out_conf->context(), out_conf->func_graph());
  MS_LOG(DEBUG) << "Convert Primitive [" << op_name << "] to MetaImpl. Old node: " << cnode->DebugString()
                << "], new node: " << new_cnode->DebugString();
  return engine->ForwardConfig(out_conf, new_conf);
}

EvalResultPtr DoTransPrimitiveFunctionEvaluator::EvalPrim(const AnalysisEnginePtr &engine,
                                                          const AbstractBasePtrList &args_abs_list, const ConfigPtr &,
                                                          const AnfNodeConfigPtr &out_conf) {
  // For PrimitiveFunction generated by CreateInstance, its args, labels, signatures and
  // implicit conversion need to be processed.
  auto do_trans_prim_func = prim_->cast<prim::DoTransPrimitiveFunctionPtr>();
  MS_EXCEPTION_IF_NULL(do_trans_prim_func);
  auto prim_func = do_trans_prim_func->function();
  MS_EXCEPTION_IF_NULL(prim_func);
  auto cnode = out_conf->node()->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto fg = cnode->func_graph();
  MS_EXCEPTION_IF_NULL(fg);

  auto prim_name = prim_func->name();
  auto op_def = mindspore::ops::GetOpDef(prim_name);
  if (op_def == nullptr) {
    MS_LOG(INTERNAL_EXCEPTION) << "DoTransPrimitiveFunction only supports Primitive with OpDef, but got " << prim_name
                               << ".";
  }
  if (cnode->size() != args_abs_list.size() + 1) {
    MS_LOG(INTERNAL_EXCEPTION) << "For Operator[" << prim_name << "], the number of cnode inputs should be "
                               << (args_abs_list.size() + 1) << ", but got " << cnode->size()
                               << ".\nnode: " << cnode->DebugString();
  }
  // Handle primitive labels.
  AddLabelsToPrimitiveFunction(prim_func);
  // Handle primitive signatures.
  auto arg_signatures = op_def->signatures_;
  prim_func->set_signatures(arg_signatures);
  prim_func->set_has_signature(!arg_signatures.empty());
  // Get init args size.
  size_t init_args_size = 0;
  // All call args and init args should have been provided.
  size_t op_args_size = op_def->args_.size();
  if (op_args_size != args_abs_list.size()) {
    MS_EXCEPTION(TypeError) << "For Operator['" << prim_name
                            << "]', the number of inputs and init args (including default arguments) should be "
                            << op_args_size << ", but got " << args_abs_list.size() << ".";
  }
  for (size_t i = 0; i < op_args_size; ++i) {
    if (op_def->args_[i].as_init_arg_) {
      ++init_args_size;
    }
  }

  // Get init args and call args.
  AnfNodePtrList prim_init_arg_nodes;
  (void)std::transform(cnode->weak_inputs().cbegin() + cnode->size() - init_args_size, cnode->weak_inputs().cend(),
                       std::back_inserter(prim_init_arg_nodes), ConvertWeakNode);
  AnfNodePtrList prim_call_arg_nodes;
  (void)std::transform(cnode->weak_inputs().cbegin() + 1, cnode->weak_inputs().cend() - init_args_size,
                       std::back_inserter(prim_call_arg_nodes), ConvertWeakNode);

  auto args_pair = std::make_pair(prim_init_arg_nodes, prim_call_arg_nodes);
  auto new_cnode = CheckAndConvertPrimitiveArgs(prim_func, args_pair, engine, out_conf, false);
  auto new_conf = engine->MakeConfig(new_cnode, out_conf->context(), out_conf->func_graph());
  MS_LOG(INFO) << "Prim: " << prim_func->name() << ", " << cnode->DebugString() << ", " << new_cnode->DebugString();
  return engine->ForwardConfig(out_conf, new_conf);
}

EvalResultPtr PrimInstanceEvaluator::EvalPrim(const AnalysisEnginePtr &engine, const AbstractBasePtrList &args_abs_list,
                                              const ConfigPtr &, const AnfNodeConfigPtr &out_conf) {
  // Convert {Prim_Class, a, b}(x, y) to {DoTrans, x, y, a, b}.
  auto op_def = ops::GetOpDef(prim_name_);
  MS_EXCEPTION_IF_NULL(op_def);
  auto primitive = std::make_shared<Primitive>(prim_name_);
  auto do_trans_primfunc = std::make_shared<prim::DoTransPrimitiveFunction>(primitive);
  AnfNodePtrList new_inputs{NewValueNode(do_trans_primfunc)};
  // Add inputs: x, y.
  auto node = out_conf->node();
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto fg = cnode->func_graph();
  MS_EXCEPTION_IF_NULL(fg);
  constexpr auto input_start_index = 1;
  AnfNodePtrList init_args_list;
  AnfNodePtrList call_args_list;
  (void)std::copy(cnode->inputs().begin() + input_start_index, cnode->inputs().end(),
                  std::back_inserter(call_args_list));
  // Add init args: a, b.
  constexpr size_t op_index = 0;
  auto partial_node = cnode->input(op_index);
  MS_EXCEPTION_IF_NULL(partial_node);
  auto partial_cnode = partial_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(partial_cnode);
  auto fn_input = partial_cnode->input(0);
  size_t init_begin_index = 2;
  if (IsValueNode<prim::UnpackCall>(fn_input) || IsPrimitive(fn_input, prim::kPrimDoUnpackCall)) {
    auto instance_nodes = GetInputsAfterUnpackCall(partial_cnode, engine, out_conf);
    (void)std::copy(instance_nodes->inputs().begin() + init_begin_index, instance_nodes->inputs().end(),
                    std::back_inserter(init_args_list));
  } else {
    init_begin_index = 1;
    (void)std::copy(partial_cnode->inputs().begin() + init_begin_index, partial_cnode->inputs().end(),
                    std::back_inserter(init_args_list));
  }
  // Create new cnode.
  std::vector<ops::OpInputArg> op_call_args;
  std::vector<ops::OpInputArg> op_init_args;
  auto op_args = op_def->args_;
  for (const auto &op_arg : op_args) {
    if (op_arg.as_init_arg_) {
      (void)op_init_args.emplace_back(op_arg);
    } else {
      (void)op_call_args.emplace_back(op_arg);
    }
  }
  auto eval_func = [&engine, &out_conf](const AnfNodePtr &node) {
    AnfNodeConfigPtr config = engine->MakeConfig(node, out_conf->context(), out_conf->func_graph());
    MS_EXCEPTION_IF_NULL(config);
    const auto &eval_result = config->ObtainEvalResult();
    MS_EXCEPTION_IF_NULL(eval_result);
    return eval_result->abstract();
  };

  // Get default args.
  auto call_nodes = GeneratePrimitiveDefaultArgs(prim_name_, call_args_list, op_call_args, eval_func, fg);
  (void)std::copy(call_nodes.begin(), call_nodes.end(), std::back_inserter(new_inputs));
  auto init_nodes = GeneratePrimitiveDefaultArgs(prim_name_, init_args_list, op_init_args, eval_func, fg);
  (void)std::copy(init_nodes.begin(), init_nodes.end(), std::back_inserter(new_inputs));

  auto new_cnode = fg->NewCNodeInOrder(new_inputs);
  auto new_conf = engine->MakeConfig(new_cnode, out_conf->context(), out_conf->func_graph());
  MS_LOG(DEBUG) << "For Primitive[" << prim_name_ << "], using old node "
                << cnode->DebugString(AnfNode::DebugStringLevel::kLevel2) << " and instance node "
                << partial_cnode->DebugString(AnfNode::DebugStringLevel::kLevel2) << "to create new node "
                << new_cnode->DebugString(AnfNode::DebugStringLevel::kLevel2);
  return engine->ForwardConfig(out_conf, new_conf);
}

EvalResultPtr FunctionalEvaluator::EvalPrim(const AnalysisEnginePtr &engine, const AbstractBasePtrList &args_abs_list,
                                            const ConfigPtr &, const AnfNodeConfigPtr &out_conf) {
  MS_EXCEPTION_IF_NULL(out_conf);
  MS_EXCEPTION_IF_NULL(out_conf->node());
  auto cnode = out_conf->node()->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  ScopeGuard scope_guard(cnode->scope());
  TraceGuard trace_guard(MakeTraceInfo<TraceResolve>(cnode->debug_info()));
  auto func_graph = cnode->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  // Convert functional inputs.
  AnfNodePtrList inputs_list;
  constexpr size_t index_op = 0;
  constexpr size_t index_data = 1;
  auto op_node = cnode->input(index_op);
  if (IsPrimitiveCNode(op_node, prim::kPrimPartial)) {
    auto op_cnode = op_node->cast<CNodePtr>();
    constexpr size_t index_partial_data = 2;
    (void)std::transform(op_cnode->weak_inputs().cbegin() + index_partial_data, op_cnode->weak_inputs().cend(),
                         std::back_inserter(inputs_list), ConvertWeakNode);
  } else if (is_method_ && IsGetAttrNode(op_node)) {
    auto op_cnode = op_node->cast<CNodePtr>();
    (void)inputs_list.emplace_back(op_cnode->input(index_data));
  }
  (void)std::transform(cnode->weak_inputs().cbegin() + index_data, cnode->weak_inputs().cend(),
                       std::back_inserter(inputs_list), ConvertWeakNode);
  if (inputs_list.size() != args_abs_list.size()) {
    MS_LOG(INTERNAL_EXCEPTION) << "For Functional[" << name_ << "], inputs size " << inputs_list.size()
                               << " is not equal to abstract size " << args_abs_list.size();
  }
  AnfNodePtr new_cnode = nullptr;
  // Check if contains Any and convert to PyExecute node.
  if (ContainsAbstractAny(args_abs_list)) {
    new_cnode = prim::ConvertFunctionalToPyExecute(functional_, inputs_list, args_abs_list, cnode, is_method_);
  } else {
    auto eval_func = [&engine, &out_conf](const AnfNodePtr &node) {
      AnfNodeConfigPtr config = engine->MakeConfig(node, out_conf->context(), out_conf->func_graph());
      MS_EXCEPTION_IF_NULL(config);
      const auto &eval_result = config->ObtainEvalResult();
      MS_EXCEPTION_IF_NULL(eval_result);
      return eval_result->abstract();
    };
    new_cnode = prim::ConvertFunctionalToPrimitive(name_, inputs_list, args_abs_list, cnode, eval_func, is_method_);
  }
  MS_LOG(DEBUG) << "Convert Functional[" << name_
                << "]. Origin cnode: " << cnode->DebugString(AnfNode::DebugStringLevel::kLevel2)
                << ", new cnode: " << new_cnode->DebugString(AnfNode::DebugStringLevel::kLevel2);
  auto fn_conf = engine->MakeConfig(new_cnode, out_conf->context(), out_conf->func_graph());
  return engine->ForwardConfig(out_conf, fn_conf);
}

EvalResultPtr ConstexprEvaluator::EvalPrim(const AnalysisEnginePtr &engine, const AbstractBasePtrList &args_abs_list,
                                           const ConfigPtr &, const AnfNodeConfigPtr &out_conf) {
  // Consider all primitive implemented python infer() real use the tuple/list arguments.
  CheckSequenceArgumentForPythonPrimitive(prim_py_, args_abs_list);
  MS_EXCEPTION_IF_NULL(prim_py_);
  auto py_args = PreparePyInputs(args_abs_list);
  prim_py_->BeginRecordAddAttr();
  py::dict output = prim_py_->RunInfer(py_args);
  prim_py_->EndRecordAddAttr();
  if (output.contains("fn")) {
    // The inputs contain variable, the constexpr will run as graph.
    py::tuple values = output["fn"];
    if (values.empty()) {
      MS_LOG(EXCEPTION) << "Can not get origin function from constexpr.";
    }
    auto inner_val = parse::ParsePythonCode(values[0]);
    MS_EXCEPTION_IF_NULL(inner_val);
    auto inner_fg = dyn_cast<FuncGraph>(inner_val);
    MS_EXCEPTION_IF_NULL(inner_fg);
    MS_EXCEPTION_IF_NULL(out_conf);
    auto cur_graph = out_conf->func_graph();
    MS_EXCEPTION_IF_NULL(cur_graph);
    auto mng = cur_graph->manager();
    MS_EXCEPTION_IF_NULL(mng);
    inner_fg->set_manager(mng);
    auto out_node = out_conf->node();
    MS_EXCEPTION_IF_NULL(out_node);
    auto out_cnode = dyn_cast<CNode>(out_node);
    MS_EXCEPTION_IF_NULL(out_cnode);
    FuncGraphPtr func_graph = out_node->func_graph();
    MS_EXCEPTION_IF_NULL(func_graph);
    std::vector<AnfNodePtr> new_cnode_inputs = {NewValueNode(inner_fg)};
    const auto &out_cnode_inputs = out_cnode->weak_inputs();
    (void)std::transform(out_cnode_inputs.cbegin() + 1, out_cnode_inputs.cend(), std::back_inserter(new_cnode_inputs),
                         [](const auto &weak_node) {
                           const auto &node = weak_node.lock();
                           MS_EXCEPTION_IF_NULL(node);
                           return node;
                         });
    auto new_node = func_graph->NewCNodeInOrder(new_cnode_inputs);
    AnalysisEnginePtr eng = out_conf->engine();
    MS_EXCEPTION_IF_NULL(eng);
    AnfNodeConfigPtr fn_conf = eng->MakeConfig(new_node, out_conf->context(), out_conf->func_graph());
    return eng->ForwardConfig(out_conf, fn_conf);
  }
  // If all inputs are constant value, use python prim evaluator.
  // Ensure input arguments are evaluated.
  auto res_abstract = EvalUndeterminedArgs(args_abs_list);
  if (res_abstract != nullptr) {
    MS_LOG(DEBUG) << "PythonPrimEvaluator eval Undetermined";
    return res_abstract;
  }
  auto forbid_reuse = prim_py_->HasAttr(GRAPH_FLAG_FORBID_REUSE_RESULT);
  if (!forbid_reuse) {
    // Try to get infer result from evaluator cache.
    EvalResultPtr eval_result = evaluator_cache_mgr_->GetValue(args_abs_list);
    if (eval_result != nullptr) {
      MS_EXCEPTION_IF_NULL(eval_result->abstract());
      return std::make_shared<EvalResult>(eval_result->abstract()->Clone(), eval_result->attribute());
    }
  }
  const auto &added_attrs = prim_py_->evaluate_added_attrs();
  MS_LOG(DEBUG) << "Output type is " << py::str(output);
  auto res_abs = PyInferRes2Abstract(prim_py_, output);
  MS_EXCEPTION_IF_NULL(res_abs);
  MS_LOG(DEBUG) << "Python InferTensor result abstract: " << res_abs->ToString();
  EvalResultPtr eval_result = std::make_shared<EvalResult>(res_abs, std::make_shared<AttrValueMap>(added_attrs));
  evaluator_cache_mgr_->SetValue(args_abs_list, eval_result);
  return eval_result;
}

EvalResultPtr MakeTupleEvaluator::EvalPrim(const AnalysisEnginePtr &, const AbstractBasePtrList &args_abs_list,
                                           const ConfigPtr &, const AnfNodeConfigPtr &out_conf) {
  std::shared_ptr<AnfNodeWeakPtrList> sequence_nodes = std::make_shared<AnfNodeWeakPtrList>();
  auto abs = std::make_shared<AbstractTuple>(args_abs_list, sequence_nodes);
  if (out_conf != nullptr) {  // 'out_conf' maybe nullptr in PyNative mode.
    if (args_abs_list.empty()) {
      MS_EXCEPTION_IF_NULL(out_conf->node());
      MS_LOG(INFO) << "For MakeTuple, the inputs should not be empty. node: " << out_conf->node()->DebugString();
    }
    static const auto enable_eliminate_unused_element = (common::GetCompileConfig("ENABLE_DDE") != "0");
    if (enable_eliminate_unused_element) {
      auto flags = GetSequenceNodeElementsUseFlags(out_conf->node());
      if (flags == nullptr) {
        SetSequenceNodeElementsUseFlags(out_conf->node(), std::make_shared<std::vector<bool>>(args_abs_list.size()));
      }
      bool has_any = fallback::ContainsSequenceAnyType(abs);
      if (has_any) {
        SetSequenceElementsUseFlagsRecursively(abs, true);
      }
      (void)sequence_nodes->emplace_back(AnfNodeWeakPtr(out_conf->node()));
    }
  }
  auto res = std::make_shared<EvalResult>(abs, std::make_shared<AttrValueMap>());
  evaluator_cache_mgr_->SetValue(args_abs_list, res);
  // pass the need_unpack tag from the AnfNode to the abstract
  if (out_conf != nullptr) {
    auto node = out_conf->node();
    constexpr auto need_unpack_str = "need_unpack";
    auto need_unpack = node->user_data<bool>(need_unpack_str);
    if (need_unpack != nullptr && *need_unpack) {
      abs->SetData<bool>(need_unpack_str, std::make_shared<bool>(true));
    }
  }
  return res;
}

EvalResultPtr MakeListEvaluator::EvalPrim(const AnalysisEnginePtr &, const AbstractBasePtrList &args_abs_list,
                                          const ConfigPtr &, const AnfNodeConfigPtr &out_conf) {
  std::shared_ptr<AnfNodeWeakPtrList> sequence_nodes = std::make_shared<AnfNodeWeakPtrList>();
  auto abs = std::make_shared<AbstractList>(args_abs_list, sequence_nodes);
  if (out_conf != nullptr) {  // 'out_conf' maybe nullptr in PyNative mode.
    if (args_abs_list.empty()) {
      MS_LOG(INFO) << "For MakeList, the inputs should not be empty. node: " << out_conf->node()->DebugString();
    }
    static const auto enable_eliminate_unused_element = (common::GetCompileConfig("ENABLE_DDE") != "0");
    if (enable_eliminate_unused_element) {
      auto flags = GetSequenceNodeElementsUseFlags(out_conf->node());
      if (flags == nullptr) {
        SetSequenceNodeElementsUseFlags(out_conf->node(), std::make_shared<std::vector<bool>>(args_abs_list.size()));
      }

      (void)sequence_nodes->emplace_back(AnfNodeWeakPtr(out_conf->node()));
      if (fallback::ContainsSequenceAnyType(abs)) {
        SetSequenceElementsUseFlagsRecursively(abs, true);
      }
    }
  }
  MS_LOG(DEBUG) << "Generate python object for new value node.";
  if (fallback::EnableFallbackListDictInplace()) {
    py::object py_list_obj = fallback::GeneratePyObj(abs);
    if (py_list_obj.ptr() != nullptr) {
      fallback::AttachPyObjToAbs(abs, py_list_obj, true);
    }
  }
  auto res = std::make_shared<EvalResult>(abs, std::make_shared<AttrValueMap>());
  evaluator_cache_mgr_->SetValue(args_abs_list, res);
  return res;
}

AbstractBasePtr CreateRealAbstract(const TypePtr &preset_type, const BaseShapePtr &shape, const AnfNodePtr &node,
                                   const AbstractBasePtrList &args_abs_list) {
  AbstractBasePtr res = nullptr;
  if (preset_type->isa<Scalar>()) {
    res = std::make_shared<AbstractScalar>(preset_type);
  } else if (preset_type->isa<List>() || preset_type->isa<Tuple>()) {
    res = fallback::GenerateAbstractSequence(shape, preset_type, true);
  } else if (preset_type->isa<TensorType>() && !preset_type->isa<AnyType>()) {
    auto tensor_type = preset_type->cast_ptr<TensorType>();
    MS_EXCEPTION_IF_NULL(tensor_type);
    auto element = std::make_shared<abstract::AbstractScalar>(kValueAny, tensor_type->element());
    res = std::make_shared<abstract::AbstractTensor>(element, shape);
  } else {
    const auto any_abstract = std::make_shared<AbstractAny>();
    // If no annotation dtype, try to use unique tensor dtype.
    auto dtype = GetLocalArgsUniqueDtype(node, args_abs_list);
    if (dtype != nullptr) {
      MS_EXCEPTION_IF_NULL(any_abstract->element());
      any_abstract->element()->set_type(dtype);
      any_abstract->set_supposed_tensor_dtype(true);
    }
    res = any_abstract;
  }
  return res;
}

EvalResultPtr PyExecuteEvaluator::EvalPrim(const AnalysisEnginePtr &, const AbstractBasePtrList &args_abs_list,
                                           const ConfigPtr &, const AnfNodeConfigPtr &out_conf) {
  MS_EXCEPTION_IF_NULL(out_conf);
  if (args_abs_list.empty()) {
    MS_LOG(INTERNAL_EXCEPTION) << "'args_abs_list' should not be empty";
  }

  // Handle for DDE.
  for (size_t i = 0; i < args_abs_list.size(); ++i) {
    MS_EXCEPTION_IF_NULL(args_abs_list[i]);
    if (args_abs_list[i]->isa<abstract::AbstractSequence>()) {
      MS_LOG(DEBUG) << "Primitive \'PyExecute\' is consuming tuple/list arguments[" << i
                    << "]: " << args_abs_list[i]->ToString();
      SetSequenceElementsUseFlagsRecursively(args_abs_list[i], true);
    }
  }

  auto node = out_conf->node();
  MS_EXCEPTION_IF_NULL(node);
  MS_LOG(DEBUG) << "The current pyexecute node: " << node->DebugString();
  // Get the type parameter.
  MS_EXCEPTION_IF_NULL(args_abs_list[0]);
  ValuePtr script_value_track = args_abs_list[0]->GetValueTrack();
  MS_EXCEPTION_IF_NULL(script_value_track);
  auto script_obj = dyn_cast_ptr<StringImm>(script_value_track);
  if (script_obj == nullptr) {
    MS_LOG(INTERNAL_EXCEPTION) << "Cast value failed, not PyObjectWrapper: " << script_value_track->ToString() << ".";
  }

  // Make global and local parameters.
  const std::string &script = script_obj->value();
  // Call python script string.
  MS_LOG(DEBUG) << "Call script: " << script << ", args: " << args_abs_list;
  // Make abstract by type and shape.
  AbstractBasePtr res = nullptr;
  // Support Tensor annotation type. Add list and tuple here later.
  TypePtr dtype = nullptr;
  TypePtr type = GetAnnotationType(node, args_abs_list);
  if (type != nullptr && type->isa<TensorType>()) {
    dtype = type->cast<TensorTypePtr>()->element();
  }
  // Create output abstract.
  if (dtype != nullptr) {
    res = std::make_shared<AbstractTensor>(dtype, std::make_shared<Shape>(ShapeVector({Shape::kShapeRankAny})));
  } else if (fallback::HasRealType(node) && fallback::HasRealShape(node)) {
    const auto &preset_type = fallback::GetRealType<AnfNode, Type>(node);
    MS_LOG(DEBUG) << "preset_type: " << preset_type->ToString();
    const auto &shape = fallback::GetRealShape<AnfNode, BaseShape>(node);
    MS_LOG(DEBUG) << "shape: " << shape->ToString();
    res = CreateRealAbstract(preset_type, shape, node, args_abs_list);
  } else if (fallback::HasRealType(node) && fallback::GetRealType<AnfNode, Type>(node)->isa<NegligibleType>()) {
    res = std::make_shared<AbstractNegligible>();
  } else {
    const auto any_abstract = std::make_shared<AbstractAny>();
    // If no annotation dtype, try to use unique tensor dtype.
    dtype = GetLocalArgsUniqueDtype(node, args_abs_list);
    if (dtype != nullptr) {
      MS_EXCEPTION_IF_NULL(any_abstract->element());
      any_abstract->element()->set_type(dtype);
      any_abstract->set_supposed_tensor_dtype(true);
    }
    res = any_abstract;
  }

  // Set input real type and shape for caller.
  if (fallback::HasRealType(node)) {
    const auto &real_type = fallback::GetRealType<AnfNode, Type>(node);
    fallback::SetRealType<AbstractBase, Type>(res, real_type);
  }
  if (fallback::HasRealShape(node)) {
    const auto &real_shape = fallback::GetRealShape<AnfNode, BaseShape>(node);
    fallback::SetRealShape<AbstractBase, BaseShape>(res, real_shape);
  }
  auto infer_result = std::make_shared<EvalResult>(res, std::make_shared<AttrValueMap>());
  evaluator_cache_mgr_->SetValue(args_abs_list, infer_result);
  return infer_result;
}

namespace {
class PyInterpretEvaluator final : public TransitionPrimEvaluator {
 public:
  PyInterpretEvaluator() : TransitionPrimEvaluator("PyInterpretEvaluator") {}
  ~PyInterpretEvaluator() override = default;
  MS_DECLARE_PARENT(PyInterpretEvaluator, TransitionPrimEvaluator);
  EvalResultPtr EvalPrim(const AnalysisEnginePtr &, const AbstractBasePtrList &args_abs_list, const ConfigPtr &,
                         const AnfNodeConfigPtr &out_conf) override {
    if (args_abs_list.empty()) {
      MS_LOG(INTERNAL_EXCEPTION) << "'args_abs_list' should not be empty";
    }
    auto node = out_conf->node();
    MS_EXCEPTION_IF_NULL(node);
    MS_LOG(DEBUG) << "The current interpret node: " << node->DebugString();

    // If the interpret node contains FuncGraph node input, need to convert the Graph node to Interpreted object.
    AnfNodePtr converted_interpret_node = ConvertPyInterpretNode(node, args_abs_list);
    if (converted_interpret_node != nullptr) {
      AnalysisEnginePtr eng = out_conf->engine();
      MS_EXCEPTION_IF_NULL(eng);
      AnfNodeConfigPtr fn_conf = eng->MakeConfig(converted_interpret_node, out_conf->context(), out_conf->func_graph());
      return eng->ForwardConfig(out_conf, fn_conf);
    }

    non_const_err_ = false;
    check_list_dict_inplace_ =
      node->has_user_data(fallback::kCheckListDictInplace) && *node->user_data<bool>(fallback::kCheckListDictInplace);

    constexpr size_t script_index = 0;
    const std::string &script = GetScriptStr(args_abs_list[script_index]);
    // Make global and local parameters.
    py::tuple params = MakeParameters(args_abs_list, script, node);
    // Would convert PyInterpret to PyExecute then.
    if (non_const_err_ || fallback::GetJitAnnotationSideEffectFromComment(node)) {
      // Make abstract by type and shape.
      AbstractBasePtr res = nullptr;
      // Support Tensor annotation type. Add list and tuple here later.
      TypePtr dtype = nullptr;
      TypePtr type = GetAnnotationType(node, args_abs_list);
      if (type != nullptr && type->isa<TensorType>()) {
        dtype = type->cast<TensorTypePtr>()->element();
      }
      // Create output abstract.
      if (dtype != nullptr) {
        res = std::make_shared<AbstractTensor>(dtype, std::make_shared<Shape>(ShapeVector({Shape::kShapeRankAny})));
      } else {
        const auto any_abstract = std::make_shared<AbstractAny>();
        // If no annotation dtype, try to use unique tensor dtype.
        dtype = GetLocalArgsUniqueDtype(node, args_abs_list);
        if (dtype != nullptr) {
          MS_EXCEPTION_IF_NULL(any_abstract->element());
          any_abstract->element()->set_type(dtype);
          any_abstract->set_supposed_tensor_dtype(true);
        }
        res = any_abstract;
      }
      auto infer_result = std::make_shared<EvalResult>(res, std::make_shared<AttrValueMap>());
      evaluator_cache_mgr_->SetValue(args_abs_list, infer_result);
      return infer_result;
    }

    // Call python script string.
    MS_LOG(DEBUG) << "Call script: " << script << ", params: " << py::str(params);
    auto obj = parse::data_converter::CallPythonScript(py::str(script), params);
    if (py::isinstance<py::none>(obj)) {
      AbstractBasePtr res = std::make_shared<abstract::AbstractNone>();
      auto infer_result = std::make_shared<EvalResult>(res, nullptr);
      evaluator_cache_mgr_->SetValue(args_abs_list, infer_result);
      return infer_result;
    }

    ValuePtr converted_val = nullptr;
    bool converted = false;
    // converted_val could be a InterpretedObject.
    if (node->has_user_data("__keep_metafg_obj_flag__")) {
      converted_val = std::make_shared<parse::InterpretedObject>(obj);
      converted = true;
    } else {
      converted = parse::ConvertData(obj, &converted_val, true);
    }
    if (!converted) {
      MS_LOG(INTERNAL_EXCEPTION) << "Convert the python object failed";
    }
    MS_EXCEPTION_IF_NULL(converted_val);
    auto fg = node->func_graph();
    MS_EXCEPTION_IF_NULL(fg);
    auto mng = fg->manager();
    MS_EXCEPTION_IF_NULL(mng);
    AddManagerForFuncGraphValue(converted_val, mng);
    if (converted_val->isa<tensor::Tensor>() && HasConstArgAttr(obj)) {
      MS_LOG(WARNING) << "The tensor " << converted_val->ToString()
                      << " which is not used for network input argument should not be set const.";
    }
    if (converted_val->isa<parse::InterpretedObject>()) {
      const auto interpreted_value = dyn_cast<parse::InterpretedObject>(converted_val);
      MS_LOG(DEBUG) << "The InterpretedObject(" << converted_val->ToString() << ") is converted by PyInterpret"
                    << " node: " << node->DebugString();
      interpreted_value->set_has_converted(true);
    }

    AbstractBasePtr res = ToAbstract(converted_val, AnalysisContext::DummyContext(), out_conf);
    if (!converted_val->ContainsValueAny()) {
      AnfNodePtr value_node = NewValueNode(converted_val);
      if (res->isa<abstract::AbstractTensor>()) {
        res->set_user_data("constant_tensor", std::make_shared<bool>(true));
      }
      value_node->set_abstract(res);
      value_node->set_debug_info(node->debug_info());
      AnalysisEnginePtr eng = out_conf->engine();
      MS_EXCEPTION_IF_NULL(eng);
      AnfNodeConfigPtr fn_conf = eng->MakeConfig(value_node, out_conf->context(), out_conf->func_graph());
      return eng->ForwardConfig(out_conf, fn_conf);
    }

    auto infer_result = std::make_shared<EvalResult>(res, std::make_shared<AttrValueMap>());
    evaluator_cache_mgr_->SetValue(args_abs_list, infer_result);
    return infer_result;
  }

  void AddManagerForFuncGraphValue(const ValuePtr &val, const FuncGraphManagerPtr &mng) const {
    // mng has been checked before using.
    MS_EXCEPTION_IF_NULL(val);
    if (val->isa<ValueSequence>()) {
      auto val_seq = val->cast<ValueSequencePtr>();
      const auto &values = val_seq->value();
      std::for_each(values.begin(), values.end(),
                    [this, &mng](const ValuePtr &e) { AddManagerForFuncGraphValue(e, mng); });
      return;
    }
    if (val->isa<ValueDictionary>()) {
      auto val_dict = val->cast<ValueDictionaryPtr>();
      const auto &values = val_dict->value();
      std::for_each(values.begin(), values.end(), [this, &mng](const std::pair<ValuePtr, ValuePtr> &pair) {
        // Key for value dictionary can not have function graph.
        AddManagerForFuncGraphValue(pair.second, mng);
      });
      return;
    }
    if (val->isa<FuncGraph>()) {
      auto val_fg = val->cast<FuncGraphPtr>();
      if (val_fg->manager() == nullptr) {
        mng->AddFuncGraph(val_fg);
        val_fg->set_manager(mng);
      }
    }
    return;
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
      MS_EXCEPTION_IF_NULL(local_abs);
      const auto &local_abs_val = local_abs->BuildValue();
      MS_EXCEPTION_IF_NULL(local_abs_val);
      MS_EXCEPTION_IF_NULL(name);
      auto py_data_name = py::str(ValueToPyData(name->BuildValue()));
      bool has_python_obj = check_list_dict_inplace_ && fallback::HasObjInExtraInfoHolder(local_abs);
      if (local_abs_val->ContainsValueAny() || has_python_obj) {
        const auto allow_fallback_runtime = (fallback::GetJitSyntaxLevel() == kLax);
        if (allow_fallback_runtime) {
          MS_LOG(INFO) << "When using JIT Fallback to handle script '" << script
                       << "', the inputs should be constant, but found variable '" << py_data_name
                       << "' to be nonconstant. To convert to PyExecute() afterwards";
          non_const_err_ = true;
        } else {
          MS_EXCEPTION(ValueError) << "When handling script '" << script << " in graph mode"
                                   << "', the inputs should be constant, but found variable '" << py_data_name
                                   << "' to be nonconstant. Try to set jit_syntax_level to LAX.";
        }
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
      MS_EXCEPTION_IF_NULL(element_name);
      MS_EXCEPTION_IF_NULL(element_abs);
      const auto &fn_py_obj = fallback::GetPyObjForFuncGraphAbstractClosure(element_abs);
      if (!py::isinstance<py::none>(fn_py_obj)) {
        (*global_params_dict)[ValueToPyData(element_name->BuildValue())] = fn_py_obj;
        MS_LOG(DEBUG) << "Found global python function object for " << element_name << ", add it to global dict.";
      }
    }
    return;
  }

  py::tuple MakeParameters(const AbstractBasePtrList &args_abs_list, const std::string &script,
                           const AnfNodePtr &node) const {
    constexpr int params_size = 3;
    auto args_size = std::count_if(args_abs_list.begin(), args_abs_list.end(),
                                   [](const AbstractBasePtr &arg) -> bool { return !arg->isa<AbstractMonad>(); });
    if (params_size != args_size) {
      MS_LOG(INTERNAL_EXCEPTION) << "Unexpected params_size: " << params_size
                                 << ", not equal to arguments.size: " << args_abs_list.size();
    }
    // The first argument is script string, ignore it.
    auto params = py::tuple(params_size - 1);

    // Make the global parameters.
    constexpr size_t global_index = 1;
    auto global_abs = args_abs_list[global_index];
    const py::object &global_params_dict = GetGlobalObject(global_abs);

    // Make the local parameters.
    constexpr size_t local_index = 2;
    auto local_dict = dyn_cast<AbstractDictionary>(args_abs_list[local_index]);  // Local parameters dict.
    if (local_dict == nullptr) {
      MS_EXCEPTION_IF_NULL(args_abs_list[local_index]);
      MS_LOG(INTERNAL_EXCEPTION) << "The third argument should be a dictionary, but got "
                                 << args_abs_list[local_index]->ToString();
    }
    auto filtered_local_dict = FilterParameters(local_dict);
    MS_LOG(DEBUG) << "arg_2, local_dict: " << local_dict->ToString()
                  << ", filtered_local_dict: " << filtered_local_dict->ToString();
    ValuePtr local_dict_value = filtered_local_dict->BuildValue();
    MS_EXCEPTION_IF_NULL(local_dict_value);
    py::dict local_params_dict = ReCheckLocalDict(filtered_local_dict);
    MS_LOG(DEBUG) << "arg_2, python local_params_dict: " << local_dict_value->ToString() << " -> "
                  << py::str(local_params_dict);
    CheckInterpretInput(filtered_local_dict, script);

    // Check if the value node in local dict is free variable.
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    const size_t local_dict_index = 3;
    auto local_dict_node = cnode->input(local_dict_index);
    MS_EXCEPTION_IF_NULL(local_dict_node);
    const size_t local_dict_key_index = 1;
    const size_t local_dict_value_index = 2;
    auto local_dict_cnode = local_dict_node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(local_dict_cnode);
    auto local_key_node = local_dict_cnode->input(local_dict_key_index);
    auto local_value_node = local_dict_cnode->input(local_dict_value_index);
    auto func = cnode->func_graph();
    if (local_key_node->isa<CNode>() && local_value_node->isa<CNode>()) {
      auto local_value_cnode = local_value_node->cast<CNodePtr>();
      size_t local_value_num = local_value_cnode->inputs().size();
      for (size_t index = 1; index < local_value_num; ++index) {
        auto key_input = local_key_node->cast<CNodePtr>()->input(index);
        auto value_input = local_value_node->cast<CNodePtr>()->input(index);
        if (IsValueNode<StringImm>(key_input)) {
          const auto &key_input_str = GetValue<std::string>(GetValueNode(key_input));
          // If exist free variable, it need push into global dict.
          if (value_input->func_graph() != nullptr && value_input->func_graph() != func) {
            global_params_dict[py::str(key_input_str)] = local_params_dict[py::str(key_input_str)];
          }
        }
      }
    }
    params[0] = global_params_dict;
    params[1] = local_params_dict;
    return params;
  }

  py::dict ReCheckLocalDict(const AbstractDictionaryPtr &filtered_local_dict) const {
    const auto &keys_values = filtered_local_dict->elements();
    py::dict local_params_dict;
    for (auto &key_value : keys_values) {
      MS_EXCEPTION_IF_NULL(key_value.second);
      ValuePtr element_value = key_value.second->BuildValue();
      MS_EXCEPTION_IF_NULL(element_value);
      auto py_data = ValueToPyData(element_value);
      MS_EXCEPTION_IF_NULL(key_value.first);
      local_params_dict[ValueToPyData(key_value.first->BuildValue())] = py_data;
    }
    return local_params_dict;
  }

  AbstractDictionaryPtr FilterParameters(const AbstractDictionaryPtr &abstract_dict) const {
    MS_EXCEPTION_IF_NULL(abstract_dict);
    std::vector<AbstractElementPair> kv;
    const auto &keys_values = abstract_dict->elements();
    // Filter out the element of Function type.
    (void)std::copy_if(keys_values.cbegin(), keys_values.cend(), std::back_inserter(kv),
                       [](const AbstractElementPair &item) {
                         MS_EXCEPTION_IF_NULL(item.second);
                         return (!item.second->isa<abstract::AbstractFunction>());
                       });
    return std::make_shared<AbstractDictionary>(kv);
  }

  bool HasConstArgAttr(const py::object &obj) const {
    constexpr char const_arg_attr[] = "const_arg";
    return py::hasattr(obj, const_arg_attr) && py::cast<bool>(py::getattr(obj, const_arg_attr));
  }

  std::string GetScriptStr(const AbstractBasePtr &abs) const {
    // When PyInterpret node is built in python, the value of script abstract should be StringImm.
    // Otherwise, the value of script should be Script type.
    MS_EXCEPTION_IF_NULL(abs);
    ValuePtr value_track = abs->GetValueTrack();
    MS_EXCEPTION_IF_NULL(value_track);
    if (value_track->isa<parse::Script>()) {
      auto script_value_track = dyn_cast_ptr<parse::Script>(value_track);
      return script_value_track->script();
    }
    if (!value_track->isa<StringImm>()) {
      MS_INTERNAL_EXCEPTION(TypeError) << "Wrong script type for PyInterpret node, script abs: " << abs->ToString();
    }
    return value_track->ToString();
  }

  py::object GetGlobalObject(const AbstractBasePtr &abs) const {
    MS_EXCEPTION_IF_NULL(abs);
    if (!abs->isa<abstract::AbstractScalar>() && !abs->isa<abstract::AbstractDictionary>()) {
      MS_INTERNAL_EXCEPTION(TypeError) << "The second argument should be a scalar(InterpretedObject) or dictionary, "
                                       << "but got " << abs->ToString();
    }
    auto val = abs->BuildValue();
    MS_EXCEPTION_IF_NULL(val);
    AbstractDictionaryPtr global_dict = nullptr;
    // Some functions in global_dict are not used and will be released early,
    // resulting in the func_graph pointer in AbstractClosure being released.
    ValuePtr globals_converted_value = nullptr;
    py::object global_params_dict;
    if (abs->isa<abstract::AbstractDictionary>()) {
      global_dict = abs->cast<abstract::AbstractDictionaryPtr>();
      auto filtered_global_dict = FilterParameters(global_dict);
      global_params_dict = ValueToPyData(filtered_global_dict->BuildValue());
    } else {
      auto global_dict_interpreted = dyn_cast<parse::InterpretedObject>(val);
      MS_EXCEPTION_IF_NULL(global_dict_interpreted);
      const py::object &global_params_dict_obj = global_dict_interpreted->obj();
      if (!parse::ConvertData(global_params_dict_obj, &globals_converted_value)) {
        MS_LOG(INTERNAL_EXCEPTION) << "Convert data failed";
      }
      MS_EXCEPTION_IF_NULL(globals_converted_value);
      // Filter global parameters dict.
      global_dict = dyn_cast<AbstractDictionary>(globals_converted_value->ToAbstract());
      if (global_dict == nullptr) {
        MS_LOG(INTERNAL_EXCEPTION) << "The second argument should be a dictionary, but got "
                                   << globals_converted_value->ToAbstract()->ToString();
      }
      auto filtered_global_dict = FilterParameters(global_dict);
      MS_LOG(DEBUG) << "arg_1, global_dict: " << global_dict->ToString()
                    << ", filtered_global_dict: " << filtered_global_dict->ToString();
      ValuePtr global_dict_value = filtered_global_dict->BuildValue();
      global_params_dict = ValueToPyData(global_dict_value);
    }
    // Add filtered global python function to global_params_dict.
    AddGlobalPythonFunction(global_dict, &global_params_dict);
    return global_params_dict;
  }

  AnfNodePtr ConvertLocalValueInputNode(const AnfNodePtr &local_node, const AbstractBasePtr &local_abs) const {
    MS_EXCEPTION_IF_NULL(local_node);
    MS_EXCEPTION_IF_NULL(local_abs);
    AnfNodePtr ret_node = nullptr;
    // Not consider AbstractDictionary scene yet.
    if (local_abs->isa<abstract::AbstractSequence>() &&
        IsOneOfPrimitiveCNode(local_node, {prim::kPrimMakeTuple, prim::kPrimMakeList})) {
      auto local_cnode = local_node->cast<CNodePtr>();
      auto local_abs_seq = local_abs->cast<abstract::AbstractSequencePtr>();
      if (local_cnode->size() - 1 != local_abs_seq->size()) {
        MS_LOG(INTERNAL_EXCEPTION) << "For node: " << local_node->DebugString() << ", input size is "
                                   << local_cnode->size() << " and abstract size is " << local_abs_seq->size()
                                   << ". Size not matched.";
      }
      const auto &local_elements_abs = local_abs_seq->elements();
      AnfNodePtrList new_inputs;
      (void)new_inputs.emplace_back(local_cnode->input(0));
      for (size_t i = 1; i < local_cnode->size(); ++i) {
        (void)new_inputs.emplace_back(ConvertLocalValueInputNode(local_cnode->input(i), local_elements_abs[i - 1]));
      }
      auto fg = local_cnode->func_graph();
      MS_EXCEPTION_IF_NULL(fg);
      ret_node = fg->NewCNode(new_inputs);
    } else {
      auto py_obj = fallback::GetPyObjForFuncGraphAbstractClosure(local_abs);
      if (py::isinstance<py::none>(py_obj)) {
        return local_node;
      }
      ret_node = NewValueNode(std::make_shared<parse::InterpretedObject>(py_obj));
    }
    MS_EXCEPTION_IF_NULL(ret_node);
    ret_node->set_debug_info(local_node->debug_info());
    return ret_node;
  }

  AnfNodePtr ConvertPyInterpretNode(const AnfNodePtr &node, const AbstractBasePtrList &args_abs_list) const {
    MS_EXCEPTION_IF_NULL(node);
    // Ensure the same node only check local dict once.
    if (node->has_user_data(fallback::kLocalDictCheck) && *node->user_data<bool>(fallback::kLocalDictCheck)) {
      return nullptr;
    }
    node->set_user_data(fallback::kLocalDictCheck, std::make_shared<bool>(true));
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    constexpr size_t interpret_min_len = 4;
    if (cnode->size() < interpret_min_len) {
      MS_LOG(INTERNAL_EXCEPTION) << "The minimum input number for PyInterpret node should be " << interpret_min_len
                                 << " but got " << cnode->size();
    }
    if (args_abs_list.size() < interpret_min_len - 1) {
      MS_LOG(INTERNAL_EXCEPTION) << "The minimum number for PyInterpret input abstract should be "
                                 << (interpret_min_len - 1) << " but got " << args_abs_list.size();
    }
    constexpr size_t local_index = 3;
    auto local_node = cnode->input(local_index);
    auto local_node_abs = args_abs_list[local_index - 1];
    MS_EXCEPTION_IF_NULL(local_node);
    MS_EXCEPTION_IF_NULL(local_node_abs);
    if (!IsPrimitiveCNode(local_node, prim::kPrimMakeDict)) {
      return nullptr;
    }
    auto local_cnode = local_node->cast<CNodePtr>();
    constexpr size_t make_dict_len = 3;
    if (local_cnode->size() != make_dict_len) {
      MS_LOG(INTERNAL_EXCEPTION) << "Make dict mode input size should be " << make_dict_len << " but got "
                                 << local_cnode->size();
    }

    const auto &check_abs_function = [](const AbstractBasePtr &input) {
      std::function<bool(const AbstractBasePtr &)> check_abs_function_inner;
      check_abs_function_inner = [&](const AbstractBasePtr &abs) {
        MS_EXCEPTION_IF_NULL(abs);
        if (abs->isa<abstract::AbstractSequence>()) {
          auto abs_seq = abs->cast<abstract::AbstractSequencePtr>();
          const auto &elements = abs_seq->elements();
          return std::any_of(elements.begin(), elements.end(),
                             [check_abs_function_inner](const AbstractBasePtr &inner_abs) {
                               return check_abs_function_inner(inner_abs);
                             });
        }
        if (abs->isa<abstract::AbstractDictionary>()) {
          auto abs_dict = abs->cast<abstract::AbstractDictionaryPtr>();
          const auto elements = abs_dict->elements();
          return std::any_of(elements.begin(), elements.end(),
                             [check_abs_function_inner](const abstract::AbstractElementPair &inner_abs) {
                               // Dictionary key can not be abstract function, no need to check.
                               return check_abs_function_inner(inner_abs.second);
                             });
        }
        return abs->isa<abstract::FuncGraphAbstractClosure>();
      };
      return check_abs_function_inner(input);
    };

    if (!check_abs_function(local_node_abs)) {
      return nullptr;
    }
    auto local_node_abs_dict = local_node_abs->cast<abstract::AbstractDictionaryPtr>();
    MS_EXCEPTION_IF_NULL(local_node_abs_dict);
    const auto &elements_pair = local_node_abs_dict->elements();
    std::vector<abstract::AbstractBasePtr> element_abs{};
    (void)std::transform(elements_pair.begin(), elements_pair.end(), std::back_inserter(element_abs),
                         [](const AbstractElementPair &pairs) { return pairs.second; });
    auto local_value_abs = std::make_shared<abstract::AbstractTuple>(element_abs);
    constexpr size_t value_index = 2;
    auto local_value_node = local_cnode->input(value_index);
    auto new_local_value_node = ConvertLocalValueInputNode(local_value_node, local_value_abs);
    std::vector<AnfNodePtr> new_local_node_inputs;
    for (size_t i = 0; i < value_index; ++i) {
      new_local_node_inputs.push_back(local_cnode->input(i));
    }
    new_local_node_inputs.push_back(new_local_value_node);
    auto fg = node->func_graph();
    MS_EXCEPTION_IF_NULL(fg);
    auto new_local_cnode = fg->NewCNode(new_local_node_inputs);
    new_local_cnode->set_debug_info(local_cnode->debug_info());
    std::vector<AnfNodePtr> new_cnode_inputs;
    for (size_t i = 0; i < local_index; ++i) {
      new_cnode_inputs.push_back(cnode->input(i));
    }
    new_cnode_inputs.push_back(new_local_cnode);
    for (size_t i = local_index + 1; i < cnode->size(); ++i) {
      new_cnode_inputs.push_back(cnode->input(i));
    }
    auto new_cnode = fg->NewCNode(new_cnode_inputs);
    new_cnode->set_debug_info(cnode->debug_info());
    new_cnode->set_user_data(fallback::kLocalDictCheck, std::make_shared<bool>(true));
    return new_cnode;
  }

 private:
  mutable bool non_const_err_{false};
  mutable bool check_list_dict_inplace_{false};
};

class EmbedEvaluator final : public SymbolicPrimEvaluator {
 public:
  EmbedEvaluator() : SymbolicPrimEvaluator("EmbedEvaluator") {}
  ~EmbedEvaluator() override = default;
  MS_DECLARE_PARENT(EmbedEvaluator, SymbolicPrimEvaluator);
  EvalResultPtr EvalPrim(const ConfigPtrList &args_conf_list) override {
    // arg: free variable to be embedded
    if (args_conf_list.size() != 1) {
      MS_LOG(INTERNAL_EXCEPTION) << "EmbedEvaluator requires 1 parameter, but got " << args_conf_list.size();
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
  MS_EXCEPTION_IF_NULL(root_g);
  for (auto &param_node : root_g->parameters()) {
    auto param = param_node->cast<ParameterPtr>();
    if (param != nullptr && param->name() == name) {
      return param;
    }
  }
  return nullptr;
}

class RefToEmbedEvaluator final : public SymbolicPrimEvaluator {
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
    bool embed_is_weight = false;
    if (node_conf->node() != nullptr && node_conf->node()->isa<Parameter>()) {
      auto param = node_conf->node()->cast_ptr<Parameter>();
      MS_EXCEPTION_IF_NULL(param);
      embed_is_weight = param->has_default();
    }
    auto refkey = ref_key_value->cast_ptr<StringImm>();
    if (refkey == nullptr || !embed_is_weight) {
      auto res = std::make_shared<AbstractScalar>(type);
      return std::make_shared<EvalResult>(res, std::make_shared<AttrValueMap>());
    }

    std::string name = refkey->value();
    MS_EXCEPTION_IF_NULL(node_conf->node());
    if (node_conf->node()->func_graph() == nullptr) {
      MS_LOG(INTERNAL_EXCEPTION) << "Should not evaluate a ValueNode, node: " << node_conf->node()->DebugString();
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

class GetAttrEvaluator final : public TransitionPrimEvaluator {
 public:
  GetAttrEvaluator() : TransitionPrimEvaluator("GetAttrEvaluator") {}
  ~GetAttrEvaluator() override = default;
  MS_DECLARE_PARENT(GetAttrEvaluator, TransitionPrimEvaluator);
  EvalResultPtr ProcessRecordWaitMothed(const AnalysisEnginePtr &engine, const AbstractBasePtrList &args_abs_list,
                                        const AnfNodeConfigPtr &out_conf) {
    constexpr auto attr_index = 1;
    auto attr_abs = args_abs_list[attr_index];
    MS_EXCEPTION_IF_NULL(attr_abs);
    // value.attr
    // GetAttr(value, attr)
    auto attr_value = attr_abs->BuildValue();
    MS_EXCEPTION_IF_NULL(attr_value);
    auto attr_str = GetValue<std::string>(attr_value);
    // Check if the value is Event, only support event.record() and event.wait()
    if (attr_str != "record" && attr_str != "wait") {
      return nullptr;
    }
    constexpr auto kValueIndex = 0;
    auto value_abs = args_abs_list[kValueIndex];
    MS_EXCEPTION_IF_NULL(value_abs);
    if (value_abs->isa<abstract::AbstractEvent>()) {
      return nullptr;
    }
    auto getattr_node = out_conf->node();
    getattr_node->set_user_data<bool>("fake_return", std::make_shared<bool>(true));
    MS_LOG(DEBUG) << "mark fake_return for getattr_node:" << getattr_node->DebugString();
    auto original_var = getattr_node->user_data<AnfNode>("original_var");
    if (original_var == nullptr) {
      return nullptr;
    }
    MS_LOG(DEBUG) << "original_var:" << original_var->DebugString();
    auto get_attr_inputs = getattr_node->cast<CNodePtr>()->inputs();
    auto func_graph = getattr_node->func_graph();
    MS_EXCEPTION_IF_NULL(func_graph);
    constexpr auto kReplaceIndex = 1;
    if (get_attr_inputs[kReplaceIndex] != original_var) {
      CNodePtr new_cnode = func_graph->NewCNode(get_attr_inputs);
      new_cnode->set_input(kReplaceIndex, original_var);
      new_cnode->set_debug_info(getattr_node->debug_info());
      AnfNodeConfigPtr fn_conf = engine->MakeConfig(new_cnode, out_conf->context(), out_conf->func_graph());
      return engine->ForwardConfig(out_conf, fn_conf);
    }
    return nullptr;
  }

  EvalResultPtr EvalPrim(const AnalysisEnginePtr &engine, const AbstractBasePtrList &args_abs_list,
                         const ConfigPtr &in_conf0, const AnfNodeConfigPtr &out_conf) override {
    constexpr auto args_min_size = 2;
    constexpr auto args_max_size = 3;
    const auto args_size = args_abs_list.size();
    if (args_size != args_min_size && args_size != args_max_size) {
      MS_LOG(EXCEPTION) << "For Primitive GetAttr, the input size should be " << args_min_size << " or "
                        << args_max_size << ", but got size: " << args_size;
    }
    auto res_abstract = EvalUndeterminedArgs(args_abs_list);
    if (res_abstract != nullptr) {
      return res_abstract;
    }

    constexpr auto attr_index = 1;
    auto attr_abs = args_abs_list[attr_index];
    MS_EXCEPTION_IF_NULL(attr_abs);

    auto attr_abs_type = attr_abs->BuildType();
    MS_EXCEPTION_IF_NULL(attr_abs_type);
    auto type_id = attr_abs_type->type_id();
    if (type_id != TypeId::kObjectTypeString) {
      MS_EXCEPTION(TypeError) << "getattr(): attribute name must be string but got: " << TypeIdToString(type_id);
    }
    EvalResultPtr res = nullptr;
    if (bound_node() != nullptr) {
      TraceGuard trace_guard(MakeTraceInfo<TraceResolve>(bound_node()->debug_info()));
      res = ProcessRecordWaitMothed(engine, args_abs_list, out_conf);
    } else {
      res = ProcessRecordWaitMothed(engine, args_abs_list, out_conf);
    }
    if (res != nullptr) {
      return res;
    }

    if (bound_node() != nullptr) {
      TraceGuard trace_guard(MakeTraceInfo<TraceResolve>(bound_node()->debug_info()));
      res = StaticGetter(engine, args_abs_list, in_conf0, out_conf);
    } else {
      res = StaticGetter(engine, args_abs_list, in_conf0, out_conf);
    }
    // Don't lookup from cache, as different out_conf with same node but different context
    // may add different entry to anfnode_config_map, like getattr primitive.
    evaluator_cache_mgr_->SetValue(args_abs_list, res);
    return res;
  }
};

class SetAttrEvaluator final : public TransitionPrimEvaluator {
 public:
  SetAttrEvaluator() : TransitionPrimEvaluator("SetAttrEvaluator") {}
  ~SetAttrEvaluator() override = default;
  MS_DECLARE_PARENT(SetAttrEvaluator, TransitionPrimEvaluator);
  EvalResultPtr EvalPrim(const AnalysisEnginePtr &engine, const AbstractBasePtrList &args_abs_list, const ConfigPtr &,
                         const AnfNodeConfigPtr &out_conf) override {
    constexpr size_t min_args_size = 3;
    constexpr size_t max_args_size = 4;
    size_t args_size = args_abs_list.size();
    if (args_size != min_args_size && args_size != max_args_size) {
      MS_LOG(EXCEPTION) << "For Primitive SetAttr, the input size should be " << min_args_size << " or "
                        << max_args_size << ", but got size: " << args_size;
    }
    // target.attr = assigned
    // SetAttr(target, attr, assigned)
    // self.cls.record()  self.cls = self.cls.record()
    // SetAttr(self.cls, attr, assigned)
    auto set_attr_node = out_conf->node();
    MS_EXCEPTION_IF_NULL(set_attr_node);
    auto set_attr_cnode = set_attr_node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(set_attr_cnode);
    constexpr size_t kAssignedIndex = 3;
    auto assigned = set_attr_cnode->input(kAssignedIndex);
    if (assigned->isa<CNode>()) {
      auto assigned_cnode = assigned->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(assigned_cnode);
      auto assigned_input = assigned_cnode->input(0);
      auto fake_return = assigned_input->user_data<bool>("fake_return");
      if (fake_return != nullptr && *fake_return) {
        MS_LOG(DEBUG) << "assigned_input has fake_return: " << assigned_input->DebugString();
        auto new_node = NewValueNode(std::make_shared<None>());
        AnfNodeConfigPtr fn_conf = engine->MakeConfig(new_node, out_conf->context(), out_conf->func_graph());
        return engine->ForwardConfig(out_conf, fn_conf);
      }
    }

    auto res_abstract = EvalUndeterminedArgs(args_abs_list);
    if (res_abstract != nullptr) {
      return res_abstract;
    }
    auto res = InterpretTensorSetDataNode(args_abs_list, out_conf);
    if (res != nullptr) {
      return res;
    }
    return InterpretSetAttrNode(args_abs_list, out_conf);
  }
};

class ResolveEvaluator final : public TransitionPrimEvaluator {
 public:
  ResolveEvaluator() : TransitionPrimEvaluator("ResolveEvaluator") {}
  ~ResolveEvaluator() override = default;
  MS_DECLARE_PARENT(ResolveEvaluator, TransitionPrimEvaluator);
  EvalResultPtr EvalPrim(const AnalysisEnginePtr &engine, const AbstractBasePtrList &args_abs_list,
                         const ConfigPtr &in_conf0, const AnfNodeConfigPtr &out_conf) override {
    constexpr auto resolve_args_size = 2;       // (namespace, symbol)
    constexpr auto resolve_with_args_size = 3;  // (namespace, symbol, arguments)
    // Inputs: namespace, symbol
    if (args_abs_list.size() != resolve_args_size && args_abs_list.size() != resolve_with_args_size) {
      MS_LOG(EXCEPTION) << "Expected args_abs_list size is 2 or 3, but has size: " << args_abs_list.size();
    }
    EvalResultPtr res = nullptr;
    if (bound_node() != nullptr) {
      TraceGuard trace_guard(MakeTraceInfo<TraceResolve>(bound_node()->debug_info()));
      res = StaticGetter(engine, args_abs_list, in_conf0, out_conf);
    } else {
      res = StaticGetter(engine, args_abs_list, in_conf0, out_conf);
    }
    return res;
  }
};

py::object GetPythonObject(const AbstractBasePtr &arg_class_type) {
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
    MS_LOG(INTERNAL_EXCEPTION) << "Cast value failed, not PyObjectWrapper: " << value_track->ToString() << ".";
  }
  if (!type_obj->isa<parse::ClassType>() && !type_obj->isa<parse::MsClassObject>()) {
    MS_LOG(EXCEPTION)
      << "CreateInstanceEvaluator the type_obj should be an object of ClassType or MsClassObject, but got "
      << type_obj->ToString() << ".";
  }
  MS_LOG(DEBUG) << "Get class type: " << type_obj->ToString() << ".";
  return type_obj->obj();
}

class CreateInstanceEvaluator final : public TransitionPrimEvaluator {
 public:
  CreateInstanceEvaluator() : TransitionPrimEvaluator("CreateInstanceEvaluator") {}
  ~CreateInstanceEvaluator() override = default;
  MS_DECLARE_PARENT(CreateInstanceEvaluator, TransitionPrimEvaluator);
  EvalResultPtr EvalPrim(const AnalysisEnginePtr &engine, const AbstractBasePtrList &args_abs_list, const ConfigPtr &,
                         const AnfNodeConfigPtr &out_conf) override {
    // Check the type parameter.
    if (args_abs_list.empty()) {
      MS_LOG(INTERNAL_EXCEPTION) << "'args_abs_list' should not be empty";
    }
    auto node = out_conf->node();
    MS_EXCEPTION_IF_NULL(node);
    constexpr size_t class_index = 0;
    auto class_obj = GetPythonObject(args_abs_list[class_index]);
    py::module mod = python_adapter::GetPyModule(parse::PYTHON_MOD_PARSE_MODULE);
    std::string class_name =
      python_adapter::CallPyModFn(mod, parse::PYTHON_MOD_GET_MS_CLASS_NAME, class_obj).cast<std::string>();
    // Get the create instance obj's parameters, `params` may contain tuple(args, kwargs).
    auto params = py::tuple(args_abs_list.size() - 1);
    bool is_prim_variable = GetParameters(args_abs_list, class_obj, class_name, &params);
    // Create primitive instance with variable arguments.
    if (is_prim_variable) {
      auto ret_val = std::make_shared<abstract::PrimInstanceAbstractClosure>(class_name, args_abs_list, node);
      return std::make_shared<EvalResult>(ret_val, std::make_shared<AttrValueMap>());
    }
    // Create class instance.
    auto obj = parse::data_converter::CreatePythonObject(class_obj, params);
    if (py::isinstance<py::none>(obj)) {
      MS_LOG(EXCEPTION) << "Create python object `" << py::str(class_obj)
                        << "` failed, only support to create 'Cell', 'Primitive' or "
                        << "user-defined Class decorated with 'jit_class'.";
    }

    // Process the object.
    TraceGuard guard(MakeTraceInfo<TraceResolve>(out_conf->node()->debug_info()));
    ValuePtr converted_res = nullptr;
    bool converted = parse::ConvertData(obj, &converted_res, true);
    if (!converted) {
      MS_LOG(INTERNAL_EXCEPTION) << "Convert the python object failed";
    }
    MS_EXCEPTION_IF_NULL(converted_res);
    // To check isolated side effect for the func graph who returns constant.
    HandleSideEffect(obj, converted_res, engine, out_conf);

    if (converted_res->isa<FuncGraph>()) {
      AddToManager(engine, converted_res->cast<FuncGraphPtr>());
    }
    AbstractBasePtr res = ToAbstract(converted_res, AnalysisContext::DummyContext(), out_conf);
    auto infer_result = std::make_shared<EvalResult>(res, std::make_shared<AttrValueMap>());
    evaluator_cache_mgr_->SetValue(args_abs_list, infer_result);
    return infer_result;
  }

  void HandleSideEffect(const py::object &obj, const ValuePtr &converted_res, const AnalysisEnginePtr &engine,
                        const AnfNodeConfigPtr &out_conf) const {
    if (engine->check_side_effect()) {
      MS_LOG(DEBUG) << "obj: " << py::str(obj) << ", converted_res: " << converted_res->ToString();
      auto prim = GetValueWithoutDoSignature(converted_res)->cast<PrimitivePtr>();
      if (prim != nullptr) {
        auto effect_info = GetPrimEffectInfo(prim);
        if (effect_info.memory || effect_info.io) {
          const auto &cnode = dyn_cast<CNode>(out_conf->node());
          MS_EXCEPTION_IF_NULL(cnode);
          MS_EXCEPTION_IF_NULL(out_conf->func_graph());
          MS_LOG(DEBUG) << "Found side-effect, cnode: " << cnode->DebugString()
                        << ", func_graph: " << out_conf->func_graph()->ToString();
          cnode->set_has_side_effect_node(true);
          out_conf->func_graph()->set_has_side_effect_node(true);
        }
      }
    }
  }

  bool GetParameters(const AbstractBasePtrList &args_abs_list, const py::object &obj, const std::string &cls_name,
                     py::tuple *params) {
    auto params_size = (*params).size();
    for (size_t i = 0; i < params_size; i++) {
      // Only support the Scalar parameters type. Bypass class type by offset with 1.
      auto arg = args_abs_list[i + 1];
      MS_EXCEPTION_IF_NULL(arg);

      auto param_value = arg->BuildValue();
      MS_EXCEPTION_IF_NULL(param_value);
      if (param_value->ContainsValueAny() && !arg->isa<AbstractFunction>()) {
        // If obj is a Primitive class and has variable arguments, just return and go through another process.
        if (py::hasattr(obj, PYTHON_PRIMITIVE_FLAG) && mindspore::ops::GetOpDef(cls_name) != nullptr) {
          return true;
        }
        MS_EXCEPTION(TypeError) << "When creating an instance of '" << cls_name
                                << "', all arguments are required to be constants, but input " << i
                                << " is a variable, which is " << arg->ToString() << ".";
      }
      py::object param = ValueToPyData(param_value, arg);
      (*params)[i] = param;
    }
    return false;
  }
};

class PartialEvaluator final : public Evaluator {
 public:
  PartialEvaluator() : Evaluator("PartialEvaluator") {}
  ~PartialEvaluator() override = default;
  EvalResultPtr Run(AnalysisEnginePtr engine, const ConfigPtrList &args_conf_list,
                    const AnfNodeConfigPtr &out_conf) override {
    if (args_conf_list.size() == 0) {
      MS_LOG(INTERNAL_EXCEPTION) << "Args size should be greater than 0";
    }
    MS_EXCEPTION_IF_NULL(out_conf);
    MS_EXCEPTION_IF_NULL(out_conf->node());
    MS_EXCEPTION_IF_NULL(args_conf_list[0]);
    const auto &arg0_eval_result = args_conf_list[0]->ObtainEvalResult();
    MS_EXCEPTION_IF_NULL(arg0_eval_result);
    auto arg0_value = arg0_eval_result->abstract();
    MS_EXCEPTION_IF_NULL(arg0_value);
    AbstractBasePtrList args_abs_list{arg0_value};
    auto cnode = out_conf->node()->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);

    // Func in hypermap(partial(Func, arg0), arg1, arg2) may become Poly Node.
    if (arg0_value->isa<AbstractProblem>()) {
      MS_EXCEPTION_IF_NULL(arg0_value->GetValueTrack());
      const auto &value_problem = arg0_value->GetValueTrack()->cast<ValueProblemPtr>();
      auto res = std::make_shared<AbstractProblem>(value_problem, out_conf->node());
      MS_LOG(DEBUG) << "AbstractProblem for node: " << out_conf->node()->DebugString()
                    << " as func is: " << arg0_value->ToString();
      auto eval_result = std::make_shared<EvalResult>(res, std::make_shared<AttrValueMap>());
      evaluator_cache_mgr_->SetValue(args_abs_list, eval_result);
      return eval_result;
    }
    auto func = CheckArg<AbstractFunction>("partial", args_abs_list, 0);
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

    (void)std::transform(args_conf_list.begin() + 1, args_conf_list.end(), std::back_inserter(args_abs_list),
                         [](const ConfigPtr &config) -> AbstractBasePtr {
                           MS_EXCEPTION_IF_NULL(config);
                           const auto &eval_result = config->ObtainEvalResult();
                           MS_EXCEPTION_IF_NULL(eval_result);
                           return eval_result->abstract();
                         });
    AbstractBasePtrList args(args_abs_list.begin() + 1, args_abs_list.end());

    if (cnode->size() != (args_conf_list.size() + 1)) {
      MS_LOG(INTERNAL_EXCEPTION) << "Out_conf node: " << cnode->DebugString()
                                 << ", args_conf_list: " << mindspore::ToString(args_conf_list);
    }
    AbstractFuncAtomPtrList partial_funcs_list;
    auto build_partial = [args, cnode, &partial_funcs_list](const AbstractFuncAtomPtr &atom_func) {
      auto new_func = std::make_shared<PartialAbstractClosure>(atom_func, args, cnode);
      partial_funcs_list.push_back(new_func);
    };
    func->Visit(build_partial);

    auto res = AbstractFunction::MakeAbstractFunction(partial_funcs_list);
    auto eval_result = std::make_shared<EvalResult>(res, std::make_shared<AttrValueMap>());
    MS_LOG(DEBUG) << "args_abs_list: " << args_abs_list << ", eval_result: " << eval_result->abstract()->ToString();
    evaluator_cache_mgr_->SetValue(args_abs_list, eval_result);
    return eval_result;
  }

  EvalResultPtr Eval(AnalysisEnginePtr, const AbstractBasePtrList &, const AnfNodeConfigPtr &) override {
    MS_LOG(INTERNAL_EXCEPTION) << "Eval() should not be called, Run() method should be called";
  }

  EvalResultPtr HandleDoSignature(const AnalysisEnginePtr &engine, const ValuePtr &signature_value,
                                  const AnfNodeConfigPtr &out_conf) const {
    MS_EXCEPTION_IF_NULL(engine);
    MS_EXCEPTION_IF_NULL(out_conf);
    MS_EXCEPTION_IF_NULL(out_conf->node());
    auto cnode = out_conf->node()->cast_ptr<CNode>();
    MS_EXCEPTION_IF_NULL(cnode);

    ScopeGuard scope_guard(out_conf->node()->scope());
    TraceGuard trace_guard(MakeTraceInfo<TraceDoSignature>(out_conf->node()->debug_info()));
    auto new_nodes_inputs = cnode->weak_inputs();
    auto new_signature_value = std::make_shared<prim::DoSignatureMetaFuncGraph>("signature", signature_value);
    auto new_sig_node = NewValueNode(new_signature_value);
    new_nodes_inputs[1] = AnfNodeWeakPtr(new_sig_node);
    FuncGraphPtr func_graph = cnode->func_graph();
    MS_EXCEPTION_IF_NULL(func_graph);
    CNodePtr new_cnode = func_graph->NewCNodeWeak(std::move(new_nodes_inputs));
    AnfNodeConfigPtr fn_conf = engine->MakeConfig(new_cnode, out_conf->context(), out_conf->func_graph());
    return engine->ForwardConfig(out_conf, fn_conf);
  }
};

class WhileLoopEvaluator final : public Evaluator {
 public:
  WhileLoopEvaluator() : Evaluator("WhileLoopEvaluator") {}
  ~WhileLoopEvaluator() override = default;

  EvalResultPtr Run(AnalysisEnginePtr engine, const ConfigPtrList &args_conf_list,
                    const AnfNodeConfigPtr &out_conf) override {
    constexpr size_t input_size = 3;
    if (args_conf_list.size() != input_size) {
      MS_LOG(INTERNAL_EXCEPTION) << "WhileLoop op expects " << input_size << " inputs, but got "
                                 << args_conf_list.size();
    }
    MS_EXCEPTION_IF_NULL(out_conf);
    MS_EXCEPTION_IF_NULL(out_conf->node());
    auto cnode = out_conf->node()->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    auto cur_graph = cnode->func_graph();
    MS_EXCEPTION_IF_NULL(cur_graph);

    AbstractBasePtrList args_abs_list;
    (void)std::transform(args_conf_list.begin(), args_conf_list.end(), std::back_inserter(args_abs_list),
                         [](const ConfigPtr &config) -> AbstractBasePtr {
                           MS_EXCEPTION_IF_NULL(config);
                           const auto &eval_result = config->ObtainEvalResult();
                           MS_EXCEPTION_IF_NULL(eval_result);
                           const auto &abs = eval_result->abstract();
                           if (abs->isa<abstract::AbstractSequence>()) {
                             SetSequenceElementsUseFlagsRecursively(abs, true);
                           }
                           return abs;
                         });

    // Get conditiona and loop func graph
    // CNode: {kPrimWhileloop, cond_func, loop_func, init_value}
    // --> while(cond_func(init_value)):
    // -->     init_value = loop_func(init_value)
    // --> return init_value
    auto cond_func = CheckArg<AbstractFunction>("while_loop", args_abs_list, kIndex0);
    auto loop_func = CheckArg<AbstractFunction>("while_loop", args_abs_list, kIndex1);
    auto init_value_abs = args_abs_list[kIndex2];
    auto init_value_node = cnode->input(kIndex3);

    // Evaluate condition and loop functions
    ConfigPtrList value_arg_conf_list = {std::make_shared<VirtualConfig>(init_value_abs)};
    EvalResultPtr cond_eval_result = engine->GetEvaluatorFor(cond_func)->Run(engine, value_arg_conf_list, nullptr);
    EvalResultPtr loop_eval_result = engine->GetEvaluatorFor(loop_func)->Run(engine, value_arg_conf_list, nullptr);
    auto loop_result_abs = loop_eval_result->abstract();
    if (!CheckTypeIdAndShapeEqual(loop_result_abs, init_value_abs)) {
      MS_EXCEPTION(ValueError) << "WhileLoop op has invalid argument, the return value of the [loop_func] "
                               << "and the [init_value] should maintain the same type and shape, but got: "
                               << loop_result_abs->ToString() << " and " << init_value_abs->ToString();
    }
    // Generate condition function graph
    auto cond_fg = cond_func->cast<abstract::FuncGraphAbstractClosurePtr>()->func_graph();
    cond_fg->debug_info()->set_name("cond_func");
    // Keep while loop not unroll
    if (cond_eval_result->abstract()->BuildValue()->ContainsValueAny() ||
        loop_result_abs->BuildValue()->ContainsValueAny()) {
      auto output = cond_fg->output();
      auto mutable_output = cond_fg->NewCNodeInOrder({NewValueNode(prim::kPrimMutable), output});
      cond_fg->set_output(mutable_output);
    }

    // Convert kPrimWhileLoop to a func graph
    auto while_loop_graph = std::make_shared<FuncGraph>();
    while_loop_graph->set_flag(FUNC_GRAPH_FLAG_CORE, true);
    while_loop_graph->debug_info()->set_name("while_loop");
    auto manager = engine->func_graph_manager();
    MS_EXCEPTION_IF_NULL(manager);
    while_loop_graph->set_manager(manager);
    auto init_param = while_loop_graph->add_parameter();
    auto cond_result = while_loop_graph->NewCNodeInOrder({NewValueNode(cond_fg), init_param});
    // Generate loop function graph
    auto loop_func_graph = std::make_shared<FuncGraph>();
    loop_func_graph->debug_info()->set_name("loop_func");
    auto loop_fg = loop_func->cast<abstract::FuncGraphAbstractClosurePtr>()->func_graph();
    auto loop_value = loop_func_graph->NewCNodeInOrder({NewValueNode(loop_fg), init_param});
    auto loop_result = loop_func_graph->NewCNodeInOrder({NewValueNode(while_loop_graph), loop_value});
    loop_func_graph->set_output(loop_result);
    loop_func_graph->set_manager(manager);
    // Generate return func
    auto return_func_graph = std::make_shared<FuncGraph>();
    return_func_graph->debug_info()->set_name("return_func");
    return_func_graph->set_output(init_param);
    return_func_graph->set_manager(manager);
    //--> def while_loop_graph(init_val):
    //-->   return cond_fg(init_val) ? loop_func_graph(init_val) : return_func_graph(init_val)
    auto cond_node =
      while_loop_graph->NewCNodeInOrder({NewValueNode(prim::kPrimCond), cond_result, NewValueNode(MakeValue(true))});
    auto sw_node = while_loop_graph->NewCNodeInOrder(
      {NewValueNode(prim::kPrimSwitch), cond_node, NewValueNode(loop_func_graph), NewValueNode(return_func_graph)});
    auto result = while_loop_graph->NewCNodeInOrder({sw_node});
    while_loop_graph->set_output(result);
    auto while_loop_graph_caller = cur_graph->NewCNodeInOrder({NewValueNode(while_loop_graph), init_value_node});
    AnfNodeConfigPtr fn_conf = engine->MakeConfig(while_loop_graph_caller, out_conf->context(), out_conf->func_graph());
    return engine->ForwardConfig(out_conf, fn_conf);
  }

  EvalResultPtr Eval(AnalysisEnginePtr, const AbstractBasePtrList &, const AnfNodeConfigPtr &) override {
    MS_LOG(INTERNAL_EXCEPTION) << "Eval() should not be called, Run() method should be called";
  }
};

class ScanEvaluator final : public Evaluator {
 public:
  ScanEvaluator() : Evaluator("ScanEvaluator") {}
  ~ScanEvaluator() override = default;

  EvalResultPtr Run(AnalysisEnginePtr engine, const ConfigPtrList &args_conf_list,
                    const AnfNodeConfigPtr &out_conf) override {
    constexpr size_t min_input_size = 3;
    constexpr size_t max_input_size = 5;
    if (args_conf_list.size() < min_input_size || args_conf_list.size() > max_input_size) {
      MS_LOG(INTERNAL_EXCEPTION) << "Scan op expects [" << min_input_size << ", " << max_input_size
                                 << "] inputs, but got " << args_conf_list.size();
    }

    MS_EXCEPTION_IF_NULL(out_conf);
    MS_EXCEPTION_IF_NULL(out_conf->node());
    auto cnode = out_conf->node()->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    auto cur_graph = cnode->func_graph();
    MS_EXCEPTION_IF_NULL(cur_graph);
    auto manager = engine->func_graph_manager();
    MS_EXCEPTION_IF_NULL(manager);

    AbstractBasePtrList args_abs_list;
    (void)std::transform(args_conf_list.begin(), args_conf_list.end(), std::back_inserter(args_abs_list),
                         [](const ConfigPtr &config) -> AbstractBasePtr {
                           MS_EXCEPTION_IF_NULL(config);
                           const auto &eval_result = config->ObtainEvalResult();
                           MS_EXCEPTION_IF_NULL(eval_result);
                           const auto &abs = eval_result->abstract();
                           if (abs->isa<abstract::AbstractSequence>()) {
                             SetSequenceElementsUseFlagsRecursively(abs, true);
                           }
                           return abs;
                         });

    // CNode: {kPrimScan, f, init, xs, length, unroll}
    // --> ys = []
    // --> for x in xs:
    // -->     init, y = f(init, x)
    // -->     ys.append(y)
    // --> return init, ys
    // Get condition and loop func graph
    auto [loop_func, init_value, xs_abs, length_value, user_unroll, is_standard_input] =
      GenerateScanArgs(args_abs_list);
    auto init_node = cnode->input(kIndex2);
    // Process Empty xs array
    if (!length_value) {
      AnfNodePtr empty_list = cur_graph->NewCNodeInOrder({NewValueNode(prim::kPrimMakeList)});
      AnfNodePtr zero_loop_output =
        cur_graph->NewCNodeInOrder({NewValueNode(prim::kPrimMakeTuple), init_node, empty_list});
      AnfNodeConfigPtr fn_conf = engine->MakeConfig(zero_loop_output, out_conf->context(), out_conf->func_graph());
      return engine->ForwardConfig(out_conf, fn_conf);
    }
    // Standardlize scan's input arguments
    if (user_unroll && !is_standard_input) {
      MS_LOG(DEBUG) << "Standardlize Scan op";
      AnfNodePtrList inputs = {cnode->inputs().begin(), cnode->inputs().begin() + 4};
      (void)inputs.emplace_back(NewValueNode(length_value));
      (void)inputs.emplace_back(NewValueNode(user_unroll));
      AnfNodePtr standard_scan_node = cur_graph->NewCNodeInOrder(inputs);
      AnfNodeConfigPtr fn_conf = engine->MakeConfig(standard_scan_node, out_conf->context(), out_conf->func_graph());
      return engine->ForwardConfig(out_conf, fn_conf);
    }

    auto [xs_node, getitem_op] = CheckXsNode(cnode->input(kIndex3), xs_abs, length_value, cur_graph);
    auto loop_func_node = CheckLoopFunc(loop_func);
    AnfNodePtr result_node = nullptr;
    AbstractBasePtr loop_result_abs = EvaluateLoopFunction(engine, loop_func, init_value, xs_abs, cnode);
    if (loop_result_abs->BuildValue()->ContainsValueAny()) {
      loop_func_node->set_flag(FUNC_GRAPH_FLAG_IGNORE_VALUE, true);
      (void)EvaluateLoopFunction(engine, loop_func, init_value, xs_abs, cnode);
      // Keep kPrimScan and get EvalResult directly, unroll later
      if (user_unroll) {
        MS_LOG(DEBUG) << "`Scan` op will be unrolled in graph optimization action later";
        return CreateScanEvalResult(loop_result_abs, length_value, args_abs_list);
      }
      MS_LOG(DEBUG) << "`Scan` op will be translated into a loop function call in type inference action";
      std::string reason = "Loop op with unroll set as false is not allow do higher order grad";
      cur_graph->set_attr(FUNC_GRAPH_ATTR_UNSUPPORT_HIGHER_GRAD_REASON, MakeValue(reason));
      result_node = RepeatLoop(cur_graph, loop_func_node, init_node, xs_node, getitem_op, length_value, manager);
    } else {
      // Unroll the loop
      MS_LOG(DEBUG) << "`Scan` op can be calculated in type inference action";
      result_node = UnrollLoop(cur_graph, loop_func_node, init_node, length_value, xs_node, getitem_op);
    }
    AnfNodeConfigPtr fn_conf = engine->MakeConfig(result_node, out_conf->context(), out_conf->func_graph());
    return engine->ForwardConfig(out_conf, fn_conf);
  }

  EvalResultPtr Eval(AnalysisEnginePtr, const AbstractBasePtrList &, const AnfNodeConfigPtr &) override {
    MS_LOG(INTERNAL_EXCEPTION) << "Eval() should not be called, Run() method should be called";
  }

 private:
  std::tuple<AbstractFunctionPtr, AbstractBasePtr, AbstractBasePtr, int64_t, bool, bool> GenerateScanArgs(
    const AbstractBasePtrList &args_abs_list) {
    // CNode: {kPrimScan, f[func], init, xs, length[int64_t], unroll[bool]}
    auto loop_func = CheckArg<AbstractFunction>("scan", args_abs_list, kIndex0);
    // Parse keywords args
    std::map<std::string, AbstractBasePtr> args_map;
    bool is_standard_input = true;
    (void)std::for_each(args_abs_list.begin() + kIndex2, args_abs_list.end(), [&args_map](const AbstractBasePtr &abs) {
      if (abs->isa<AbstractKeywordArg>()) {
        auto keyword = abs->cast<AbstractKeywordArgPtr>();
        args_map[keyword->get_key()] = keyword->get_arg();
      }
    });
    // Get input argument with default value
    auto GetArgsAbs = [&args_abs_list, &args_map, &is_standard_input](const AbstractBasePtr &default_abs,
                                                                      const std::string &name, const size_t index) {
      auto iter = args_map.find(name);
      if (iter != args_map.end()) {
        is_standard_input = false;
        return iter->second;
      }
      if (index < args_abs_list.size() && !args_abs_list[index]->isa<AbstractKeywordArg>()) {
        return args_abs_list[index];
      }
      is_standard_input = false;
      return default_abs;
    };
    auto length_abs = GetArgsAbs(std::make_shared<AbstractNone>(), "length", kIndex3);
    int64_t length_value = GetLengthValue(args_abs_list[kIndex2], length_abs);
    auto unroll_abs = GetArgsAbs(std::make_shared<AbstractScalar>(true), "unroll", kIndex4);
    auto unroll = unroll_abs->BuildValue();
    if (!unroll->isa<BoolImm>()) {
      MS_EXCEPTION(TypeError) << "For `Scan` op, expect argument [unroll] to be bool, but got "
                              << args_abs_list[kIndex4]->ToString();
    }
    return std::make_tuple(loop_func, args_abs_list[kIndex1], args_abs_list[kIndex2], length_value,
                           GetValue<bool>(unroll), is_standard_input);
  }

  int64_t GetLengthValue(const AbstractBasePtr &xs_abs, const AbstractBasePtr &length_abs) {
    if (xs_abs->isa<AbstractNone>() && length_abs->isa<AbstractNone>()) {
      MS_EXCEPTION(ValueError) << "For `Scan` op, argument [xs] and [length] cannot be None at the same time";
    }
    size_t xs_size = 0;
    if (xs_abs->isa<AbstractSequence>()) {
      xs_size = xs_abs->cast<AbstractSequencePtr>()->size();
    } else if (!xs_abs->isa<AbstractNone>()) {
      MS_EXCEPTION(TypeError) << "For `Scan` op, expect abstract of argument [xs] to be AbstractTuple, "
                              << "AbstractList or AbstractNone, but got: " << xs_abs->ToString();
    }
    auto xs_size_value = SizeToLong(xs_size);
    if (length_abs->isa<AbstractNone>()) {
      return xs_size_value;
    }
    if (length_abs->BuildValue()->isa<Int64Imm>()) {
      auto length_abs_value = GetValue<int64_t>(length_abs->BuildValue());
      if (length_abs_value == xs_size_value || xs_abs->isa<AbstractNone>()) {
        return length_abs_value;
      }
      MS_EXCEPTION(ValueError) << "For `Scan` op, supposed to have the same value of [length] argument "
                               << "as the length of [xs], but got: " << length_abs_value << " and: " << xs_size_value;
    }
    MS_LOG(EXCEPTION) << "For `Scan` op, supposed to have int argument [length], but got: " << length_abs->ToString();
  }

  AbstractBasePtr GetItemAbs(const AbstractBasePtr &array) {
    if (array->isa<AbstractSequence>()) {
      auto abs_seq = array->cast<AbstractSequencePtr>();
      if (!abs_seq->empty()) {
        const auto &ele = abs_seq->elements();
        return ele[kIndex0];
      }
    }
    return std::make_shared<abstract::AbstractNone>();
  }

  std::tuple<AnfNodePtr, PrimitivePtr> CheckXsNode(const AnfNodePtr &xs_node, const AbstractBasePtr &xs_abs,
                                                   const int64_t length_value, const FuncGraphPtr &cur_graph) {
    std::map<TypeId, PrimitivePtr> getitem_op_map = {{kObjectTypeTuple, prim::kPrimTupleGetItem},
                                                     {kObjectTypeList, prim::kPrimListGetItem},
                                                     {kMetaTypeNone, prim::kPrimTupleGetItem}};
    MS_EXCEPTION_IF_NULL(xs_node);
    AnfNodePtr processed_array = xs_node;
    auto type = xs_abs->GetType();
    MS_EXCEPTION_IF_NULL(type);
    auto type_id = type->type_id();
    auto iter = getitem_op_map.find(type_id);
    if (iter == getitem_op_map.end()) {
      MS_LOG(EXCEPTION) << "`Scan` op has invalid xs argument";
    }
    PrimitivePtr getitem_op = iter->second;
    // Process None xs_node
    if (type_id == kMetaTypeNone) {
      std::vector<AnfNodePtr> xs_nodes(length_value + 1, NewValueNode(static_cast<int64_t>(0)));
      xs_nodes[kIndex0] = NewValueNode(prim::kPrimMakeTuple);
      processed_array = cur_graph->NewCNodeInOrder(xs_nodes);
    }
    return std::make_tuple(processed_array, getitem_op);
  }

  FuncGraphPtr CheckLoopFunc(const AbstractFunctionPtr &loop_func) {
    auto loop_func_ptr = loop_func->cast<abstract::FuncGraphAbstractClosurePtr>();
    MS_EXCEPTION_IF_NULL(loop_func_ptr);
    const FuncGraphPtr &loop_func_node = loop_func_ptr->func_graph();
    MS_EXCEPTION_IF_NULL(loop_func_node);
    constexpr size_t loop_func_expect_input_size = 2;
    auto loop_func_params = loop_func_node->get_inputs();
    auto loop_func_input_size = loop_func_params.size();
    if (loop_func_input_size != loop_func_expect_input_size) {
      MS_EXCEPTION(ValueError) << "For `Scan` op, loop_func expects two arguments, but got: " << loop_func_input_size;
    }
    return loop_func_node;
  }

  AbstractBasePtr EvaluateLoopFunction(AnalysisEnginePtr engine, const AbstractFunctionPtr &loop_func,
                                       const AbstractBasePtr &init_value, const AbstractBasePtr &xs_abs,
                                       const AnfNodePtr &cnode) {
    auto abs_item = GetItemAbs(xs_abs);
    ConfigPtrList value_arg_conf_list = {std::make_shared<VirtualConfig>(init_value),
                                         std::make_shared<VirtualConfig>(abs_item)};
    auto loop_result = engine->GetEvaluatorFor(loop_func)->Run(engine, value_arg_conf_list, nullptr);
    auto loop_result_abs = loop_result->abstract();
    if (!loop_result_abs) {
      MS_LOG(EXCEPTION) << "Failed to evaluate loop function.";
    }
    SetSequenceElementsUseFlagsRecursively(loop_result_abs, true);
    auto loop_result_tuple = loop_result_abs->cast<abstract::AbstractTuplePtr>();
    constexpr size_t loop_result_size = 2;
    if (!loop_result_tuple || loop_result_tuple->size() != loop_result_size) {
      MS_EXCEPTION(ValueError) << "For `Scan` op, the return value of parameter [loop_func] "
                               << "must be a tuple with two elements, but got: " << loop_result_abs->ToString();
    }
    const auto &ele_abs = loop_result_tuple->elements()[0];
    constexpr auto kCheckArg = "CheckArg";
    if (!cnode->has_user_data(kCheckArg)) {
      if (!CheckTypeIdAndShapeEqual(ele_abs, init_value)) {
        MS_EXCEPTION(ValueError) << "Scan op has invalid argument, the first element of [loop_func]'s output "
                                 << "and the [init_value] should maintain the same type and shape, but got: "
                                 << ele_abs->ToString() << " and " << init_value->ToString();
      }
      cnode->set_user_data(kCheckArg, std::make_shared<bool>(true));
    }

    return loop_result_abs;
  }

  EvalResultPtr CreateScanEvalResult(const AbstractBasePtr &loop_result_abs, int64_t length_value,
                                     const AbstractBasePtrList &args_abs_list) {
    auto loop_result_tuple = loop_result_abs->cast<AbstractTuplePtr>();
    const auto &loop_result_eles = loop_result_tuple->elements();
    AbstractBasePtrList ys_abs_list(length_value, loop_result_eles[kIndex1]);
    auto ys_abs = std::make_shared<AbstractList>(ys_abs_list);
    AbstractBasePtrList scan_abs_list = {loop_result_eles[kIndex0], ys_abs};
    // Loop_func return (init, y1) --> kPrimScan evalresult (init, [y1, ..., yn])
    auto result_abs = std::make_shared<AbstractTuple>(scan_abs_list);
    auto eval_result = std::make_shared<EvalResult>(result_abs, std::make_shared<AttrValueMap>());
    evaluator_cache_mgr_->SetValue(args_abs_list, eval_result);
    return eval_result;
  }

  AnfNodePtr UnrollLoop(const FuncGraphPtr &cur_graph, const FuncGraphPtr &loop_func_node, const AnfNodePtr &init_node,
                        int64_t length_value, const AnfNodePtr &xs_node, const PrimitivePtr &getitem_op) {
    AnfNodePtrList ys_result{NewValueNode(prim::kPrimMakeList)};
    AnfNodePtr loop_init = init_node;
    for (int64_t i = 0; i < length_value; ++i) {
      auto item =
        cur_graph->NewCNodeInOrder({NewValueNode(getitem_op), xs_node, NewValueNode(static_cast<int64_t>(i))});
      auto func_output = cur_graph->NewCNodeInOrder({NewValueNode(loop_func_node), loop_init, item});
      loop_init = cur_graph->NewCNodeInOrder(
        {NewValueNode(prim::kPrimTupleGetItem), func_output, NewValueNode(static_cast<int64_t>(0))});
      auto new_y = cur_graph->NewCNodeInOrder(
        {NewValueNode(prim::kPrimTupleGetItem), func_output, NewValueNode(static_cast<int64_t>(1))});
      (void)ys_result.emplace_back(new_y);
    }
    auto loop_ys = cur_graph->NewCNodeInOrder(ys_result);
    return cur_graph->NewCNodeInOrder({NewValueNode(prim::kPrimMakeTuple), loop_init, loop_ys});
  }

  AnfNodePtr RepeatLoop(const FuncGraphPtr &cur_graph, const FuncGraphPtr &loop_func_node, const AnfNodePtr &init_node,
                        const AnfNodePtr &xs_node, const PrimitivePtr &getitem_op, int64_t length_value,
                        const FuncGraphManagerPtr &manager) {
    // Build alternative func graph for scan op
    auto scan_func_graph = std::make_shared<FuncGraph>();
    scan_func_graph->debug_info()->set_name("scan");
    scan_func_graph->set_manager(manager);
    auto index_param = scan_func_graph->add_parameter();
    auto xs_param = scan_func_graph->add_parameter();
    auto init_param = scan_func_graph->add_parameter();
    auto ys_param = scan_func_graph->add_parameter();
    // def loop_func_graph():
    // --> x = xs[i]
    // --> init, y = f(init, x)
    // --> ys.append(y)
    // --> i = i+1
    // --> return top_func(i, init, ys)
    auto loop_func_graph = std::make_shared<FuncGraph>();
    auto item = loop_func_graph->NewCNodeInOrder({NewValueNode(getitem_op), xs_param, index_param});
    auto func_output = loop_func_graph->NewCNodeInOrder({NewValueNode(loop_func_node), init_param, item});
    auto new_init = loop_func_graph->NewCNodeInOrder(
      {NewValueNode(prim::kPrimTupleGetItem), func_output, NewValueNode(static_cast<int64_t>(0))});
    auto new_y = loop_func_graph->NewCNodeInOrder(
      {NewValueNode(prim::kPrimTupleGetItem), func_output, NewValueNode(static_cast<int64_t>(1))});
    auto new_ys = loop_func_graph->NewCNodeInOrder({NewValueNode(prim::kPrimListAppend), ys_param, new_y});
    auto new_index = loop_func_graph->NewCNodeInOrder(
      {NewValueNode(prim::kPrimScalarAdd), index_param, NewValueNode(static_cast<int64_t>(1))});
    auto output =
      loop_func_graph->NewCNodeInOrder({NewValueNode(scan_func_graph), new_index, xs_param, new_init, new_ys});
    loop_func_graph->set_output(output);
    // def return_func():
    // --> return (init, ys)
    auto return_func_graph = std::make_shared<FuncGraph>();
    auto return_func_graph_node =
      return_func_graph->NewCNodeInOrder({NewValueNode(prim::kPrimMakeTuple), init_param, ys_param});
    return_func_graph->set_output(return_func_graph_node);
    // def scan_func_graph(i, init, ys):
    // --> return (i < len(xs)) ? loop_func_graph(): return_func()
    CNodePtr compare_node =
      scan_func_graph->NewCNodeInOrder({NewValueNode(prim::kPrimScalarLt), index_param, NewValueNode(length_value)});
    auto cond_node =
      scan_func_graph->NewCNodeInOrder({NewValueNode(prim::kPrimCond), compare_node, NewValueNode(MakeValue(true))});
    auto switch_node = scan_func_graph->NewCNodeInOrder(
      {NewValueNode(prim::kPrimSwitch), cond_node, NewValueNode(loop_func_graph), NewValueNode(return_func_graph)});
    auto result = scan_func_graph->NewCNodeInOrder({switch_node});
    scan_func_graph->set_output(result);
    // Call loop_func first and init ys as a list with one element, to avoid TypeJoined Problem
    auto top_scan_func_graph = std::make_shared<FuncGraph>();
    auto top_xs_param = top_scan_func_graph->add_parameter();
    auto top_init_param = top_scan_func_graph->add_parameter();
    top_scan_func_graph->set_manager(manager);
    auto first_item = top_scan_func_graph->NewCNodeInOrder(
      {NewValueNode(getitem_op), top_xs_param, NewValueNode(static_cast<int64_t>(0))});
    auto first_loop_func_output =
      top_scan_func_graph->NewCNodeInOrder({NewValueNode(loop_func_node), top_init_param, first_item});
    auto new_init_node = top_scan_func_graph->NewCNodeInOrder(
      {NewValueNode(prim::kPrimTupleGetItem), first_loop_func_output, NewValueNode(static_cast<int64_t>(0))});
    auto y_node = top_scan_func_graph->NewCNodeInOrder(
      {NewValueNode(prim::kPrimTupleGetItem), first_loop_func_output, NewValueNode(static_cast<int64_t>(1))});
    auto ys_node = top_scan_func_graph->NewCNodeInOrder({NewValueNode(prim::kPrimMakeList), y_node});
    auto mutable_index_node =
      top_scan_func_graph->NewCNodeInOrder({NewValueNode(prim::kPrimMutable), NewValueNode(static_cast<int64_t>(1))});
    auto top_result_node = top_scan_func_graph->NewCNodeInOrder(
      {NewValueNode(scan_func_graph), mutable_index_node, top_xs_param, new_init_node, ys_node});
    top_scan_func_graph->set_output(top_result_node);
    auto result_node = cur_graph->NewCNodeInOrder({NewValueNode(top_scan_func_graph), xs_node, init_node});
    return result_node;
  }
};

class ForiLoopEvaluator final : public Evaluator {
 public:
  ForiLoopEvaluator() : Evaluator("ForiLoopEvaluator") {}
  ~ForiLoopEvaluator() override = default;

  EvalResultPtr Run(AnalysisEnginePtr engine, const ConfigPtrList &args_conf_list,
                    const AnfNodeConfigPtr &out_conf) override {
    constexpr size_t min_input_size = 4;
    constexpr size_t max_input_size = 5;
    if (args_conf_list.size() < min_input_size || args_conf_list.size() > max_input_size) {
      MS_LOG(INTERNAL_EXCEPTION) << "For `ForiLoop` op, expects [" << min_input_size << ", " << max_input_size
                                 << "] inputs, but got " << args_conf_list.size();
    }
    MS_EXCEPTION_IF_NULL(out_conf);
    MS_EXCEPTION_IF_NULL(out_conf->node());
    auto cnode = out_conf->node()->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    auto cur_graph = cnode->func_graph();
    MS_EXCEPTION_IF_NULL(cur_graph);
    auto manager = engine->func_graph_manager();
    MS_EXCEPTION_IF_NULL(manager);

    AbstractBasePtrList args_abs_list;
    (void)std::transform(args_conf_list.begin(), args_conf_list.end(), std::back_inserter(args_abs_list),
                         [](const ConfigPtr &config) -> AbstractBasePtr {
                           MS_EXCEPTION_IF_NULL(config);
                           const auto &eval_result = config->ObtainEvalResult();
                           MS_EXCEPTION_IF_NULL(eval_result);
                           return eval_result->abstract();
                         });
    // {kPrimFroiLoop, lower_index, upper_index, loop_func, init_val, unroll}
    // Get condition and loop func graph
    constexpr size_t lower_index = 0;
    constexpr size_t upper_index = 1;
    constexpr size_t loop_func_index = 2;
    constexpr size_t init_index = 3;
    auto lower_index_node = cnode->input(lower_index + 1);
    auto upper_index_node = cnode->input(upper_index + 1);
    auto loop_func = CheckArg<AbstractFunction>("fori_loop", args_abs_list, loop_func_index);
    auto init_value = args_abs_list[init_index];
    auto init_node = cnode->input(init_index + 1);
    auto loop_func_abs = loop_func->cast<abstract::FuncGraphAbstractClosurePtr>();
    MS_EXCEPTION_IF_NULL(loop_func_abs);
    auto loop_func_node = loop_func_abs->func_graph();
    auto length_node = GetLength(args_abs_list[lower_index], args_abs_list[upper_index]);
    AnfNodePtr final_node = nullptr;

    if (length_node != nullptr) {
      MS_LOG(DEBUG) << "`ForiLoop` op has constant boundary parameters, [lower] and [upper]: "
                    << args_abs_list[lower_index]->ToString() << " and " << args_abs_list[upper_index]->ToString();
      // Build Scan loop func graph based on loop_func
      FuncGraphPtr scan_loop_func_graph = BuildLoopFuncGraphOfScan(loop_func_node);
      scan_loop_func_graph->set_manager(manager);
      // Call Scan_op instead
      // (_, result), _ = {prim::kPrimScan(scan_loop_func_graph), (lower, init_val), None, length_node, unroll_node}
      auto unroll_node = GetUnroll(args_abs_list, max_input_size);
      auto scan_init = cur_graph->NewCNode({NewValueNode(prim::kPrimMakeTuple), lower_index_node, init_node});
      auto scan_output =
        cur_graph->NewCNode({NewValueNode(prim::kPrimScan), NewValueNode(scan_loop_func_graph), scan_init,
                             NewValueNode(std::make_shared<None>()), length_node, unroll_node});
      auto carry_output = cur_graph->NewCNode(
        {NewValueNode(prim::kPrimTupleGetItem), scan_output, NewValueNode(static_cast<int64_t>(0))});
      final_node = cur_graph->NewCNodeInOrder(
        {NewValueNode(prim::kPrimTupleGetItem), carry_output, NewValueNode(static_cast<int64_t>(1))});
    } else {
      MS_LOG(DEBUG) << "`ForiLoop` op has variable boundary parameters, [lower] and [upper]: "
                    << args_abs_list[lower_index]->ToString() << " and " << args_abs_list[upper_index]->ToString();
      // Build while_loop cond_func graph
      FuncGraphPtr cond_func_graph = BuildCondFuncGraphOfWhileLoop();
      cond_func_graph->set_manager(manager);
      // Build while loop loop_func graph
      FuncGraphPtr loop_func_graph = BuildLoopFuncGraphOfWhileLoop(loop_func_node);
      loop_func_graph->set_manager(manager);
      // Call while_loop op instead
      // _, _, result = {prim::kPrimWhileLoop(cond_func_graph, loop_func_graph, (lower, upper, init_val)}
      auto params =
        cur_graph->NewCNode({NewValueNode(prim::kPrimMakeTuple), lower_index_node, upper_index_node, init_node});
      auto while_loop_output = cur_graph->NewCNode(
        {NewValueNode(prim::kPrimWhileLoop), NewValueNode(cond_func_graph), NewValueNode(loop_func_graph), params});
      final_node = cur_graph->NewCNodeInOrder(
        {NewValueNode(prim::kPrimTupleGetItem), while_loop_output, NewValueNode(static_cast<int64_t>(2))});
    }
    AnfNodeConfigPtr fn_conf = engine->MakeConfig(final_node, out_conf->context(), out_conf->func_graph());
    return engine->ForwardConfig(out_conf, fn_conf);
  }

  EvalResultPtr Eval(AnalysisEnginePtr, const AbstractBasePtrList &, const AnfNodeConfigPtr &) override {
    MS_LOG(INTERNAL_EXCEPTION) << "Eval() should not be called, Run() method should be called";
  }

 private:
  ValueNodePtr GetLength(const AbstractBasePtr &lower_abs, const AbstractBasePtr &upper_abs) {
    auto lower_value = lower_abs->BuildValue()->cast<Int64ImmPtr>();
    auto upper_value = upper_abs->BuildValue()->cast<Int64ImmPtr>();
    if (!lower_value || !upper_value) {
      return nullptr;
    }
    int64_t length = upper_value->value() - lower_value->value();
    return NewValueNode(length);
  }

  AnfNodePtr GetUnroll(const AbstractBasePtrList &args_abs_list, const size_t max_input_size) {
    if (args_abs_list.size() < max_input_size) {
      return NewValueNode(true);
    }
    const std::string unroll_key = "unroll";
    auto unroll_node_abs = args_abs_list[max_input_size - 1];
    if (unroll_node_abs->isa<AbstractKeywordArg>()) {
      auto keyword = unroll_node_abs->cast<AbstractKeywordArgPtr>();
      if (keyword->get_key() != unroll_key) {
        MS_EXCEPTION(TypeError) << "ForiLoop op has invalid keyword argument: " << keyword->get_key();
      }
      unroll_node_abs = keyword->get_arg();
    }
    if (!unroll_node_abs->isa<AbstractScalar>()) {
      MS_EXCEPTION(TypeError) << "ForiLoop op has invalid [unroll] argument: " << unroll_node_abs->ToString();
    }
    auto unroll = unroll_node_abs->BuildValue();
    if (!unroll || !unroll->isa<BoolImm>()) {
      MS_EXCEPTION(TypeError) << "ForiLoop op supposed to have bool argument [unroll], but got "
                              << unroll_node_abs->ToString();
    }
    return NewValueNode(unroll);
  }

  /**
   * \brief Build scan loop func graph.
   *
   * \example
   *     # python
   *     def scan_loop_func_graph(loop_carry, _):
   *       index, item = loop_carry
   *       body_output = loop_func_node(index, item)
   *       new_index = index + 1
   *       return (new_index, body_output), item
   *
   * \param[in] loop_func_node loop function node.
   *
   * \return The built scan loop func graph.
   **/
  FuncGraphPtr BuildLoopFuncGraphOfScan(const FuncGraphPtr &loop_func_node) {
    auto scan_loop_func_graph = std::make_shared<FuncGraph>();
    auto loop_carry = scan_loop_func_graph->add_parameter();
    (void)scan_loop_func_graph->add_parameter();
    auto index = scan_loop_func_graph->NewCNodeInOrder(
      {NewValueNode(prim::kPrimTupleGetItem), loop_carry, NewValueNode(static_cast<int64_t>(0))});
    auto item = scan_loop_func_graph->NewCNodeInOrder(
      {NewValueNode(prim::kPrimTupleGetItem), loop_carry, NewValueNode(static_cast<int64_t>(1))});
    auto body_output = scan_loop_func_graph->NewCNodeInOrder({NewValueNode(loop_func_node), index, item});
    auto new_index = scan_loop_func_graph->NewCNodeInOrder(
      {NewValueNode(prim::kPrimScalarAdd), index, NewValueNode(static_cast<int64_t>(1))});
    auto tuple_result =
      scan_loop_func_graph->NewCNodeInOrder({NewValueNode(prim::kPrimMakeTuple), new_index, body_output});
    auto scan_loop_func_output = scan_loop_func_graph->NewCNodeInOrder(
      {NewValueNode(prim::kPrimMakeTuple), tuple_result, NewValueNode(static_cast<int64_t>(0))});
    scan_loop_func_graph->set_output(scan_loop_func_output);
    return scan_loop_func_graph;
  }

  /**
   * \brief Build while cond func graph.
   *
   * \example
   *     # python
   *     def cond_func_graph(loop_carry):
   *       cond_index, cond_upper, _ = loop_carry
   *       return cond_index < cond_upper
   *
   * \return The built while condition func graph.
   **/
  FuncGraphPtr BuildCondFuncGraphOfWhileLoop() {
    auto cond_func_graph = std::make_shared<FuncGraph>();
    auto cond_carry = cond_func_graph->add_parameter();
    auto cond_index = cond_func_graph->NewCNodeInOrder(
      {NewValueNode(prim::kPrimTupleGetItem), cond_carry, NewValueNode(static_cast<int64_t>(0))});
    auto cond_uppper = cond_func_graph->NewCNodeInOrder(
      {NewValueNode(prim::kPrimTupleGetItem), cond_carry, NewValueNode(static_cast<int64_t>(1))});
    const std::string less_module_name = "mindspore.ops.composite.multitype_ops.less_impl";
    ValuePtr less_op = prim::GetPythonOps("less", less_module_name);
    CNodePtr cond_node = cond_func_graph->NewCNodeInOrder({NewValueNode(less_op), cond_index, cond_uppper});
    cond_func_graph->set_output(cond_node);
    return cond_func_graph;
  }

  /**
   * \brief Build while loop func graph.
   *
   * \example
   *     # python
   *     def loop_func_graph(loop_carry):
   *       loop_index, loop_upper, loop_x = loop_carry
   *       body_output = loop_func_node(loop_index, loop_x)
   *       new_index = loop_index + 1
   *       return (new_index, loop_upper, body_output)
   *
   * \param[in] loop_func_node loop function node.
   *
   * \return The built while loop func graph.
   **/
  FuncGraphPtr BuildLoopFuncGraphOfWhileLoop(const FuncGraphPtr &loop_func_node) {
    auto loop_func_graph = std::make_shared<FuncGraph>();
    auto loop_carry = loop_func_graph->add_parameter();
    auto loop_index = loop_func_graph->NewCNodeInOrder(
      {NewValueNode(prim::kPrimTupleGetItem), loop_carry, NewValueNode(static_cast<int64_t>(0))});
    auto loop_uppper = loop_func_graph->NewCNodeInOrder(
      {NewValueNode(prim::kPrimTupleGetItem), loop_carry, NewValueNode(static_cast<int64_t>(1))});
    auto loop_x = loop_func_graph->NewCNodeInOrder(
      {NewValueNode(prim::kPrimTupleGetItem), loop_carry, NewValueNode(static_cast<int64_t>(2))});
    std::string add_module_name = "mindspore.ops.composite.multitype_ops.add_impl";
    ValuePtr add_op = prim::GetPythonOps("add", add_module_name);
    auto body_output = loop_func_graph->NewCNodeInOrder({NewValueNode(loop_func_node), loop_index, loop_x});
    auto new_index =
      loop_func_graph->NewCNodeInOrder({NewValueNode(add_op), loop_index, NewValueNode(static_cast<int64_t>(1))});
    auto loop_output =
      loop_func_graph->NewCNodeInOrder({NewValueNode(prim::kPrimMakeTuple), new_index, loop_uppper, body_output});
    loop_func_graph->set_output(loop_output);
    return loop_func_graph;
  }
};

class RaiseEvaluator final : public TransitionPrimEvaluator {
 public:
  RaiseEvaluator() : TransitionPrimEvaluator("RaiseEvaluator") {}
  ~RaiseEvaluator() override = default;
  MS_DECLARE_PARENT(RaiseEvaluator, TransitionPrimEvaluator);
  EvalResultPtr EvalPrim(const AnalysisEnginePtr &, const AbstractBasePtrList &args_abs_list, const ConfigPtr &,
                         const AnfNodeConfigPtr &out_conf) override {
    MS_EXCEPTION_IF_NULL(out_conf);
    // Handle for DDE.
    for (size_t i = 0; i < args_abs_list.size(); ++i) {
      MS_EXCEPTION_IF_NULL(args_abs_list[i]);
      if (args_abs_list[i]->isa<abstract::AbstractSequence>()) {
        MS_LOG(DEBUG) << "Primitive \'Raise\' is consuming tuple/list arguments[" << i
                      << "]: " << args_abs_list[i]->ToString();
        SetSequenceElementsUseFlagsRecursively(args_abs_list[i], true);
      }
    }
    auto node = out_conf->node();
    MS_EXCEPTION_IF_NULL(node);
    auto cur_graph = node->func_graph();
    MS_EXCEPTION_IF_NULL(cur_graph);
    if (args_abs_list.empty()) {
      // Process raise.
      MS_LOG(INTERNAL_EXCEPTION) << "No active exception to reraise.";
    }
    const auto &cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);

    // Return Any directly if meet variable condition or content.
    bool is_variable_condition = raiseutils::HasVariableCondition(cur_graph);
    bool has_variable = false;
    size_t index_begin = 2;
    size_t index_end = cnode->size() - 1;
    for (size_t index = index_begin; index < cnode->size(); ++index) {
      if (raiseutils::CheckHasVariable(args_abs_list[index - 1])) {
        has_variable = true;
        break;
      }
    }
    if (is_variable_condition || has_variable) {
      AbstractBasePtr res = std::make_shared<AbstractNegligible>();
      cnode->set_has_side_effect_node(true);
      cur_graph->set_has_side_effect_node(true);
      MS_LOG(DEBUG) << "Found side-effect, cnode: " << cnode->DebugString()
                    << ", func_graph: " << cur_graph->ToString();
      auto infer_result = std::make_shared<EvalResult>(res, std::make_shared<AttrValueMap>());
      evaluator_cache_mgr_->SetValue(args_abs_list, infer_result);
      return infer_result;
    }

    // Continue to handle raise in compile time.
    std::shared_ptr<raiseutils::KeyValueInfo> key_value = std::make_shared<raiseutils::KeyValueInfo>();
    std::string exception_type =
      raiseutils::GetExceptionType(args_abs_list[0], cnode->input(index_end), key_value, false);
    std::string exception_string;
    // Process raise ValueError()
    if (args_abs_list.size() == 1) {
      RaiseConstant(exception_type);
    }
    // Processed in units of nodes. Raise ValueError(xxxx)
    for (size_t index = index_begin; index < cnode->size() - 1; ++index) {
      const auto input = cnode->input(index);
      auto input_abs = args_abs_list[index - 1];
      MS_EXCEPTION_IF_NULL(input_abs);
      const bool need_symbol = raiseutils::CheckNeedSymbol(input_abs);
      if (need_symbol) {
        exception_string += "'";
      }
      bool need_comma = !IsPrimitiveCNode(input, prim::kPrimMakeTuple);
      exception_string += raiseutils::GetExceptionString(input_abs, input, key_value, need_symbol, need_comma);
      if (need_symbol) {
        exception_string += "'";
      }
      constexpr auto end_index = 2;
      if (index < cnode->size() - end_index) {
        exception_string += ", ";
      }
    }
    bool need_out_symbol = cnode->size() > 4;
    if (need_out_symbol) {
      exception_string = "(" + exception_string + ")";
    }
    RaiseConstant(exception_type, exception_string);
    MS_LOG(EXCEPTION) << "Constant raise is not raising exception correctly";
  }

 private:
  void RaiseConstant(const std::string &type, const std::string &exception_string = "") {
    auto iter = exception_types_map.find(type);
    if (iter == exception_types_map.end()) {
      MS_LOG(EXCEPTION) << "Unsupported exception type: " << type
                        << ". Raise only support some Python standard exception types: "
                        << SupportedExceptionsToString();
    }
    ExceptionType error_type = iter->second;
    if (exception_string.empty()) {
      MS_EXCEPTION(error_type);
    } else {
      MS_EXCEPTION(error_type) << exception_string;
    }
  }
};

class WithEnterEvaluator final : public TransitionPrimEvaluator {
 public:
  WithEnterEvaluator() : TransitionPrimEvaluator("WithEnterEvaluator") {}
  ~WithEnterEvaluator() override = default;
  MS_DECLARE_PARENT(WithEnterEvaluator, TransitionPrimEvaluator);
  EvalResultPtr EvalPrim(const AnalysisEnginePtr &engine, const AbstractBasePtrList &args_abs_list, const ConfigPtr &,
                         const AnfNodeConfigPtr &out_conf) override {
    MS_EXCEPTION_IF_NULL(out_conf);
    MS_EXCEPTION_IF_NULL(out_conf->node());
    auto node = out_conf->node()->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(node);
    auto cur_graph = node->func_graph();
    MS_EXCEPTION_IF_NULL(cur_graph);

    if (args_abs_list.size() != 1) {
      MS_LOG(INTERNAL_EXCEPTION) << "The enter node has wrong input." << node->debug_info();
    }

    // Check class object
    constexpr size_t cls_index = 0;
    MS_EXCEPTION_IF_NULL(args_abs_list[cls_index]);
    auto cls_val = args_abs_list[cls_index]->BuildValue();
    MS_EXCEPTION_IF_NULL(cls_val);
    auto value_obj = cls_val->cast<parse::MsClassObjectPtr>();
    if (value_obj == nullptr) {
      MS_EXCEPTION(TypeError) << "Only support jit_class instance, but got " << cls_val->ToString();
    }
    auto cls_obj = value_obj->obj();

    const std::string call_func = "__enter__";
    if (!py::hasattr(cls_obj, common::SafeCStr(call_func))) {
      MS_LOG(EXCEPTION) << value_obj->name() << " has no " << call_func << " function, please check the code.";
    }
    py::object call_obj = py::getattr(cls_obj, common::SafeCStr(call_func));
    FuncGraphPtr call_func_graph = parse::ConvertToFuncGraph(call_obj);
    if (call_func_graph == nullptr) {
      MS_LOG(INTERNAL_EXCEPTION) << "Parse python object " << call_func << " failed.";
    }
    FuncGraphManagerPtr manager = engine->func_graph_manager();
    MS_EXCEPTION_IF_NULL(manager);
    manager->AddFuncGraph(call_func_graph);

    std::vector<AnfNodePtr> enter_inputs{NewValueNode(call_func_graph)};
    //  __enter__(self)
    auto call_enter_node = cur_graph->NewCNodeInOrder(enter_inputs);
    // Continue to eval call_enter_node.
    AnfNodeConfigPtr fn_conf = engine->MakeConfig(call_enter_node, out_conf->context(), out_conf->func_graph());
    return engine->ForwardConfig(out_conf, fn_conf);
  }
};

class WithExitEvaluator final : public TransitionPrimEvaluator {
 public:
  WithExitEvaluator() : TransitionPrimEvaluator("WithExitEvaluator") {}
  ~WithExitEvaluator() override = default;
  MS_DECLARE_PARENT(WithExitEvaluator, TransitionPrimEvaluator);
  EvalResultPtr EvalPrim(const AnalysisEnginePtr &engine, const AbstractBasePtrList &args_abs_list, const ConfigPtr &,
                         const AnfNodeConfigPtr &out_conf) override {
    MS_EXCEPTION_IF_NULL(out_conf);
    MS_EXCEPTION_IF_NULL(out_conf->node());
    auto node = out_conf->node()->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(node);
    auto cur_graph = node->func_graph();
    MS_EXCEPTION_IF_NULL(cur_graph);

    if (args_abs_list.size() != 1) {
      MS_LOG(INTERNAL_EXCEPTION) << "The exit node has wrong input." << node->debug_info();
    }

    // Check class object
    constexpr size_t cls_index = 0;
    MS_EXCEPTION_IF_NULL(args_abs_list[cls_index]);
    auto cls_val = args_abs_list[cls_index]->BuildValue();
    MS_EXCEPTION_IF_NULL(cls_val);
    auto value_obj = cls_val->cast<parse::MsClassObjectPtr>();
    if (value_obj == nullptr) {
      MS_EXCEPTION(TypeError) << "Only support jit_class instance, but got " << cls_val->ToString();
    }
    auto cls_obj = value_obj->obj();

    const std::string call_func = "__exit__";
    if (!py::hasattr(cls_obj, common::SafeCStr(call_func))) {
      MS_LOG(EXCEPTION) << value_obj->name() << " has no " << call_func << " function, please check the code.";
    }
    py::object call_obj = py::getattr(cls_obj, common::SafeCStr(call_func));
    FuncGraphPtr call_func_graph = parse::ConvertToFuncGraph(call_obj);
    if (call_func_graph == nullptr) {
      MS_LOG(INTERNAL_EXCEPTION) << "Parse python object " << call_func << " failed.";
    }
    FuncGraphManagerPtr manager = engine->func_graph_manager();
    MS_EXCEPTION_IF_NULL(manager);
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

class CondEvaluator final : public TransitionPrimEvaluator {
 public:
  CondEvaluator() : TransitionPrimEvaluator("CondEvaluator") {}
  ~CondEvaluator() override = default;
  MS_DECLARE_PARENT(CondEvaluator, TransitionPrimEvaluator);
  EvalResultPtr EvalPrim(const AnalysisEnginePtr &engine, const AbstractBasePtrList &args_abs_list, const ConfigPtr &,
                         const AnfNodeConfigPtr &out_conf) override {
    auto res_abstract = EvalUndeterminedArgs(args_abs_list);
    if (res_abstract != nullptr) {
      return res_abstract;
    }
    MS_EXCEPTION_IF_NULL(out_conf);
    MS_EXCEPTION_IF_NULL(out_conf->node());
    auto cnode = out_conf->node()->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    auto cur_graph = cnode->func_graph();
    MS_EXCEPTION_IF_NULL(cur_graph);
    constexpr size_t input_size = 2;
    if (args_abs_list.size() != input_size) {
      MS_LOG(INTERNAL_EXCEPTION) << "The input size to cond node should be " << std::to_string(input_size)
                                 << ", but got " << std::to_string(args_abs_list.size());
    }

    AnfNodePtr new_node = nullptr;
    constexpr size_t cond_abs_index = 0;
    constexpr size_t cond_input_index = 1;
    constexpr size_t flag_input_index = 2;
    auto cond_abs = args_abs_list[cond_abs_index];
    auto cond_node = cnode->input(cond_input_index);
    auto flag_node = cnode->input(flag_input_index);
    MS_EXCEPTION_IF_NULL(cond_abs);
    if (cond_abs->isa<AbstractAny>()) {
      // If the input to cond node is AbstractAny, genenrate pyexecute node 'bool(input)';
      const auto script_str = std::make_shared<StringImm>("bool(__input__)");

      const auto input_str = std::make_shared<StringImm>("__input__");
      std::vector<AnfNodePtr> key_value_names_list{NewValueNode(prim::kPrimMakeTuple)};
      (void)key_value_names_list.emplace_back(NewValueNode(input_str));
      const auto key_value_name_tuple = cur_graph->NewCNode(key_value_names_list);

      std::vector<AnfNodePtr> key_value_list{NewValueNode(prim::kPrimMakeTuple), cond_node};
      const auto key_value_tuple = cur_graph->NewCNode(key_value_list);
      new_node =
        fallback::CreatePyExecuteCNodeInOrder(cnode, NewValueNode(script_str), key_value_name_tuple, key_value_tuple);
      fallback::SetRealType<AnfNode, Type>(new_node, std::make_shared<TensorType>(kBool));
      fallback::SetRealShape(new_node, std::make_shared<abstract::Shape>(std::vector<int64_t>{Shape::kShapeDimAny}));
    } else if (cond_abs->isa<AbstractTensor>() && is_while_condition(flag_node)) {
      // When the condition of while is a tensor, do not use standard_method.tensor_bool
      // to avoid turning the tensor into scalar to cause a loop.
      constexpr auto operations_module = "mindspore.ops.operations";
      auto cast_op = python_adapter::GetPyFn(operations_module, kCastOpName)();
      auto cast_node = NewValueNode(parse::data_converter::PyDataToValue(cast_op));
      auto type_node = NewValueNode(TypeIdToType(kNumberTypeBool));
      new_node = cur_graph->NewCNodeInOrder({cast_node, cond_node, type_node});
      new_node->set_debug_info(cnode->debug_info());
    } else if (cond_abs->isa<AbstractFunction>()) {
      auto abs = std::make_shared<AbstractScalar>(std::make_shared<BoolImm>(true), kBool);
      return std::make_shared<EvalResult>(abs, std::make_shared<AttrValueMap>());
    } else {
      // The logic of truth value testing:
      //   1. If the object has __bool__ attribute, call __bool__()
      //   2. Else if the object has __len__ attribute, call __len__()
      //   3. Else return true.
      auto cond_type = cond_abs->BuildType();
      MS_EXCEPTION_IF_NULL(cond_type);
      auto cond_type_id = cond_type->type_id();
      constexpr auto bool_attr_str = "__bool__";
      constexpr auto len_attr_str = "__len__";
      ValuePtr prim_func;
      if (!pipeline::Resource::GetMethodPtr(cond_type_id, bool_attr_str).empty()) {
        prim_func = prim::GetPythonOps(parse::NAMED_PRIMITIVE_BOOL);
      } else if (!pipeline::Resource::GetMethodPtr(cond_type_id, len_attr_str).empty()) {
        prim_func = prim::GetPythonOps(parse::NAMED_PRIMITIVE_CHECK_LEN);
      } else {
        prim_func = prim::GetPythonOps(parse::NAMED_PRIMITIVE_REAL_BOOL);
      }
      auto prim_fg = dyn_cast<FuncGraph>(prim_func);
      MS_EXCEPTION_IF_NULL(prim_fg);
      auto mng = cur_graph->manager();
      MS_EXCEPTION_IF_NULL(mng);
      prim_fg->set_manager(mng);
      new_node = cur_graph->NewCNodeInOrder({NewValueNode(prim_fg), cond_node});
    }
    AnfNodeConfigPtr fn_conf = engine->MakeConfig(new_node, out_conf->context(), out_conf->func_graph());
    return engine->ForwardConfig(out_conf, fn_conf);
  }

  bool is_while_condition(const AnfNodePtr &flag_node) const {
    MS_EXCEPTION_IF_NULL(flag_node);
    auto vnode = GetValueNode(flag_node);
    MS_EXCEPTION_IF_NULL(vnode);
    return GetValue<bool>(vnode);
  }
};

class DoUnpackCallEvaluator final : public TransitionPrimEvaluator {
 public:
  DoUnpackCallEvaluator() : TransitionPrimEvaluator("DoUnpackCallEvaluator") {}
  ~DoUnpackCallEvaluator() override = default;
  MS_DECLARE_PARENT(DoUnpackCallEvaluator, TransitionPrimEvaluator);
  EvalResultPtr EvalPrim(const AnalysisEnginePtr &engine, const AbstractBasePtrList &args_abs_list, const ConfigPtr &,
                         const AnfNodeConfigPtr &out_conf) override {
    auto node = out_conf->node();
    MS_EXCEPTION_IF_NULL(node);
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    auto fg = cnode->func_graph();
    MS_EXCEPTION_IF_NULL(fg);
    // Unpack call for primitive.
    auto unpack_prim_node = UnpackCallForPrimitive(args_abs_list, cnode, engine, out_conf);
    if (unpack_prim_node != nullptr) {
      auto new_node = GetInputsAfterUnpackCall(unpack_prim_node, engine, out_conf);
      MS_LOG(DEBUG) << "Unpack call for primitive: convert " << cnode->DebugString() << " to "
                    << new_node->DebugString();
      AnfNodeConfigPtr fn_conf = engine->MakeConfig(new_node, out_conf->context(), out_conf->func_graph());
      return engine->ForwardConfig(out_conf, fn_conf);
    }
    // Unpack call for functional.
    auto unpack_functional_node = UnpackCallForFunctional(args_abs_list, cnode, engine, out_conf);
    if (unpack_functional_node != nullptr) {
      auto new_node = GetInputsAfterUnpackCall(unpack_functional_node, engine, out_conf);
      MS_LOG(DEBUG) << "Unpack call for functional: convert " << cnode->DebugString() << " to "
                    << new_node->DebugString();
      AnfNodeConfigPtr fn_conf = engine->MakeConfig(new_node, out_conf->context(), out_conf->func_graph());
      return engine->ForwardConfig(out_conf, fn_conf);
    }
    // Convert DoUnpackCall into Unpackcall which inherits from MetaFuncGraph.
    auto unpack_call = std::make_shared<prim::UnpackCall>(parse::NAMED_METAGRAPH_UNPACKCALL);
    AnfNodePtrList unpack_call_inputs{NewValueNode(unpack_call)};
    constexpr size_t input_start_index = 1;
    (void)std::copy(cnode->inputs().begin() + input_start_index, cnode->inputs().end(),
                    std::back_inserter(unpack_call_inputs));
    auto unpack_call_node = fg->NewCNodeInOrder(unpack_call_inputs);
    MS_LOG(DEBUG) << "Convert DoUnpackCall into Unpackcall: convert " << cnode->DebugString() << " to "
                  << unpack_call_node->DebugString();
    AnfNodeConfigPtr fn_conf = engine->MakeConfig(unpack_call_node, out_conf->context(), out_conf->func_graph());
    return engine->ForwardConfig(out_conf, fn_conf);
  }

  CNodePtr UnpackCallForFunctional(const AbstractBasePtrList &args_abs_list, const CNodePtr &cnode,
                                   const AnalysisEnginePtr &engine, const AnfNodeConfigPtr &out_conf) {
    constexpr auto index_fn = 0;
    auto fn_abs = args_abs_list[index_fn];
    MS_EXCEPTION_IF_NULL(fn_abs);
    // {DoUnpack, GetAttr(x, method_name), arg1, arg2} -> {Functional, x, unpack_arg1, unpack_arg2}
    if (fn_abs->isa<PartialAbstractClosure>()) {
      auto partial_abs = fn_abs->cast<PartialAbstractClosurePtr>();
      auto partial_fn_abs = partial_abs->fn();
      MS_EXCEPTION_IF_NULL(partial_fn_abs);
      if (!partial_fn_abs->isa<FunctionalAbstractClosure>()) {
        return nullptr;
      }
      const auto &method_name = partial_fn_abs->cast<FunctionalAbstractClosurePtr>()->name();
      const auto &functional = BuildMethodFunctional(method_name);
      // Get x.
      constexpr auto index_input = 1;
      auto op_node = cnode->input(index_input);
      if (!IsGetAttrNode(op_node)) {
        return nullptr;
      }
      auto x_node = op_node->cast<CNodePtr>()->input(index_input);
      AnfNodePtrList new_inputs{NewValueNode(functional), x_node};
      // Create Functional code.
      constexpr auto input_start_index = 2;
      (void)std::copy(cnode->inputs().cbegin() + input_start_index, cnode->inputs().cend(),
                      std::back_inserter(new_inputs));
      auto fg = cnode->func_graph();
      MS_EXCEPTION_IF_NULL(fg);
      return fg->NewCNodeInOrder(new_inputs);
    }
    return nullptr;
  }

  CNodePtr UnpackCallForPrimitive(const AbstractBasePtrList &args_abs_list, const CNodePtr &cnode,
                                  const AnalysisEnginePtr &engine, const AnfNodeConfigPtr &out_conf) {
    constexpr auto fn_index = 0;
    auto fn_abs = args_abs_list[fn_index];
    auto fg = out_conf->func_graph();
    MS_EXCEPTION_IF_NULL(fg);
    // {DoUnpackCall, {ClassType, arg1, arg2, ...}, input1, input2, ...}
    if (fn_abs->isa<PrimInstanceAbstractClosure>()) {
      auto prim_instance_abs = fn_abs->cast<PrimInstanceAbstractClosurePtr>();
      return UnpackCallForPrimInstanceAbstract(prim_instance_abs, cnode, engine, out_conf);
    }
    // {DoUnpackCall, ClassType, arg1, arg2, ...}
    if (!fn_abs->isa<PartialAbstractClosure>()) {
      return nullptr;
    }
    auto partial_abs = fn_abs->cast<PartialAbstractClosurePtr>();
    auto partial_fn_abs = partial_abs->fn();
    if (partial_fn_abs == nullptr || !(partial_fn_abs->isa<PrimitiveAbstractClosure>())) {
      return nullptr;
    }
    auto partial_prim = partial_fn_abs->cast<PrimitiveAbstractClosurePtr>()->prim();
    if (IsPrimitiveEquals(partial_prim, prim::kPrimCreateInstance)) {
      constexpr auto class_type_index = 0;
      auto class_obj = GetPythonObject(partial_abs->args()[class_type_index]);
      // Check if primitive.
      if (!py::hasattr(class_obj, PYTHON_PRIMITIVE_FLAG)) {
        return nullptr;
      }
      py::module mod = python_adapter::GetPyModule(parse::PYTHON_MOD_PARSE_MODULE);
      std::string prim_name =
        python_adapter::CallPyModFn(mod, parse::PYTHON_MOD_GET_MS_CLASS_NAME, class_obj).cast<std::string>();
      return UnpackCallForCreateInstance(prim_name, cnode, fg);
    }
    return nullptr;
  }

  CNodePtr UnpackCallForCreateInstance(const std::string &prim_name, const CNodePtr &cnode, const FuncGraphPtr &fg) {
    // {DoUnpackCall, ClassType, arg1, arg2, ...} -> {ClassType, unpack_arg1, unpack_arg2, ...}
    if (ops::GetOpDef(prim_name) == nullptr) {
      return nullptr;
    }
    constexpr auto class_type_index = 1;
    constexpr auto input_start_index = 2;
    AnfNodePtrList new_inputs{cnode->input(class_type_index)};
    (void)std::copy(cnode->inputs().begin() + input_start_index, cnode->inputs().end(), std::back_inserter(new_inputs));
    return fg->NewCNodeInOrder(new_inputs);
  }

  CNodePtr UnpackCallForPrimInstanceAbstract(const PrimInstanceAbstractClosurePtr &prim_instance_abs,
                                             const CNodePtr &cnode, const AnalysisEnginePtr &engine,
                                             const AnfNodeConfigPtr &out_conf) {
    // {DoUnpackCall, {ClassType, arg1, arg2, ...}, input1, input2, ...} ->
    // {DoTrans, unpack_input1, unpack_input2, ..., arg1, arg2, ...}
    auto fg = out_conf->func_graph();
    MS_EXCEPTION_IF_NULL(fg);
    const auto &prim_name = prim_instance_abs->prim_name();
    auto op_def = ops::GetOpDef(prim_name);
    if (op_def == nullptr) {
      return nullptr;
    }
    auto do_trans_primfunc = std::make_shared<prim::DoTransPrimitiveFunction>(std::make_shared<Primitive>(prim_name));
    AnfNodePtrList new_inputs{NewValueNode(do_trans_primfunc)};
    // Get call args.
    constexpr auto call_start_index = 2;
    (void)std::copy(cnode->inputs().begin() + call_start_index, cnode->inputs().end(), std::back_inserter(new_inputs));
    // Get init args.
    auto instance_node = prim_instance_abs->instance_node();
    MS_EXCEPTION_IF_NULL(instance_node);
    auto instance_cnode = instance_node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(instance_cnode);
    AnfNodePtrList init_args_list;
    constexpr auto init_start_index = 1;
    (void)std::copy(instance_cnode->inputs().begin() + init_start_index, instance_cnode->inputs().end(),
                    std::back_inserter(init_args_list));
    std::vector<ops::OpInputArg> op_init_args;
    auto op_args = op_def->args_;
    for (const auto &op_arg : op_args) {
      if (op_arg.as_init_arg_) {
        (void)op_init_args.emplace_back(op_arg);
      }
    }
    auto eval_func = [&engine, &out_conf](const AnfNodePtr &node) {
      AnfNodeConfigPtr config = engine->MakeConfig(node, out_conf->context(), out_conf->func_graph());
      MS_EXCEPTION_IF_NULL(config);
      const auto &eval_result = config->ObtainEvalResult();
      MS_EXCEPTION_IF_NULL(eval_result);
      return eval_result->abstract();
    };
    // Get init arguments(including default arguments).
    auto init_inputs = GeneratePrimitiveDefaultArgs(prim_name, init_args_list, op_init_args, eval_func, fg);
    (void)std::copy(init_inputs.begin(), init_inputs.end(), std::back_inserter(new_inputs));
    return fg->NewCNodeInOrder(new_inputs);
  }
};

class TraceGraphEvaluator final : public TransitionPrimEvaluator {
 public:
  TraceGraphEvaluator() : TransitionPrimEvaluator("TraceGraphEvaluator") {}
  ~TraceGraphEvaluator() override = default;
  MS_DECLARE_PARENT(TraceGraphEvaluator, TransitionPrimEvaluator);
  EvalResultPtr EvalPrim(const AnalysisEnginePtr &engine, const AbstractBasePtrList &args_abs_list, const ConfigPtr &,
                         const AnfNodeConfigPtr &out_conf) override {
    MS_EXCEPTION_IF_NULL(out_conf);
    const auto &node = out_conf->node();
    MS_EXCEPTION_IF_NULL(node);
    const auto &cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    const auto &fg = cnode->func_graph();
    MS_EXCEPTION_IF_NULL(fg);
    const auto &trace_recorder = trace::TraceRecorder::GetInstance();
    const auto &trace_top_graph = trace_recorder->InitTopGraph(node->debug_info());
    py::tuple py_inputs(args_abs_list.size());
    for (size_t i = 0; i < args_abs_list.size(); ++i) {
      const auto &param = trace_top_graph->add_parameter();
      param->set_abstract(args_abs_list[i]);
      py_inputs[i] = trace_recorder->InitTraceGraphInputs(args_abs_list[i], param);
    }
    const auto &trace_prim_node = cnode->input(0);
    const auto &trace_prim_cnode = trace_prim_node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(trace_prim_cnode);
    constexpr auto obj_index = 3;
    const auto &obj_node = trace_prim_cnode->input(obj_index);
    if (!obj_node->isa<ValueNode>()) {
      MS_LOG(EXCEPTION) << "Trace node missing function object.";
    }
    const auto &obj_value_node = obj_node->cast<ValueNodePtr>();
    const auto &obj_value = GetValueNode<parse::InterpretedObjectPtr>(obj_value_node);
    MS_EXCEPTION_IF_NULL(obj_value);
    const py::object &func_obj = obj_value->obj();
    py::object cell_obj = py::none();
    constexpr auto cell_index = 4;
    if (trace_prim_cnode->inputs().size() > cell_index) {
      const auto &cell_node = trace_prim_cnode->input(cell_index);
      if (!cell_node->isa<ValueNode>()) {
        MS_LOG(EXCEPTION) << "Trace node missing cell object.";
      }
      const auto &cell_value_node = cell_node->cast<ValueNodePtr>();
      MS_EXCEPTION_IF_NULL(cell_value_node);
      const auto &cell_value = GetValueNode<parse::InterpretedObjectPtr>(cell_value_node);
      MS_EXCEPTION_IF_NULL(cell_value);
      cell_obj = cell_value->obj();
    }
    py::tuple output;
    try {
      output = python_adapter::CallPyFn("mindspore.graph.jit_trace", "nested_run", func_obj, cell_obj, *py_inputs);
    } catch (const std::exception &e) {
      MS_LOG(EXCEPTION) << "Encounter error: " << e.what() << " When compiling nested trace graph.";
    }
    const py::list &file_names = output[0];
    const py::list &linenos = output[1];
    const py::tuple &output_args = output[2];
    const auto &func_graph = trace_recorder->BuildEndGraph(file_names, linenos, py::args(output_args), true);
    auto eng = out_conf->engine();
    AddToManager(eng, trace_top_graph);
    AnfNodePtrList new_inputs{NewValueNode(trace_top_graph)};
    const auto &origin_inputs = cnode->inputs();
    for (size_t i = 1; i < origin_inputs.size(); ++i) {
      (void)new_inputs.emplace_back(origin_inputs[i]);
    }
    const auto &new_node = fg->NewCNodeInOrder(new_inputs);
    AnfNodeConfigPtr fn_conf = engine->MakeConfig(new_node, out_conf->context(), out_conf->func_graph());
    return engine->ForwardConfig(out_conf, fn_conf);
  }
};

class StreamSendEvaluator final : public TransitionPrimEvaluator {
 public:
  StreamSendEvaluator() : TransitionPrimEvaluator("StreamSendEvaluator") {}
  ~StreamSendEvaluator() override = default;
  MS_DECLARE_PARENT(StreamSendEvaluator, TransitionPrimEvaluator);
  EvalResultPtr EvalPrim(const AnalysisEnginePtr &, const AbstractBasePtrList &args_abs_list, const ConfigPtr &,
                         const AnfNodeConfigPtr &out_conf) override {
    auto infer_result = std::make_shared<EvalResult>(args_abs_list[0], std::make_shared<AttrValueMap>());
    evaluator_cache_mgr_->SetValue(args_abs_list, infer_result);
    return infer_result;
  }
};

class StreamRecvEvaluator final : public TransitionPrimEvaluator {
 public:
  StreamRecvEvaluator() : TransitionPrimEvaluator("StreamRecvEvaluator") {}
  ~StreamRecvEvaluator() override = default;
  MS_DECLARE_PARENT(StreamRecvEvaluator, TransitionPrimEvaluator);
  EvalResultPtr EvalPrim(const AnalysisEnginePtr &, const AbstractBasePtrList &args_abs_list, const ConfigPtr &,
                         const AnfNodeConfigPtr &out_conf) override {
    auto infer_result = std::make_shared<EvalResult>(args_abs_list[0], std::make_shared<AttrValueMap>());
    evaluator_cache_mgr_->SetValue(args_abs_list, infer_result);
    return infer_result;
  }
};

class GetStreamInfoEvaluator final : public TransitionPrimEvaluator {
 public:
  GetStreamInfoEvaluator() : TransitionPrimEvaluator("GetStreamInfoEvaluator") {}
  ~GetStreamInfoEvaluator() override = default;
  MS_DECLARE_PARENT(GetStreamInfoEvaluator, TransitionPrimEvaluator);
  EvalResultPtr EvalPrim(const AnalysisEnginePtr &, const AbstractBasePtrList &args_abs_list, const ConfigPtr &,
                         const AnfNodeConfigPtr &out_conf) override {
    // Set AbstractAny to ensure that the GetStreamInfo operator in constant loops remains in the subgraph, preventing
    // it from being optimized away by constant folding.
    const auto any_abstract = std::make_shared<AbstractAny>();
    auto infer_result = std::make_shared<EvalResult>(any_abstract, std::make_shared<AttrValueMap>());
    evaluator_cache_mgr_->SetValue(args_abs_list, infer_result);
    return infer_result;
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
    {prim::kPrimScalarPow, R{prim::ScalarPow, true, nullptr, true}},
    {prim::kPrimScalarUadd, R{prim::ScalarUAdd, true, nullptr, true}},
    {prim::kPrimScalarUsub, R{prim::ScalarUSub, true, nullptr, true}},
    {prim::kPrimScalarLog, R{prim::ScalarLog, true, nullptr, true}},
    {prim::kPrimBitXor, R{prim::BitXor, true, nullptr, true}},
    {prim::kPrimBitLeftShift, R{prim::BitLeftShift, true, nullptr, true}},
    {prim::kPrimBitRightShift, R{prim::BitRightShift, true, nullptr, true}},
    {prim::kPrimScalarNe, R{prim::ScalarNe, true, std::make_shared<Bool>(), true}},
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

PrimEvaluatorMap prim_evaluator_constructors = PrimEvaluatorMap();
std::mutex PrimEvaluatorConstructorMutex;

void InitPrimEvaluatorConstructors() {
  PrimEvaluatorMap &constructor = prim_evaluator_constructors;

  for (const auto &iter : GetPrimitiveInferMap()) {
    constructor[iter.first] = InitStandardPrimEvaluator(iter.first, iter.second);
  }

  for (const auto &iter : GetUniformPrimitiveToImplMap()) {
    constructor[iter.first] =
      InitUniformPrimEvaluator(iter.first, iter.second.impl_, iter.second.eval_value_, iter.second.specify_out_type_);
  }
  constructor[prim::kPrimEmbed] = std::make_shared<EmbedEvaluator>();
  constructor[prim::kPrimRefToEmbed] = std::make_shared<RefToEmbedEvaluator>();
  constructor[prim::kPrimGetAttr] = std::make_shared<GetAttrEvaluator>();
  constructor[prim::kPrimSetAttr] = std::make_shared<SetAttrEvaluator>();
  constructor[prim::kPrimResolve] = std::make_shared<ResolveEvaluator>();
  constructor[prim::kPrimCreateInstance] = std::make_shared<CreateInstanceEvaluator>();
  constructor[prim::kPrimPartial] = std::make_shared<PartialEvaluator>();
  constructor[prim::kPrimPyInterpret] = std::make_shared<PyInterpretEvaluator>();
  constructor[prim::kPrimMakeTuple] = std::make_shared<MakeTupleEvaluator>();
  constructor[prim::kPrimMakeList] = std::make_shared<MakeListEvaluator>();
  constructor[prim::kPrimRaise] = std::make_shared<RaiseEvaluator>();
  constructor[prim::kPrimWithEnter] = std::make_shared<WithEnterEvaluator>();
  constructor[prim::kPrimDoUnpackCall] = std::make_shared<DoUnpackCallEvaluator>();
  constructor[prim::kPrimWithExit] = std::make_shared<WithExitEvaluator>();
  constructor[prim::kPrimCond] = std::make_shared<CondEvaluator>();
  constructor[prim::kPrimWhileLoop] = std::make_shared<WhileLoopEvaluator>();
  constructor[prim::kPrimScan] = std::make_shared<ScanEvaluator>();
  constructor[prim::kPrimForiLoop] = std::make_shared<ForiLoopEvaluator>();
  constructor[prim::kPrimTraceGraph] = std::make_shared<TraceGraphEvaluator>();
  constructor[prim::kPrimStreamSend] = std::make_shared<StreamSendEvaluator>();
  constructor[prim::kPrimStreamRecv] = std::make_shared<StreamRecvEvaluator>();
  constructor[prim::kPrimGetStreamInfo] = std::make_shared<GetStreamInfoEvaluator>();
}

void InitBuiltinPrimEvaluatorConstructors() {
  PrimEvaluatorMap &constructor = prim_evaluator_constructors;
  constructor[prim::kPrimInnerAbs] = std::make_shared<InnerAbsEvaluator>();
  constructor[prim::kPrimInnerRound] = std::make_shared<InnerRoundEvaluator>();
}
}  // namespace

void ClearPrimEvaluatorMap() {
  prim_evaluator_constructors.clear();
  GetFrontendPrimitiveInferMapPtr()->clear();
  GetUniformPrimitiveToImplMap().clear();
  prim::GetFunctionalConvertCache().clear();
}

bool IsInWhiteList(const PrimitivePtr &primitive) {
  MS_EXCEPTION_IF_NULL(primitive);

  using WhiteList = mindspore::HashMap<PrimitivePtr, bool, PrimitiveHasher, PrimitiveEqual>;

  static WhiteList whitelist = {{prim::kPrimPartial, true}};
  auto iter = whitelist.find(primitive);
  if (iter != whitelist.end()) {
    return iter->second;
  }

  auto found = abstract::GetFrontendPrimitiveInferImpl(primitive);
  if (found.has_value()) {
    auto infer = found.value();
    return infer.IsInWhiteList();
  }

  auto uni_iter = GetUniformPrimitiveToImplMap().find(primitive);
  if (uni_iter != GetUniformPrimitiveToImplMap().end()) {
    return uni_iter->second.in_white_list_;
  }

  return true;
}

PrimEvaluatorMap &GetPrimEvaluatorConstructors() {
  PrimEvaluatorMap &constructor = prim_evaluator_constructors;
  if (!constructor.empty()) {
    return constructor;
  }
  std::lock_guard<std::mutex> initLock(PrimEvaluatorConstructorMutex);
  if (constructor.empty()) {
    InitPrimEvaluatorConstructors();
    InitBuiltinPrimEvaluatorConstructors();
  }

  return constructor;
}

AbstractBasePtr SensitivityTransform(const AbstractBasePtr &spec) {
  auto f_spec = dyn_cast_ptr<AbstractFunction>(spec);
  if (f_spec != nullptr) {
    return std::make_shared<AbstractScalar>(kValueAny, std::make_shared<EnvType>());
  }
  return spec->Clone();
}
}  // namespace abstract
}  // namespace mindspore
