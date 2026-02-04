/**
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

#include "include/frontend/operator/composite/grad_operation.h"

#include <regex>
#include <vector>
#include "ir/anf.h"
#include "ir/core_ops_primitive.h"
#include "ir/func_graph_flag.h"
#include "ir/graph_utils.h"
#include "utils/trace_info.h"
#include "utils/ms_context.h"
#include "utils/compile_config.h"
#include "abstract/abstract_value.h"
#include "abstract/abstract_function.h"
#include "frontend/operator/ops.h"
#include "frontend/operator/composite/composite.h"
#include "frontend/jit/ps/parse/resolve.h"

namespace mindspore {
namespace prim {
using mindspore::abstract::AbstractBase;
using mindspore::abstract::AbstractBasePtr;
using mindspore::abstract::AbstractClass;
using mindspore::abstract::AbstractFunction;
using mindspore::abstract::AbstractFunctionPtr;
using mindspore::abstract::AbstractNone;
using mindspore::abstract::AbstractSequence;
using mindspore::abstract::AbstractSequencePtr;
using mindspore::abstract::AbstractTuple;
using mindspore::abstract::AbstractTuplePtr;
using mindspore::abstract::AbstractUndetermined;
using mindspore::abstract::EnvSetSparseResultMgr;
using mindspore::abstract::FuncGraphAbstractClosure;

namespace {
bool IsTupleAllTensor(const AbstractTuplePtr &tuple_arg) {
  MS_EXCEPTION_IF_NULL(tuple_arg);
  for (size_t i = 0; i < tuple_arg->size(); ++i) {
    if (!(*tuple_arg)[i]->isa<AbstractUndetermined>() &&
        !((*tuple_arg)[i]->isa<AbstractTuple>() && IsTupleAllTensor((*tuple_arg)[i]->cast<AbstractTuplePtr>()))) {
      return false;
    }
  }
  return true;
}

bool EnableGradFirstForTuple(const AbstractTuplePtr &tuple_arg, bool enable_tuple_grad) {
  return tuple_arg->size() > 1 && (*tuple_arg)[1]->isa<AbstractTuple>() && enable_tuple_grad &&
         IsTupleAllTensor((*tuple_arg)[1]->cast<AbstractTuplePtr>());
}

bool EnableGradForScalar(const AbstractBasePtr &abs) {
  return (MsContext::GetInstance()->get_param<bool>(MS_CTX_GRAD_FOR_SCALAR) ||
          common::GetCompileConfig("GRAD_FOR_SCALAR") == "1") &&
         abs->BuildType() != nullptr && abs->BuildType()->isa<Number>();
}

bool CanGradArgument(const AbstractTuplePtr &tuple_arg, size_t pos) {
  MS_EXCEPTION_IF_NULL(tuple_arg);
  return tuple_arg->size() > pos && (*tuple_arg)[pos] != nullptr &&
         (((*tuple_arg)[pos]->BuildValue() != nullptr && (*tuple_arg)[pos]->BuildValue()->ContainsValueAny()) ||
          EnableGradForScalar((*tuple_arg)[pos]));
}

void GenerateFuncGraphByPosition(const FuncGraphPtr &fg, const AbstractTuplePtr &tuple_arg, const AbstractTuplePtr &pos,
                                 bool return_ids = false) {
  if (pos == nullptr) {
    MS_LOG(EXCEPTION) << "Return grad by position, but the grad_position is empty!";
  }
  if (pos->empty()) {
    MS_LOG(EXCEPTION) << "grad_position should not be empty when grad by position.";
  }
  AnfNodePtr tuple_parameter = fg->add_parameter();
  (void)fg->add_parameter();  // The 'grad_position' parameter.
  // Collect all parameters by 'grad_position'.
  std::vector<AnfNodePtr> pos_elements = {NewValueNode(prim::kPrimMakeTuple)};
  CNodePtr current_element = nullptr;
  for (size_t i = 0; i < pos->size(); ++i) {
    auto val = pos->elements()[i]->BuildValue();
    MS_EXCEPTION_IF_NULL(val);
    auto val_imm = dyn_cast<Int64Imm>(val);
    if (val_imm == nullptr) {
      MS_LOG(INTERNAL_EXCEPTION) << "Position element must be Int64Imm, but got " << val->ToString();
    }
    auto int_val = LongToSize(val_imm->value());
    ++int_val;  // Ignore the env position.
    if (int_val >= tuple_arg->size()) {
      MS_EXCEPTION(IndexError) << "Position index " << (int_val - 1) << " is exceed input size.";
    }
    if (!CanGradArgument(tuple_arg, int_val)) {
      continue;
    }
    current_element =
      fg->NewCNodeInOrder({NewValueNode(prim::kPrimTupleGetItem), tuple_parameter, NewValueNode(SizeToLong(int_val))});
    if (return_ids) {
      current_element =
        fg->NewCNodeInOrder({NewValueNode(kPrimMakeTuple), NewValueNode(SizeToLong(int_val) - 1), current_element});
    }
    pos_elements.push_back(current_element);
  }

  // The returned result may vary for grad result element number.
  // A single value if only one result, a tuple for multiple results, or a empty tuple for no result.
  //
  // Notice that even if the user set 'grad_position' as multiple choices,
  // the 'CanGradArgument' may change it to only one choice or none choice.
  constexpr size_t args_least_size = 2;
  if (pos_elements.size() == args_least_size) {
    fg->set_output(current_element);
  } else if (pos_elements.size() > args_least_size) {
    fg->set_output(fg->NewCNodeInOrder(pos_elements));
  } else {  // The 'pos' is empty AbstractTuple.
    auto empty_tuple_value = std::make_shared<ValueTuple>(ValuePtrList());
    auto empty_tuple = NewValueNode(empty_tuple_value);
    fg->set_output(empty_tuple);
  }
}
}  // namespace

FuncGraphPtr Tail::GenerateTailFuncGraph(const AbstractSequencePtr &sequence_arg) const {
  MS_EXCEPTION_IF_NULL(sequence_arg);
  FuncGraphPtr fg = std::make_shared<FuncGraph>();
  fg->set_flag(FUNC_GRAPH_FLAG_CORE, true);
  if (fg->debug_info() != nullptr) {
    fg->debug_info()->set_name("tail");
  }

  AnfNodePtr tuple_parameter = fg->add_parameter();
  std::vector<AnfNodePtr> elements;
  PrimitivePtr op = nullptr;
  if (sequence_arg->isa<AbstractTuple>()) {
    (void)elements.emplace_back(NewValueNode(prim::kPrimMakeTuple));
    op = prim::kPrimTupleGetItem;
  } else {
    (void)elements.emplace_back(NewValueNode(prim::kPrimMakeList));
    op = prim::kPrimListGetItem;
  }

  // Remove the first element to make a new sequence.
  for (size_t i = 1; i < sequence_arg->size(); ++i) {
    elements.push_back(fg->NewCNodeInOrder({NewValueNode(op), tuple_parameter, NewValueNode(SizeToLong(i))}));
  }
  if (elements.size() > 1) {
    fg->set_output(fg->NewCNodeInOrder(elements));
    return fg;
  }

  // No element left, return empty tuple.
  if (sequence_arg->isa<AbstractTuple>()) {
    auto empty_tuple_value = std::make_shared<ValueTuple>(ValuePtrList());
    auto empty_tuple = NewValueNode(empty_tuple_value);
    fg->set_output(empty_tuple);
  }
  // No element left, return empty list.
  auto empty_tuple_value = std::make_shared<ValueTuple>(ValuePtrList());
  auto empty_tuple = NewValueNode(empty_tuple_value);
  fg->set_output(empty_tuple);
  return fg;
}

FuncGraphPtr Tail::GenerateGradFuncGraph(const AbstractTuplePtr &tuple_arg, const AbstractTuplePtr &position) const {
  MS_EXCEPTION_IF_NULL(tuple_arg);
  FuncGraphPtr fg = std::make_shared<FuncGraph>();
  fg->set_flag(FUNC_GRAPH_FLAG_CORE, true);
  if (fg->debug_info() != nullptr) {
    fg->debug_info()->set_name("grad_tail");
  }

  if (tail_type_ == kGradFirst) {
    AnfNodePtr tuple_parameter = fg->add_parameter();
    if (CanGradArgument(tuple_arg, 1) || EnableGradFirstForTuple(tuple_arg, enable_tuple_grad_first_)) {
      fg->set_output(
        fg->NewCNodeInOrder({NewValueNode(prim::kPrimTupleGetItem), tuple_parameter, NewValueNode(SizeToLong(1))}));
    } else {
      fg->set_output(NewValueNode(std::make_shared<ValueTuple>(ValuePtrList())));
    }
    return fg;
  }

  if (tail_type_ == kGradByPosition) {
    GenerateFuncGraphByPosition(fg, tuple_arg, position, return_ids_);
    return fg;
  }

  if (tail_type_ == kGradAll) {
    AnfNodePtr tuple_parameter = fg->add_parameter();
    std::vector<AnfNodePtr> elements = {NewValueNode(prim::kPrimMakeTuple)};
    for (size_t i = 1; i < tuple_arg->size(); ++i) {
      MS_EXCEPTION_IF_NULL((*tuple_arg)[i]);
      if (CanGradArgument(tuple_arg, i)) {
        elements.push_back(
          fg->NewCNodeInOrder({NewValueNode(prim::kPrimTupleGetItem), tuple_parameter, NewValueNode(SizeToLong(i))}));
      }
    }

    // We should deal with 'get_all=True' as other options later:
    // "The returned result may vary for grad result element number.
    // A single value if only one result, a tuple for multiple results, or a empty tuple for no result.
    //
    // Notice that even if the user set 'get_all=True' and pass multiple inputs,
    // the 'CanGradArgument' may change it to only one gradient output or no gradient."
    constexpr size_t args_least_size = 2;
    if (elements.size() >= args_least_size) {
      fg->set_output(fg->NewCNodeInOrder(elements));
      return fg;
    }
    // Empty tuple.
    auto empty_tuple_value = std::make_shared<ValueTuple>(ValuePtrList());
    auto empty_tuple = NewValueNode(empty_tuple_value);
    fg->set_output(empty_tuple);
    return fg;
  }
  MS_LOG(INTERNAL_EXCEPTION) << "'tail_type_' is not for GradOperation, but " << tail_type_;
}

FuncGraphPtr Tail::GenerateFuncGraph(const AbstractBasePtrList &args_abs_list) {
  // To handle normal tail.
  if (args_abs_list.size() < 1) {
    MS_LOG(EXCEPTION) << "'Tail' requires at least 1 argument, but got " << args_abs_list.size();
  }
  if (tail_type_ >= kNotGrad) {
    AbstractSequencePtr sequence_arg = dyn_cast<AbstractSequence>(args_abs_list[0]);
    if (sequence_arg == nullptr) {
      MS_LOG(EXCEPTION) << "'Tail' arg0 must be tuple or list, but got " << args_abs_list[0]->ToString();
    }
    return GenerateTailFuncGraph(sequence_arg);
  }

  // To handle for GradOperation tail.
  constexpr size_t args_max_size = 2;
  if (args_abs_list.size() > args_max_size) {
    MS_LOG(EXCEPTION) << "'Tail' requires at most 2 arguments for GradOperation, but got " << args_abs_list.size();
  }
  AbstractTuplePtr tuple_arg = dyn_cast<AbstractTuple>(args_abs_list[0]);
  if (tuple_arg == nullptr) {
    MS_LOG(EXCEPTION) << "'Tail' arg0 must be tuple, but got " << args_abs_list[0]->ToString();
  }
  if (args_abs_list.size() == args_max_size) {
    AbstractTuplePtr pos = dyn_cast<AbstractTuple>(args_abs_list[1]);
    if (pos == nullptr) {
      MS_LOG(EXCEPTION) << "'Tail' arg1 'position' must be tuple, but got " << args_abs_list[1]->ToString();
    }
    return GenerateGradFuncGraph(tuple_arg, pos);
  }
  return GenerateGradFuncGraph(tuple_arg);
}

namespace {
AnfNodePtr CreateGradOutputs(const FuncGraphPtr &k_child, const AnfNodePtr &gradient, const AnfNodePtr &f_app,
                             bool has_aux, bool get_value) {
  if (get_value) {
    return k_child->NewCNodeInOrder({NewValueNode(kPrimMakeTuple), f_app, gradient});
  }
  if (!has_aux) {
    return gradient;
  }
  PrimitivePtr get_tuple_item_op = prim::kPrimTupleGetItem;
  PrimitivePtr make_tuple_op = prim::kPrimMakeTuple;
  std::vector<AnfNodePtr> elements = {NewValueNode(make_tuple_op)};
  (void)elements.emplace_back(
    k_child->NewCNodeInOrder({NewValueNode(get_tuple_item_op), f_app, NewValueNode(static_cast<int64_t>(1))}));
  auto aux_output = k_child->NewCNodeInOrder(elements);
  auto unpack_node =
    k_child->NewCNodeInOrder({NewValueNode(get_tuple_item_op), aux_output, NewValueNode(static_cast<int64_t>(0))});
  return k_child->NewCNodeInOrder({NewValueNode(kPrimMakeTuple), gradient, unpack_node});
}
}  // namespace

GradOperation::GradOperation(const std::string &name, bool get_all, bool get_by_list, bool sens_param,
                             bool get_by_position, bool has_aux, bool get_value, bool return_ids, bool merge_forward)
    : MetaFuncGraph(name),
      get_all_(get_all),
      get_by_list_(get_by_list),
      sens_param_(sens_param),
      get_by_position_(get_by_position),
      has_aux_(has_aux),
      get_value_(get_value),
      return_ids_(return_ids),
      merge_forward_(merge_forward) {
  if (get_by_position) {
    signatures_ =
      // def grad(func:read, weight_list:ref, position_list:ref):
      std::vector<Signature>({{"func", SignatureEnumRW::kRWRead, SignatureEnumKind::kKindDefault},
                              {"weight_list", SignatureEnumRW::kRWRef, SignatureEnumKind::kKindDefault},
                              {"position_list", SignatureEnumRW::kRWRef, SignatureEnumKind::kKindDefault}});
  } else if (get_by_list) {
    signatures_ =
      // def grad(func:read, weight_list:ref):
      std::vector<Signature>({{"func", SignatureEnumRW::kRWRead, SignatureEnumKind::kKindDefault},
                              {"weight_list", SignatureEnumRW::kRWRef, SignatureEnumKind::kKindDefault}});
  }
}

FuncGraphPtr GradOperation::GetGrad(const AnfNodePtr &j, const AnfNodePtr &weights, const AnfNodePtr &position,
                                    const FuncGraphPtr &forward_graph, bool is_weights_none) const {
  FuncGraphPtr k_child = std::make_shared<FuncGraph>();
  k_child->set_flag(FUNC_GRAPH_FLAG_CORE, true);
  k_child->set_flag(FUNC_GRAPH_FLAG_K_GRAPH, true);

  AnfNodePtr position_node = nullptr;
  if (position != nullptr) {
    position_node = position;
  }

  std::vector<AnfNodePtr> inputs;
  inputs.push_back(j);
  for (size_t i = 0; i < forward_graph->parameters().size(); ++i) {
    inputs.push_back(k_child->add_parameter());
  }
  auto k_app = k_child->NewCNodeInOrder(inputs);

  auto tuple_get_item = NewValueNode(prim::kPrimTupleGetItem);
  auto f_app = k_child->NewCNodeInOrder({tuple_get_item, k_app, NewValueNode(static_cast<int64_t>(0))});
  auto bprop = k_child->NewCNodeInOrder({tuple_get_item, k_app, NewValueNode(static_cast<int64_t>(1))});

  GradByParameter(k_child, f_app, bprop, weights, position_node, forward_graph, is_weights_none);
  return k_child;
}

CNodePtr GradOperation::SetNodeByParameter(const CNodePtr &grad, const FuncGraphPtr &fg) const {
  CNodePtr fv_bprop;
  if (!weight_value_->isa<abstract::AbstractTuple>()) {
    auto weight_ref = dyn_cast<abstract::AbstractRefTensor>(weight_value_);
    if (weight_ref != nullptr) {
      auto weight_key = weight_ref->ref_key_value()->cast<RefKeyPtr>();
      MS_EXCEPTION_IF_NULL(weight_key);
      auto param_name = weight_key->value();
      fv_bprop = fg->NewCNodeInOrder({NewValueNode(kPrimMakeTuple), NewValueNode(param_name), grad});
    } else {
      MS_LOG(INTERNAL_EXCEPTION) << "Abstract of parameter should be AbstractRefTensor, but got "
                                 << weight_value_->ToString();
    }
  } else {
    std::vector<AnfNodePtr> params;
    abstract::AbstractTuplePtr weight_tuple = weight_value_->cast<abstract::AbstractTuplePtr>();
    const auto &elements = weight_tuple->elements();
    params.push_back(NewValueNode(prim::kPrimMakeTuple));
    for (size_t i = 0; i < weight_tuple->size(); i++) {
      auto weight_ref = dyn_cast<abstract::AbstractRefTensor>(elements[i]);
      if (weight_ref != nullptr) {
        auto weight_key = weight_ref->ref_key_value()->cast<RefKeyPtr>();
        MS_EXCEPTION_IF_NULL(weight_key);
        auto param_name = weight_key->value();
        auto grad_value =
          fg->NewCNodeInOrder({NewValueNode(prim::kPrimTupleGetItem), grad, NewValueNode(static_cast<int64_t>(i))});
        fv_bprop = fg->NewCNodeInOrder({NewValueNode(kPrimMakeTuple), NewValueNode(param_name), grad_value});
        params.push_back(fv_bprop);
      } else {
        MS_LOG(INTERNAL_EXCEPTION) << "Abstract of parameter should be AbstractRefTensor, but got "
                                   << weight_value_->ToString();
      }
    }
    fv_bprop = fg->NewCNodeInOrder(params);
  }
  return fv_bprop;
}

// Do grad by the parameter of GradOperation.
void GradOperation::GradByParameter(const FuncGraphPtr &k_child, const AnfNodePtr &f_app, const AnfNodePtr &bprop,
                                    const AnfNodePtr &weights, const AnfNodePtr &position,
                                    const FuncGraphPtr &forward_graph, bool is_weights_none) const {
  MS_EXCEPTION_IF_NULL(k_child);

  AnfNodePtr bprop_arg = nullptr;
  if (sens_param_) {
    bprop_arg = k_child->add_parameter();
  } else {
    auto ones_like = prim::GetPythonOps("_ones_like_for_grad");
    bprop_arg = k_child->NewCNodeInOrder({NewValueNode(ones_like), f_app});
  }
  AnfNodePtr b_app = k_child->NewCNodeInOrder({bprop, bprop_arg});
  // Add sense parameter flag for bound_node_.
  if (b_app->isa<CNode>() && sens_param_) {
    b_app->cast<CNodePtr>()->AddAttr("sens_param_", MakeValue(true));
  }

  CNodePtr fv_bprop = nullptr;
  if (get_by_list_) {
    if (is_weights_none) {
      fv_bprop = k_child->NewCNodeInOrder({NewValueNode(prim::kPrimMakeTuple)});
    } else {
      // Python code: grads = hyper_map(F.partial(env_get, env), weights)
      AnfNodePtr env =
        k_child->NewCNodeInOrder({NewValueNode(prim::kPrimTupleGetItem), b_app, NewValueNode(static_cast<int64_t>(0))});
      AnfNodePtr partial_env_get =
        k_child->NewCNodeInOrder({NewValueNode(prim::kPrimPartial), NewValueNode(prim::GetPythonOps("env_get")), env});
      MetaFuncGraphPtr hyper_map = std::make_shared<HyperMap>();
      fv_bprop = k_child->NewCNodeInOrder({NewValueNode(hyper_map), partial_env_get, weights});
      if (return_ids_) {
        fv_bprop = SetNodeByParameter(fv_bprop, k_child);
      }
    }
  }

  CNodePtr inputs_bprop = nullptr;
  if (get_by_position_) {
    TailPtr tail_grad_by_position = std::make_shared<Tail>("tail_grad_by_position", kGradByPosition, return_ids_);
    inputs_bprop = k_child->NewCNodeInOrder({NewValueNode(tail_grad_by_position), b_app, position});
  } else if (get_all_) {
    TailPtr tail_grad_all = std::make_shared<Tail>("tail_grad_all", kGradAll);
    inputs_bprop = k_child->NewCNodeInOrder({NewValueNode(tail_grad_all), b_app});
  }

  // Gradients wrt inputs and parameters
  if (fv_bprop != nullptr && inputs_bprop != nullptr) {
    auto make_tuple = k_child->NewCNodeInOrder({NewValueNode(kPrimMakeTuple), inputs_bprop, fv_bprop});
    k_child->set_output(CreateGradOutputs(k_child, make_tuple, f_app, has_aux_, get_value_));
    return;
  }

  // Gradients wrt parameters
  if (fv_bprop != nullptr) {
    k_child->set_output(CreateGradOutputs(k_child, fv_bprop, f_app, has_aux_, get_value_));
    return;
  }

  // Gradients wrt inputs
  if (inputs_bprop != nullptr) {
    k_child->set_output(CreateGradOutputs(k_child, inputs_bprop, f_app, has_aux_, get_value_));
    return;
  }
  // Gradients wrt first input.
  // b_app returns (EnvInstance(grads wrt params), grads wrt input0, grads wrt input1, ...),
  // so obtain first input grad by setting tail_type of Tail to kGradFirst.
  TailPtr tail_grad_first = std::make_shared<Tail>("tail_grad_first", kGradFirst);
  tail_grad_first->set_enable_tuple_grad_first(forward_graph->has_flag("enable_tuple_grad_first"));
  auto tail_grad_first_cnode = k_child->NewCNodeInOrder({NewValueNode(tail_grad_first), b_app});
  k_child->set_output(CreateGradOutputs(k_child, tail_grad_first_cnode, f_app, has_aux_, get_value_));
}

namespace {
// Check if primal func graph has the primitive returned sparse result in its bprop().
void CheckPrimBpropReturnSparse(const FuncGraphPtr &primal_graph) {
  bool has_sparse_bprop_prim = false;
  (void)TopoSort(primal_graph->return_node(), SuccDeeperSimple,
                 [&has_sparse_bprop_prim](const AnfNodePtr &node) -> IncludeType {
                   MS_EXCEPTION_IF_NULL(node);
                   if (has_sparse_bprop_prim) {
                     return EXCLUDE;
                   }
                   PrimitivePtr prim = nullptr;
                   if (node->isa<CNode>()) {
                     prim = GetCNodePrimitiveWithoutDoSignature(node);
                   } else {
                     prim = GetPrimitiveWithoutDoSignature(node);
                   }
                   if (prim != nullptr) {
                     bool sparse_bprop = GetPrimitiveFlag(prim, GRAPH_FLAG_BPROP_RETURN_SPARSE);
                     if (sparse_bprop) {
                       MS_LOG(DEBUG) << "prim: " << prim->ToString() << " has attr 'bprop_return_sparse'";
                       has_sparse_bprop_prim = true;
                       return EXCLUDE;
                     }
                   }
                   return FOLLOW;
                 });
  if (has_sparse_bprop_prim) {
    primal_graph->set_flag(FUNC_GRAPH_FLAG_SPARSE_BPROP, true);
    EnvSetSparseResultMgr::GetInstance().Set(true);
  }
}
}  // namespace

// Generate the graph.
FuncGraphPtr GradOperation::GenerateFuncGraph(const abstract::AbstractBasePtrList &args_abs_list) {
  if (args_abs_list.empty()) {
    MS_LOG(EXCEPTION)
      << "'GradOperation' requires a forward network or function as an input, while the input is empty.";
  }

  constexpr size_t fn_index = 0;
  auto fn_abs = args_abs_list[fn_index];
  constexpr size_t len_with_weight = 2;
  constexpr size_t weights_index = 1;
  if (args_abs_list.size() >= len_with_weight) {
    weight_value_ = args_abs_list[weights_index];
  }
  MS_EXCEPTION_IF_NULL(fn_abs);
  if (fn_abs->isa<abstract::AbstractClass>()) {
    auto class_abs = dyn_cast<abstract::AbstractClass>(fn_abs);
    auto class_val = class_abs->BuildValue();
    MS_EXCEPTION_IF_NULL(class_val);
    auto class_obj = class_val->cast<parse::MsClassObjectPtr>();
    MS_EXCEPTION_IF_NULL(class_obj);
    auto obj_name = std::regex_replace(class_obj->name(), std::regex("MsClassObject:"), "");
    MS_LOG(EXCEPTION) << "For 'GradOperation', the first argument must be a 'Function' or 'Cell' type "
                      << "object, but got object with jit_class type" << obj_name << ".";
  }
  abstract::AbstractFunctionPtr fn = dyn_cast<abstract::AbstractFunction>(fn_abs);
  if (fn == nullptr) {
    MS_LOG(EXCEPTION) << "For 'GradOperation', the first argument must be a 'Function' or 'Cell', but got "
                      << args_abs_list[0]->ToString();
  }

  auto real_fn = fn->cast_ptr<abstract::FuncGraphAbstractClosure>();
  if (real_fn == nullptr) {
    MS_LOG(EXCEPTION) << "For 'GradOperation', the first argument must be a 'Function' or 'Cell', but got "
                      << fn->ToString();
  }
  FuncGraphPtr forward_graph = real_fn->func_graph();
  MS_EXCEPTION_IF_NULL(forward_graph);

  if (has_aux_) {
    GradAuxPtr aux_fn = std::make_shared<GradAux>("aux_fn");
    auto output_cnode = forward_graph->output();
    auto aux_fn_cnode = forward_graph->NewCNodeInOrder({NewValueNode(aux_fn), output_cnode, NewValueNode(get_value_)});
    forward_graph->set_output(aux_fn_cnode);
  }

  forward_graph->set_flag(FUNC_GRAPH_FLAG_DEFER_INLINE, true);

  // Check if primal func graph has the primitive returned sparse result in its bprop().
  CheckPrimBpropReturnSparse(forward_graph);

  FuncGraphPtr grad_fg = nullptr;
  {
    TraceGuard g(MakeTraceInfo<TraceGradOperation>(forward_graph->debug_info()));
    grad_fg = std::make_shared<FuncGraph>();
  }
  auto nparam = forward_graph->parameters().size();

  std::ostringstream ss;
  ss << "grad{" << nparam << "}";
  grad_fg->set_flag(FUNC_GRAPH_FLAG_CORE, true);
  grad_fg->set_flag("grad_fg", true);
  if (grad_fg->debug_info() != nullptr) {
    grad_fg->debug_info()->set_name(ss.str());
  }
  ParameterPtr param_graph = grad_fg->add_parameter();

  bool is_weights_empty_or_none = false;
  AnfNodePtr weights = nullptr;
  AnfNodePtr position = nullptr;
  if (args_abs_list.size() > weights_index) {
    auto weights_abs = args_abs_list[weights_index];
    MS_EXCEPTION_IF_NULL(weights_abs);
    if (weights_abs->isa<abstract::AbstractSequence>()) {
      if (weights_abs->cast<abstract::AbstractSequencePtr>()->empty()) {
        is_weights_empty_or_none = true;
      }
    }
  }
  if (get_by_position_) {
    weights = grad_fg->add_parameter();
    position = grad_fg->add_parameter();
  } else if (get_by_list_) {
    weights = grad_fg->add_parameter();
    // Check if weights is None.
    if (!is_weights_empty_or_none && args_abs_list.size() > weights_index) {
      auto weights_abs = args_abs_list[weights_index];
      MS_EXCEPTION_IF_NULL(weights_abs);
      if (weights_abs->isa<abstract::AbstractNone>()) {
        is_weights_empty_or_none = true;
      }
    }
  }

  std::vector<AnfNodePtr> inputs;
  inputs.push_back(NewValueNode(prim::kPrimJ));
  inputs.push_back(param_graph);
  auto j = grad_fg->NewCNodeInOrder(inputs);
  if (merge_forward_) {
    j->set_user_data<bool>("merge_forward", std::make_shared<bool>(true));
  }
  // df is checked in GetGrad
  FuncGraphPtr k_child = nullptr;
  {
    TraceGuard guard(MakeTraceInfo<TraceGradOperation>(forward_graph->debug_info()));
    k_child = GetGrad(j, weights, position, forward_graph, is_weights_empty_or_none);
    k_child->set_flag(FUNC_GRAPH_FLAG_ARGS_NO_EXPAND, true);
  }
  grad_fg->set_output(NewValueNode(k_child));

  return grad_fg;
}
}  // namespace prim
}  // namespace mindspore
