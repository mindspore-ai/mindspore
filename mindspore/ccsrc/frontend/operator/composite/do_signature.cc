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

#include "frontend/operator/composite/do_signature.h"
#include <algorithm>
#include <utility>

#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/param_validator.h"
#include "frontend/operator/cc_implementations.h"
#include "frontend/operator/ops.h"
#include "include/utils/amp.h"
#include "utils/convert_utils.h"
#include "include/utils/operator/primitive_utils.h"
#include "frontend/jit/ps/static_analysis/prim.h"
#include "frontend/jit/ps/static_analysis/prim_utils.h"
#include "frontend/jit/ps/static_analysis/auto_monad.h"
#include "ir/anf.h"
#include "ir/dtype.h"
#include "ir/dtype/ref.h"
#include "ir/func_graph_flag.h"
#include "ir/core_ops_primitive.h"
#include "ops/op_def.h"
#include "utils/flags.h"
#include "primitive/arithmetic_ops.h"
#include "primitive/auto_generate/gen_ops_primitive_c.h"
#include "primitive/auto_generate/gen_ops_primitive_i.h"
#include "primitive/auto_generate/gen_ops_primitive_l.h"
#include "primitive/auto_generate/gen_ops_primitive_m.h"
#include "primitive/auto_generate/gen_ops_primitive_s.h"
#include "primitive/auto_generate/gen_ops_primitive_t.h"

namespace mindspore {
// namespace to support composite operators definition
namespace prim {
namespace {
using TypeInfoPair = std::pair<std::vector<TypeId>, std::vector<bool>>;

const std::vector<Signature> &GetSignature(const ValuePtr &function) {
  static const auto empty = std::vector<Signature>();
  if (function->isa<Primitive>() && function->cast<PrimitivePtr>()->has_signature()) {
    return function->cast<PrimitivePtr>()->signatures();
  } else if (function->isa<MetaFuncGraph>()) {
    return function->cast<MetaFuncGraphPtr>()->signatures();
  }
  return empty;
}

void ProcessDefault(const FuncGraphPtr &graph, const std::string &func_name, const AbstractBasePtrList &args_abs_list,
                    const std::vector<Signature> &signature, AnfNodePtrList *op_inputs,
                    std::vector<TypePtr> *input_types) {
  auto args_size = args_abs_list.size();
  if (args_size != op_inputs->size() || op_inputs->size() != input_types->size()) {
    MS_LOG(INTERNAL_EXCEPTION) << "For " << func_name << ", the number of args_abs_list is " << args_size
                               << ", but the number of cnodes is " << op_inputs->size() << ", the number of types is "
                               << input_types->size();
  }
  std::set<std::string> sig_names;
  for (const auto &sig : signature) {
    (void)sig_names.insert(sig.name);
  }
  std::map<std::string, AnfNodePtr> key_node_map;
  std::map<std::string, TypePtr> key_type_map;
  AnfNodePtrList new_op_inputs;
  std::vector<TypePtr> new_input_types;
  for (size_t i = 0; i < args_size; ++i) {
    if (args_abs_list[i]->isa<abstract::AbstractKeywordArg>()) {
      const auto &key = args_abs_list[i]->cast<abstract::AbstractKeywordArgPtr>()->get_key();
      if (sig_names.find(key) == sig_names.end()) {
        MS_LOG(EXCEPTION) << "Got an unexpected keyword argument '" << key << "' for '" << func_name << "'.";
      }
      key_node_map[key] =
        graph->NewCNodeInOrder({NewValueNode(prim::kPrimExtractKeywordArg), NewValueNode(key), op_inputs->at(i)});
      key_type_map[key] = input_types->at(i);
    } else {
      (void)new_op_inputs.emplace_back(op_inputs->at(i));
      (void)new_input_types.emplace_back(input_types->at(i));
    }
  }
  for (size_t i = new_op_inputs.size(); i < signature.size(); ++i) {
    const auto &arg_name = signature[i].name;
    const auto &iter = key_node_map.find(arg_name);
    if (iter != key_node_map.end()) {
      (void)new_op_inputs.emplace_back(iter->second);
      (void)new_input_types.emplace_back(key_type_map[arg_name]);
    } else {
      auto default_value = signature[i].default_value;
      if (default_value == nullptr) {
        MS_LOG(EXCEPTION) << "For '" << func_name << "', the size of input should be " << signature.size()
                          << ", but got " << args_size << ". Please check inputs of the operator.";
      }
      auto type = default_value->type() != nullptr ? default_value->type() : std::make_shared<TypeNone>();
      (void)new_op_inputs.emplace_back(NewValueNode(default_value));
      (void)new_input_types.emplace_back(type);
    }
  }
  *op_inputs = new_op_inputs;
  *input_types = new_input_types;
}

TypeInfoPair GetTypeInfo(const std::vector<TypePtr> &input_types) {
  TypeInfoPair type_info_pair;
  for (const auto &arg_type : input_types) {
    MS_EXCEPTION_IF_NULL(arg_type);
    TypeId type_id = kTypeUnknown;
    bool is_tensor = false;
    if (arg_type->isa<Number>()) {
      type_id = arg_type->cast<NumberPtr>()->type_id();
      is_tensor = false;
    } else if (arg_type->isa<TensorType>()) {
      auto elem_type = arg_type->cast<TensorTypePtr>()->element();
      MS_EXCEPTION_IF_NULL(elem_type);
      type_id = elem_type->type_id();
      is_tensor = true;
    }
    (void)type_info_pair.first.emplace_back(type_id);
    (void)type_info_pair.second.emplace_back(is_tensor);
  }
  return type_info_pair;
}

void CheckSigSize(const ValuePtr &function, const size_t &sig_size, const bool &has_var,
                  const AbstractBasePtrList &args_abs_list, const std::string &func_name) {
  if (sig_size > 0) {
    if (has_var) {
      if (sig_size - 1 > args_abs_list.size()) {
        MS_LOG(EXCEPTION) << "Function " << func_name
                          << "'s input length less than PositionalKeyword Signature length.";
      }
      return;
    }
    // Consider the case where there are monads in primitive's args_abs_list.
    size_t args_size = args_abs_list.size();
    if (function->isa<Primitive>()) {
      auto prim = function->cast<PrimitivePtr>();
      if (prim->HasAttr(GRAPH_FLAG_SIDE_EFFECT_MEM) || prim->HasAttr(GRAPH_FLAG_SIDE_EFFECT_IO)) {
        args_size -= GetAbstractMonadNum(args_abs_list);
      }
    }
    if (args_size > sig_size) {
      MS_LOG(EXCEPTION) << "Function " << func_name << "'s input length is not equal to Signature length.";
    }
  }
}

void CheckPrimInputType(const ValuePtr &function, const AbstractBasePtrList &args_abs_list) {
  if (!function->isa<Primitive>()) {
    return;
  }
  auto prim = function->cast<PrimitivePtr>();
  const auto &prim_name = prim->name();
  auto op_def = mindspore::ops::GetOpDef(prim_name);
  if (op_def == nullptr) {
    return;
  }
  auto op_args = op_def->args_;
  std::vector<ops::OpInputArg> op_call_args;
  (void)std::copy_if(op_args.cbegin(), op_args.cend(), std::back_inserter(op_call_args),
                     [](const ops::OpInputArg &arg) { return !arg.as_init_arg_; });
  auto args_size = args_abs_list.size() - GetAbstractMonadNum(args_abs_list);
  if (args_size > op_call_args.size()) {
    MS_EXCEPTION(TypeError) << "For Operator[" << prim_name
                            << "], the number of inputs should be less than or equal to " << op_call_args.size()
                            << ", but got " << args_size << ".";
  }
  for (size_t i = 0; i < args_abs_list.size(); ++i) {
    auto abs = args_abs_list[i];
    auto op_arg = op_call_args[i];
    if (!op_arg.arg_handler_.empty()) {
      continue;
    }
    if (abs->isa<abstract::AbstractKeywordArg>()) {
      continue;
    }
    if (abstract::ValidateArgSpecialType(prim_name, abs, op_arg)) {
      continue;
    }
    auto cast_dtypes = op_arg.cast_dtype_;
    bool match = std::any_of(cast_dtypes.cbegin(), cast_dtypes.cend(),
                             [&abs](const ops::OP_DTYPE &dtype) { return abstract::ValidateArgsType(abs, dtype); });
    if (!match) {
      MS_EXCEPTION(TypeError) << prim::BuildOpInputsErrorMsg(op_def, op_arg.arg_name_, abs->BuildType());
    }
  }
}

SignatureEnumRW GetSignatureEnumRW(size_t index, const std::vector<Signature> &signature, bool has_var) {
  SignatureEnumRW sig = SignatureEnumRW::kRWDefault;
  // If sig_size is 0 use default.
  std::size_t sig_size = signature.size();
  if (index < sig_size) {
    sig = signature[index].rw;
  } else if (has_var && index >= sig_size) {
    sig = signature[sig_size - 1].rw;
  }
  return sig;
}

TypePtr GetMixedPrecisionTargetType(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  if (func_graph->has_flag(GRAPH_FLAG_MIX_PRECISION_FP32)) {
    return kFloat32;
  } else if (func_graph->has_flag(GRAPH_FLAG_MIX_PRECISION_FP16)) {
    return kFloat16;
  } else if (func_graph->has_flag(GRAPH_FLAG_MIX_PRECISION_BF16)) {
    return kBFloat16;
  } else {
    return nullptr;
  }
}

bool GetImplicitPromoteType(const std::vector<Signature> &signature, const std::set<size_t> &write_indices,
                            TypeInfoPair *args_type_info) {
  if (signature.empty()) {
    return false;
  }
  std::vector<SignatureEnumDType> dtypes;
  (void)std::transform(signature.begin(), signature.end(), std::back_inserter(dtypes),
                       [](const Signature &sig) { return sig.dtype; });
  int64_t empty_dtype_count = std::count(dtypes.begin(), dtypes.end(), SignatureEnumDType::kDTypeEmptyDefaultValue);
  if (static_cast<int64_t>(dtypes.size()) == empty_dtype_count) {
    return false;
  }
  auto args_size = dtypes.size();
  auto args_type_id = args_type_info->first;
  auto args_is_tensor = args_type_info->second;
  if (args_size > args_type_id.size()) {
    // It is possible that op_inputs size is larger than signatures size in vmap.
    MS_LOG(INTERNAL_EXCEPTION) << "For auto type cast, the number of args should be greater than or equal to "
                               << args_size << ", but got " << args_type_id.size() << ".";
  }
  auto sig_type_map = GetSignatureTypeMap(dtypes, args_type_id, args_is_tensor, write_indices);
  for (size_t i = 0; i < args_size; ++i) {
    auto it = sig_type_map.find(dtypes[i]);
    if (it == sig_type_map.end()) {
      continue;
    }
    (args_type_info->first)[i] = (it->second).first;
    (args_type_info->second)[i] = (it->second).second;
  }
  return true;
}

TypeInfoPair UpdateTypeInfoForAmp(const std::vector<TypePtr> &input_types, const TypeInfoPair &type_info,
                                  std::vector<bool> *matched_sequence) {
  TypeInfoPair amp_type_info = type_info;
  // Collect Tensor[Float] in Tuple/List.
  for (size_t i = 0; i < input_types.size(); ++i) {
    auto arg_type = input_types[i];
    MS_EXCEPTION_IF_NULL(arg_type);
    TypePtrList sequence_elements;
    if (arg_type->isa<Tuple>()) {
      auto tuple_type = arg_type->cast<TuplePtr>();
      if (tuple_type->dynamic_len()) {
        continue;
      }
      sequence_elements = tuple_type->elements();
    } else if (arg_type->isa<List>()) {
      auto list_type = arg_type->cast<ListPtr>();
      if (list_type->dynamic_len()) {
        continue;
      }
      sequence_elements = list_type->elements();
    } else {
      continue;
    }
    for (const auto &elem : sequence_elements) {
      if (elem->isa<TensorType>()) {
        auto tensor_type = elem->cast<TensorTypePtr>()->element();
        MS_EXCEPTION_IF_NULL(tensor_type);
        auto type_id = tensor_type->type_id();
        if (IsFloatTensor(type_id, true)) {
          (void)amp_type_info.first.emplace_back(type_id);
          (void)amp_type_info.second.emplace_back(true);
          (*matched_sequence)[i] = true;
        }
      }
    }
  }
  return amp_type_info;
}

AnfNodePtr DoTypeCastForSequenceElement(const FuncGraphPtr &func_graph, const TypePtrList &elem_type_list,
                                        const AnfNodePtr &seq_node, const TypeId &amp_type_id, bool is_tuple) {
  bool need_cast = false;
  auto make_prim = is_tuple ? prim::kPrimMakeTuple : prim::kPrimMakeList;
  auto getitem_prim = is_tuple ? prim::kPrimTupleGetItem : prim::kPrimListGetItem;
  AnfNodePtrList elem_inputs{NewValueNode(make_prim)};
  for (size_t i = 0; i < elem_type_list.size(); ++i) {
    auto elem_node = func_graph->NewCNodeInOrder({NewValueNode(getitem_prim), seq_node, NewValueNode(SizeToLong(i))});
    if (elem_type_list[i]->isa<TensorType>()) {
      auto tensor_type = elem_type_list[i]->cast<TensorTypePtr>()->element();
      MS_EXCEPTION_IF_NULL(tensor_type);
      auto type_id = tensor_type->type_id();
      if (IsFloatTensor(type_id, true) && type_id != amp_type_id) {
        auto cast_node = func_graph->NewCNodeAfter(
          elem_node, {NewValueNode(prim::kPrimCast), elem_node, NewValueNode(static_cast<int64_t>(amp_type_id))});
        (void)elem_inputs.emplace_back(cast_node);
        MS_LOG(DEBUG) << "Do type cast for sequence[" << i << "]: " << cast_node->DebugString();
        need_cast = true;
        continue;
      }
    }
    (void)elem_inputs.emplace_back(elem_node);
  }
  return need_cast ? func_graph->NewCNodeInOrder(elem_inputs) : nullptr;
}

void DoTypeCastForFloatTensorInSequence(const FuncGraphPtr &func_graph, const std::vector<TypePtr> &input_types,
                                        const TypeId &amp_type_id, const std::vector<bool> &matched_sequence,
                                        AnfNodePtrList *op_inputs) {
  for (size_t idx = 0; idx < matched_sequence.size(); ++idx) {
    if (!matched_sequence[idx]) {
      continue;
    }
    auto input_type = input_types[idx];
    MS_EXCEPTION_IF_NULL(input_type);
    if (input_type->isa<Tuple>()) {
      const auto &tuple_elems = input_type->cast<TuplePtr>()->elements();
      auto new_node = DoTypeCastForSequenceElement(func_graph, tuple_elems, (*op_inputs)[idx], amp_type_id, true);
      if (new_node != nullptr) {
        (*op_inputs)[idx] = new_node;
      }
    } else if (input_type->isa<List>()) {
      const auto &list_elems = input_type->cast<ListPtr>()->elements();
      auto new_node = DoTypeCastForSequenceElement(func_graph, list_elems, (*op_inputs)[idx], amp_type_id, false);
      if (new_node != nullptr) {
        (*op_inputs)[idx] = new_node;
      }
    }
  }
}

bool GetAutoMixedPrecisionType(const FuncGraphPtr &func_graph, const ValuePtr &function,
                               const std::vector<TypePtr> &input_types, TypeInfoPair *target_type_info,
                               std::vector<AnfNodePtr> *op_inputs) {
  if (!function->isa<Primitive>() || func_graph->amp_strategy() == nullptr || !func_graph->amp_strategy()->IsEnable()) {
    return false;
  }
  const auto &prim_name = function->cast<PrimitivePtr>()->name();
  auto strategy_info = GetPrimCastStrategyInfo(func_graph->amp_strategy(), prim_name);
  if (strategy_info.strategy == amp::PrimCastStrategy::Ignore) {
    return false;
  }

  // Get auto mixed-precision target type id.
  std::vector<bool> matched_sequence(input_types.size());
  auto amp_type_info = UpdateTypeInfoForAmp(input_types, *target_type_info, &matched_sequence);
  TypeId amp_type_id;
  if (strategy_info.strategy == amp::PrimCastStrategy::AutoPromote) {
    amp_type_id = GetMixPrecisionPromoteType(amp_type_info.first, amp_type_info.second);
    if (amp_type_id == kTypeUnknown) {
      return false;
    }
  } else {
    amp_type_id = strategy_info.dtype->type_id();
  }
  MS_LOG(DEBUG) << "For Operator[" << prim_name << "], its amp strategy is " << strategy_info.strategy
                << ", and target type_id is" << TypeIdToString(amp_type_id);

  // Process Tensor[Float] in Tuple/List.
  DoTypeCastForFloatTensorInSequence(func_graph, input_types, amp_type_id, matched_sequence, op_inputs);
  // Get target type id.
  for (size_t i = 0; i < target_type_info->first.size(); ++i) {
    if (IsFloatTensor((target_type_info->first)[i], (target_type_info->second)[i])) {
      (target_type_info->first)[i] = amp_type_id;
    }
  }
  return true;
}

void DoTypeCast(const FuncGraphPtr &func_graph, const ValuePtr &func, const std::set<size_t> &write_indices,
                const std::pair<TypeInfoPair, TypeInfoPair> &type_info_pair, std::vector<AnfNodePtr> *op_inputs,
                const AnfNodePtr &old_cnode) {
  auto source_type_info = type_info_pair.first;
  auto target_type_info = type_info_pair.second;
  for (size_t i = 0; i < source_type_info.first.size(); ++i) {
    TypeId source_type_id = (source_type_info.first)[i];
    TypeId target_type_id = (target_type_info.first)[i];
    bool source_is_tensor = (source_type_info.second)[i];
    bool target_is_tensor = (target_type_info.second)[i];
    if (source_type_id == kTypeUnknown || target_type_id == kTypeUnknown) {
      continue;
    }
    if (source_type_id == target_type_id && source_is_tensor == target_is_tensor) {
      continue;
    }
    if (write_indices.find(i) != write_indices.end()) {
      MS_EXCEPTION(TypeError) << ErrorMessageForConvertRefDtype(func, TypeIdToString(source_type_id),
                                                                TypeIdToString(target_type_id), i);
    }
    auto param = (*op_inputs)[i];
    auto target_type_node = NewValueNode(static_cast<int64_t>(target_type_id));
    MS_LOG(DEBUG) << "Do type cast for Primitive[" << func->ToString() << "], source_is_tensor: " << source_is_tensor
                  << ", target_is_tensor: " << target_is_tensor
                  << ", source_type_id: " << TypeIdToString(source_type_id)
                  << ", target_type_id: " << TypeIdToString(target_type_id);
    // For generating new_cnode to replace old_cnode, we insert kPrimCast before old_cnode to maintain orderlist
    if (!source_is_tensor && target_is_tensor) {
      // Scalar needs to be converted to Tensor.
      auto source_type_node = NewValueNode(static_cast<int64_t>(source_type_id));
      AnfNodePtrList scalar_to_tensor_inputs = {NewValueNode(prim::kPrimScalarToTensor), param, source_type_node};
      param = (old_cnode == nullptr ? func_graph->NewCNodeAfter(param, scalar_to_tensor_inputs)
                                    : func_graph->NewCNodeBefore(old_cnode, scalar_to_tensor_inputs));
      MS_LOG(DEBUG) << "Using " << (old_cnode == nullptr ? "param" : "old cnode") << " as anchor to insert cast op.";
      (*op_inputs)[i] = func_graph->NewCNodeAfter(param, {NewValueNode(prim::kPrimCast), param, target_type_node});
    } else {
      // If target type is not Tensor but scalar, use ScalarCast.
      PrimitivePtr cast_op = target_is_tensor ? prim::kPrimCast : prim::kPrimScalarCast;
      AnfNodePtrList cast_inputs = {NewValueNode(cast_op), param, target_type_node};
      MS_LOG(DEBUG) << "Using " << (old_cnode == nullptr ? "param" : "old cnode") << " as anchor to insert cast op.";
      (*op_inputs)[i] = (old_cnode == nullptr ? func_graph->NewCNodeAfter(param, cast_inputs)
                                              : func_graph->NewCNodeBefore(old_cnode, cast_inputs));
    }
  }
}

void InsertCastForToFloat(const FuncGraphPtr &func_graph, const TypePtr &cast_type, AnfNodePtr *param, TypePtr *type) {
  auto source_tensor_type = (*type)->cast<TensorTypePtr>();
  if (source_tensor_type != nullptr) {
    const auto &source_element = source_tensor_type->element();
    if (cast_type != nullptr && (IsSubType(source_element, kFloat) || IsSubType(source_element, kBFloat)) &&
        *source_element != *cast_type) {
      auto cast = prim::GetPythonOps("_cast", "mindspore.ops.functional");
      *param = func_graph->NewCNodeAfter(*param, {NewValueNode(cast), *param, NewValueNode(cast_type)});
      *type = cast_type->type_id() == kNumberTypeFloat16
                ? kTensorTypeFP16
                : (cast_type->type_id() == kNumberTypeBFloat16 ? kTensorTypeBF16 : kTensorTypeFP32);
    }
  }
}
}  // namespace

std::vector<AnfNodePtr> GetNewInputsBySignatures(const FuncGraphPtr &func_graph, const std::string &func_name,
                                                 const ValuePtr &function, const AbstractBasePtrList &args_abs_list,
                                                 const std::vector<AnfNodePtr> &params_list,
                                                 const AnfNodePtr &old_cnode) {
  // args: original inputs
  auto &signature = GetSignature(function);
  std::size_t sig_size = signature.size();
  auto has_var = (sig_size > 0 && signature[sig_size - 1].kind == SignatureEnumKind::kKindVarPositional);
  CheckSigSize(function, sig_size, has_var, args_abs_list, func_name);
  CheckPrimInputType(function, args_abs_list);
  std::vector<AnfNodePtr> op_inputs;
  std::set<size_t> write_indices;
  std::vector<TypePtr> input_types;
  bool is_inplace_prim = function->isa<Primitive>() && function->cast<PrimitivePtr>()->inplace_prim();
  auto cast_type = GetMixedPrecisionTargetType(func_graph);
  // Assume, the write input of op is always the first input. We check if any write op,
  // and add cast op on other inputs to keep the same type with assigned parameter.
  for (size_t i = 0; i < args_abs_list.size(); ++i) {
    MS_EXCEPTION_IF_NULL(args_abs_list[i]);
    AnfNodePtr param = params_list[i];
    SignatureEnumRW sig = GetSignatureEnumRW(i, signature, has_var);
    TypePtr type = args_abs_list[i]->BuildType();
    if (type && type->isa<RefType>()) {
      if (sig == SignatureEnumRW::kRWRead) {
        InsertCastForToFloat(func_graph, cast_type, &param, &type);
      } else if (sig == SignatureEnumRW::kRWWrite) {
        (void)write_indices.insert(i);
      }
      // If sig is SignatureEnumRW::kRWRef, not do anything.
    } else if (is_inplace_prim && sig == SignatureEnumRW::kRWWrite) {
      (void)write_indices.insert(i);
    } else if (type && IfRaiseExceptionForCheckParameter(func_name, function, sig, type)) {
      MS_EXCEPTION(TypeError) << "Function " << func_name << "'s input " << i << " should be a Parameter or a Tensor, "
                              << "but got " << type->ToString() << ".";
    }
    MS_LOG(DEBUG) << "Function " << func_name << "'s input " << i << " "
                  << param->DebugString(AnfNode::DebugStringLevel::kLevel2) << " abs " << args_abs_list[i]->ToString()
                  << " type " << type->ToString() << ".";
    input_types.push_back(type);
    op_inputs.push_back(param);
  }
  // process default
  auto positional_size = has_var ? signature.size() - 1 : signature.size();
  if (args_abs_list.size() < positional_size) {
    ProcessDefault(func_graph, func_name, args_abs_list, signature, &op_inputs, &input_types);
  }
  // Record type info.
  auto source_type_info = GetTypeInfo(input_types);
  auto target_type_info = source_type_info;
  // Auto mixed precision.
  bool amp_type_changed = GetAutoMixedPrecisionType(func_graph, function, input_types, &target_type_info, &op_inputs);
  // Implicit type promotion.
  bool promote_type_changed = GetImplicitPromoteType(signature, write_indices, &target_type_info);
  // Do type cast.
  if (promote_type_changed || amp_type_changed) {
    DoTypeCast(func_graph, function, write_indices, std::make_pair(source_type_info, target_type_info), &op_inputs,
               old_cnode);
  }
  return op_inputs;
}

AnfNodePtr GenerateCNodeBySignatures(const FuncGraphPtr &func_graph, const std::string &func_name,
                                     const ValuePtr &function, const AbstractBasePtrList &args_abs_list,
                                     const AnfNodePtrList &old_node_inputs) {
  auto new_inputs = GetNewInputsBySignatures(func_graph, func_name, function, args_abs_list, old_node_inputs);
  AnfNodePtrList op_inputs{NewValueNode(function)};
  (void)std::copy(new_inputs.begin(), new_inputs.end(), std::back_inserter(op_inputs));
  return func_graph->NewCNodeInOrder(op_inputs);
}

FuncGraphPtr DoSignatureMetaFuncGraph::GenerateFuncGraph(const AbstractBasePtrList &args_abs_list) {
  FuncGraphPtr func_graph = std::make_shared<FuncGraph>();

  for (size_t i = 0; i < args_abs_list.size(); ++i) {
    (void)func_graph->add_parameter();
  }
  auto new_cnode = GenerateCNodeBySignatures(func_graph, name_, function_, args_abs_list, func_graph->parameters());
  func_graph->set_output(new_cnode);
  func_graph->set_flag(FUNC_GRAPH_FLAG_CORE, true);
  return func_graph;
}

bool IfRaiseExceptionForCheckParameter(const std::string &func_name, const ValuePtr &function,
                                       const SignatureEnumRW &sig, const TypePtr &type) {
  auto is_type_ref = (sig == SignatureEnumRW::kRWWrite) &&
                     !((type->type_id() == kObjectTypeRef) || (type->type_id() == kObjectTypeRefKey) ||
                       (type->type_id() == kMetaTypeNone));
  if (is_type_ref && (!function->isa<Primitive>() || !function->cast<PrimitivePtr>()->inplace_prim())) {
    return true;
  }
  return false;
}
}  // namespace prim
}  // namespace mindspore
