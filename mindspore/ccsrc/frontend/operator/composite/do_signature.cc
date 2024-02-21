/**
 * Copyright 2019-2024 Huawei Technologies Co., Ltd
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
#include "frontend/optimizer/opt.h"
#include "include/common/utils/convert_utils.h"
#include "include/common/pybind_api/api_register.h"
#include "ir/anf.h"
#include "ir/dtype.h"
#include "ops/op_def.h"
#include "mindspore/core/utils/flags.h"
#include "mindspore/core/ops/arithmetic_ops.h"
#include "mindspore/core/ops/auto_generate/gen_ops_primitive.h"

namespace mindspore {
// namespace to support composite operators definition
namespace prim {
const std::map<TypeId, size_t> type_map = {{kNumberTypeBool, 1},    {kNumberTypeInt8, 2},    {kNumberTypeUInt8, 3},
                                           {kNumberTypeInt16, 4},   {kNumberTypeInt32, 5},   {kNumberTypeInt64, 6},
                                           {kNumberTypeFloat16, 7}, {kNumberTypeFloat32, 8}, {kNumberTypeFloat64, 9}};
namespace {
const std::vector<Signature> &GetSignature(const ValuePtr &function) {
  static const auto empty = std::vector<Signature>();
  if (function->isa<Primitive>() && function->cast<PrimitivePtr>()->has_signature()) {
    return function->cast<PrimitivePtr>()->signatures();
  } else if (function->isa<MetaFuncGraph>()) {
    return function->cast<MetaFuncGraphPtr>()->signatures();
  }
  return empty;
}

void ProcessDefault(const std::string &func_name, size_t actual_param_number, const std::vector<Signature> &signature,
                    bool has_var, std::vector<AnfNodePtr> *op_inputs) {
  std::size_t sig_size = signature.size();
  auto positional_size = sig_size;
  if (has_var) {
    positional_size = sig_size - 1;
  }
  if (actual_param_number < positional_size) {
    for (size_t i = actual_param_number; i < sig_size; ++i) {
      auto default_value = signature[i].default_value;
      if (default_value == nullptr) {
        MS_LOG(EXCEPTION) << "For '" << func_name << "', the size of input should be " << sig_size << ", but got "
                          << actual_param_number << ". Please check inputs of the operator.";
      } else {
        (*op_inputs).push_back(NewValueNode(default_value));
      }
    }
  }
}

void GetTypeInfo(const std::vector<TypePtr> &input_types, std::vector<TypeId> *args_type_id,
                 std::vector<bool> *args_has_tensor) {
  for (const auto &arg_type : input_types) {
    if (arg_type->isa<Number>()) {
      (void)args_type_id->emplace_back(arg_type->cast<NumberPtr>()->type_id());
      (void)args_has_tensor->emplace_back(false);
    } else if (arg_type->isa<TensorType>()) {
      auto elem_type = arg_type->cast<TensorTypePtr>()->element();
      MS_EXCEPTION_IF_NULL(elem_type);
      (void)args_type_id->emplace_back(elem_type->type_id());
      (void)args_has_tensor->emplace_back(true);
    } else {
      (void)args_type_id->emplace_back(kTypeUnknown);
      (void)args_has_tensor->emplace_back(false);
    }
  }
}

TypeId GetConversionType(const TypeId &current, const TypeId &saved_type_id, bool arg_is_tensor, bool contain_tensor) {
  if (current == saved_type_id) {
    return current;
  }
  if (current == kTypeUnknown || saved_type_id == kTypeUnknown) {
    return kTypeUnknown;
  }
  // Tensor + Scalar
  if (arg_is_tensor && !contain_tensor) {
    return ConvertTypeBetweenTensorAndScalar(current, saved_type_id);
  }
  // Scalar + Tensor
  if (!arg_is_tensor && contain_tensor) {
    return ConvertTypeBetweenTensorAndScalar(saved_type_id, current);
  }
  // Tensor + Tensor, Scalar + Scalar
  return ConvertTypeForTensorsOrScalars(current, saved_type_id);
}

std::map<SignatureEnumDType, std::pair<TypeId, bool>> GetSignatureTypeMap(const std::vector<SignatureEnumDType> &dtypes,
                                                                          const std::vector<TypeId> &args_type_id,
                                                                          const std::vector<bool> &args_has_tensor,
                                                                          size_t args_size) {
  std::map<SignatureEnumDType, std::pair<TypeId, bool>> sig_type_map;
  for (size_t i = 0; i < args_size; ++i) {
    const auto &it = sig_type_map.find(dtypes[i]);
    if (it == sig_type_map.end()) {
      (void)sig_type_map.insert(std::make_pair(dtypes[i], std::make_pair(args_type_id[i], args_has_tensor[i])));
    } else {
      TypeId saved_type_id = (it->second).first;
      bool contain_tensor = (it->second).second;
      TypeId target_type_id = GetConversionType(args_type_id[i], saved_type_id, args_has_tensor[i], contain_tensor);
      it->second = std::make_pair(target_type_id, args_has_tensor[i] || contain_tensor);
    }
  }
  return sig_type_map;
}

void DoAutoCast(const std::vector<Signature> &signature, const std::vector<TypePtr> &input_types,
                const FuncGraphPtr &graph, const std::pair<ValuePtr, std::set<size_t>> &write_indices_pair,
                std::vector<AnfNodePtr> *op_inputs) {
  MS_EXCEPTION_IF_NULL(graph);
  std::vector<SignatureEnumDType> dtypes;
  (void)std::transform(signature.begin(), signature.end(), std::back_inserter(dtypes),
                       [](const Signature &sig) { return sig.dtype; });
  int64_t empty_dtype_count = std::count(dtypes.begin(), dtypes.end(), SignatureEnumDType::kDTypeEmptyDefaultValue);
  if (dtypes.empty() || static_cast<int64_t>(dtypes.size()) == empty_dtype_count) {
    return;
  }
  auto args_size = signature.size();
  if (args_size != input_types.size() || args_size != op_inputs->size()) {
    MS_LOG(INTERNAL_EXCEPTION) << "For auto type cast, the number of args should be " << args_size
                               << ", but got input_types size: " << input_types.size()
                               << ", op_inputs size: " << op_inputs->size();
  }

  std::vector<TypeId> args_type_id;
  std::vector<bool> args_has_tensor;
  GetTypeInfo(input_types, &args_type_id, &args_has_tensor);
  auto sig_type_map = GetSignatureTypeMap(dtypes, args_type_id, args_has_tensor, args_size);
  for (size_t i = 0; i < args_size; ++i) {
    auto it = sig_type_map.find(dtypes[i]);
    if (it == sig_type_map.end()) {
      continue;
    }
    TypeId current_type_id = args_type_id[i];
    TypeId target_type_id = (it->second).first;
    if (current_type_id == kTypeUnknown || target_type_id == kTypeUnknown) {
      continue;
    }
    bool arg_is_tensor = args_has_tensor[i];
    bool contain_tensor = (it->second).second;
    bool need_scalar_to_tensor = !arg_is_tensor && contain_tensor;
    auto func = write_indices_pair.first;
    auto write_indices = write_indices_pair.second;
    if ((current_type_id != target_type_id || need_scalar_to_tensor) && write_indices.find(i) != write_indices.end()) {
      RaiseExceptionForConvertRefDtype(func, TypeIdToString(current_type_id), TypeIdToString(target_type_id), i);
    }
    auto param = (*op_inputs)[i];
    auto target_type_node = NewValueNode(static_cast<int64_t>(target_type_id));
    if (need_scalar_to_tensor) {
      auto current_type_node = NewValueNode(static_cast<int64_t>(current_type_id));
      param = graph->NewCNodeAfter(param, {NewValueNode(prim::kPrimScalarToTensor), param, current_type_node});
      (*op_inputs)[i] = graph->NewCNodeAfter(param, {NewValueNode(prim::kPrimCast), param, target_type_node});
    } else if (current_type_id != target_type_id) {
      PrimitivePtr cast_op = contain_tensor ? prim::kPrimCast : prim::kPrimScalarCast;
      (*op_inputs)[i] = graph->NewCNodeAfter(param, {NewValueNode(cast_op), param, target_type_node});
    }
  }
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
}  // namespace

std::vector<AnfNodePtr> GetNewInputsBySignatures(const FuncGraphPtr &func_graph, const std::string &func_name,
                                                 const ValuePtr &function, const AbstractBasePtrList &args_abs_list,
                                                 const std::vector<AnfNodePtr> &params_list) {
  // args: original inputs
  auto &signature = GetSignature(function);
  std::size_t sig_size = signature.size();
  auto has_var = (sig_size > 0 && signature[sig_size - 1].kind == SignatureEnumKind::kKindVarPositional);
  CheckSigSize(function, sig_size, has_var, args_abs_list, func_name);
  std::vector<AnfNodePtr> op_inputs;
  std::set<size_t> write_indices;
  std::vector<TypePtr> input_types;
  auto cast_type = GetMixedPrecisionTargetType(func_graph);
  // Assume, the write input of op is always the first input. We check if any write op,
  // and add cast op on other inputs to keep the same type with assigned parameter.
  for (size_t i = 0; i < args_abs_list.size(); ++i) {
    AnfNodePtr param = params_list[i];
    if (args_abs_list[i] == nullptr) {
      op_inputs.push_back(param);
      continue;
    }

    SignatureEnumRW sig = GetSignatureEnumRW(i, signature, has_var);
    TypePtr type = args_abs_list[i]->BuildType();
    if (type && type->isa<RefType>()) {
      if (sig == SignatureEnumRW::kRWRead) {
        auto source_tensor_type = type->cast<TensorTypePtr>();
        if (source_tensor_type != nullptr) {
          auto source_element = source_tensor_type->element();
          if (cast_type != nullptr && (IsSubType(source_element, kFloat) || IsSubType(source_element, kBFloat)) &&
              *source_element != *cast_type) {
            auto cast = prim::GetPythonOps("cast", "mindspore.ops.functional");
            param = func_graph->NewCNodeAfter(param, {NewValueNode(cast), param, NewValueNode(cast_type)});
            type = cast_type->type_id() == kNumberTypeFloat16
                     ? kTensorTypeFP16
                     : (cast_type->type_id() == kNumberTypeBFloat16 ? kTensorTypeBF16 : kTensorTypeFP32);
          }
        }
      } else if (sig == SignatureEnumRW::kRWWrite) {
        write_indices.insert(i);
      }
      // If sig is SignatureEnumRW::kRWRef, not do anything.
    } else if (sig == SignatureEnumRW::kRWWrite &&
               !((type->type_id() == kObjectTypeRef) || (type->type_id() == kObjectTypeRefKey))) {
      RaiseExceptionForCheckParameter(func_name, i, type->ToString());
    }
    MS_LOG(DEBUG) << "Function " << func_name << "'s input " << i << " " << param->DebugString(2) << " abs "
                  << args_abs_list[i]->ToString() << " type " << type->ToString() << ".";
    input_types.push_back(type);
    op_inputs.push_back(param);
  }
  // process default
  ProcessDefault(func_name, args_abs_list.size(), signature, has_var, &op_inputs);
  auto write_indices_pair = std::make_pair(function, write_indices);
  DoAutoCast(signature, input_types, func_graph, write_indices_pair, &op_inputs);
  return op_inputs;
}

AnfNodePtr GenerateCNode(const FuncGraphPtr &func_graph, const std::string &func_name, const ValuePtr &function,
                         const AbstractBasePtrList &args_abs_list, const AnfNodePtrList &old_node_inputs) {
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
  auto new_cnode = GenerateCNode(func_graph, name_, function_, args_abs_list, func_graph->parameters());
  func_graph->set_output(new_cnode);
  func_graph->set_flag(FUNC_GRAPH_FLAG_CORE, true);
  return func_graph;
}

void RaiseExceptionForConvertRefDtype(const ValuePtr &func, const std::string &ref_type, const std::string &target_type,
                                      size_t index) {
  std::ostringstream buffer;
  if (func->isa<Primitive>()) {
    auto prim = func->cast<PrimitivePtr>();
    auto args_names_value = prim->GetAttr("input_names");
    if (args_names_value != nullptr) {
      auto args_names = GetValue<std::vector<std::string>>(args_names_value);
      if (index < args_names.size()) {
        buffer << " the argument[" << args_names[index] << "]'s data type of primitive[" << prim->name() << "] is ";
      }
    }
  }
  if (buffer.str().empty()) {
    buffer << " so data type ";
  }
  MS_EXCEPTION(TypeError) << "Data type conversion of 'Parameter' is not supported," << buffer.str() << ref_type
                          << ", which cannot be converted to data type " << target_type << " automatically.\n";
}

void RaiseExceptionForCheckParameter(const std::string &func_name, size_t i, const std::string &source_type) {
  MS_EXCEPTION(TypeError) << "Function " << func_name << "'s input " << i << " should be a Parameter, but "
                          << source_type << ".";
}
}  // namespace prim
}  // namespace mindspore
