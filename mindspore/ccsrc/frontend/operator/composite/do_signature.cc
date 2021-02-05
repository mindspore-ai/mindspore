/**
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

#include "frontend/operator/composite/do_signature.h"
#include <algorithm>
#include <utility>

#include "abstract/abstract_value.h"
#include "ir/anf.h"
#include "ir/dtype.h"
#include "abstract/dshape.h"
#include "abstract/param_validator.h"
#include "frontend/operator/cc_implementations.h"
#include "frontend/optimizer/opt.h"
#include "pybind_api/api_register.h"

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
    return function->cast<PrimitivePyPtr>()->signatures();
  } else if (function->isa<MetaFuncGraph>()) {
    return function->cast<MetaFuncGraphPtr>()->signatures();
  }
  return empty;
}

void ProcessDefault(const std::string &func_name, size_t actual_param_number, const std::vector<Signature> &signature,
                    bool has_var, std::vector<AnfNodePtr> *const op_inputs) {
  std::size_t sig_size = signature.size();
  auto positional_size = sig_size;
  if (has_var) {
    positional_size = sig_size - 1;
  }
  if (actual_param_number < positional_size) {
    for (size_t i = actual_param_number; i < sig_size; ++i) {
      auto default_value = signature[i].default_value;
      if (default_value == nullptr) {
        MS_LOG(EXCEPTION) << "Function " << func_name << "'s input length is not equal to Signature length.";
      } else {
        (*op_inputs).push_back(NewValueNode(default_value));
      }
    }
  }
}

void SetMaxType(TypeId *max_type_id, size_t *max_type_number, const TypeId type_id, const size_t type_number) {
  *max_type_id = type_id;
  *max_type_number = type_number;
}

bool GetTensorOrScalarTypeInfo(TypePtr arg_type_origin, TypeId *arg_type_id, TypeId *arg_type = nullptr) {
  if (arg_type_origin->isa<TensorType>()) {
    auto tensor = arg_type_origin->cast<TensorTypePtr>();
    auto tensor_type = tensor->element();
    MS_EXCEPTION_IF_NULL(tensor_type);
    *arg_type_id = tensor_type->type_id();
    if (arg_type != nullptr) {
      *arg_type = kObjectTypeTensorType;
    }
    return true;
  }
  if (arg_type_origin->isa<Number>()) {
    auto scalar_type = arg_type_origin->cast<NumberPtr>();
    MS_EXCEPTION_IF_NULL(scalar_type);
    *arg_type_id = scalar_type->type_id();
    if (arg_type != nullptr) {
      *arg_type = kObjectTypeNumber;
    }
    return true;
  }
  return false;
}

TypeId GetMaxTypeId(const std::vector<TypePtr> &input_types, std::vector<size_t> indices,
                    const std::set<size_t> &write_indices) {
  TypeId max_type_id = kTypeUnknown;
  size_t max_type_number = 0;
  bool has_int8 = false;
  bool has_scalar_int64 = false;
  bool has_scalar_float32 = false;
  for (const auto &index : indices) {
    TypeId arg_type_id = kTypeUnknown;
    TypeId arg_type = kTypeUnknown;
    if (!GetTensorOrScalarTypeInfo(input_types[index], &arg_type_id, &arg_type)) {
      continue;
    }
    if (arg_type != kObjectTypeTensorType) {
      if (arg_type_id == kNumberTypeInt64) {
        has_scalar_int64 = true;
      } else if (arg_type_id == kNumberTypeFloat32) {
        has_scalar_float32 = true;
      }
      continue;
    }
    auto it = type_map.find(arg_type_id);
    if (it == type_map.end()) {
      continue;
    }
    if (arg_type_id == kNumberTypeInt8) {
      has_int8 = true;
    }
    if (max_type_id == kTypeUnknown) {
      SetMaxType(&max_type_id, &max_type_number, arg_type_id, it->second);
      continue;
    }
    if (it->second > max_type_number) {
      SetMaxType(&max_type_id, &max_type_number, arg_type_id, it->second);
    }
  }

  if (max_type_id == kNumberTypeUInt8 && has_int8) {
    max_type_id = kNumberTypeInt16;
  }
  // if bool is the max type, see if there is scalar input
  // if so, it means that max is bool tensor, use scalar type instead.
  // for example: Tensor([True, True]) * 2, expect result is Tensor([2, 2])
  if (max_type_id == kNumberTypeBool) {
    if (has_scalar_int64) {
      max_type_id = kNumberTypeInt64;
    }
    if (has_scalar_float32) {
      max_type_id = kNumberTypeFloat32;
    }
  }
  if (max_type_id != kNumberTypeFloat16 && max_type_id != kNumberTypeFloat32 && max_type_id != kNumberTypeFloat64 &&
      max_type_id != kTypeUnknown && has_scalar_float32) {
    max_type_id = kNumberTypeFloat32;
  }
  return max_type_id;
}

// Get the largest type of index in the same SignatureEnumDType of arguments.
using MaxTypeMap = std::map<SignatureEnumDType, TypeId>;
MaxTypeMap GetMaxDtype(const std::vector<SignatureEnumDType> &dtypes, const std::vector<TypePtr> &input_types,
                       const std::set<size_t> &write_indices) {
  // record index for signature.dtypes of the same type
  // eg. [T, T1, T, T2, T, T1, T3] -> {{T:(0,2,4)}, {T1:(1,5)}, {T2:(3)}, {T3:(6)}}
  std::map<SignatureEnumDType, std::vector<size_t>> type_indices;
  for (size_t i = 0; i < dtypes.size(); ++i) {
    auto it = type_indices.find(dtypes[i]);
    if (it == type_indices.end()) {
      (void)type_indices.insert(std::make_pair(dtypes[i], std::vector<size_t>{i}));
    } else {
      it->second.push_back(i);
    }
  }
  std::map<SignatureEnumDType, TypeId> dst_type;
  for (auto it = type_indices.begin(); it != type_indices.end(); (void)++it) {
    auto type = it->first;
    auto indices = it->second;
    // If the number of arguments belonging to the same SignatureEnumDType is less than 2, skip it.
    if (indices.size() < 2) {
      continue;
    }
    bool has_tensor = false;
    for (const auto &index : indices) {
      auto arg_value = input_types[index];
      if (arg_value->isa<TensorType>()) {
        has_tensor = true;
        break;
      }
    }
    if (!has_tensor) {
      (void)dst_type.insert(std::make_pair(type, kTypeUnknown));
      continue;
    }
    (void)dst_type.insert(std::make_pair(type, GetMaxTypeId(input_types, indices, write_indices)));
  }
  return dst_type;
}

AnfNodePtr DoCast(const AnfNodePtr &param, const TypeId &type_id, const FuncGraphPtr &graph) {
  auto prim_cast_class = prim::GetPythonOps("Cast", "mindspore.ops.operations");
  MS_EXCEPTION_IF_NULL(prim_cast_class);
  auto dtype_node = NewValueNode(TypeIdToType(type_id));
  auto cast_node = NewCNode({NewValueNode(prim_cast_class)}, graph);
  return graph->NewCNodeAfter(param, {cast_node, param, dtype_node});
}

void DoAutoCast(const std::string &func_name, const std::vector<Signature> &signature,
                const std::vector<TypePtr> &input_types, const FuncGraphPtr &graph,
                std::vector<AnfNodePtr> *const op_inputs, const std::set<size_t> &write_indices) {
  std::vector<SignatureEnumDType> dtypes;
  (void)std::transform(signature.begin(), signature.end(), std::back_inserter(dtypes),
                       [](const Signature &sig) { return sig.dtype; });
  int64_t empty_dtype_count = std::count(dtypes.begin(), dtypes.end(), SignatureEnumDType::kDTypeEmptyDefaultValue);
  if (dtypes.empty() || static_cast<int64_t>(dtypes.size()) == empty_dtype_count) {
    return;
  }
  // Stat the index of the arguments with the largest type in the same SignatureEnumDType.
  std::map<SignatureEnumDType, TypeId> dst_type = GetMaxDtype(dtypes, input_types, write_indices);
  // Identify which arg requires auto cast
  for (size_t i = 0; i < input_types.size(); ++i) {
    auto it = dst_type.find(dtypes[i]);
    if (it == dst_type.end() || it->second == kTypeUnknown) {
      continue;
    }
    auto rw_it = write_indices.find(i);
    auto is_write = (rw_it != write_indices.end());

    TypeId arg_type_id = kTypeUnknown;
    auto arg_value = input_types[i];
    (void)GetTensorOrScalarTypeInfo(arg_value, &arg_type_id);
    auto it_map = type_name_map.find(arg_type_id);
    if (it_map == type_name_map.end()) {
      continue;
    }
    if (is_write) {
      if (arg_type_id != it->second) {
        auto it_name_map = type_name_map.find(it->second);
        if (it_name_map == type_name_map.end()) {
          continue;
        }
        RaiseExceptionForConvertRefDtype(func_name, it_map->second, it_name_map->second);
      }
      continue;
    }
    if ((arg_value->isa<TensorType>()) && arg_type_id == it->second) {
      continue;
    }
    MS_LOG(DEBUG) << "do cast for inputs " << i << " " << (*op_inputs)[i + 1]->ToString() << " " << arg_type_id
                  << " to " << it->second;
    (*op_inputs)[i + 1] = DoCast((*op_inputs)[i + 1], it->second, graph);
  }
}

AnfNodePtr BuildNewCNode(const FuncGraphPtr &func_graph, const std::string &func_name, const ValuePtr &function,
                         const AbstractBasePtrList &args_spec_list, const std::vector<AnfNodePtr> &params_list) {
  // args: original inputs
  auto &signature = GetSignature(function);
  std::size_t sig_size = signature.size();
  auto has_var = (sig_size > 0 && signature[sig_size - 1].kind == SignatureEnumKind::kKindVarPositional);
  if (sig_size > 0) {
    if (has_var) {
      if (sig_size - 1 > args_spec_list.size()) {
        MS_LOG(EXCEPTION) << "Function " << func_name
                          << "'s input length less than PositionalKeyword Signature length.";
      }
    } else if (args_spec_list.size() > sig_size) {
      MS_LOG(EXCEPTION) << "Function " << func_name << "'s input length is not equal to Signature length.";
    }
  }
  std::vector<AnfNodePtr> op_inputs;
  std::set<size_t> write_indices;
  std::vector<TypePtr> input_types;
  op_inputs.push_back(NewValueNode(function));
  auto cast_type = parse::GetMixedPrecisionTargetType(func_graph);
  // Assume, the write input of op is always the first input. We check if any write op,
  // and add cast op on other inputs to keep the same type with assigned parameter.
  for (size_t i = 0; i < args_spec_list.size(); ++i) {
    AnfNodePtr param = params_list[i];
    if (args_spec_list[i] == nullptr) {
      op_inputs.push_back(param);
      continue;
    }
    SignatureEnumRW sig = SignatureEnumRW::kRWDefault;
    // If sig_size is 0 use default.
    if (sig_size > 0 && i < sig_size) {
      sig = signature[i].rw;
    } else if (has_var && i >= sig_size) {
      sig = signature[sig_size - 1].rw;
    }

    TypePtr type = args_spec_list[i]->BuildType();
    if (type && type->isa<RefType>()) {
      if (sig == SignatureEnumRW::kRWRead) {
        auto source_tensor_type = type->cast<TensorTypePtr>();
        if (source_tensor_type != nullptr) {
          auto source_element = source_tensor_type->element();
          if (cast_type != nullptr && IsSubType(source_element, kFloat) && *source_element != *cast_type) {
            auto cast = prim::GetPythonOps("cast", "mindspore.ops.functional");
            param = func_graph->NewCNodeAfter(param, {NewValueNode(cast), param, NewValueNode(cast_type)});
            type = cast_type->type_id() == kNumberTypeFloat16 ? kTensorTypeFP16 : kTensorTypeFP32;
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
                  << args_spec_list[i]->ToString() << " type " << type->ToString();
    input_types.push_back(type);
    op_inputs.push_back(param);
  }
  // process default
  ProcessDefault(func_name, args_spec_list.size(), signature, has_var, &op_inputs);
  DoAutoCast(func_name, signature, input_types, func_graph, &op_inputs, write_indices);
  return func_graph->NewCNodeInOrder(op_inputs);
}
}  // namespace

AnfNodePtr GenerateCNode(const FuncGraphPtr &func_graph, const std::string &func_name, const ValuePtr &function,
                         const AbstractBasePtrList &args_spec_list, const AnfNodePtrList &old_node_inputs) {
  auto new_cnode = BuildNewCNode(func_graph, func_name, function, args_spec_list, old_node_inputs);
  return new_cnode;
}

FuncGraphPtr DoSignatureMetaFuncGraph::GenerateFuncGraph(const AbstractBasePtrList &args_spec_list) {
  FuncGraphPtr func_graph = std::make_shared<FuncGraph>();

  for (size_t i = 0; i < args_spec_list.size(); ++i) {
    (void)func_graph->add_parameter();
  }
  auto new_cnode = BuildNewCNode(func_graph, name_, function_, args_spec_list, func_graph->parameters());
  func_graph->set_output(new_cnode);
  func_graph->set_flag(FUNC_GRAPH_FLAG_CORE, true);
  return func_graph;
}

void RaiseExceptionForConvertRefDtype(const std::string &func_name, const std::string &ref_type,
                                      const std::string &target_type) {
  MS_LOG(EXCEPTION) << "In op '" << func_name << "', \n"
                    << "the type of writable argument is '" << ref_type << "', "
                    << "but the largest type in the same SignatureEumDtype is '" << target_type
                    << "'. The writable arg type is not equal to the largest type, "
                    << "so can not cast automatically.";
}
void RaiseExceptionForCheckParameter(const std::string &func_name, int i, const std::string &source_type) {
  MS_EXCEPTION(TypeError) << "Function " << func_name << "'s input " << i << " should be a Parameter, but "
                          << source_type << ".";
}
}  // namespace prim
}  // namespace mindspore
