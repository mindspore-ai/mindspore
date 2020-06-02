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

#include "operator/composite/do_signature.h"
#include <algorithm>
#include <utility>

#include "pipeline/static_analysis/abstract_value.h"
#include "ir/anf.h"
#include "pipeline/static_analysis/dshape.h"
#include "pipeline/static_analysis/param_validator.h"
#include "operator/cc_implementations.h"
#include "optimizer/opt.h"
#include "utils/symbolic.h"
#include "./common.h"
#include "pybind_api/api_register.h"

namespace mindspore {
// namespace to support composite operators definition
namespace prim {
namespace {
using PatternListType = std::initializer_list<BaseRef>;
const std::map<TypeId, size_t> type_map = {{kNumberTypeBool, 1},    {kNumberTypeInt8, 2},    {kNumberTypeUInt8, 3},
                                           {kNumberTypeInt16, 4},   {kNumberTypeInt32, 5},   {kNumberTypeInt64, 6},
                                           {kNumberTypeFloat16, 7}, {kNumberTypeFloat32, 8}, {kNumberTypeFloat64, 9}};

const std::vector<Signature> &GetSignature(const ValuePtr &function) {
  static const auto empty = std::vector<Signature>();
  if (function->isa<Primitive>() && function->cast<PrimitivePtr>()->has_signature()) {
    return function->cast<PrimitivePyPtr>()->signatures();
  } else if (function->isa<MetaFuncGraph>()) {
    return function->cast<MetaFuncGraphPtr>()->signatures();
  }
  return empty;
}

void ProcessDefault(const std::string &func_name, const AbstractBasePtrList &args_spec_list,
                    const std::vector<Signature> &signature, bool has_var, std::vector<AnfNodePtr> *const op_inputs) {
  std::size_t sig_size = signature.size();
  auto positional_size = sig_size;
  if (has_var) {
    positional_size = sig_size - 1;
  }
  if (args_spec_list.size() < positional_size) {
    for (size_t i = args_spec_list.size(); i < sig_size; ++i) {
      auto default_value = signature[i].default_value;
      if (default_value == nullptr) {
        MS_LOG(EXCEPTION) << "Function " << func_name << "'s input length is not equal to Signature length.";
      } else {
        (*op_inputs).push_back(NewValueNode(default_value));
      }
    }
  }
}
bool CompareTensorScalarType(const TypeId &tensor_type, const size_t &t_type_number, const TypeId &scalar_type,
                             const size_t &s_type_number) {
  if (scalar_type == kNumberTypeFloat16 || scalar_type == kNumberTypeFloat32 || scalar_type == kNumberTypeFloat64) {
    if (tensor_type == kNumberTypeFloat16 || tensor_type == kNumberTypeFloat32 || tensor_type == kNumberTypeFloat64) {
      return t_type_number >= s_type_number;
    }
    return false;
  }
  return true;
}

void SetMaxType(TypeId *max_type_id, TypeId *max_type, size_t *max_type_number, const TypeId type_id, const TypeId type,
                const size_t type_number) {
  *max_type_id = type_id;
  *max_type = type;
  *max_type_number = type_number;
}

bool GetTensorOrScalarTypeInfo(AbstractBasePtr arg_value, bool is_write, TypeId *arg_type_id,
                               TypeId *arg_type = nullptr) {
  if (arg_value->isa<abstract::AbstractRef>()) {
    if (is_write) {
      arg_value = arg_value->cast<abstract::AbstractRefPtr>()->ref_origin();
    } else {
      arg_value = arg_value->cast<abstract::AbstractRefPtr>()->ref();
    }
  }
  if (arg_value->isa<abstract::AbstractTensor>()) {
    auto tensor = arg_value->cast<abstract::AbstractTensorPtr>();
    auto tensor_type = tensor->element()->BuildType();
    MS_EXCEPTION_IF_NULL(tensor_type);
    *arg_type_id = tensor_type->type_id();
    if (arg_type != nullptr) {
      *arg_type = kObjectTypeTensorType;
    }
    return true;
  }
  if (arg_value->isa<abstract::AbstractScalar>()) {
    auto scalar = arg_value->cast<abstract::AbstractScalarPtr>();
    auto scalar_type = scalar->BuildType();
    MS_EXCEPTION_IF_NULL(scalar_type);
    *arg_type_id = scalar_type->type_id();
    if (arg_type != nullptr) {
      *arg_type = kObjectTypeNumber;
    }
    return true;
  }
  return false;
}

TypeId GetMaxTypeId(const abstract::AbstractBasePtrList &args_spec_list, std::vector<size_t> indices,
                    const std::set<size_t> &write_indices) {
  TypeId max_type_id = kTypeUnknown;
  TypeId max_type = kTypeUnknown;
  size_t max_type_number = 0;
  bool has_int8 = false;
  for (const auto &index : indices) {
    TypeId arg_type_id = kTypeUnknown;
    TypeId arg_type = kTypeUnknown;
    auto is_write = (write_indices.find(index) != write_indices.end());
    if (!GetTensorOrScalarTypeInfo(args_spec_list[index], is_write, &arg_type_id, &arg_type)) {
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
      SetMaxType(&max_type_id, &max_type, &max_type_number, arg_type_id, arg_type, it->second);
      continue;
    }

    if (max_type == arg_type) {
      if (it->second > max_type_number) {
        SetMaxType(&max_type_id, &max_type, &max_type_number, arg_type_id, arg_type, it->second);
      }
    } else {
      if (arg_type == kObjectTypeTensorType) {
        if (CompareTensorScalarType(arg_type_id, it->second, max_type_id, max_type_number)) {
          SetMaxType(&max_type_id, &max_type, &max_type_number, arg_type_id, arg_type, it->second);
        }
      } else {
        if (!CompareTensorScalarType(max_type_id, max_type_number, arg_type_id, it->second)) {
          SetMaxType(&max_type_id, &max_type, &max_type_number, arg_type_id, arg_type, it->second);
        }
      }
    }
  }

  if (max_type_id == kNumberTypeUInt8 && has_int8 == true) {
    max_type_id = kNumberTypeInt16;
  }
  return max_type_id;
}

// Get the largest type of index in the same SignatureEnumDType of arguments.
std::map<SignatureEnumDType, TypeId> GetMaxDtype(const std::vector<SignatureEnumDType> &dtypes,
                                                 const abstract::AbstractBasePtrList &args_spec_list,
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
      AbstractBasePtr arg_value = args_spec_list[index];
      if (arg_value->isa<abstract::AbstractRef>()) {
        arg_value = arg_value->cast<abstract::AbstractRefPtr>()->ref();
      }
      if (arg_value->isa<abstract::AbstractTensor>()) {
        has_tensor = true;
        break;
      }
    }
    if (!has_tensor) {
      (void)dst_type.insert(std::make_pair(type, kTypeUnknown));
      continue;
    }
    (void)dst_type.insert(std::make_pair(type, GetMaxTypeId(args_spec_list, indices, write_indices)));
  }
  return dst_type;
}

AnfNodePtr DoCast(const AnfNodePtr &param, const TypeId &type_id, const FuncGraphPtr &graph) {
  auto prim_cast_class = prim::GetPythonOps("Cast", "mindspore.ops.operations");
  MS_EXCEPTION_IF_NULL(prim_cast_class);
  auto dtype_node = NewValueNode(TypeIdToType(type_id));
  auto cast_node = NewCNode({NewValueNode(prim_cast_class)}, graph);
  return NewCNode({cast_node, param, dtype_node}, graph);
}

void DoAutoCast(const std::string &func_name, const std::vector<Signature> &signature,
                const abstract::AbstractBasePtrList &args_spec_list, const FuncGraphPtr &graph,
                std::vector<AnfNodePtr> *const op_inputs, const std::set<size_t> &write_indices) {
  std::vector<SignatureEnumDType> dtypes;
  (void)std::transform(signature.begin(), signature.end(), std::back_inserter(dtypes),
                       [](const Signature &sig) { return sig.dtype; });
  int empty_dtype_count = std::count(dtypes.begin(), dtypes.end(), SignatureEnumDType::kDTypeEmptyDefaultValue);
  if (dtypes.empty() || static_cast<int>(dtypes.size()) == empty_dtype_count) {
    return;
  }
  // Stat the index of the arguments with the largest type in the same SignatureEnumDType.
  std::map<SignatureEnumDType, TypeId> dst_type = GetMaxDtype(dtypes, args_spec_list, write_indices);
  // Identify which arg requires auto cast
  for (size_t i = 0; i < args_spec_list.size(); ++i) {
    auto it = dst_type.find(dtypes[i]);
    if (it == dst_type.end() || it->second == kTypeUnknown) {
      continue;
    }
    auto rw_it = write_indices.find(i);
    auto is_write = (rw_it != write_indices.end());

    TypeId arg_type_id = kTypeUnknown;
    AbstractBasePtr arg_value = args_spec_list[i];
    (void)GetTensorOrScalarTypeInfo(arg_value, is_write, &arg_type_id);
    auto it_map = type_map.find(arg_type_id);
    if (it_map == type_map.end()) {
      continue;
    }
    if (is_write) {
      if (arg_type_id != it->second) {
        MS_LOG(EXCEPTION) << "In op '" << func_name << "', argument '" << args_spec_list[i]
                          << "' can not cast type from '" << TypeIdLabel(arg_type_id) << "' to '"
                          << TypeIdLabel(it->second) << "' automatically.";
      }
      continue;
    }
    if (arg_value->isa<abstract::AbstractTensor>() && arg_type_id == it->second) {
      continue;
    }
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
  op_inputs.push_back(NewValueNode(function));
  // Assume, the write input of op is always the first input. We check if any write op,
  // and add cast op on other inputs to keep the same type with assigned parameter.
  for (size_t i = 0; i < args_spec_list.size(); ++i) {
    AnfNodePtr param = params_list[i];
    if (args_spec_list[i] == nullptr) {
      op_inputs.push_back(param);
      continue;
    }
    SignatureEnumRW sig = SignatureEnumRW::kRWDefault;
    // If sig_size is 0 use defalut.
    if (sig_size > 0 && i < sig_size) {
      sig = signature[i].rw;
    } else if (has_var && i >= sig_size) {
      sig = signature[sig_size - 1].rw;
    }

    TypePtr type = args_spec_list[i]->GetTypeTrack();
    if (type && type->type_id() == kObjectTypeRef) {
      if (sig == SignatureEnumRW::kRWRead) {
        param = func_graph->NewCNode({NewValueNode(prim::kPrimGetRefValue), param});
      } else if (sig == SignatureEnumRW::kRWWrite) {
        param = func_graph->NewCNode({NewValueNode(prim::kPrimGetRefOrigin), param});
        write_indices.insert(i);
      }
      // If sig is SignatureEnumRW::kRWRef, not do anything.
    } else if (sig == SignatureEnumRW::kRWWrite && type->type_id() != kObjectTypeRefKey) {
      MS_EXCEPTION(TypeError) << "Function " << func_name << "'s input " << i << " should be a Parameter.";
    }
    op_inputs.push_back(param);
  }
  // process default
  ProcessDefault(func_name, args_spec_list, signature, has_var, &op_inputs);
  DoAutoCast(func_name, signature, args_spec_list, func_graph, &op_inputs, write_indices);
  return func_graph->NewCNode(op_inputs);
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
  func_graph->set_flags(FUNC_GRAPH_FLAG_CORE, true);
  return func_graph;
}
}  // namespace prim
}  // namespace mindspore
