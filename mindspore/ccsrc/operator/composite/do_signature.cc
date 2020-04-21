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

const std::vector<Signature> &GetSignature(const ValuePtr &function) {
  static const auto empty = std::vector<Signature>();
  if (function->isa<Primitive>()) {
    return function->cast<PrimitivePtr>()->signatures();
  } else if (function->isa<MetaFuncGraph>()) {
    return function->cast<MetaFuncGraphPtr>()->signatures();
  }
  return empty;
}

void ProcessDefault(const std::string &func_name, const AbstractBasePtrList &args_spec_list,
                    const std::vector<Signature> &signature, bool has_var, std::vector<AnfNodePtr> *op_inputs) {
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

// Get the largest type of index in the same SignatureEnumDType of arguments.
std::map<SignatureEnumDType, size_t> GetMaxDtypeIndex(const std::vector<SignatureEnumDType> &dtypes,
                                                      const abstract::AbstractBasePtrList &args_spec_list) {
  // record index for signature.dtypes of the same type
  // eg. [T, T1, T, T2, T, T1, T3] -> {{T:(0,2,4)}, {T1:(1,5)}, {T2:(3)}, {T3:(6)}}
  std::map<SignatureEnumDType, std::vector<size_t>> type_indexs;
  for (size_t i = 0; i < dtypes.size(); ++i) {
    auto it = type_indexs.find(dtypes[i]);
    if (it == type_indexs.end()) {
      (void)type_indexs.insert(std::make_pair(dtypes[i], std::vector<size_t>{i}));
    } else {
      it->second.push_back(i);
    }
  }
  // example:sig_dtype:[T,   T1,       T,         T2,    T,         T1,   T3,     T4,  T4]
  // and args type:    [int, Tensor,   Tensor,    float, Tensor,    int,  Tensor, int, float]
  // result:{{T:2},{T1:1}}
  std::map<SignatureEnumDType, size_t> dst_type;
  for (auto it = type_indexs.begin(); it != type_indexs.end(); (void)++it) {
    auto type = it->first;
    auto indexs = it->second;
    // If the number of arguments belonging to the same SignatureEnumDType is less than 2, skip it.
    if (indexs.size() < 2) {
      continue;
    }

    for (const auto &index : indexs) {
      AbstractBasePtr arg_value = args_spec_list[index];
      if (arg_value->isa<abstract::AbstractRef>()) {
        arg_value = arg_value->cast<abstract::AbstractRefPtr>()->ref();
      }

      if (arg_value->isa<abstract::AbstractTensor>()) {
        (void)dst_type.insert(std::make_pair(type, index));
        break;
      }
    }
  }
  return dst_type;
}

AnfNodePtr DoCast(const AnfNodePtr &param, const AnfNodePtr &source_param, const FuncGraphPtr &graph) {
  // op and module import path
  auto prim_dtype = prim::GetPythonOps("dtype", "mindspore.ops.functional");
  MS_EXCEPTION_IF_NULL(prim_dtype);
  // op and module import path
  auto prim_cast_class = prim::GetPythonOps("Cast", "mindspore.ops.operations");
  MS_EXCEPTION_IF_NULL(prim_cast_class);
  auto dtype_node = NewCNode({NewValueNode(prim_dtype), source_param}, graph);
  auto cast_node = NewCNode({NewValueNode(prim_cast_class)}, graph);
  return NewCNode({cast_node, param, dtype_node}, graph);
}

void DoAutoCast(const std::vector<Signature> &signature, const abstract::AbstractBasePtrList &args_spec_list,
                const FuncGraphPtr &graph, std::vector<AnfNodePtr> *op_inputs) {
  std::vector<SignatureEnumDType> dtypes;
  (void)std::transform(signature.begin(), signature.end(), std::back_inserter(dtypes),
                       [](const Signature &sig) { return sig.dtype; });
  int empty_dtype_count = std::count(dtypes.begin(), dtypes.end(), SignatureEnumDType::kDTypeEmptyDefaultValue);
  if (dtypes.empty() || static_cast<int>(dtypes.size()) == empty_dtype_count) {
    return;
  }
  // Stat the index of the arguments with the largest type in the same SignatureEnumDType.
  std::map<SignatureEnumDType, size_t> dst_type = GetMaxDtypeIndex(dtypes, args_spec_list);
  // Identify which arg requires auto cast
  for (size_t i = 0; i < args_spec_list.size(); ++i) {
    AbstractBasePtr arg_value = args_spec_list[i];
    if (arg_value->isa<abstract::AbstractRef>()) {
      arg_value = arg_value->cast<abstract::AbstractRefPtr>()->ref();
    }
    auto it = dst_type.find(dtypes[i]);
    if (it == dst_type.end() || it->second == i || !arg_value->isa<abstract::AbstractScalar>()) {
      continue;
    }
    // get source node for cast
    AnfNodePtr source_node = (*op_inputs)[it->second + 1];
    (*op_inputs)[i + 1] = DoCast((*op_inputs)[i + 1], source_node, graph);
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
  op_inputs.push_back(NewValueNode(function));
  // Assume, the write input of op is always the first input. We check if any write op,
  // and add cast op on other inputs to keep the same type with assigned parameter.
  AnfNodePtr assign_source = nullptr;
  for (size_t i = 0; i < args_spec_list.size(); ++i) {
    AnfNodePtr param = params_list[i];
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
        assign_source = func_graph->NewCNode({NewValueNode(prim::kPrimGetRefOrigin), param});
        param = func_graph->NewCNode({NewValueNode(prim::kPrimGetRefKey), param});
      }
      // If sig is SignatureEnumRW::kRWRef, not do anything.
    }
    // add cast op here
    if (assign_source != nullptr && sig != SignatureEnumRW::kRWWrite) {
      param = DoCast(param, assign_source, func_graph);
    }
    op_inputs.push_back(param);
  }
  // process default
  ProcessDefault(func_name, args_spec_list, signature, has_var, &op_inputs);
  DoAutoCast(signature, args_spec_list, func_graph, &op_inputs);
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
