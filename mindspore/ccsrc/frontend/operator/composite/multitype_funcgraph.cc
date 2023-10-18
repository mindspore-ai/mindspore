
/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "frontend/operator/composite/multitype_funcgraph.h"

#include "abstract/abstract_function.h"
#include "abstract/dshape.h"
#include "frontend/optimizer/opt.h"
#include "utils/ms_context.h"
#include "pipeline/jit/ps/fallback.h"
#include "include/common/pybind_api/api_register.h"
#include "include/common/fallback.h"
#include "ir/signature.h"
#include "ir/dtype.h"
#include "pipeline/jit/ps/debug/trace.h"

namespace mindspore {
// namespace to support composite operators definition
namespace prim {
MultitypeFuncGraph::MultitypeFuncGraph(const std::string &name) : MetaFuncGraph(name) {
  fn_cache_.clear();
  // def multitype(*args:ref):
  signatures_ = std::vector<Signature>({{"args", SignatureEnumRW::kRWRef, SignatureEnumKind::kKindVarPositional}});
}

void MultitypeFuncGraph::Register(const TypePtrList &types, specialize_fn s_fn) {
  MS_LOG(DEBUG) << "Register type (" << ::mindspore::ToString(types) << ".";
  auto result = fn_cache_.emplace(types, s_fn);
  if (!result.second) {
    MS_LOG(INTERNAL_EXCEPTION) << "Cannot register as (" << ::mindspore::ToString(types) << ", already registered.";
  }
}

void MultitypeFuncGraph::Register(const TypePtrList &types, const py::function &py_fn) {
  MS_LOG(DEBUG) << "Register type (" << ::mindspore::ToString(types) << ", " << py::str(py_fn.cast<py::object>())
                << ").";
  if (fn_cache_.find(types) != fn_cache_.end()) {
    MS_LOG(INTERNAL_EXCEPTION) << "Cannot register as (" << ::mindspore::ToString(types) << ", already registered.";
  }
  fn_cache_py_[types] = py_fn;
}

void MultitypeFuncGraph::PyRegister(const py::tuple &tuple, const py::function &py_fn) {
  TypePtrList types;
  for (size_t it = 0; it < tuple.size(); ++it) {
    py::object type_in = tuple[it];
    TypePtr type_ptr = nullptr;
    if (py::isinstance<py::str>(type_in)) {
      auto type_name = type_in.cast<std::string>();
      type_ptr = StringToType(type_name);
      if (type_ptr == nullptr) {
        MS_LOG(INTERNAL_EXCEPTION) << type_name << " convert from string error ";
      }
    } else if (py::isinstance<Type>(type_in)) {
      type_ptr = type_in.cast<TypePtr>();
    } else {
      MS_LOG(INTERNAL_EXCEPTION) << "Register must be string or `mindspore.dtype.Type`";
    }
    types.push_back(type_ptr);
  }
  Register(types, py_fn);
}

namespace {
bool HasUMonadType(const TypePtrList &types) {
  auto types_size = types.size();
  // If UMonad is the only type, ignore it.
  if (types_size > 1) {
    auto last_type = types[types_size - 1];
    if (IsIdentidityOrSubclass(last_type, kUMonadType)) {
      MS_LOG(DEBUG) << "Have Extra UMonad type";
      return true;
    }
  }
  return false;
}

size_t GetTypesPrefixMatchedNum(const TypePtrList &types, const TypePtrList &sign) {
  for (size_t i = 0; i < sign.size(); ++i) {
    if (!IsIdentidityOrSubclass(types[i], sign[i])) {
      return i;
    }
  }
  return sign.size();
}

std::string IntToNumber(const std::string &v) {
  static mindspore::HashMap<std::string, std::string> int_to_number{
    {"Int64", "Number"}, {"Int32", "Number"}, {"Int8", "Number"}};
  auto iter = int_to_number.find(v);
  if (iter != int_to_number.end()) {
    return iter->second;
  } else {
    return v;
  }
}

std::vector<mindspore::TypePtrList> GetSortedCache(const TypeListMap<py::function> &fn_cache_py_,
                                                   const TypePtrList &types, size_t match_max_idx) {
  std::vector<mindspore::TypePtrList> cache_vec;
  (void)std::transform(fn_cache_py_.begin(), fn_cache_py_.end(), back_inserter(cache_vec),
                       [](const auto &fcp) { return fcp.first; });

  for (auto it = cache_vec.begin(); it != cache_vec.end();) {
    if (GetTypesPrefixMatchedNum(types, *it) != match_max_idx) {
      it = cache_vec.erase(it);
    } else {
      ++it;
    }
  }

  auto comparator = [match_max_idx](const mindspore::TypePtrList &a, const mindspore::TypePtrList &b) {
    if (a.size() > b.size()) {
      return false;
    }
    if (a.size() < b.size()) {
      return true;
    }
    for (size_t i = match_max_idx; i < a.size(); ++i) {
      if (a[i]->type_id() == b[i]->type_id()) {
        continue;
      }
      return a[i]->type_id() < b[i]->type_id();
    }
    return false;
  };
  std::sort(cache_vec.begin(), cache_vec.end(), comparator);
  return cache_vec;
}
}  // namespace

// Return Exact match if exists,  else return non ambiguous sub class match
// Return py::none() if matching is ambiguous
const std::tuple<py::function, bool, size_t> MultitypeFuncGraph::SignMatch(const TypePtrList &types) {
  // Exact match
  size_t match_max_idx = 0;
  for (auto &item : fn_cache_py_) {
    bool has_extra_u_monad = false;
    TypePtrList sign = item.first;
    auto types_size = types.size();
    if (sign.size() != types_size) {
      // Don't take the UMonad type into account.
      has_extra_u_monad = (types_size > 1) && (sign.size() == (types_size - 1)) && HasUMonadType(types);
      if (!has_extra_u_monad) {
        continue;
      }
    }
    size_t match_idx = GetTypesPrefixMatchedNum(types, sign);
    if (match_idx > match_max_idx) {
      match_max_idx = match_idx;
    }
    if (match_idx == sign.size()) {
      return std::make_tuple(item.second, has_extra_u_monad, sign.size());
    }
  }
  return std::make_tuple(py::none(), false, match_max_idx);
}

const std::string MultitypeFuncGraph::PrintMatchFailLog(const TypeListMap<py::function>, const TypePtrList &types,
                                                        size_t match_max_idx, bool has_any) {
  std::ostringstream buffer1;
  py::list types_list;
  bool external_flag = false;
  buffer1 << "<";
  for (size_t i = 0; i < types.size(); ++i) {
    if (types[i]->type_id() == kMetaTypeExternal) {
      external_flag = true;
    }
    std::string types_to_int = IntToNumber(TypeIdLabel(types[i]->type_id()));
    types_list.append(types_to_int);
    if (i != types.size() - 1) {
      buffer1 << types_to_int << ", ";
    } else {
      buffer1 << types_to_int << ">";
    }
  }
  if (has_any && match_max_idx >= types_list.size()) {
    MS_LOG(EXCEPTION)
      << "In the inputs of operation '" << name_
      << "', there are unsupported syntax in graph mode. Those codes would be fallen back to python interpreter, "
      << "which is not supported for operation '" << name_ << "'.";
  }

  std::ostringstream buffer2;
  if (match_max_idx == 1) {
    buffer2 << "When first argument is '" << types_list[0].str() << "', ";
  }
  if (match_max_idx > 1) {
    buffer2 << "When arguments are given as ";
    for (size_t i = 0; i < match_max_idx; ++i) {
      buffer2 << "'" << types_list[i].str() << "', ";
    }
  }

  std::ostringstream oss;
  oss << "For operation '" << name_ << "', current input arguments types are " << buffer1.str() << ". The "
      << (match_max_idx + 1) << "-th argument type '" << types_list[match_max_idx].str() << "' is not supported now.\n"
      << buffer2.str() << "the support arguments types of '" << name_ << "' operation as follows:\n";
  const std::vector<mindspore::TypePtrList> cache_vec = GetSortedCache(fn_cache_py_, types, match_max_idx);
  for (auto &item : cache_vec) {
    oss << "<";
    for (size_t i = 0; i < item.size(); ++i) {
      std::string item_str = item[i]->ToString();
      (void)item_str.erase(std::remove(item_str.begin(), item_str.end(), ' '), item_str.end());
      if (i != item.size() - 1) {
        oss << item_str << ", ";
      } else {
        oss << item_str << ">\n";
      }
    }
  }

  if (!doc_url_.empty()) {
    oss << "For more details with '" << name_ << "', please refer to " << doc_url_ << "\n";
  } else if (external_flag) {
    oss << "For more details with 'External', please refer to "
           "https://www.mindspore.cn/search?inputValue=%27External%27%20TypeError\n";
  }

  return oss.str();
}

FuncGraphPtr MultitypeFuncGraph::GenerateFromTypes(const TypePtrList &types) {
  auto [py_fn, has_extra_u_monad, match_max_idx] = SignMatch(types);
  std::ostringstream buffer;
  buffer << types;
  bool has_any = std::any_of(types.begin(), types.end(), [](const TypePtr &type) { return type->isa<AnyType>(); });
  if (!py_fn.is_none() && (!has_any || name_ == "add_backward")) {
    FuncGraphPtr func_graph = parse::ParsePythonCode(py_fn);
    if (func_graph == nullptr) {
      MS_LOG(INTERNAL_EXCEPTION) << "Fail to parse overload function " << buffer.str() << ".";
    }
    MS_LOG(DEBUG) << "Find overload function " << buffer.str() << ", function: " << func_graph->ToString() << ".";
    if (has_extra_u_monad) {
      MS_LOG(DEBUG) << "Add extra UMoand type for func_graph: " << func_graph->ToString() << ".";
      func_graph->add_parameter();
    }
    return func_graph;
  }
  auto stub = GenerateStubFunc(types);
  if (stub != nullptr) {
    MS_LOG(DEBUG) << "GenerateStubFunc " << buffer.str() << ", function: " << stub->ToString() << ".";
    return stub;
  }

  const auto allow_fallback_runtime = (fallback::GetJitSyntaxLevel() == kLax);
  bool has_dic = std::any_of(types.begin(), types.end(), [](const TypePtr &type) { return type->isa<Dictionary>(); });
  if (allow_fallback_runtime && (!need_raise_ || !has_dic)) {
    FuncGraphPtr func_graph = std::make_shared<FuncGraph>();
    AnfNodePtrList node_inputs{};
    for (auto type : types) {
      node_inputs.push_back(func_graph->add_parameter());
    }
    if (name_ == "ones_like_leaf") {
      AnfNodePtr template_node = fallback::GenerateOnesOrZerosLikeNode(func_graph, node_inputs[0], "ones_like");
      func_graph->set_output(template_node);
      return func_graph;
    }
    if (name_ == "zeros_like_leaf") {
      AnfNodePtr template_node = fallback::GenerateOnesOrZerosLikeNode(func_graph, node_inputs[0], "zeros_like");
      func_graph->set_output(template_node);
      return func_graph;
    }
    auto ret_node = fallback::GeneratePyInterpretNodeWithScriptSrc(func_graph, types, node_inputs, node_expr_src_);
    if (ret_node != nullptr) {
      func_graph->set_output(ret_node);
      return func_graph;
    }
  }

  auto match_fail_log = PrintMatchFailLog(fn_cache_py_, types, match_max_idx, has_any);
  MS_LOG(EXCEPTION) << match_fail_log;
}
}  // namespace prim
}  // namespace mindspore
