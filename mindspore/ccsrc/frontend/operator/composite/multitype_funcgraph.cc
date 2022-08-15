
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
#include "include/common/pybind_api/api_register.h"
#include "ir/signature.h"
#include "ir/dtype.h"
#include "pipeline/jit/debug/trace.h"

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
    MS_LOG(EXCEPTION) << "Cannot register as (" << ::mindspore::ToString(types) << ", already registered.";
  }
}

void MultitypeFuncGraph::Register(const TypePtrList &types, const py::function &py_fn) {
  MS_LOG(DEBUG) << "Register type (" << ::mindspore::ToString(types) << ", " << py::str(py_fn.cast<py::object>())
                << ").";
  if (fn_cache_.find(types) != fn_cache_.end()) {
    MS_LOG(EXCEPTION) << "Cannot register as (" << ::mindspore::ToString(types) << ", already registered.";
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
        MS_LOG(EXCEPTION) << type_name << " convert from string error ";
      }
    } else if (py::isinstance<Type>(type_in)) {
      type_ptr = type_in.cast<TypePtr>();
    } else {
      MS_LOG(EXCEPTION) << "Register must be string or `mindspore.dtype.Type`";
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
}  // namespace

// Return Exact match if exists,  else return non ambiguous sub class match
// Return py::none() if matching is ambiguous
const std::pair<py::function, bool> MultitypeFuncGraph::SignMatch(const TypePtrList &types) {
  // Exact match
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
    auto match = true;
    for (size_t i = 0; i < sign.size(); ++i) {
      if (!IsIdentidityOrSubclass(types[i], sign[i])) {
        match = false;
        break;
      }
    }
    if (!match) {
      continue;
    }
    return std::pair(item.second, has_extra_u_monad);
  }
  return std::pair(py::none(), false);
}

FuncGraphPtr MultitypeFuncGraph::GenerateFromTypes(const TypePtrList &types) {
  auto py_fn_pair = SignMatch(types);
  auto py_fn = py_fn_pair.first;
  std::ostringstream buffer;
  buffer << types;
  if (!py_fn.is_none()) {
    FuncGraphPtr func_graph = parse::ParsePythonCode(py_fn);
    if (func_graph == nullptr) {
      MS_LOG(EXCEPTION) << "Fail to parse overload function " << buffer.str() << ".";
    }
    MS_LOG(DEBUG) << "Find overload function " << buffer.str() << ", function: " << func_graph->ToString() << ".";
    if (py_fn_pair.second) {
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
  std::ostringstream oss;
  oss << "The supported types of overload function `" << name_ << "` is: ";
  size_t idx = 0;
  for (auto &item : fn_cache_py_) {
    oss << item.first;
    if (idx != fn_cache_py_.size() - 1) {
      oss << ", ";
    } else {
      oss << ".";
    }
    idx++;
  }
  oss << "\n";
  MS_LOG(EXCEPTION) << "The '" << name_ << "' operation does not support the type " << buffer.str() << ".\n"
                    << oss.str();
}
}  // namespace prim
}  // namespace mindspore
