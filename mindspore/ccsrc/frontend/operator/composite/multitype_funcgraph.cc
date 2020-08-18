
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
#include <utility>
#include <sstream>

#include "abstract/abstract_function.h"
#include "abstract/dshape.h"
#include "frontend/optimizer/opt.h"
#include "utils/ms_context.h"
#include "pybind_api/api_register.h"
#include "./common.h"
#include "ir/signature.h"
#include "debug/trace.h"

namespace mindspore {
// namespace to support composite operators definition
namespace prim {
MultitypeFuncGraph::MultitypeFuncGraph(const std::string &name) : MetaFuncGraph(name) {
  fn_cache_.clear();
  signatures_ = std::vector<Signature>({// def multitype(*args:ref):
                                        {"args", SignatureEnumRW::kRWRef, SignatureEnumKind::kKindVarPositional}});
}

void MultitypeFuncGraph::Register(const TypePtrList &types, specialize_fn s_fn) {
  MS_LOG(DEBUG) << "Register type (" << ::mindspore::ToString(types) << ".";
  auto fn = fn_cache_.find(types);
  if (fn != fn_cache_.end()) {
    MS_LOG(EXCEPTION) << "Cannot register as (" << ::mindspore::ToString(types) << ", already registered.";
  }
  fn_cache_[types] = s_fn;
}

void MultitypeFuncGraph::Register(const TypePtrList &types, const py::function &py_fn) {
  MS_LOG(DEBUG) << "Register type (" << ::mindspore::ToString(types) << ", " << std::string(py_fn.str()) << ").";
  auto fn = fn_cache_.find(types);
  if (fn != fn_cache_.end()) {
    MS_LOG(EXCEPTION) << "Cannot register as (" << ::mindspore::ToString(types) << ", already registered.";
  }
  fn_cache_py_[types] = py_fn;
}

void MultitypeFuncGraph::Register(const std::vector<std::string> &types_name, const py::function &py_fn) {
  TypePtrList types;
  for (auto &type_name : types_name) {
    auto type_ptr = StringToType(type_name);
    if (type_ptr == nullptr) {
      MS_LOG(EXCEPTION) << type_name << " convert from string error ";
    }
    types.push_back(type_ptr);
  }
  Register(types, py_fn);
}

void MultitypeFuncGraph::PyRegister(const py::tuple &tuple, const py::function &py_fn) {
  std::vector<std::string> types_name;
  for (size_t it = 0; it < tuple.size(); ++it) {
    py::object name_py = tuple[it];
    if (py::isinstance<py::str>(name_py)) {
      types_name.push_back(name_py.cast<std::string>());
      continue;
    }
    MS_LOG(EXCEPTION) << "Register must be string";
  }
  Register(types_name, py_fn);
}
static TypePtr UnwrapRef(const TypePtr &type) {
  if (type->isa<RefType>()) {
    return type->cast<RefTypePtr>()->subtype();
  }
  return type;
}

// Return Exact match if exists,  else return non ambiguous sub class match
// Return py::none() if matching is ambiguous
const py::function MultitypeFuncGraph::SignMatch(const TypePtrList &types) {
  // Exact match
  for (auto &item : fn_cache_py_) {
    TypePtrList sign = item.first;
    if (sign.size() != types.size()) {
      continue;
    }
    auto match = true;
    for (size_t i = 0; i < sign.size(); ++i) {
      if (!IsIdentidityOrSubclass(UnwrapRef(types[i]), sign[i])) {
        match = false;
        break;
      }
    }
    if (!match) {
      continue;
    }
    return item.second;
  }
  return py::none();
}

FuncGraphPtr MultitypeFuncGraph::GenerateFromTypes(const TypePtrList &types) {
  auto py_fn = SignMatch(types);
  std::ostringstream buffer;
  buffer << types;
  if (py_fn != py::none()) {
    FuncGraphPtr func_graph = parse::ParsePythonCode(py_fn);
    if (func_graph == nullptr) {
      MS_LOG(EXCEPTION) << "Fail to parse overload function " << buffer.str();
    }
    MS_LOG(DEBUG) << "Find overload function " << buffer.str() << ", function: " << func_graph->ToString();
    return func_graph;
  }
  auto stub = GenerateStubFunc(types);
  if (stub != nullptr) {
    MS_LOG(DEBUG) << "GenerateStubFunc " << buffer.str() << ", function: " << stub->ToString();
    return stub;
  }
  std::ostringstream oss;
  oss << "There are " << fn_cache_py_.size() << " prototypes for overload function `" << name_
      << "`, corresponding location info:\n";
  int idx = 0;
  for (auto &item : fn_cache_py_) {
    FuncGraphPtr func_graph = parse::ParsePythonCode(item.second);
    if (func_graph == nullptr) {
      MS_LOG(WARNING) << "Fail to parse Python code for function `" << name_ << "`.";
      continue;
    }
    oss << ++idx << ". " << item.first << "\n  " << trace::GetDebugInfo(func_graph->debug_info()) << "\n";
  }
  MS_LOG(EXCEPTION) << "The '" << name_ << "' operation does not support the type " << buffer.str() << "\n"
                    << oss.str();
}

REGISTER_PYBIND_DEFINE(MultitypeFuncGraph_, ([](const py::module *m) {
                         (void)py::class_<MultitypeFuncGraph, MetaFuncGraph, std::shared_ptr<MultitypeFuncGraph>>(
                           *m, "MultitypeFuncGraph_")
                           .def(py::init<std::string &>())
                           .def("register_fn", &MultitypeFuncGraph::PyRegister);
                       }));
}  // namespace prim
}  // namespace mindspore
