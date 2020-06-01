
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

#include "operator/composite/multitype_funcgraph.h"
#include <algorithm>
#include <utility>
#include <sstream>

#include "ir/anf.h"
#include "ir/func_graph.h"
#include "pipeline/static_analysis/abstract_value.h"
#include "pipeline/static_analysis/abstract_function.h"
#include "pipeline/static_analysis/dshape.h"
#include "pipeline/static_analysis/param_validator.h"
#include "operator/cc_implementations.h"
#include "optimizer/opt.h"
#include "utils/symbolic.h"
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
FuncGraphPtr MultitypeFuncGraph::GenerateFromTypes(const TypePtrList &types) {
  bool find_fn = false;
  py::function py_fn;
  for (auto &item : fn_cache_py_) {
    TypePtrList sign = item.first;
    if (sign.size() != types.size()) {
      continue;
    }
    bool match = true;
    for (size_t i = 0; i < sign.size(); ++i) {
      if (!IsIdentidityOrSubclass(UnwrapRef(types[i]), sign[i])) {
        match = false;
        break;
      }
    }
    if (!match) {
      continue;
    }
    find_fn = true;
    py_fn = item.second;
    break;
  }
  std::ostringstream buffer;
  buffer << types;
  if (find_fn) {
    FuncGraphPtr func_graph = parse::ParsePythonCode(py_fn);
    if (func_graph == nullptr) {
      MS_LOG(EXCEPTION) << "Fail to parse overload function " << buffer.str();
    }
    MS_LOG(DEBUG) << "Find overload function " << buffer.str() << ", function: " << func_graph->ToString();
    return func_graph;
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
