/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_BPROP_MINDIR_RESOLVE_NODE_RESOLVE_H
#define MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_BPROP_MINDIR_RESOLVE_NODE_RESOLVE_H

#include <string>
#include <memory>
#include "frontend/optimizer/optimizer.h"
#include "frontend/optimizer/irpass.h"
#include "frontend/optimizer/anf_visitor.h"

namespace mindspore {
namespace opt {
namespace irpass {
class ResolveNodeResolve : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    Reset();
    AnfVisitor::Match(prim::kPrimResolve, {IsVNode, IsVNode})(node);

    if (!is_match_) {
      return nullptr;
    }
    py::module mod = python_adapter::GetPyModule(parse::PYTHON_MOD_PARSE_MODULE);
    auto symbol_obj = python_adapter::CallPyModFn(mod, parse::PYTHON_MOD_RESOLVE_FUNCTION, obj_, symbol_);
    auto module_str = py::str(getattr(symbol_obj, "__module__")).cast<std::string>();
    auto name_str = py::str(getattr(symbol_obj, "__name__")).cast<std::string>();
    auto mindir_name_space = std::make_shared<MindIRNameSpace>(module_str);
    auto mindir_symbol = std::make_shared<MindIRSymbol>(name_str);
    auto fg = node->func_graph();
    MS_EXCEPTION_IF_NULL(fg);
    return fg->NewCNode(
      {node->cast_ptr<CNode>()->input(0), NewValueNode(mindir_name_space), NewValueNode(mindir_symbol)});
  }

  void Visit(const ValueNodePtr &vnode) override {
    if (IsValueNode<parse::NameSpace>(vnode)) {
      auto name_space = GetValueNode<parse::NameSpacePtr>(vnode);
      MS_EXCEPTION_IF_NULL(name_space);
      obj_ = name_space->namespace_obj();
    } else if (IsValueNode<parse::Symbol>(vnode)) {
      auto symbol_value = GetValueNode<parse::SymbolPtr>(vnode);
      MS_EXCEPTION_IF_NULL(symbol_value);
      symbol_ = symbol_value->symbol();
    }
    if (!py::isinstance<py::none>(obj_) && !symbol_.empty()) {
      is_match_ = true;
    }
  }

  void Reset() {
    is_match_ = false;
    symbol_.clear();
    obj_ = py::none();
  }

 private:
  bool is_match_{false};
  py::object obj_{py::none()};
  std::string symbol_;
};
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_BPROP_MINDIR_RESOLVE_NODE_RESOLVE_H
