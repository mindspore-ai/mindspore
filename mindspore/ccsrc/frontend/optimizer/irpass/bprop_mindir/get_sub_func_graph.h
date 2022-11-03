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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_BPROP_MINDIR_GET_SUB_FUNC_GRAPH_H
#define MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_BPROP_MINDIR_GET_SUB_FUNC_GRAPH_H

#include <string>
#include <memory>
#include "frontend/optimizer/optimizer.h"
#include "frontend/optimizer/irpass.h"
#include "frontend/optimizer/anf_visitor.h"

namespace mindspore {
namespace opt {
namespace irpass {
class GetSubFuncGraph : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    Reset();
    AnfVisitor::Match(prim::kPrimResolve, {IsVNode, IsVNode})(node);

    if (!is_match_) {
      return nullptr;
    }
    auto module = python_adapter::GetPyModule(name_space_);
    if (!module || py::isinstance<py::none>(module)) {
      MS_LOG(EXCEPTION) << "Can not get python module: " << name_space_;
    }
    auto func_obj = module.attr(symbol_.c_str());
    ValuePtr convert_result = nullptr;
    bool converted = parse::ConvertData(func_obj, &convert_result);
    if (!converted) {
      MS_LOG(EXCEPTION) << "Failed to convert data for " << py::str(func_obj);
    }
    if (!convert_result->isa<FuncGraph>()) {
      MS_LOG(EXCEPTION) << "The result of convert should be a func_graph, but got " << convert_result->ToString();
    }
    return NewValueNode(convert_result);
  }

  void Visit(const ValueNodePtr &vnode) override {
    if (IsValueNode<MindIRNameSpace>(vnode)) {
      auto mindir_name_space = GetValueNode<MindIRNameSpacePtr>(vnode);
      MS_EXCEPTION_IF_NULL(mindir_name_space);
      name_space_ = mindir_name_space->name_space();
    } else if (IsValueNode<MindIRSymbol>(vnode)) {
      auto mindir_symbol = GetValueNode<MindIRSymbolPtr>(vnode);
      MS_EXCEPTION_IF_NULL(mindir_symbol);
      symbol_ = mindir_symbol->symbol();
    }
    if (!name_space_.empty() && !symbol_.empty()) {
      is_match_ = true;
    }
  }

  void Reset() {
    is_match_ = false;
    name_space_.clear();
    symbol_.clear();
  }

 private:
  bool is_match_{false};
  std::string name_space_;
  std::string symbol_;
};
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_BPROP_MINDIR_GET_CLASS_TYPE_H
