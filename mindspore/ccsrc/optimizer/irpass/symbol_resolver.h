/**
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

#ifndef MINDSPORE_CCSRC_OPTIMIZER_IRPASS_SYMBOL_RESOLVER_H_
#define MINDSPORE_CCSRC_OPTIMIZER_IRPASS_SYMBOL_RESOLVER_H_

#include <string>
#include <memory>

#include "optimizer/optimizer.h"
#include "optimizer/irpass.h"
#include "ir/visitor.h"
#include "operator/ops.h"
#include "pipeline/parse/data_converter.h"
#include "pipeline/parse/python_adapter.h"

namespace mindspore {
namespace opt {
namespace irpass {
// {prim::kPrimResolve, Ns, Sym}
class ResolverResolve : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &optimizer, const AnfNodePtr &node) override {
    Reset();
    AnfVisitor::Match(prim::kPrimResolve, {IsVNode, IsVNode})(node);
    if (sym_ != nullptr) {
      return parse::ResolveSymbol(optimizer->manager(), ns_, sym_, node);
    }
    return nullptr;
  }

  void Visit(const ValueNodePtr &vnode) override {
    if (IsValueNode<parse::NameSpace>(vnode)) {
      ns_ = GetValueNode<parse::NameSpacePtr>(vnode);
    } else if (ns_ != nullptr && IsValueNode<parse::Symbol>(vnode)) {
      sym_ = GetValueNode<parse::SymbolPtr>(vnode);
    }
  }

  void Reset() {
    ns_ = nullptr;
    sym_ = nullptr;
  }

 private:
  parse::NameSpacePtr ns_{nullptr};
  parse::SymbolPtr sym_{nullptr};
};

// {prim::kPrimGetAttr, Ns, Str}
class ResolverGetattr : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &optimizer, const AnfNodePtr &node) override {
    Reset();
    AnfVisitor::Match(prim::kPrimGetAttr, {IsVNode, IsVNode})(node);
    if (sym_ != nullptr) {
      return parse::ResolveSymbol(optimizer->manager(), ns_, sym_, node);
    }
    return nullptr;
  }

  void Visit(const AnfNodePtr &node) override {
    if (IsValueNode<parse::NameSpace>(node)) {
      ns_ = GetValueNode<parse::NameSpacePtr>(node);
    } else if (ns_ != nullptr && IsValueNode<StringImm>(node)) {
      auto str = GetValue<std::string>(GetValueNode(node));
      sym_ = std::make_shared<parse::Symbol>(str);
    }
  }

  void Reset() {
    ns_ = nullptr;
    sym_ = nullptr;
  }

 private:
  parse::NameSpacePtr ns_{nullptr};
  parse::SymbolPtr sym_{nullptr};
};
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_OPTIMIZER_IRPASS_SYMBOL_RESOLVER_H_
