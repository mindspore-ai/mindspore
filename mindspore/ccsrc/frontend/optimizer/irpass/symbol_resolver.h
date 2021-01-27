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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_SYMBOL_RESOLVER_H_
#define MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_SYMBOL_RESOLVER_H_

#include <string>
#include <memory>
#include <vector>

#include "frontend/optimizer/optimizer.h"
#include "frontend/optimizer/optimizer_caller.h"
#include "frontend/optimizer/irpass.h"
#include "frontend/optimizer/anf_visitor.h"
#include "frontend/operator/ops.h"
#include "ir/pattern_matcher.h"
#include "pipeline/jit/parse/data_converter.h"
#include "pipeline/jit/parse/python_adapter.h"
#include "pipeline/jit/parse/parse_base.h"

namespace mindspore {
namespace opt {
namespace irpass {
const char PARSE_SUPER_NAME[] = "namespace";

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
class ResolverGetAttr : public AnfVisitor {
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

// {prim::kPrimGetAttr, {prim::kPrimResolve, ns_node, sym_node}, attr_node}
class ResolverGetAttrResolve : public OptimizerCaller {
 public:
  AnfNodePtr operator()(const OptimizerPtr &optimizer, const AnfNodePtr &node) override {
    PatternNode<AnfNodePtr> ns_node, sym_node, attr_node;
    auto ResolveAttrLambda = [&node, &ns_node, &sym_node, &attr_node, &optimizer]() -> AnfNodePtr {
      auto node_to_getattr = node->cast<CNodePtr>()->input(1);
      std::string attr_as_string = GetValueNode<StringImmPtr>(attr_node.GetNode(node))->value();

      auto ns_ = GetValueNode<parse::NameSpacePtr>(ns_node.GetNode(node));
      auto sym_ = GetValueNode<parse::SymbolPtr>(sym_node.GetNode(node));
      if (ns_->module() == parse::RESOLVE_NAMESPACE_NAME_CLASS_MEMBER && sym_->symbol() != PARSE_SUPER_NAME) {
        // deal with the case of getting attr from a class member
        // and avoid the case of getting attr from self (the result of ParseSuper)
        auto result = parse::ResolveCellwithAttr(optimizer->manager(), ns_, sym_, node_to_getattr, attr_as_string);
        return result;
      }
      return nullptr;
    };
    MATCH_REPLACE_LAMBDA_IF(
      node, PPrimitive(prim::kPrimGetAttr, PPrimitive(prim::kPrimResolve, ns_node, sym_node), attr_node),
      ResolveAttrLambda, attr_node.CheckFunc(IsValueNode<StringImm>, node));

    return nullptr;
  }
};

class ResolverResolveAndGetAttr : public OptimizerCaller {
 public:
  ResolverResolveAndGetAttr() {
    resolver_optimizers_ = {std::make_shared<ResolverGetAttrResolve>(), std::make_shared<ResolverResolve>(),
                            std::make_shared<ResolverGetAttr>()};
  }
  ~ResolverResolveAndGetAttr() = default;

  AnfNodePtr operator()(const OptimizerPtr &optimizer, const AnfNodePtr &node) override {
    AnfNodePtr new_node;
    for (const auto &resolver_opt : resolver_optimizers_) {
      new_node = (*resolver_opt)(optimizer, node);
      if (new_node != nullptr) {
        return new_node;
      }
    }
    return nullptr;
  }

 private:
  std::vector<OptimizerCallerPtr> resolver_optimizers_{};
};
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_SYMBOL_RESOLVER_H_
