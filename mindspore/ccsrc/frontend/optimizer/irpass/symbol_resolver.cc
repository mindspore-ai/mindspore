/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "frontend/optimizer/irpass/symbol_resolver.h"

#include <string>
#include <memory>
#include <vector>

namespace mindspore {
namespace opt {
namespace irpass {
// {prim::kPrimGetAttr, {prim::kPrimResolve, namespace, symbol}, attr}
// {prim::kPrimGetAttr, namespace, attr}
// {prim::kPrimGetAttr, bool, attr}
// {prim::kPrimResolve, namespace, symbol}
AnfNodePtr ResolverGetAttrResolve::operator()(const OptimizerPtr &optimizer, const AnfNodePtr &node) {
  constexpr char PARSE_SUPER_NAME[] = "namespace";
  constexpr size_t namespace_index = 1;
  constexpr size_t symbol_index = 2;

  PatternNode<AnfNodePtr> resolve_node, ns_node, sym_node, attr_node, bool_node;
  auto GetAttrResolveLambda = [&node, &resolve_node, &attr_node, &optimizer]() -> AnfNodePtr {
    auto inner = resolve_node.GetNode(node);
    auto attr = attr_node.GetNode(node);
    if (IsPrimitiveCNode(inner, prim::kPrimResolve)) {
      auto resolve_cnode = inner->cast<CNodePtr>();
      auto namespace_node = resolve_cnode->input(namespace_index);
      auto symbol_node = resolve_cnode->input(symbol_index);
      if (!IsValueNode<parse::NameSpace>(namespace_node) || !IsValueNode<parse::Symbol>(symbol_node)) {
        return nullptr;
      }
      // deal with the case of getting attr from a class member
      // and avoid the case of getting attr from self (the result of ParseSuper)
      auto ns = GetValueNode<parse::NameSpacePtr>(namespace_node);
      auto sym = GetValueNode<parse::SymbolPtr>(symbol_node);
      if (ns->module() == parse::RESOLVE_NAMESPACE_NAME_CLASS_MEMBER && sym->symbol() != PARSE_SUPER_NAME) {
        return parse::ResolveCellwithAttr(optimizer->manager(), ns, sym, inner, attr);
      }
    }
    return nullptr;
  };

  auto GetAttrLambda = [&node, &ns_node, &attr_node, &optimizer]() -> AnfNodePtr {
    auto ns = GetValueNode<parse::NameSpacePtr>(ns_node.GetNode(node));
    auto str = GetValue<std::string>(GetValueNode(attr_node.GetNode(node)));
    parse::SymbolPtr sym = std::make_shared<parse::Symbol>(str);
    return parse::ResolveSymbol(optimizer->manager(), ns, sym, node);
  };

  auto ResolveLambda = [&node, &ns_node, &sym_node, &optimizer]() -> AnfNodePtr {
    auto ns = GetValueNode<parse::NameSpacePtr>(ns_node.GetNode(node));
    auto sym = GetValueNode<parse::SymbolPtr>(sym_node.GetNode(node));
    auto manager = optimizer->manager();
    return parse::ResolveSymbol(manager, ns, sym, node);
  };

  // {prim::kPrimGetAttr, {prim::kPrimResolve, namespace, symbol}, attr}
  MATCH_REPLACE_LAMBDA_IF(node, PPrimitive(prim::kPrimGetAttr, resolve_node, attr_node), GetAttrResolveLambda,
                          attr_node.CheckFunc(IsValueNode<StringImm>, node));
  // {prim::kPrimGetAttr, namespace, attr}
  MATCH_REPLACE_LAMBDA_IF(
    node, PPrimitive(prim::kPrimGetAttr, ns_node, attr_node), GetAttrLambda,
    ns_node.CheckFunc(IsValueNode<parse::NameSpace>, node) && attr_node.CheckFunc(IsValueNode<StringImm>, node));
  // {prim::kPrimGetAttr, bool, attr}
  MATCH_REPLACE_IF(
    node, PPrimitive(prim::kPrimGetAttr, bool_node, attr_node), bool_node,
    bool_node.CheckFunc(IsValueNode<BoolImm>, node) && attr_node.CheckFunc(IsValueNode<StringImm>, node));
  // {prim::kPrimResolve, namespace, symbol}
  MATCH_REPLACE_LAMBDA_IF(
    node, PPrimitive(prim::kPrimResolve, ns_node, sym_node), ResolveLambda,
    ns_node.CheckFunc(IsValueNode<parse::NameSpace>, node) && sym_node.CheckFunc(IsValueNode<parse::Symbol>, node));
  return nullptr;
}
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
