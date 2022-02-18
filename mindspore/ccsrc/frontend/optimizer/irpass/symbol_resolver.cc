/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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
// {prim::kPrimGetAttr, {getitem, {prim::kPrimResolve, namespace, symbol}, index}, attr}
// {prim::kPrimGetAttr, namespace, attr}
// {prim::kPrimGetAttr, bool, attr}
// {prim::kPrimResolve, namespace, symbol}
AnfNodePtr Resolver::operator()(const OptimizerPtr &optimizer, const AnfNodePtr &node) {
  PatternNode<AnfNodePtr> getattr_operand, ns_node, sym_node, attr_node, bool_node;
  auto GetAttrResolveLambda = [&node, &getattr_operand, &attr_node, &optimizer]() -> AnfNodePtr {
    auto getattr_operand_node = getattr_operand.GetNode(node);
    auto attr = attr_node.GetNode(node);

    // {prim::kPrimGetAttr, {prim::kPrimResolve, namespace, symbol}, attr}
    if (IsPrimitiveCNode(getattr_operand_node, prim::kPrimResolve)) {
      auto [name_space, symbol] = parse::GetNamespaceAndSymbol(getattr_operand_node);
      auto module_name = name_space->module();
      constexpr std::string_view parse_super_name = "namespace";
      if (module_name.find(parse::RESOLVE_NAMESPACE_NAME_CLASS_MEMBER) != std::string::npos &&
          symbol->symbol() != parse_super_name) {
        auto obj = parse::GetSymbolObject(name_space, symbol, node);
        return parse::ResolveCellWithAttr(optimizer->manager(), obj, getattr_operand_node, attr);
      }
    }

    // {prim::kPrimGetAttr, {getitem, {prim::kPrimResolve, namespace, symbol}, index}, attr}
    auto operand_cnode = getattr_operand_node->cast<CNodePtr>();
    constexpr size_t getitem_inputs_size = 3;
    if (operand_cnode != nullptr && operand_cnode->size() == getitem_inputs_size) {
      constexpr auto prim_index = 0;
      constexpr auto resolve_index = 1;
      constexpr auto index_index = 2;
      auto prim_node = operand_cnode->input(prim_index);
      auto resolve_node = operand_cnode->input(resolve_index);
      auto index_node = operand_cnode->input(index_index);
      if (!parse::IsResolveNodeWithGetItem(prim_node) || !IsPrimitiveCNode(resolve_node, prim::kPrimResolve)) {
        return nullptr;
      }
      auto [name_space, symbol] = parse::GetNamespaceAndSymbol(resolve_node);
      auto obj = parse::GetObjectFromSequence(name_space, symbol, resolve_node, index_node);
      if (py::isinstance<py::tuple>(obj) || py::isinstance<py::list>(obj)) {
        return parse::ResolveSequenceWithAttr(optimizer->manager(), obj, resolve_node, attr, operand_cnode);
      }
      return parse::ResolveCellWithAttr(optimizer->manager(), obj, resolve_node, attr);
    }
    return nullptr;
  };

  auto GetAttrLambda = [&node, &ns_node, &attr_node, &optimizer]() -> AnfNodePtr {
    auto name_space = GetValueNode<parse::NameSpacePtr>(ns_node.GetNode(node));
    auto str = GetValue<std::string>(GetValueNode(attr_node.GetNode(node)));
    parse::SymbolPtr symbol = std::make_shared<parse::Symbol>(str);
    auto manager = optimizer->manager();
    return parse::ResolveSymbol(manager, name_space, symbol, node);
  };

  auto ResolveLambda = [&node, &ns_node, &sym_node, &optimizer]() -> AnfNodePtr {
    auto name_space = GetValueNode<parse::NameSpacePtr>(ns_node.GetNode(node));
    auto symbol = GetValueNode<parse::SymbolPtr>(sym_node.GetNode(node));
    auto manager = optimizer->manager();
    return parse::ResolveSymbol(manager, name_space, symbol, node);
  };

  // {prim::kPrimGetAttr, {prim::kPrimResolve, namespace, symbol}, attr}
  // {prim::kPrimGetAttr, {getitem, {prim::kPrimResolve, namespace, symbol}, index}, attr}
  MATCH_REPLACE_LAMBDA_IF(node, PPrimitive(prim::kPrimGetAttr, getattr_operand, attr_node), GetAttrResolveLambda,
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
