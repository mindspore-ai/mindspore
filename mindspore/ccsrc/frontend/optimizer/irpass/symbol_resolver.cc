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
// {prim::kPrimGetAttr, {prim::kPrimTupleGetItem, {prim::kPrimResolve, namespace, symbol}, index}, attr}
// {prim::kPrimGetAttr, {prim::kPrimResolve, namespace, symbol}, attr}
// {prim::kPrimGetAttr, namespace, attr}
// {prim::kPrimGetAttr, bool, attr}
// {prim::kPrimResolve, namespace, symbol}
AnfNodePtr Resolver::operator()(const OptimizerPtr &optimizer, const AnfNodePtr &node) {
  PatternNode<AnfNodePtr> getattr_operand, ns_node, sym_node, attr_node, bool_node;
  auto GetAttrResolveLambda = [&node, &getattr_operand, &attr_node, &optimizer]() -> AnfNodePtr {
    auto getattr_operand_node = getattr_operand.GetNode(node);
    auto attr = attr_node.GetNode(node);
    constexpr auto recursive_level = 3;
    MS_LOG(DEBUG) << "getattr_operand_node: " << getattr_operand_node->DebugString(recursive_level);

    // {prim::GetAttr, {{prim::Resolve, ..., 'getitem'}, {prim::Resolve, ...}, index}, attr}
    auto getitem_cnode = getattr_operand_node->cast<CNodePtr>();
    constexpr size_t getitem_inputs_size = 3;
    if (getitem_cnode != nullptr && getitem_cnode->size() == getitem_inputs_size) {
      constexpr size_t prim_index = 0;
      auto resolve_getitem_node = getitem_cnode->input(prim_index);
      constexpr size_t resolve_index = 1;
      auto resolve_node = getitem_cnode->input(resolve_index);
      if (IsPrimitiveCNode(resolve_getitem_node, prim::kPrimResolve) &&
          IsPrimitiveCNode(resolve_node, prim::kPrimResolve)) {
        auto resolve_getitem_cnode = resolve_getitem_node->cast<CNodePtr>();
        auto resolve_getitem_symbol = GetValueNode<parse::SymbolPtr>(resolve_getitem_cnode->input(2));
        constexpr auto getitem_symbol = "getitem";
        if (resolve_getitem_symbol->symbol() == getitem_symbol) {
          constexpr size_t position_index = 2;
          auto index_node = getitem_cnode->input(position_index);
          auto [name_space, symbol] = parse::GetNamespaceAndSymbol(resolve_node);
          auto obj = parse::GetObjectFromSequence(name_space, symbol, resolve_node, index_node);
          if (py::isinstance<py::tuple>(obj) || py::isinstance<py::list>(obj)) {
            bool should_incorporate_getattr = true;
            std::vector<AnfNodePtr> inputs;
            inputs.push_back(NewValueNode(prim::kPrimMakeTuple));
            auto sequence = obj.cast<py::sequence>();
            for (size_t i = 0; i < sequence.size(); ++i) {
              if (!parse::data_converter::IsCellInstance(sequence[i])) {
                should_incorporate_getattr = false;
                break;
              }
              auto res = parse::ResolveCellWithAttr(optimizer->manager(), sequence[i], resolve_node, attr);
              inputs.emplace_back(res);
            }
            if (should_incorporate_getattr) {
              auto make_tuple_node = getitem_cnode->func_graph()->NewCNodeInOrder(inputs);
              auto resolve_getitem_name_space = GetValueNode<parse::NameSpacePtr>(resolve_getitem_cnode->input(1));
              auto resolved_getitem_node =
                ResolveSymbol(optimizer->manager(), resolve_getitem_name_space, resolve_getitem_symbol, node);
              auto out =
                getitem_cnode->func_graph()->NewCNodeInOrder({resolved_getitem_node, make_tuple_node, index_node});
              return out;
            }
          } else {
            return parse::ResolveCellWithAttr(optimizer->manager(), obj, resolve_node, attr);
          }
        }
      }
    }

    // {prim::GetAttr, {prim::Resolve, ...}}
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
    return nullptr;
  };

  auto GetAttrLambda = [&node, &ns_node, &attr_node, &optimizer]() -> AnfNodePtr {
    auto name_space = GetValueNode<parse::NameSpacePtr>(ns_node.GetNode(node));
    auto str = GetValue<std::string>(GetValueNode(attr_node.GetNode(node)));
    parse::SymbolPtr symbol = std::make_shared<parse::Symbol>(str);
    return parse::ResolveSymbol(optimizer->manager(), name_space, symbol, node);
  };

  auto ResolveLambda = [&node, &ns_node, &sym_node, &optimizer]() -> AnfNodePtr {
    auto name_space = GetValueNode<parse::NameSpacePtr>(ns_node.GetNode(node));
    auto symbol = GetValueNode<parse::SymbolPtr>(sym_node.GetNode(node));
    auto manager = optimizer->manager();
    return parse::ResolveSymbol(manager, name_space, symbol, node);
  };

  // {prim::kPrimGetAttr, {prim::kPrimResolve, namespace, symbol}, attr}
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
