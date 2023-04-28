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

namespace mindspore {
namespace opt {
namespace irpass {
// {prim::kPrimGetAttr, object, attr}
// {prim::kPrimSetAttr, object, attr, assigned}
// {prim::kPrimResolve, namespace, symbol}
AnfNodePtr Resolver::operator()(const OptimizerPtr &optimizer, const AnfNodePtr &node) {
  PatternNode<AnfNodePtr> object, attr, setattr_target, setattr_attr, setattr_assigned, ns_node, sym_node;
  auto GetAttrLambda = [&node, &object, &attr, &optimizer]() -> AnfNodePtr {
    auto object_node = object.GetNode(node);
    auto attr_node = attr.GetNode(node);
    // {prim::kPrimGetAttr, {getitem, {prim::kPrimResolve, namespace, symbol}, index}, attr}
    // {prim::kPrimGetAttr, {getitem, {prim::kPrimGetAttr, ResolveNode, member}, index}, attr}
    if (parse::IsGetItemCNode(object_node)) {
      return parse::ResolveGetItemWithAttr(optimizer->manager(), object_node, attr_node, node);
    }
    // {prim::kPrimGetAttr, {prim::kPrimResolve, namespace, symbol}, attr}
    if (IsPrimitiveCNode(object_node, prim::kPrimResolve)) {
      // 'node' is getattr node.
      return parse::ResolveSymbolWithAttr(optimizer->manager(), object_node, attr_node, node);
    }
    // {prim::kPrimGetAttr, namespace, attr}
    if (IsValueNode<parse::NameSpace>(object_node)) {
      auto name_space = GetValueNode<parse::NameSpacePtr>(object_node);
      auto attr_str = GetValue<std::string>(GetValueNode(attr_node));
      parse::SymbolPtr symbol = std::make_shared<parse::Symbol>(attr_str);
      auto ret = parse::ResolveSymbol(optimizer->manager(), name_space, symbol, node);
      // If object has no attribute, ret will be null. The attribute may not be used.
      // Let getattr be resolved in static_analysis.
      if (IsValueNode<TypeNull>(ret)) {
        return nullptr;
      }
      return ret;
    }
    // {prim::kPrimGetAttr, MsClassObject, attr}
    if (IsValueNode<parse::MsClassObject>(object_node)) {
      auto ms_class = GetValueNode<parse::MsClassObjectPtr>(object_node)->obj();
      return parse::ResolveMsClassWithAttr(ms_class, attr_node, node);
    }
    return nullptr;
  };

  auto SetAttrLambda = [&node, &setattr_target, &setattr_attr, &setattr_assigned, &optimizer]() -> AnfNodePtr {
    auto target_node = setattr_target.GetNode(node);
    auto attr_node = setattr_attr.GetNode(node);
    auto assigned_node = setattr_assigned.GetNode(node);
    MS_LOG(DEBUG) << "Found setattr: " << target_node->DebugString() << ", " << attr_node->DebugString() << ", "
                  << assigned_node->DebugString();
    // If target_node is not a InterpretedObject, but a resolve CNode, we should convert it here.
    // {prim::kPrimSetAttr, {prim::kPrimResolve, namespace, symbol}, attr, assigned}
    if (IsPrimitiveCNode(target_node, prim::kPrimResolve)) {
      // 'node' is setattr node.
      const auto allow_fallback_runtime = (MsContext::GetInstance()->GetJitSyntaxLevel() == kLax);
      if (!allow_fallback_runtime) {
        MS_LOG(EXCEPTION) << "Not support setattr during JIT Fallback disabled.";
      }
      return parse::ResolveInterpretedObjectOfSetAttr(target_node, attr_node, assigned_node);
    }
    return nullptr;
  };

  auto ResolveLambda = [&node, &ns_node, &sym_node, &optimizer]() -> AnfNodePtr {
    auto name_space = GetValueNode<parse::NameSpacePtr>(ns_node.GetNode(node));
    auto symbol = GetValueNode<parse::SymbolPtr>(sym_node.GetNode(node));
    auto manager = optimizer->manager();
    auto ret = parse::ResolveSymbol(manager, name_space, symbol, node);
    // If object has no attribute, ret will be null. The attribute may not be used.
    // Let getattr be resolved in static_analysis.
    if (IsValueNode<TypeNull>(ret)) {
      return nullptr;
    }
    return ret;
  };

  // {prim::kPrimGetAttr, object, attr}
  MATCH_REPLACE_LAMBDA_IF(node, PPrimitive(prim::kPrimGetAttr, object, attr), GetAttrLambda,
                          attr.CheckFunc(IsValueNode<StringImm>, node));
  // {prim::kPrimSetAttr, object, attr, assigned_value}
  MATCH_REPLACE_LAMBDA_IF(node, PPrimitive(prim::kPrimSetAttr, setattr_target, setattr_attr, setattr_assigned),
                          SetAttrLambda, setattr_attr.CheckFunc(IsValueNode<StringImm>, node));
  // {prim::kPrimResolve, namespace, symbol}
  MATCH_REPLACE_LAMBDA_IF(
    node, PPrimitive(prim::kPrimResolve, ns_node, sym_node), ResolveLambda,
    ns_node.CheckFunc(IsValueNode<parse::NameSpace>, node) && sym_node.CheckFunc(IsValueNode<parse::Symbol>, node));
  return nullptr;
}
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
