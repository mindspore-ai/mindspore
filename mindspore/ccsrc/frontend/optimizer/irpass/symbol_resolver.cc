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

#include "include/common/fallback.h"
#include "mindspore/core/ir/cell.h"
#include "mindspore/core/ops/structure_ops.h"
#include "mindspore/core/ops/framework_ops.h"
#include "pipeline/jit/ps/fallback.h"

namespace mindspore {
namespace opt {
namespace irpass {
// {prim::kPrimGetAttr, object, attr}
// {prim::kPrimSetAttr, object, attr, assigned}
// {prim::kPrimResolve, namespace, symbol}
AnfNodePtr Resolver::operator()(const OptimizerPtr &optimizer, const AnfNodePtr &node) {
  PatternNode<AnfNodePtr> object;
  PatternNode<AnfNodePtr> attr;
  PatternNode<AnfNodePtr> setattr_target;
  PatternNode<AnfNodePtr> setattr_attr;
  PatternNode<AnfNodePtr> setattr_assigned;
  PatternNode<AnfNodePtr> ns_node;
  PatternNode<AnfNodePtr> sym_node;
  auto GetAttrLambda = [&node, &object, &attr, &optimizer]() -> AnfNodePtr {
    auto object_node = object.GetNode(node);
    auto attr_node = attr.GetNode(node);
    MS_LOG(DEBUG) << "Handle getattr, object_node: " << object_node->DebugString()
                  << ", attr_node: " << attr_node->DebugString();
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
      auto res = parse::ResolveSymbol(optimizer->manager(), name_space, symbol, node);
      // If object has no attribute, res will be null. The attribute may not be used.
      // Let getattr be resolved in static_analysis.
      if (IsValueNode<TypeNull>(res)) {
        return nullptr;
      }
      return res;
    }
    // Handle premature converted Cell or MsClass object.
    // {prim::kPrimGetAttr, ValueNode<FuncGraph>(), attr}
    if (IsValueNode<FuncGraph>(object_node)) {
      auto func_value = GetValueNode<FuncGraphPtr>(object_node);
      MS_EXCEPTION_IF_NULL(func_value);
      auto python_obj = func_value->python_obj();
      if (python_obj != nullptr) {
        auto wrapper = dyn_cast_ptr<parse::PyObjectWrapper>(python_obj);
        MS_EXCEPTION_IF_NULL(wrapper);
        auto cls_obj = wrapper->obj();
        if (py::isinstance<Cell>(cls_obj) || py::hasattr(cls_obj, PYTHON_MS_CLASS)) {
          return parse::ResolveClassObjectWithAttr(cls_obj, attr_node, node);
        }
      }
      return nullptr;
    }
    // {prim::kPrimGetAttr, MsClassObject, attr}
    if (IsValueNode<parse::MsClassObject>(object_node)) {
      auto ms_class = GetValueNode<parse::MsClassObjectPtr>(object_node)->obj();
      return parse::ResolveClassObjectWithAttr(ms_class, attr_node, node);
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
      const auto allow_fallback_runtime = (fallback::GetJitSyntaxLevel() == kLax);
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
    auto res = parse::ResolveSymbol(manager, name_space, symbol, node);
    MS_LOG(DEBUG) << "Finish ResolveSymbol, namespace: " << name_space->ToString() << ", symbol: " << symbol->ToString()
                  << ", result: " << (res != nullptr ? res->ToString() : "null");
    // If object has no attribute, res will be null. The attribute may not be used.
    // Let getattr be resolved in static_analysis.
    if (IsValueNode<TypeNull>(res)) {
      return nullptr;
    }
    if (fallback::HasPyObjectInNode(node)) {
      MS_LOG(DEBUG) << "Resolved node has python object, attach it to the node after resolve.";
      fallback::SetPyObjectToNode(res, fallback::GetPyObjectFromNode(node));
    }
    return res;
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
