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
#include <string>
#include <memory>
#include "ir/anf.h"
#include "frontend/optimizer/optimizer.h"
#include "frontend/optimizer/irpass.h"
#include "frontend/optimizer/irpass/bprop_mindir/reslove_primitive_attr.h"
namespace mindspore {
namespace opt {
namespace irpass {
namespace {
inline CNode *GetCallNode(const AnfNodePtr &node) {
  if (!node->isa<CNode>()) {
    return nullptr;
  }
  auto apply_node = dyn_cast_ptr<CNode>(node);
  if (apply_node->inputs().empty()) {
    MS_LOG(DEBUG) << " CNode input is empty!";
    return nullptr;
  }
  const auto &func_node = apply_node->input(0);
  MS_EXCEPTION_IF_NULL(func_node);
  auto func_c_node = dyn_cast_ptr<CNode>(func_node);
  if (func_c_node == nullptr) {
    return nullptr;
  }
  return func_c_node;
}

inline bool IsGetAttrPrimNode(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto primitive = GetValuePtr<Primitive>(node);
  if (primitive == nullptr) {
    return false;
  }
  return primitive->name() == "getattr";
}

inline Primitive *GetPrimNode(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto primitive = GetValuePtr<Primitive>(node);
  return primitive;
}

inline bool IsGetAttrDictStringImmValueNode(const AnfNodePtr &node) {
  auto str = GetValuePtr<StringImm>(node);
  if (str == nullptr) {
    return false;
  }
  return str->value() == "get_attr_dict";
}
}  // namespace
bool ReslovePrimitiveAttr::IsStringAttrValueNode(const AnfNodePtr &node) {
  auto attr_name = GetValuePtr<StringImm>(node);
  if (attr_name == nullptr) {
    return false;
  }
  attr_name_ = attr_name->value();
  return true;
}

bool ReslovePrimitiveAttr::IsCallPrimitiveAttrDictNode(const AnfNodePtr &node) {
  auto call_node = GetCallNode(node);
  if (call_node == nullptr) {
    return false;
  }
  return IsGetAttrDictFuncNode(call_node);
}

bool ReslovePrimitiveAttr::IsCNodeMinIRMetaGraphGetItem(const AnfNodePtr &node) {
  auto cnode = dyn_cast_ptr<CNode>(node);
  if (cnode == nullptr) {
    return false;
  }
  auto meta_func = GetValuePtr<MindIRMetaFuncGraph>(cnode->input(0));
  if (meta_func == nullptr) {
    return false;
  }

  if (meta_func->name() != "getitem") {
    return false;
  }

  auto dict_node = cnode->input(1);
  auto string_node = cnode->input(2);
  return IsStringAttrValueNode(string_node) && IsCallPrimitiveAttrDictNode(dict_node);
}

bool ReslovePrimitiveAttr::IsGetAttrDictFuncNode(const CNode *node) {
  constexpr auto kGetAttrSize = 3;
  if (node->inputs().size() != kGetAttrSize) {
    return false;
  }
  auto attr_prim_node = node->input(0);
  auto prim_node = node->input(1);
  auto attr_name_node = node->input(2);
  primitive_ = GetPrimNode(prim_node);
  return IsGetAttrPrimNode(attr_prim_node) && primitive_ != nullptr && IsGetAttrDictStringImmValueNode(attr_name_node);
}
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
