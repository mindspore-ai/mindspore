/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include "include/backend/optimizer/visitor.h"

#include <vector>
#include <memory>
#include <algorithm>
#include "include/backend/optimizer/pattern_engine.h"
#include "utils/any.h"
#include "ir/anf.h"
#include "ir/func_graph.h"
#include "utils/log_adapter.h"

namespace mindspore {
bool CheckIfNeedExpand(const std::vector<BaseRef> &list) {
  return std::any_of(list.begin(), list.end(), [](const BaseRef &any) { return utils::isa<Seq>(any); });
}

std::shared_ptr<VectorRef> ExpandList(const std::vector<BaseRef> &list) {
  std::shared_ptr<VectorRef> new_list = std::make_shared<VectorRef>();
  for (auto &item : list) {
    if (utils::isa<Seq>(item)) {
      const Seq &seq = utils::cast<Seq>(item);
      new_list->insert(new_list->end(), seq.begin(), seq.end());
    } else {
      new_list->push_back(item);
    }
  }
  return new_list;
}

static BaseRef GetVar(const BaseRef &x) {
  if (utils::isa<AnfNodePtr>(x)) {
    auto node = utils::cast<AnfNodePtr>(x);
    MS_LOG(DEBUG) << "TypeString [" + node->type_name() + "]";
    if (node->isa<VarNode>()) {
      MS_LOG(DEBUG) << "IsVarNode " + node->cast<VarNodePtr>()->var_->ToString();
      return node->cast<VarNodePtr>()->var_;
    }
  }
  return x;
}

bool Visitor::Visit(const VectorRef &v_any, VectorRef *const values_ref, BaseRef *const visit_out) const {
  std::vector<BaseRef> out;
  for (const auto &element : v_any) {
    out.push_back(element);
    values_ref->push_back(GetVar(element));
  }
  if (visit_out != nullptr) {
    *visit_out = ExpandList(out);
  }
  return true;
}

bool Visitor::Visit(const BaseRef &any, VectorRef *const values_ref, BaseRef *const visit_out) const {
  if (utils::isa<Seq>(any)) {
    return Visit(utils::cast<Seq>(any), values_ref, visit_out);
  } else if (utils::isa<AnfNodePtr>(any)) {
    auto nodeptr = utils::cast<AnfNodePtr>(any);
    AnfNodePtr output;
    AnfNodePtr *p_output = &output;
    if (visit_out == nullptr) {
      p_output = nullptr;
    }
    Visit(nodeptr, values_ref, p_output);
    if (visit_out != nullptr) {
      *visit_out = output;
    }
    return true;
  }
  MS_LOG(DEBUG) << "VisitError, not support type to Visit: " + any.ToString();
  return false;
}

void Visitor::Visit(const AnfNodePtr &node, VectorRef *const values_ref, AnfNodePtr *output) const {
  if (node->isa<CNode>()) {
    Visit(node->cast<CNodePtr>(), values_ref, output);
    return;
  }

  if (node->isa<ValueNode>()) {
    Visit(node->cast<ValueNodePtr>(), values_ref, output);
    return;
  }

  if (output != nullptr) {
    *output = node;
  }
}

void Visitor::Visit(const CNodePtr &cnode, VectorRef *const values_ref, AnfNodePtr *output) const {
  // if output is nullptr, it's not required to make the new CNode node.
  if (output == nullptr) {
    for (auto &inp : cnode->inputs()) {
      auto var = GetVar(inp);
      values_ref->push_back(var);
    }
    if (cnode->func_graph() != nullptr) {
      values_ref->push_back(GetVar(cnode->func_graph()));
    } else {
      values_ref->push_back(GetVar(cnode->func_graph_as_var()));
    }
    return;
  }

  std::vector<AnfNodePtr> new_inputs;
  std::vector<BaseRef> after_cnode_fn;
  std::shared_ptr<VectorRef> out;
  for (auto &input : cnode->inputs()) {
    after_cnode_fn.push_back(input);
    values_ref->push_back(GetVar(input));
  }
  if (CheckIfNeedExpand(after_cnode_fn)) {
    out = ExpandList(after_cnode_fn);
  }

  std::vector<BaseRef> &outs = after_cnode_fn;
  if (out != nullptr) {
    outs = out->elements();
  }

  for (auto &any_item : outs) {
    if (!utils::isa<AnfNodePtr>(any_item)) {
      MS_LOG(EXCEPTION) << "VisitError, fn not return the same type AnfNodePtr";
    }
    new_inputs.push_back(utils::cast<AnfNodePtr>(any_item));
  }

  BaseRef any_fg;
  AnfNodePtr new_cnode = nullptr;
  if (cnode->func_graph() != nullptr) {
    any_fg = cnode->func_graph();
    values_ref->push_back(GetVar(any_fg));
    if (!utils::isa<FuncGraphPtr>(any_fg)) {
      MS_LOG(EXCEPTION) << "VisitError, fn not return the same type FuncGraphPtr";
    }
    new_cnode = std::make_shared<CNode>(new_inputs, utils::cast<FuncGraphPtr>(any_fg));
  } else {
    any_fg = cnode->func_graph_as_var();
    values_ref->push_back(GetVar(any_fg));
    if (utils::isa<VarPtr>(any_fg)) {
      new_cnode = std::make_shared<CNode>(new_inputs, utils::cast<VarPtr>(any_fg));
    } else if (utils::isa<FuncGraphPtr>(any_fg)) {
      new_cnode = std::make_shared<CNode>(new_inputs, utils::cast<FuncGraphPtr>(any_fg));
    } else {
      MS_LOG(EXCEPTION) << "VisitError, fn not return VarPtr or FuncGraphPtr";
    }
  }
  new_cnode->set_abstract(cnode->abstract());
  *output = new_cnode;
}

void Visitor::Visit(const ValueNodePtr &vnode, VectorRef *const values_ref, AnfNodePtr *output) const {
  values_ref->push_back(GetVar(vnode->value()));
  const BaseRef &value = utils::cast<ValuePtr>(vnode->value());
  if (utils::isa<ValuePtr>(value)) {
    if (output != nullptr) {
      auto ct = NewValueNode(utils::cast<ValuePtr>(value));
      ct->set_abstract(vnode->abstract());
      *output = ct;
    }
    return;
  }
  MS_LOG(EXCEPTION) << "Visit result is not ValuePtr.";
}
}  // namespace mindspore
