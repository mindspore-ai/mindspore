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

#include "pre_activate/common/visit.h"

#include <vector>
#include <memory>
#include <algorithm>
#include <utility>

#include "pre_activate/common/pattern_engine.h"
#include "utils/any.h"
#include "ir/anf.h"
#include "ir/func_graph.h"
#include "utils/log_adapter.h"

/* namespace to support utils definition */
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

bool DefaultVisitor::Visit(const VectorRef &v_any, BaseRef *const visit_out) const {
  std::vector<BaseRef> out;
  (void)std::transform(v_any.begin(), v_any.end(), std::back_inserter(out),
                       [this](const BaseRef &item) { return fn_(item); });
  if (visit_out != nullptr) {
    *visit_out = ExpandList(out);
  }
  return true;
}

bool DefaultVisitor::Visit(const BaseRef &any, BaseRef *const visit_out) const {
  if (utils::isa<Seq>(any)) {
    return Visit(utils::cast<Seq>(any), visit_out);
  } else if (utils::isa<AnfNodePtr>(any)) {
    auto nodeptr = utils::cast<AnfNodePtr>(any);
    AnfNodePtr output;
    AnfNodePtr *p_output = &output;
    if (visit_out == nullptr) {
      p_output = nullptr;
    }
    Visit(nodeptr, fn_, p_output);
    if (visit_out != nullptr) {
      *visit_out = output;
    }
    return true;
  }
  MS_LOG(DEBUG) << "VisitError, not support type to Visit: " + any.ToString();
  return false;
}

void DefaultVisitor::Visit(const AnfNodePtr &node, const VisitFn &fn, AnfNodePtr *output) const {
  if (node->isa<CNode>()) {
    Visit(node->cast<CNodePtr>(), fn, output);
    return;
  }

  if (node->isa<ValueNode>()) {
    Visit(node->cast<ValueNodePtr>(), fn, output);
    return;
  }

  if (output != nullptr) {
    *output = node;
  }
}

void DefaultVisitor::Visit(const CNodePtr &cnode, const VisitFn &fn, AnfNodePtr *output) const {
  // if output is nullptr, it's not required to make the new CNode node.
  if (output == nullptr) {
    for (auto &inp : cnode->inputs()) {
      (void)fn(inp);
    }

    if (cnode->func_graph() != nullptr) {
      (void)fn(cnode->func_graph());
    } else {
      (void)fn(cnode->func_graph_as_var());
    }
    return;
  }

  std::vector<AnfNodePtr> new_inputs;
  std::vector<BaseRef> after_cnode_fn;
  std::shared_ptr<VectorRef> out;
  (void)std::transform(cnode->inputs().begin(), cnode->inputs().end(), std::back_inserter(after_cnode_fn), fn);
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
    any_fg = fn(cnode->func_graph());
    if (!utils::isa<FuncGraphPtr>(any_fg)) {
      MS_LOG(EXCEPTION) << "VisitError, fn not return the same type FuncGraphPtr";
    }
    new_cnode = std::make_shared<CNode>(new_inputs, utils::cast<FuncGraphPtr>(any_fg));
  } else {
    any_fg = fn(cnode->func_graph_as_var());
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

void DefaultVisitor::Visit(const ValueNodePtr &vnode, const VisitFn &fn, AnfNodePtr *output) const {
  const BaseRef &value = utils::cast<ValuePtr>(fn(vnode->value()));
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
