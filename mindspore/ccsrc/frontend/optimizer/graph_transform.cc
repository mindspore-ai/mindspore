/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "frontend/optimizer/graph_transform.h"
#include <vector>
#include <algorithm>
#include <string>
#include "ir/graph_utils.h"

namespace mindspore {
/* namespace to support opt */
namespace opt {
// check cnode input values, whether it is tuple input
bool CNodeHasTupleInput(const CNodePtr &cnode) {
  auto &inputs = cnode->inputs();
  for (size_t i = 1; i < inputs.size(); i++) {
    if (IsValueNode<FuncGraph>(inputs[i])) {
      continue;
    }
    if (IsValueNode<Primitive>(inputs[i])) {
      // unexpected high order primitvie  as cnode input when transform graph
      MS_LOG(WARNING) << "CheckTupleInput, got unexpected primitive as input" << cnode->DebugString();
      return false;
    }
    auto abs = inputs[i]->abstract();
    if (abs == nullptr) {
      MS_LOG(WARNING) << "CheckTupleInput, got abstract nullptr for node:" << cnode->DebugString();
      return false;
    }
    if (abs->isa<abstract::AbstractTuple>()) {
      return true;
    }
  }
  return false;
}

bool FuncGraphHasTupleInput(const FuncGraphPtr &fg) {
  auto &params = fg->parameters();
  for (auto &param : params) {
    if (param->abstract()->isa<abstract::AbstractTuple>()) {
      return true;
    }
  }
  return false;
}

std::vector<AnfNodePtr> TransformTupleArgument(const FuncGraphPtr &fg, const AnfNodePtr &node,
                                               const abstract::AbstractTuplePtr &abs) {
  auto &elements = abs->elements();
  std::vector<AnfNodePtr> tuple_node_expanded;
  for (size_t i = 0; i < elements.size(); i++) {
    auto elem_node = fg->NewCNode({NewValueNode(prim::kPrimTupleGetItem), node, NewValueNode(SizeToLong(i))});
    elem_node->set_abstract(elements[i]);
    if (elements[i]->isa<abstract::AbstractTuple>()) {
      auto nodes = TransformTupleArgument(fg, elem_node, elements[i]->cast<abstract::AbstractTuplePtr>());
      tuple_node_expanded.insert(tuple_node_expanded.end(), nodes.begin(), nodes.end());
    } else {
      tuple_node_expanded.push_back(elem_node);
    }
  }
  return tuple_node_expanded;
}

AnfNodePtr TransformCallGraph(const FuncGraphPtr &trans_fg, const CNodePtr &cnode) {
  auto &cinputs = cnode->inputs();
  auto fg = cnode->func_graph();
  std::vector<AnfNodePtr> inputs;
  inputs.push_back(NewValueNode(trans_fg));
  for (size_t i = 1; i < cinputs.size(); i++) {
    auto abs = cinputs[i]->abstract();
    if (abs == nullptr) {
      MS_LOG(EXCEPTION) << "TransformCallGraph:Node abstract should not be nullptr" << cinputs[i]->DebugString();
    }
    if (abs->isa<abstract::AbstractTuple>()) {
      auto nodes = TransformTupleArgument(fg, cinputs[i], abs->cast<abstract::AbstractTuplePtr>());
      inputs.insert(inputs.end(), nodes.begin(), nodes.end());
    } else {
      inputs.push_back(cinputs[i]);
    }
  }
  auto new_node = fg->NewCNode(inputs);
  new_node->set_abstract(cnode->abstract());
  return new_node;
}

AnfNodePtr TransformPartial(const FuncGraphPtr &trans_fg, const CNodePtr &cnode) {
  auto &cinputs = cnode->inputs();
  auto fg = cnode->func_graph();
  std::vector<AnfNodePtr> inputs;
  inputs.push_back(NewValueNode(prim::kPrimPartial));
  inputs.push_back(NewValueNode(trans_fg));
  for (size_t i = 2; i < cinputs.size(); i++) {
    auto abs = cinputs[i]->abstract();
    if (abs == nullptr) {
      MS_LOG(EXCEPTION) << "TransformPartial:Node abstract should not be nullptr" << cinputs[i]->DebugString();
    }
    if (abs->isa<abstract::AbstractTuple>()) {
      auto nodes = TransformTupleArgument(fg, cinputs[i], abs->cast<abstract::AbstractTuplePtr>());
      inputs.insert(inputs.end(), nodes.begin(), nodes.end());
    } else {
      inputs.push_back(cinputs[i]);
    }
  }
  auto new_node = fg->NewCNode(inputs);
  new_node->set_abstract(cnode->abstract());
  return new_node;
}

AnfNodePtr TransformSwitchCall(const AnfNodePtr &swtich_node, const CNodePtr &cnode) {
  auto &cinputs = cnode->inputs();
  auto fg = cnode->func_graph();
  std::vector<AnfNodePtr> inputs;
  inputs.push_back(swtich_node);
  for (size_t i = 1; i < cinputs.size(); i++) {
    auto abs = cinputs[i]->abstract();
    if (abs == nullptr) {
      MS_LOG(EXCEPTION) << "TransformSwitchCall:Node abstract should not be nullptr" << cinputs[i]->DebugString();
    }
    if (abs->isa<abstract::AbstractTuple>()) {
      auto nodes = TransformTupleArgument(fg, cinputs[i], abs->cast<abstract::AbstractTuplePtr>());
      inputs.insert(inputs.end(), nodes.begin(), nodes.end());
    } else {
      inputs.push_back(cinputs[i]);
    }
  }
  auto new_node = fg->NewCNode(inputs);
  new_node->set_abstract(cnode->abstract());
  return new_node;
}
}  // namespace opt
}  // namespace mindspore
