/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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
#include "frontend/optimizer/irpass/view_inplace_utils.h"

#include "frontend/optimizer/irpass.h"
#include "include/frontend/optimizer/optimizer.h"
#include "ir/graph_utils.h"

namespace mindspore {
namespace opt {
namespace irpass {

namespace {
AnfNodePtr GetValidFuncCallNode(const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  if (IsPrimitiveCNode(cnode->input(kIndex0), prim::kPrimSwitch)) {
    auto switch_cnode = cnode->input(kIndex0)->cast<CNodePtr>();
    auto true_fg = GetValueNode<FuncGraphPtr>(switch_cnode->input(kIndex2));
    auto false_fg = GetValueNode<FuncGraphPtr>(switch_cnode->input(kIndex3));
    auto true_index = IsFuncOutputSameWithParamNode(true_fg);
    if (true_index == -1) {
      return nullptr;
    }
    auto false_index = IsFuncOutputSameWithParamNode(false_fg);
    if (true_index != false_index) {
      return nullptr;
    }
    return cnode->input(true_index + 1);
  } else if (auto fg = GetValueNode<FuncGraphPtr>(cnode->input(kIndex0)); fg != nullptr) {
    auto index = IsFuncOutputSameWithParamNode(fg);
    if (index == -1) {
      return nullptr;
    }
    return cnode->input(index + 1);
  }
  return nullptr;
}
}  // namespace

constexpr auto kOutputSameWithParamIndex = "output_same_with_param_index";
constexpr auto kIsCheckOutputSameWithParamIndex = "is_check_output_same_with_param_index";

bool IsViewOutput(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto abs = node->abstract();
  if (abs != nullptr && abs->isa<abstract::AbstractRefTensor>()) {
    const auto ref = abs->cast<abstract::AbstractRefPtr>();
    if (ref->is_view_output()) {
      return true;
    }
  }
  return false;
}

bool IsViewNode(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<CNode>()) {
    return false;
  }
  auto prim = GetValueNode<PrimitivePtr>(node->cast<CNodePtr>()->input(0));
  return prim != nullptr && prim->graph_view_prim();
}

bool IsInplaceNode(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<CNode>()) {
    return false;
  }
  auto prim = GetValueNode<PrimitivePtr>(node->cast<CNodePtr>()->input(0));
  return prim != nullptr && prim->inplace_prim();
}

std::pair<CNodePtr, bool> IsCreatedByViewOp(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (IsViewNode(node)) {
    auto cnode = node->cast<CNodePtr>();
    return {cnode, true};
  }
  const auto &abs = node->abstract();
  if (abs != nullptr && abs->isa<abstract::AbstractRefTensor>()) {
    auto ref = abs->cast<abstract::AbstractRefPtr>();
    if (ref->is_view_output()) {
      auto view_op = abs->user_data<CNode>(kOriginalViewOp);
      if (view_op != nullptr) {
        return {view_op, true};
      }
    }
  }
  return {nullptr, IsViewOutput(node)};
}

bool IsVirtualViewCNode(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  return cnode != nullptr && cnode->HasAttr(kIsVirtualViewOp);
}

AnfNodePtr CheckUMonad(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!HasAbstractUMonad(node)) {
    MS_LOG(EXCEPTION) << "Need to be umonad, but got: " << node->DebugString();
  }
  return node;
}

std::string GetRefKey(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  const auto abs = node->abstract();
  if (abs == nullptr || !abs->isa<abstract::AbstractRefTensor>()) {
    return "";
  }
  auto abs_ref = abs->cast<abstract::AbstractRefPtr>();
  auto ref_key_value = abs_ref->ref_key_value()->cast<StringImmPtr>();
  return ref_key_value == nullptr ? "" : ref_key_value->value();
}

void ReplaceInplaceNodeForCNode(const CNodePtr &cnode, const std::unordered_map<AnfNodePtr, AnfNodePtr> &inplace_input,
                                const FuncGraphManagerPtr &manager, const FuncGraphPtr &func_graph,
                                bool need_ignore_fv) {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(manager);
  auto find_replaced_node = [&inplace_input](const AnfNodePtr &node) -> AnfNodePtr {
    auto it = inplace_input.find(node);
    if (it == inplace_input.end()) {
      return nullptr;
    }
    // Find the final inplaced cnode to replace
    // For example:
    // %1 = Inplace(%0)
    // %2 = Inplace(%1)
    // %3 = Depend(%0, U) ==> %3 = Depend(%2, U)
    AnfNodePtr replaced_node = it->second;
    it = inplace_input.find(replaced_node);
    while (it != inplace_input.end()) {
      replaced_node = it->second;
      it = inplace_input.find(replaced_node);
    }
    return replaced_node;
  };

  // Replace cnode inputs from inplace input to inplace output
  for (size_t i = 1; i < cnode->size(); ++i) {
    auto original_input = cnode->input(i);
    if (need_ignore_fv && original_input->func_graph() != func_graph) {
      continue;
    }
    auto replaced_node = find_replaced_node(original_input);
    if (replaced_node == nullptr) {
      continue;
    }
    MS_LOG(INFO) << "Replace cnode : " << cnode->DebugString() << " input from: " << original_input->DebugString()
                 << " to: " << replaced_node->DebugString() << " for inplace ops replacement.";
    manager->SetEdge(cnode, i, replaced_node);
  }
}

std::vector<bool> GetInplaceChangedParamIndex(const FuncGraphPtr &fg) {
  // CallNode: {fg, param1, param2, param3}
  // Return whether each param is changed by inplace op in this funcgraph
  std::unordered_map<std::string, size_t> params_map;
  const auto &params = fg->parameters();
  std::vector<bool> inplace_modified_param(params.size(), false);
  for (size_t i = 0; i < params.size(); ++i) {
    auto ref_key = GetRefKey(params[i]);
    if (!ref_key.empty()) {
      params_map[ref_key] = i;
    }
  }
  for (auto node : TopoSort(fg->get_return())) {
    if (!IsCNode(node)) {
      continue;
    }
    if (IsInplaceNode(node)) {
      auto iter = params_map.find(GetRefKey(node));
      if (iter != params_map.end()) {
        inplace_modified_param[iter->second] = true;
      }
    }
    // Sub func and view_output of param changed need to be checked later
  }
  return inplace_modified_param;
}

int IsFuncOutputSameWithParamNode(const FuncGraphPtr &fg) {
  // Is being checked, return directly to avoid loop
  if (fg == nullptr || fg->has_attr(kIsCheckOutputSameWithParamIndex)) {
    return -1;
  }
  if (fg->has_attr(kOutputSameWithParamIndex)) {
    return GetValue<int>(fg->get_attr(kOutputSameWithParamIndex));
  }
  fg->set_flag(kIsCheckOutputSameWithParamIndex, true);
  const auto &params = fg->parameters();
  // Scene1 [Unsupported]:
  // If output of func is maketuple
  // %1 = func(param1, param2) [return maketuple(param1, param2)]
  // %2 = TupleGetItem(%1, 0) ==> %2 equal to param1
  // Scene2 [This func supported]:
  // If output of func is normal tensor, same as param1
  // %1 = func(param1, param2) [return param1]
  // %2 = Op(param1, param2)
  // ==> Same as: %2 = Op(%1, param2)
  auto current_node = fg->output();
  MS_EXCEPTION_IF_NULL(current_node);
  while (!current_node->isa<Parameter>()) {
    if (!IsCNode(current_node)) {
      break;
    }
    auto cnode = current_node->cast<CNodePtr>();
    if (IsPrimitiveCNode(cnode, prim::kPrimDepend)) {
      current_node = cnode->input(kIndex1);
      continue;
    }

    if (IsInplaceNode(current_node)) {
      const auto &prim = GetCNodePrimitive(cnode);
      MS_EXCEPTION_IF_NULL(prim);
      const auto &indexes = prim->inplace_input_indexes();
      if (indexes.size() != 1) {
        break;
      }
      current_node = cnode->input(indexes[0] + 1);
      continue;
    }

    auto poss_node = GetValidFuncCallNode(cnode);
    if (poss_node == nullptr) {
      break;
    }
    current_node = poss_node;
  }
  auto it = std::find(params.begin(), params.end(), current_node);
  auto index = -1;
  if (it != params.end()) {
    index = static_cast<int>(std::distance(params.begin(), it));
  }
  fg->erase_flag(kIsCheckOutputSameWithParamIndex);
  fg->set_attr(kOutputSameWithParamIndex, MakeValue(index));
  return index;
}

}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
