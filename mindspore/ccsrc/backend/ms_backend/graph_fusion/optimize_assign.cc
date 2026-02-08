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
#include "backend/ms_backend/graph_fusion/optimize_assign.h"

#include <algorithm>
#include <functional>
#include <vector>
#include <string>
#include <map>
#include <unordered_set>
#include <utility>

#include "primitive/sequence_ops.h"
#include "primitive/nn_optimizer_ops.h"
#include "primitive/framework_ops.h"
#include "include/backend/common/pass_manager/helper.h"
#include "include/backend/common/kernel_graph/anf_runtime_algorithm.h"
#include "include/utils/anfalgo.h"
#include "ir/graph_utils.h"
#include "backend/ms_backend/graph_fusion/graph_kernel_helper.h"
#include "primitive/auto_generate/gen_ops_primitive_a.h"
#include "primitive/auto_generate/gen_ops_primitive_d.h"
#include "primitive/auto_generate/gen_ops_primitive_l.h"
#include "primitive/auto_generate/gen_ops_primitive_m.h"
#include "primitive/auto_generate/gen_ops_primitive_r.h"
#include "primitive/auto_generate/gen_ops_primitive_t.h"
#include "primitive/auto_generate/gen_ops_primitive_u.h"

namespace mindspore::graphkernel {
namespace {
using OutIndexParamPair = std::pair<size_t, AnfNodePtr>;

/// \brief find the output and assign values to be replaced.
/// \return map from [index of src-node in outputs] to pair <index of Assign in outputs, external dest-node>
std::map<size_t, OutIndexParamPair> FindAssignAndOutputVal(const CNodePtr &fg_cnode) {
  // Check output includes assign
  auto func_graph = common::AnfAlgo::GetCNodeFuncGraphPtr(fg_cnode);
  MS_EXCEPTION_IF_NULL(func_graph);
  auto out_tuple = func_graph->output()->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(out_tuple);
  std::map<size_t, OutIndexParamPair> output_replace_map;

  if (!IsPrimitiveCNode(out_tuple, prim::kPrimMakeTuple)) {
    return output_replace_map;
  }

  // Trans parameter to the real input
  auto ParameterToInput = [&func_graph, &fg_cnode](const AnfNodePtr &p) {
    const auto &params = func_graph->parameters();
    size_t i = std::find(params.begin(), params.end(), p) - params.begin();
    return i == params.size() ? nullptr : fg_cnode->input(i + 1);
  };

  const auto &fg_outputs = out_tuple->inputs();
  for (size_t i = 1; i < fg_outputs.size(); ++i) {
    auto out = fg_outputs[i];
    if (IsPrimitiveCNode(out, prim::kPrimAssign)) {
      auto assign_val = out->cast<CNodePtr>()->input(2);
      auto assign_parameter = out->cast<CNodePtr>()->input(1);
      auto iter = std::find(fg_outputs.begin() + 1, fg_outputs.end(), assign_val);
      if (iter != fg_outputs.end()) {
        size_t assign_val_index = static_cast<size_t>(iter - fg_outputs.begin());
        auto assign_to = ParameterToInput(assign_parameter);
        if (assign_to != nullptr && assign_val_index > 0) {
          output_replace_map[assign_val_index - 1] = std::make_pair(i - 1, assign_to);
        }
      }
    }
  }
  return output_replace_map;
}

bool HasPathToParamUser(const AnfNodePtr &gk_node, const AnfNodePtr &param_user, const AnfNodePtr &getitem) {
  auto mng = common::AnfAlgo::GetCNodeFuncGraphPtr(gk_node)->manager();
  MS_EXCEPTION_IF_NULL(mng);
  bool result = false;
  auto IncludeUser = [&result, &gk_node, &getitem](const AnfNodePtr &node) {
    if (node == getitem) {
      return EXCLUDE;
    }
    if (node == gk_node) {
      result = true;
      return EXCLUDE;
    }
    return result ? EXCLUDE : FOLLOW;
  };
  static_cast<void>(DeepLinkedGraphSearch(param_user, IncludeUser));
  return result;
}

std::unordered_set<AnfNodePtr> HasPathToReturn(const FuncGraphPtr &func_graph) {
  std::unordered_set<AnfNodePtr> result;
  std::function<void(const AnfNodePtr &node)> dfs;
  dfs = [&result, &dfs](const AnfNodePtr &node) {
    if (IsPrimitiveCNode(node, prim::kPrimReturn) || IsPrimitiveCNode(node, prim::kPrimDepend)) {
      result.insert(node);
      dfs(node->cast<CNodePtr>()->input(1));
    } else if (IsPrimitiveCNode(node, prim::kPrimMakeTuple)) {
      result.insert(node);
      auto inputs = node->cast<CNodePtr>()->inputs();
      for (size_t i = 1; i < inputs.size(); ++i) {
        dfs(inputs[i]);
      }
    }
  };
  dfs(func_graph->get_return());
  return result;
}

void KeepExecOrder(const FuncGraphPtr &func_graph, const AnfNodePtr &getitem, const CNodeIndexPair &getitem_user,
                   const AnfNodePtr &assign_to_node, const FuncGraphManagerPtr &mng) {
  // Insert update_state_node, need mount a monad node.
  auto u = NewValueNode(kUMonad);
  u->set_abstract(kUMonad->ToAbstract());
  AnfNodePtrList update_state_inputs = {NewValueNode(prim::kPrimUpdateState), u, getitem};
  auto update_state_node = func_graph->NewCNode(update_state_inputs);
  update_state_node->set_abstract(getitem->abstract());
  func_graph->AddNode(update_state_node);

  // Insert load_node
  AnfNodePtrList load_inputs = {NewValueNode(prim::kPrimLoad), assign_to_node, update_state_node};
  auto load_node = func_graph->NewCNode(load_inputs);
  load_node->set_abstract(assign_to_node->abstract());
  func_graph->AddNode(load_node);
  const auto &getitem_user_cnode = getitem_user.first->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(getitem_user_cnode);
  getitem_user_cnode->set_input(getitem_user.second, load_node);
}

int64_t GetitemIndex(const AnfNodePtr &getitem) {
  auto index_node = getitem->cast<CNodePtr>()->input(kInputNodeOutputIndexInTupleGetItem);
  auto value_ptr = GetValueNode(index_node);
  return GetValue<int64_t>(value_ptr);
}

void UpdateUsersOfGraphKernel(const FuncGraphPtr &func_graph, const AnfNodePtr &cnode, const AnfNodePtr &assign_to,
                              int64_t removed_index, int64_t assign_idx,
                              const std::unordered_set<AnfNodePtr> &outputs) {
  auto mng = func_graph->manager();
  MS_EXCEPTION_IF_NULL(mng);
  for (const auto &getitem_iter : mng->node_users()[cnode]) {
    auto getitem = getitem_iter.first;
    if (!IsPrimitiveCNode(getitem, prim::kPrimTupleGetItem)) {
      continue;
    }
    if (GetitemIndex(getitem) != removed_index) {
      continue;
    }
    auto getitem_users = mng->node_users()[getitem];  // get a copy of getitem's users before replacing

    for (const auto &getitem_user_iter : getitem_users) {
      auto getitem_user = getitem_user_iter.first;
      // if `getitem` will be returned, we can't optimize this getitem.
      // because we can't keep exec_order outside the kernel graph.
      if (outputs.find(getitem_user) != outputs.end()) {
        return;
      }
      // 1. Data users may not link directly to its input, they may segregated by Depend node.
      // 2. If the `cnode` has another path to the getitem_user, it's unnecessary to add depend node to
      //    keep exec_order.
      if (HasPathToParamUser(cnode, getitem_user, getitem)) {
        const auto &getitem_user_cnode = getitem_user->cast<CNodePtr>();
        MS_EXCEPTION_IF_NULL(getitem_user_cnode);
        getitem_user_cnode->set_input(getitem_user_iter.second, assign_to);
        continue;
      }
      KeepExecOrder(func_graph, getitem, getitem_user_iter, assign_to, mng);
    }
    // the index of TupleGetItem should be changed from the output index of the replaced node to the assign node
    auto item_idx = opt::CreateValueNodeWithKernelInfo(func_graph, MakeValue(assign_idx));
    auto getitem_cnode = getitem->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(getitem_cnode);
    getitem_cnode->set_input(kInputNodeOutputIndexInTupleGetItem, item_idx);
    break;
  }
}

bool RepalceOutputByParameter(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  auto todos = TopoSort(func_graph->get_return());

  bool changed = false;
  auto outputs = HasPathToReturn(func_graph);
  for (const auto &n : todos) {
    if (!common::AnfAlgo::IsGraphKernel(n)) {
      continue;
    }
    auto cnode = n->cast<CNodePtr>();
    auto replaceable_nodes = FindAssignAndOutputVal(cnode);
    if (replaceable_nodes.empty()) {
      continue;
    }
    changed = true;
    for (const auto &[index, idx_node] : replaceable_nodes) {
      UpdateUsersOfGraphKernel(func_graph, cnode, idx_node.second, static_cast<int64_t>(index),
                               static_cast<int64_t>(idx_node.first), outputs);
    }
  }
  return changed;
}
}  // namespace

bool OptimizeAssign::Run(const FuncGraphPtr &func_graph) {
  auto mng = func_graph->manager();
  if (mng == nullptr) {
    mng = Manage(func_graph, true);
    func_graph->set_manager(mng);
  }
  auto res = RepalceOutputByParameter(func_graph);
  if (res) {
    mng->RemoveRoots();
    mng->KeepRoots({func_graph});
  }
  return res;
}
}  // namespace mindspore::graphkernel
