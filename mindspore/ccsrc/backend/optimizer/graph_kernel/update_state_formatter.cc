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
#include "backend/optimizer/graph_kernel/update_state_formatter.h"

#include <vector>
#include <set>
#include <memory>
#include <utility>
#include <algorithm>
#include "backend/session/anf_runtime_algorithm.h"
#include "backend/kernel_compiler/common_utils.h"
#include "backend/optimizer/graph_kernel/graph_kernel_helper.h"
#include "backend/optimizer/graph_kernel/eliminate_redundant_output.h"

namespace mindspore {
namespace opt {
AnfNodePtrList GetUpdateStateList(const FuncGraphPtr &func_graph) {
  auto todos = TopoSort(func_graph->get_return());
  AnfNodePtrList result;
  std::copy_if(todos.begin(), todos.end(), std::back_inserter(result),
               [](const AnfNodePtr &node) { return IsPrimitiveCNode(node, prim::kPrimUpdateState); });
  return result;
}

AnfNodePtrList SpreadTuples(const AnfNodePtrList &nodes, size_t begin_index) {
  AnfNodePtrList result;
  for (size_t i = begin_index; i < nodes.size(); i++) {
    if (IsPrimitiveCNode(nodes[i], prim::kPrimMakeTuple)) {
      auto mt = nodes[i]->cast<CNodePtr>();
      // recursively spread all inner tuples.
      auto mt_inputs = SpreadTuples(mt->inputs(), 1);
      result.insert(result.end(), mt_inputs.begin(), mt_inputs.end());
    } else {
      result.push_back(nodes[i]);
    }
  }
  return result;
}

bool SpreadUpdateState::Run(const FuncGraphPtr &func_graph) {
  auto todos = GetUpdateStateList(func_graph);
  bool changed = false;
  for (auto node : todos) {
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    if (cnode->size() <= kUpdateStateRealInput) continue;
    auto inputs = SpreadTuples(cnode->inputs(), kUpdateStateRealInput);
    if (inputs.size() + 2 != cnode->size() || inputs[0] != cnode->input(2)) {
      AnfNodePtrList node_inputs = {cnode->input(0), cnode->input(1)};
      node_inputs.insert(node_inputs.end(), inputs.begin(), inputs.end());
      cnode->set_inputs(node_inputs);
      changed = true;
    }
  }

  if (changed) {
    auto mng = func_graph->manager();
    MS_EXCEPTION_IF_NULL(mng);
    mng->RemoveRoots();
    mng->KeepRoots({func_graph});
  }
  return changed;
}

bool ShrinkUpdateState::Run(const FuncGraphPtr &func_graph) {
  auto todos = GetUpdateStateList(func_graph);
  bool changed = false;
  for (auto node : todos) {
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    if (cnode->size() <= kUpdateStateRealInput) continue;
    AnfNodePtrList mt_inputs = SpreadTuples(cnode->inputs(), kUpdateStateRealInput);
    AbstractBasePtrList abs_list;
    std::transform(mt_inputs.begin(), mt_inputs.end(), std::back_inserter(abs_list),
                   [](const AnfNodePtr &inp) { return inp->abstract(); });
    mt_inputs.insert(mt_inputs.begin(), NewValueNode(prim::kPrimMakeTuple));
    auto mt_node = func_graph->NewCNode(mt_inputs);
    mt_node->set_abstract(std::make_shared<abstract::AbstractTuple>(abs_list));
    mt_node->set_kernel_info(std::make_shared<device::KernelInfo>());

    AnfNodePtrList inputs = {cnode->input(0), cnode->input(1), mt_node};
    cnode->set_inputs(inputs);
    changed = true;
  }

  if (changed) {
    auto mng = func_graph->manager();
    MS_EXCEPTION_IF_NULL(mng);
    mng->RemoveRoots();
    mng->KeepRoots({func_graph});
  }
  return changed;
}

bool ExtendOutputForUpdateState::Run(const FuncGraphPtr &func_graph) {
  auto todos = FindGraphKernelsWithMultiOutput(func_graph);
  auto mng = func_graph->manager();
  MS_EXCEPTION_IF_NULL(mng);
  bool changed = false;
  for (const auto &node : todos) {
    GetGraphKernelGetitemList(mng, node, &getitems_, false);
    if (getitems_.empty()) continue;
    FindIndexesToUpdateState(mng);
    if (indexes_.empty()) continue;
    auto sub_func_graph = AnfAlgo::GetCNodeFuncGraphPtr(node);
    FilterIndexes(sub_func_graph);
    if (indexes_.empty()) continue;
    for (auto idx : indexes_) {
      changed = ProcessIndex(func_graph, sub_func_graph, idx) || changed;
    }
  }
  if (changed) {
    std::make_shared<SpreadUpdateState>()->Run(func_graph);
    std::make_shared<EliminateHangingOutput>()->Run(func_graph);
  }
  return changed;
}

void ExtendOutputForUpdateState::FindIndexesToUpdateState(const FuncGraphManagerPtr &mng) {
  indexes_.clear();
  external_user_type_.clear();
  external_user_type_.resize(getitems_.size(), ExternalUserType::kNormalOp);
  for (size_t i = 0; i < getitems_.size(); ++i) {
    const AnfNodePtr &getitem = getitems_[i];
    if (getitem == nullptr) continue;

    const auto &getitem_user = mng->node_users()[getitem];
    auto IsUpdateState = [](const std::pair<AnfNodePtr, int> &user) {
      return IsPrimitiveCNode(user.first, prim::kPrimUpdateState);
    };
    if (std::all_of(getitem_user.begin(), getitem_user.end(), IsUpdateState)) {
      external_user_type_[i] = ExternalUserType::kUpdateState;
      indexes_.push_back(i);
    } else if (std::any_of(getitem_user.begin(), getitem_user.end(), IsUpdateState)) {
      external_user_type_[i] = ExternalUserType::kMix;
      indexes_.push_back(i);
    }
  }
}

void ExtendOutputForUpdateState::FilterIndexes(const FuncGraphPtr &func_graph) {
  auto output_node = func_graph->output()->cast<CNodePtr>();
  // do not process the side-effect nodes.
  indexes_.erase(std::remove_if(indexes_.begin(), indexes_.end(),
                                [&output_node](size_t i) { return IsSideEffectNode(output_node->input(i + 1)); }),
                 indexes_.end());
}

std::vector<size_t> ExtendOutputForUpdateState::FindAllOutputs(const FuncGraphPtr &func_graph, size_t index) {
  auto output_node = func_graph->output()->cast<CNodePtr>();
  auto index_node = output_node->input(index);
  std::vector<size_t> group;

  // if the `out_node` is a user (direct or indirect) of the `index_node`, returns true
  auto DependsOnIndexNode = [&index_node](const AnfNodePtr &out_node) -> bool {
    bool result = false;
    auto IncludeFunc = [&result, &index_node](const AnfNodePtr &node) {
      if (node == index_node) {
        result = true;
        return EXCLUDE;
      }
      return result ? EXCLUDE : FOLLOW;
    };
    static_cast<void>(DeepLinkedGraphSearch(out_node, IncludeFunc));
    return result;
  };

  for (size_t i = 1; i < output_node->size(); i++) {
    auto out = output_node->input(i);
    // only process the nodes that depend on index_node.
    if (!DependsOnIndexNode(out)) continue;

    // 1. always extend to the side-effect nodes
    // 2. if the external users are only UpdateState, the related output will be eliminated,
    //    so only the getitem with realkernel user can be extended to.
    if (IsSideEffectNode(out) ||
        (getitems_[i - 1] != nullptr && external_user_type_[i - 1] != ExternalUserType::kUpdateState)) {
      group.push_back(i - 1);
    }
  }
  return group;
}

bool ExtendOutputForUpdateState::ProcessIndex(const FuncGraphPtr &func_graph, const FuncGraphPtr &sub_func_graph,
                                              size_t index) {
  auto group = FindAllOutputs(sub_func_graph, index + 1);
  AnfNodePtr new_node = nullptr;
  if (group.size() == 1 && group[0] == index) {
    return false;
  }
  if (group.empty()) {
    // the output is not side-effect node, but it hasn't realkernel user.
    // replace the getitem with a value node that is unrelated to the original node.
    // and this value node will be removed at the later pass.
    MS_LOG(INFO) << "The " << getitems_[index]->fullname_with_scope() << " only has UpdateState user.";
    new_node = NewValueNode(kUMonad)->cast<AnfNodePtr>();
    new_node->set_abstract(kUMonad->ToAbstract());
  } else {
    // Create MakeTuple, even though the group size is 1, the following pass will spread the MakeTuple,
    // so it's unnecessary to set abstract for it.
    AnfNodePtrList mt_input = {NewValueNode(prim::kPrimMakeTuple)};
    std::transform(group.begin(), group.end(), std::back_inserter(mt_input),
                   [this](size_t idx) { return getitems_[idx]; });
    new_node = func_graph->NewCNode(mt_input)->cast<AnfNodePtr>();
  }
  auto mng = func_graph->manager();
  MS_EXCEPTION_IF_NULL(mng);
  for (auto user : mng->node_users()[getitems_[index]]) {
    user.first->cast<CNodePtr>()->set_input(user.second, new_node);
  }
  return true;
}

bool MergeOutputForUpdateState::Run(const FuncGraphPtr &func_graph) {
  auto todos = GetUpdateStateList(func_graph);
  bool changed = false;
  for (auto node : todos) {
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    AnfNodePtrList inputs = {cnode->input(0), cnode->input(1)};
    std::set<AnfNodePtr> node_set;
    for (size_t i = 2; i < cnode->size(); ++i) {
      auto input = cnode->input(i);
      if (IsPrimitiveCNode(input, prim::kPrimTupleGetItem)) {
        // only keep one GetItem for that link to the same node.
        auto gt_input = input->cast<CNodePtr>()->input(kRealInputNodeIndexInTupleGetItem);
        if (node_set.insert(gt_input).second) {
          inputs.push_back(input);
        }
      } else if (!HasAbstractUMonad(input)) /*filter the UMonad that was added in "ExtendOutputForUpdateState" */ {
        if (node_set.insert(input).second) {
          inputs.push_back(input);
        }
      }
    }

    if (inputs.size() < cnode->size()) {
      cnode->set_inputs(inputs);
      changed = true;
    }
  }
  if (changed) {
    auto mng = func_graph->manager();
    MS_EXCEPTION_IF_NULL(mng);
    mng->RemoveRoots();
    mng->KeepRoots({func_graph});
  }
  return changed;
}
}  // namespace opt
}  // namespace mindspore
