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
#include "backend/optimizer/graph_kernel/optimize_assign.h"

#include <algorithm>
#include <vector>
#include <string>
#include <map>

#include "base/core_ops.h"
#include "backend/optimizer/common/helper.h"
#include "backend/session/anf_runtime_algorithm.h"
#include "backend/optimizer/graph_kernel/graph_kernel_helper.h"

namespace mindspore {
namespace opt {
namespace {
kernel::KernelBuildInfoPtr BuildSelectKernelBuildInfo(const std::vector<std::string> &inputs_format,
                                                      const std::vector<TypeId> &inputs_type,
                                                      const std::vector<std::string> &output_formats,
                                                      const std::vector<TypeId> &output_types, const CNodePtr &cnode) {
  kernel::KernelBuildInfo::KernelBuildInfoBuilder graph_info_builder;
  graph_info_builder.SetInputsFormat(inputs_format);
  graph_info_builder.SetInputsDeviceType(inputs_type);
  graph_info_builder.SetOutputsFormat(output_formats);
  graph_info_builder.SetOutputsDeviceType(output_types);
  graph_info_builder.SetProcessor(AnfAlgo::GetProcessor(cnode));
  graph_info_builder.SetKernelType(KernelType::AKG_KERNEL);
  graph_info_builder.SetFusionType(kernel::FusionType::OPAQUE);
  return graph_info_builder.Build();
}

/**
 * If an Assign's source node was outputted with this Assign, the src-node should be removed from output list,
 * external users can use the dest-node under the premise of correct execution order.
 * This function find out the [index of src node in output list] and [external dest-node].
 * Note:
 * 1. Assign is always in output list. (links to external Depend node)
 * 2. Assign's dest-node should be a Parameter.
 */
std::map<size_t, AnfNodePtr> FindAssignAndOutputVal(const CNodePtr &fg_cnode) {
  // Check output includes assign
  auto func_graph = AnfAlgo::GetCNodeFuncGraphPtr(fg_cnode);
  MS_EXCEPTION_IF_NULL(func_graph);
  auto out_cnode = func_graph->output()->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(out_cnode);
  std::map<size_t, AnfNodePtr> output_replace_map;

  if (!IsPrimitiveCNode(out_cnode, prim::kPrimMakeTuple)) {
    return output_replace_map;
  }

  // Trans parameter to the real input
  auto ParameterToInput = [&func_graph, &fg_cnode](const AnfNodePtr &p) {
    const auto &params = func_graph->parameters();
    size_t i = std::find(params.begin(), params.end(), p) - params.begin();
    return i == params.size() ? nullptr : fg_cnode->input(i + 1);
  };

  const auto &inputs = out_cnode->inputs();
  for (const auto &out : inputs) {
    if (IsPrimitiveCNode(out, prim::kPrimAssign)) {
      auto assign_val = out->cast<CNodePtr>()->input(2);
      auto assign_parameter = out->cast<CNodePtr>()->input(1);
      auto iter = std::find(inputs.begin() + 1, inputs.end(), assign_val);
      if (iter != inputs.end()) {
        size_t assign_val_index = iter - inputs.begin();
        auto assign_to = ParameterToInput(assign_parameter);
        if (assign_to != nullptr && assign_val_index > 0) {
          output_replace_map[assign_val_index - 1] = assign_to;
        }
      }
    }
  }
  return output_replace_map;
}

bool IsPriorDependSatisfied(const AnfNodePtr &gk_node, const AnfNodePtr &par_user_node, const AnfNodePtr &exclude,
                            const FuncGraphManagerPtr &mng) {
  auto IncludeUser = [exclude](const AnfNodePtr &node) -> IncludeType {
    if (exclude == node) {
      return EXCLUDE;
    }
    return FOLLOW;
  };

  auto users = DeepUsersSearch(gk_node, IncludeUser, mng);
  return std::any_of(users.begin(), users.end(), [par_user_node](AnfNodePtr user) { return user == par_user_node; });
}

void AddControlDepend(const FuncGraphPtr &func_graph, const AnfNodePtr &gk_node, const AnfNodePtr &par_user_node,
                      const AnfNodePtr &par, const FuncGraphManagerPtr &mng) {
  AnfNodePtrList cd_inputs = {NewValueNode(prim::kPrimControlDepend), gk_node, par_user_node};
  auto cd_node = func_graph->NewCNode(cd_inputs);
  func_graph->AddNode(cd_node);
  AnfNodePtrList dep_inputs = {NewValueNode(prim::kPrimDepend), par_user_node, cd_node};
  auto dp_node = func_graph->NewCNode(dep_inputs);
  func_graph->AddNode(dp_node);
  const auto &post_users = mng->node_users()[par_user_node];
  for (auto user : post_users) {
    (user.first)->cast<CNodePtr>()->set_input(user.second, dp_node);
  }
}

int64_t GetitemIndex(const AnfNodePtr &getitem) {
  auto index_node = getitem->cast<CNodePtr>()->input(kInputNodeOutputIndexInTupleGetItem);
  auto value_ptr = GetValueNode(index_node);
  return GetValue<int64_t>(value_ptr);
}

void UpdateUsersOfGraphKernel(const FuncGraphPtr &func_graph, const AnfNodePtr &cnode, const FuncGraphManagerPtr &mng,
                              const AnfNodePtr &assign_to, int64_t removed_index) {
  auto graph_kernel_fg = AnfAlgo::GetCNodeFuncGraphPtr(cnode);
  MS_EXCEPTION_IF_NULL(graph_kernel_fg);
  // Check the removed output has data_user or not
  bool has_data_user = false;
  AnfNodePtr getitem = nullptr;
  const auto &users = mng->node_users()[cnode];
  for (const auto &user : users) {
    auto index = GetitemIndex(user.first);
    if (index == removed_index) {
      getitem = user.first;
      const auto &assign_val_users = mng->node_users()[user.first];
      has_data_user = has_data_user || HasDataUser(assign_val_users, mng);
      break;
    }
  }
  if (getitem == nullptr) return;

  if (has_data_user) {
    // check and add ControlDepend to ensure the right execution order.
    for (auto getitem_user : mng->node_users()[getitem]) {
      if (IsPriorDependSatisfied(cnode, getitem_user.first, getitem, mng)) {
        continue;
      }
      AddControlDepend(func_graph, cnode, getitem_user.first, assign_to, mng);
      mng->RemoveRoots();
      mng->KeepRoots({func_graph});
    }
  }

  mng->Replace(getitem, assign_to);
}

bool RepalceOutputByParameter(const FuncGraphPtr &func_graph) {
  auto todos = TopoSort(func_graph->get_return());
  MS_EXCEPTION_IF_NULL(func_graph);
  auto mng = func_graph->manager();
  MS_EXCEPTION_IF_NULL(mng);

  bool changed = false;
  for (const auto &n : todos) {
    if (!AnfAlgo::IsGraphKernel(n)) continue;
    auto cnode = n->cast<CNodePtr>();
    auto replaceable_nodes = FindAssignAndOutputVal(cnode);
    if (replaceable_nodes.empty()) continue;
    changed = true;
    for (const auto &iter : replaceable_nodes) {
      UpdateUsersOfGraphKernel(func_graph, cnode, mng, iter.second, iter.first);
    }
  }
  return changed;
}

bool ReplaceAssignByInplaceAssignInGraphkernel(const FuncGraphPtr &func_graph) {
  auto mng = func_graph->manager();
  MS_EXCEPTION_IF_NULL(mng);
  auto todos = TopoSort(func_graph->get_return());
  bool changed = false;
  for (const auto &n : todos) {
    if (!AnfAlgo::CheckPrimitiveType(n, prim::kPrimAssign)) continue;
    changed = true;
    auto cnode = n->cast<CNodePtr>();
    AnfNodePtrList inputs = {NewValueNode(prim::kPrimInplaceAssign->Clone()), cnode->input(1), cnode->input(2),
                             cnode->input(2)};
    auto new_cnode = func_graph->NewCNode(inputs);
    AnfAlgo::SetNodeAttr("fake_output", MakeValue(true), new_cnode);
    new_cnode->set_abstract(inputs.back()->abstract());
    new_cnode->set_kernel_info(std::make_shared<device::KernelInfo>());
    std::vector<std::string> input_formats = AnfAlgo::GetAllInputFormats(cnode);
    std::vector<TypeId> input_types = AnfAlgo::GetAllInputDeviceTypes(cnode);
    input_formats.push_back(input_formats.back());
    input_types.push_back(input_types.back());
    std::vector<std::string> output_formats = {input_formats.back()};
    std::vector<TypeId> output_types = {input_types.back()};
    auto graph_sel_info = BuildSelectKernelBuildInfo(input_formats, input_types, output_formats, output_types, cnode);
    AnfAlgo::SetSelectKernelBuildInfo(graph_sel_info, new_cnode.get());
    mng->Replace(cnode, new_cnode);
  }
  return changed;
}

bool RepalceAssignByInplaceAssign(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  auto mng = func_graph->manager();
  MS_EXCEPTION_IF_NULL(mng);
  auto todos = TopoSort(func_graph->get_return());

  auto changed = false;
  for (const auto &n : todos) {
    if (!AnfAlgo::IsGraphKernel(n)) continue;
    auto graph_kernel_fg = AnfAlgo::GetCNodeFuncGraphPtr(n);
    MS_EXCEPTION_IF_NULL(graph_kernel_fg);
    changed = ReplaceAssignByInplaceAssignInGraphkernel(graph_kernel_fg) || changed;
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
  return RepalceAssignByInplaceAssign(func_graph);
}
}  // namespace opt
}  // namespace mindspore
