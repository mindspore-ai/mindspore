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

#include "plugin/ascend/kernel_executor/hyper_offload/hyper_offload.h"
#include <vector>
#include <memory>

#include "utils/compile_config.h"
#include "include/utils/anfalgo.h"
#include "include/backend/common/kernel_graph/anf_runtime_algorithm.h"
#include "primitive/framework_ops.h"
#include "primitive/other_ops.h"
#include "primitive/auto_generate/gen_ops_primitive_c.h"
#include "primitive/auto_generate/gen_ops_primitive_t.h"
#include "plugin/ascend/kernel_executor/kernel_select_ascend.h"
#include "plugin/ascend/kernel_executor/rts/rt_kernel_build.h"
#include "plugin/ascend/graph_optimizer/stream_assign/acl_stream_assign.h"
#include "mindspore/ops/ops_utils/op_constants.h"

namespace mindspore {
namespace device {
namespace ascend {
namespace hyper_offload {
namespace {
using UserInfo = std::pair<KernelWithIndex, std::vector<KernelWithIndex>>;
using UserInfoList = std::vector<UserInfo>;
using AbsoluteDistance = std::pair<KernelWithIndex, size_t>;
using AbsoluteDistanceList = std::vector<AbsoluteDistance>;
using RelativeDistance = std::pair<std::pair<KernelWithIndex, KernelWithIndex>, size_t>;
using RelativeDistanceList = std::vector<RelativeDistance>;
using ReplaceInfo = std::pair<KernelWithIndex, std::vector<KernelWithIndex>>;
using ReplaceInfoList = std::vector<ReplaceInfo>;
using OffloadInfo = std::pair<KernelWithIndex, ReplaceInfoList>;
using OffloadInfoList = std::vector<OffloadInfo>;
using ConditionSwitchInfoPtr = std::shared_ptr<ConditionSwitchInfo>;
constexpr auto kSwitchTrueBranchNum = "TrueBranchNodeNum";
constexpr auto kSwitchFalseBranchNum = "FalseBranchNodeNum";

bool NeedOffloadParameter(const KernelGraphPtr &kernel_graph) {
  return kernel_graph->has_user_data("offload_parameter");
}

bool NeedOffloadActivation(const KernelGraphPtr &kernel_graph) {
  return kernel_graph->has_user_data("offload_activation");
}

bool NeedOffload(const KernelGraphPtr &kernel_graph) {
  static bool enable_hyper_offload = common::GetEnv("MS_DEV_HYPER_OFFLOAD") != "1";
  if (enable_hyper_offload) {
    return false;
  }
  return NeedOffloadParameter(kernel_graph) || NeedOffloadActivation(kernel_graph);
}

bool IsD2HNode(const AnfNodePtr &node) {
  return IsOneOfPrimitiveCNode(node, {prim::kPrimCopyToHost, prim::kPrimCopyToHostExt});
}

bool IsH2DNode(const AnfNodePtr &node) { return IsOneOfPrimitiveCNode(node, {prim::kPrimCopyToDevice}); }

bool IsViewOrInplaceNode(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<CNode>()) {
    return false;
  }
  const auto &prim = GetValueNode<PrimitivePtr>(node->cast<CNodePtr>()->input(0));
  return prim != nullptr && (prim->graph_view_prim() || prim->inplace_prim());
}

CNodePtr BuildToHostNode(const KernelGraphPtr &kernel_graph, const AnfNodePtr &data_node) {
  const std::string &prim_name = prim::kPrimCopyToHost->name();
  auto prim = std::make_shared<Primitive>(prim_name);
  AnfNodePtrList inputs{NewValueNode(prim), data_node};
  auto sync_node = kernel_graph->NewValueNode(MakeValue(false));
  sync_node->set_abstract(std::make_shared<abstract::AbstractScalar>(false));
  (void)inputs.emplace_back(sync_node);
  auto ret_node = kernel_graph->NewCNode(inputs);
  ret_node->set_abstract(data_node->abstract());
  device::ascend::GenerateKernelBuildInfo(ret_node, RT_KERNEL);
  auto kernel_mod_ptr = kernel::RtOpBuild(ret_node);
  MS_EXCEPTION_IF_NULL(kernel_mod_ptr);
  AnfAlgo::SetKernelMod(kernel_mod_ptr, ret_node.get());
  return ret_node;
}

CNodePtr BuildInplaceToHostNode(const KernelGraphPtr &kernel_graph, const AnfNodePtr &data_node,
                                const AnfNodePtr &update_node, const AnfNodePtr &depend_node) {
  const std::string &prim_name = prim::kPrimCopyToHostExt->name();
  auto prim = std::make_shared<Primitive>(prim_name);
  AnfNodePtrList inputs{NewValueNode(prim), data_node, update_node, depend_node};
  auto sync_node = kernel_graph->NewValueNode(MakeValue(false));
  sync_node->set_abstract(std::make_shared<abstract::AbstractScalar>(false));
  (void)inputs.emplace_back(sync_node);
  auto ret_node = kernel_graph->NewCNode(inputs);
  ret_node->set_abstract(data_node->abstract());
  device::ascend::GenerateKernelBuildInfo(ret_node, RT_KERNEL);
  auto kernel_mod_ptr = kernel::RtOpBuild(ret_node);
  MS_EXCEPTION_IF_NULL(kernel_mod_ptr);
  AnfAlgo::SetKernelMod(kernel_mod_ptr, ret_node.get());
  return ret_node;
}

CNodePtr BuildToDeviceNode(const KernelGraphPtr &kernel_graph, const AnfNodePtr &data_node) {
  const std::string &prim_name = prim::kPrimCopyToDevice->name();
  auto prim = std::make_shared<Primitive>(prim_name);
  auto depend_node = kernel_graph->NewValueNode(MakeValue(false));
  depend_node->set_abstract(std::make_shared<abstract::AbstractScalar>(false));
  auto ret_node = kernel_graph->NewCNode({NewValueNode(prim), data_node, depend_node, depend_node});
  ret_node->set_abstract(data_node->abstract());
  device::ascend::GenerateKernelBuildInfo(ret_node, RT_KERNEL);
  auto kernel_mod_ptr = kernel::RtOpBuild(ret_node);
  MS_EXCEPTION_IF_NULL(kernel_mod_ptr);
  AnfAlgo::SetKernelMod(kernel_mod_ptr, ret_node.get());
  return ret_node;
}

bool IsValidOffloadNode(const UserInfo &user_info, bool offload_activation) {
  auto node = user_info.first.first;
  MS_EXCEPTION_IF_NULL(node);
  if (!AnfUtils::IsRealKernel(node)) {
    return false;
  }
  if (IsOneOfPrimitiveCNode(node, {prim::kPrimMoveTo, prim::kPrimTensorMove, prim::kPrimConditionSwitch})) {
    return false;
  }
  if (std::any_of(user_info.second.begin(), user_info.second.end(), [](const auto &e) { return IsD2HNode(e.first); })) {
    return false;
  }
  return true;
}

UserInfoList CollectAllNodeUsers(const CNodePtrList &nodes) {
  UserInfoList user_info_list;
  for (const CNodePtr &node : nodes) {
    for (size_t i = 0; i < node->inputs().size(); ++i) {
      const auto &input = node->input(i);
      if (input->abstract() == nullptr || !input->abstract()->isa<abstract::AbstractTensor>()) {
        continue;
      }
      const auto &cur_kernel_with_index = common::AnfAlgo::VisitKernelWithReturnType(input, 0, false);
      if (!cur_kernel_with_index.first->isa<CNode>()) {
        continue;
      }
      auto iter = std::find_if(user_info_list.begin(), user_info_list.end(),
                               [&cur_kernel_with_index](const auto &e) { return cur_kernel_with_index == e.first; });
      auto cur_user = KernelWithIndex{node, i};
      if (iter == user_info_list.end()) {
        auto new_user_info = UserInfo(cur_kernel_with_index, {cur_user});
        (void)user_info_list.emplace_back(new_user_info);
      } else {
        (iter->second).emplace_back(cur_user);
      }
    }
  }
  return user_info_list;
}

AbsoluteDistanceList GenerateAbsoluteDistanceList(const UserInfo &user_info, const CNodePtrList &execution_order) {
  auto node = user_info.first.first;
  auto node_iter = std::find(execution_order.begin(), execution_order.end(), node);
  MS_EXCEPTION_IF_CHECK_FAIL(node_iter != execution_order.end(), "Failed to find node in execution order.");
  auto node_distance = std::distance(execution_order.begin(), node_iter);
  AbsoluteDistanceList absolute_distance_list{AbsoluteDistance{user_info.first, node_distance}};
  const auto &node_users = user_info.second;
  for (const auto &node_user : node_users) {
    auto user_node = node_user.first;
    auto node_user_iter = std::find(execution_order.begin(), execution_order.end(), user_node);
    if (node_user_iter == execution_order.end()) {
      MS_LOG(EXCEPTION) << "Failed to find in executor for user node: " << user_node->DebugString() << ", node is "
                        << node->DebugString();
    }
    size_t absolute_distance = std::distance(execution_order.begin(), node_user_iter);
    (void)absolute_distance_list.emplace_back(AbsoluteDistance{node_user, absolute_distance});
  }
  std::sort(absolute_distance_list.begin(), absolute_distance_list.end(),
            [](const auto &e1, const auto &e2) { return e1.second < e2.second; });
  return absolute_distance_list;
}

RelativeDistanceList GenerateRelativeDistanceList(const AbsoluteDistanceList &absolute_distance_list) {
  RelativeDistanceList relative_distance_list;
  for (size_t i = 1; i < absolute_distance_list.size(); ++i) {
    std::pair<KernelWithIndex, KernelWithIndex> current_node_pairs{absolute_distance_list[i - 1].first,
                                                                   absolute_distance_list[i].first};
    int current_distance = absolute_distance_list[i].second - absolute_distance_list[i - 1].second;
    (void)relative_distance_list.emplace_back(RelativeDistance{current_node_pairs, IntToSize(current_distance)});
  }
  return relative_distance_list;
}

ReplaceInfoList GenerateReplaceInfoList(const RelativeDistanceList &relative_distance_list) {
  static int select_distance = std::stoi(common::GetCompileConfig("HYPER_OFFLOAD_SELECT_DISTANCE"));
  std::vector<bool> need_offload;
  for (const auto &cur_distance_element : relative_distance_list) {
    (void)need_offload.emplace_back(((int)cur_distance_element.second > select_distance));
  }
  ReplaceInfoList replace_info_list;
  if (!std::any_of(need_offload.begin(), need_offload.end(), [](bool offload) { return offload; })) {
    return replace_info_list;
  }
  // With input:
  // relative_distance_list: (((node1, node2), distance1), ((node2, node3), distance2), ((node3, node4), distance3))
  // need_offload: (False, True, False)
  // generate replace distance info:
  // ((node2, (node3, node4)), )
  for (size_t i = 0; i < relative_distance_list.size(); ++i) {
    const auto &new_need_change_node = relative_distance_list[i].first.second;
    if (need_offload[i]) {
      const auto &upper_bound = relative_distance_list[i].first.first;
      std::vector<KernelWithIndex> need_change_nodes{new_need_change_node};
      replace_info_list.emplace_back(ReplaceInfo(upper_bound, need_change_nodes));
      continue;
    }
    if (replace_info_list.empty()) {
      continue;
    }
    (void)replace_info_list.back().second.emplace_back(new_need_change_node);
  }
  MS_EXCEPTION_IF_CHECK_FAIL(!replace_info_list.empty(), "Replace info list should not be empty");
  return replace_info_list;
}

std::optional<OffloadInfo> GenerateOffloadInfo(const UserInfo &user_info, const CNodePtrList &execution_order,
                                               bool offload_activation) {
  if (!IsValidOffloadNode(user_info, offload_activation)) {
    return std::nullopt;
  }
  const auto &absolute_distance_list = GenerateAbsoluteDistanceList(user_info, execution_order);
  if (absolute_distance_list.size() == 1) {
    return std::nullopt;
  }
  const auto &relative_distance_list = GenerateRelativeDistanceList(absolute_distance_list);
  const auto &replace_info_list = GenerateReplaceInfoList(relative_distance_list);
  if (replace_info_list.empty()) {
    return std::nullopt;
  }
  return OffloadInfo{user_info.first, replace_info_list};
}

OffloadInfoList GenerateOffloadInfoList(const KernelGraphPtr &kernel_graph, const UserInfoList &user_info_list) {
  const auto &execution_order = kernel_graph->execution_order();
  OffloadInfoList offload_info_list;
  bool offload_activation = NeedOffloadActivation(kernel_graph);
  for (const auto &cur_user_info : user_info_list) {
    const auto &offload_info_opt = GenerateOffloadInfo(cur_user_info, execution_order, offload_activation);
    if (!offload_info_opt.has_value()) {
      continue;
    }
    const auto &offload_info = offload_info_opt.value();
    (void)offload_info_list.emplace_back(offload_info);
  }
  return offload_info_list;
}

void ChangeExecutionOrderByOffloadInfo(const OffloadInfo &offload_info, const std::vector<KernelWithIndex> &outputs,
                                       const KernelGraphPtr &kernel_graph, const UserInfoList &user_info_list,
                                       CNodePtrList &new_execution_order) {
  const auto &data_node_info = offload_info.first;
  auto data_node = data_node_info.first;
  auto data_node_abstract = data_node->abstract();
  MS_EXCEPTION_IF_NULL(data_node_abstract);
  if (!data_node_abstract->isa<abstract::AbstractTensor>()) {
    if (!data_node_abstract->isa<abstract::AbstractTuple>()) {
      MS_LOG(EXCEPTION) << "Found non-except abstract for node " << data_node->fullname_with_scope()
                        << " with abstract: " << data_node_abstract->ToString();
    }
    auto index_node = kernel_graph->NewValueNode(MakeValue(int64_t(data_node_info.second)));
    index_node->set_abstract(std::make_shared<abstract::AbstractScalar>(int64_t(data_node_info.second)));
    data_node = kernel_graph->NewCNode(
      {NewValueNode(std::make_shared<Primitive>(prim::kPrimTupleGetItem->name())), data_node, index_node});
    data_node->set_abstract(data_node_abstract->cast<abstract::AbstractTuplePtr>()->elements()[data_node_info.second]);
  }
  auto to_remote_node = BuildToHostNode(kernel_graph, data_node);
  for (const auto &cur_replace_info : offload_info.second) {
    auto to_device_node = BuildToDeviceNode(kernel_graph, to_remote_node);
    auto upper_bound_node = cur_replace_info.first.first;
    const auto &change_nodes_infos = cur_replace_info.second;
    MS_EXCEPTION_IF_CHECK_FAIL(!change_nodes_infos.empty(), "change_nodes should not be empty");
    for (const auto &change_node_info : change_nodes_infos) {
      auto change_node = change_node_info.first->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(change_node);
      int index = change_node_info.second;
      MS_EXCEPTION_IF_CHECK_FAIL(index >= 0, "fail to found index.");
      change_node->set_input(index, to_device_node);
    }
    auto position = std::find(new_execution_order.begin(), new_execution_order.end(), change_nodes_infos[0].first);
    new_execution_order.insert(position, to_device_node);
  }
  bool is_output =
    std::any_of(outputs.begin(), outputs.end(), [&data_node](const auto &e) { return e.first == data_node; });
  if (is_output) {
    auto output = kernel_graph->output()->cast<CNodePtr>();
    MS_EXCEPTION_IF_CHECK_FAIL(IsPrimitiveCNode(output, prim::kPrimMakeTuple), "output not make tuple");
    int index = -1;
    for (size_t i = 1; i < output->inputs().size(); ++i) {
      if (output->input(i) == data_node) {
        index = SizeToInt(i);
        break;
      }
    }
    MS_EXCEPTION_IF_CHECK_FAIL(index > 0, "Fail to find index");
    auto to_device_node = BuildToDeviceNode(kernel_graph, to_remote_node);
    output->set_input(IntToSize(index), to_device_node);
    new_execution_order.emplace_back(to_device_node);
  }
  auto data_iter = std::find(new_execution_order.begin(), new_execution_order.end(), data_node_info.first);
  MS_EXCEPTION_IF_CHECK_FAIL(data_iter != new_execution_order.end(), "Fail to find node in execution order.");
  new_execution_order.insert(data_iter + 1, to_remote_node);
}

std::optional<AnfNodePtr> FindLastRealUser(const AnfNodePtr &node, const UserInfoList &user_info_list) {
  int64_t output_index = 0;
  if (IsPrimitiveCNode(node, prim::kPrimTupleGetItem)) {
    auto cnode = node->cast<CNodePtr>();
    constexpr size_t index_pos = 2;
    auto index_value_node = cnode->input(index_pos)->cast<ValueNodePtr>();
    output_index = GetValue<int64_t>(index_value_node->value());
  }
  auto node_iter = std::find_if(user_info_list.begin(), user_info_list.end(), [&node, output_index](const auto &e) {
    if (IsPrimitiveCNode(node, prim::kPrimTupleGetItem)) {
      auto real_data = node->cast<CNodePtr>()->input(1);
      return e.first.first == real_data && (int64_t)e.first.second == output_index;
    }
    return e.first.first == node && (int64_t)e.first.second == output_index;
  });
  if (node_iter == user_info_list.end()) {
    return std::nullopt;
  }
  const auto &node_users = node_iter->second;
  auto last_user = node_users[node_users.size() - 1].first;
  // ToRemote will be placed right after the data is changed.
  if (IsD2HNode(last_user)) {
    return std::nullopt;
  }
  return last_user;
}

void InsertHyperOffloadNodeToExecutionOrder(const KernelGraphPtr &kernel_graph) {
  if (!NeedOffload(kernel_graph)) {
    return;
  }
  const auto &user_info_list = CollectAllNodeUsers(kernel_graph->execution_order());
  const auto &offload_info_list = GenerateOffloadInfoList(kernel_graph, user_info_list);
  MS_LOG(INFO) << "offload_info_list size: " << offload_info_list.size();

  const auto &outputs = common::AnfAlgo::GetAllOutputWithIndex(kernel_graph->output());
  CNodePtrList new_execution_order = kernel_graph->execution_order();
  static int offload_total_number = std::stoi(common::GetCompileConfig("HYPER_OFFLOAD_SELECT_NUM"));
  int current_offload_number = 0;
  for (const auto &offload_info : offload_info_list) {
    current_offload_number++;
    if (current_offload_number >= offload_total_number) {
      break;
    }
    const auto &cur_name = offload_info.first.first->fullname_with_scope();
    ChangeExecutionOrderByOffloadInfo(offload_info, outputs, kernel_graph, user_info_list, new_execution_order);
  }
  kernel_graph->set_execution_order(new_execution_order);
}

void AdjustHyperOffloadNodePosition(const KernelGraphPtr &kernel_graph) {
  if (!NeedOffload(kernel_graph)) {
    return;
  }
  const auto &nodes = kernel_graph->execution_order();
  static int prefetch_distance = std::stoi(common::GetCompileConfig("HYPER_OFFLOAD_PREFETCH_DISTANCE"));
  CNodePtrList new_execution_order{};
  for (const auto &node : nodes) {
    if (IsH2DNode(node)) {
      auto cnode = node->cast<CNodePtr>();
      auto data_node = cnode->input(1);
      if (data_node->isa<CNode>() && !IsD2HNode(data_node)) {
        MS_LOG(EXCEPTION) << "Unexpected data cnode " << data_node->DebugString();
      }
      if (new_execution_order.empty()) {
        (void)new_execution_order.emplace_back(node);
        continue;
      }
      size_t upper_bound_index = 0;
      if (data_node->isa<CNode>()) {
        auto upper_bound_iter = std::find(new_execution_order.begin(), new_execution_order.end(), data_node);
        MS_EXCEPTION_IF_CHECK_FAIL(upper_bound_iter != new_execution_order.end(), "Fail to find data node");
        upper_bound_index = std::distance(new_execution_order.begin(), upper_bound_iter) + 1;
      }
      auto data_size = AnfAlgo::GetOutputTensorMemSize(node, 0);
      MS_LOG(INFO) << "For node with size: " << data_size << ", upper bound index: " << upper_bound_index
                   << ", lower bound index: " << (new_execution_order.size() - 1);
      if (node->has_user_data("prefetch_distance")) {
        size_t custom_prefetch_distance = *(node->user_data<size_t>("prefetch_distance"));
        if (new_execution_order.size() - 1 - custom_prefetch_distance <= upper_bound_index) {
          MS_LOG(EXCEPTION) << "Prefetch distance is too large";
        }
        (void)new_execution_order.insert(
          new_execution_order.begin() + new_execution_order.size() - custom_prefetch_distance, node);
      } else {
        (void)new_execution_order.insert(new_execution_order.begin() + new_execution_order.size() - prefetch_distance,
                                         node);
      }
      continue;
    }
    (void)new_execution_order.emplace_back(node);
  }
  kernel_graph->set_execution_order(new_execution_order);
}

std::optional<ParameterPtr> FindParameterByRefName(const KernelGraphPtr &kernel_graph, const std::string &name) {
  for (const auto &input : kernel_graph->parameters()) {
    if (!input->isa<Parameter>()) {
      continue;
    }
    auto param = input->cast<ParameterPtr>();
    if (param->name() == name) {
      return param;
    }
  }
  return std::nullopt;
}

std::optional<ParameterPtr> GetRemoteParameter(const KernelGraphPtr &kernel_graph, const mindspore::AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto abstract = node->abstract();
  if (abstract == nullptr) {
    return std::nullopt;
  }
  if (!abstract->isa<abstract::AbstractRefTensor>()) {
    return std::nullopt;
  }
  auto abstract_ref = abstract->cast<abstract::AbstractRefPtr>();
  const auto &ref_value = abstract_ref->ref_key_value()->cast<StringImmPtr>();
  MS_EXCEPTION_IF_NULL(ref_value);
  const auto &ref_name = ref_value->value();
  const auto &ref_node_res = FindParameterByRefName(kernel_graph, ref_name);
  if (!ref_node_res.has_value()) {
    return std::nullopt;
  }
  const auto &parameter = ref_node_res.value();
  const auto &device_str = AnfAlgo::GetParameterDeviceStr(parameter);
  if (device_str != kToRemote) {
    return std::nullopt;
  }
  return parameter;
}

void ReplaceParameterLoadWithHyperOffloadNode(const KernelGraphPtr &kernel_graph) {
  static const bool enable_hyper_offload = common::GetEnv("MS_DEV_HYPER_OFFLOAD") != "1";
  if (enable_hyper_offload) {
    return;
  }
  const auto &nodes = kernel_graph->execution_order();
  CNodePtrList new_execution_order{};
  for (const auto &node : nodes) {
    CNodePtrList to_remote_nodes{};
    if (IsPrimitiveCNode(node, prim::kPrimConditionSwitch)) {
      for (size_t i = 0; i < node->inputs().size(); ++i) {
        auto input = node->input(i);
        auto remote_param_res = GetRemoteParameter(kernel_graph, input);
        if (!remote_param_res.has_value()) {
          continue;
        }
        auto remote_param = remote_param_res.value();
        auto to_device_node = BuildToDeviceNode(kernel_graph, remote_param);
        to_device_node->set_abstract(remote_param->abstract());
        node->set_input(i, to_device_node);
        (void)new_execution_order.emplace_back(to_device_node);
      }
      (void)new_execution_order.emplace_back(node);
      continue;
    }
    for (size_t i = 0; i < node->inputs().size(); ++i) {
      auto input = node->input(i);
      auto real_input = common::AnfAlgo::VisitKernelWithReturnType(input, 0, false, {prim::kPrimLoad}).first;
      if (IsPrimitiveCNode(real_input, prim::kPrimLoad)) {
        auto data_node = real_input->cast<CNodePtr>()->input(1);
        data_node = common::AnfAlgo::VisitKernelWithReturnType(data_node, 0, false, {prim::kPrimLoad}).first;
        if (!data_node->isa<Parameter>()) {
          continue;
        }
        if (!GetRemoteParameter(kernel_graph, data_node).has_value()) {
          continue;
        }
        auto to_device_node = BuildToDeviceNode(kernel_graph, data_node);
        to_device_node->set_abstract(real_input->abstract());
        node->set_input(i, to_device_node);
        (void)new_execution_order.emplace_back(to_device_node);
        continue;
      }
      const auto &param_res = GetRemoteParameter(kernel_graph, input);
      if (!param_res.has_value()) {
        continue;
      }
      auto changed_node = input;
      if (real_input->isa<Parameter>()) {
        auto to_device_node = BuildToDeviceNode(kernel_graph, input);
        to_device_node->set_abstract(dyn_cast<abstract::AbstractRefTensor>(input->abstract())->CloneAsTensor());
        node->set_input(i, to_device_node);
        changed_node = to_device_node;
        (void)new_execution_order.emplace_back(to_device_node);
      }
      (void)to_remote_nodes.emplace_back(BuildInplaceToHostNode(kernel_graph, changed_node, param_res.value(), node));
    }
    (void)new_execution_order.emplace_back(node);
    const auto &inline_sub_graph_kernels = kernel_graph->inline_sub_graph_kernels();
    auto iter = inline_sub_graph_kernels.find(node);
    if (iter != inline_sub_graph_kernels.end()) {
      for (const auto &to_remote_node : to_remote_nodes) {
        kernel_graph->AddInlineSubgraphKernel(to_remote_node, iter->second);
      }
    }
    for (const auto &to_remote_node : to_remote_nodes) {
      (void)new_execution_order.emplace_back(to_remote_node);
    }
  }
  kernel_graph->set_execution_order(new_execution_order);
}

// Set branch information attributes for the Switch node
void SetBranchInfoForSwitchNode(const ConditionSwitchInfoPtr &switch_info) {
  MS_EXCEPTION_IF_NULL(switch_info);
  auto switch_cnode = switch_info->switch_cnode;
  MS_EXCEPTION_IF_NULL(switch_cnode);
  switch_cnode->AddAttr(kSwitchTrueBranchNum, MakeValue(switch_info->true_branch_node_num));
  switch_cnode->AddAttr(kSwitchFalseBranchNum, MakeValue(switch_info->false_branch_node_num));
}

// Handle the ConditionGather node, reorder the control flow nodes for the condition gather node
CNodePtrList ReorderControlFlowNodesForConditionGather(std::vector<ConditionSwitchInfoPtr> *switch_index,
                                                       CNodePtrList *true_branch_cnodes,
                                                       CNodePtrList *false_branch_cnodes, const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(switch_index);
  MS_EXCEPTION_IF_NULL(true_branch_cnodes);
  MS_EXCEPTION_IF_NULL(false_branch_cnodes);
  MS_EXCEPTION_IF_NULL(cnode);
  auto lastest_switch_info = switch_index->back();
  CNodePtrList switch_kernel_cnodes;
  MS_EXCEPTION_IF_NULL(lastest_switch_info);
  auto true_branch_size = lastest_switch_info->true_branch_node_num;
  // Mark true branch range
  if (true_branch_size > true_branch_cnodes->size()) {
    MS_LOG(EXCEPTION) << "Invalid true branch node num: " << true_branch_size << " , "
                      << lastest_switch_info->true_branch_node_num;
  }
  switch_kernel_cnodes.insert(switch_kernel_cnodes.end(), true_branch_cnodes->end() - true_branch_size,
                              true_branch_cnodes->end());
  true_branch_cnodes->resize(true_branch_cnodes->size() - true_branch_size);
  // Mark false branch range
  auto false_branch_size = lastest_switch_info->false_branch_node_num;
  if (false_branch_size > false_branch_cnodes->size()) {
    MS_LOG(EXCEPTION) << "Invalid false branch node num: " << false_branch_size << " , "
                      << lastest_switch_info->false_branch_node_num;
  }
  switch_kernel_cnodes.insert(switch_kernel_cnodes.end(), false_branch_cnodes->end() - false_branch_size,
                              false_branch_cnodes->end());
  false_branch_cnodes->resize(false_branch_cnodes->size() - false_branch_size);
  MS_VLOG(VL_HYPER_OFFLOAD_INFO) << "Switch scope, true branch size: " << true_branch_size
                                 << ", false branch size: " << false_branch_size;
  switch_kernel_cnodes.emplace_back(cnode);
  SetBranchInfoForSwitchNode(lastest_switch_info);
  switch_index->pop_back();

  return switch_kernel_cnodes;
}

void UpdateOuterSwitchInfo(const std::vector<ConditionSwitchInfoPtr> &switch_index, CNodePtrList *true_branch_cnodes,
                           CNodePtrList *false_branch_cnodes, const CNodePtrList &switch_kernel_cnodes,
                           const std::string &subgraph_name) {
  if (switch_index.empty()) {
    return;
  }
  // Nested control flow, update outer switch info
  auto &outer_switch_info = switch_index.back();
  bool is_true_branch = subgraph_name == outer_switch_info->true_subgraph;
  if (is_true_branch) {
    MS_EXCEPTION_IF_NULL(true_branch_cnodes);
    true_branch_cnodes->insert(true_branch_cnodes->end(), switch_kernel_cnodes.begin(), switch_kernel_cnodes.end());
    outer_switch_info->true_branch_node_num += switch_kernel_cnodes.size();
  } else {
    MS_EXCEPTION_IF_NULL(false_branch_cnodes);
    false_branch_cnodes->insert(false_branch_cnodes->end(), switch_kernel_cnodes.begin(), switch_kernel_cnodes.end());
    outer_switch_info->false_branch_node_num += switch_kernel_cnodes.size();
  }
}

// Create a ConditionSwitchInfo object for the ConditionSwitch cnode
ConditionSwitchInfoPtr CreateConditionSwitchInfo(const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  const auto &inline_sub_graph_names = cnode->GetAttr(kInlineSubGraphName);
  if (inline_sub_graph_names == nullptr || !inline_sub_graph_names->isa<ValueTuple>()) {
    MS_LOG(EXCEPTION) << "Invalid condition switch node: " << cnode->DebugString();
  }
  const auto &tuple_name = inline_sub_graph_names->cast<ValueTuplePtr>()->value();
  constexpr size_t tuple_name_size = 2;
  if (tuple_name.size() != tuple_name_size) {
    MS_LOG(EXCEPTION) << "Invalid condition switch node: " << cnode->DebugString()
                      << ", tuple name size: " << tuple_name.size();
  }
  MS_LOG(INFO) << "Condition switch cnode: " << cnode->DebugString()
               << ", sub graph name: " << inline_sub_graph_names->ToString();
  constexpr size_t true_branch_index = 1;
  constexpr size_t false_branch_index = 0;
  auto true_branch_name = GetValue<std::string>(tuple_name[true_branch_index]);
  auto false_branch_name = GetValue<std::string>(tuple_name[false_branch_index]);
  auto cur_switch_info = std::make_shared<ConditionSwitchInfo>(cnode, true_branch_name, false_branch_name);
  return cur_switch_info;
}
}  // namespace

void ExecutionOrderOptimizeWithHyperOffload(const KernelGraphPtr &kernel_graph) {
  static const bool enable_hyper_offload = common::GetEnv("MS_DEV_HYPER_OFFLOAD") != "1";
  if (enable_hyper_offload) {
    return;
  }
  ReplaceParameterLoadWithHyperOffloadNode(kernel_graph);
  InsertHyperOffloadNodeToExecutionOrder(kernel_graph);
  AdjustHyperOffloadNodePosition(kernel_graph);
}

void AdjustExecutionOrderForHyperOffloadOps(const KernelGraphPtr &kernel_graph) {
  static const bool enable_hyper_offload = common::GetEnv("MS_DEV_HYPER_OFFLOAD") != "1";
  if (enable_hyper_offload) {
    return;
  }
  CNodePtrList new_execution_order{};
  CNodePtrList remain_h2d_nodes{};
  for (const auto node : kernel_graph->execution_order()) {
    if (IsH2DNode(node)) {
      (void)remain_h2d_nodes.emplace_back(node);
      continue;
    }
    // ToRemote node should execute right after data is created.
    if (IsD2HNode(node)) {
      constexpr size_t data_index = 1;
      auto data_node = node->input(data_index);
      if (data_node->isa<CNode>()) {
        auto data_node_iter = std::find(new_execution_order.begin(), new_execution_order.end(), data_node);
        MS_EXCEPTION_IF_CHECK_FAIL(data_node_iter != new_execution_order.end(), "Find data_node failed.");
        new_execution_order.insert(data_node_iter + 1, node);
      } else {
        (void)new_execution_order.emplace_back(node);
      }
      continue;
    }
    const auto &inputs = node->inputs();
    if (IsViewOrInplaceNode(node)) {
      auto new_end = std::remove_if(remain_h2d_nodes.begin(), remain_h2d_nodes.end(), [&](const CNodePtr &h2d_node) {
        constexpr size_t h2d_data_idx = 1;
        const auto &h2d_data = h2d_node->input(h2d_data_idx);
        bool found =
          std::any_of(inputs.begin(), inputs.end(), [&h2d_data](const auto &input) { return input == h2d_data; });
        if (found) {
          (void)new_execution_order.emplace_back(h2d_node);
        }
        return found;
      });
      remain_h2d_nodes.erase(new_end, remain_h2d_nodes.end());
      new_execution_order.emplace_back(node);
      continue;
    }

    for (const auto &input : inputs) {
      if (!IsH2DNode(input)) {
        continue;
      }
      auto remain_iter = std::find(remain_h2d_nodes.begin(), remain_h2d_nodes.end(), input);
      if (remain_iter != remain_h2d_nodes.end()) {
        remain_h2d_nodes.erase(remain_iter);
        (void)new_execution_order.emplace_back(input->cast<CNodePtr>());
      }
    }
    (void)new_execution_order.emplace_back(node);
  }
  for (const auto &node : remain_h2d_nodes) {
    (void)new_execution_order.emplace_back(node);
  }
  kernel_graph->set_execution_order(new_execution_order);
}

void AddEventToHyperOffloadOps(const KernelGraphPtr &kernel_graph) {
  static const bool enable_hyper_offload = common::GetEnv("MS_DEV_HYPER_OFFLOAD") != "1";
  if (enable_hyper_offload) {
    return;
  }
  CNodePtrList new_execution_order{};
  const auto &nodes = kernel_graph->execution_order();
  const auto &all_node_users = CollectAllNodeUsers(nodes);
  std::unordered_map<AnfNodePtr, AnfNodePtrList> copy_to_remote_event_map;
  int distance = std::stoi(common::GetCompileConfig("HYPER_OFFLOAD_RELEASE_DISTANCE"));
  for (size_t i = 0; i < nodes.size(); ++i) {
    const auto &node = nodes[i];
    if (IsD2HNode(node)) {
      auto process_stream_id = AnfAlgo::GetStreamId(node);
      auto wait_stream_id = 0;

      auto cnode = node->cast<CNodePtr>();
      auto data_node = cnode->input(1);
      const auto &send_recv_pair =
        AclStreamAssign::GetInstance().CreateSendReceive(NOT_NULL(kernel_graph), process_stream_id, wait_stream_id);
      (void)new_execution_order.emplace_back(node);
      (void)new_execution_order.emplace_back(send_recv_pair.first);
      auto data_node_last_user = FindLastRealUser(data_node, all_node_users);
      AnfNodePtr insert_node = nullptr;
      if (data_node_last_user.has_value()) {
        auto last_user_node = data_node_last_user.value();
        auto last_user_node_iter = std::find(nodes.begin(), nodes.end(), last_user_node);
        MS_EXCEPTION_IF_CHECK_FAIL(last_user_node_iter != nodes.end(), "Fail to find last user node.");
        insert_node = *(last_user_node_iter + distance);
      } else {
        auto node_iter = std::find(nodes.begin(), nodes.end(), node);
        MS_EXCEPTION_IF_CHECK_FAIL(node_iter != nodes.end(), "Fail to find node");
        auto insert_node_iter = std::next(node_iter, distance);
        MS_EXCEPTION_IF_CHECK_FAIL(insert_node_iter != nodes.end(), "Failed to find insert node position");
        insert_node = *(insert_node_iter);
      }
      MS_EXCEPTION_IF_NULL(insert_node);
      auto copy_to_remote_event_map_iter = copy_to_remote_event_map.find(insert_node);
      if (copy_to_remote_event_map_iter == copy_to_remote_event_map.end()) {
        copy_to_remote_event_map[insert_node] = {send_recv_pair.second};
      } else {
        copy_to_remote_event_map[insert_node].emplace_back(send_recv_pair.second);
      }
    } else {
      (void)new_execution_order.emplace_back(node);
    }
    auto iter = copy_to_remote_event_map.find(node);
    if (iter != copy_to_remote_event_map.end()) {
      const auto &all_recv_nodes = iter->second;
      for (const auto recv_node : all_recv_nodes) {
        (void)new_execution_order.emplace_back(recv_node->cast<CNodePtr>());
      }
      copy_to_remote_event_map.erase(iter);
    }
  }
  MS_EXCEPTION_IF_CHECK_FAIL(copy_to_remote_event_map.empty(), "Remain recv node to insert.");
  kernel_graph->set_execution_order(new_execution_order);
}

// Reorder the control flow nodes in the execution order, keep true branch and false branch nodes together
void ReorderControlFlowNodes(const KernelGraphPtr &kernel_graph) {
  static const bool use_hyper_offload_with_sliding_window =
    common::GetEnv("MS_DEV_HYPER_OFFLOAD") == "1" && common::GetCompileConfig("ENABLE_HYPER_OFFLOAD_SLIDE") == "1";
  if (!use_hyper_offload_with_sliding_window) {
    return;
  }
  MS_EXCEPTION_IF_NULL(kernel_graph);
  const auto &cnodes = kernel_graph->execution_order();
  std::vector<ConditionSwitchInfoPtr> switch_index;
  CNodePtrList true_branch_cnodes;
  CNodePtrList false_branch_cnodes;
  CNodePtrList new_execution_order{};
  const auto &cnode_branch_map = kernel_graph->inline_sub_graph_kernels();
  for (const auto &cnode : cnodes) {
    MS_EXCEPTION_IF_NULL(cnode);
    // Not in control flow node
    if (switch_index.empty()) {
      if (!IsPrimitiveCNode(cnode, prim::kPrimConditionSwitch)) {
        new_execution_order.emplace_back(cnode);
      } else {
        auto cur_switch_info = CreateConditionSwitchInfo(cnode);
        switch_index.push_back(cur_switch_info);
        new_execution_order.emplace_back(cnode);
      }
      continue;
    }

    const auto &branch_subgraph_name_iter = cnode_branch_map.find(cnode);
    std::string branch_subgraph_name = "";
    if (branch_subgraph_name_iter != cnode_branch_map.end()) {
      branch_subgraph_name = branch_subgraph_name_iter->second;
    } else if (!IsPrimitiveCNode(cnode, prim::kPrimConditionGather) || switch_index.size() != 1) {
      MS_LOG(EXCEPTION) << "Invalid node in control flow: " << cnode->DebugString();
    }

    // ConditionGather, process the whole control flow
    if (IsPrimitiveCNode(cnode, prim::kPrimConditionGather)) {
      auto switch_kernel_cnodes =
        ReorderControlFlowNodesForConditionGather(&switch_index, &true_branch_cnodes, &false_branch_cnodes, cnode);
      // Single control flow, just append to the execution order
      if (switch_index.empty()) {
        new_execution_order.insert(new_execution_order.end(), switch_kernel_cnodes.begin(), switch_kernel_cnodes.end());
      } else {
        UpdateOuterSwitchInfo(switch_index, &true_branch_cnodes, &false_branch_cnodes, switch_kernel_cnodes,
                              branch_subgraph_name);
      }
      continue;
    }

    auto &lastest_switch_info = switch_index.back();
    MS_EXCEPTION_IF_NULL(lastest_switch_info);
    if (lastest_switch_info->true_subgraph == branch_subgraph_name) {
      true_branch_cnodes.emplace_back(cnode);
      lastest_switch_info->true_branch_node_num++;
    } else {
      false_branch_cnodes.emplace_back(cnode);
      lastest_switch_info->false_branch_node_num++;
    }

    // Nested control flow scene
    if (IsPrimitiveCNode(cnode, prim::kPrimConditionSwitch)) {
      auto cur_switch_info = CreateConditionSwitchInfo(cnode);
      switch_index.push_back(cur_switch_info);
    }
  }
  kernel_graph->set_execution_order(new_execution_order);
}
}  // namespace hyper_offload
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
