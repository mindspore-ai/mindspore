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

#include "backend/common/graph_kernel/inplace_assign_builder.h"

#include <algorithm>
#include <memory>
#include <utility>
#include <set>
#include <string>
#include <map>
#include <vector>
#include "mindspore/core/ops/core_ops.h"
#include "ir/tensor.h"
#include "include/common/utils/utils.h"
#include "include/common/debug/anf_ir_dump.h"
#include "utils/log_adapter.h"
#include "kernel/kernel.h"
#include "kernel/common_utils.h"
#include "backend/common/graph_kernel/graph_kernel_helper.h"
#include "backend/common/graph_kernel/core/graph_kernel_utils.h"

namespace mindspore::graphkernel {
namespace {
CNodePtr CreateAssign(const FuncGraphPtr &sub_graph,
                      const std::vector<std::pair<InplaceAssignerInfo, AnfNodePtr>> &parameters_infos, size_t idx) {
  if (idx >= parameters_infos.size()) {
    MS_LOG(EXCEPTION) << "idx " << idx << " is out of range [0, " << parameters_infos.size() << ")";
  }
  MS_EXCEPTION_IF_NULL(sub_graph);

  const auto &target_node = parameters_infos[idx].first.op_node;
  const auto &new_parameter = parameters_infos[idx].second;

  auto node = CreateCNode({NewValueNode(prim::kPrimAssign), new_parameter, target_node}, sub_graph,
                          {GetFormat(target_node), GetShape(target_node), GetType(target_node)});
  return node;
}

size_t GetItemIdx(const AnfNodePtr &node) {
  if (!IsPrimitiveCNode(node, prim::kPrimTupleGetItem)) {
    MS_LOG(EXCEPTION) << "Expect TupleGetItem node, but got " << common::AnfAlgo::GetCNodeName(node);
  }
  auto get_item_cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(get_item_cnode);
  auto value_input = get_item_cnode->input(kInputNodeOutputIndexInTupleGetItem);
  MS_EXCEPTION_IF_NULL(value_input);
  auto value_node = value_input->cast<ValueNodePtr>();
  MS_EXCEPTION_IF_NULL(value_node);
  auto item_idx = LongToSize(GetValue<int64_t>(value_node->value()));
  return item_idx;
}
}  // namespace

void InplaceAssignBuilder::CorrectKernelBuildInfo(
  const AnfNodePtr &composite_node, const std::vector<std::pair<InplaceAssignerInfo, AnfNodePtr>> &inplace_infos) {
  // Change kernel build info.
  auto kernel_info = dynamic_cast<device::KernelInfo *>(composite_node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  const auto &origin_kernel_build_info = kernel_info->GetMutableSelectKernelBuildInfo();
  MS_EXCEPTION_IF_NULL(origin_kernel_build_info);
  auto origin_inputs_format = origin_kernel_build_info->GetAllInputFormats();
  auto origin_inputs_type = origin_kernel_build_info->GetAllInputDeviceTypes();

  std::vector<std::string> &new_inputs_format = origin_inputs_format;
  std::vector<TypeId> &new_inputs_type = origin_inputs_type;
  for (const auto &inplace_info : inplace_infos) {
    if (inplace_info.first.inplace_to_origin_input < 0) {
      auto &new_input = inplace_info.second;
      auto kernel_with_index = common::AnfAlgo::VisitKernel(new_input, 0);
      new_inputs_format.push_back(AnfAlgo::GetOutputFormat(kernel_with_index.first, kernel_with_index.second));
      new_inputs_type.push_back(AnfAlgo::GetOutputDeviceDataType(kernel_with_index.first, kernel_with_index.second));
    }
  }

  auto new_selected_info = BuildSelectKernelBuildInfo(
    new_inputs_format, new_inputs_type, origin_kernel_build_info->GetAllOutputFormats(),
    origin_kernel_build_info->GetAllOutputDeviceTypes(), origin_kernel_build_info->processor());
  AnfAlgo::SetSelectKernelBuildInfo(new_selected_info, composite_node.get());
}

void InplaceAssignBuilder::CreateAssignNodeAndCorrectReturn(
  const FuncGraphPtr &sub_graph,
  const std::vector<std::pair<InplaceAssignerInfo, AnfNodePtr>> &parameters_infos) const {
  std::map<size_t, size_t> target_indices;
  for (size_t i = 0; i < parameters_infos.size(); ++i) {
    target_indices[parameters_infos[i].first.real_output_index + 1] = i;
  }

  // Change output to Assign node.
  auto output = sub_graph->output();
  if (IsPrimitiveCNode(output, prim::kPrimMakeTuple)) {
    auto output_cnode = output->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(output_cnode);
    for (size_t i = 1; i < output_cnode->inputs().size(); ++i) {
      std::map<size_t, size_t>::const_iterator cur_input = target_indices.find(i);
      if (cur_input == target_indices.end()) {
        continue;
      }
      auto inplace = CreateAssign(sub_graph, parameters_infos, cur_input->second);
      output_cnode->set_input(i, inplace);
    }
  } else if (parameters_infos.size() == 1) {
    auto inplace = CreateAssign(sub_graph, parameters_infos, 0);
    sub_graph->set_output(inplace);
  }
}

CNodePtr InplaceAssignBuilder::CreateCleanCompositeNode(const InplaceAssignerInfo &op_info,
                                                        const FuncGraphPtr &main_graph, TypeId dst_type) {
  std::set<TypeId> data_support = {kNumberTypeFloat16, kNumberTypeFloat32, kNumberTypeFloat64};

  if (!std::any_of(data_support.cbegin(), data_support.cend(), [&dst_type](TypeId type) { return dst_type == type; })) {
    MS_LOG(EXCEPTION) << "For CreateCleanCompositeNode, the data type: " << TypeIdToString(dst_type, true)
                      << " is not in supported list: [float16, float32, float64].";
  }

  // Create zero value which will be broadcast to target shape.
  auto format = GetFormat(op_info.op_node);
  auto dtype = (dst_type == kNumberTypeFloat16) ? kNumberTypeFloat32 : dst_type;
  ValueNodePtr value_node;
  if (dtype == kNumberTypeFloat32) {
    value_node =
      CreateScalarTensorValueNode<float>({format, {1}, TypeIdToType(dtype)}, static_cast<float>(0), sizeof(float));
  } else {
    value_node =
      CreateScalarTensorValueNode<double>({format, {1}, TypeIdToType(dtype)}, static_cast<double>(0), sizeof(double));
  }

  // Create composite op's sub-graph.
  auto new_sub_graph = std::make_shared<FuncGraph>();

  AnfNodePtr broadcast_input_node;
  if (dst_type == kNumberTypeFloat16) {
    AnfNodePtrList cast_inputs = {NewValueNode(prim::kPrimCast), value_node};
    auto cast_node_inner = CreateCNode(cast_inputs, new_sub_graph, {format, {1}, TypeIdToType(dst_type)});
    SetNodeAttrSafely("dst_type", kFloat32, cast_node_inner);
    broadcast_input_node = cast_node_inner;
  } else {
    broadcast_input_node = value_node;
  }

  // Create broadcast basic op.
  auto dst_shape_vec = GetShape(op_info.op_node);
  AnfNodePtrList clean_inputs = {NewValueNode(prim::kPrimBroadcastTo), broadcast_input_node};
  auto broadcast_to_node_inner =
    CreateCNode(clean_inputs, new_sub_graph, {format, dst_shape_vec, GetType(op_info.op_node)});
  SetNodeAttrSafely("shape", MakeValue(GetDeviceShape(op_info.op_node)), broadcast_to_node_inner);

  // Makeup sub-graph.
  new_sub_graph->set_output(broadcast_to_node_inner);
  auto broadcast_to_composite_node = main_graph->NewCNode({NewValueNode(new_sub_graph)});
  broadcast_to_composite_node->set_abstract(broadcast_to_node_inner->abstract());
  SetNewKernelInfo(broadcast_to_composite_node, new_sub_graph, {}, {broadcast_to_node_inner});
  auto graph_attr =
    GkUtils::ExtractGraphKernelName(TopoSort(new_sub_graph->get_return()), "", "inplace_assign_builder");
  new_sub_graph->set_attr(FUNC_GRAPH_ATTR_GRAPH_KERNEL, MakeValue(graph_attr));
  new_sub_graph->set_attr("composite_type", MakeValue("inplace_assign_builder"));

  return broadcast_to_composite_node;
}

void InplaceAssignBuilder::ProcessOriginCNode(
  const AnfNodePtr &composite_node,
  const std::vector<std::pair<InplaceAssignerInfo, AnfNodePtr>> &info_and_inplace_assignee_addr) {
  auto sub_graph = common::AnfAlgo::GetCNodeFuncGraphPtr(composite_node);
  auto mng_sub = sub_graph->manager();
  if (mng_sub == nullptr) {
    mng_sub = Manage(sub_graph, false);
    sub_graph->set_manager(mng_sub);
  }

  // Add input
  std::vector<std::pair<InplaceAssignerInfo, AnfNodePtr>> parameters_infos;
  std::vector<AnfNodePtr> additonal_inputs;
  for (const auto &[target_node_info, input] : info_and_inplace_assignee_addr) {
    // Add attribute to target node.
    SetTargetAttrs(target_node_info.op_node);

    // add parameter
    if (target_node_info.inplace_to_origin_input < 0) {
      auto parameter = sub_graph->add_parameter();
      parameter->set_abstract(input->abstract());
      parameter->set_kernel_info(input->kernel_info_ptr());
      (void)parameters_infos.emplace_back(target_node_info, parameter);
      (void)additonal_inputs.emplace_back(input);
    } else {
      auto params = sub_graph->parameters();
      (void)parameters_infos.emplace_back(target_node_info,
                                          params[IntToSize(target_node_info.inplace_to_origin_input)]);
    }
  }

  auto inputs = composite_node->cast<CNodePtr>()->inputs();
  (void)inputs.insert(inputs.end(), additonal_inputs.begin(), additonal_inputs.end());
  composite_node->cast<CNodePtr>()->set_inputs(inputs);

  CreateAssignNodeAndCorrectReturn(sub_graph, parameters_infos);
  CorrectKernelBuildInfo(composite_node, info_and_inplace_assignee_addr);
}

std::vector<InplaceAssignUserInfo> InplaceAssignBuilder::FindOriginCNodeUsers(
  const AnfNodePtr &composite_node,
  const std::vector<std::pair<InplaceAssignerInfo, AnfNodePtr>> &info_and_inplace_assignee_addr,
  const FuncGraphManagerPtr &mng) const {
  std::vector<InplaceAssignUserInfo> user_node_infos;

  std::map<size_t, AnfNodePtr> real_indices_and_input_node;
  for (auto &[info, clean] : info_and_inplace_assignee_addr) {
    (void)real_indices_and_input_node.emplace(info.real_output_index, clean);
  }

  if (info_and_inplace_assignee_addr[0].first.real_output_num <= 1) {
    // Find users directly.
    auto users = mng->node_users()[composite_node];
    for (const auto &[user, index] : users) {
      user_node_infos.push_back({info_and_inplace_assignee_addr[0].second, composite_node, user, IntToSize(index)});
    }
  } else {
    std::vector<std::pair<AnfNodePtr, AnfNodePtr>> getitem_user_nodes;
    auto users = mng->node_users()[composite_node];
    for (const auto &node_index : users) {
      // 1. First, find TupleGetItem nodes.
      const auto &user_node = node_index.first;
      if (!IsPrimitiveCNode(user_node, prim::kPrimTupleGetItem)) {
        continue;
      }
      auto item_idx = GetItemIdx(user_node);
      const auto iter = real_indices_and_input_node.find(item_idx);
      if (iter != real_indices_and_input_node.end()) {
        (void)getitem_user_nodes.emplace_back(user_node, iter->second);
      }
    }
    // 2. Find users of TupleGetItem nodes.
    for (size_t i = 0; i < getitem_user_nodes.size(); ++i) {
      const auto &getitem_node = getitem_user_nodes[i].first;
      const auto &broadcast_to_node = getitem_user_nodes[i].second;
      auto real_users = mng->node_users()[getitem_node];
      for (const auto &[user, index] : real_users) {
        user_node_infos.push_back({broadcast_to_node, getitem_node, user, IntToSize(index)});
      }
    }
  }

  return user_node_infos;
}

void InplaceAssignBuilder::ProcessOriginCNodeUser(
  const FuncGraphPtr &main_graph, const AnfNodePtr &composite_node,
  const std::vector<std::pair<InplaceAssignerInfo, AnfNodePtr>> &info_and_inplace_assignee_addr,
  const FuncGraphManagerPtr &mng) const {
  // 1. Find users.
  auto user_nodes = FindOriginCNodeUsers(composite_node, info_and_inplace_assignee_addr, mng);
  for (const auto &iter : user_nodes) {
    // 2. Make sure modified composite node running first, So firstly, create depend_node, then add edge to connect
    // work_node, broadcast_node and depend_node to keep order.
    AnfNodePtrList depend_inputs = {NewValueNode(prim::kPrimDepend), iter.inplace_assignee_addr, iter.work_node};
    auto depend_node = main_graph->NewCNode(depend_inputs);
    depend_node->set_abstract(iter.inplace_assignee_addr->abstract());
    main_graph->AddNode(depend_node);
    auto user_cnode = iter.user_node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(user_cnode);
    user_cnode->set_input(iter.user_input_idx, depend_node);
  }
}
}  // namespace mindspore::graphkernel
