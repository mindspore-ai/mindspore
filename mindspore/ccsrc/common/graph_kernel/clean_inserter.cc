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

#include "common/graph_kernel/clean_inserter.h"

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
#include "backend/common/session/kernel_graph.h"
#include "common/graph_kernel/graph_kernel_helper.h"
#include "common/graph_kernel/core/graph_kernel_utils.h"

namespace mindspore::graphkernel {
namespace {
CNodePtr CreateAssign(const FuncGraphPtr &sub_graph,
                      const std::vector<std::pair<CleanZeroUserInfo, AnfNodePtr>> &parameters_infos, size_t idx) {
  if (idx >= parameters_infos.size()) {
    MS_LOG(EXCEPTION) << "idx " << idx << " is out of range [0, " << parameters_infos.size() << ")";
  }
  MS_EXCEPTION_IF_NULL(sub_graph);

  const auto &target_node = parameters_infos[idx].first.op_node;
  const auto &new_parameter = parameters_infos[idx].second;

  auto node =
    CreateCNode({NewValueNode(prim::kPrimAssign), new_parameter, target_node}, sub_graph,
                {.format = GetFormat(target_node), .shape = GetShape(target_node), .type = GetType(target_node)});
  return node;
}
}  // namespace

void CleanInserter::CorrectKernelBuildInfo(const AnfNodePtr &composite_node,
                                           const std::vector<std::pair<CleanZeroUserInfo, AnfNodePtr>> &clean_infos) {
  // Change kernel build info.
  auto kernel_info = dynamic_cast<device::KernelInfo *>(composite_node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  const auto &origin_kernel_build_info = kernel_info->GetMutableSelectKernelBuildInfo();
  MS_EXCEPTION_IF_NULL(origin_kernel_build_info);
  auto origin_inputs_format = origin_kernel_build_info->GetAllInputFormats();
  auto origin_inputs_type = origin_kernel_build_info->GetAllInputDeviceTypes();

  std::vector<std::string> &new_inputs_format = origin_inputs_format;
  std::vector<TypeId> &new_inputs_type = origin_inputs_type;
  for (const auto &clean_info : clean_infos) {
    auto &new_input = clean_info.second;
    auto kernel_with_index = common::AnfAlgo::VisitKernel(new_input, 0);
    new_inputs_format.push_back(AnfAlgo::GetOutputFormat(kernel_with_index.first, kernel_with_index.second));
    new_inputs_type.push_back(AnfAlgo::GetOutputDeviceDataType(kernel_with_index.first, kernel_with_index.second));
  }

  auto new_selected_info = BuildSelectKernelBuildInfo(
    new_inputs_format, new_inputs_type, origin_kernel_build_info->GetAllOutputFormats(),
    origin_kernel_build_info->GetAllOutputDeviceTypes(), origin_kernel_build_info->processor());
  AnfAlgo::SetSelectKernelBuildInfo(new_selected_info, composite_node.get());
}

void CleanInserter::CreateAssignNodeAndCorrectReturn(
  const FuncGraphPtr &sub_graph, const std::vector<std::pair<CleanZeroUserInfo, AnfNodePtr>> &parameters_infos) {
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
      auto iter = target_indices.find(i);
      if (iter == target_indices.end()) continue;
      auto inplace = CreateAssign(sub_graph, parameters_infos, iter->second);
      output_cnode->set_input(i, inplace);
    }
  } else if (parameters_infos.size() == 1) {
    auto inplace = CreateAssign(sub_graph, parameters_infos, 0);
    sub_graph->set_output(inplace);
  }
}

CNodePtr CleanInserter::InsertUpdateState(const FuncGraphPtr &main_graph, const AnfNodePtr &node) const {
  // Insert update_state_node, need mount a monad node.
  auto u = NewValueNode(kUMonad);
  u->set_abstract(kUMonad->ToAbstract());
  AnfNodePtrList update_state_inputs = {NewValueNode(prim::kPrimUpdateState), u, node};
  auto update_state_cnode = main_graph->NewCNode(update_state_inputs);
  update_state_cnode->set_abstract(kUMonad->ToAbstract());
  main_graph->AddNode(update_state_cnode);
  return update_state_cnode;
}

CNodePtr CleanInserter::CreateCleanCompositeNode(const CleanZeroUserInfo &op_info, const FuncGraphPtr &main_graph,
                                                 TypeId dst_type) {
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
    value_node = CreateScalarTensorValueNode<float>({.format = format, .shape = {1}, .type = TypeIdToType(dtype)},
                                                    static_cast<float>(0), sizeof(float));
  } else {
    value_node = CreateScalarTensorValueNode<double>({.format = format, .shape = {1}, .type = TypeIdToType(dtype)},
                                                     static_cast<double>(0), sizeof(double));
  }

  // Create composite op's sub-graph.
  auto new_sub_graph = std::make_shared<FuncGraph>();

  AnfNodePtr broadcast_input_node;
  if (dst_type == kNumberTypeFloat16) {
    AnfNodePtrList cast_inputs = {NewValueNode(prim::kPrimCast), value_node};
    auto cast_node_inner =
      CreateCNode(cast_inputs, new_sub_graph, {.format = format, .shape = {1}, .type = TypeIdToType(dst_type)});
    SetNodeAttrSafely("dst_type", MakeValue("float32"), cast_node_inner);
    broadcast_input_node = cast_node_inner;
  } else {
    broadcast_input_node = value_node;
  }

  // Create broadcast basic op.
  auto dst_shape_vec = GetShape(op_info.op_node);
  AnfNodePtrList clean_inputs = {NewValueNode(prim::kPrimBroadcastTo), broadcast_input_node};
  auto broadcast_to_node_inner = CreateCNode(
    clean_inputs, new_sub_graph, {.format = format, .shape = dst_shape_vec, .type = GetType(op_info.op_node)});
  SetNodeAttrSafely("shape", MakeValue(GetDeviceShape(op_info.op_node)), broadcast_to_node_inner);

  // Makeup sub-graph.
  new_sub_graph->set_output(broadcast_to_node_inner);
  auto broadcast_to_composite_node = main_graph->NewCNode({NewValueNode(new_sub_graph)});
  broadcast_to_composite_node->set_abstract(broadcast_to_node_inner->abstract());
  SetNewKernelInfo(broadcast_to_composite_node, new_sub_graph, {}, {broadcast_to_node_inner});
  auto graph_attr = GkUtils::ExtractGraphKernelName(TopoSort(new_sub_graph->get_return()), "", "atomic_clean");
  new_sub_graph->set_attr(FUNC_GRAPH_ATTR_GRAPH_KERNEL, MakeValue(graph_attr));
  new_sub_graph->set_attr("composite_type", MakeValue("atomic_clean"));

  return broadcast_to_composite_node;
}

void CleanInserter::ProcessOriginCNode(
  const AnfNodePtr &composite_node,
  const std::vector<std::pair<CleanZeroUserInfo, AnfNodePtr>> &info_and_broadcast_to_nodes, bool atomic_add_attr) {
  auto sub_graph = common::AnfAlgo::GetCNodeFuncGraphPtr(composite_node);
  auto mng_sub = sub_graph->manager();
  if (mng_sub == nullptr) {
    mng_sub = Manage(sub_graph, false);
    sub_graph->set_manager(mng_sub);
  }

  // Add input
  std::vector<std::pair<CleanZeroUserInfo, AnfNodePtr>> parameters_infos;
  for (const auto &[atomic_add_info, new_input] : info_and_broadcast_to_nodes) {
    // Add atomic attribute to target node.
    if (atomic_add_attr) SetNodeAttrSafely("enable_atomic_add", MakeValue(true), atomic_add_info.op_node);

    // add parameter
    auto parameter = sub_graph->add_parameter();
    parameter->set_abstract(new_input->abstract());
    parameter->set_kernel_info(new_input->kernel_info_ptr());
    (void)parameters_infos.emplace_back(atomic_add_info, parameter);
  }

  auto inputs = composite_node->cast<CNodePtr>()->inputs();
  (void)std::transform(info_and_broadcast_to_nodes.cbegin(), info_and_broadcast_to_nodes.cend(),
                       std::back_inserter(inputs),
                       [](const std::pair<CleanZeroUserInfo, AnfNodePtr> &pair_item) { return pair_item.second; });
  composite_node->cast<CNodePtr>()->set_inputs(inputs);

  CreateAssignNodeAndCorrectReturn(sub_graph, parameters_infos);
  CorrectKernelBuildInfo(composite_node, info_and_broadcast_to_nodes);
}
}  // namespace mindspore::graphkernel
