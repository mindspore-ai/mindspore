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
#include "backend/common/graph_kernel/add_stitch_atomic_clean_gpu.h"

#include <algorithm>
#include <string>
#include "mindspore/core/ops/core_ops.h"
#include "ir/tensor.h"
#include "include/common/utils/utils.h"
#include "utils/log_adapter.h"
#include "kernel/kernel.h"
#include "kernel/common_utils.h"
#include "backend/common/graph_kernel/graph_kernel_helper.h"
#include "backend/common/graph_kernel/core/graph_kernel_utils.h"
#include "include/backend/kernel_graph.h"

namespace mindspore::graphkernel {
void StitchAtomicCleanInserter::CorrectKernelBuildInfo(
  const AnfNodePtr &composite_node, const std::vector<std::pair<InplaceAssignerInfo, AnfNodePtr>> &inplace_infos) {
  // Change kernel build info.
  auto kernel_info = dynamic_cast<device::KernelInfo *>(composite_node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  const auto &origin_kernel_build_info = kernel_info->GetMutableSelectKernelBuildInfo();
  auto origin_outputs_format = origin_kernel_build_info->GetAllOutputFormats();
  auto origin_outputs_type = origin_kernel_build_info->GetAllOutputDeviceTypes();
  auto origin_processor = origin_kernel_build_info->processor();

  std::vector<std::string> new_inputs_format = origin_kernel_build_info->GetAllInputFormats();
  std::vector<TypeId> new_inputs_type = origin_kernel_build_info->GetAllInputDeviceTypes();
  std::vector<std::string> new_outputs_format;
  std::vector<TypeId> new_outputs_type;
  for (size_t i = 0; i < origin_outputs_format.size(); ++i) {
    new_outputs_format.push_back(origin_outputs_format[i]);
    new_outputs_type.push_back(origin_outputs_type[i]);
  }

  auto kernel_with_index = common::AnfAlgo::VisitKernel(inplace_infos[0].second, 0);
  new_inputs_format.push_back(AnfAlgo::GetOutputFormat(kernel_with_index.first, kernel_with_index.second));
  new_inputs_type.push_back(AnfAlgo::GetOutputDeviceDataType(kernel_with_index.first, kernel_with_index.second));

  auto new_selected_info = BuildSelectKernelBuildInfo(new_inputs_format, new_inputs_type, new_outputs_format,
                                                      new_outputs_type, origin_processor);
  AnfAlgo::SetSelectKernelBuildInfo(new_selected_info, composite_node.get());
}

void StitchAtomicCleanInserter::AddDepend(const FuncGraphPtr &main_graph, const AnfNodePtr &clean_node,
                                          const AnfNodePtr &composite_node, const AnfNodePtr &user_node,
                                          int index) const {
  // Create depend node to hold execution order.
  AnfNodePtrList d_inputs = {NewValueNode(prim::kPrimDepend), clean_node, composite_node};
  auto depend_cnode = main_graph->NewCNode(d_inputs);
  depend_cnode->set_abstract(clean_node->abstract());
  main_graph->AddNode(depend_cnode);

  auto user_cnode = user_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(user_cnode);
  user_cnode->set_input(IntToSize(index), depend_cnode);
}

CNodePtr StitchAtomicCleanInserter::CreateAssignNode(const FuncGraphPtr &sub_graph, const AnfNodePtr &new_parameter,
                                                     const InplaceAssignerInfo &info) const {
  // add assign
  AnfNodePtr out_node = info.op_node;  // Use result data itself

  auto assign_node = CreateCNode({NewValueNode(prim::kPrimAssign), new_parameter, out_node}, sub_graph,
                                 {GetFormat(out_node), GetShape(out_node), GetType(out_node)});
  common::AnfAlgo::EraseNodeAttr(kAttrStitch, out_node);
  SetNodeAttrSafely(kAttrStitch, MakeValue("common"), assign_node);
  return assign_node;
}

void StitchAtomicCleanInserter::ProcessOriginCNode(
  const AnfNodePtr &composite_node,
  const std::vector<std::pair<InplaceAssignerInfo, AnfNodePtr>> &info_and_inplace_assignee_addr) {
  auto sub_graph = common::AnfAlgo::GetCNodeFuncGraphPtr(composite_node);
  auto mng_sub = sub_graph->manager();
  if (mng_sub == nullptr) {
    mng_sub = Manage(sub_graph, false);
    sub_graph->set_manager(mng_sub);
  }

  auto [atomic_add_info, new_input] = info_and_inplace_assignee_addr[0];

  // add input
  auto inputs = composite_node->cast<CNodePtr>()->inputs();
  inputs.push_back(new_input);
  composite_node->cast<CNodePtr>()->set_inputs(inputs);

  // add parameter
  auto parameter = sub_graph->add_parameter();
  parameter->set_abstract(new_input->abstract());
  parameter->set_kernel_info(new_input->kernel_info_ptr());

  auto assign = CreateAssignNode(sub_graph, parameter, atomic_add_info);

  // Replace atomic ReduceSum's user with atomic clean output, and add depend op after assign to avoid
  // elimination.
  std::vector<std::pair<AnfNodePtr, int>> reduce_user_nodes =
    FindInnerCNodeUsers(stitch_node_, atomic_add_info.op_node);
  bool connected = false;
  for (const auto &[user_node, index] : reduce_user_nodes) {
    auto user_cnode = user_node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(user_cnode);
    user_cnode->set_input(IntToSize(index), parameter);
    if (!connected) {
      std::vector<std::pair<AnfNodePtr, int>> user_user = FindInnerCNodeUsers(stitch_node_, user_cnode);
      if (!user_user.empty()) {
        auto pair = user_user[0];
        AddDepend(sub_graph, user_cnode, assign, pair.first, pair.second);
      }
      connected = true;
    }
    CorrectKernelBuildInfo(composite_node, info_and_inplace_assignee_addr);
  }

  auto old_graph_name = GetValue<std::string>(sub_graph->get_attr(FUNC_GRAPH_ATTR_GRAPH_KERNEL));
  auto new_graph_name = GkUtils::ExtractGraphKernelName(TopoSort(sub_graph->get_return()), "", "atomic_add");
  sub_graph->set_attr(FUNC_GRAPH_ATTR_GRAPH_KERNEL, MakeValue(new_graph_name));
  MS_LOG(INFO) << "Convert " << old_graph_name << " to atomic add graph " << new_graph_name;
}

std::vector<std::pair<AnfNodePtr, int>> StitchAtomicCleanInserter::FindInnerCNodeUsers(const AnfNodePtr &inner_node,
                                                                                       const CNodePtr &target) const {
  auto node = inner_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(node);
  auto sub_graph = common::AnfAlgo::GetCNodeFuncGraphPtr(node);
  auto mng_sub = sub_graph->manager();
  if (mng_sub == nullptr) {
    mng_sub = Manage(sub_graph, false);
    sub_graph->set_manager(mng_sub);
  }
  std::vector<std::pair<AnfNodePtr, int>> inner_user_nodes;
  auto users = mng_sub->node_users()[target];
  (void)std::transform(users.cbegin(), users.cend(), std::back_inserter(inner_user_nodes),
                       [](const std::pair<AnfNodePtr, int> &pair) { return pair; });
  return inner_user_nodes;
}

std::pair<bool, InplaceAssignerInfo> StitchAtomicCleanInserter::IsStitchWithAtomic(const AnfNodePtr &anf_node) {
  if (!common::AnfAlgo::IsGraphKernel(anf_node)) {
    return {false, InplaceAssignerInfo()};
  }
  auto node = anf_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(node);
  auto sub_graph = common::AnfAlgo::GetCNodeFuncGraphPtr(node);
  AnfNodePtrList kernel_nodes;
  kernel::GetValidKernelNodes(sub_graph, &kernel_nodes);
  for (auto &n : kernel_nodes) {
    if (common::AnfAlgo::HasNodeAttr(kAttrStitch, n->cast<CNodePtr>()) &&
        common::AnfAlgo::GetNodeAttr<std::string>(n, kAttrStitch) == "atomic" &&
        IsPrimitiveCNode(n, prim::kPrimReduceSum)) {
      MS_LOG(INFO) << "GOT STITCH WITH ATOMIC!!!";
      InplaceAssignerInfo info;
      info.op_node = n->cast<CNodePtr>();
      stitch_node_ = anf_node;
      return {true, info};
    }
  }
  return {false, InplaceAssignerInfo()};
}

bool StitchAtomicCleanInserter::Run(const FuncGraphPtr &func_graph) {
  auto kernel_graph = std::dynamic_pointer_cast<session::KernelGraph>(func_graph);
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto mng = kernel_graph->manager();
  if (mng == nullptr) {
    mng = Manage(kernel_graph, true);
    kernel_graph->set_manager(mng);
  }

  bool changed = false;
  auto topo_nodes = TopoSort(kernel_graph->get_return());
  for (const auto &node : topo_nodes) {
    // if stitch attr exists, add atomic clean op depends on the attr
    auto [is_stitch, atomic_add_info] = IsStitchWithAtomic(node);
    if (is_stitch) {
      InsertAtomicClean(kernel_graph, node, {atomic_add_info}, mng);
      changed = true;
    }
  }

  if (changed) {
    GkUtils::UpdateFuncGraphManager(mng, func_graph);
  }

  return changed;
}
}  // namespace mindspore::graphkernel
