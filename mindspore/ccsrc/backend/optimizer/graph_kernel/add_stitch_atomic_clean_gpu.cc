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

#include "backend/optimizer/graph_kernel/add_stitch_atomic_clean_gpu.h"
#include <algorithm>
#include <functional>
#include <list>
#include <map>
#include <memory>
#include <utility>
#include <set>
#include <stack>
#include <string>
#include <tuple>
#include <vector>
#include "base/core_ops.h"
#include "ir/tensor.h"
#include "utils/utils.h"
#include "utils/log_adapter.h"
#include "backend/kernel_compiler/kernel.h"
#include "backend/kernel_compiler/common_utils.h"
#include "backend/optimizer/graph_kernel/graph_kernel_helper.h"
#include "backend/session/anf_runtime_algorithm.h"
#include "backend/session/kernel_graph.h"
#include "debug/anf_ir_dump.h"

namespace mindspore {
namespace opt {
void StitchAtomicCleanInsertter::CorrectKernelBuildInfo(const AnfNodePtr &composite_node, const AnfNodePtr &new_input) {
  // Change kernel build info.
  auto kernel_info = static_cast<device::KernelInfo *>(composite_node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  const auto &origin_kernel_build_info = kernel_info->GetMutableSelectKernelBuildInfo();
  auto origin_inputs_format = origin_kernel_build_info->GetAllInputFormats();
  auto origin_outputs_format = origin_kernel_build_info->GetAllOutputFormats();
  auto origin_inputs_type = origin_kernel_build_info->GetAllInputDeviceTypes();
  auto origin_outputs_type = origin_kernel_build_info->GetAllOutputDeviceTypes();
  auto origin_processor = origin_kernel_build_info->processor();

  std::vector<std::string> &new_inputs_format = origin_inputs_format;
  std::vector<TypeId> &new_inputs_type = origin_inputs_type;
  std::vector<std::string> new_outputs_format;
  std::vector<TypeId> new_outputs_type;
  for (size_t i = 0; i < origin_outputs_format.size(); ++i) {
    new_outputs_format.push_back(origin_outputs_format[i]);
    new_outputs_type.push_back(origin_outputs_type[i]);
  }

  auto kernel_with_index = AnfAlgo::VisitKernel(new_input, 0);
  new_inputs_format.push_back(AnfAlgo::GetOutputFormat(kernel_with_index.first, kernel_with_index.second));
  new_inputs_type.push_back(AnfAlgo::GetOutputDeviceDataType(kernel_with_index.first, kernel_with_index.second));

  kernel::KernelBuildInfo::KernelBuildInfoBuilder new_info_builder;
  new_info_builder.SetInputsFormat(new_inputs_format);
  new_info_builder.SetInputsDeviceType(new_inputs_type);
  new_info_builder.SetOutputsFormat(new_outputs_format);
  new_info_builder.SetOutputsDeviceType(new_outputs_type);
  new_info_builder.SetProcessor(origin_processor);
  new_info_builder.SetKernelType(KernelType::AKG_KERNEL);
  new_info_builder.SetFusionType(kernel::FusionType::OPAQUE);
  auto new_selected_info = new_info_builder.Build();
  AnfAlgo::SetSelectKernelBuildInfo(new_selected_info, composite_node.get());
}

CNodePtr StitchAtomicCleanInsertter::CreateInplaceAssignNodeAndCorrectReturn(const FuncGraphPtr &sub_graph,
                                                                             const AnfNodePtr &new_parameter) {
  // add inplaceassign
  AnfNodePtr out_node = atomic_add_node_;  // Use result data itself, and set attr "fake_out" true.
  auto inplace_assign_node =
    CreateCNode({NewValueNode(prim::kPrimInplaceAssign), new_parameter, atomic_add_node_, out_node}, sub_graph,
                {.format = GetFormat(out_node), .shape = GetShape(out_node), .type = GetType(out_node)});
  SetNodeAttrSafely("fake_output", MakeValue(true), inplace_assign_node);
  AnfAlgo::EraseNodeAttr(kAttrStitch, atomic_add_node_);
  SetNodeAttrSafely(kAttrStitch, MakeValue("common"), inplace_assign_node);
  return inplace_assign_node;
}

void StitchAtomicCleanInsertter::ProcessOriginCNode(const KernelGraphPtr &main_graph, const AnfNodePtr &composite_node,
                                                    const AnfNodePtr &new_input, const FuncGraphManagerPtr &mng) {
  auto sub_graph = AnfAlgo::GetCNodeFuncGraphPtr(composite_node);
  auto mng_sub = sub_graph->manager();
  if (mng_sub == nullptr) {
    mng_sub = Manage(sub_graph, false);
    sub_graph->set_manager(mng_sub);
  }

  // add input
  auto inputs = composite_node->cast<CNodePtr>()->inputs();
  inputs.push_back(new_input);
  composite_node->cast<CNodePtr>()->set_inputs(inputs);

  // add parameter
  auto parameter = sub_graph->add_parameter();
  parameter->set_abstract(new_input->abstract());
  parameter->set_kernel_info(new_input->kernel_info_ptr());

  auto inplace_assign = CreateInplaceAssignNodeAndCorrectReturn(sub_graph, parameter);

  // Replace atomic ReduceSum's user with atomic clean output, and add depend op after inplaceassign to avoid
  // elimination.
  std::vector<std::pair<AnfNodePtr, int>> reduce_user_nodes = FindInnerCNodeUsers(stitch_node_, atomic_add_node_);
  bool connected = false;
  for (const auto &[user_node, index] : reduce_user_nodes) {
    auto user_cnode = user_node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(user_cnode);
    user_cnode->set_input(index, parameter);
    if (!connected) {
      std::vector<std::pair<AnfNodePtr, int>> user_user = FindInnerCNodeUsers(stitch_node_, user_cnode);
      if (!user_user.empty()) {
        auto pair = user_user[0];
        AddDepend(sub_graph, user_cnode, inplace_assign, pair.first, pair.second);
      }
      connected = true;
    }
    CorrectKernelBuildInfo(composite_node, new_input);
  }

  auto old_graph_name = GetValue<std::string>(sub_graph->get_attr(FUNC_GRAPH_ATTR_GRAPH_KERNEL));
  auto new_graph_name = ExtractGraphKernelName(TopoSort(sub_graph->get_return()), "", "atomic_add");
  sub_graph->set_attr(FUNC_GRAPH_ATTR_GRAPH_KERNEL, MakeValue(new_graph_name));
  MS_LOG(INFO) << "Convert " << old_graph_name << " to atomic add graph " << new_graph_name;
}

std::vector<std::pair<AnfNodePtr, int>> StitchAtomicCleanInsertter::FindInnerCNodeUsers(const AnfNodePtr &inner_node,
                                                                                        const CNodePtr &target) {
  auto node = inner_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(node);
  auto sub_graph = AnfAlgo::GetCNodeFuncGraphPtr(node);
  auto mng_sub = sub_graph->manager();
  if (mng_sub == nullptr) {
    mng_sub = Manage(sub_graph, false);
    sub_graph->set_manager(mng_sub);
  }
  std::vector<std::pair<AnfNodePtr, int>> inner_user_nodes;
  auto users = mng_sub->node_users()[target];
  std::transform(users.cbegin(), users.cend(), std::back_inserter(inner_user_nodes),
                 [](const std::pair<AnfNodePtr, int> &pair) { return pair; });
  return inner_user_nodes;
}

bool StitchAtomicCleanInsertter::IsStitchWithAtomic(const AnfNodePtr &anf_node) {
  if (!AnfAlgo::IsGraphKernel(anf_node)) return false;
  auto node = anf_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(node);
  auto sub_graph = AnfAlgo::GetCNodeFuncGraphPtr(node);
  AnfNodePtrList kernel_nodes;
  kernel::GetValidKernelNodes(sub_graph, &kernel_nodes);
  for (auto &n : kernel_nodes) {
    if (AnfAlgo::HasNodeAttr(kAttrStitch, n->cast<CNodePtr>()) &&
        AnfAlgo::GetNodeAttr<std::string>(n, kAttrStitch) == "atomic" && IsPrimitiveCNode(n, prim::kPrimReduceSum)) {
      MS_LOG(INFO) << "GOT STITCH WITH ATOMIC!!!";
      atomic_add_node_ = n->cast<CNodePtr>();
      stitch_node_ = anf_node;
      return true;
    }
  }
  return false;
}

bool StitchAtomicCleanInsertter::Run(const FuncGraphPtr &func_graph) {
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
    if (IsStitchWithAtomic(node)) {
      InsertAtomicClean(kernel_graph, node, mng);
      changed = true;
    }
  }

  if (changed) {
    mng->RemoveRoots();
    mng->KeepRoots({func_graph});
  }

  return changed;
}
}  // namespace opt
}  // namespace mindspore
