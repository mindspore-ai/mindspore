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

#include <vector>
#include <string>
#include <algorithm>
#include <memory>
#include <utility>
#include "mindspore/core/ops/core_ops.h"
#include "include/common/utils/utils.h"
#include "backend/common/graph_kernel/graph_kernel_helper.h"
#include "ir/tensor.h"
#include "ir/manager.h"
#include "kernel/kernel_build_info.h"
#include "kernel/common_utils.h"
#include "include/backend/kernel_info.h"
#include "backend/common/graph_kernel/decrease_transfer_precision.h"

namespace mindspore::graphkernel {
static const size_t GK_MIN_SIZE = 2;  // 2

int64_t ObtainGetItemIndex(const AnfNodePtr &getitem) {
  auto index_node = getitem->cast<CNodePtr>()->input(kInputNodeOutputIndexInTupleGetItem);
  auto value_ptr = GetValueNode(index_node);
  return GetValue<int64_t>(value_ptr);
}

bool IsPreNodeReduce(const FuncGraphPtr &, const AnfNodePtr &node, bool is_tuple_out, size_t index) {
  auto gk_graph = common::AnfAlgo::GetCNodeFuncGraphPtr(node);
  MS_EXCEPTION_IF_NULL(gk_graph);
  if (is_tuple_out) {
    auto tuple_output = gk_graph->output()->cast<CNodePtr>();
    if (common::AnfAlgo::GetCNodeName(tuple_output) != prim::kPrimMakeTuple->name()) {
      MS_LOG(EXCEPTION) << "Expect MakeTuple node, but got " << common::AnfAlgo::GetCNodeName(tuple_output);
    }
    auto input_node = tuple_output->input(index + 1);
    if (common::AnfAlgo::GetCNodeName(input_node) == prim::kPrimReduceSum->name()) {
      return true;
    }
  }
  return false;
}

size_t GetGraphKernelSize(const AnfNodePtr &node) {
  auto gk_graph = common::AnfAlgo::GetCNodeFuncGraphPtr(node);
  MS_EXCEPTION_IF_NULL(gk_graph);
  return gk_graph->GetOrderedCnodes().size();
}

bool IsCandidateNode(const AnfNodePtr &node) {
  bool is_gk = common::AnfAlgo::IsGraphKernel(node);
  if (is_gk) {
    auto num = GetGraphKernelSize(node);
    if (num > GK_MIN_SIZE) {
      auto sub_graph = common::AnfAlgo::GetCNodeFuncGraphPtr(node);
      auto graph_name = GetValue<std::string>(sub_graph->get_attr(FUNC_GRAPH_ATTR_GRAPH_KERNEL));
      if (graph_name.find("atomic") == std::string::npos) {
        return true;
      }
    }
  }
  return false;
}

bool IsAllUserCandidateNode(const AnfNodeIndexSet &users) {
  // check whether all user are graph kernel when more than one users for the in_node
  bool result = std::all_of(users.begin(), users.end(), [](const std::pair<AnfNodePtr, int> &node_index) {
    return IsCandidateNode(node_index.first);
  });
  return result;
}

bool DecreaseTransferPrecision::Run(const FuncGraphPtr &func_graph) {
  auto mng = func_graph->manager();
  if (mng == nullptr) {
    mng = Manage(func_graph, true);
    func_graph->set_manager(mng);
  }
  auto users_map = mng->node_users();
  auto todos = TopoSort(func_graph->get_return());
  bool changed = false;
  for (const auto &node : todos) {
    auto is_candidate = IsCandidateNode(node);
    if (is_candidate) {
      auto cnode = node->cast<CNodePtr>();
      for (size_t index = 1; index < cnode->size(); index++) {
        auto dtype = AnfAlgo::GetInputDeviceDataType(node, index - 1);
        if (dtype != kNumberTypeFloat32) {
          continue;
        }
        auto item = cnode->input(index);
        if (!item->cast<CNodePtr>()) {
          continue;
        }
        auto in_node = item->cast<CNodePtr>();
        if (IsPrimitive(in_node->input(0), prim::kPrimTupleGetItem)) {
          auto tuple_node = in_node->input(1);
          auto tuple_index = ObtainGetItemIndex(in_node);
          auto has_reduce_output = IsPreNodeReduce(func_graph, tuple_node, true, LongToSize(tuple_index));
          auto fail_flag = !IsCandidateNode(tuple_node) ||
                           (users_map[in_node].size() > 1 && IsAllUserCandidateNode(users_map[in_node])) ||
                           has_reduce_output;
          if (fail_flag) {
            continue;
          }
          // mutate father
          (void)ProcessFather(func_graph, tuple_node, true, LongToSize(tuple_index));
          in_node->set_abstract(std::make_shared<abstract::AbstractTensor>(kFloat16, GetShape(in_node)));
          // mutate sons
          for (auto each_out : users_map[in_node]) {
            (void)ProcessSon(func_graph, each_out.first, IntToSize(each_out.second));
          }
        }
        if (IsCandidateNode(in_node)) {
          auto fail_flag = !IsAllUserCandidateNode(users_map[in_node]);
          if (fail_flag) {
            continue;
          }
          // mutate father
          (void)ProcessFather(func_graph, in_node, false, 0);
          // mutate sons
          (void)ProcessSon(func_graph, cnode, index);
        }
      }
    }
  }
  return changed;
}

bool DecreaseTransferPrecision::ProcessFather(const FuncGraphPtr &, const AnfNodePtr &node, bool is_tuple_out,
                                              size_t index) const {
  auto gk_graph = common::AnfAlgo::GetCNodeFuncGraphPtr(node);
  MS_EXCEPTION_IF_NULL(gk_graph);
  auto mng = gk_graph->manager();
  MS_EXCEPTION_IF_NULL(mng);

  // lambda func for cast fp32 to fp16
  auto func_add_cast_fp16 = [&gk_graph](const AnfNodePtr &old_output) {
    AnfNodePtrList inputs = {NewValueNode(prim::kPrimCast), old_output};
    auto cnode = gk_graph->NewCNode(inputs);
    MS_EXCEPTION_IF_NULL(cnode);
    gk_graph->AddNode(cnode);
    cnode->set_abstract(std::make_shared<abstract::AbstractTensor>(kFloat16, GetShape(old_output)));
    cnode->set_scope(old_output->scope());
    SetNodeAttrSafely(kAttrDstType, kFloat16, cnode);
    cnode->set_kernel_info(std::make_shared<device::KernelInfo>());
    std::vector<std::string> cnode_input_format = {AnfAlgo::GetOutputFormat(old_output, 0)};
    std::vector<TypeId> cnode_input_type = {kNumberTypeFloat32};
    std::vector<std::string> cnode_output_format = {AnfAlgo::GetOutputFormat(old_output, 0)};
    std::vector<TypeId> cnode_output_type = {kNumberTypeFloat16};
    kernel::KernelBuildInfo::KernelBuildInfoBuilder graph_info_builder;
    graph_info_builder.SetInputsFormat(cnode_input_format);
    graph_info_builder.SetInputsDeviceType(cnode_input_type);
    graph_info_builder.SetOutputsFormat(cnode_output_format);
    graph_info_builder.SetOutputsDeviceType(cnode_output_type);
    graph_info_builder.SetProcessor(kernel::GetProcessorFromContext());
    graph_info_builder.SetKernelType(KernelType::AKG_KERNEL);
    graph_info_builder.SetFusionType(kernel::kPatternOpaque);
    auto info_1 = graph_info_builder.Build();
    AnfAlgo::SetSelectKernelBuildInfo(info_1, cnode.get());
    return cnode;
  };

  if (!is_tuple_out) {
    auto old_output = gk_graph->output()->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(old_output);
    if (common::AnfAlgo::GetCNodeName(old_output) == prim::kPrimCast->name() &&
        AnfAlgo::GetInputDeviceDataType(old_output, 0) == kNumberTypeFloat16 &&
        AnfAlgo::GetOutputDeviceDataType(old_output, 0) == kNumberTypeFloat32) {
      auto real_output = old_output->input(1);
      gk_graph->set_output(real_output);
    } else {
      auto cnode = func_add_cast_fp16(old_output);
      gk_graph->set_output(cnode);
    }

    // get kernel build info
    node->set_abstract(std::make_shared<abstract::AbstractTensor>(kFloat16, GetShape(node)));
    auto gk_builder_info =
      std::make_shared<kernel::KernelBuildInfo::KernelBuildInfoBuilder>(AnfAlgo::GetSelectKernelBuildInfo(node));
    std::vector<TypeId> gk_output_type = {kNumberTypeFloat16};
    gk_builder_info->SetOutputsDeviceType(gk_output_type);
    AnfAlgo::SetSelectKernelBuildInfo(gk_builder_info->Build(), node.get());
    return true;
  } else {
    // cast for graph kernel with make tuple output
    auto tuple_output = gk_graph->output()->cast<CNodePtr>();
    if (common::AnfAlgo::GetCNodeName(tuple_output) != prim::kPrimMakeTuple->name()) {
      MS_LOG(EXCEPTION) << "Expect MakeTuple node, but got " << common::AnfAlgo::GetCNodeName(tuple_output);
    }
    auto input_node = tuple_output->input(index + 1);
    auto cnode = func_add_cast_fp16(input_node);
    tuple_output->set_input(index + 1, cnode);

    // Update MakeTuple node abstract
    AbstractBasePtrList abstract_list;
    for (size_t i = 1; i < tuple_output->size(); ++i) {
      (void)abstract_list.emplace_back(tuple_output->input(i)->abstract());
    }
    tuple_output->set_abstract(std::make_shared<abstract::AbstractTuple>(abstract_list));

    // Update Graph Kernel abstract
    node->set_abstract(tuple_output->abstract());

    // Update Graph Kernel Build Kernel Info
    auto old_builder_info = AnfAlgo::GetSelectKernelBuildInfo(node);
    auto gk_builder_info = std::make_shared<kernel::KernelBuildInfo::KernelBuildInfoBuilder>(old_builder_info);
    auto origin_outputs_type = old_builder_info->GetAllOutputDeviceTypes();
    std::vector<TypeId> gk_output_type;
    for (size_t i = 0; i < origin_outputs_type.size(); ++i) {
      gk_output_type.push_back(origin_outputs_type[i]);
    }
    gk_output_type[index] = kNumberTypeFloat16;
    gk_builder_info->SetOutputsDeviceType(gk_output_type);
    AnfAlgo::SetSelectKernelBuildInfo(gk_builder_info->Build(), node.get());

    return true;
  }
}

bool DecreaseTransferPrecision::ProcessSon(const FuncGraphPtr &, const AnfNodePtr &node, size_t index) const {
  auto gk_graph = common::AnfAlgo::GetCNodeFuncGraphPtr(node);
  MS_EXCEPTION_IF_NULL(gk_graph);
  auto mng = gk_graph->manager();
  MS_EXCEPTION_IF_NULL(mng);
  auto old_input = gk_graph->get_inputs()[index - 1];
  MS_EXCEPTION_IF_NULL(old_input);

  auto user_nodes = mng->node_users()[old_input];
  // get kernel build info
  auto gk_builder_info =
    std::make_shared<kernel::KernelBuildInfo::KernelBuildInfoBuilder>(AnfAlgo::GetSelectKernelBuildInfo(node));
  auto ori_input_format = AnfAlgo::GetAllInputDeviceTypes(node);
  std::vector<TypeId> &new_inputs_type = ori_input_format;
  new_inputs_type[index - 1] = kNumberTypeFloat16;
  gk_builder_info->SetInputsDeviceType(new_inputs_type);
  AnfAlgo::SetSelectKernelBuildInfo(gk_builder_info->Build(), node.get());
  AbstractBasePtr old_abstract = node->abstract()->Clone();
  node->set_abstract(old_abstract);

  for (const auto &user : user_nodes) {
    auto user_node = user.first;
    if (IsPrimitiveCNode(user_node, prim::kPrimCast) &&
        AnfAlgo::GetOutputDeviceDataType(user_node, 0) == kNumberTypeFloat16) {
      (void)mng->Replace(user_node, old_input);
      return true;
    }
  }

  auto tensor_input = node->cast<CNodePtr>()->input(index);
  AnfNodePtrList inputs = {NewValueNode(prim::kPrimCast), old_input};
  auto cnode = gk_graph->NewCNode(inputs);
  gk_graph->AddNode(cnode);
  cnode->set_abstract(old_input->abstract());
  cnode->set_scope(old_input->scope());
  SetNodeAttrSafely(kAttrDstType, kFloat32, cnode);
  MS_EXCEPTION_IF_NULL(cnode);
  old_input->set_abstract(std::make_shared<abstract::AbstractTensor>(kFloat16, GetShape(old_input)));
  cnode->set_kernel_info(std::make_shared<device::KernelInfo>());
  std::vector<std::string> cnode_input_format = {AnfAlgo::GetOutputFormat(tensor_input, 0)};
  std::vector<TypeId> cnode_input_type = {kNumberTypeFloat16};
  std::vector<std::string> cnode_output_format = {AnfAlgo::GetOutputFormat(tensor_input, 0)};
  std::vector<TypeId> cnode_output_type = {kNumberTypeFloat32};
  kernel::KernelBuildInfo::KernelBuildInfoBuilder node_info_builder;
  node_info_builder.SetInputsFormat(cnode_input_format);
  node_info_builder.SetInputsDeviceType(cnode_input_type);
  node_info_builder.SetOutputsFormat(cnode_output_format);
  node_info_builder.SetOutputsDeviceType(cnode_output_type);
  node_info_builder.SetProcessor(kernel::GetProcessorFromContext());
  node_info_builder.SetKernelType(KernelType::AKG_KERNEL);
  node_info_builder.SetFusionType(kernel::kPatternOpaque);
  auto info_1 = node_info_builder.Build();
  AnfAlgo::SetSelectKernelBuildInfo(info_1, cnode.get());
  (void)mng->Replace(old_input, cnode);
  return true;
}
}  // namespace mindspore::graphkernel
