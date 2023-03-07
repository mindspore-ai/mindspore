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
#include "backend/common/graph_kernel/raise_reduction_precision.h"

#include <memory>
#include "mindspore/core/ops/core_ops.h"
#include "include/common/utils/utils.h"
#include "backend/common/optimizer/helper.h"
#include "backend/common/graph_kernel/core/graph_kernel_utils.h"
#include "backend/common/graph_kernel/graph_kernel_helper.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "ir/tensor.h"
#include "kernel/kernel_build_info.h"
#include "kernel/common_utils.h"
#include "include/backend/kernel_info.h"

namespace mindspore::graphkernel {
bool RaiseReductionPrecision::IsFp16ReduceSum(const AnfNodePtr &node) const {
  return IsPrimitiveCNode(node, prim::kPrimReduceSum) && AnfAlgo::GetInputDeviceDataType(node, 0) == kNumberTypeFloat16;
}

AnfNodePtr RaiseReductionPrecision::CreateCast(const AnfNodePtr &input, const TypePtr &dst_type,
                                               const std::string &format) const {
  auto func_graph = input->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  AnfNodePtrList inputs = {NewValueNode(prim::kPrimCast), input};
  auto cnode = CreateCNode(inputs, func_graph, {format, GetShape(input), dst_type});
  SetNodeAttrSafely(kAttrDstType, dst_type, cnode);
  common::AnfAlgo::SetNodeAttr(kIsBackendCast, MakeValue(true), cnode);
  return cnode;
}

AnfNodePtr RaiseReductionPrecision::CreateReduceSum(const AnfNodePtr &node, const AnfNodePtr &input) const {
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  cnode->set_input(1, input);
  cnode->set_abstract(std::make_shared<abstract::AbstractTensor>(kFloat32, GetShape(node)));
  kernel::KernelBuildInfo::KernelBuildInfoBuilder info_builder;
  info_builder.SetInputsFormat({AnfAlgo::GetInputFormat(node, 0)});
  info_builder.SetInputsDeviceType({kFloat32->type_id()});
  info_builder.SetOutputsFormat({AnfAlgo::GetOutputFormat(node, 0)});
  info_builder.SetOutputsDeviceType({kFloat32->type_id()});
  info_builder.SetProcessor(AnfAlgo::GetProcessor(node));
  info_builder.SetKernelType(KernelType::AKG_KERNEL);
  info_builder.SetFusionType(kernel::kPatternOpaque);
  AnfAlgo::SetSelectKernelBuildInfo(info_builder.Build(), cnode.get());
  return node;
}

void RaiseReductionPrecision::ReplaceNode(const AnfNodePtr &reduce_node, const AnfNodePtr &cast_node) const {
  auto mng = reduce_node->func_graph()->manager();
  MS_EXCEPTION_IF_NULL(mng);
  // use a copy of user, since the following `mng->Replace` will change the original users of reduce_node.
  auto users = mng->node_users()[reduce_node];
  for (const auto &user : users) {
    auto user_node = user.first;
    size_t user_index = IntToSize(user.second);
    if (IsPrimitiveCNode(user_node, prim::kPrimCast) &&
        AnfAlgo::GetOutputDeviceDataType(user_node, 0) == kNumberTypeFloat32) {
      if (!(mng->Replace(user_node, reduce_node))) {
        MS_LOG(ERROR) << "Fail to replace node[" << user_node->fullname_with_scope() << "] with node["
                      << reduce_node->fullname_with_scope() << "]";
      }
    } else {
      if (user_node->isa<CNode>()) {
        user_node->cast<CNodePtr>()->set_input(user_index, cast_node);
      }
    }
  }
}

bool RaiseReductionPrecision::Process(const FuncGraphPtr &func_graph) const {
  auto mng = func_graph->manager();
  if (mng == nullptr) {
    mng = Manage(func_graph, true);
    func_graph->set_manager(mng);
  }
  auto todos = TopoSort(func_graph->get_return());
  bool changed = false;
  for (auto node : todos) {
    if (IsFp16ReduceSum(node)) {
      auto cast1 = CreateCast(node->cast<CNodePtr>()->input(1), kFloat32, AnfAlgo::GetInputFormat(node, 0));
      auto new_reduce = CreateReduceSum(node, cast1);
      auto cast2 = CreateCast(new_reduce, kFloat16, AnfAlgo::GetOutputFormat(node, 0));
      ReplaceNode(node, cast2);
      changed = true;
    }
  }
  if (changed) {
    mng->RemoveRoots();
    mng->KeepRoots({func_graph});
  }
  return changed;
}

bool RaiseReductionPrecision::Run(const FuncGraphPtr &func_graph) {
  auto mng = func_graph->manager();
  if (mng == nullptr) {
    mng = Manage(func_graph, true);
    func_graph->set_manager(mng);
  }
  bool changed = false;
  auto todos = TopoSort(func_graph->get_return());
  for (const auto &node : todos) {
    if (common::AnfAlgo::IsGraphKernel(node)) {
      auto sub_func_graph = common::AnfAlgo::GetCNodeFuncGraphPtr(node);
      MS_ERROR_IF_NULL(sub_func_graph);
      changed = Process(sub_func_graph) || changed;
    }
  }
  if (changed) {
    GkUtils::UpdateFuncGraphManager(mng, func_graph);
  }
  return changed;
}
}  // namespace mindspore::graphkernel
