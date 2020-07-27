/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "backend/optimizer/ascend/format_type/split_unsupported_transdata.h"
#include <vector>
#include <memory>
#include "backend/session/anf_runtime_algorithm.h"

namespace mindspore {
namespace opt {
const BaseRef SplitUnsupportedTransData::DefinePattern() const {
  VarPtr X = std::make_shared<Var>();
  return VectorRef({prim::KPrimTransData, X});
}

const AnfNodePtr SplitUnsupportedTransData::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                    const EquivPtr &) const {
  if (node == nullptr || !node->isa<CNode>() || !AnfAlgo::IsRealKernel(node)) {
    return nullptr;
  }
  auto ori_trans_data = node->cast<CNodePtr>();
  if (AnfAlgo::GetCNodeName(ori_trans_data) != prim::KPrimTransData->name()) {
    return nullptr;
  }
  auto kernel_info = AnfAlgo::GetSelectKernelBuildInfo(ori_trans_data);
  MS_EXCEPTION_IF_NULL(kernel_info);
  if (kernel_info->GetInputNum() != 1 || kernel_info->GetOutputNum() != 1) {
    MS_LOG(EXCEPTION) << "Transdata node's kernel info's input and output format size is not 1"
                      << ori_trans_data->DebugString();
  }
  return SplitTransData(func_graph, ori_trans_data);
}
AnfNodePtr SplitUnsupportedTransData::SplitTransData(const FuncGraphPtr &func_graph, const CNodePtr &trans_node) const {
  auto kernel_info = AnfAlgo::GetSelectKernelBuildInfo(trans_node);
  if (kHWSpecialFormatSet.find(kernel_info->GetInputFormat(0)) == kHWSpecialFormatSet.end() ||
      kHWSpecialFormatSet.find(kernel_info->GetOutputFormat(0)) == kHWSpecialFormatSet.end()) {
    return trans_node;
  }
  auto builder_info_to_default = std::make_shared<kernel::KernelBuildInfo::KernelBuildInfoBuilder>(kernel_info);
  auto builder_info_to_special_foramt = std::make_shared<kernel::KernelBuildInfo::KernelBuildInfoBuilder>(kernel_info);
  builder_info_to_default->SetOutputsFormat({kOpFormat_DEFAULT});
  builder_info_to_special_foramt->SetInputsFormat({kOpFormat_DEFAULT});
  std::vector<AnfNodePtr> next_trans_node_inputs = {
    NewValueNode(std::make_shared<Primitive>(prim::KPrimTransData->name())), trans_node};
  auto next_trans_node = func_graph->NewCNode(next_trans_node_inputs);
  next_trans_node->set_abstract(trans_node->abstract());
  AnfAlgo::SetSelectKernelBuildInfo(builder_info_to_default->Build(), trans_node.get());
  AnfAlgo::SetSelectKernelBuildInfo(builder_info_to_special_foramt->Build(), next_trans_node.get());
  return next_trans_node;
}
}  // namespace opt
}  // namespace mindspore
