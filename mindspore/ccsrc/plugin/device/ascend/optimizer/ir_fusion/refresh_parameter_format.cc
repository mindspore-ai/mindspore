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

#include "plugin/device/ascend/optimizer/ir_fusion/refresh_parameter_format.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "include/common/utils/utils.h"
#include "include/backend/kernel_info.h"
#include "backend/common/optimizer/helper.h"

namespace mindspore {
namespace opt {
void DoRefresh(const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  size_t input_num = common::AnfAlgo::GetInputTensorNum(cnode);
  for (size_t input_index = 0; input_index < input_num; input_index++) {
    auto input_kernel_node = common::AnfAlgo::VisitKernel(common::AnfAlgo::GetInputNode(cnode, input_index), 0).first;
    MS_EXCEPTION_IF_NULL(input_kernel_node);
    if (input_kernel_node->isa<Parameter>()) {
      std::shared_ptr<kernel::KernelBuildInfo::KernelBuildInfoBuilder> builder =
        std::make_shared<kernel::KernelBuildInfo::KernelBuildInfoBuilder>();
      MS_EXCEPTION_IF_NULL(builder);
      auto cnode_input_format = AnfAlgo::GetInputFormat(cnode, input_index);
      auto kernel_node_format = AnfAlgo::GetOutputFormat(input_kernel_node, 0);
      auto dtype = AnfAlgo::GetOutputDeviceDataType(input_kernel_node, 0);
      if (kernel_node_format != cnode_input_format) {
        builder->SetOutputsFormat({cnode_input_format});
        builder->SetOutputsDeviceType({dtype});
        AnfAlgo::SetSelectKernelBuildInfo(builder->Build(), input_kernel_node.get());
      }
    }
  }
}

bool RefreshParameterFormat::Run(const FuncGraphPtr &func_graph) {
  if (func_graph == nullptr) {
    MS_LOG(WARNING) << "func_graph is nullptr.";
    return false;
  }
  std::vector<AnfNodePtr> node_list = TopoSort(func_graph->get_return());
  for (auto node : node_list) {
    if (node == nullptr || !node->isa<CNode>()) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    if (cnode == nullptr) {
      continue;
    }
    auto node_name = common::AnfAlgo::GetCNodeName(cnode);
    if (node_name == kBNTrainingUpdateOpName) {
      DoRefresh(cnode);
    }
  }
  return true;
}
}  // namespace opt
}  // namespace mindspore
