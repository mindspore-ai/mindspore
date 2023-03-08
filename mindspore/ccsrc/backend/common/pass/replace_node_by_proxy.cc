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
#include "backend/common/pass/replace_node_by_proxy.h"
#include <vector>
#include <memory>
#include "include/backend/kernel_info.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "kernel/kernel_build_info.h"

namespace mindspore {
namespace opt {
kernel::KernelBuildInfoPtr ReplaceNodeByProxy::GenerateKernelBuildInfo(const CNodePtr &cnode) const {
  MS_EXCEPTION_IF_NULL(cnode);
  std::vector<std::string> inputs_device_format;
  std::vector<std::string> outputs_device_format;
  std::vector<TypeId> inputs_device_type;
  std::vector<TypeId> outputs_device_type;
  kernel::KernelBuildInfo::KernelBuildInfoBuilder builder;
  size_t input_num = common::AnfAlgo::GetInputTensorNum(cnode);
  for (size_t input_index = 0; input_index < input_num; ++input_index) {
    inputs_device_format.push_back(AnfAlgo::GetInputFormat(cnode, input_index));
    inputs_device_type.push_back(AnfAlgo::GetInputDeviceDataType(cnode, input_index));
  }
  size_t output_num = AnfAlgo::GetOutputTensorNum(cnode);
  for (size_t output_index = 0; output_index < output_num; ++output_index) {
    outputs_device_format.push_back(AnfAlgo::GetOutputFormat(cnode, output_index));
    outputs_device_type.push_back(AnfAlgo::GetOutputDeviceDataType(cnode, output_index));
  }
  builder.SetFusionType(AnfAlgo::GetFusionType(cnode));
  builder.SetProcessor(AnfAlgo::GetProcessor(cnode));
  builder.SetKernelType(AnfAlgo::GetKernelType(cnode));

  builder.SetInputsFormat(inputs_device_format);
  builder.SetOutputsFormat(outputs_device_format);
  builder.SetInputsDeviceType(inputs_device_type);
  builder.SetOutputsDeviceType(outputs_device_type);
  return builder.Build();
}

bool ReplaceNodeByProxy::Run(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  auto manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  std::vector<AnfNodePtr> node_list = TopoSort(func_graph->get_return());
  for (auto node : node_list) {
    if (node != nullptr && node->isa<CNode>() && common::AnfAlgo::GetCNodeName(node) == kEmbeddingLookupOpName) {
      TraceGuard guard(std::make_shared<TraceOpt>(node->debug_info()));
      auto cnode = node->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(cnode);
      auto prim = std::make_shared<Primitive>(kEmbeddingLookupProxyOpName);
      MS_EXCEPTION_IF_NULL(prim);
      std::vector<AnfNodePtr> proxy_inputs = {NewValueNode(prim)};
      (void)proxy_inputs.insert(proxy_inputs.cend(), cnode->inputs().cbegin() + 1, cnode->inputs().cend());
      AnfNodePtr proxy_node = func_graph->NewCNode(proxy_inputs);
      MS_EXCEPTION_IF_NULL(proxy_node);

      auto kernel_info = std::make_shared<device::KernelInfo>();
      MS_EXCEPTION_IF_NULL(kernel_info);
      proxy_node->set_kernel_info(kernel_info);

      AbstractBasePtrList abstract_list;
      common::AnfAlgo::CopyNodeAttr(kAttrPsKey, cnode, proxy_node);
      common::AnfAlgo::CopyNodeAttr("offset", cnode, proxy_node);
      abstract_list.push_back(cnode->abstract());
      auto abstract_tuple = std::make_shared<abstract::AbstractTuple>(abstract_list);
      MS_EXCEPTION_IF_NULL(abstract_tuple);
      proxy_node->set_abstract(abstract_tuple);

      auto kernel_build_info = GenerateKernelBuildInfo(cnode);
      AnfAlgo::SetSelectKernelBuildInfo(kernel_build_info, proxy_node.get());

      if (!manager->Replace(cnode, proxy_node)) {
        MS_LOG(EXCEPTION) << "Replace node by proxy node failed.";
      }
    }
  }
  return true;
}
}  // namespace opt
}  // namespace mindspore
