/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "backend/ms_backend/graph_fusion/kernel_packet/kernel_packet_common_test_suite.h"
#include "include/utils/anfalgo.h"
#include "runtime/hardware_abstract/kernel_base/graph_fusion/graph_kernel/kernel_packet/kernel_packet_infer_functor.h"

namespace mindspore::graphkernel::test {
CNodePtrList TestKernelPacket::GetAllPacketNodes(const FuncGraphPtr &fg) {
  auto cnodes = GetAllCNodes(fg);
  CNodePtrList packet_nodes;
  std::copy_if(cnodes.begin(), cnodes.end(), std::back_inserter(packet_nodes),
               [](const CNodePtr &node) { return common::AnfAlgo::HasNodeAttr(kAttrKernelPacketNode, node); });
  return packet_nodes;
}

bool TestKernelPacket::InferPacketNode(const CNodePtr &packet_node, const NodeShapeVector &input_real_shape) {
  auto sub_fg = common::AnfAlgo::GetNodeAttr<FuncGraphPtr>(packet_node, kAttrFuncGraph);
  MS_EXCEPTION_IF_NULL(sub_fg);
  MS_EXCEPTION_IF_NULL(sub_fg->symbol_engine());
  AbstractBasePtrList args;
  for (size_t i = 1; i < packet_node->size(); i++) {
    auto iter = input_real_shape.find(packet_node->input(i));
    auto ori_abs = packet_node->input(i)->abstract();
    MS_EXCEPTION_IF_NULL(ori_abs);
    if (iter == input_real_shape.end()) {
      args.push_back(ori_abs);
    } else {
      args.push_back(std::make_shared<abstract::AbstractTensor>(ori_abs->GetType(), iter->second));
    }
  }
  return sub_fg->symbol_engine()->Infer(args);
}
}  // namespace mindspore::graphkernel::test
