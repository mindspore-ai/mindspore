/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "plugin/device/ascend/optimizer/mindir/ascend_mindir_op_adapter.h"
#include "plugin/device/ascend/optimizer/mindir/reg_ascend_vm_op_adaptation_info.h"
#include "include/common/utils/anfalgo.h"
#include "include/backend/anf_runtime_algorithm.h"

namespace mindspore {
namespace opt {
namespace {
AnfNodePtr CreateNodeWithCheck(const AnfNodePtr &node, const KernelGraphPtr kernel_graph,
                               bool is_ascend_mindir = false) {
  if (!node || !AnfUtils::IsRealCNodeKernel(node)) {
    return node;
  }
  auto op_name = common::AnfAlgo::GetCNodeName(node);
  auto op_adaptation_info = OpAdaptationInfoRegister::GetOpAdaptationInfo(op_name, kAscendDevice, true);
  if (!op_adaptation_info || op_adaptation_info->is_ascend_mindir() != is_ascend_mindir) {
    return node;
  }
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);

  auto me_op_name = op_adaptation_info->me_op_name();
  auto backend_op_name = op_adaptation_info->backend_op_name();
  auto pre_check_func = op_adaptation_info->pre_check_func();
  auto input_to_attr_map = op_adaptation_info->input_attr_map();
  // Step1: rename to default op name
  common::AnfAlgo::SetNodeAttr(kAttrMeOpName, MakeValue(me_op_name), cnode);
  OpAdaptationInfoRegister::RenamePrimitiveName(cnode, me_op_name, backend_op_name);
  // Step2: pre_check
  if (pre_check_func) {
    if (!pre_check_func(cnode)) {
      MS_LOG(DEBUG) << "Pre check function return Not Change for op " << cnode->fullname_with_scope();
      return cnode;
    }
  }
  if (!input_to_attr_map.empty()) {
    auto target_op = OpAdaptationInfoRegister::CreateTargetOp(cnode, *op_adaptation_info);
    if (target_op == nullptr) {
      MS_LOG(DEBUG) << "Create target op failed for node " << cnode->fullname_with_scope();
      return cnode;
    }
    if (kernel_graph != nullptr) {
      kernel_graph->FrontBackendlMapUpdate(cnode, target_op);
    }
    return target_op;
  }
  return node;
}
}  // namespace

constexpr auto kHasRunMindIR = "HasRunMindIR";
const AnfNodePtr AscendMindIROpAdapter::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                                const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  if (common::AnfAlgo::HasNodeAttr(kHasRunMindIR, cnode)) {
    return node;
  }
  // There are other UnifyMindIR pass before AscendMindIROpAdapter which may create new nodes.
  if (graph->has_flag(kAttrMutableKernel) && AnfUtils::IsRealCNodeKernel(node)) {
    AnfAlgo::SetDynamicAttrToPrim(common::AnfAlgo::GetCNodePrimitive(node));
  }
  auto ret_node = CreateNodeWithCheck(node, graph->cast<KernelGraphPtr>(), true);
  if (ret_node != node) {
    common::AnfAlgo::SetNodeAttr(kHasRunMindIR, MakeValue(true), ret_node);
  }
  return ret_node;
}
}  // namespace opt
}  // namespace mindspore
