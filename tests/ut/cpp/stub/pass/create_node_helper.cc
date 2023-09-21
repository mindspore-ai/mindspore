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
#include "plugin/device/ascend/optimizer/create_node_helper.h"

#include <memory>
#include <vector>
#include <set>
#include "ops/array_op_name.h"
#include "ops/framework_ops.h"
#include "plugin/device/ascend/optimizer/mindir/reg_ascend_vm_op_adaptation_info.h"
#include "include/common/utils/utils.h"
#include "include/common/utils/anfalgo.h"
#include "plugin/device/ascend/optimizer/ascend_helper.h"
#include "include/backend/optimizer/helper.h"
#include "plugin/device/ascend/kernel/tbe/tbe_dynamic_shape_util.h"
#include "include/backend/optimizer/op_adaptation_info_factory.h"

namespace mindspore::opt {
AnfNodePtr CreateNodeHelper::CreateNodeWithCheck(const AnfNodePtr &node, bool is_ascend_mindir) {
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
  auto ret_node = ConvertToTargetOp(cnode, op_adaptation_info);
  if ((ret_node != nullptr) && (ret_node != cnode)) {
    MS_LOG(INFO) << "Replace op " << cnode->fullname_with_scope() << " debug string:" << cnode->DebugString()
                 << " with " << ret_node->fullname_with_scope() << " debug string:" << ret_node->DebugString()
                 << ", is dynamic shape:" << common::AnfAlgo::IsDynamicShape(node);
  }
  return ret_node;
}

CNodePtr CreateNodeHelper::ConvertToTargetOp(const CNodePtr &origin_op, OpAdaptationInfo *op_adaptation_info) {
  MS_EXCEPTION_IF_NULL(origin_op);
  MS_EXCEPTION_IF_NULL(op_adaptation_info);
  auto me_op_name = op_adaptation_info->me_op_name();
  auto default_op_name = op_adaptation_info->backend_op_name();
  auto target_op_name = op_adaptation_info->target_op_name();
  auto input_to_attr_map = op_adaptation_info->input_attr_map();
  // rename to default op name
  OpAdaptationInfoRegister::RenamePrimitiveName(origin_op, me_op_name, default_op_name);
  // Check supported if the op need
  auto graph = origin_op->func_graph();
  auto kernel_graph = graph->cast<KernelGraphPtr>();

  if (!input_to_attr_map.empty()) {
    auto target_op = OpAdaptationInfoRegister::CreateTargetOp(origin_op, *op_adaptation_info);
    if (target_op == nullptr) {
      MS_LOG(DEBUG) << "Create target op failed for node " << origin_op->fullname_with_scope();
      return origin_op;
    }
    if (kernel_graph != nullptr) {
      kernel_graph->FrontBackendlMapUpdate(origin_op, target_op);
    }
    return target_op;
  }

  return origin_op;
}
}  // namespace mindspore::opt
