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
#include "mindspore/core/ops/framework_ops.h"
#include "plugin/device/ascend/optimizer/mindir/reg_ascend_vm_op_adaptation_info.h"
#include "include/common/utils/utils.h"
#include "include/common/utils/anfalgo.h"
#include "plugin/device/ascend/optimizer/ascend_helper.h"
#include "include/backend/optimizer/helper.h"

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
    MS_LOG(DEBUG) << "Replace op " << cnode->fullname_with_scope() << " debug string:" << cnode->DebugString()
                  << " with " << ret_node->fullname_with_scope() << " debug string:" << ret_node->DebugString()
                  << ", is dynamic shape:" << common::AnfAlgo::IsDynamicShape(node);
  }
  return ret_node;
}

CNodePtr CreateNodeHelper::ConvertToTargetOp(const CNodePtr &origin_op, OpAdaptationInfo *op_adaptation_info) {
  MS_EXCEPTION_IF_NULL(origin_op);
  MS_EXCEPTION_IF_NULL(op_adaptation_info);
  auto me_op_name = op_adaptation_info->me_op_name();
  auto backend_op_name = op_adaptation_info->backend_op_name();
  auto pre_check_func = op_adaptation_info->pre_check_func();
  auto need_tbe_check = op_adaptation_info->need_tbe_check_supported();
  auto input_to_attr_map = op_adaptation_info->input_attr_map();
  // Step1: rename to default op name
  common::AnfAlgo::SetNodeAttr(kAttrMeOpName, MakeValue(me_op_name), origin_op);
  OpAdaptationInfoRegister::RenamePrimitiveName(origin_op, me_op_name, backend_op_name);
  // Step2: pre_check
  if (pre_check_func) {
    if (!pre_check_func(origin_op)) {
      MS_LOG(DEBUG) << "Pre check function return Not Change for op " << origin_op->fullname_with_scope();
      return origin_op;
    }
  }

  // Check supported if the op need
  auto graph = origin_op->func_graph();
  auto kernel_graph = graph->cast<KernelGraphPtr>();
  if (need_tbe_check) {
    auto is_dynamic = common::AnfAlgo::IsDynamicShape(origin_op);
    // when cnode is a dynamic shape node, if origin op supported, use origin op
    if (is_dynamic) {
      auto ret = CheckAICoreSupported(origin_op);
      if (ret) {
        MS_LOG(DEBUG) << "Origin op " << origin_op->fullname_with_scope() << " is supported in this configuration";
        return origin_op;
      }
    }

    auto target_op = OpAdaptationInfoRegister::CreateTargetOp(origin_op, *op_adaptation_info);
    if (target_op == nullptr) {
      MS_LOG(DEBUG) << "Create target op failed for node " << origin_op->fullname_with_scope();
      return origin_op;
    }

    auto ret = CheckAICoreSupported(target_op);
    if (!ret) {
      MS_LOG(DEBUG) << "Target op " << target_op->fullname_with_scope() << " is supported in this configuration";
      return origin_op;
    }

    if (kernel_graph != nullptr) {
      kernel_graph->FrontBackendlMapUpdate(origin_op, target_op);
    }
    return target_op;
  }

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
