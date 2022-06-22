/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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
#include "backend/common/pass/convert_const_input_to_attr.h"
#include <vector>
#include <algorithm>
#include "backend/common/optimizer/const_input_to_attr.h"
#include "include/common/utils/utils.h"
#include "utils/ms_context.h"
#include "mindspore/core/ops/core_ops.h"
#include "include/common/utils/anfalgo.h"

namespace mindspore {
namespace opt {
const AnfNodePtr ConvertConstInputToAttr::Process(const FuncGraphPtr &, const AnfNodePtr &node,
                                                  const EquivPtr &) const {
  if (node == nullptr || !AnfUtils::IsRealCNodeKernel(node)) {
    return nullptr;
  }

  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  ConstInputToAttrInfoRegister reg;
  if (!ConstInputToAttrInfoRegistry::Instance().GetRegisterByOpName(common::AnfAlgo::GetCNodeName(cnode), &reg)) {
    return nullptr;
  }
  if (common::AnfAlgo::GetCNodeName(cnode) == prim::kPrimEmbeddingLookup->name() ||
      common::AnfAlgo::GetCNodeName(cnode) == prim::kPrimEmbeddingLookupCommGrad->name()) {
    if (!common::AnfAlgo::HasNodeAttr(kAttrPrimitiveTarget, cnode)) {
      return nullptr;
    }
  }
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto device = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  if (common::AnfAlgo::GetCNodeName(cnode) == prim::kPrimGatherD->name()) {
    if (device != kGPUDevice) {
      return nullptr;
    }
  }
  if ((common::AnfAlgo::GetCNodeName(cnode) == kStridedSliceGradOpName) && (device == kAscendDevice)) {
    return nullptr;
  }
  if (common::AnfAlgo::IsDynamicShape(cnode)) {
    if (device == kGPUDevice) {
      if (DynamicShapeConstInputToAttrGPU.find(common::AnfAlgo::GetCNodeName(cnode)) ==
          DynamicShapeConstInputToAttrGPU.end()) {
        MS_LOG(INFO) << "current node is dynamic shape " << cnode->fullname_with_scope();
        return nullptr;
      }
    } else if (device == kCPUDevice) {
      if (DynamicShapeConstInputToAttrCPU.find(common::AnfAlgo::GetCNodeName(cnode)) ==
          DynamicShapeConstInputToAttrCPU.end()) {
        MS_LOG(INFO) << "current node is dynamic shape " << cnode->fullname_with_scope();
        return nullptr;
      }
    } else {
      if (DynamicShapeConstInputToAttr.find(common::AnfAlgo::GetCNodeName(cnode)) ==
          DynamicShapeConstInputToAttr.end()) {
        MS_LOG(INFO) << "current node is dynamic shape " << cnode->fullname_with_scope();
        return nullptr;
      }
    }
  }

  ConstInputToAttr(cnode, reg.GetConstInputAttrInfo());
  return node;
}
}  // namespace opt
}  // namespace mindspore
