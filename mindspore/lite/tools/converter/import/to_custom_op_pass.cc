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

#include <memory>
#include "ops/auto_generate/gen_lite_ops.h"
#include "ops/custom.h"
#include "ops/op_utils.h"
#include "mindspore/ccsrc/include/common/utils/utils.h"
#include "tools/converter/import/to_custom_op_pass.h"

using mindspore::ops::kNameGatherDGradV2;
using mindspore::ops::kNameMaskedFill;

namespace mindspore {
namespace opt {
bool ToCustomOpPass::Run(const FuncGraphPtr &graph) {
  MS_ASSERT(graph != nullptr);
  auto manager = graph->manager();
  MS_ASSERT(manager != nullptr);
  auto node_list = TopoSort(graph->get_return());

  for (auto &node : node_list) {
    if (!utils::isa<CNodePtr>(node)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    MS_ASSERT(cnode != nullptr);
    auto value_node = cnode->input(0);
    auto prim = GetValueNode<PrimitivePtr>(value_node);
    if (prim == nullptr) {
      MS_LOG(DEBUG) << "this is a call cnode, which input[0] is fg.";
      continue;
    }

    auto func = ToCustomOpRegistry::GetInstance()->GetToCustomOpFunc(prim->name());
    if (func == nullptr) {
      continue;
    }

    auto ret = func(cnode);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "failed to convert normal cnode node to custom cnode";
      return false;
    }
  }
  return true;
}

int GatherDGradV2ToCustomOp(const CNodePtr &cnode) {
  auto custom_prim = std::make_shared<mindspore::ops::Custom>();
  custom_prim->set_type(kNameGatherDGradV2);
  cnode->set_input(kAnfPrimitiveIndex, NewValueNode(custom_prim->GetPrim()));
  return RET_OK;
}

int MaskedFillToCustomOp(const CNodePtr &cnode) {
  auto custom_prim = std::make_shared<mindspore::ops::Custom>();
  custom_prim->set_type(kNameMaskedFill);
  cnode->set_input(kAnfPrimitiveIndex, NewValueNode(custom_prim->GetPrim()));
  return RET_OK;
}

REGISTER_TO_CUSTOM_OP(kNameGatherDGradV2, GatherDGradV2ToCustomOp);
REGISTER_TO_CUSTOM_OP(kNameMaskedFill, MaskedFillToCustomOp);
}  // namespace opt
}  // namespace mindspore
