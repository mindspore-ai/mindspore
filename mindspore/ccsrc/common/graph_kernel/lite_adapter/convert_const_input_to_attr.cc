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
#include "common/graph_kernel/lite_adapter/convert_const_input_to_attr.h"

#include "backend/common/optimizer/const_input_to_attr.h"
#include "utils/anf_utils.h"

namespace mindspore::graphkernel {
bool ConvertConstInputToAttr::Run(const FuncGraphPtr &func_graph) {
  bool changed = false;
  auto nodes = TopoSort(func_graph->get_return());
  for (auto node : nodes) {
    if (node == nullptr || !AnfUtils::IsRealCNodeKernel(node)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    opt::ConstInputToAttrInfoRegister reg;
    if (!opt::ConstInputToAttrInfoRegistry::Instance().GetRegisterByOpName(AnfUtils::GetCNodeName(cnode), &reg)) {
      continue;
    }
    changed = true;
    opt::ConstInputToAttr(cnode, reg.GetConstInputAttrInfo());
  }
  return changed;
}
}  // namespace mindspore::graphkernel
