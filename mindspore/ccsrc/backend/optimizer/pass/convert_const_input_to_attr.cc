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
#include "backend/optimizer/pass/convert_const_input_to_attr.h"

#include <vector>
#include <string>
#include <memory>

#include "backend/optimizer/common/const_input_to_attr_registry.h"
#include "backend/optimizer/common/helper.h"
#include "utils/utils.h"
#include "utils/ms_context.h"
#include "base/core_ops.h"
#include "backend/session/anf_runtime_algorithm.h"
#include "backend/kernel_compiler/common_utils.h"

namespace mindspore {
namespace opt {
const AnfNodePtr ConvertConstInputToAttr::Process(const FuncGraphPtr &, const AnfNodePtr &node,
                                                  const EquivPtr &) const {
  if (node == nullptr || !AnfAlgo::IsRealCNodeKernel(node)) {
    return nullptr;
  }
  std::vector<AnfNodePtr> todos;
  if (AnfAlgo::IsGraphKernel(node)) {
    auto sub_graph = AnfAlgo::GetCNodeFuncGraphPtr(node);
    MS_EXCEPTION_IF_NULL(sub_graph);
    kernel::GetValidKernelNodes(sub_graph, &todos);
  } else {
    todos.push_back(node);
  }

  for (auto &t : todos) {
    CNodePtr cnode = t->cast<CNodePtr>();
    ConstInputToAttrInfoRegister reg;
    if (!ConstInputToAttrInfoRegistry::Instance().GetRegisterByOpName(AnfAlgo::GetCNodeName(cnode), &reg)) {
      continue;
    }
    if (AnfAlgo::GetCNodeName(cnode) == prim::kPrimEmbeddingLookup->name() ||
        AnfAlgo::GetCNodeName(cnode) == prim::kPrimEmbeddingLookupCommGrad->name()) {
      if (!AnfAlgo::HasNodeAttr(kAttrPrimitiveTarget, cnode)) {
        continue;
      }
    }
    if (AnfAlgo::GetCNodeName(cnode) == prim::kPrimGatherD->name()) {
      auto ms_context = MsContext::GetInstance();
      if (ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET) != kGPUDevice) {
        continue;
      }
    }
    if (AnfAlgo::IsDynamicShape(cnode) &&
        DynamicShapeConstInputToAttr.find(AnfAlgo::GetCNodeName(cnode)) == DynamicShapeConstInputToAttr.end()) {
      MS_LOG(INFO) << "current node is dynamic shape " << cnode->fullname_with_scope();
      continue;
    }
    ConstInputToAttr(cnode, reg.GetConstInputAttrInfo());
  }
  return node;
}
}  // namespace opt
}  // namespace mindspore
