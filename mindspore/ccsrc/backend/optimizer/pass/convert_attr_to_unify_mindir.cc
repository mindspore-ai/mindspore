/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "backend/optimizer/pass/convert_attr_to_unify_mindir.h"
#include "utils/check_convert_utils.h"
#include "backend/session/anf_runtime_algorithm.h"
#include "backend/kernel_compiler/common_utils.h"

namespace mindspore {
namespace opt {
const AnfNodePtr ConvertAttrToUnifyMindIR::Process(const FuncGraphPtr &, const AnfNodePtr &node,
                                                   const EquivPtr &) const {
  if (node == nullptr || !AnfAlgo::IsRealCNodeKernel(node)) {
    return nullptr;
  }

  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto inputs = cnode->inputs();
  AnfNodePtr op = inputs[0];
  MS_EXCEPTION_IF_NULL(op);
  if (IsValueNode<Primitive>(op)) {
    auto prim = GetValueNode<PrimitivePtr>(op);
    MS_EXCEPTION_IF_NULL(prim);
    auto attrs = prim->attrs();
    std::string type_name = prim->name();
    for (auto attr : attrs) {
      bool converted = CheckAndConvertUtils::ConvertAttrValueToString(type_name, attr.first, &attr.second);
      if (converted) {
        prim->set_attr(attr.first, attr.second);
      }
      bool converted_ir_attr = CheckAndConvertUtils::CheckIrAttrtoOpAttr(type_name, attr.first, &attr.second);
      if (converted_ir_attr) {
        prim->set_attr(attr.first, attr.second);
      }
    }
  }

  return node;
}
}  // namespace opt
}  // namespace mindspore
