/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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
#include "backend/common/pass/custom_op_const_input_to_attr.h"

#include <memory>

#include "utils/hash_set.h"
#include "backend/common/pass/const_input_to_attr.h"
#include "include/common/utils/anfalgo.h"

namespace mindspore {
namespace opt {
const AnfNodePtr CustomOpConstInputToAttr::Process(const FuncGraphPtr &, const AnfNodePtr &node,
                                                   const EquivPtr &) const {
  if (node == nullptr || !AnfUtils::IsRealCNodeKernel(node)) {
    return nullptr;
  }
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  if (!IsPrimitiveCNode(cnode, prim::kPrimCustom)) {
    return nullptr;
  }

  mindspore::HashSet<size_t> attr_indices;
  GetCustomOpAttrIndex(common::AnfAlgo::GetCNodePrimitive(cnode), &attr_indices);
  if (attr_indices.empty()) {
    return nullptr;
  }

  return ConstInputToAttr(cnode, attr_indices);
}
}  // namespace opt
}  // namespace mindspore
