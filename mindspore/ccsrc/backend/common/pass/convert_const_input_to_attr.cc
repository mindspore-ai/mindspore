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

#include <algorithm>
#include "backend/common/optimizer/const_input_to_attr.h"
#include "backend/common/optimizer/reg_cpu_const_input_to_attr.h"
#include "backend/common/optimizer/reg_gpu_const_input_to_attr.h"
#include "backend/common/optimizer/reg_ascend_const_input_to_attr.h"
#include "include/common/utils/utils.h"
#include "utils/ms_context.h"
#include "include/common/utils/anfalgo.h"

namespace mindspore::opt {
const AnfNodePtr ConvertConstInputToAttr::Process(const FuncGraphPtr &, const AnfNodePtr &node,
                                                  const EquivPtr &) const {
  if (node == nullptr || !AnfUtils::IsRealCNodeKernel(node)) {
    return nullptr;
  }
  auto name = common::AnfAlgo::GetCNodeName(node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  std::string primitive_target;
  if (common::AnfAlgo::HasNodeAttr(kAttrPrimitiveTarget, cnode)) {
    primitive_target = common::AnfAlgo::GetNodeAttr<std::string>(cnode, kAttrPrimitiveTarget);
  }
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto backend = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  if (backend != primitive_target && !primitive_target.empty()) {
    MS_LOG(WARNING) << "primitive target does not match backend: " << backend
                    << ", primitive_target: " << primitive_target;
    backend = primitive_target;
  }
  auto is_dynamic_shape = common::AnfAlgo::IsDynamicShape(node);
  auto attr_index = ConstInputToAttrRegister::GetInstance().GetConstToAttr(name, backend, is_dynamic_shape);
  if (attr_index.empty()) {
    return nullptr;
  }
  ConstInputToAttr(cnode, attr_index);
  return node;
}
}  // namespace mindspore::opt
