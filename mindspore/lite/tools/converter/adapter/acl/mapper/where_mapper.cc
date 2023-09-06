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

#include "tools/converter/adapter/acl/mapper/where_mapper.h"
#include <memory>
#include "tools/converter/adapter/acl/common/utils.h"
#include "tools/converter/adapter/acl/mapper/primitive_mapper_register.h"
#include "tools/converter/adapter/acl/mapper/tbe_op_def.h"
#include "plugin/device/cpu/kernel/nnacl/op_base.h"

namespace mindspore {
namespace lite {
STATUS WhereMapper::Mapper(const CNodePtr &cnode) {
  ValueNodePtr value_node = nullptr;
  PrimitivePtr src_prim = nullptr;
  if (GetValueNodeAndPrimFromCnode(cnode, &value_node, &src_prim) != lite::RET_OK) {
    MS_LOG(ERROR) << "Get value node and primitive from cnode failed.";
    return lite::RET_ERROR;
  }
  auto inputs = cnode->inputs();
  if (inputs.size() != C2NUM && inputs.size() != C4NUM) {
    MS_LOG(ERROR) << "Invalid input size:" << inputs.size();
    return RET_ERROR;
  }
  PrimitivePtr dst_prim = nullptr;
  if (inputs.size() == C2NUM) {
    dst_prim = std::make_shared<acl::Where>();
    auto attr_value = src_prim->GetAttr("is_nonzero");
    if (attr_value != nullptr && GetValue<bool>(attr_value)) {
      dst_prim = std::make_shared<acl::NonZeroV2>();
    }
  } else {
    dst_prim = std::make_shared<acl::SelectV2>();
  }
  if (MoveAttrMap(cnode, dst_prim) != RET_OK) {
    MS_LOG(ERROR) << "WhereMapper moves attributes failed.";
    return RET_ERROR;
  }
  value_node->set_value(dst_prim);
  return lite::RET_OK;
}
REGISTER_PRIMITIVE_MAPPER(kNameWhere, WhereMapper)
}  // namespace lite
}  // namespace mindspore
