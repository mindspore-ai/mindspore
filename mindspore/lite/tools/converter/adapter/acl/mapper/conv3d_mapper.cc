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

#include "tools/converter/adapter/acl/mapper/conv3d_mapper.h"
#include "tools/converter/adapter/acl/mapper/primitive_mapper_register.h"
#include "ops/op_utils.h"
#include "mindspore/core/ops/op_name.h"

namespace mindspore {
namespace lite {
STATUS Conv3DMapper::Mapper(const CNodePtr &cnode) {
  ValueNodePtr value_node = nullptr;
  PrimitivePtr src_prim = nullptr;
  if (GetValueNodeAndPrimFromCnode(cnode, &value_node, &src_prim) != lite::RET_OK) {
    MS_LOG(ERROR) << "Get primitive from cnode failed, node name:" << cnode->fullname_with_scope() << "!";
    return lite::RET_ERROR;
  }
  if (src_prim == nullptr) {
    MS_LOG(ERROR) << "src_prm is nullptr!";
    return lite::RET_ERROR;
  }
  src_prim->AddAttr(ops::kFormat, MakeValue("NCDHW"));
  if (src_prim->HasAttr(ops::kStride)) {
    src_prim->AddAttr(ops::kStrides, src_prim->GetAttr(ops::kStride));
  }
  if (src_prim->HasAttr(ops::kPad)) {
    src_prim->AddAttr(ops::kPads, src_prim->GetAttr(ops::kPad));
    src_prim->AddAttr(ops::kPadList, src_prim->GetAttr(ops::kPad));
  }
  if (src_prim->HasAttr(ops::kGroup)) {
    src_prim->AddAttr(ops::kGroups, src_prim->GetAttr(ops::kGroup));
  }
  return lite::RET_OK;
}

REGISTER_PRIMITIVE_MAPPER(kNameConv3D, Conv3DMapper)
}  // namespace lite
}  // namespace mindspore
