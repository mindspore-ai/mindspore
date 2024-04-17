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

#include "tools/converter/adapter/acl/mapper/concat_mapper.h"
#include <memory>
#include <string>
#include "ir/anf.h"
#include "ir/primitive.h"
#include "ops/array_ops.h"
#include "tools/converter/adapter/acl/mapper/primitive_mapper_register.h"
#include "src/common/log_util.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace lite {
STATUS ConcatMapper::Mapper(const CNodePtr &cnode) {
  CHECK_NULL_RETURN(cnode);
  if (RenameNode(cnode) != RET_OK) {
    MS_LOG(ERROR) << "Concat rename failed.";
    return RET_ERROR;
  }

  ValueNodePtr value_node = nullptr;
  PrimitivePtr src_prim = nullptr;
  if (GetValueNodeAndPrimFromCnode(cnode, &value_node, &src_prim) != lite::RET_OK) {
    MS_LOG(ERROR) << "Get value node and primitive from cnode failed.";
    return lite::RET_ERROR;
  }

  if (src_prim->HasAttr("axis")) {
    auto dst_prim = std::make_shared<Primitive>(prim::kPrimConcatD->name());
    CHECK_NULL_RETURN(dst_prim);
    dst_prim->SetAttrs(src_prim->attrs());
    value_node->set_value(dst_prim);
  }

  if (AddAttrForDynInputPrimitive(cnode) != RET_OK) {
    MS_LOG(ERROR) << "Concat mapper failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

STATUS ConcatMapper::RenameNode(const CNodePtr &cnode) {
  const std::string kNamePercent = "%";
  std::string name = cnode->fullname_with_scope();
  std::string::size_type pos = 0;
  while ((pos = name.find(kNamePercent)) != name.npos) {
    name = name.replace(pos, kNamePercent.size(), "");
  }
  cnode->set_fullname_with_scope(name);
  return lite::RET_OK;
}

REGISTER_PRIMITIVE_MAPPER(kNameConcat, ConcatMapper)
}  // namespace lite
}  // namespace mindspore
