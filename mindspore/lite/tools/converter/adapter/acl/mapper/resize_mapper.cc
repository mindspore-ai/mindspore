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

#include "tools/converter/adapter/acl/mapper/resize_mapper.h"
#include <memory>
#include "tools/converter/adapter/acl/mapper/primitive_mapper_register.h"
#include "tools/converter/adapter/acl/mapper/tbe_op_def.h"
#include "ops/op_utils.h"
#include "src/common/log_util.h"

namespace mindspore {
namespace lite {
namespace {
constexpr auto kNameInputNum = 3;
}

STATUS ResizeMapper::Mapper(const CNodePtr &cnode) {
  if (cnode->inputs().size() != kNameInputNum) {
    MS_LOG(WARNING) << "Input of resize must be " << kNameInputNum << ", real size: " << cnode->inputs().size();
    return lite::RET_OK;
  }
  ValueNodePtr value_node = nullptr;
  PrimitivePtr src_prim = nullptr;
  if (GetValueNodeAndPrimFromCnode(cnode, &value_node, &src_prim) != lite::RET_OK) {
    MS_LOG(ERROR) << "Get primitive from cnode failed.";
    return lite::RET_ERROR;
  }
  auto val_ptr = src_prim->GetAttr(ops::kMethod);
  CHECK_NULL_RETURN(val_ptr);
  int64_t method = GetValue<int64_t>(val_ptr);
  PrimitivePtr dst_prim = nullptr;
  if (method == static_cast<int64_t>(mindspore::ResizeMethod::NEAREST)) {
    dst_prim = std::make_shared<acl::ResizeNearestNeighborV2>();
  } else {
    MS_LOG(WARNING) << "Not support method " << method;
    return lite::RET_OK;
  }
  CHECK_NULL_RETURN(dst_prim);
  dst_prim->SetAttrs(src_prim->attrs());
  value_node->set_value(dst_prim);
  return RET_OK;
}

REGISTER_PRIMITIVE_MAPPER(kNameResize, ResizeMapper)
}  // namespace lite
}  // namespace mindspore
