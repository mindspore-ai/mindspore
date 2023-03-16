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

#include "tools/converter/adapter/acl/mapper/silu_mapper.h"
#include <memory>
#include <vector>
#include "tools/converter/adapter/acl/mapper/primitive_mapper_register.h"
#include "tools/converter/adapter/acl/mapper/tbe_op_def.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "tools/converter/adapter/acl/common/utils.h"
#include "ops/op_utils.h"
#include "src/common/log_util.h"
#include "mindspore/core/ops/op_name.h"
namespace mindspore {
namespace lite {
namespace {
constexpr auto kNameInputNum = 1;
}  // namespace

STATUS SiLUMapper::Mapper(const CNodePtr &cnode) {
  if (cnode->inputs().size() != kNameInputNum) {
    MS_LOG(WARNING) << "Input of resize must be " << kNameInputNum << ", real silu: " << cnode->inputs().size()
                    << ", cnode " << cnode->fullname_with_scope();
    return lite::RET_OK;
  }
  ValueNodePtr value_node = nullptr;
  PrimitivePtr src_prim = nullptr;
  if (GetValueNodeAndPrimFromCnode(cnode, &value_node, &src_prim) != lite::RET_OK) {
    MS_LOG(ERROR) << "Get primitive from cnode failed, cnode " << cnode->fullname_with_scope();
    return lite::RET_ERROR;
  }
  if (value_node == nullptr) {
    MS_LOG(ERROR) << "value node is nullptr.";
    return RET_ERROR;
  }
  if (src_prim == nullptr) {
    MS_LOG(ERROR) << "src prim is nullptr.";
    return RET_ERROR;
  }
  auto dst_prim = std::make_shared<acl::Swish>();
  if (dst_prim == nullptr) {
    MS_LOG(ERROR) << "make swish failed.";
    return RET_ERROR;
  }
  value_node->set_value(dst_prim);
  return RET_OK;
}
REGISTER_PRIMITIVE_MAPPER(kNameSiLU, SiLUMapper)
}  // namespace lite
}  // namespace mindspore
