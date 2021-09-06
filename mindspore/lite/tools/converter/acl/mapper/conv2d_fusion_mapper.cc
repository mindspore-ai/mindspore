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

#include "tools/converter/acl/mapper/conv2d_fusion_mapper.h"
#include "memory"
#include "tools/converter/acl/mapper/primitive_mapper_register.h"

namespace mindspore {
namespace lite {
STATUS Conv2DFusionMapper::Mapper(const CNodePtr &cnode) {
  ValueNodePtr value_node = nullptr;
  PrimitivePtr src_prim = nullptr;
  if (GetValueNodeAndPrimFromCnode(cnode, &value_node, &src_prim) != lite::RET_OK) {
    MS_LOG(ERROR) << "Get primitive from cnode failed.";
    return lite::RET_ERROR;
  }
  auto dst_prim = std::make_shared<ops::Conv2D>();
  MS_ASSERT(dst_prim != nullptr);
  dst_prim->SetAttrs(src_prim->attrs());

  auto status = AttrAdjust(dst_prim, ops::kStride);
  if (status != lite::RET_OK) {
    MS_LOG(ERROR) << "adjust stride failed.";
    return status;
  }
  status = AttrAdjust(dst_prim, ops::kDilation);
  if (status != lite::RET_OK) {
    MS_LOG(ERROR) << "adjust dilation failed.";
    return status;
  }
  value_node->set_value(dst_prim);
  return lite::RET_OK;
}

REGISTER_PRIMITIVE_MAPPER(kNameConv2DFusion, Conv2DFusionMapper)
}  // namespace lite
}  // namespace mindspore
