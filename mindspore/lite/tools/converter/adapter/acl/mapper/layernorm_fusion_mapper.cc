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

#include "tools/converter/adapter/acl/mapper/layernorm_fusion_mapper.h"
#include "memory"
#include "tools/converter/adapter/acl/mapper/primitive_mapper_register.h"
#include "tools/converter/adapter/acl/mapper/tbe_op_def.h"
#include "src/common/log_util.h"
#include "ops/op_utils.h"

namespace mindspore {
namespace lite {
STATUS LayerNormFusionMapper::Mapper(const CNodePtr &cnode) {
  ValueNodePtr value_node = nullptr;
  PrimitivePtr src_prim = nullptr;
  if (GetValueNodeAndPrimFromCnode(cnode, &value_node, &src_prim) != lite::RET_OK) {
    MS_LOG(ERROR) << "Get primitive from cnode failed.";
    return lite::RET_ERROR;
  }
  ops::LayerNorm layer_norm;
  PrimitivePtr dst_prim = layer_norm.GetPrim();
  CHECK_NULL_RETURN(dst_prim);
  dst_prim->SetAttrs(src_prim->attrs());
  auto status = AttrAdjust(dst_prim, ops::kStride);
  if (status != lite::RET_OK) {
    MS_LOG(ERROR) << "adjust stride failed.";
    return status;
  }
  value_node->set_value(dst_prim);
  return lite::RET_OK;
}

REGISTER_PRIMITIVE_MAPPER(kNameLayerNormFusion, LayerNormFusionMapper)
}  // namespace lite
}  // namespace mindspore
