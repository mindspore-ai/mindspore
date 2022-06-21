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

#include "tools/converter/adapter/acl/mapper/conv2d_fusion_mapper.h"
#include "memory"
#include "tools/converter/adapter/acl/mapper/primitive_mapper_register.h"
#include "tools/converter/adapter/acl/mapper/tbe_op_def.h"
#include "src/common/log_util.h"
#include "ops/op_utils.h"

namespace mindspore {
namespace lite {
STATUS Conv2DFusionMapper::Mapper(const CNodePtr &cnode) {
  ValueNodePtr value_node = nullptr;
  PrimitivePtr src_prim = nullptr;
  if (GetValueNodeAndPrimFromCnode(cnode, &value_node, &src_prim) != lite::RET_OK) {
    MS_LOG(ERROR) << "Get primitive from cnode failed.";
    return lite::RET_ERROR;
  }
  ops::Conv2D conv2d_op;
  PrimitivePtr dst_prim = conv2d_op.GetPrim();
#ifndef SUPPORT_SD3403_DAVINCI
  bool is_depth_wise = false;
  auto depth_wise_ptr = src_prim->GetAttr(ops::kIsDepthWise);
  if (depth_wise_ptr != nullptr) {
    is_depth_wise = GetValue<bool>(depth_wise_ptr);
  }
  if (is_depth_wise) {
    dst_prim = std::make_shared<acl::DepthwiseConv2dNative>();
  }
#endif
  CHECK_NULL_RETURN(dst_prim);
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
  status = AdjustAttrPad(dst_prim);
  if (status != lite::RET_OK) {
    MS_LOG(ERROR) << "adjust pad failed.";
    return status;
  }
  value_node->set_value(dst_prim);
  return lite::RET_OK;
}

REGISTER_PRIMITIVE_MAPPER(kNameConv2DFusion, Conv2DFusionMapper)
}  // namespace lite
}  // namespace mindspore
