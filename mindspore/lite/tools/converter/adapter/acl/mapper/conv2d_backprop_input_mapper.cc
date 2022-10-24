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

#include "tools/converter/adapter/acl/mapper/conv2d_backprop_input_mapper.h"
#include <memory>
#include <vector>
#include "tools/converter/adapter/acl/mapper/tbe_op_def.h"
#include "tools/converter/adapter/acl/mapper/primitive_mapper_register.h"
#include "src/common/log_util.h"
#include "ops/op_utils.h"

namespace mindspore {
namespace lite {
namespace {
constexpr auto kNameFormat = "format";
constexpr auto kPrimIndex = 0;
constexpr auto kFilterIndex = 1;
constexpr auto kOutBackpropIndex = 2;
constexpr auto kInputSizeIndex = 3;
constexpr auto kInputNum = 4;
}  // namespace

STATUS Conv2DBackpropInputMapper::Mapper(const CNodePtr &cnode) {
  ValueNodePtr value_node = nullptr;
  PrimitivePtr src_prim = nullptr;
  if (GetValueNodeAndPrimFromCnode(cnode, &value_node, &src_prim) != lite::RET_OK) {
    MS_LOG(ERROR) << "Get primitive from cnode failed.";
    return lite::RET_ERROR;
  }
  PrimitivePtr dst_prim = std::make_shared<acl::Conv2DBackpropInputV2>();
  CHECK_NULL_RETURN(dst_prim);
  dst_prim->SetAttrs(src_prim->attrs());
  if (AttrAdjust(dst_prim, ops::kDilation) != lite::RET_OK) {
    MS_LOG(ERROR) << "Adjust dilation failed.";
    return lite::RET_ERROR;
  }
  if (AttrAdjust(dst_prim, ops::kStride) != lite::RET_OK) {
    MS_LOG(ERROR) << "Adjust stride failed.";
    return lite::RET_ERROR;
  }
  if (AdjustAttrFormat(dst_prim, kNameFormat) != lite::RET_OK) {
    MS_LOG(ERROR) << "Adjust format failed.";
    return lite::RET_ERROR;
  }
  if (AdjustAttrPad(dst_prim) != lite::RET_OK) {
    MS_LOG(ERROR) << "Adjust pad failed.";
    return lite::RET_ERROR;
  }
  value_node->set_value(dst_prim);

  if (AdjustInputOrder(cnode) != lite::RET_OK) {
    MS_LOG(ERROR) << "Adjust input order failed.";
    return lite::RET_ERROR;
  }
  return lite::RET_OK;
}

STATUS Conv2DBackpropInputMapper::AdjustInputOrder(const CNodePtr &cnode) {
  // original input order: out_backprop, filter, input_sizes
  // new order: input_sizes, filter, out_backprop
  if (cnode->inputs().size() != kInputNum) {
    MS_LOG(ERROR) << "Input num must be " << kInputNum << ",real num " << cnode->inputs().size();
    return lite::RET_ERROR;
  }
  auto inputs = cnode->inputs();
  std::vector<AnfNodePtr> new_inputs = {inputs[kPrimIndex], inputs[kInputSizeIndex], inputs[kOutBackpropIndex],
                                        inputs[kFilterIndex]};
  cnode->set_inputs(new_inputs);
  return lite::RET_OK;
}

REGISTER_PRIMITIVE_MAPPER(kNameConv2DBackpropInput, Conv2DBackpropInputMapper)
}  // namespace lite
}  // namespace mindspore
