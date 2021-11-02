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
#include <vector>
#include <map>
#include <string>
#include "memory"
#include "tools/converter/adapter/acl/mapper/primitive_mapper_register.h"
#include "tools/converter/adapter/acl/mapper/tbe_op_def.h"
#include "src/common/log_util.h"

namespace mindspore {
namespace lite {
static const std::map<int64_t, std::string> kPadModToStrMap = {
  {PadMode::PAD, "CALCULATED"},
  {PadMode::SAME, "SAME"},
  {PadMode::VALID, "VALID"},
};
namespace {
constexpr auto kNamePaddingMode = "padding";
constexpr auto kIsOriPadMode = "is_ori_pad_mode";
}  // namespace

STATUS Conv2DFusionMapper::Mapper(const CNodePtr &cnode) {
  ValueNodePtr value_node = nullptr;
  PrimitivePtr src_prim = nullptr;
  if (GetValueNodeAndPrimFromCnode(cnode, &value_node, &src_prim) != lite::RET_OK) {
    MS_LOG(ERROR) << "Get primitive from cnode failed.";
    return lite::RET_ERROR;
  }
  bool is_depth_wise = false;
  auto depth_wise_ptr = src_prim->GetAttr(ops::kIsDepthWise);
  if (depth_wise_ptr != nullptr) {
    is_depth_wise = GetValue<bool>(depth_wise_ptr);
  }
  PrimitivePtr dst_prim = nullptr;
  if (!is_depth_wise) {
    dst_prim = std::make_shared<ops::Conv2D>();
  } else {
    dst_prim = std::make_shared<acl::DepthwiseConv2dNative>();
  }
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

STATUS Conv2DFusionMapper::AdjustAttrPad(const PrimitivePtr &prim) {
  // attr pad val
  auto pad_ptr = prim->GetAttr(ops::kPadList);
  if (pad_ptr == nullptr) {
    std::vector<int64_t> pad_list = {0, 0, 0, 0};
    prim->AddAttr(ops::kPadList, MakeValue(pad_list));
  }
  // attr pad mode
  if (prim->GetAttr(kIsOriPadMode) != nullptr) {
    bool is_ori_pad_mode = GetValue<bool>(prim->GetAttr(kIsOriPadMode));
    if (!is_ori_pad_mode) {
      MS_LOG(INFO) << "No need to add attr padding mode";
      return lite::RET_OK;
    }
    auto pad_mode_val = prim->GetAttr(ops::kPadMode);
    if (pad_mode_val != nullptr) {
      auto pad_mode = GetValue<int64_t>(pad_mode_val);
      if (kPadModToStrMap.find(pad_mode) != kPadModToStrMap.end()) {
        std::string padding_mode = kPadModToStrMap.at(pad_mode);
        prim->AddAttr(kNamePaddingMode, MakeValue(padding_mode));
      }
    }
  }
  return lite::RET_OK;
}

REGISTER_PRIMITIVE_MAPPER(kNameConv2DFusion, Conv2DFusionMapper)
}  // namespace lite
}  // namespace mindspore
