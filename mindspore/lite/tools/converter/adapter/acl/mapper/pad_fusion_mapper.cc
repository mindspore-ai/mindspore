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

#include "tools/converter/adapter/acl/mapper/pad_fusion_mapper.h"
#include <memory>
#include <map>
#include <string>
#include "tools/converter/adapter/acl/mapper/primitive_mapper_register.h"
#include "tools/converter/adapter/acl/mapper/tbe_op_def.h"
#include "ops/op_utils.h"
#include "src/common/log_util.h"

namespace mindspore {
namespace lite {
static const std::map<std::string, PrimitivePtr> kPadTypeMap = {
  {"Pad", std::make_shared<acl::PadV1>()},
  {"PadV2", std::make_shared<acl::PadV2>()},
  {"PadV3", std::make_shared<acl::PadV3>()},
  {"MirrorPad", std::make_shared<acl::MirrorPad>()},
};

namespace {
constexpr size_t kNamePadInputNum = 3;
constexpr auto kNamePadContiguous = "pad_contiguous";
}  // namespace

STATUS PadFusionMapper::Mapper(const CNodePtr &cnode) {
  ValueNodePtr value_node = nullptr;
  PrimitivePtr src_prim = nullptr;
  if (GetValueNodeAndPrimFromCnode(cnode, &value_node, &src_prim) != lite::RET_OK) {
    MS_LOG(ERROR) << "Get primitive from cnode failed.";
    return lite::RET_ERROR;
  }
  std::string origin_name;
  if (src_prim->GetAttr(kNamePadContiguous) != nullptr) {
    origin_name = "PadV3";
  }
  auto value_ptr = src_prim->GetAttr(ops::kOriginalOpName);
  if (value_ptr != nullptr) {
    origin_name = GetValue<std::string>(value_ptr);
  }
  PrimitivePtr dst_prim = nullptr;
  if (kPadTypeMap.find(origin_name) != kPadTypeMap.end()) {
    dst_prim = kPadTypeMap.at(origin_name);
  }
  CHECK_NULL_RETURN(dst_prim);
  dst_prim->SetAttrs(src_prim->attrs());
  AdjustPadAttr(dst_prim);
  if (origin_name != "Pad") {
    if (ConvertAttrToInput(cnode, dst_prim) != lite::RET_OK) {
      MS_LOG(ERROR) << "Convert attr to input failed.";
      return lite::RET_ERROR;
    }
  }
  value_node->set_value(dst_prim);
  return lite::RET_OK;
}

void PadFusionMapper::AdjustPadAttr(const PrimitivePtr &dst_prim) {
  static std::map<int64_t, std::string> kPadModeToStrMap = {
    {PaddingMode::CONSTANT, "constant"},
    {PaddingMode::REFLECT, "reflect"},
    {PaddingMode::SYMMETRIC, "edge"},
  };
  auto pad_mode_value = dst_prim->GetAttr(ops::kPaddingMode);
  if (pad_mode_value != nullptr) {
    auto pad_mode = GetValue<int64_t>(pad_mode_value);
    if (kPadModeToStrMap.find(pad_mode) != kPadModeToStrMap.end()) {
      dst_prim->AddAttr(ops::kMode, MakeValue(kPadModeToStrMap[pad_mode]));
      dst_prim->DelAttr(ops::kPaddingMode);
    }
  }
}

STATUS PadFusionMapper::ConvertAttrToInput(const CNodePtr &cnode, const PrimitivePtr &prim) {
  if (cnode->size() != kNamePadInputNum) {
    MS_LOG(INFO) << "No need to add attr to input, real input num: " << cnode->size();
    return lite::RET_OK;
  }
  auto func_graph = cnode->func_graph();
  CHECK_NULL_RETURN(func_graph);
  auto status = AddFloatAttrToInput(func_graph, cnode, prim, ops::kConstantValue, false);
  if (status != lite::RET_OK) {
    MS_LOG(ERROR) << "Add constant value to input failed.";
    return lite::RET_ERROR;
  }
  return lite::RET_OK;
}

REGISTER_PRIMITIVE_MAPPER(kNamePadFusion, PadFusionMapper)
}  // namespace lite
}  // namespace mindspore
