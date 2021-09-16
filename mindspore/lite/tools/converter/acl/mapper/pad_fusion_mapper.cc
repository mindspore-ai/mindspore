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

#include "tools/converter/acl/mapper/pad_fusion_mapper.h"
#include <memory>
#include <map>
#include <string>
#include "tools/converter/acl/mapper/primitive_mapper_register.h"
#include "tools/converter/acl/mapper/tbe_op_def.h"
#include "ops/op_utils.h"

namespace mindspore {
namespace lite {
namespace {
constexpr size_t kNumFlagTwo = 2;
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
  auto dst_prim = std::make_shared<acl::PadV3>();
  MS_ASSERT(dst_prim != nullptr);
  dst_prim->SetAttrs(src_prim->attrs());
  AdjustPadAttr(dst_prim);

  if (cnode->size() != kNamePadInputNum) {
    MS_LOG(INFO) << "No need to add attr to input, input num: " << cnode->size();
    value_node->set_value(dst_prim);
    return lite::RET_OK;
  }
  auto func_graph = cnode->func_graph();
  if (func_graph == nullptr) {
    MS_LOG(ERROR) << "Func graph is nullptr.";
    return lite::RET_ERROR;
  }
  int status = AddAttrToInput(func_graph, cnode, dst_prim, ops::kConstantValue, kNumFlagTwo);
  if (status != lite::RET_OK) {
    MS_LOG(ERROR) << "Add constant value to input failed.";
    return lite::RET_ERROR;
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

  dst_prim->AddAttr(kNamePadContiguous, MakeValue(true));
}

REGISTER_PRIMITIVE_MAPPER(kNamePadFusion, PadFusionMapper)
}  // namespace lite
}  // namespace mindspore
