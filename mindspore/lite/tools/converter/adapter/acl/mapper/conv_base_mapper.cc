/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "tools/converter/adapter/acl/mapper/conv_base_mapper.h"
#include <vector>
#include <map>
#include "ops/op_utils.h"
#include "nnacl/op_base.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace lite {
static const std::map<int64_t, std::string> kPadModToStrMap = {
  {PadMode::PAD, "CALCULATED"},
  {PadMode::SAME, "SAME"},
  {PadMode::VALID, "VALID"},
};
namespace {
constexpr auto kNamePaddingMode = "padding";
}  // namespace

STATUS ConvBaseMapper::AdjustAttrPad(const PrimitivePtr &prim) {
  // attr pad val
  MS_CHECK_TRUE_MSG(prim != nullptr, lite::RET_ERROR, "prim is nullptr.");
  auto pad_ptr = prim->GetAttr(ops::kPadList);
  if (pad_ptr == nullptr) {
    std::vector<int64_t> pad_list = {0, 0, 0, 0};
    prim->AddAttr(ops::kPadList, MakeValue(pad_list));
  }
  // attr pad mode
  if (prim->GetAttr(ops::kIsOriginalPadMode) != nullptr) {
    bool is_ori_pad_mode = GetValue<bool>(prim->GetAttr(ops::kIsOriginalPadMode));
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
}  // namespace lite
}  // namespace mindspore
