/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_OPS_PAD_FUSION_H_
#define MINDSPORE_CORE_OPS_PAD_FUSION_H_
#include "ops/pad.h"
#include "abstract/abstract_value.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
constexpr auto kNamePadFusion = "PadFusion";
class PadFusion : public Pad {
 public:
  PadFusion() : Pad(kNamePadFusion) { InitIOName({"x"}, {"y"}); }
  ~PadFusion() = default;
  MS_DECLARE_PARENT(PadFusion, Pad);
  void Init(const PaddingMode &padding_mode, const float constant_value);
  void set_padding_mode(const PaddingMode &padding_mode);
  void set_constant_value(const float constant_value);
  PaddingMode get_padding_mode() const;
  float get_constant_value() const;
};
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_PAD_FUSION_H_
