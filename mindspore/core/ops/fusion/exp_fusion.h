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

#ifndef MINDSPORE_CORE_OPS_EXP_FUSION_H_
#define MINDSPORE_CORE_OPS_EXP_FUSION_H_
#include "ops/exp.h"
#include "abstract/abstract_value.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
constexpr auto kNameExpFusion = "ExpFusion";
class ExpFusion : public Exp {
 public:
  ExpFusion() : Exp(kNameExpFusion) { InitIOName({"x"}, {"y"}); }
  ~ExpFusion() = default;
  MS_DECLARE_PARENT(ExpFusion, Exp);
  void Init(const float base = -1.0, const float scale = 1.0, const float shift = 0.0);
  void set_base(const float base);
  void set_scale(const float scale);
  void set_shift(const float shift);
  float get_base() const;
  float get_scale() const;
  float get_shift() const;
};
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_EXP_FUSION_H_
