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

#ifndef MINDSPORE_CORE_OPS_MUL_FUSION_H_
#define MINDSPORE_CORE_OPS_MUL_FUSION_H_
#include "ops/mul.h"
#include "abstract/abstract_value.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
constexpr auto kNameMulFusion = "MulFusion";
class MulFusion : public Mul {
 public:
  MulFusion() : Mul(kNameMulFusion) { InitIOName({"x", "y"}, {"output"}); }
  ~MulFusion() = default;
  MS_DECLARE_PARENT(MulFusion, Mul);
  void Init(const ActivationType &activation_type);
  void set_activation_type(const ActivationType &activation_type);
  ActivationType get_activation_type() const;
};
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_MUL_FUSION_H_
