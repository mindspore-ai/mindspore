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

#ifndef MINDSPORE_CORE_OPS_ACTIVATION_H_
#define MINDSPORE_CORE_OPS_ACTIVATION_H_
#include "ops/primitive_c.h"
#include "abstract/abstract_value.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
constexpr auto kNameActivation = "Activation";
class Activation : public PrimitiveC {
 public:
  Activation() : PrimitiveC(kNameActivation) {}
  ~Activation() = default;
  MS_DECLARE_PARENT(Activation, PrimitiveC);
  void Init(const float alpha = 0.2, const float min_val = -1.0, const float max_val = 1.0,
            const ActivationType &activation_type = NO_ACTIVATION);
  void set_alpha(const float alpha);
  void set_min_val(const float min_val);
  void set_max_val(const float max_val);
  void set_activation_type(const ActivationType &activation_type);
  float get_alpha() const;
  float get_min_val() const;
  float get_max_val() const;
  ActivationType get_activation_type() const;
};
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_ACTIVATION_H_
